import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import statistics
import random
import json
import subprocess
import sys
import time
import math
from collections import deque
from pathlib import Path

import tqdm
import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from tensordict import TensorDict

from fast_td3_utils import (
    EmpiricalNormalization,
    IdentityNormalizer,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_eval_snapshot,
    save_params,
    get_ddp_state_dict,
    load_ddp_state_dict,
    mark_step,
)
from hyperparams import get_args

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False


def setup_distributed(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    is_distributed = world_size > 1
    if is_distributed:
        print(
            f"Initializing distributed training with rank {rank}, world size {world_size}"
        )
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
        torch.cuda.set_device(rank)
    return is_distributed



def log_git_state(save_dir: str) -> None:
    """Record git commit/branch/status of this repo and unitree_rl_lab to log dir."""
    import subprocess as _sp
    from pathlib import Path as _P

    repos = {
        "FastTD3_unitree": _P(__file__).resolve().parents[1],
    }
    unitree_path = os.environ.get("UNITREE_RL_LAB_PATH")
    if unitree_path:
        repos["unitree_rl_lab"] = _P(unitree_path)

    git_dir = _P(save_dir) / "params"
    git_dir.mkdir(parents=True, exist_ok=True)
    git_file = git_dir / "git_state.txt"
    lines = []
    for repo_name, repo_path in repos.items():
        lines.append(f"===== {repo_name} ({repo_path}) =====")
        for label, cmd in [
            ("commit", ["git", "rev-parse", "HEAD"]),
            ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            ("status", ["git", "status", "--porcelain"]),
        ]:
            try:
                result = _sp.check_output(
                    cmd, cwd=str(repo_path), text=True, stderr=_sp.DEVNULL
                ).strip()
                lines.append(f"{label}: {result}")
            except Exception:
                lines.append(f"{label}: <unavailable>")
        lines.append("")
    with open(git_file, "w") as f:
        f.write("\n".join(lines))


def main(rank: int, world_size: int):
    is_distributed = setup_distributed(rank, world_size)

    args = get_args()
    if rank == 0:
        print(args)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"
    checkpoint_prefix = args.checkpoint_prefix or run_name
    eval_checkpoint_path = os.path.join(args.save_dir, "eval", f"{run_name}_rank{rank}_eval_latest.pt")

    def checkpoint_path(step=None, final: bool = False) -> str:
        if final and not args.save_final_as_step:
            filename = f"{run_name}_final.pt"
        else:
            if step is None:
                raise ValueError("step must be provided for step-named checkpoints")
            filename = f"{checkpoint_prefix}_{step}.pt"
        return os.path.join(args.save_dir, filename)

    def export_unitree_params(envs) -> None:
        if not args.export_unitree_params or rank != 0:
            return

        import inspect
        import shutil

        from isaaclab.utils.io import dump_yaml
        from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

        params_dir = os.path.join(args.save_dir, "params")
        dump_yaml(os.path.join(params_dir, "env.yaml"), envs.env_cfg)
        dump_yaml(os.path.join(params_dir, "agent.yaml"), vars(args))
        export_deploy_cfg(envs.envs.unwrapped, args.save_dir)

        env_cfg_file = inspect.getfile(envs.env_cfg.__class__)
        shutil.copy(
            env_cfg_file,
            os.path.join(params_dir, os.path.basename(env_cfg_file)),
        )

    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = (
        "cuda"
        if args.cuda and torch.cuda.is_available()
        else "mps" if args.cuda and torch.backends.mps.is_available() else "cpu"
    )
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    writer = None
    if args.log_tensorboard and rank == 0:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=args.save_dir, flush_secs=10)

    if args.use_wandb and rank == 0:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    neptune_run = None
    if args.use_neptune and rank == 0:
        import neptune

        neptune_run = neptune.init_run(project=args.neptune_project, name=run_name)
        neptune_run["config"] = vars(args)

    # Use different seeds per rank to avoid synchronization issues
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    if not (args.env_name.startswith("Isaac-") and "G1" in args.env_name):
        raise ValueError(
            "Only IsaacLab Unitree G1 environments are supported in this Unitree-only fork"
        )

    # Validate use_tanh / action_bounds combination
    if args.use_tanh and args.action_bounds is None:
        raise ValueError(
            "use_tanh=True requires action_bounds to be set (e.g., --use_tanh --action_bounds 1.0)"
        )
    if not args.use_tanh and args.action_bounds is not None:
        raise ValueError(
            "use_tanh=False does not support action_bounds; "
            "remove --action_bounds or set --use_tanh"
        )

    from environments.isaaclab_env import IsaacLabEnv

    isaac_device = f"cuda:{rank}" if device.type == "cuda" else device.type
    envs = IsaacLabEnv(
        args.env_name,
        isaac_device,
        args.num_envs,
        args.seed + rank,
        action_bounds=args.action_bounds,
    )

    export_unitree_params(envs)
    if rank == 0:
        log_git_state(args.save_dir)

    n_act = envs.num_actions
    n_obs = envs.num_obs if type(envs.num_obs) == int else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if type(envs.num_privileged_obs) == int
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = IdentityNormalizer()
        critic_obs_normalizer = IdentityNormalizer()

    if args.reward_normalization:
        reward_normalizer = RewardNormalizer(
            gamma=args.gamma,
            device=device,
            g_max=min(abs(args.v_min), abs(args.v_max)),
        )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
        "std_min": args.std_min,
        "std_max": args.std_max,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    if args.agent == "fasttd3":
        from fast_td3 import Actor, Critic

        actor_cls = Actor
        critic_cls = Critic

        actor_kwargs.update(
            {
                "sim_type": args.sim_type,
                "sim_dimension": args.sim_dimension,
                "seq_len": args.actor_seq_len,
                "use_tanh": args.use_tanh,
            }
        )
        critic_kwargs.update(
            {
                "sim_type": args.sim_type,
                "sim_dimension": args.sim_dimension,
                "seq_len": args.critic_seq_len,
            }
        )

        if rank == 0:
            print("Using FastTD3")
    elif args.agent == "fasttd3_simbav2":
        if args.sim_type:
            raise ValueError("SimNorm options are only supported with agent='fasttd3'")

        from fast_td3_simbav2 import Actor, Critic

        actor_cls = Actor
        critic_cls = Critic

        if rank == 0:
            print("Using FastTD3 + SimbaV2")
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
                "alpha_init": 1.0 / (args.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": args.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
                "alpha_init": 1.0 / (args.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
                "num_blocks": args.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
    else:
        raise ValueError(f"Agent {args.agent} not supported")

    actor = actor_cls(**actor_kwargs)
    if is_distributed:
        actor = DDP(actor, device_ids=[rank])
    from tensordict import from_module

    actor_detach = actor_cls(**actor_kwargs)
    # Copy params to actor_detach without grad
    from_module(actor.module if hasattr(actor, "module") else actor).data.to_module(
        actor_detach
    )
    policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs)
    if is_distributed:
        qnet = DDP(qnet, device_ids=[rank])
    qnet_target = critic_cls(**critic_kwargs)  # Create a separate instance
    qnet_target.load_state_dict(get_ddp_state_dict(qnet))

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    # Add learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate(step: int):
        save_eval_snapshot(
            step,
            actor,
            obs_normalizer,
            args,
            eval_checkpoint_path,
            extra_state={"curriculum_snapshot": envs.snapshot_curriculum()},
        )
        repo_root = Path(__file__).resolve().parents[1]
        eval_env = os.environ.copy()
        eval_env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{eval_env['PYTHONPATH']}"
            if eval_env.get("PYTHONPATH")
            else str(repo_root)
        )
        command = [
            sys.executable,
            "-m",
            "fast_td3.eval_unitree",
            "--checkpoint",
            str(Path(eval_checkpoint_path).resolve()),
            "--env-name",
            args.env_name,
            "--device",
            isaac_device,
            "--num-envs",
            str(args.eval_num_envs),
            "--seed",
            str(args.seed + args.eval_seed_offset + rank),
            "--action-bounds",
            str(args.action_bounds) if args.action_bounds is not None else "None",
            "--amp-dtype",
            args.amp_dtype,
        ]
        if args.amp:
            command.append("--amp")
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=repo_root,
            env=eval_env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Evaluation subprocess failed"
                f"\nstdout:\n{result.stdout}"
                f"\nstderr:\n{result.stderr}"
            )
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            payload = json.loads(line)
            return payload["eval_avg_return"], payload["eval_avg_length"]
        raise RuntimeError(f"Evaluation subprocess did not return JSON:\n{result.stdout}")

    def scalar_value(value):
        if isinstance(value, torch.Tensor):
            return value.detach().mean().item()
        return float(value)

    def collect_episode_info_scalars(ep_infos):
        if not ep_infos:
            return {}

        scalars = {}
        for key in ep_infos[0]:
            values = []
            for ep_info in ep_infos:
                if key not in ep_info:
                    continue
                value = ep_info[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor([value], device=device)
                value = value.detach().to(device)
                if len(value.shape) == 0:
                    value = value.unsqueeze(0)
                values.append(value.flatten())
            if not values:
                continue
            tag = key if "/" in key else f"Episode/{key}"
            scalars[tag] = torch.cat(values).mean().item()
        return scalars

    def collect_train_scalars(logs, ep_infos, rewbuffer, lenbuffer, speed, step):
        scalars = {
            "Perf/total_fps": speed * envs.num_envs * world_size,
            "Perf/control_steps_per_second": speed,
            "Perf/world_size": world_size,
            "Loss/actor_loss": scalar_value(logs["actor_loss"]),
            "Loss/qf_loss": scalar_value(logs["qf_loss"]),
            "Loss/actor_learning_rate": actor_scheduler.get_last_lr()[0],
            "Loss/critic_learning_rate": q_scheduler.get_last_lr()[0],
            "Value/qf_max": scalar_value(logs["qf_max"]),
            "Value/qf_min": scalar_value(logs["qf_min"]),
            "Grad/actor_norm": scalar_value(logs["actor_grad_norm"]),
            "Grad/critic_norm": scalar_value(logs["critic_grad_norm"]),
            "Train/env_reward": scalar_value(logs["env_rewards"]),
            "Train/buffer_reward": scalar_value(logs["buffer_rewards"]),
            "Train/total_timesteps": step * envs.num_envs * world_size,
        }

        if len(rewbuffer) > 0:
            scalars["Train/mean_reward"] = statistics.mean(rewbuffer)
            scalars["Train/mean_episode_length"] = statistics.mean(lenbuffer)
        if "eval_avg_return" in logs:
            scalars["Eval/avg_return"] = scalar_value(logs["eval_avg_return"])
            scalars["Eval/avg_length"] = scalar_value(logs["eval_avg_length"])
        scalars.update(envs.curriculum_scalars())
        scalars.update(collect_episode_info_scalars(ep_infos))
        return scalars

    def write_scalar_logs(scalars, step):
        if rank != 0:
            return
        if writer is not None:
            for tag, value in scalars.items():
                writer.add_scalar(tag, value, step)
        if args.use_wandb:
            wandb.log(scalars, step=step)
        if neptune_run is not None:
            for tag, value in scalars.items():
                neptune_run[tag].log(value, step=step)

    def update_main(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = actor(next_observations) + clipped_noise
            if args.use_tanh:
                next_state_actions = next_state_actions.clamp(-1.0, 1.0)
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )

            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = (
                qnet.module.get_value(F.softmax(qf1, dim=1))
                if hasattr(qnet, "module")
                else qnet.get_value(F.softmax(qf1, dim=1))
            )
            qf2_value = (
                qnet.module.get_value(F.softmax(qf2, dim=1))
                if hasattr(qnet, "module")
                else qnet.get_value(F.softmax(qf2, dim=1))
            )
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        # Handle DDP module by accessing .module attribute
        src_module = src.module if hasattr(src, "module") else src
        tgt_module = tgt.module if hasattr(tgt, "module") else tgt

        src_ps = [p.data for p in src_module.parameters()]
        tgt_ps = [p.data for p in tgt_module.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if args.compile:
        compile_mode = args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        update_pol = torch.compile(update_pol, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        # Keep stateful running-stat normalizers eager. Inductor/Triton can
        # fail compiling their fused mean/var + buffer update kernels.
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs(
            random_start_init=args.random_start_init
        )
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset(random_start_init=args.random_start_init)
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        load_ddp_state_dict(actor, torch_checkpoint["actor_state_dict"])
        if torch_checkpoint["obs_normalizer_state"] is not None:
            obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        if torch_checkpoint["critic_obs_normalizer_state"] is not None:
            critic_obs_normalizer.load_state_dict(
                torch_checkpoint["critic_obs_normalizer_state"]
            )
        load_ddp_state_dict(qnet, torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(envs.num_envs, dtype=torch.float, device=device)
    cur_episode_length = torch.zeros(envs.num_envs, dtype=torch.float, device=device)
    start_time = None
    desc = ""

    while global_step < args.total_timesteps:
        mark_step()
        logs_dict = TensorDict()
        if (
            start_time is None
            and global_step >= args.measure_burnin + args.learning_starts
        ):
            start_time = time.time()
            measure_burnin = global_step

        with torch.no_grad(), autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            norm_obs = normalize_obs(obs)
            actions = policy(obs=norm_obs, dones=dones)

        next_obs, rewards, dones, infos = envs.step(actions.float())
        truncations = infos["time_outs"]

        if "episode" in infos:
            ep_infos.append(infos["episode"])
        elif "log" in infos:
            ep_infos.append(infos["log"])
        cur_reward_sum += rewards
        cur_episode_length += 1
        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            rewbuffer.extend(cur_reward_sum[done_ids].detach().cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[done_ids].detach().cpu().numpy().tolist())
            cur_reward_sum[done_ids] = 0
            cur_episode_length[done_ids] = 0

        if args.reward_normalization:
            update_stats(rewards, dones.float())

        if envs.asymmetric_obs:
            next_critic_obs = infos["observations"]["critic"]
        # Compute 'true' next_obs and next_critic_obs for saving
        true_next_obs = torch.where(
            dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
        )
        if envs.asymmetric_obs:
            true_next_critic_obs = torch.where(
                dones[:, None] > 0,
                infos["observations"]["raw"]["critic_obs"],
                next_critic_obs,
            )

        transition = TensorDict(
            {
                "observations": obs,
                "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                "next": {
                    "observations": true_next_obs,
                    "rewards": torch.as_tensor(
                        rewards, device=device, dtype=torch.float
                    ),
                    "truncations": truncations.long(),
                    "dones": dones.long(),
                },
            },
            batch_size=(envs.num_envs,),
            device=device,
        )
        if envs.asymmetric_obs:
            transition["critic_observations"] = critic_obs
            transition["next"]["critic_observations"] = true_next_critic_obs
        rb.extend(transition)

        obs = next_obs
        if envs.asymmetric_obs:
            critic_obs = next_critic_obs

        if global_step > args.learning_starts:
            for i in range(args.num_updates):
                data = rb.sample(max(1, args.batch_size // args.num_envs))
                data["observations"] = normalize_obs(data["observations"])
                data["next"]["observations"] = normalize_obs(
                    data["next"]["observations"]
                )
                if envs.asymmetric_obs:
                    data["critic_observations"] = normalize_critic_obs(
                        data["critic_observations"]
                    )
                    data["next"]["critic_observations"] = normalize_critic_obs(
                        data["next"]["critic_observations"]
                    )
                raw_rewards = data["next"]["rewards"]
                data["next"]["rewards"] = normalize_reward(raw_rewards)

                logs_dict = update_main(data, logs_dict)
                if args.num_updates > 1:
                    if i % args.policy_frequency == 1:
                        logs_dict = update_pol(data, logs_dict)
                else:
                    if global_step % args.policy_frequency == 0:
                        logs_dict = update_pol(data, logs_dict)

                soft_update(qnet, qnet_target, args.tau)

            if args.log_interval > 0 and global_step % args.log_interval == 0 and start_time is not None:
                speed = (global_step - measure_burnin) / (time.time() - start_time)
                if rank == 0:
                    pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                with torch.no_grad():
                    logs = {
                        "actor_loss": logs_dict["actor_loss"].mean(),
                        "qf_loss": logs_dict["qf_loss"].mean(),
                        "qf_max": logs_dict["qf_max"].mean(),
                        "qf_min": logs_dict["qf_min"].mean(),
                        "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                        "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                        "env_rewards": rewards.mean(),
                        "buffer_rewards": raw_rewards.mean(),
                    }

                    if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                        local_eval_avg_return, local_eval_avg_length = evaluate(global_step)
                        eval_results = torch.tensor(
                            [local_eval_avg_return, local_eval_avg_length],
                            device=device,
                        )
                        if is_distributed:
                            torch.distributed.all_reduce(
                                eval_results, op=torch.distributed.ReduceOp.AVG
                            )

                        if rank == 0:
                            global_avg_return = eval_results[0].item()
                            global_avg_length = eval_results[1].item()
                            print(
                                f"Evaluating at global step {global_step}: Avg Return={global_avg_return:.2f}"
                            )
                            logs["eval_avg_return"] = global_avg_return
                            logs["eval_avg_length"] = global_avg_length

                scalar_logs = collect_train_scalars(
                    logs, ep_infos, rewbuffer, lenbuffer, speed, global_step
                )
                write_scalar_logs(scalar_logs, global_step)
                ep_infos.clear()

            if (
                args.save_interval > 0
                and global_step > 0
                and global_step % args.save_interval == 0
                and rank == 0
            ):
                print(f"Saving model at global step {global_step}")
                save_params(
                    global_step,
                    actor,
                    qnet,
                    qnet_target,
                    obs_normalizer,
                    critic_obs_normalizer,
                    args,
                    checkpoint_path(global_step),
                )

        global_step += 1
        actor_scheduler.step()
        q_scheduler.step()
        if rank == 0:
            pbar.update(1)

    if rank == 0:
        save_params(
            global_step,
            actor,
            qnet,
            qnet_target,
            obs_normalizer,
            critic_obs_normalizer,
            args,
            checkpoint_path(global_step, final=True),
        )
        if writer is not None:
            writer.flush()
            writer.close()

    if neptune_run is not None:
        neptune_run.stop()

    envs.close()

    # Cleanup distributed training
    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
