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

from tensordict import TensorDict

from fast_td3_utils import (
    EmpiricalNormalization,
    IdentityNormalizer,
    RewardNormalizer,
    SimpleReplayBuffer,
    save_eval_snapshot,
    save_params,
    mark_step,
)
from hyperparams import get_args

torch.set_float32_matmul_precision("high")


def main():
    args = get_args()
    print(args)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"
    checkpoint_prefix = args.checkpoint_prefix or run_name
    eval_checkpoint_path = os.path.join(args.save_dir, "eval", f"{run_name}_eval_latest.pt")

    def checkpoint_path(step=None, final: bool = False) -> str:
        if final and not args.save_final_as_step:
            filename = f"{run_name}_final.pt"
        else:
            if step is None:
                raise ValueError("step must be provided for step-named checkpoints")
            filename = f"{checkpoint_prefix}_{step}.pt"
        return os.path.join(args.save_dir, filename)

    def export_unitree_params(envs) -> None:
        if not args.export_unitree_params:
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
    if args.log_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=args.save_dir, flush_secs=10)

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if not args.cuda:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device_rank}")
        elif torch.backends.mps.is_available():
            device = torch.device(f"mps:{args.device_rank}")
        else:
            raise ValueError("No GPU available")
    print(f"Using device: {device}")

    if not args.env_name.startswith("Isaac-"):
        raise ValueError(
            "fast_td3/train.py only supports IsaacLab/Unitree environments; "
            f"got env_name={args.env_name!r}"
        )

    from environments.isaaclab_env import IsaacLabEnv

    envs = IsaacLabEnv(
        args.env_name,
        str(device),
        args.num_envs,
        args.seed,
        action_bounds=args.action_bounds,
    )

    if args.render_interval > 0:
        raise NotImplementedError("Rendering is not supported for IsaacLab environments")

    export_unitree_params(envs)

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
    action_low, action_high = -1.0, 1.0

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
            }
        )
        critic_kwargs.update(
            {
                "sim_type": args.sim_type,
                "sim_dimension": args.sim_dimension,
                "seq_len": args.critic_seq_len,
            }
        )

        print("Using FastTD3")
    elif args.agent == "fasttd3_simbav2":
        if args.sim_type:
            raise ValueError("SimNorm options are only supported with agent='fasttd3'")

        from fast_td3_simbav2 import Actor, Critic

        actor_cls = Actor
        critic_cls = Critic

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

    from tensordict import from_module

    actor_detach = actor_cls(**actor_kwargs)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs)
    qnet_target = critic_cls(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

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
        eval_script = Path(__file__).resolve().with_name("eval_unitree.py")
        command = [
            sys.executable,
            str(eval_script),
            "--checkpoint",
            eval_checkpoint_path,
            "--env-name",
            args.env_name,
            "--device",
            str(device),
            "--num-envs",
            str(args.eval_num_envs),
            "--seed",
            str(args.seed + args.eval_seed_offset),
            "--action-bounds",
            str(args.action_bounds),
            "--amp-dtype",
            args.amp_dtype,
        ]
        if args.amp:
            command.append("--amp")
        result = subprocess.run(command, text=True, capture_output=True)
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
            "Perf/total_fps": speed * envs.num_envs,
            "Perf/control_steps_per_second": speed,
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
            "Train/total_timesteps": step * envs.num_envs,
        }

        if len(rewbuffer) > 0:
            scalars["Train/mean_reward"] = statistics.mean(rewbuffer)
            scalars["Train/mean_episode_length"] = statistics.mean(lenbuffer)
        if "eval_avg_return" in logs:
            scalars["Eval/avg_return"] = scalar_value(logs["eval_avg_return"])
            scalars["Eval/avg_length"] = scalar_value(logs["eval_avg_length"])
        scalars.update(collect_episode_info_scalars(ep_infos))
        return scalars

    def write_scalar_logs(scalars, step):
        if writer is not None:
            for tag, value in scalars.items():
                writer.add_scalar(tag, value, step)
        if args.use_wandb:
            wandb.log(scalars, step=step)

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

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
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
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
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
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

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
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
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

    try:
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
                            print(f"Evaluating at global step {global_step}")
                            eval_avg_return, eval_avg_length = evaluate(global_step)
                            logs["eval_avg_return"] = eval_avg_return
                            logs["eval_avg_length"] = eval_avg_length

                    scalar_logs = collect_train_scalars(
                        logs, ep_infos, rewbuffer, lenbuffer, speed, global_step
                    )
                    write_scalar_logs(scalar_logs, global_step)
                    ep_infos.clear()

                if (
                    args.save_interval > 0
                    and global_step > 0
                    and global_step % args.save_interval == 0
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
            pbar.update(1)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        envs.close()

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


if __name__ == "__main__":
    main()
