import argparse
import json
import sys
from pathlib import Path

import torch


def _prepend_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    sys.path[:] = [path for path in sys.path if path != repo_root_str]
    sys.path.insert(0, repo_root_str)
    return repo_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Unitree FastTD3 checkpoint in a separate Isaac Sim process."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the FastTD3 checkpoint to evaluate.")
    parser.add_argument("--env-name", required=True, help="IsaacLab/Unitree task id.")
    parser.add_argument("--device", default="cuda:0", help="IsaacLab device.")
    parser.add_argument("--num-envs", type=int, required=True, help="Number of parallel eval environments.")
    parser.add_argument("--seed", type=int, required=True, help="Evaluation seed.")
    parser.add_argument(
        "--action-bounds",
        type=float,
        default=1.0,
        help="Action clamp scale used by the training env.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable autocast during evaluation.")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16", help="Autocast dtype.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _prepend_repo_root()

    from fast_td3.environments.isaaclab_env import IsaacLabEnv
    from fast_td3.unitree_policy import load_policy

    device = torch.device(args.device)
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    envs = IsaacLabEnv(
        args.env_name,
        args.device,
        args.num_envs,
        args.seed,
        action_bounds=args.action_bounds,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    envs.apply_curriculum_snapshot(checkpoint.get("curriculum_snapshot"))
    policy = load_policy(args.checkpoint).to(device).eval()

    num_envs = envs.num_envs
    episode_returns = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    done_masks = torch.zeros(num_envs, dtype=torch.bool, device=device)

    try:
        with envs.frozen_curriculum():
            obs = envs.reset(random_start_init=False)
            for _ in range(envs.max_episode_steps):
                with torch.inference_mode(), torch.amp.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                ):
                    actions = policy(obs.to(device))

                next_obs, rewards, dones, _ = envs.step(actions.float())
                rewards = rewards.to(device)
                dones = dones.to(device).bool()

                episode_returns = torch.where(
                    ~done_masks, episode_returns + rewards, episode_returns
                )
                episode_lengths = torch.where(
                    ~done_masks, episode_lengths + 1, episode_lengths
                )
                done_masks = torch.logical_or(done_masks, dones)
                if done_masks.all():
                    break
                obs = next_obs
    finally:
        envs.close()

    result = {
        "eval_avg_return": episode_returns.mean().item(),
        "eval_avg_length": episode_lengths.mean().item(),
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
