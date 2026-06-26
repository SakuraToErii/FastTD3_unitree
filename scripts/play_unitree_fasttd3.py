#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import site
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _prioritize_venv_site_packages() -> None:
    """Keep Isaac Sim's pip_prebundle from shadowing installed Python deps."""
    site_paths: list[str] = []
    for path in site.getsitepackages():
        site_paths.append(str(Path(path).resolve()))
    user_site = site.getusersitepackages()
    if user_site:
        site_paths.append(str(Path(user_site).resolve()))

    insert_at = 0
    protected_paths = {str(ROOT)}
    for idx, path in enumerate(sys.path):
        if str(Path(path or ".").resolve()) in protected_paths:
            insert_at = idx + 1

    for site_path in reversed(site_paths):
        for existing in list(sys.path):
            if str(Path(existing or ".").resolve()) == site_path:
                sys.path.remove(existing)
                sys.path.insert(insert_at, existing)

    typing_extensions = sys.modules.get("typing_extensions")
    if typing_extensions is None:
        return
    module_file = getattr(typing_extensions, "__file__", "")
    if "pip_prebundle" in module_file:
        del sys.modules["typing_extensions"]


_prioritize_venv_site_packages()

from isaaclab.app import AppLauncher

from fast_td3.unitree_bridge import (
    UNITREE_LOG_EXPERIMENT_NAME,
    UNITREE_TASK,
    add_unitree_source_path,
    unitree_rl_lab_root,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play and export a Unitree FastTD3 checkpoint.")
    parser.add_argument("--unitree_rl_lab_path", default=None, help="Path to unitree_rl_lab.")
    parser.add_argument("--task", default=UNITREE_TASK, help="Unitree Isaac Lab task name.")
    parser.add_argument("--experiment_name", default=UNITREE_LOG_EXPERIMENT_NAME)
    parser.add_argument("--load_run", default=None, help="Run directory under logs/rsl_rl/<experiment_name>.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path or filename inside --load_run.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--video", action="store_true", default=False, help="Record one play video.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
    parser.add_argument("--action_bounds", type=float, default=None, help="Override checkpoint action bound (only meaningful when use_tanh=True).")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument("--export_only", action="store_true", help="Only export exported/policy.*; do not launch sim.")
    parser.add_argument("--skip_export", action="store_true", help="Skip exported/policy.pt and policy.onnx export.")
    parser.add_argument("--no_export_jit", action="store_true", help="Do not export exported/policy.pt.")
    parser.add_argument("--no_export_onnx", action="store_true", help="Do not export exported/policy.onnx.")
    parser.add_argument("--skip_verify", action="store_true", help="Skip saved TorchScript verification.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video:
        args.enable_cameras = True
    return args


def _log_root(args: argparse.Namespace) -> Path:
    return unitree_rl_lab_root(args.unitree_rl_lab_path) / "logs" / "rsl_rl" / args.experiment_name


def _latest_run_dir(log_root: Path) -> Path:
    runs = [path for path in log_root.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run directories found in {log_root}")
    return sorted(runs, key=lambda path: path.name)[-1]


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = [path for path in run_dir.glob("model_*.pt") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {run_dir}")
    return sorted(checkpoints, key=lambda path: (_checkpoint_step(path), path.name))[-1]


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    log_root = _log_root(args)
    run_dir = log_root / args.load_run if args.load_run else None

    if args.checkpoint:
        checkpoint = Path(args.checkpoint).expanduser()
        if checkpoint.is_file():
            return checkpoint.resolve()
        if checkpoint.is_absolute():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if run_dir is None:
            run_dir = _latest_run_dir(log_root)
        checkpoint = run_dir / checkpoint
        if checkpoint.is_file():
            return checkpoint.resolve()
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if run_dir is None:
        run_dir = _latest_run_dir(log_root)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return _latest_checkpoint(run_dir).resolve()


def _verify_scripted_policy(policy, scripted_policy, obs_dim: int) -> float:
    import torch

    from fast_td3.unitree_policy import UnitreeTorchScriptPolicy

    torch.manual_seed(0)
    # Compare against the export wrapper, which bakes in action_bounds — not
    # the raw Policy.forward (that does not apply action_bounds).
    export_policy = UnitreeTorchScriptPolicy(policy).to("cpu").eval()
    max_abs_diff = 0.0
    test_obs = (
        torch.zeros(1, obs_dim, dtype=torch.float32),
        torch.randn(8, obs_dim, dtype=torch.float32),
    )
    with torch.inference_mode():
        for obs in test_obs:
            expected = export_policy(obs)
            actual = scripted_policy(obs)
            diff = torch.max(torch.abs(expected - actual)).item()
            max_abs_diff = max(max_abs_diff, diff)
            if not torch.allclose(expected, actual, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    "Saved TorchScript policy does not match the FastTD3 checkpoint "
                    f"(max_abs_diff={max_abs_diff:.3g})."
                )
    return max_abs_diff


def _export_policy(args: argparse.Namespace, checkpoint_path: Path) -> None:
    import torch

    from fast_td3.unitree_policy import (
        checkpoint_actor_dims,
        export_policy_as_jit,
        export_policy_as_onnx,
        load_policy,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _, obs_dim, _ = checkpoint_actor_dims(checkpoint)
    policy = load_policy(checkpoint_path).to("cpu").eval()
    if args.action_bounds is not None:
        policy.action_bounds = args.action_bounds

    if policy.action_bounds is not None:
        print(f"[INFO] Baking action_bounds={policy.action_bounds} into exported policy")
    export_dir = checkpoint_path.parent / "exported"
    if not args.no_export_jit:
        pt_path = export_policy_as_jit(policy, export_dir, filename="policy.pt")
        print(f"[INFO] Exported TorchScript policy: {pt_path}")
        if not args.skip_verify:
            loaded_policy = torch.jit.load(str(pt_path), map_location="cpu").eval()
            max_abs_diff = _verify_scripted_policy(policy, loaded_policy, obs_dim)
            print(f"[INFO] Verified TorchScript output: max_abs_diff={max_abs_diff:.3g}")
    if not args.no_export_onnx:
        onnx_path = export_policy_as_onnx(
            policy,
            export_dir,
            obs_dim,
            filename="policy.onnx",
            opset_version=args.opset,
        )
        print(f"[INFO] Exported ONNX policy: {onnx_path}")

    deploy_cfg = checkpoint_path.parent / "params" / "deploy.yaml"
    if not deploy_cfg.exists():
        print(f"[WARN] Missing deploy config for g1_ctrl: {deploy_cfg}")


def _policy_obs(obs):
    if isinstance(obs, dict):
        return obs["policy"]
    return obs


def _play(args: argparse.Namespace, checkpoint_path: Path, simulation_app) -> None:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import unitree_rl_lab.tasks  # noqa: F401
    from fast_td3.unitree_policy import load_policy
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.utils.dict import print_dict
    from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args.video:
        video_kwargs = {
            "video_folder": str(checkpoint_path.parent / "videos" / "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording play video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    action_bounds = args.action_bounds
    if action_bounds is None:
        action_bounds = checkpoint.get("args", {}).get("action_bounds", None)

    device = env.unwrapped.device
    policy = load_policy(checkpoint_path).to(device).eval()
    dt = env.unwrapped.step_dt

    print(f"[INFO] Loading FastTD3 checkpoint from: {checkpoint_path}")
    print(f"[INFO] Using action_bounds={action_bounds}")

    obs, _ = env.reset()
    obs = _policy_obs(obs)
    timestep = 0
    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                if action_bounds is not None:
                    actions = torch.clamp(actions, -1.0, 1.0) * action_bounds
                obs, _, _, _, _ = env.step(actions)
                obs = _policy_obs(obs)

            if args.video:
                timestep += 1
                if timestep >= args.video_length:
                    break

            sleep_time = dt - (time.time() - start_time)
            if args.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    add_unitree_source_path(args.unitree_rl_lab_path)
    checkpoint_path = _resolve_checkpoint(args)
    print(f"[INFO] FastTD3 checkpoint: {checkpoint_path}")

    if not args.skip_export:
        _export_policy(args, checkpoint_path)
    if args.export_only:
        return

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        _play(args, checkpoint_path, simulation_app)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
