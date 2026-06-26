#!/usr/bin/env python3
"""Headless smoke check: confirm terminal observations reach the FastTD3 wrapper.

Launches the IsaacLab sim app, builds the FastTD3 Unitree alias env through the real
``IsaacLabEnv`` wrapper, steps until at least one sub-env is done, and asserts that
``infos["observations"]["terminal"]`` is populated with the right shapes. Exits 0 on
success.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FASTTD3_DIR = ROOT / "fast_td3"
for p in (str(ROOT), str(FASTTD3_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from unitree_bridge import FASTTD3_UNITREE_ALIAS, register_unitree_alias


def main() -> int:
    from environments.isaaclab_env import IsaacLabEnv

    device = "cuda:0"
    # IsaacLabEnv.__init__ launches the SimulationApp first, then registers the
    # alias — do not call register_unitree_alias() here, or isaaclab imports fail
    # before the sim app exposes the `omni` modules.
    envs = IsaacLabEnv(
        FASTTD3_UNITREE_ALIAS,
        device=device,
        num_envs=8,
        seed=0,
        action_bounds=None,
    )
    print(
        f"[smoke] env built: num_envs={envs.num_envs} num_obs={envs.num_obs} "
        f"asymmetric={envs.asymmetric_obs}"
    )

    # Imported after the sim app is up so `omni.*` is available to isaaclab.envs.
    from environments.terminal_obs_env import FastTD3ManagerBasedRLEnv

    unwrapped = envs.envs.unwrapped
    print(f"[smoke] underlying env class: {type(unwrapped).__name__}")
    assert isinstance(unwrapped, FastTD3ManagerBasedRLEnv), (
        f"expected FastTD3ManagerBasedRLEnv, got {type(unwrapped).__name__}"
    )

    envs.reset(random_start_init=True)
    actions = torch.zeros(envs.num_envs, envs.num_actions, device=device)

    seen_terminal = False
    saw_done = False
    for step in range(envs.max_episode_steps + 5):
        actions = torch.zeros(envs.num_envs, envs.num_actions, device=device)
        obs, rew, dones, infos = envs.step(actions)
        terminal = infos["observations"].get("terminal")
        n_done = int(dones.sum().item())
        if n_done == 0:
            continue
        saw_done = True
        print(f"[smoke] step {step}: n_done={n_done} terminal_present={terminal is not None}")
        assert terminal is not None, "dones occurred but terminal obs is None"
        assert terminal["obs"].shape[0] == n_done, (
            f"terminal obs rows {terminal['obs'].shape[0]} != n_done {n_done}"
        )
        assert terminal["obs"].shape[1] == envs.num_obs, (
            f"terminal obs dim {terminal['obs'].shape[1]} != num_obs {envs.num_obs}"
        )
        if envs.asymmetric_obs:
            assert terminal["critic_obs"] is not None, "asymmetric but critic terminal None"
            assert terminal["critic_obs"].shape[0] == n_done
        else:
            assert terminal["critic_obs"] is None
        seen_terminal = True
        break

    envs.close()
    if not saw_done:
        print("[smoke] WARN: no done occurred within max_steps; could not validate terminal obs")
        return 2
    if not seen_terminal:
        print("[smoke] FAIL: done occurred but terminal obs never validated")
        return 1
    print("[smoke] OK: terminal observations flow through and have correct shapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())