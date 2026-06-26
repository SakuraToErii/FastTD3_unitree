import os
import sys
from pathlib import Path


UNITREE_TASK = "Unitree-G1-29dof-Velocity"
FASTTD3_UNITREE_ALIAS = "Isaac-Unitree-G1-29dof-Velocity"
UNITREE_LOG_EXPERIMENT_NAME = "unitree_g1_29dof_velocity"
DEFAULT_UNITREE_RL_LAB_PATH = Path("/home/ordis/projects/unitree_rl_lab")

# FastTD3 alias routes through a ManagerBasedRLEnv subclass that captures the true
# terminal observation before IsaacLab resets the env (see
# fast_td3/environments/terminal_obs_env.py). The stock Unitree task keeps its
# original ``isaaclab.envs:ManagerBasedRLEnv`` entry point and is unaffected.


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def unitree_rl_lab_root(path: str | os.PathLike[str] | None = None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    if os.environ.get("UNITREE_RL_LAB_PATH"):
        return Path(os.environ["UNITREE_RL_LAB_PATH"]).expanduser().resolve()
    return DEFAULT_UNITREE_RL_LAB_PATH


def add_unitree_source_path(path: str | os.PathLike[str] | None = None) -> Path:
    root = unitree_rl_lab_root(path)
    source = root / "source" / "unitree_rl_lab"
    if not source.exists():
        raise FileNotFoundError(f"Unitree RL Lab source not found: {source}")
    source_s = str(source)
    if source_s not in sys.path:
        sys.path.insert(0, source_s)
    return root


def add_fasttd3_script_path() -> None:
    root = repo_root()
    for path in (root, root / "fast_td3"):
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)


def register_unitree_alias(path: str | os.PathLike[str] | None = None) -> str:
    add_unitree_source_path(path)

    import gymnasium as gym
    import unitree_rl_lab.tasks  # noqa: F401
    try:
        from environments.terminal_obs_env import FastTD3ManagerBasedRLEnv
    except ModuleNotFoundError:
        from fast_td3.environments.terminal_obs_env import FastTD3ManagerBasedRLEnv

    try:
        gym.spec(FASTTD3_UNITREE_ALIAS)
        return FASTTD3_UNITREE_ALIAS
    except gym.error.Error:
        pass

    spec = gym.spec(UNITREE_TASK)
    gym.register(
        id=FASTTD3_UNITREE_ALIAS,
        entry_point=FastTD3ManagerBasedRLEnv,
        disable_env_checker=True,
        kwargs=dict(spec.kwargs),
    )
    return FASTTD3_UNITREE_ALIAS


def default_policy_dir(path: str | os.PathLike[str] | None = None) -> Path:
    return (
        unitree_rl_lab_root(path)
        / "deploy"
        / "robots"
        / "g1_29dof"
        / "config"
        / "policy"
        / "velocity"
    )
