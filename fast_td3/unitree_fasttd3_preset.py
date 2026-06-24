"""Canonical defaults for FastTD3 on Unitree G1-29dof velocity."""

from typing import Final


UNITREE_FASTTD3_TASK: Final[str] = "Unitree-G1-29dof-Velocity"
UNITREE_FASTTD3_ENV_NAME: Final[str] = "Isaac-Unitree-G1-29dof-Velocity"
UNITREE_FASTTD3_EXP_NAME: Final[str] = "UnitreeFastTD3"
UNITREE_FASTTD3_PROJECT: Final[str] = "UnitreeFastTD3"
UNITREE_FASTTD3_LOG_EXPERIMENT_NAME: Final[str] = "unitree_g1_29dof_velocity"

UNITREE_FASTTD3_NUM_ENVS: Final[int] = 2048
UNITREE_FASTTD3_TOTAL_TIMESTEPS: Final[int] = 50000
UNITREE_FASTTD3_EVERY_ENV_BUFFER_SIZE: Final[int] = 1024
UNITREE_FASTTD3_TOTAL_BATCH_SIZE: Final[int] = 32768
UNITREE_FASTTD3_NUM_STEPS: Final[int] = 1
UNITREE_FASTTD3_NUM_UPDATES: Final[int] = 4
UNITREE_FASTTD3_RENDER_INTERVAL: Final[int] = 0
UNITREE_FASTTD3_EVAL_INTERVAL: Final[int] = 1000
UNITREE_FASTTD3_ACTION_BOUNDS: Final[float] = 1.0
UNITREE_FASTTD3_COMPILE: Final[bool] = True
UNITREE_FASTTD3_COMPILE_MODE: Final[str] = "reduce-overhead"
UNITREE_FASTTD3_OBS_NORMALIZATION: Final[bool] = False
UNITREE_FASTTD3_REWARD_NORMALIZATION: Final[bool] = False

UNITREE_FASTTD3_LAUNCHER_DEFAULTS: Final[tuple[tuple[str, str], ...]] = (
    ("--num_envs", str(UNITREE_FASTTD3_NUM_ENVS)),
    ("--total_timesteps", str(UNITREE_FASTTD3_TOTAL_TIMESTEPS)),
    ("--buffer_size", str(UNITREE_FASTTD3_EVERY_ENV_BUFFER_SIZE)),
    ("--batch_size", str(UNITREE_FASTTD3_TOTAL_BATCH_SIZE)),
    ("--num_steps", str(UNITREE_FASTTD3_NUM_STEPS)),
    ("--num_updates", str(UNITREE_FASTTD3_NUM_UPDATES)),
    ("--render_interval", str(UNITREE_FASTTD3_RENDER_INTERVAL)),
    ("--eval_interval", str(UNITREE_FASTTD3_EVAL_INTERVAL)),
    ("--action_bounds", str(UNITREE_FASTTD3_ACTION_BOUNDS)),
    ("--compile_mode", UNITREE_FASTTD3_COMPILE_MODE),
)

UNITREE_FASTTD3_BOOL_DEFAULTS: Final[tuple[tuple[str, str, bool, bool], ...]] = (
    ("--compile", "--no_compile", UNITREE_FASTTD3_COMPILE, True),
    ("--use_wandb", "--no_use_wandb", False, True),
    ("--obs_normalization", "--no_obs_normalization", UNITREE_FASTTD3_OBS_NORMALIZATION, True),
    ("--reward_normalization", "--no_reward_normalization", UNITREE_FASTTD3_REWARD_NORMALIZATION, False),
)
