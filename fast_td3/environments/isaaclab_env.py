from contextlib import contextmanager
from typing import Optional

import gymnasium as gym
import torch

_APP_LAUNCHER = None
_SIMULATION_APP = None
_SIMULATION_DEVICE = None


def _register_unitree_alias_if_needed(task_name: str) -> str:
    try:
        from unitree_bridge import FASTTD3_UNITREE_ALIAS, register_unitree_alias
    except ModuleNotFoundError:
        from fast_td3.unitree_bridge import FASTTD3_UNITREE_ALIAS, register_unitree_alias

    if task_name == FASTTD3_UNITREE_ALIAS:
        return register_unitree_alias()
    return task_name


def _ensure_simulation_app(device: str):
    global _APP_LAUNCHER, _SIMULATION_APP, _SIMULATION_DEVICE

    if _SIMULATION_APP is None:
        from isaaclab.app import AppLauncher

        _APP_LAUNCHER = AppLauncher(headless=True, device=device)
        _SIMULATION_APP = _APP_LAUNCHER.app
        _SIMULATION_DEVICE = device
    elif device != _SIMULATION_DEVICE:
        raise ValueError(
            "IsaacLabEnv can only reuse one Isaac Sim app per process; "
            f"already launched on {_SIMULATION_DEVICE!r}, requested {device!r}."
        )

    return _SIMULATION_APP


class IsaacLabEnv:
    """Wrapper for IsaacLab Unitree environments used by FastTD3."""

    def __init__(
        self,
        task_name: str,
        device: str,
        num_envs: int,
        seed: int,
        action_bounds: Optional[float] = None,
    ):
        _ensure_simulation_app(device)

        task_name = _register_unitree_alias_if_needed(task_name)

        import isaaclab_tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=num_envs,
        )
        env_cfg.seed = seed
        self.env_cfg = env_cfg
        self.seed = seed
        self.envs = gym.make(task_name, cfg=env_cfg, render_mode=None)

        self.num_envs = self.envs.unwrapped.num_envs
        self.max_episode_steps = self.envs.unwrapped.max_episode_length
        self.action_bounds = action_bounds
        self.num_obs = self.envs.unwrapped.single_observation_space["policy"].shape[0]
        self.asymmetric_obs = "critic" in self.envs.unwrapped.single_observation_space
        if self.asymmetric_obs:
            self.num_privileged_obs = self.envs.unwrapped.single_observation_space[
                "critic"
            ].shape[0]
        else:
            self.num_privileged_obs = 0
        self.num_actions = self.envs.unwrapped.single_action_space.shape[0]

    def snapshot_curriculum(self) -> dict:
        snapshot = {}
        snapshot["commands"] = self._snapshot_command_ranges()
        snapshot["terrain"] = self._snapshot_terrain_levels()
        return snapshot

    def apply_curriculum_snapshot(self, snapshot: Optional[dict]) -> None:
        if not snapshot:
            return
        self._apply_command_ranges(snapshot.get("commands", {}))
        self._apply_terrain_levels(snapshot.get("terrain", {}))

    @contextmanager
    def frozen_curriculum(self):
        curriculum_manager = getattr(self.envs.unwrapped, "curriculum_manager", None)
        if curriculum_manager is None:
            yield
            return

        original_compute = curriculum_manager.compute
        curriculum_manager.compute = lambda env_ids=None: None
        try:
            yield
        finally:
            curriculum_manager.compute = original_compute

    def reset(self, random_start_init: bool = True) -> torch.Tensor:
        obs_dict, _ = self.envs.reset()
        if random_start_init:
            self.envs.unwrapped.episode_length_buf = torch.randint_like(
                self.envs.unwrapped.episode_length_buf, high=int(self.max_episode_steps)
            )
        return obs_dict["policy"]

    def reset_with_critic_obs(
        self, random_start_init: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _ = self.envs.reset()
        if random_start_init:
            self.envs.unwrapped.episode_length_buf = torch.randint_like(
                self.envs.unwrapped.episode_length_buf, high=int(self.max_episode_steps)
            )
        return obs_dict["policy"], obs_dict["critic"]

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.action_bounds is not None:
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, infos = self.envs.step(actions)
        dones = (terminations | truncations).to(dtype=torch.long)
        obs = obs_dict["policy"]
        critic_obs = obs_dict["critic"] if self.asymmetric_obs else None
        info_ret = dict(infos)
        info_ret["time_outs"] = truncations
        info_ret["observations"] = {"critic": critic_obs}
        # NOTE: There's really no way to get the raw observations from IsaacLab
        # We just use the 'reset_obs' as next_obs, unfortunately.
        # See https://github.com/isaac-sim/IsaacLab/issues/1362
        info_ret["observations"]["raw"] = {
            "obs": obs,
            "critic_obs": critic_obs,
        }
        return obs, rew, dones, info_ret

    def render(self):
        raise NotImplementedError(
            "We don't support rendering for IsaacLab environments"
        )

    def close(self):
        self.envs.close()

    def _snapshot_command_ranges(self) -> dict:
        env = self.envs.unwrapped
        commands = {}
        for command_name in ("base_velocity",):
            try:
                command_term = env.command_manager.get_term(command_name)
            except (AttributeError, KeyError):
                continue
            command_data = {}
            for attr_name in ("ranges", "limit_ranges"):
                ranges = getattr(command_term.cfg, attr_name, None)
                if ranges is None:
                    continue
                command_data[attr_name] = {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in vars(ranges).items()
                }
            commands[command_name] = command_data
        return commands

    def _apply_command_ranges(self, commands: dict) -> None:
        env = self.envs.unwrapped
        for command_name, command_data in commands.items():
            try:
                command_term = env.command_manager.get_term(command_name)
            except (AttributeError, KeyError):
                continue
            for attr_name, ranges_data in command_data.items():
                ranges = getattr(command_term.cfg, attr_name, None)
                if ranges is None:
                    continue
                for key, value in ranges_data.items():
                    if not hasattr(ranges, key):
                        continue
                    if isinstance(value, list):
                        value = tuple(value)
                    setattr(ranges, key, value)

    def curriculum_scalars(self) -> dict[str, float]:
        scalars = {}
        env = self.envs.unwrapped
        try:
            command_term = env.command_manager.get_term("base_velocity")
        except (AttributeError, KeyError):
            command_term = None
        if command_term is not None:
            ranges = command_term.cfg.ranges
            for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                value = getattr(ranges, name, None)
                if value is None:
                    continue
                scalars[f"Curriculum/base_velocity/{name}_min"] = float(value[0])
                scalars[f"Curriculum/base_velocity/{name}_max"] = float(value[1])

        terrain = getattr(env.scene, "terrain", None)
        if terrain is not None and hasattr(terrain, "terrain_levels"):
            levels = terrain.terrain_levels.float()
            scalars["Curriculum/terrain/level_mean"] = levels.mean().item()
            scalars["Curriculum/terrain/level_max"] = levels.max().item()

        try:
            episode_sums = env.reward_manager._episode_sums
            reward_term = env.reward_manager.get_term_cfg("track_lin_vel_xy")
        except (AttributeError, KeyError):
            return scalars
        reward = episode_sums["track_lin_vel_xy"].mean() / env.max_episode_length_s
        scalars["Curriculum/base_velocity/track_lin_vel_xy_mean"] = reward.item()
        scalars["Curriculum/base_velocity/track_lin_vel_xy_threshold"] = float(
            reward_term.weight * 0.8
        )
        return scalars

    def _snapshot_terrain_levels(self) -> dict:
        terrain = getattr(self.envs.unwrapped.scene, "terrain", None)
        if terrain is None or not hasattr(terrain, "terrain_levels"):
            return {}
        return {"terrain_levels": terrain.terrain_levels.detach().cpu()}

    def _apply_terrain_levels(self, terrain_data: dict) -> None:
        terrain = getattr(self.envs.unwrapped.scene, "terrain", None)
        if terrain is None or not hasattr(terrain, "terrain_levels"):
            return
        source_levels = terrain_data.get("terrain_levels")
        if source_levels is None:
            return
        target_levels = terrain.terrain_levels
        source_levels = source_levels.to(device=target_levels.device, dtype=target_levels.dtype)
        if source_levels.numel() < target_levels.numel():
            return
        if source_levels.shape != target_levels.shape:
            source_levels = source_levels[: target_levels.numel()].reshape_as(target_levels)

        target_levels.copy_(source_levels)
        if getattr(terrain, "terrain_origins", None) is None:
            return

        env_ids = torch.arange(target_levels.shape[0], device=target_levels.device)
        no_move = torch.zeros_like(target_levels, dtype=torch.bool)
        terrain.update_env_origins(env_ids, no_move, no_move)
