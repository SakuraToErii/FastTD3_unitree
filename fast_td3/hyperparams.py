import argparse
import os
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class BaseArgs:
    env_name: str = "Isaac-Unitree-G1-29dof-Velocity"
    """the id of the IsaacLab/Unitree environment"""
    agent: str = "fasttd3"
    """the agent to use: currently support [fasttd3, fasttd3_simbav2]"""
    seed: int = 3407
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, cudnn.deterministic=True; default False to match PPO"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    device_rank: int = 0
    """the rank of the device"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    project: str = "FastTD3"
    """the project name"""
    use_wandb: bool = True
    """whether to use wandb"""
    use_neptune: bool = False
    """whether to use neptune logging"""
    neptune_project: str = "FastTD3"
    """the neptune project name"""
    log_tensorboard: bool = True
    """whether to write TensorBoard logs into save_dir"""
    log_interval: int = 500
    """the interval to write training logs"""
    checkpoint_path: str = None
    """the path to the checkpoint file"""
    num_envs: int = 2048
    """the number of environments to run in parallel"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    critic_learning_rate: float = 3e-4
    """the learning rate of the critic"""
    actor_learning_rate: float = 3e-4
    """the learning rate for the actor"""
    critic_learning_rate_end: float = 3e-4
    """the learning rate of the critic at the end of training"""
    actor_learning_rate_end: float = 3e-4
    """the learning rate for the actor at the end of training"""
    buffer_size: int = 1024
    """the replay memory buffer size per environment"""
    num_steps: int = 1
    """the number of steps to use for the multi-step return"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.1
    """target smoothing coefficient"""
    batch_size: int = 32768
    """the batch size of sample from the replay memory"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    std_min: float = 0.001
    """the minimum scale of noise"""
    std_max: float = 0.3
    """the maximum scale of noise"""
    learning_starts: int = 10
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    num_updates: int = 4
    """the number of updates to perform per step"""
    init_scale: float = 0.01
    """the scale of the initial parameters"""
    num_atoms: int = 251
    """the number of atoms"""
    v_min: float = -10.0
    """the minimum value of the support"""
    v_max: float = 10.0
    """the maximum value of the support"""
    critic_hidden_dim: int = 1024
    """the hidden dimension of the critic network"""
    actor_hidden_dim: int = 512
    """the hidden dimension of the actor network"""
    critic_num_blocks: int = 2
    """(SimbaV2 only) the number of blocks in the critic network"""
    actor_num_blocks: int = 1
    """(SimbaV2 only) the number of blocks in the actor network"""
    use_cdq: bool = True
    """whether to use Clipped Double Q-learning"""
    measure_burnin: int = 3
    """Number of burn-in iterations for speed measure."""
    eval_interval: int = 1000
    """the interval to evaluate the model"""
    eval_num_envs: int = 128
    """the number of environments to use in the separate evaluation process"""
    eval_seed_offset: int = 1000003
    """large prime added to the training seed for the evaluation environment"""
    render_interval: int = 0
    """the interval to render the model; IsaacLab rendering is unsupported here"""
    compile: bool = True
    """whether to use torch.compile."""
    compile_mode: str = "reduce-overhead"
    """the mode of torch.compile."""
    obs_normalization: bool = False
    """whether to enable observation normalization"""
    reward_normalization: bool = False
    """whether to enable reward normalization"""
    use_grad_norm_clipping: bool = False
    """whether to use gradient norm clipping."""
    max_grad_norm: float = 0.0
    """the maximum gradient norm"""
    amp: bool = True
    """whether to use amp"""
    amp_dtype: str = "bf16"
    """the dtype of the amp"""
    disable_bootstrap: bool = False
    """Whether to disable bootstrap in the critic learning"""

    use_tanh: bool = False
    """if True, Actor output uses Tanh (bounded [-1,1]); requires action_bounds to be set.
    If False, Actor output is unbounded (matches PPO); action_bounds must be None."""

    action_bounds: float = None
    """when use_tanh=True, scales clamped [-1,1] output to [-action_bounds, action_bounds].
    Must be None when use_tanh=False. Must not be None when use_tanh=True."""


    weight_decay: float = 0.1
    """the weight decay of the optimizer"""
    save_interval: int = 5000
    """the interval to save the model"""
    save_dir: str = "models"
    """the directory where checkpoints are saved"""
    checkpoint_prefix: str = None
    """the checkpoint filename prefix; defaults to the run name"""
    save_final_as_step: bool = False
    """save the final checkpoint as <prefix>_<global_step>.pt instead of <run_name>_final.pt"""
    export_unitree_params: bool = False
    """export Unitree deploy/env/agent params into save_dir/params"""
    random_start_init: bool = True
    """randomize initial episode lengths after reset, matching Unitree/RSL-RL PPO training"""

    sim_type: str = ""
    """SimNorm mode: '', sim_actor, sim_critic, or sim_both"""
    sim_dimension: int = 64
    """the dimension of the sim module"""
    critic_seq_len: int = 8
    """the number of simplices used in the critic head"""
    actor_seq_len: int = 8
    """the number of simplices used in the actor head"""


def _parse_bool(value: str) -> bool:
    if value.lower() in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _option_aliases(name: str) -> list[str]:
    aliases = [f"--{name}"]
    hyphen_name = name.replace("_", "-")
    if hyphen_name != name:
        aliases.append(f"--{hyphen_name}")
    return aliases


def _argparse_cli(args_class: type[BaseArgs]) -> BaseArgs:
    parser = argparse.ArgumentParser()
    defaults: dict[str, Any] = {}
    for field in fields(args_class):
        name = field.name
        default = getattr(args_class, name)
        defaults[name] = default
        option_aliases = _option_aliases(name)

        if isinstance(default, bool):
            group = parser.add_mutually_exclusive_group()
            group.add_argument(
                *option_aliases,
                dest=name,
                nargs="?",
                const=True,
                type=_parse_bool,
            )
            group.add_argument(
                *_option_aliases(f"no_{name}"),
                dest=name,
                action="store_false",
            )
            continue

        value_type = field.type if default is None else type(default)
        parser.add_argument(*option_aliases, dest=name, type=value_type)

    parser.set_defaults(**defaults)
    return args_class(**vars(parser.parse_args()))


def _cli(args_class: type[BaseArgs]) -> BaseArgs:
    return _argparse_cli(args_class)


def get_args():
    """
    Parse command-line arguments for the Unitree/IsaacLab single-GPU trainer.
    """
    args = _cli(BaseArgs)
    if not args.env_name.startswith("Isaac-"):
        raise ValueError(
            "fast_td3/train.py only supports IsaacLab/Unitree environments; "
            f"got env_name={args.env_name!r}"
        )
    return args
