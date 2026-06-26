import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
from .fast_td3_utils import EmpiricalNormalization
from .fast_td3 import Actor
from .fast_td3_simbav2 import Actor as ActorSimbaV2


class Policy(nn.Module):
    """Inference wrapper: actor + obs normalizer.

    ``use_tanh`` is read from the checkpoint's ``args`` dict so that the
    exported ONNX/TorchScript policy matches the training-time actor:
    - ``use_tanh=True``  → ``Tanh(MLP(normalizer(obs)))``, bounded [-1, 1]
    - ``use_tanh=False`` → ``MLP(normalizer(obs))``, unbounded (PPO-style)
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        args: dict,
        agent: str = "fasttd3",
    ):
        super().__init__()

        num_envs = args["num_envs"]
        actor_hidden_dim = args["actor_hidden_dim"]

        actor_kwargs = dict(
            n_obs=n_obs,
            n_act=n_act,
            num_envs=num_envs,
            device="cpu",
            hidden_dim=actor_hidden_dim,
            std_min=args.get("std_min", 0.05),
            std_max=args.get("std_max", 0.8),
        )

        if agent == "fasttd3":
            actor_cls = Actor
            actor_kwargs.update(
                {
                    "init_scale": args["init_scale"],
                    "sim_type": args.get("sim_type", ""),
                    "sim_dimension": args.get("sim_dimension", 64),
                    "seq_len": args.get("actor_seq_len", 8),
                    "use_tanh": args.get("use_tanh", False),
                }
            )
        elif agent == "fasttd3_simbav2":
            actor_cls = ActorSimbaV2

            actor_num_blocks = args["actor_num_blocks"]
            actor_kwargs["use_tanh"] = args.get("use_tanh", False)
            actor_kwargs.update(
                {
                    "scaler_init": math.sqrt(2.0 / actor_hidden_dim),
                    "scaler_scale": math.sqrt(2.0 / actor_hidden_dim),
                    "alpha_init": 1.0 / (actor_num_blocks + 1),
                    "alpha_scale": 1.0 / math.sqrt(actor_hidden_dim),
                    "expansion": 4,
                    "c_shift": 3.0,
                    "num_blocks": actor_num_blocks,
                }
            )
        else:
            raise ValueError(f"Agent {agent} not supported")

        self.actor = actor_cls(
            **actor_kwargs,
        )
        self.obs_normalizer = EmpiricalNormalization(shape=n_obs, device="cpu")

        self.actor.eval()
        self.obs_normalizer.eval()

        # action_bounds scales the Tanh-bounded actor output to
        # [-action_bounds, action_bounds]. At training time this scaling is
        # applied inside IsaacLabEnv.step; the C++ deploy side does NOT apply
        # it (only the per-joint scale from deploy.yaml), so it must be baked
        # into the exported policy to keep deployment consistent with training.
        self.action_bounds = args.get("action_bounds", None)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = self.obs_normalizer(obs)
        actions = self.actor(norm_obs)
        return actions

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.distributions.Normal:
        actions = self.forward(obs)
        return torch.distributions.Normal(actions, torch.ones_like(actions) * 1e-8)


class FrozenEmpiricalNormalizer(nn.Module):
    """TorchScript-friendly inference-only copy of EmpiricalNormalization."""

    def __init__(self, normalizer: EmpiricalNormalization):
        super().__init__()
        self.eps = float(normalizer.eps)
        self.register_buffer("_mean", normalizer._mean.detach().clone())
        self.register_buffer("_std", normalizer._std.detach().clone())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self._mean) / (self._std + self.eps)


class UnitreeTorchScriptPolicy(nn.Module):
    """Unitree/RSL-RL style TorchScript policy: forward(obs) -> actions.

    When the source policy has ``action_bounds`` set (i.e. ``use_tanh=True``
    with a non-None ``action_bounds``), the exported policy bakes in the
    ``clamp(actions, -1, 1) * action_bounds`` mapping so its output is in
    ``[-action_bounds, action_bounds]`` — exactly the range the training
    environment consumed. The C++ deploy side only applies the per-joint scale
    from ``deploy.yaml`` and has no knowledge of ``action_bounds``.
    """

    def __init__(self, policy: Policy):
        super().__init__()
        self.actor = copy.deepcopy(policy.actor)
        if isinstance(policy.obs_normalizer, EmpiricalNormalization):
            self.normalizer = FrozenEmpiricalNormalizer(policy.obs_normalizer)
        else:
            self.normalizer = nn.Identity()
        self.actor.eval()
        self.normalizer.eval()

        action_bounds = getattr(policy, "action_bounds", None)
        # Bool attribute (TorchScript-compatible) gates the clamp+scale branch;
        # the scalar buffer carries the value. ONNX constant-folds the branch
        # away when action_bounds is None.
        self.apply_bounds = action_bounds is not None
        self.register_buffer(
            "action_bounds",
            torch.tensor(float(action_bounds) if action_bounds is not None else 1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actions = self.actor(self.normalizer(obs))
        if self.apply_bounds:
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_bounds
        return actions

    @torch.jit.export
    def reset(self):
        pass


def checkpoint_actor_dims(torch_checkpoint: dict) -> tuple[str, int, int]:
    agent = torch_checkpoint["args"].get("agent", "fasttd3")
    actor_state_dict = torch_checkpoint["actor_state_dict"]
    if agent == "fasttd3":
        n_obs = actor_state_dict["net.0.weight"].shape[-1]
        n_act = actor_state_dict["fc_mu.0.weight"].shape[0]
    elif agent == "fasttd3_simbav2":
        # HyperEmbedder appends one constant feature before its first linear layer.
        n_obs = actor_state_dict["embedder.w.w.weight"].shape[-1] - 1
        n_act = actor_state_dict["predictor.mean_bias"].shape[0]
    else:
        raise ValueError(f"Agent {agent} not supported")
    return agent, n_obs, n_act


def load_policy(checkpoint_path):
    torch_checkpoint = torch.load(
        f"{checkpoint_path}", map_location="cpu", weights_only=False
    )
    args = torch_checkpoint["args"]

    agent, n_obs, n_act = checkpoint_actor_dims(torch_checkpoint)

    policy = Policy(
        n_obs=n_obs,
        n_act=n_act,
        args=args,
        agent=agent,
    )
    policy.actor.load_state_dict(torch_checkpoint["actor_state_dict"])

    obs_normalizer_state = torch_checkpoint.get("obs_normalizer_state")
    if not obs_normalizer_state:
        policy.obs_normalizer = nn.Identity()
    else:
        policy.obs_normalizer.load_state_dict(obs_normalizer_state)

    return policy


def script_policy(policy: Policy) -> torch.jit.ScriptModule:
    export_policy = UnitreeTorchScriptPolicy(policy).to("cpu").eval()
    return torch.jit.script(export_policy)


def export_policy_as_jit(policy: Policy, path: str | Path, filename: str = "policy.pt") -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / filename
    script_policy(policy).save(str(output_path))
    return output_path


def export_policy_as_onnx(
    policy: Policy,
    path: str | Path,
    obs_dim: int,
    filename: str = "policy.onnx",
    opset_version: int = 18,
) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / filename
    export_policy = UnitreeTorchScriptPolicy(policy).to("cpu").eval()
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            export_policy,
            dummy_obs,
            str(output_path),
            export_params=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
            opset_version=opset_version,
        )
    return output_path
