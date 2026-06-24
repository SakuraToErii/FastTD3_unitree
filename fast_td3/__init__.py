"""
Fast TD3 is a high-performance implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3)
with distributional critics for reinforcement learning.
"""

# Core model components
from fast_td3.fast_td3 import Actor, Critic, DistributionalQNetwork
from fast_td3.fast_td3_utils import EmpiricalNormalization, SimpleReplayBuffer
from fast_td3.unitree_policy import (
    Policy,
    UnitreeTorchScriptPolicy,
    checkpoint_actor_dims,
    export_policy_as_jit,
    export_policy_as_onnx,
    load_policy,
    script_policy,
)

__all__ = [
    # Core model components
    "Actor",
    "Critic",
    "DistributionalQNetwork",
    "EmpiricalNormalization",
    "SimpleReplayBuffer",
    "Policy",
    "UnitreeTorchScriptPolicy",
    "checkpoint_actor_dims",
    "export_policy_as_jit",
    "export_policy_as_onnx",
    "load_policy",
    "script_policy",
]
