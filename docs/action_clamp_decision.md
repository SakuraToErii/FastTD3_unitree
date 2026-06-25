# Action Output Design: use_tanh + action_bounds

> Date: 2026-06-25
> Context: FastTD3_unitree alignment with unitree_rl_lab PPO

## Design

A single `use_tanh` hyperparameter controls the Actor's output activation and the entire action processing chain:

### `use_tanh=False` (default, matches PPO)

```
Actor:     MLP(obs) → unbounded
IsaacLabEnv.step:   no clamp, no scaling → raw action to JointPositionAction
Target smoothing:   no clamp
action_bounds:      must be None (error otherwise)
```

The Actor output is unbounded, identical to PPO's `MLP(obs)` mean output. The environment's `JointPositionAction` (`action × 0.25 + default`) and the physics engine's joint limits serve as the only safety boundaries.

### `use_tanh=True` (classic TD3, opt-in)

```
Actor:     Tanh(MLP(obs)) → [-1, 1]
IsaacLabEnv.step:   clamp(-1, 1) × action_bounds → [-action_bounds, action_bounds]
Target smoothing:   clamp(-1, 1)
action_bounds:      must be set (error if None)
```

The Actor output is bounded by Tanh. `action_bounds` scales the Tanh output to a wider range. The clamp in `IsaacLabEnv.step` is a safety net for exploration noise exceeding Tanh's range.

### Validation

| `use_tanh` | `action_bounds` | Result |
|---|---|---|
| `True` | set (e.g., 1.0, 4.0) | ✅ Valid |
| `True` | `None` | ❌ Error: requires action_bounds |
| `False` | `None` | ✅ Valid (default) |
| `False` | set | ❌ Error: not supported without Tanh |

### Why no separate `action_low` / `action_high`

When `use_tanh=True`, the target policy smoothing clamp range is always [-1, 1] (Tanh's natural range), regardless of `action_bounds`. The `action_bounds` scaling happens only in `IsaacLabEnv.step` (environment-facing), not in the critic's target computation (which operates in pre-scaling action space). So there's no need for separate variables.

## Files Changed

| File | Change |
|---|---|
| `fast_td3/hyperparams.py` | Added `use_tanh: bool = False`; `action_bounds` default `None`; updated docstrings |
| `fast_td3/fast_td3.py` | Actor takes `use_tanh`; conditionally appends `nn.Tanh()` to `fc_mu` |
| `fast_td3/fast_td3_simbav2.py` | `HyperPolicy` takes `use_tanh`; conditionally applies `torch.tanh()` |
| `fast_td3/train.py` | Validation; removed `action_low/high`; target smoothing uses `use_tanh`; passes `use_tanh` to Actor |
| `fast_td3/train_multigpu.py` | Same as `train.py` |
| `fast_td3/unitree_policy.py` | Passes `use_tanh` from checkpoint args to Actor/HyperPolicy |
| `scripts/play_unitree_fasttd3.py` | `action_bounds` fallback default `None` |
| `fast_td3/eval_unitree.py` | `action_bounds` default `None`; accepts `"None"` string |

## Backward Compatibility

- `nn.Tanh` has no learnable parameters → state_dict keys unchanged → old checkpoints load without errors
- Old checkpoints (trained with Tanh) will produce different outputs when loaded without Tanh → **must retrain**
- Old checkpoints' `args` dict won't have `use_tanh` → defaults to `False` (no Tanh)
