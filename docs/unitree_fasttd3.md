# Unitree FastTD3 Workflow

This project uses a FastTD3 checkpoint format, not an RSL-RL checkpoint format.
Do not feed FastTD3 `model_*.pt` files to `unitree_rl_lab/scripts/rsl_rl/play.py`: that script builds an `OnPolicyRunner` and expects checkpoint keys such as `model_state_dict`, `optimizer_state_dict`, `iter`, and `infos`.

Use the FastTD3-specific scripts below instead.

## Environment

Activate Isaac Lab through its virtual environment activation script, not by calling `.venv/bin/python` directly. The activation script sets the Isaac Sim paths required for `import isaacsim`.

```bash
source /home/ordis/projects/IsaacLab/.venv/bin/activate
cd /home/ordis/projects/algs/FastTD3
```

## Training

Use:

```bash
python scripts/train_unitree_fasttd3.py \
  --unitree_rl_lab_path /home/ordis/projects/unitree_rl_lab \
  --task Unitree-G1-29dof-Velocity \
  --run_name fasttd3_seed1 \
  --seed 1
```

`scripts/train_unitree_fasttd3.py` wraps `fast_td3/train.py` and changes the output layout to match the Unitree log/run style:

```text
/home/ordis/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/
  2026-xx-xx_xx-xx-xx_fasttd3_seed1/
    model_<step>.pt
    params/
      agent.yaml
      deploy.yaml
      env.yaml
      velocity_env_cfg.py
```

The `model_<step>.pt` file is still a FastTD3 checkpoint. It is named and placed like a Unitree/RSL-RL run artifact, but it is not loadable by RSL-RL `OnPolicyRunner`.

The `params/deploy.yaml` file is exported from the same Unitree Isaac Lab environment instance used by training, so it carries the deployment contract needed later by `g1_ctrl`: joint mapping, step timing, stiffness, damping, default joint positions, action scale/offset, observation terms, observation scales, and history settings.

By default, the Unitree launcher passes these FastTD3 options:

```text
--save_dir <unitree_rl_lab>/logs/rsl_rl/unitree_g1_29dof_velocity/<run>
--checkpoint_prefix model
--save_final_as_step
--export_unitree_params
```

So the final checkpoint is saved as `model_<global_step>.pt` rather than the old FastTD3 `models/<run>_final.pt` layout.

### Action Output Mode (`use_tanh`)

By default, `use_tanh=False`: the Actor outputs an unbounded MLP result (matching PPO's `act_inference`), and `action_bounds` must be `None`. No clipping is applied — the environment's `JointPositionAction` (`action * 0.25 + default`) and the physics engine's joint limits are the only safety boundaries.

To use classic TD3 bounded output, pass `--use_tanh --action_bounds 1.0`. The Actor appends `nn.Tanh()` (output [-1, 1]) and `IsaacLabEnv.step` scales it to `[-action_bounds, action_bounds]`. Target policy smoothing clamps to [-1, 1].

Invalid combinations raise `ValueError` at startup. See `docs/action_clamp_decision.md` for the full design rationale.

Rationale: with Tanh + `scale=0.25`, the maximum joint deviation was only ±0.25 rad (14.3°), far below the G1's actual joint ranges. PPO has no Tanh and no clip, so it can reach the full joint range.

Evaluation runs in a separate Isaac Sim process through `fast_td3/eval_unitree.py`. The trainer writes a temporary policy snapshot under `<run>/eval/`, including the current Unitree curriculum state, and the eval process uses `eval_num_envs=128` and `seed + eval_seed_offset` (`1000003` by default). This keeps eval reset/step/curriculum updates from touching the training environment while evaluating at the same current curriculum difficulty. The implementation record is in `docs/eval_isolation.md` and corresponds to commit `77c7efd`.

## Playing And Exporting

Use:

```bash
python scripts/play_unitree_fasttd3.py \
  --unitree_rl_lab_path /home/ordis/projects/unitree_rl_lab \
  --load_run 2026-xx-xx_xx-xx-xx_fasttd3_seed1 \
  --checkpoint model_50000.pt \
  --headless
```

`scripts/play_unitree_fasttd3.py` does three things:

1. Resolves a FastTD3 checkpoint from a Unitree-style run directory.
2. Exports deployable policy files beside that checkpoint:

```text
<run>/
  exported/
    policy.pt
    policy.onnx
```

3. Creates the Unitree Isaac Lab play environment and runs the FastTD3 policy in simulation.

If `--checkpoint` is omitted, the script uses the latest `model_*.pt` in the selected run. If `--load_run` is also omitted, it uses the latest run directory under:

```text
/home/ordis/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity
```

To export only and skip simulation:

```bash
python scripts/play_unitree_fasttd3.py \
  --unitree_rl_lab_path /home/ordis/projects/unitree_rl_lab \
  --load_run 2026-xx-xx_xx-xx-xx_fasttd3_seed1 \
  --checkpoint model_50000.pt \
  --export_only
```

This replaces the old `scripts/export_unitree_fasttd3.py` path. That script was removed because export now belongs to the FastTD3 play flow.

The policy loading/export implementation lives in:

```text
fast_td3/unitree_policy.py
```

This file replaced the old generic `fast_td3/fast_td3_deploy.py` name. It is still needed by `scripts/play_unitree_fasttd3.py`.

## Real Robot Deployment Package

For `g1_ctrl`, point `policy_dir` at the run directory, not at `exported/`.

Required layout:

```text
<run>/
  params/
    deploy.yaml
  exported/
    policy.onnx
```

Example:

```yaml
FSM:
  Velocity:
    policy_dir: /home/ordis/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-xx-xx_xx-xx-xx_fasttd3_seed1
```

Do not copy only `policy.onnx` into another model directory unless you also bring the matching `params/deploy.yaml`. The ONNX and `deploy.yaml` must come from the same training/deployment contract.

## Unitree-Only Scope

This fork is intentionally limited to Unitree G1 and `unitree_rl_lab`.

Active files:

```text
scripts/train_unitree_fasttd3.py
scripts/play_unitree_fasttd3.py
scripts/train.sh
fast_td3/train.py
fast_td3/eval_unitree.py
fast_td3/hyperparams.py
fast_td3/fast_td3.py
fast_td3/fast_td3_utils.py
fast_td3/environments/isaaclab_env.py
fast_td3/unitree_bridge.py
fast_td3/unitree_policy.py
```

Optional helpers if you still use them:

```text
fast_td3/fast_td3_simbav2.py      # optional Simplicial Embeddings agent variant
fast_td3/train_multigpu.py        # optional multi-GPU launcher
```

Other benchmark and deployment stacks are outside this fork.
