# Evaluation Isolation Record

Initial implementation commit: `77c7efd Isolate Unitree FastTD3 evaluation`

This note records why Unitree FastTD3 evaluation is isolated from the training process, what was implemented, and what tradeoffs were accepted.

## Problem

The previous evaluation path reused the training `IsaacLabEnv`. That meant evaluation called `reset()` and `step()` on the same environment object that training was using.

For Unitree locomotion this is not just a logging concern:

- IsaacLab curriculum is updated during environment reset.
- Unitree command curriculum mutates the current `base_velocity` command ranges.
- Terrain curriculum mutates per-environment terrain levels and origins.
- Evaluation rollouts can therefore advance or disturb training curriculum state.
- Resetting after evaluation also breaks the ongoing off-policy training rollout state.

I considered creating a second eval env inside the same Python process, but rejected it. Isaac Sim steps the whole stage through a shared simulation context, so a same-process second env can still share physics stepping and stage-level state. Different scene names reduce path conflicts, but do not provide the process-level isolation needed here.

## Implemented Behavior

Evaluation now runs in a separate Python process through `fast_td3/eval_unitree.py`.

The training process does this on each eval interval:

1. Saves a temporary policy snapshot under `<run>/eval/`.
2. Adds `curriculum_snapshot` to that checkpoint.
3. Launches `fast_td3/eval_unitree.py` with `sys.executable`.
4. Parses the final JSON line returned by the eval process.

The eval process does this:

1. Starts its own Isaac Sim / IsaacLab environment.
2. Uses `eval_num_envs` environments, default `128`.
3. Uses a seed separated from training by `eval_seed_offset`.
4. Loads the FastTD3 actor and observation normalizer from the temporary snapshot.
5. Applies the training curriculum snapshot before evaluation reset.
6. Freezes its own curriculum manager while evaluating.
7. Returns `eval_avg_return` and `eval_avg_length` as JSON.

This keeps eval reset, eval step, and eval curriculum updates out of the training process.

## Files

- `fast_td3/train.py`: single-GPU trainer writes eval checkpoints and launches the eval subprocess.
- `fast_td3/train_multigpu.py`: multi-GPU trainer does the same per rank, then averages eval metrics across ranks.
- `fast_td3/eval_unitree.py`: standalone eval subprocess entry point.
- `fast_td3/environments/isaaclab_env.py`: exposes curriculum snapshot, apply, freeze, and close helpers.
- `fast_td3/fast_td3_utils.py`: `save_eval_snapshot()` writes the minimal actor/normalizer eval state, and `save_params()` keeps full training checkpoints.
- `fast_td3/hyperparams.py`: adds `eval_num_envs`, default `128`, and `eval_seed_offset`, default `1000003`.
- `docs/unitree_fasttd3.md`: points readers to this implementation note.

## Seeds

Single-GPU eval seed:

```text
seed + eval_seed_offset
```

Multi-GPU eval seed:

```text
seed + eval_seed_offset + rank
```

The default `eval_seed_offset` is `1000003`, a large prime. The goal is to avoid accidental training/eval seed overlap while preserving deterministic reproduction from the base seed.

## Curriculum State

Evaluation should not advance training curriculum, but it should measure the policy at the same current curriculum difficulty. To support that, the temporary eval checkpoint stores:

- current command ranges for `base_velocity`
- current terrain levels, when terrain curriculum is active

The eval process applies that snapshot to its own environment before reset, then freezes curriculum compute during evaluation. That means eval sees the same current difficulty but cannot mutate training state.

If training uses more environments than evaluation, the terrain-level snapshot is truncated to the eval environment count. This keeps eval lightweight while still sampling from the current training terrain curriculum state.

## Policy And Gradients

The eval process receives the current policy through the temporary eval snapshot. That snapshot contains:

- actor state dict
- observation normalizer state
- training args
- global step
- curriculum snapshot

It intentionally does not contain critic networks, target critic networks, optimizers, reward normalizer state, or replay buffer data. The eval process does not create a replay buffer.

The eval rollout uses `torch.inference_mode()` and `policy.eval()`, so it does not build an autograd graph and does not send gradients back to the training process. The evaluator only returns scalar metrics through JSON.

## Tradeoffs

The main tradeoff is cost. A separate process is heavier than a same-process eval call:

- each eval starts another Isaac Sim process
- GPU memory pressure is higher
- eval latency is higher

The benefit is stronger isolation:

- no training env reset from eval
- no training env step from eval
- no training curriculum mutation from eval
- no shared Isaac Sim stage or simulation context with training

If evaluation becomes too expensive, prefer increasing `eval_interval` or disabling eval temporarily instead of returning to same-process eval.

## Remaining Notes

The temporary eval checkpoints live under `<run>/eval/` and are intentionally not named `model_*.pt`. The play/export script only resolves `model_*.pt` from the run root, so eval snapshots should not be selected accidentally for deployment export.
