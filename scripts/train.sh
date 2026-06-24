#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

UNITREE_RL_LAB_PATH="${UNITREE_RL_LAB_PATH:-/home/ordis/projects/unitree_rl_lab}"
SEEDS="${SEEDS:-3407}"

for seed in $SEEDS; do
  exp_name="UnitreeFastTD3_seed${seed}"

  echo "Training seed=${seed}"
  python scripts/train_unitree_fasttd3.py \
    --unitree_rl_lab_path "$UNITREE_RL_LAB_PATH" \
    --task Unitree-G1-29dof-Velocity \
    --exp_name "$exp_name" \
    --project UnitreeFastTD3 \
    --run_name "fasttd3_seed${seed}" \
    --seed "$seed"
done
