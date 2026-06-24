#!/usr/bin/env bash
set -euo pipefail

UNITREE_RL_LAB_PATH="${UNITREE_RL_LAB_PATH:-/home/ordis/projects/unitree_rl_lab}"
SEEDS="${SEEDS:-1}"

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
