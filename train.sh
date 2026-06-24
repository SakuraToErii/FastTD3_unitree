set -e
echo "=============================================="
echo "开始多次种子循环运行"
echo "=============================================="
# 使用不同种子循环运行5次
for seed in 1; do
  echo "=============================================="
  echo "开始运行第 $seed 次实验 (seed=$seed)"
  echo "=============================================="
#--num_envs=256 \
  python fast_td3/train.py \
    --env_name="G1JoystickFlatTerrain" \
    --exp_name="G1_flat_$seed" \
    --project="G1Moon" \
    --seed=$seed \
    --buffer_size=10240 \
    --batch_size=10240 \
    --num_envs=512 \
    --total_timesteps=100000 \
    --render_interval=5000 \
    --use_grad_norm_clipping \
    --max_grad_norm=10.0 \
    --learning_starts=50 \
    --obs_normalization \
    --reward_normalization \
    --use_tuned_reward \
    --use_domain_randomization 
    # --use_dynamic_reset

  echo "第 $seed 次实验完成"
  echo ""
done

echo "=============================================="
echo "所有实验完成！"
echo "==============================================" 
