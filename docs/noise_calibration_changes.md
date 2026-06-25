# 噪声机制说明

## 两类噪声

FastTD3 有两类噪声，均为静态绝对尺度，不依赖关节物理参数或动作幅度：

- **探索噪声**（exploration）：采样时给 actor 输出叠加的高斯噪声，episode 级 sticky 重采样，std 从 `U[std_min, std_max]` 采样。
- **策略噪声**（target policy smoothing）：critic 更新时给 target actor next-action 叠加的 clipped 高斯噪声。

## 探索噪声

```python
noise = torch.randn_like(act) * self.noise_scales
return act + noise
```

- `noise_scales`：`(num_envs, 1)`，每个环境在 episode 开始时从 `U[std_min, std_max]` 采样一个值，整个 episode 保持不变（sticky）。episode 结束（done）时重采样。
- `std_max` 随 `global_step` cosine 退火到 `std_max_end`（`learning_starts` 之前保持满 `std_max`）。
- 超参：`std_min`（默认 0.001）、`std_max`（默认 0.3）、`std_max_end`（默认 0.1，设为 `None` 时退火到 `std_min`）。

## 策略噪声（target policy smoothing）

```python
clipped_noise = (torch.randn_like(actions) * policy_noise).clamp(-noise_clip, noise_clip)
```

- 标准 TD3 target policy smoothing：在 critic 更新时给 target actor 的 next-action 叠加 clipped 高斯噪声，防止 critic 利用 Q 函数的尖锐峰值。
- 超参：`policy_noise`（默认 0.1）、`noise_clip`（默认 0.2，≈2× `policy_noise`）。

## 监控指标

TensorBoard / wandb 中以下指标用于判断探索噪声是否合理：

- `Train/expl_std_mean`：当前探索噪声 std 均值。
- `Train/std_max_cur`：当前退火后的 `std_max`。
- `Train/action_abs_mean`：actor 输出绝对值均值。
- `Train/noise_action_ratio`：噪声 std / 动作幅度，用于判断探索是否被动作量级淹没。
