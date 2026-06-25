# 噪声标定改进记录

> 本文件记录对 FastTD3 探索噪声与策略噪声机制所做的实现改动（按"实现了什么"而非"修复了什么"表述）。

## 背景

FastTD3 有两类噪声：
- **探索噪声**（exploration）：采样时给 actor 输出叠加的高斯噪声，episode 级 sticky 重采样。
- **策略噪声**（target policy smoothing）：critic 更新时给 target actor next-action 叠加的 clipped 高斯噪声。

改动前两者均为"固定绝对幅度 + 单标量广播到全部 29 个动作维"，且不随训练步数衰减。本次工作将其升级为按关节物理量纲缩放、并随训练退火。

## 实现内容

### 1. 实现了 per-dimension 噪声缩放

- **`fast_td3/environments/isaaclab_env.py`**：新增 `IsaacLabEnv.action_std_scales` 属性与 `_compute_action_std_scales()` 方法。从 live articulation 的 `root_physx_view` 读取每个关节的 `effort_limit`（`get_dof_max_forces`）与 `stiffness`（`get_dof_stiffness`），按 `0.25 × effort / stiffness`（即"使 actuator 力矩饱和的 action 幅度"，与 Unitree mimic action scale 同口径）计算 per-dim 单位，再归一化到 `[0, 1]`（最大行程关节保持配置的 `std_max`）。读取失败时优雅回退到全 1 向量，保持旧行为。
- **`fast_td3/fast_td3.py` / `fast_td3/fast_td3_simbav2.py`**：两个 `Actor` 类均新增 `action_std_scales` 参数并注册为 buffer。`explore()` 中噪声由 `randn × noise_scales` 改为 `randn × noise_scales × action_std_scales`，使小行程关节（如腕部）噪声小于大行程关节（如髋部）。
- **`fast_td3/train.py` / `fast_td3/train_multigpu.py`**：`actor_kwargs` 注入 `envs.action_std_scales`。

`action_std_scales` 注册为 **非持久化 buffer**（`persistent=False`）：不进入 `actor_state_dict`，因此加载旧 checkpoint（无此 key）或新 checkpoint 都不会触发 strict load 报错；play/export 仅跑确定性 `forward`，本就不需要噪声缩放。它仍作为模块属性可被训练循环读取（监控指标、target noise 均可用）。`checkpoint_actor_dims` 仅依据 `fc_mu` 权重形状推断维度，不受影响。

### 2. 实现了探索噪声 `std_max` 的 cosine 退火

- **`fast_td3/hyperparams.py`**：新增超参 `std_max_end: float = None`（`None` 时退火到 `std_min`）。
- **`fast_td3/train.py` / `fast_td3/train_multigpu.py`**：新增 `current_std_max(step)` 与 `apply_exploration_schedule(step)`。在 `learning_starts` 之前保持满 `std_max`（warm-up 平台，保证 buffer 初期高覆盖随机探索），之后按 cosine 从 `std_max` 退火到 `std_max_end`，风格与已有 LR `CosineAnnealingLR` 一致。每步在 `policy()` 调用前更新 `actor.std_max` / `actor_detach.std_max`；resume 时按 `global_step` 重算一次。`explore()` 中 done 重采样读取 `self.std_max`，故重采样区间自动跟随退火。

### 3. 实现了 target policy smoothing 的 per-dim clip

- **`fast_td3/train.py` / `fast_td3/train_multigpu.py`**：target policy noise 由全局 `clamp(-noise_clip, noise_clip)` 改为 `clamp × action_std_scales`，使平滑邻域跟随各关节控制权限。`use_tanh=True` 分支保留原 `clamp(-1,1)`；`use_tanh=False` 下不再无约束放行，而是依靠 per-dim clip 约束平滑范围。

### 4. 统一了 `policy_noise` / `noise_clip` 取值与文档

- **`fast_td3/hyperparams.py`**：`policy_noise` 默认 `0.2 → 0.1`，`noise_clip` 默认 `0.5 → 0.2`（≈2× `policy_noise`，使 clip 在有意义比例的样本上触发，不再形同虚设）。
- **`scripts/train.sh`**：移除冗余的 `--policy_noise 0.1 --std_max 0.3` 覆盖，噪声超参统一沿用 `hyperparams.py` 默认值，消除"默认值/脚本值/文档值"三方漂移。
- **`AGENTS.md`**：关键超参表与算法说明同步更新为 per-dim 噪声 + 退火描述。

### 5. 实现了噪声标定的可观测指标

- **`fast_td3/train.py` / `fast_td3/train_multigpu.py`** 的 `collect_train_scalars` 新增四项 TensorBoard/wandb 指标：
  - `Train/expl_std_mean`：当前探索噪声 std 均值。
  - `Train/std_max_cur`：当前退火后的 `std_max`。
  - `Train/action_abs_mean`：actor 输出绝对值均值。
  - `Train/noise_action_ratio`：噪声幅度 / 动作幅度，用于判断探索是否被动作量级淹没。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `fast_td3/hyperparams.py` | `policy_noise`/`noise_clip` 默认值；新增 `std_max_end` |
| `fast_td3/fast_td3.py` | `Actor` 新增 `action_std_scales`；`explore` per-dim 噪声 |
| `fast_td3/fast_td3_simbav2.py` | SimbaV2 `Actor` 同步 per-dim 噪声 |
| `fast_td3/environments/isaaclab_env.py` | 新增 `action_std_scales` 推导 |
| `fast_td3/train.py` | kwargs 注入、退火、per-dim target clip、监控指标 |
| `fast_td3/train_multigpu.py` | 与单卡同步 |
| `scripts/train.sh` | 移除冗余噪声覆盖 |
| `AGENTS.md` | 超参表与说明同步 |

## 验证

- 全部改动文件 `py_compile` 通过。
- 单元级 smoke test：构造 `Actor(action_std_scales=...)`，验证 `explore` 输出形状、`deterministic=True` 等价于 `forward`、`std_max` cosine 退火曲线（`learning_starts` 前平台、中点约半值、终点降至 `std_min`）、per-dim 广播形状正确。
- IsaacLab 运行时 `action_std_scales` 的实际数值需在真实训练启动后由 `Train/noise_action_ratio` 等指标复核。
