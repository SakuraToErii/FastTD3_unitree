# TODO — 对齐 unitree_rl_lab PPO 的环境配置与 Actor 输出

> 对比对象：`unitree_rl_lab/scripts/rsl_rl/train.py`（RSL-RL PPO）vs 本仓库 `scripts/train.sh` → `scripts/train_unitree_fasttd3.py` → `fast_td3/train.py`
>
> 范围：**排除算法自身的超参与更新逻辑**（如 gamma、lr、batch_size、num_updates、replay buffer、distributional critic 等），仅关注 **环境配置** 和 **Actor 策略输出到环境的链路**。

---

## 结论

本仓库 **已基本实现** 与 PPO 环境配置的一致性。以下方面已确认一致：

- ✅ 任务注册与 env_cfg 加载（同一个 `velocity_env_cfg.py`，同一个 `parse_env_cfg`）
- ✅ `num_envs` / `seed` / `device` 覆盖方式
- ✅ `random_start_init=True` ↔ PPO `init_at_random_ep_len=True`
- ✅ 观测组（policy + critic）、`history_length=5`、`flatten_history_dim=True`
- ✅ `time_outs` / `truncations` 处理（`is_finite_horizon=False`，两者都会在 extras 中添加 `time_outs`）
- ✅ `deploy.yaml` 导出（同一个 `export_deploy_cfg` 函数）
- ✅ `params/env.yaml` + `params/agent.yaml` + env_cfg 源文件拷贝
- ✅ 日志目录布局（`logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`）
- ✅ Play 环境使用 `play_env_cfg_entry_point`（`RobotPlayEnvCfg`）
- ✅ Action manager（`JointPositionAction`, `scale=0.25`, `use_default_offset=True`）
- ✅ `step_dt` = `sim.dt × decimation` = `0.005 × 4` = `0.02s`（50Hz）
- ✅ 策略导出格式（`actor(normalizer(obs))` → TorchScript / ONNX，opset 18，input `obs` / output `actions`）

以下是需要关注的差异，按优先级排列。

---

## 🔴 高优先级 — 影响训练行为

### 1. 动作裁剪（action clipping）不一致

| | PPO | FastTD3 |
|---|---|---|
| **配置** | `clip_actions=None`（`BasePPORunnerCfg` 未设置，默认 None） | `action_bounds=1.0`（`BaseArgs` 默认） |
| **行为** | `RslRlVecEnvWrapper.step` **不裁剪**，动作直接送入 `action_manager` | `IsaacLabEnv.step` 执行 `torch.clamp(actions, -1.0, 1.0) * action_bounds` |
| **影响** | PPO 的 Gaussian 采样可以产生超出 [-1,1] 的动作，直接进入 `JointPositionAction`（`action × 0.25 + default_offset`） | FastTD3 的探索噪声被限制在 [-1,1]，超出部分被截断 |

**分析**：
- FastTD3 的 Actor 末层使用 `nn.Tanh()`，确定性输出已在 [-1,1] 内，所以 **推理/部署阶段** 的裁剪是 no-op，不影响导出策略的行为。
- 差异仅存在于 **训练探索阶段**：PPO 允许探索噪声产生超出 [-1,1] 的动作，FastTD3 会截断。
- `play_unitree_fasttd3.py` 和 `eval_unitree.py` 也应用了相同的 `clamp(-1, 1) * action_bounds`，而 PPO 的 `play.py` 不裁剪。

**待办**：
- [x] 已完成：`action_bounds` 默认改为 `None`，去掉训练/play/eval 中的 clamp（关闭裁剪，完全对齐 PPO）
- [x] 已记录：`docs/action_clamp_decision.md`（Tanh actor + bounded exploration），而非环境配置差异
- [x] 已完成：play 和 eval 的 fallback 默认改为 `None`

### 2. ✅ 已解决 — torch 后端设置对齐

已在 `train.py` 和 `train_multigpu.py` 补齐与 PPO 一致的 torch 后端设置：
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
# torch_deterministic 默认改为 False（对齐 PPO 的 cudnn.deterministic=False）
```

---

## 🟡 中优先级 — 功能缺失（不影响默认训练行为）

### 3. 不支持 Hydra env_cfg 覆盖

| | PPO | FastTD3 |
|---|---|---|
| **机制** | `@hydra_task_config` 装饰器，允许通过 Hydra CLI 覆盖 env_cfg 任意参数（reward 权重、观测项、事件等） | `parse_env_cfg` 直接加载，无 Hydra 覆盖 |

**影响**：如果需要通过命令行调整 env_cfg 参数（如 `env.rewards.track_lin_vel_xy.weight=2.0`），PPO 支持而 FastTD3 不支持。不使用 Hydra 覆盖时，两者 env_cfg 完全一致。

**待办**：
- [ ] 评估是否需要为 FastTD3 launcher 添加 Hydra 支持，或提供等效的 env_cfg 覆盖机制
- [ ] 如果暂不需要，在文档中注明此限制

### 4. 训练阶段不支持视频录制

| | PPO | FastTD3 |
|---|---|---|
| **视频** | `--video` / `--video_length` / `--video_interval` | `AppLauncher(headless=True, device=device)`，不支持 |

**影响**：无法在训练过程中录制视频。`play_unitree_fasttd3.py` 已支持 `--video`，但训练阶段不支持。

**待办**：
- [ ] 如果需要训练阶段视频，将 `AppLauncher` 改为使用完整 CLI args（`AppLauncher.add_app_launcher_args`）
- [ ] 否则在文档中注明训练阶段不支持视频录制

### 5. AppLauncher 参数受限

| | PPO | FastTD3 |
|---|---|---|
| **AppLauncher** | `AppLauncher.add_app_launcher_args(parser)` → 完整 CLI（physics, livestream, enable_cameras 等） | `AppLauncher(headless=True, device=device)` → 仅 headless + device |

**影响**：无法通过 CLI 控制物理引擎类型、livestream、fabric 等参数。

**待办**：
- [ ] 如果需要更灵活的 AppLauncher 参数，改为使用 `AppLauncher.add_app_launcher_args`
- [ ] 否则保持现状（当前已硬编码 headless，对训练功能无影响）

---

## 🟢 低优先级 — 细微差异

### 6. Git 状态日志

- PPO：`runner.add_git_repo_to_log(__file__)` 记录 git 状态
- FastTD3：无 git 状态记录

**待办**：
- [x] 已完成：`log_git_state()` 函数记录 FastTD3 和 unitree_rl_lab 的 commit/branch/status 到 `params/git_state.txt`

### 7. Neptune logger 支持

- PPO：支持 tensorboard / wandb / neptune
- FastTD3：支持 tensorboard / wandb（通过 `use_wandb` + `log_tensorboard`）

**待办**：
- [x] 已完成：添加 `use_neptune` / `neptune_project` 参数，`train.py` 和 `train_multigpu.py` 均支持

### 8. 分布式训练入口

- PPO：`--distributed` 标志在同一脚本中支持多卡
- FastTD3：独立的 `fast_td3/train_multigpu.py` 脚本

**影响**：入口不同但功能等效。需确保 `train_multigpu.py` 与 `train.py` 保持同步（AGENTS.md 已提及此契约）。

**待办**：
- [ ] 确认 `train_multigpu.py` 中上述 #1~#2 的修改也需同步

---

## 📋 不需要修改的项（已一致或属于算法自身差异）

以下差异 **无需修改**，因为它们要么已一致，要么属于算法自身的预期差异：

- **Actor 输出激活函数**：PPO 的 MLP 无输出激活（unbounded Gaussian mean），FastTD3 使用 `Tanh`（bounded）。这是算法网络架构差异，属于预期。
- **探索方式**：PPO 从 `Normal(mean, std)` 采样，FastTD3 对确定性输出加 clipped Gaussian 噪声。这是算法探索策略差异。
- **Checkpoint 格式**：PPO 用 `model_state_dict` / `optimizer_state_dict` / `iter`，FastTD3 用 `actor_state_dict` / `obs_normalizer_state` / `args` 等。这是算法实现差异，AGENTS.md 已记录。
- **观测归一化**：两者默认均关闭（PPO `empirical_normalization=False`，FastTD3 `obs_normalization=False`）。
- **推理/部署阶段动作裁剪**：FastTD3 的 Tanh 输出已在 [-1,1] 内，`clamp(-1,1) * 1.0` 是 no-op，与 PPO 行为一致。
- **OMP_NUM_THREADS / TORCHDYNAMO_INLINE_INAUILT_NN_MODULES**：FastTD3 特有的性能优化，不影响训练正确性。
- **torch.compile / AMP**：算法性能优化，不影响环境配置。
