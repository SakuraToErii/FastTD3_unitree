# AGENTS.md — FastTD3_unitree

> 给接手这个仓库的 AI / 开发者的第一印象指南。读这一份即可了解仓库定位、目录结构、运行方式、以及所有容易踩坑的契约。

## 一句话定位

这是 **FastTD3 算法**（Twin Delayed DDPG + 分布式 Critic）针对 **Unitree G1-29dof 人形 + `unitree_rl_lab` / Isaac Lab** 的专用兼容与桥接分支。原版 FastTD3 是一个通用 RL 基准实现，并没有在 Unitree 的 Isaac Lab 训练、仿真验证、真机部署链路里配置过；本仓库做的工作就是把它接进去，并保证产物能被 `g1_ctrl` 真机控制器直接使用。

- 远端：`git@github.com:SakuraToErii/FastTD3_unitree.git`，默认分支 `main`。
- 范围被刻意限定在 **Unitree G1 + unitree_rl_lab**，其它通用 benchmark / 部署栈已清理掉（见 `cfd28ad cleanup: make FastTD3 unitree-only`）。

## 关键背景知识（最容易踩坑的部分）

1. **虚拟环境必须先激活，不要直接调 `.venv/bin/python`。** 激活脚本会注入 Isaac Sim 路径，否则 `import isaacsim` 失败：
   ```bash
   source /home/ordis/projects/IsaacLab/.venv/bin/activate
   cd /home/ordis/projects/algs/FastTD3_unitree
   ```
2. **FastTD3 的 checkpoint 不是 RSL-RL checkpoint。** 训练产物虽然按 Unitree 风格命名成 `model_<step>.pt` 并放进 `logs/rsl_rl/<exp>/<run>/`，但里面的 key 是 FastTD3 的（`actor_state_dict`、`obs_normalizer_state`、`args`、`curriculum_snapshot` 等），**不能**交给 `unitree_rl_lab/scripts/rsl_rl/play.py`（它需要 `OnPolicyRunner` 的 `model_state_dict`/`optimizer_state_dict`/`iter`/`infos`）。播放/导出必须用本仓库的 `scripts/play_unitree_fasttd3.py`。
3. **真机部署契约 = `exported/policy.onnx` + 同一次训练导出的 `params/deploy.yaml`，缺一不可。** `g1_ctrl` 的 `policy_dir` 要指向 **run 目录**（同时含 `params/` 和 `exported/`），不要只拷贝 `policy.onnx`。`deploy.yaml` 记录了关节映射、控制周期、刚度阻尼、默认关节位置、动作缩放/偏置、观测项/缩放、历史长度等，与策略配套。
4. **评估在独立进程里跑**，不会动训练环境的 curriculum。设计原因见 `docs/eval_isolation.md`（commit `77c7efd`）。如果嫌评估太贵，调大 `eval_interval` 或暂时关掉评估，**不要**改回同进程评估。

## 外部依赖布局

仓库假设以下同级目录结构（路径写死在 `fast_td3/unitree_bridge.py` 的 `DEFAULT_UNITREE_RL_LAB_PATH`）：

```text
~/projects/
  IsaacLab/        # 含 _isaac_sim/ 与 .venv/（本仓库用的虚拟环境就在这里）
  unitree_rl_lab/  # Unitree 的任务定义、env_cfg、export_deploy_cfg 等
  algs/FastTD3_unitree/   # 本仓库
```

`--unitree_rl_lab_path` 可覆盖默认路径；运行时会把 `unitree_rl_lab/source/unitree_rl_lab` 加到 `sys.path` 最前面，并把 Unitree 的 `Unitree-G1-29dof-Velocity` 任务在 `gym` 里注册一个别名 `Isaac-Unitree-G1-29dof-Velocity`（FastTD3 内部用带 `Isaac-` 前缀的 env_name）。

## 目录结构

```text
FastTD3_unitree/
├── README.md                  # 面向用户的使用说明（中文）
├── AGENTS.md                  # 本文件
├── train.sh                   # 顶层 wrapper，转发到 scripts/train.sh
├── docs/
│   ├── unitree_fasttd3.md     # 主流程文档（训练/播放/导出/部署）
│   └── eval_isolation.md      # 评估进程隔离的设计记录
├── scripts/
│   ├── train.sh               # 批量训练入口（按 SEEDS 循环）
│   ├── train_unitree_fasttd3.py  # Unitree 风格训练 launcher（核心入口）
│   └── play_unitree_fasttd3.py   # checkpoint 播放 + 导出 ONNX/JIT（核心入口）
└── fast_td3/
    ├── __init__.py            # 导出 Actor/Critic/Policy/load_policy 等
    ├── hyperparams.py         # BaseArgs 训练超参（dataclass + CLI 解析）
    ├── fast_td3.py            # Actor / Critic / DistributionalQNetwork / SimNorm
    ├── fast_td3_simbav2.py    # 可选 SimbaV2 actor 变体（agent=fasttd3_simbav2）
    ├── fast_td3_utils.py      # ReplayBuffer / EmpiricalNormalization / save_*
    ├── train.py               # 单卡训练主循环（约 770 行）
    ├── train_multigpu.py      # 可选多卡 DDP 训练入口（约 830 行）
    ├── eval_unitree.py        # 独立进程评估入口（被 train.py 用 subprocess 拉起）
    ├── environments/
    │   └── isaaclab_env.py     # IsaacLabEnv 包装：reset/step/curriculum snapshot/freeze
    ├── unitree_bridge.py      # 路径解析 + 任务别名注册 + 默认路径常量
    └── unitree_policy.py      # Policy 加载 / TorchScript / ONNX 导出
```

## 运行方式

### 训练
```bash
source /home/ordis/projects/IsaacLab/.venv/bin/activate
cd /home/ordis/projects/algs/FastTD3_unitree
python scripts/train_unitree_fasttd3.py \
  --unitree_rl_lab_path /home/ordis/projects/unitree_rl_lab \
  --task Unitree-G1-29dof-Velocity \
  --run_name fasttd3_seed1 \
  --seed 1
```
`scripts/train.sh` 是批量入口，默认 `SEEDS=3407`、`num_envs=2048`、`buffer_size=1024`，噪声相关超参沿用 `hyperparams.py` 默认值（`policy_noise=0.1`、`noise_clip=0.2`、`std_max=0.3`、`std_max_end=0.1`），可通过环境变量 `SEEDS` / `UNITREE_RL_LAB_PATH` 覆盖。

launcher 做的事：解析 Unitree 特有参数 → 把 venv site-packages 提前到 `sys.path`（防止 Isaac Sim 的 `pip_prebundle` 阴影覆盖依赖，并会删掉被污染的 `typing_extensions` 模块）→ 注册任务别名 → 构造 Unitree 风格的 `<timestamp>_<run_name>` 日志目录 → 通过 `runpy.run_path` 以 `__main__` 跑 `fast_td3/train.py`，并默认补上 `--save_dir`、`--checkpoint_prefix model`、`--save_final_as_step`、`--export_unitree_params`。

### 输出目录布局
```text
unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/
  <timestamp>_fasttd3_seed1/
    model_<step>.pt            # FastTD3 checkpoint（不是 RSL-RL）
    params/
      agent.yaml               # 训练超参
      env.yaml                 # IsaacLab env_cfg
      deploy.yaml              # ★ g1_ctrl 真机部署契约
      velocity_env_cfg.py      # env_cfg 源文件副本
    eval/                      # 临时评估快照（不是 model_*.pt，不会被导出脚本选中）
    exported/                  # play 脚本产出
      policy.pt                # TorchScript
      policy.onnx              # ONNX
    events.out.tfevents.*      # TensorBoard
```
`params/deploy.yaml` 由 `unitree_rl_lab.utils.export_deploy_cfg.export_deploy_cfg` 从训练用的同一个 env 实例导出。

### 播放与导出
```bash
python scripts/play_unitree_fasttd3.py \
  --unitree_rl_lab_path /home/ordis/projects/unitree_rl_lab \
  --load_run <timestamp>_fasttd3_seed1 \
  --checkpoint model_50000.pt \
  --headless            # 或 --export_only 只导出不启仿真
```
省略 `--checkpoint` 自动取 run 目录里 step 最大的 `model_*.pt`；省略 `--load_run` 自动取最新 run。导出会做 TorchScript 数值校验（`_verify_scripted_policy`，atol=1e-6），并在缺 `params/deploy.yaml` 时告警。

### 真机部署
```yaml
# g1_ctrl 配置
FSM:
  Velocity:
    policy_dir: /home/ordis/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/<run>
```

## 算法与代码要点

- **FastTD3**：TD3 + 分布式 Critic（C51 风格，默认 `num_atoms=251`、`v_min=-10`、`v_max=10`）+ Clipped Double Q（`use_cdq=True`）。Actor 输出确定性动作并加 Gaussian 探索噪声（`std_min`/`std_max`，episode 级 sticky 重采样，`std_max` 随 `global_step` cosine 退火到 `std_max_end`）。探索噪声为静态绝对尺度 `randn × noise_scales`，不依赖 per-joint 物理参数。target policy smoothing 使用 `randn × policy_noise` clamp 到 `±noise_clip`，同样是静态绝对尺度。延迟策略更新 `policy_frequency=2`，每步 `num_updates=4` 次梯度。Actor 输出激活由 `use_tanh` 控制（默认 `False`，无界输出对齐 PPO；`True` 时附加 Tanh 并配合 `action_bounds` 缩放）。
- **SimNorm / SimbaV2**：可选。`sim_type ∈ {"", "sim_actor", "sim_critic", "sim_both"}`（Simplicial Normalization，arXiv 2204.00616）；`agent=fasttd3_simbav2` 用 `fast_td3_simbav2.py` 的 SimbaV2 actor（带可学习 Scaler、`num_blocks` 等）。
- **性能优化**：默认 `torch.compile`（`reduce-overhead`）、AMP `bf16`、`torch.set_float32_matmul_precision("high")`、`TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1`、`OMP_NUM_THREADS=1`。
- **IsaacLabEnv**（`fast_td3/environments/isaaclab_env.py`）：每个进程只起一个 Isaac Sim app（单例 `_SIMULATION_APP`，不允许换 device）；支持 asymmetric obs（critic obs）；`random_start_init=True` 在 reset 后随机化 `episode_length_buf` 以对齐 RSL-RL PPO 训练。提供 curriculum 的 `snapshot_curriculum` / `apply_curriculum_snapshot` / `frozen_curriculum` 三个助手，分别用于评估进程复刻当前难度、冻结 curriculum。
- **评估隔离**：`train.py` 每到 `eval_interval`（默认 1000）就 `save_eval_snapshot`（只含 actor + normalizer + args + global_step + curriculum_snapshot，**不含** critic/optimizer/buffer），用 `subprocess.run([sys.executable, "-m", "fast_td3.eval_unitree", ...])` 拉起独立进程，解析其最后一行 JSON（`eval_avg_return`、`eval_avg_length`）。eval 用 `eval_num_envs=128`、`seed + eval_seed_offset`（`1000003`），多卡再加 `rank`。eval 快照放在 `<run>/eval/` 且不叫 `model_*.pt`，避免被导出脚本误选。
- **策略导出**（`fast_td3/unitree_policy.py`）：`Policy = actor + EmpiricalNormalization`；`use_tanh` 从 checkpoint 的 `args` 读取，保证导出策略与训练时一致；导出时转成 `UnitreeTorchScriptPolicy`（用 `FrozenEmpiricalNormalizer` 让 normalizer 对 TorchScript 友好，并 `@torch.jit.export reset()` 提供空实现以兼容 RSL-RL 接口），再 `torch.jit.script` 存 `policy.pt`，并 `torch.onnx.export`（opset 默认 18，输入名 `obs`、输出名 `actions`）存 `policy.onnx`。`checkpoint_actor_dims` 从 state_dict 形状反推 obs/act 维度（注意 SimbaV2 的 embedder 会多一个常量特征，需减 1）。

## 关键超参默认值（`fast_td3/hyperparams.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `env_name` | `Isaac-Unitree-G1-29dof-Velocity` | 必须 `Isaac-` 前缀，否则 `train.py` 报错 |
| `num_envs` | 2048 | `scripts/train.sh` 对齐为 2048 |
| `total_timesteps` | 100000 | 注意默认训练步数较少，真训练需调大 |
| `buffer_size` | 1024 | 每环境 replay 大小 |
| `batch_size` | 32768 | |
| `num_atoms / v_min / v_max` | 251 / -10 / 10 | 分布式 critic support |
| `critic_hidden_dim / actor_hidden_dim` | 1024 / 512 | |
| `gamma / tau` | 0.99 / 0.1 | |
| `policy_noise / noise_clip` | 0.1 / 0.2 | target policy smoothing；noise_clip ≈ 2×policy_noise 使 clip 实际生效 |
| `std_min / std_max / std_max_end` | 0.001 / 0.3 / 0.1 | 探索噪声区间；`std_max_end=None` 时才退火到 `std_min` |
| `policy_frequency / num_updates` | 2 / 4 | |
| `eval_interval / eval_num_envs / eval_seed_offset` | 1000 / 128 / 1000003 | |
| `use_tanh` | False | Actor 末层是否使用 Tanh；False=无界(对齐 PPO)，True=有界[-1,1] |
| `action_bounds` | None | 仅 use_tanh=True 时生效，clamp[-1,1]×action_bounds；use_tanh=False 时必须为 None |
| `save_final_as_step / export_unitree_params` | False / False | launcher 默认补成 True |
| `random_start_init` | True | 对齐 RSL-RL PPO 的随机初始 episode 长度 |

CLI 解析支持 `--foo` 与 `--foo-bar` 两种写法，布尔参数用 `--flag`/`--no_flag`（或 `--flag true`）。

## 提示给接手 AI 的清单

- 改训练流程优先动 `fast_td3/train.py`（单卡）和 `fast_td3/train_multigpu.py`（多卡，两者结构高度相似，改一处记得同步另一处）。
- 改 Unitree 衔接（路径、任务注册、日志布局、deploy.yaml 导出）动 `scripts/*_unitree_fasttd3.py` 与 `fast_td3/unitree_bridge.py`。
- 改策略结构 / 导出格式动 `fast_td3/fast_td3.py`、`fast_td3/unitree_policy.py`；改了 actor 结构要同步更新 `checkpoint_actor_dims` 的维度推断。
- 改评估动 `fast_td3/eval_unitree.py` + `fast_td3/environments/isaaclab_env.py` 的 curriculum 助手；`train.py` 里 subprocess 调用约定是「最后一行 JSON」。
- 涉及 `params/deploy.yaml` 的任何字段变化都要同步检查 `g1_ctrl` 侧能否消费。
- 详细操作流程以 `docs/unitree_fasttd3.md` 为准。
