# Unitree FastTD3

这是一个面向 Unitree G1-29dof 和 `unitree_rl_lab` 的 FastTD3 专用版本。
需要事先已安装好 Isaacsim、Isaaclab 和 unitree_rl_lab，并且在 Isaaclab 虚拟环境里运行。
```text
~/
isaacsim/
projects/
  IsaacLab/
    _isaac_sim/
    .venv/
  unitree_rl_lab/
  algs/
    FastTD3/
```

项目目标不是保留通用强化学习任务集合，而是把 FastTD3 接到 Unitree 的 Isaac Lab 训练、仿真验证和真机部署链路里。当前训练产物会按 `unitree_rl_lab` 的 run 目录风格保存，并且会携带真机部署必须使用的 `params/deploy.yaml`。

详细流程见 [docs/unitree_fasttd3.md](docs/unitree_fasttd3.md)。

## 这个项目做什么

- 用 FastTD3 训练 `unitree_rl_lab` 里的 Unitree G1 velocity task。
- 保存 Unitree 风格的训练目录：`model_<step>.pt` 加 `params/`。
- 用 FastTD3 专用 play 脚本加载 checkpoint、仿真验证，并导出 `exported/policy.pt` 和 `exported/policy.onnx`。
- 为 `g1_ctrl` 保留同一策略包里的 `params/deploy.yaml`，避免只替换 ONNX 导致部署契约不匹配。

## 环境

不要直接调用 `.venv/bin/python`。先激活 Isaac Lab 虚拟环境，让 Isaac Sim 路径正确注入：

```bash
source /home/ordis/projects/IsaacLab/.venv/bin/activate
cd /home/ordis/projects/algs/FastTD3
```

## 训练

```bash
python scripts/train_unitree_fasttd3.py \
  --unitree_rl_lab_path /home/ordis/projects/unitree_rl_lab \
  --task Unitree-G1-29dof-Velocity \
  --run_name fasttd3_seed1 \
  --seed 1
```

训练输出默认写到：

```text
/home/ordis/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/
  <timestamp>_fasttd3_seed1/
    model_<step>.pt
    params/
      agent.yaml
      deploy.yaml
      env.yaml
      velocity_env_cfg.py
```

注意：`model_<step>.pt` 是 FastTD3 checkpoint，不是 RSL-RL checkpoint，不能直接交给 `unitree_rl_lab/scripts/rsl_rl/play.py`。

## 播放与导出

```bash
python scripts/play_unitree_fasttd3.py \
  --unitree_rl_lab_path /home/ordis/projects/unitree_rl_lab \
  --load_run <timestamp>_fasttd3_seed1 \
  --checkpoint model_50000.pt \
  --headless
```

只导出、不启动仿真：

```bash
python scripts/play_unitree_fasttd3.py \
  --unitree_rl_lab_path /home/ordis/projects/unitree_rl_lab \
  --load_run <timestamp>_fasttd3_seed1 \
  --checkpoint model_50000.pt \
  --export_only
```

导出后 run 目录会包含：

```text
<run>/
  params/
    deploy.yaml
  exported/
    policy.pt
    policy.onnx
```

## 真机部署要点

`g1_ctrl` 的 `policy_dir` 必须指向同时包含 `params/` 和 `exported/` 的 run 目录，不能指向 `exported/`：

```yaml
FSM:
  Velocity:
    policy_dir: /home/ordis/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/<run>
```

不要只交换 `policy.onnx`。`params/deploy.yaml` 记录了关节映射、控制周期、刚度阻尼、默认关节位置、动作缩放、观测项和历史长度等部署契约，必须和同一次训练导出的策略配套使用。

## 主要文件

```text
scripts/train_unitree_fasttd3.py      # Unitree 风格训练入口
scripts/play_unitree_fasttd3.py       # FastTD3 checkpoint 播放与导出
scripts/train.sh                      # 批量训练入口
fast_td3/train.py                     # 单卡 FastTD3 训练主循环
fast_td3/train_multigpu.py            # 可选多卡训练入口
fast_td3/eval_unitree.py              # 独立进程评估入口
fast_td3/hyperparams.py               # Unitree FastTD3 参数
fast_td3/environments/isaaclab_env.py # Isaac Lab 环境包装
fast_td3/unitree_bridge.py            # unitree_rl_lab 路径和任务注册桥接
fast_td3/unitree_policy.py            # checkpoint 加载与 policy 导出
```
