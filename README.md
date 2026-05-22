# livelybot_pi_rl_baseline

基于 NVIDIA Isaac Gym 的双足机器人强化学习基线,内置 PPO 训练流程,并提供从 Isaac Gym 到 MuJoCo 的 sim2sim 部署框架。本仓库支持的任务:

| 任务名 (`--task`) | 机器人 | 实验目录 (`logs/<...>`) | 说明 |
|---|---|---|---|
| `pai_ppo` | Pai 12-DoF | `Pai_ppo` | 默认行走任务 |
| `a1_ppo` | A1-legs_V1 12-DoF | `A1_ppo` | A1 双足行走 |
| `a1_jump_ppo` | A1-legs_V1 12-DoF | `A1_jump_ppo` | A1 跳跃 |

下面以 **A1 机器人** 为例,完整走一遍 安装 → 训练 → play 评估 → sim2sim 部署 流程。

---

## 1. 环境准备

### 1.1 创建训练用 conda 环境(Isaac Gym)

Isaac Gym 官方需要 Python 3.8 + numpy 1.23,所以训练和 play 用一个独立的 conda 环境。

```bash
# 创建并激活环境
conda create -n pi_env python=3.8 -y
conda activate pi_env

# 安装 PyTorch(CUDA 版本要 ≤ 你显卡驱动支持的版本,执行 nvidia-smi 查看)
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 锁定 numpy 版本
conda install numpy=1.23
```

### 1.2 安装 Isaac Gym Preview 4

1. 在 NVIDIA 官网下载 [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym)
2. 解压后安装:
   ```bash
   cd isaacgym/python && pip install -e .
   ```
3. 测试是否安装成功:
   ```bash
   cd ../examples && python 1080_balls_of_solitude.py
   ```
   如果有图形界面弹出且球体正常滚动,说明 Isaac Gym OK。如有问题,参考 `isaacgym/docs/index.html`。

### 1.3 安装本仓库

```bash
git clone https://github.com/XUEHAIXU/shaungzuzoulu.git
cd shaungzuzoulu
pip install -e .
```

`setup.py` 会一并安装 `wandb`、`tensorboard`、`tqdm`、`opencv-python`、`mujoco>=3.2,<3.3`、`mujoco-python-viewer`、`pyyaml`、`matplotlib`。

### 1.4 (可选) 创建 MuJoCo sim2sim 用的环境

[humanoid/scripts/sim2sim_a1.py](humanoid/scripts/sim2sim_a1.py) 用的是较新的 MuJoCo viewer API,推荐单独建一个环境跑 sim2sim:

```bash
conda create -n mujoco python=3.11 -y
conda activate mujoco
pip install "mujoco>=3.5" torch numpy scipy pyyaml
```

> **为什么要拆两个环境?** 训练侧的 Isaac Gym 锁死了 Python 3.8 + numpy 1.23,而新的 mujoco viewer 在 Python 3.11 / numpy 2.x 下更稳。仓库本身在两边都能 `pip install -e .`。

---

## 2. 训练 A1(`a1_ppo`)

A1 行走训练任务对应 [humanoid/envs/a1/a1_config.py](humanoid/envs/a1/a1_config.py),`experiment_name = "A1_ppo"`,默认 `run_name = "v1"`。

```bash
conda activate pi_env

# 训练:4096 个并行环境,无界面,run 名为 v1
python humanoid/scripts/train.py --task=a1_ppo --run_name v1 --headless --num_envs 4096
```

训练产物保存到:

```
logs/A1_ppo/<日期_时间>_v1/
├── model_0.pt
├── model_100.pt
├── ...
└── events.out.tfevents...      # TensorBoard 日志
```

实时观察训练曲线:

```bash
tensorboard --logdir logs/A1_ppo
```

训练 A1 跳跃任务:

```bash
python humanoid/scripts/train.py --task=a1_jump_ppo --run_name v1 --headless --num_envs 4096
```

### 常用命令行参数

| 参数 | 含义 |
|---|---|
| `--task` | 任务名,如 `a1_ppo` / `a1_jump_ppo` / `pai_ppo` |
| `--run_name` | 这次实验的名字,会拼到日志目录 |
| `--num_envs` | 并行环境数,显存够建议 4096 |
| `--headless` | 无界面运行(训练时强烈建议) |
| `--sim_device` | `cuda:0` / `cuda:1` / `cpu` |
| `--rl_device` | RL 计算设备,**必须与 `--sim_device` 一致** |
| `--max_iterations` | 最大迭代步数 |
| `--seed` | 随机种子 |
| `--resume` `--load_run` `--checkpoint` | 断点续训 |

完整列表见 [humanoid/utils/helpers.py](humanoid/utils/helpers.py)。

> **注意:** `CUDA_VISIBLE_DEVICES` 在 Isaac Gym 下不生效,必须用 `--sim_device` / `--rl_device` 显式指定。

---

## 3. play —— 评估并导出策略

`play.py` 会加载训练好的策略,在 Isaac Gym 中可视化运行,**同时自动导出 TorchScript (`.pt`) 与 ONNX (`.onnx`)**,供 sim2sim 与真机部署使用。

```bash
conda activate pi_env

# 评估 a1_ppo / run_name=v1 的最新模型
python humanoid/scripts/play.py --task=a1_ppo --run_name v1
```

运行后会:

1. 在 Isaac Gym 里实时渲染若干个 A1 机器人,观察策略效果(无 `--headless` 时弹窗,按 `v` 可关闭/开启渲染)
2. 把策略导出到:
   ```
   logs/A1_ppo/exported/policies/
   ├── policy_1.pt           # TorchScript,sim2sim_a1.py 默认加载
   └── policy_1.onnx         # ONNX,真机/C++ 部署
   ```

### play 常用控制

- 加载特定 checkpoint:`--load_run <子目录名> --checkpoint <iter>`
- 录视频:在 [humanoid/scripts/play.py](humanoid/scripts/play.py) 顶部把 `RENDER` 设为 `True`,会自动写到 `videos/A1_ppo/`

---

## 4. sim2sim —— 在 MuJoCo 中验证策略

[humanoid/scripts/sim2sim_a1.py](humanoid/scripts/sim2sim_a1.py) 使用 [humanoid/scripts/configs/a1.yaml](humanoid/scripts/configs/a1.yaml) 描述模型路径、PD 增益、obs 归一化等参数,**确保 MuJoCo 侧的观测构造、关节顺序、扭矩裁剪与 Isaac Gym 训练时完全一致**。

### 4.1 准备配置

打开 [humanoid/scripts/configs/a1.yaml](humanoid/scripts/configs/a1.yaml),修改 `policy_path` 指向你刚才 play 导出的 `.pt`:

```yaml
policy_path: "${LEGGED_GYM_ROOT_DIR}/logs/A1_ppo/exported/policies/policy_1.pt"
xml_path:    "${LEGGED_GYM_ROOT_DIR}/resources/robots/A1-legs_V1/mjcf/A1-legs_V1.xml"
```

`${LEGGED_GYM_ROOT_DIR}` 会自动展开为本仓库根目录。

### 4.2 运行 sim2sim

```bash
conda activate mujoco          # 见 1.4 节
cd humanoid/scripts
python sim2sim_a1.py configs/a1.yaml
```

MuJoCo viewer 弹出后,机器人会先用默认关节角站立 `warmup_steps=100` 步,再切换到策略控制。

### 4.3 键盘控制(在终端窗口生效,不是 viewer 窗口)

| 按键 | 作用 |
|---|---|
| `w` / `s` | vx +0.1 / -0.1 |
| `a` / `d` | yaw +0.1 / -0.1 (左转 / 右转) |
| `j` / `l` | vy +0.1 / -0.1 (左移 / 右移) |
| `space` | 速度指令清零 |
| `r` | 复位机器人状态 |

终端会实时显示当前 `vx / vy / yaw / base_z / step`。

---

## 5. 资源目录速览

```
shaungzuzoulu/
├── humanoid/
│   ├── algo/ppo/                 # PPO 实现 (on_policy_runner.py 等)
│   ├── envs/
│   │   ├── base/                 # legged_robot 基类
│   │   ├── pai/                  # Pai 任务
│   │   └── a1/                   # ★ A1 任务
│   │       ├── a1_config.py      # A1Cfg / A1CfgPPO
│   │       ├── a1_env.py         # A1FreeEnv
│   │       ├── a1_jump_config.py
│   │       └── a1_jump_env.py
│   ├── scripts/
│   │   ├── train.py              # 训练入口
│   │   ├── play.py               # 评估 + 导出 .pt/.onnx
│   │   ├── sim2sim.py            # Pai sim2sim
│   │   ├── sim2sim_a1.py         # ★ A1 sim2sim
│   │   └── configs/a1.yaml       # ★ A1 sim2sim 配置
│   └── utils/                    # 工具类(helpers, task_registry, logger)
├── resources/robots/
│   └── A1-legs_V1/               # ★ URDF / MJCF / mesh / textures
├── logs/                         # 训练输出(已 gitignore)
└── setup.py
```

---

## 6. 添加新机器人 / 新任务

1. 在 `humanoid/envs/<your_robot>/` 下新建 `<robot>_config.py` 和 `<robot>_env.py`,从 `LeggedRobotCfg` / `LeggedRobotCfgPPO` 继承
2. 把 URDF/MJCF/mesh 放到 `resources/robots/<your_robot>/`
3. 在 [humanoid/envs/__init__.py](humanoid/envs/__init__.py) 用 `task_registry.register(...)` 注册新任务
4. 想跑 sim2sim 的话,新建 `humanoid/scripts/configs/<your_robot>.yaml`,并参考 `sim2sim_a1.py` 写一个对应脚本(注意 MJCF 与 URDF 之间的关节顺序是否一致)

`cfg` 中所有非零的 `reward_scales` 都会在每一步累加;把某项设为 `0` 即可关闭对应奖励。

---

## 7. 常见问题

- **`ImportError: libpython3.8.so.1.0`** —— 训练环境必须激活 `pi_env`(Python 3.8)。
- **`numpy.bool` deprecated / numpy 版本错误** —— Isaac Gym 强依赖 numpy 1.23,`conda install numpy=1.23`。
- **`cudaErrorNoKernelImageForDevice`** —— PyTorch CUDA 版本与显卡驱动不匹配,重装匹配的 PyTorch。
- **MuJoCo viewer 黑屏** —— 检查显卡驱动 / EGL,或换 `mujoco-python-viewer`。
- **play 不导出 onnx** —— 检查 [humanoid/scripts/play.py](humanoid/scripts/play.py) 顶部 `EXPORT_POLICY = True`。

---

## Acknowledgments

本仓库基于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 与 HighTorque Robotics 的 [livelybot_pi_rl_baseline](https://github.com/HighTorque-Robotics/livelybot_pi_rl_baseline) 修改而来。
