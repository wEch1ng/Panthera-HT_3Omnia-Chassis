# Panthera Lab · 三全向轮单臂实验平台

同一台 Panthera 轮式单臂，逐步尝试视觉控制、遥操作、模仿学习、强化学习和端到端策略。本仓库把**机器人本体、传感器、控制、任务、训练与显示**分开组织，当前可运行的任务是“腕部 D405 观察乒乓球、机械臂调节球拍保持球心居中”。

**默认演示**：原单机械臂第 1500 轮策略控制六个关节，三个全向轮由键盘控制。原生 MuJoCo 全局窗口、右下角相机图像、侧栏外力滑块和丢球自动重开均保留。另附第 40 轮九动作残差 PPO，属于实验基线，并未证明优于原控制器。

## 在另一台电脑开始

优先使用 **Linux x86_64 + Python 3.12 + 图形驱动**。已测试 CPU 推理；有独立显卡更适合高精度网格的实时显示。不需要 ROS、不需要访问原来的单臂仓库，也不需要先重新训练或重新转换网格。

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在终端执行：

```bash
git clone https://github.com/wEch1ng/Panthera-HT_3Omnia-Chassis.git
cd Panthera-HT_3Omnia-Chassis
uv sync --frozen
uv run --frozen panthera doctor
uv run --frozen panthera view
```

依赖版本由 `pyproject.toml` 和 `uv.lock` 锁定；默认安装 CPU 版 PyTorch。第一次安装需要网络，正常运行不需要下载权重。URDF、STL/GLB/OBJ、相机和球拍资源、两个策略权重都随工程提供；文件路径由安装位置解析，和用户名、当前工作目录无关。

Ubuntu 缺少图形库时：

```bash
sudo apt-get update
sudo apt-get install libgl1 libegl1 libglfw3 libglib2.0-0
```

`doctor` 的默认检查不打开窗口；要同时测试 RGB-D 渲染和实际推理：

```bash
uv run --frozen panthera doctor --render
```

无桌面 Linux 机器可运行：

```bash
MUJOCO_GL=egl uv run --frozen panthera doctor --render
MUJOCO_GL=egl uv run --frozen panthera eval --episodes 2 --seconds 20
```

EGL 不可用时需安装相应驱动/软件 OpenGL；不要把有桌面的 `view` 当成无窗口命令。Windows、macOS 的图形兼容性尚未在本工程验收，尤其 macOS 的原生 MuJoCo 窗口有独立的主线程要求。这里提供的“可复现”首先指已验证的 Linux 环境，不承诺未经测试的平台。

## 快速命令

|用途|命令|
|---|---|
|默认单臂策略 + 键盘底盘|`uv run panthera view`|
|实验性轮臂残差 PPO|`uv run panthera view --task pingpong-residual`|
|指定新权重|`uv run panthera view --task pingpong-residual --checkpoint outputs/runs/my-run/last.pt`|
|独立评测|`uv run panthera eval --task pingpong-arm --episodes 6 --seconds 20`|
|训练新残差 PPO|`uv run panthera train --iterations 40 --workers 4 --output outputs/runs/my-run`|
|继续同一训练|`uv run panthera train --iterations 40 --workers 4 --output outputs/runs/my-run --resume`|
|录制 RGB-D 与电机反馈|`uv run panthera record --seconds 5 --output outputs/recordings/demo.npz`|
|短时性能分析|`uv run panthera view --profile-seconds 8 --profile-output outputs/profile.json`|

无窗口命令前可添加 `MUJOCO_GL=egl`。训练输出放到 `outputs/`，不会覆盖 `checkpoints/` 中的演示权重。`--resume` 包含网络与优化器状态，但暂不保存并行环境和所有随机数状态，不保证中断恢复后逐位一致。

## 演示操作

先点击仿真窗口，使用英文输入法；方向均相对底盘自身。

|按键/操作|作用|
|---|---|
|W / S|前进 / 后退|
|A / D|左移 / 右移|
|Q / E|逆时针 / 顺时针自转|
|松键 / X|平滑停车 / 目标速度归零|
|Space / R|暂停继续 / 重开回合|
|方向键|短时推球扰动|
|C / 侧栏 Clear all|清除外力|
|双击物体，再 Ctrl + 右键拖拽|MuJoCo 原生平移施力|
|双击物体，再 Ctrl + 左键拖拽|MuJoCo 原生力矩扰动|
|M|仅残差 PPO 模式：切换键盘/PPO底盘|

右侧展开 **Control**：`Ball_Fx/Fy/Fz_N`、`Paddle_Fx/Fy/Fz_N`、`Base_Fx/Fy/Fz_N` 分别是世界坐标下的球、球拍和底盘外力，单位 N。保持非零会持续施力；设为零、按 C 或重开回合清除。它们不是实际电机，也不进入策略输入。

D405 连续丢球 0.4 秒，或视觉测到球持续远离恢复区域，会重开回合并清空跟踪、动作历史和延迟队列。重开是仿真器恢复初始状态，不表示机器人会自行捡球。释放移动键后才能继续驾驶。

## 文件结构

```text
assets/
  robots/panthera/
    urdf/robot.urdf       # 完整轮臂 + URDF相机；另有独立arm/chassis
    meshes/              # 原始机械臂、底盘CAD网格
    mjcf/                # 已转换MuJoCo组件，不需首次转换
  tools/paddle/          # 球拍、球、许可证
checkpoints/
  pingpong/              # 原单臂与实验残差策略
  manifest.json          # 来源、用途、SHA256
configs/                 # 基线参数、可读默认配置说明
src/panthera_lab/
  robot/                 # 关节命名、布局、电机参数
  control/               # 三全向轮正逆解、视觉基线
  perception/            # RGB-D采集、颜色分割、球面拟合、跟踪
  sim/                   # 通用本体组装、动力学/电机、外力
  tasks/pingpong/         # 球拍场景、回合、单臂适配、残差任务
  learning/              # PPO网络、更新与训练入口
  runtime/               # 回合重开
  apps/                  # 原生查看器、评测、自检、录制
  interfaces.py          # SensorPacket / RobotCommand / SensorPolicy
  resources.py           # 源码和安装包通用的资源路径
examples/                # 接入图像策略、注册新任务的可运行示例
scripts/                 # 资产验证、打包/迁移工具
reports/                 # 重构验证与标注为历史的旧结果
outputs/                 # 本机新训练/评测/录制，不提交Git
archive/                 # 原始导出与旧实验完整归档，不参与默认运行
```

找模型看 `assets/robots/panthera/urdf/robot.urdf`；改轮子解算看 `control/omni_drive.py`；改识别看 `perception/`；新增项目从 `tasks/` 开始。不要把训练脚本重新放进 URDF 文件夹。

## 如何扩展

- **相同本体的新任务**：复用 `sim.model.robot_xml()` 与 `RobotPlant`，在任务自己的场景函数加入物体、工具和奖励；平衡球不是通用本体的依赖。
- **新策略**：接受 `SensorPacket`，输出六关节目标位置和三轮目标速度；可直接使用原始 RGB-D，无须先做球检测。`examples/image_policy.py` 演示接口，但只是持位策略，不是已训练的端到端网络。
- **不同回合逻辑/查看器任务**：通过任务工厂注册，或用 `--task module:function` 加载自己的 `TaskBundle`，不必修改原生查看器。详见 [架构与新增任务](docs/architecture.md)。
- **模仿学习数据**：`record` 输出 RGB、深度、电机位置/速度/反馈力矩、时间和对应低层指令。无球真值、无底盘真值；只是采集基础，尚无完整数据集管理/训练算法。
- **实机**：保留相同传感器/指令边界，替换相机、CAN驱动和时钟适配层。本工程目前没有实机驱动，见 [电机与迁移限制](docs/motors.md)。

## 当前行为与边界

识别复用原单臂 RGB-D 方法：橙色分割 → 深度反投影 → 20 mm 球面拟合 → 连续帧速度估计。单臂策略使用 42 维视觉/关节/历史特征，输出 6 个关节增量；底盘不参与其推理。当前相机内参由58°视场假设推导，URDF安装外参尚需实机标定。

原单臂策略在固定底座上训练；移植后使用机械臂编码器正解的底盘相对坐标，不输入底盘/球的世界真值。持续行驶、打滑和较强外力下可能掉球。三全向轮是球形支撑 + 限幅驱动方向牵引的近似模型，没有显式被动滚子。视觉深度、反馈力矩和热模型也尚不足以直接实机部署。

学习策略与基线的旧成绩见 `reports/history/`，带有模型演化过程中的限制；不是新框架的实机可靠性承诺。重构验证见 [验证记录](docs/validation.md)。

## 测试、发布与旧文件

```bash
uv sync --frozen --extra dev --extra assets
MUJOCO_GL=egl uv run pytest -q
uv run python scripts/validate_assets.py
uv build
```

构建的 wheel 包含必要资产、配置、权重和许可证，可安装到其他路径后运行。完整源码迁移可复制整个仓库，或运行 `python scripts/export_project.py` 生成轻量分发包（不含 `.venv`、缓存和归档重复资源）。

旧目录的原始文件保存在 `archive/`，没有丢弃；旧代码仅作历史参考，不维护其中的绝对路径/老启动命令。新运行路径只使用 `src/`、`assets/`、`configs/`、`checkpoints/`。

代码和模型的许可来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)；保留现有上游许可证，不把全部CAD资源擅自重新授权。
