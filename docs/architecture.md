# 架构与扩展

## 分层

1. `robot`：机器人的命名和硬件约束。`RobotLayout` 从模型名称查询关节位置、速度和执行器索引，不要求未来添加的物体遵守旧数组偏移。
2. `sim.model`：组装底盘、完整单臂和URDF相机。返回XML，可由任务扩充，再编译MuJoCo。
3. `sim.plant.RobotPlant`：480 Hz动力学、三全向轮接触近似、电机限幅。对外提供电机反馈与原始相机包、接收目标角度/轮速。
4. `perception`：RGB-D相机和橙球识别/跟踪。相机渲染内部需要仿真几何，输出仅是像素与深度；策略没有得到物体真值。
5. `tasks`：场景物体、初始化、观测适配、动作时序、回合结束。原单臂兼容层和残差PPO任务共用同一机器人。
6. `apps`：运行、原生界面、评测、录制。独立评测可以用真值计分，但不得把它回传给策略。
7. `learning`：网络、PPO更新、并行采样。新运行写到用户指定的`outputs`目录，内置权重只读。

## 传感器/动作约定

`SensorPacket(time_s, camera, motors)`：

- `camera.rgb`：uint8 RGB，H×W×3；默认128×96。
- `camera.depth`：对齐到RGB，光轴距离（米），不是欧氏斜距。
- `camera.fovy`：58°假设值；实机适配应扩展为完整已标定内参。
- `motors.position/velocity/torque`：9维，左轮、右轮、后轮、关节1..6；输出轴rad、rad/s、N·m。
- `time_s`：当前仿真采样时间。未来驱动器必须提供真实时间戳，不得用固定帧率掩盖丢帧。

`RobotCommand(arm_position, wheel_velocity)`：6个绝对关节目标角度 + 3个轮速。底层伺服统一限制力矩与速度。当前内置旧策略仍使用自身42维特征适配器；图像策略可直接消费原始包，不必改动球检测器。

## 新增任务

在`src/panthera_lab/tasks/your_task/`中：

1. `scene.py`调用`robot_xml()`，添加自己的工具、物体和场景。
2. 编写一个任务类，具有`env`（含`m/d/cam/latest_rgb/external_force_callback`）、`reset(seed)`、`step(action,wheel_override=None)`、`close()`。
3. `reset`返回策略观测；`step`返回`(observation, reward, done, failure)`。查看器仅在failure时自动重开，不强制使用训练时长。
4. 工厂返回`TaskBundle(task, policy, iteration, arm_only)`。若任务内部推理，`policy=None`；外部策略需要`actor(torch_tensor)`返回动作。
5. 在registry里注册一个名称，或用`--task your_module:make_task`显式加载。

见`examples/custom_task.py`，这是包装现有任务并展示工厂接口的完整小例子。架构支持新任务，但还没提供通用Gymnasium/ROS/CAN适配器，不能把接口示例当作已实现能力。

## 端到端方向

先运行`examples/image_policy.py`确认无球本体与原始图像接口可用，然后换入自己的模型。使用`panthera record`采集同步RGB-D、电机反馈和动作；具体损失、时序网络、数据筛选与训练器由新的任务/learning模块实现。

## 文件与权重

`resources.resource()`支持源码checkout和wheel内置资源；可用`PANTHERA_RESOURCE_ROOT`显式指定完整资源根。切换当前目录不影响加载。不使用sys.path注入、不读取个人目录、不依赖另一个仓库。

`checkpoints/manifest.json`记录权重来源和哈希。参数维度一致不等于语义一致：原单臂42维与残差PPO42维不同，不能混用权重。不同架构应使用独立task和检查点元数据。
