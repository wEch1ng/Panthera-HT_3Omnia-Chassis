# 配置

`visual_baseline.json` 是残差PPO使用的四参数视觉基线，保持旧训练值。

本体频率、关节命名、默认姿态及相机尺寸集中在 `src/panthera_lab/robot/config.py`；电机表在同目录 `motor_limits.py` 与 `arm_motor_limits.py`。这些值与现有权重有关，修改后需要重新验证，不能仅凭相同维度继续使用旧权重。

任务、权重路径、输出目录、训练规模、评测时长通过 CLI 参数指定。未建立一套与代码重复、却不起作用的配置文件。
