# 合并模型的 MuJoCo 可视化

在工作区根目录运行：

```bash
Chassis_Panthera_Combined/.venv/bin/python Chassis_Panthera_Combined/mujoco/visualize_mujoco.py
```

程序读取 `../robot_color.urdf`，转换网格并生成本目录的 `scene.xml`。
底盘通过 `base_free_joint` 自由平移和旋转，受重力和地面接触作用。
三个轮关节和六个机械臂关节可动，默认仅对这九个关节进行偏置力补偿，浮动基座不施加补偿。
轮子沿用 STL 凸包碰撞，尚未单独建模全向轮滚子，也未添加底盘行驶控制器。
安装位姿沿用 URDF 中的零偏移；地面高度按零位网格最低点设置。

鼠标左键旋转，右键平移，滚轮缩放。双击连杆后 Ctrl+右键拖动施力。
空格暂停，R 复位，G 切换重力补偿。

无窗口验证及截图：

```bash
Chassis_Panthera_Combined/.venv/bin/python Chassis_Panthera_Combined/mujoco/visualize_mujoco.py --check
MUJOCO_GL=egl Chassis_Panthera_Combined/.venv/bin/python Chassis_Panthera_Combined/mujoco/visualize_mujoco.py --screenshot Chassis_Panthera_Combined/mujoco/preview.png
```

迁移到其他电脑后需重新创建 Python 环境，并安装 `mujoco numpy trimesh Pillow`。
