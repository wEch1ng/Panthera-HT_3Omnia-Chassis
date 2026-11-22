# 底盘 + Panthera 机械臂

本文件夹包含完整模型资源，可独立移动，不依赖原来的两个模型目录。

## 在模型查看器中打开

1. 把整个 `Chassis_Panthera_Combined` 文件夹拖入查看器（或使用文件夹导入），不要只导入 URDF。
2. 选择 `robot.urdf`：外观和碰撞均使用 STL，优先用于检查完整模型。
3. 若查看器支持 GLB，可选择 `robot_color.urdf`，使用机械臂原始彩色外观。

两份 URDF 的机械结构相同；所有网格路径均相对于 URDF 所在目录。
STL 版本不包含机械臂原 GLB 的材质配色。

## 安装关系

使用 follower 机械臂，底盘根节点为 `base_link`，机械臂基座为 `arm_base_link`。
二者通过 `arm_mount_joint` 固定连接，默认 `xyz="0 0 0" rpy="0 0 0"`。
这是基座坐标系重合的假设；如有实际安装偏移，请修改所使用 URDF 中该固定关节的 origin。
保留源模型的质量、惯性、关节轴与限位；夹爪固定，相机默认关闭。
