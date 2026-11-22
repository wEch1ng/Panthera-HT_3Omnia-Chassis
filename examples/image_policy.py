"""Sensor API example: accept images, hold a pose. This is NOT a trained image model.

Replace act() with your network, or record demonstrations with `panthera record`.
No MjData or world coordinates are accepted by this policy.
"""

import numpy as np
from panthera_lab.interfaces import SensorPacket, RobotCommand
from panthera_lab.robot.config import HOME


class ImagePolicy:
    def reset(self):
        pass

    def act(self, sensors: SensorPacket) -> RobotCommand:
        assert sensors.camera.rgb.ndim == 3
        return RobotCommand(HOME.copy(), np.zeros(3))


if __name__ == "__main__":
    from panthera_lab.sim.plant import RobotPlant
    from panthera_lab.sim.model import robot_xml, compile_model
    import mujoco as mj

    plant = RobotPlant(compile_model(robot_xml()))
    policy = ImagePolicy()
    try:
        plant.d.qpos[plant.layout.arm_qpos] = HOME
        mj.mj_forward(plant.m, plant.d)
        policy.reset()
        for _ in range(30):
            command = policy.act(plant.read_sensors())
            plant.apply_command(command.arm_position, command.wheel_velocity)
        print("Image-policy sensor/action interface OK (no ball task, no learned network).")
    finally:
        plant.close()
