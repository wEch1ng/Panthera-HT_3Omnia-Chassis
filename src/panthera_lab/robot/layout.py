"""Resolve named robot channels once; new bodies must not silently shift observations."""

import numpy as np
from .config import MOTOR_JOINTS


class RobotLayout:
    def __init__(self, model):
        self.joints = np.array([model.joint(name).id for name in MOTOR_JOINTS])
        self.qpos = model.jnt_qposadr[self.joints].copy()
        self.dofs = model.jnt_dofadr[self.joints].copy()
        self.actuators = np.array([model.actuator(name + "_torque").id for name in MOTOR_JOINTS])
        self.wheel_qpos, self.arm_qpos = self.qpos[:3], self.qpos[3:]
        self.wheel_dofs, self.arm_dofs = self.dofs[:3], self.dofs[3:]
        self.arm_limits = model.jnt_range[self.joints[3:]].copy()
