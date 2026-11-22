"""Sensor/action boundary usable by vision-feature, imitation or image policies."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np
from panthera_lab.robot.motor_limits import MotorFeedback


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray  # H x W x 3 uint8, RGB
    depth: np.ndarray  # H x W, optical-axis metres aligned to RGB
    fovy: float


@dataclass(frozen=True)
class SensorPacket:
    time_s: float
    camera: CameraFrame
    motors: MotorFeedback


@dataclass(frozen=True)
class RobotCommand:
    arm_position: np.ndarray  # six output-joint angles in rad
    wheel_velocity: np.ndarray  # left/right/back output speeds in rad/s


class SensorPolicy(Protocol):
    def reset(self) -> None: ...
    def act(self, sensors: SensorPacket) -> RobotCommand: ...
