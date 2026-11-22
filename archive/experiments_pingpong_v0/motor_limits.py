"""Module-output units. Wheel data: user catalog printed page 4; arm: reference repo."""
from dataclasses import dataclass
import numpy as np
import arm_motor_limits as arm

NAMES = ('left_wheel_joint','right_wheel_joint','back_wheel_joint')+tuple(f'joint{i}' for i in range(1,7))
MODELS = ('HTDW-5022-02-DNE',)*3+arm.MOTOR_MODEL
RATED_TORQUE = np.r_[[3.5]*3,arm.RATED_TORQUE_NM]
STALL_TORQUE = np.r_[[13.]*3,arm.STALL_TORQUE_NM]
RATED_SPEED = np.r_[[82.]*3,arm.RATED_SPEED_RPM]*np.pi/30
NO_LOAD_SPEED = np.r_[[123.]*3,arm.NO_LOAD_SPEED_RPM]*np.pi/30
RATED_CURRENT = np.r_[[2.]*3,arm.RATED_CURRENT_A]
PEAK_CURRENT = np.r_[[10.]*3,arm.PEAK_CURRENT_A]
SAFETY = .9


def torque_cap(velocity):
    # Same conservative approximation as reference fast_xyz_env, not a measured TN curve.
    return SAFETY*STALL_TORQUE*np.clip(1-np.abs(velocity)/(SAFETY*NO_LOAD_SPEED),0,1)


def clip_command(requested, velocity):
    cap=torque_cap(velocity)
    torque=np.clip(requested,-cap,cap)
    # Above rated-speed margin, allow braking only; never clamp the physical state.
    return np.where((np.abs(velocity)>=SAFETY*RATED_SPEED)&(torque*velocity>0),0.,torque)


@dataclass(frozen=True)
class MotorFeedback:
    position: np.ndarray  # output-shaft radians; accumulated wheel angle
    velocity: np.ndarray  # output-shaft rad/s
    torque: np.ndarray  # output-shaft Nm; measured/estimated motor feedback

    def observation(self):
        return np.r_[self.position,self.velocity/RATED_SPEED,self.torque/RATED_TORQUE].astype(np.float32)
