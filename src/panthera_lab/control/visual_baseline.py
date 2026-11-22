"""Deterministic visual baseline; retained as residual PPO reference."""

from collections import deque
import numpy as np
from panthera_lab.robot.config import HOME, WHEEL_RADIUS as R


class SensorPolicy:
    """Controller boundary: fixed calibration, RGB-D detection and wheel encoders only."""

    def __init__(self, origin, basis, home_R, ik, axes):
        self.origin, self.basis, self.home_R, self.ik, self.axes = origin, basis, home_R, ik, axes

    def reset(self, wheel_positions):
        self.prev = None
        self.vel = np.zeros(2)
        self.wheel_ref = wheel_positions.copy()
        self.delay = deque([(HOME.copy(), np.zeros(3)) for _ in range(2)])
        self.visible = 0
        self.frames = 0
        self.missing = 0

    def command(self, point, wheel_positions, gains):
        self.frames += 1
        kp, kd, wp, wd = gains
        if point is not None:
            self.visible += 1
            err_local = self.basis.T @ (point - self.origin)
            err_local[2] = 0
            e = (self.home_R @ err_local)[:2]
            raw = np.zeros(2) if self.prev is None else (e - self.prev) * 30 / (1 + self.missing)
            self.vel = 0.55 * self.vel + 0.45 * raw
            self.prev = e
            self.missing = 0
        else:
            self.missing += 1
            e = (
                np.zeros(2)
                if self.prev is None or self.missing > 3
                else self.prev + self.vel * self.missing / 30
            )
            if self.missing > 3:
                self.vel[:] = 0
                self.prev = None
        tilt = np.clip(kp * e + kd * self.vel, -0.13, 0.13)
        target = HOME + self.ik @ [tilt[1], -tilt[0]]
        tangents = np.column_stack((self.axes[:, 1], -self.axes[:, 0]))
        trips = (wheel_positions - self.wheel_ref) * R
        odom = (2 / 3) * (tangents.T @ trips)
        speed = np.clip(wp * e + wd * self.vel - 0.6 * odom, -0.18, 0.18)
        omega = tangents @ speed / R
        self.delay.append((target, omega))
        target, omega = self.delay.popleft()
        return target, omega
