"""Nine-action residual PPO task. Inputs/reward use sensor estimates only."""

import json
from collections import deque
import numpy as np
from panthera_lab.tasks.pingpong.env import Env
from panthera_lab.robot.config import HOME
from panthera_lab.robot.motor_limits import RATED_SPEED, SAFETY
from panthera_lab.resources import resource


class Task:
    def __init__(self, force_controls=False):
        self.env = Env(force_controls=force_controls)
        self.gains = json.loads(resource("configs", "visual_baseline.json").read_text())["gains"]
        self.episode = 0

    def reset(self, seed):
        self.env.reset(seed)
        self.rng = np.random.default_rng(seed + 12)
        self.t = 0
        self.action = np.zeros(9)
        self.queue = deque([np.zeros(9) for _ in range(2)])
        self.sample()
        return self.obs()

    def sample(self):
        e = self.env
        point = e.observe()
        # Noise/dropout is applied to the measured RGB-D estimate, not world truth.
        if point is not None:
            point += self.rng.normal(0, 0.0007, 3)
            if self.rng.random() < 0.02:
                point = None
        self.target, self.omega = e.actor.command(
            point, e.motor_feedback().position[:3], self.gains
        )

    def obs(self):
        e = self.env
        a = e.actor
        f = e.motor_feedback()
        ball = np.r_[
            np.zeros(2) if a.prev is None else a.prev / 0.08,
            a.vel / 0.5,
            float(a.missing == 0),
            min(a.missing / 6, 1),
        ]
        return np.r_[
            ball,
            (f.position[:3] - a.wheel_ref) * 0.1,
            (f.position[3:] - HOME) / 0.2,
            f.velocity / RATED_SPEED,
            f.torque / np.r_[[3.5] * 3, [6, 10, 10, 6, 2, 2]],
            self.action,
        ].astype(np.float32)

    def step(self, action, wheel_override=None):
        e = self.env
        action = np.clip(action, -1, 1)
        self.queue.append(action.copy())
        applied = self.queue.popleft()
        e.apply_command(
            self.target + applied[3:] * 0.025,
            self.omega + applied[:3] * 0.8 if wheel_override is None else wheel_override,
        )
        self.t += 1
        # Hidden disturbance is an environment event only, never an observation.
        if self.t == 60:
            angle = self.rng.uniform(0, 2 * np.pi)
            e.d.qvel[e.ball_vadr : e.ball_vadr + 2] += 0.04 * np.array(
                [np.cos(angle), np.sin(angle)]
            )
        self.sample()
        a = e.actor
        f = e.motor_feedback()
        distance = 0.15 if a.prev is None else np.linalg.norm(a.prev)
        reward = (
            1.0
            - min(distance / 0.04, 4.0)
            - 0.04 * np.sum((action - self.action) ** 2)
            - 0.02 * np.sum(action**2)
            - 0.002 * np.sum(np.abs(f.torque * f.velocity))
            - 0.3 * (a.missing > 0)
        )
        speed_fail = bool(np.any(np.abs(f.velocity) > SAFETY * RATED_SPEED * 1.01))
        failure = a.missing > 12 or distance > 0.1 or speed_fail
        if failure:
            reward -= 5
        self.action = action.copy()
        return self.obs(), reward, failure or self.t >= 300, failure

    def close(self):
        self.env.renderer.close()
