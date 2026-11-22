"""Original fixed-base recovery network, using encoder FK in chassis frame."""

from collections import deque
import numpy as np
import torch
import mujoco as mj
from panthera_lab.tasks.pingpong.env import Env
from panthera_lab.robot.config import HOME
from panthera_lab.resources import resource
from panthera_lab.learning.ppo import ActorCritic
from panthera_lab.perception.rgbd_ball import RGBDBallTracker
from panthera_lab.robot.motor_limits import RATED_SPEED, SAFETY


class SingleArmTask:
    def __init__(self, force_controls=False, checkpoint=None):
        self.env = Env(force_controls=force_controls)
        e = self.env
        ck = torch.load(
            checkpoint or resource("checkpoints", "pingpong", "arm_recovery.pt"),
            map_location="cpu",
            weights_only=True,
        )
        assert ck["args"]["observation_id"] == "d405-rgbd-recovery-v1"
        self.policy = ActorCritic(42, 6)
        self.policy.load_state_dict(ck["model"])
        self.policy.eval()
        self.args = ck["args"]
        self.iteration = ck["iter"]
        self.tracker = RGBDBallTracker()
        self.fk = mj.MjData(e.m)
        self.grip = (
            e.m.body("gripper").id
            if any(e.m.body(i).name == "gripper" for i in range(e.m.nbody))
            else int(e.m.body_parentid[e.m.body("paddle").id])
        )
        self.jac = np.zeros((3, e.m.nv))
        self.rot = self.jac.copy()
        self.limits = np.array([e.m.jnt_range[e.m.joint(f"joint{i}").id] for i in range(1, 7)])

    def reset(self, seed):
        self.missing_frames = 0
        self.outside_frames = 0
        self.failure_reason = ""
        self.env.reset(seed)
        self.tracker.reset()
        self.history = np.zeros((3, 6))
        self.delay = deque([np.zeros(6) for _ in range(2)])
        self.target = HOME.copy()
        self.t = 0
        return self.observe()

    def observe(self):
        e = self.env
        f = e.motor_feedback()
        k = self.fk
        # Independent kinematics: floating base remains identity, only arm encoders enter.
        mj.mj_resetData(e.m, k)
        k.qpos[e.layout.arm_qpos] = f.position[3:]
        k.qvel[e.layout.arm_dofs] = f.velocity[3:]
        mj.mj_forward(e.m, k)
        local = e.observe()
        point = (
            None if local is None else k.cam_xpos[e.cam] + k.cam_xmat[e.cam].reshape(3, 3) @ local
        )
        point, velocity, visible, age = self.tracker.update(point)
        c = k.geom_xpos[e.blade]
        normal = k.geom_xmat[e.blade].reshape(3, 3)[:, 2]
        g = k.xmat[self.grip].reshape(3, 3)
        u = g[:, 0]
        w = -g[:, 2]
        rel = np.zeros(3) if point is None else point - c
        mj.mj_jacGeom(e.m, k, self.jac, self.rot, e.blade)
        extra = [
            rel @ normal / 0.1,
            velocity @ normal / 0.5,
            visible,
            min(age / 0.3, 5),
            (c[2] - self.args["target_height"]) / 0.1,
            self.args["target_u"] / 0.08,
            self.args["target_w"] / 0.08,
            self.jac[2, e.layout.arm_dofs] @ f.velocity[3:] / 0.5,
        ]
        self.observation = np.r_[
            [rel @ u / 0.08, rel @ w / 0.08, velocity @ u / 0.5, velocity @ w / 0.5],
            (f.position[3:] - HOME) / 0.2,
            f.velocity[3:] * 0.1,
            self.history.ravel(),
            extra,
        ].astype(np.float32)
        self.env.gravity = k.qfrc_bias[e.layout.arm_dofs].copy()
        # Episode termination uses the current RGB-D measurement, never ball truth.
        self.missing_frames = self.missing_frames + 1 if local is None else 0
        outside = local is not None and (np.hypot(rel @ u, rel @ w) > 0.20 or rel @ normal < -0.08)
        self.outside_frames = self.outside_frames + 1 if outside else 0
        self.failure_reason = (
            "D405 lost ball for 0.4 s"
            if self.missing_frames >= 12
            else "D405 ball outside recovery area"
            if self.outside_frames >= 6
            else ""
        )
        return self.observation

    def step(self, unused=None, wheel_override=None):
        with torch.no_grad():
            a = np.clip(self.policy.actor(torch.tensor(self.observation)).numpy(), -1, 1)
        self.delay.append(a)
        applied = self.delay.popleft()
        f = self.env.motor_feedback()
        self.target = np.clip(self.target + applied * 0.04, self.limits[:, 0], self.limits[:, 1])
        delta = SAFETY * RATED_SPEED[3:] / 30
        self.target = np.clip(self.target, f.position[3:] - delta, f.position[3:] + delta)
        self.env.apply_command(
            self.target, np.zeros(3) if wheel_override is None else wheel_override
        )
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = a
        self.t += 1
        obs = self.observe()
        failure = bool(self.failure_reason)
        return obs, 0.0, failure, failure

    def close(self):
        self.env.renderer.close()
