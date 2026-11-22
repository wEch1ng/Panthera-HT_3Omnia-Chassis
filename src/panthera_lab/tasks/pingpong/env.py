"""Ping-pong resets, measurements and evaluator; separate from reusable robot plant."""

import numpy as np
import mujoco as mj
from panthera_lab.robot.config import HOME
from panthera_lab.sim.plant import RobotPlant
from panthera_lab.tasks.pingpong.scene import build_model
from panthera_lab.control.visual_baseline import SensorPolicy
from panthera_lab.perception.rgbd_ball import locate_ball


class Env(RobotPlant):
    def __init__(self, force_controls=False):
        super().__init__(build_model(force_controls))
        m = self.m
        self.blade = m.geom("paddle_blade").id
        self.ball = m.geom("ball_geom").id
        ball_joint = m.joint("ball_joint").id
        self.ball_qadr = int(m.jnt_qposadr[ball_joint])
        self.ball_vadr = int(m.jnt_dofadr[ball_joint])
        k = self.kin
        cam = self.cam
        self.origin = k.cam_xmat[cam].reshape(3, 3).T @ (k.geom_xpos[self.blade] - k.cam_xpos[cam])
        self.basis = k.cam_xmat[cam].reshape(3, 3).T @ k.geom_xmat[self.blade].reshape(3, 3)
        mj.mj_jacGeom(m, k, self.jac, self.jacr, self.blade)
        self.ik = np.linalg.pinv(self.jacr[:2, self.layout.arm_dofs])
        self.axes = k.xaxis[self.layout.joints[:3]].copy()
        self.home_R = k.geom_xmat[self.blade].reshape(3, 3).copy()
        self.actor = SensorPolicy(self.origin, self.basis, self.home_R, self.ik, self.axes)

    def observe(self):
        frame = self.camera.capture(self.d)
        self.latest_rgb = frame.rgb
        local, _ = locate_ball(frame.rgb, frame.depth, frame.fovy)
        return local

    def reset(self, seed):
        m, d = self.m, self.d
        rng = np.random.default_rng(seed)
        mj.mj_resetData(m, d)
        d.qpos[self.layout.arm_qpos] = HOME
        mj.mj_forward(m, d)
        self.center = d.geom_xpos[self.blade].copy()
        normal = d.geom_xmat[self.blade].reshape(3, 3)[:, 2]
        assert normal[2] > 0.999
        a, b = rng.uniform(0, 2 * np.pi, 2)
        offset = rng.uniform(0.012, 0.045) * np.array([np.cos(a), np.sin(a)])
        d.qpos[self.ball_qadr : self.ball_qadr + 3] = self.center + [*offset, 0.0281]
        d.qvel[self.ball_vadr : self.ball_vadr + 2] = rng.uniform(0.015, 0.09) * np.array(
            [np.cos(b), np.sin(b)]
        )
        mj.mj_forward(m, d)
        self.actor.reset(d.qpos[self.layout.wheel_qpos])
        self.wdist = 0.0
        self.lastbase = d.qpos[:2].copy()

    def error(self):
        v = self.d.geom_xpos[self.ball] - self.d.geom_xpos[self.blade]
        rot = self.d.geom_xmat[self.blade].reshape(3, 3)
        return (v - rot[:, 2] * np.dot(v, rot[:, 2]))[:2]

    def advance(self, gains):
        self.last_motor_observation = self.motor_feedback().observation()
        target, omega = self.actor.command(
            self.observe(), self.motor_feedback().position[:3], gains
        )
        self.apply_command(target, omega)

    def training_loss(self, g, seed):
        self.reset(seed)
        errors = []
        for _ in range(90):
            self.advance(g)
            errors.append(
                85.0 if self.actor.missing else min(85.0, 1000 * np.linalg.norm(self.actor.prev))
            )
        return (
            np.mean(errors) + max(errors[-30:]) + 85 * (1 - self.actor.visible / self.actor.frames)
        )

    def episode(self, g, seed, sec):
        self.reset(seed)
        err = []
        held = []
        for _ in range(round(sec * 30)):
            self.advance(g)
            self.wdist += np.linalg.norm(self.d.qpos[:2] - self.lastbase)
            self.lastbase[:] = self.d.qpos[:2]
            err.append(np.linalg.norm(self.error()))
            held.append(any(set(c.geom) == {self.ball, self.blade} for c in self.d.contact))
            if (
                err[-1] > 0.085
                or self.d.geom_xpos[self.ball, 2] < self.d.geom_xpos[self.blade, 2] - 0.02
            ):
                break
        ok = len(err) == round(sec * 30) and max(err[-30:]) < 0.02 and all(held[-30:])
        lost = sec - len(err) / 30
        return dict(
            visible_fraction=self.actor.visible / self.actor.frames,
            seed=seed,
            success=bool(ok),
            duration_s=len(err) / 30,
            mean_mm=float(np.mean(err) * 1000),
            last_second_max_mm=float(max(err[-30:]) * 1000),
            base_travel_mm=self.wdist * 1000,
            loss=float(100 * lost + np.mean(err) * 1000 + max(err[-30:]) * 1000),
        )
