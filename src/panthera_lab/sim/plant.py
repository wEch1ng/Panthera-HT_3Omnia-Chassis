"""Task-independent motor/omni-contact plant; only sensor packets leave this layer."""

import numpy as np
import mujoco as mj
from panthera_lab.robot.config import HOME, WHEEL_BODIES, SUBSTEPS
from panthera_lab.robot.layout import RobotLayout
from panthera_lab.robot.motor_limits import MotorFeedback, clip_command
from panthera_lab.perception.camera import WristCamera


class RobotPlant:
    def __init__(self, model):
        self.m = model
        self.d = mj.MjData(model)
        self.layout = RobotLayout(model)
        self.external_force_callback = None
        self.wheels = {
            model.geom(n + "_contact").id: (model.body(n).id, model.joint(n + "_joint").id)
            for n in WHEEL_BODIES
        }
        self.floor = model.geom("floor").id
        self.force = np.zeros(6)
        self.jac = np.zeros((3, model.nv))
        self.jacr = self.jac.copy()
        self.camera = WristCamera(model)
        self.cam = self.camera.camera_id
        self.renderer = self.camera.renderer
        self.opt = self.camera.options
        self.kin = mj.MjData(model)
        self.kin.qpos[self.layout.arm_qpos] = HOME
        mj.mj_forward(model, self.kin)
        self.gravity = self.kin.qfrc_bias[self.layout.arm_dofs].copy()

    def motor_feedback(self):
        return MotorFeedback(
            self.d.qpos[self.layout.qpos].copy(),
            self.d.qvel[self.layout.dofs].copy(),
            self.d.actuator_force[self.layout.actuators].copy(),
        )

    def read_sensors(self):
        from panthera_lab.interfaces import SensorPacket

        frame = self.camera.capture(self.d)
        self.latest_rgb = frame.rgb
        return SensorPacket(self.d.time, frame, self.motor_feedback())

    def close(self):
        self.camera.close()

    def apply_command(self, target, omega):
        m, d = self.m, self.d
        for _ in range(SUBSTEPS):
            if self.external_force_callback is not None:
                self.external_force_callback(m, d)
            feedback = self.motor_feedback()
            requested = np.r_[
                0.6 * (omega - feedback.velocity[:3]),
                self.gravity
                + (target - feedback.position[3:]) * [240, 360, 300, 180, 100, 80]
                - feedback.velocity[3:] * [18, 24, 20, 12, 8, 6],
            ]
            d.ctrl[self.layout.actuators] = clip_command(requested, feedback.velocity)
            d.qfrc_applied[:] = 0
            for i, c in enumerate(d.contact):
                g1, g2 = c.geom
                wheel = self.wheels.get(g2 if g1 == self.floor else g1 if g2 == self.floor else -1)
                if wheel is None:
                    continue
                body, j = wheel
                axis = d.xaxis[j]
                t = np.array([axis[1], -axis[0], 0.0])
                mj.mj_jac(m, d, self.jac, None, c.pos, body)
                row = t @ self.jac
                mj.mj_contactForce(m, d, i, self.force)
                limit = 0.8 * abs(self.force[0])
                d.qfrc_applied[:] += row * np.clip(-110 * (row @ d.qvel), -limit, limit)
            mj.mj_step(m, d)
