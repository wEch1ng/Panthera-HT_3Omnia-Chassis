"""Viewer-only mouse forces. This object is never passed to the policy."""
import numpy as np
import mujoco as mj


class MouseForce:
    def __init__(self, model, data, scene, options):
        self.model, self.data, self.scene, self.options = model, data, scene, options
        self.perturb = mj.MjvPerturb()
        self.last_wrench = np.zeros(6)
        self.drag_button = None
        for name in ('mjVIS_SELECT', 'mjVIS_PERTFORCE', 'mjVIS_PERTOBJ'):
            options.flags[getattr(mj.mjtVisFlag, name)] = True

    @property
    def selected_name(self):
        body = self.perturb.select
        return (self.model.body(body).name or f'body {body}') if body > 0 else 'none'

    def select(self, relx, rely, aspect):
        point = np.zeros(3)
        geom, flex, skin = (np.full(1, -1, np.int32) for _ in range(3))
        body = mj.mjv_select(self.model, self.data, self.options, aspect,
                             relx, rely, self.scene, point, geom, flex, skin)
        if body <= 0:
            return False
        self.release()
        self.perturb.select = body
        self.perturb.skinselect = int(skin[0])
        self.perturb.flexselect = int(flex[0])
        self.perturb.localpos[:] = self.data.xmat[body].reshape(3, 3).T @ (point - self.data.xpos[body])
        return True

    def begin(self, rotate=False, button=None):
        if self.perturb.select <= 0:
            return False
        mj.mjv_initPerturb(self.model, self.data, self.scene, self.perturb)
        self.perturb.active = mj.mjtPertBit.mjPERT_ROTATE if rotate else mj.mjtPertBit.mjPERT_TRANSLATE
        self.drag_button = button
        return True

    def move(self, dx, dy, horizontal=False):
        if not self.perturb.active:
            return
        if self.perturb.active == mj.mjtPertBit.mjPERT_ROTATE:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if horizontal else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if horizontal else mj.mjtMouse.mjMOUSE_MOVE_V
        mj.mjv_movePerturb(self.model, self.data, action, dx, dy, self.scene, self.perturb)

    def release(self):
        self.perturb.active = 0
        self.perturb.active2 = 0
        self.drag_button = None
        self.last_wrench[:] = 0
        self.data.xfrc_applied[:] = 0

    def reset(self):
        self.release()
        self.perturb.select = 0

    def apply_forces(self, model, data, pulses=()):
        """Re-evaluate the spring at every physics substep, not just video frames."""
        data.xfrc_applied[:] = 0
        if self.perturb.active and self.perturb.select > 0:
            mj.mjv_applyPerturbForce(model, data, self.perturb)
        # Pulses and dragging add, even when they target the same body.
        for body, axis, force, _ in pulses:
            data.xfrc_applied[model.body(body).id, axis] += force
        self.last_wrench[:] = data.xfrc_applied[self.perturb.select] if self.perturb.select > 0 else 0
