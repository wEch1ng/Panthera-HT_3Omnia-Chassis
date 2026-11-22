"""World-frame external wrench inputs exposed in MuJoCo's native Control panel."""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj

# All values are forces in newtons. These controls are not robot motors.
FORCE_TARGETS = (('ball', 'Ball', .02), ('paddle', 'Paddle', 5.), ('base_link', 'Base', 30.))


def add_force_controls(root):
    world, actuators = root.find('worldbody'), root.find('actuator')
    for body_name, label, limit in FORCE_TARGETS:
        body = world.find(f".//body[@name='{body_name}']")
        site = f'ui_force_{body_name}'
        ET.SubElement(body, 'site', name=site, size='.001', rgba='0 0 0 0', group='5')
        for axis in 'xyz':
            # UI-only controls: no dynamics from this actuator itself. Their values
            # feed xfrc_applied in world coordinates below, independently of motors.
            ET.SubElement(actuators, 'general', name=f'{label}_F{axis}_N', site=site,
                          gear='0 0 0 0 0 0', gainprm='0', group='5',
                          ctrllimited='true', ctrlrange=f'{-limit} {limit}')


class NativeForces:
    def __init__(self, model):
        self.ids = [model.body(name).id for name, _, _ in FORCE_TARGETS]
        self.perturb = mj.MjvPerturb()
        self.pulses = []

    def snapshot(self, source):
        # Copy under the native viewer lock; the physics loop reads its own copy.
        for field in ('active', 'active2', 'select', 'skinselect', 'flexselect', 'scale', 'localmass'):
            setattr(self.perturb, field, getattr(source, field))
        for field in ('localpos', 'refpos', 'refquat', 'refselpos'):
            getattr(self.perturb, field)[:] = getattr(source, field)

    def clear(self, data):
        data.ctrl[9:] = 0
        data.xfrc_applied[:] = 0
        self.perturb.active = self.perturb.active2 = 0
        self.pulses.clear()

    def apply(self, model, data):
        data.xfrc_applied[:] = 0
        if self.perturb.select > 0 and self.perturb.active:
            mj.mjv_applyPerturbForce(model, data, self.perturb)
        for i, body in enumerate(self.ids):
            data.xfrc_applied[body, :3] += np.clip(data.ctrl[9+3*i:12+3*i],
                model.actuator_ctrlrange[9+3*i:12+3*i, 0],
                model.actuator_ctrlrange[9+3*i:12+3*i, 1])
        for axis, force, _ in self.pulses:
            data.xfrc_applied[self.ids[0], axis] += force
