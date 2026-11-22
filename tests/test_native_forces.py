"""Native external-force controls must not become motor telemetry or policy inputs."""

import numpy as np
import mujoco as mj
from panthera_lab.tasks.pingpong.arm_recovery import SingleArmTask
from panthera_lab.sim.forces import NativeForces
from panthera_lab.runtime.session import InferenceSession


def test_integration():
    task = SingleArmTask(force_controls=True)
    try:
        session = InferenceSession(task)
        env = task.env
        forces = NativeForces(env.m)
        env.external_force_callback = forces.apply
        assert env.m.nu == 18 and env.motor_feedback().observation().shape == (27,)
        assert session.observation.shape == (42,)
        for index, (body, axis, value) in enumerate(
            [("ball", 0, 0.006), ("paddle", 1, 1.0), ("base_link", 2, 10.0)]
        ):
            session.restart()
            forces.clear(env.d)
            actuator = env.m.actuator(
                f"{dict(ball='Ball', paddle='Paddle', base_link='Base')[body]}_F{'xyz'[axis]}_N"
            ).id
            env.d.ctrl[actuator] = value
            forces.apply(env.m, env.d)
            assert env.d.xfrc_applied[env.m.body(body).id, axis] == value
            assert np.linalg.norm(env.d.xfrc_applied) > 0
            # It is a force channel, not a physical motor or electrical feedback.
            mj.mj_forward(env.m, env.d)
            np.testing.assert_array_equal(env.d.actuator_force[9:], 0)
            session.advance(wheel_override=np.zeros(3))
            assert env.d.ctrl[actuator] == value
            assert env.motor_feedback().torque.shape == (9,)
        # Check actual dynamics by paired force/no-force runs, not just array values.
        final = []
        for value in (0.0, 0.006):
            session.restart()
            forces.clear(env.d)
            env.d.ctrl[env.m.actuator("Ball_Fx_N").id] = value
            session.advance(wheel_override=np.zeros(3))
            final.append(env.d.qvel[15:18].copy())
        assert np.linalg.norm(final[1] - final[0]) > 0.001
        forces.clear(env.d)
        np.testing.assert_array_equal(env.d.ctrl[9:], 0)
        session.restart()
        np.testing.assert_array_equal(env.d.xfrc_applied, 0)
        for _ in range(60):
            assert not session.advance(wheel_override=np.zeros(3))
        print(
            "PASS: native control -> force -> motion; 9 motor signals / 42 policy inputs unchanged; clear/reset; continued inference."
        )
    finally:
        task.close()
