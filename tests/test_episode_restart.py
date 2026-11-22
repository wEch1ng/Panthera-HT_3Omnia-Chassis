"""Integration checks for brief occlusion, actual escape and clean repeat episodes."""

import numpy as np
from panthera_lab.tasks.pingpong.arm_recovery import SingleArmTask
from panthera_lab.runtime.session import InferenceSession
from panthera_lab.robot.config import HOME
import mujoco as mj


def test_integration():
    task = SingleArmTask()
    try:
        session = InferenceSession(task)
        env = task.env
        initial_qpos = env.d.qpos.copy()
        initial_qvel = env.d.qvel.copy()
        initial_obs = session.observation.copy()
        observe = env.observe

        # An intermittent RGB-D dropout must not restart a recoverable trial.
        env.observe = lambda: None
        for _ in range(3):
            assert not session.advance(wheel_override=np.zeros(3))
        env.observe = observe
        assert not session.advance(wheel_override=np.zeros(3))
        assert session.episode == 1 and task.missing_frames == 0
        assert np.isfinite(session.observation).all()

        # Move the simulated ball out of the view; termination still sees only RGB-D.
        for _ in range(2):
            env.d.qpos[16:19] = [3.0, 3.0, 0.02]
            env.d.qvel[15:21] = 0.0
            mj.mj_forward(env.m, env.d)
            count = session.episode
            for frame in range(15):
                if session.advance(wheel_override=np.zeros(3)):
                    break
            else:
                raise AssertionError("Lost ball did not trigger automatic restart")
            assert session.episode == count + 1 and frame + 1 == 12
            assert "D405 lost ball" in session.last_restart_reason
            np.testing.assert_allclose(env.d.qpos, initial_qpos)
            np.testing.assert_allclose(env.d.qvel, initial_qvel)
            np.testing.assert_allclose(session.observation, initial_obs)
            np.testing.assert_array_equal(task.history, 0.0)
            np.testing.assert_array_equal(np.asarray(task.delay), 0.0)
            np.testing.assert_array_equal(task.tracker.velocity, 0.0)
            np.testing.assert_array_equal(env.d.xfrc_applied, 0.0)
            np.testing.assert_array_equal(task.target, HOME)
            assert task.t == 0 and task.tracker.age == 0.0 and not task.failure_reason
            assert not session.advance(wheel_override=np.zeros(3))
        # Continue fresh inference; this must not loop on a stale terminal state.
        for _ in range(60):
            assert not session.advance(wheel_override=np.zeros(3))
        print(
            "PASS: brief occlusion retained; two actual ball escapes each restart after 12 frames; "
            "history, velocity, targets, forces and observations reset; fresh inference continues."
        )
    finally:
        task.close()
