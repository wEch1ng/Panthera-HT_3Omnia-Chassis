import numpy as np
from panthera_lab.tasks.pingpong.arm_recovery import SingleArmTask
from panthera_lab.tasks.pingpong.residual import Task


def test_arm_policy_uses_camera_and_arm_encoders_only():
    task = SingleArmTask()
    try:
        task.reset(81000)
        point = task.env.observe()
        assert point is not None
        feedback = task.env.motor_feedback()
        task.env.observe = lambda: point.copy()
        task.env.motor_feedback = lambda: feedback
        task.tracker.reset()
        before = task.observe().copy()
        task.env.d.qpos[:3] += [2, 3, 1]
        task.env.d.qpos[task.env.ball_qadr : task.env.ball_qadr + 3] += [1, 2, 3]
        task.tracker.reset()
        after = task.observe()
        np.testing.assert_array_equal(before, after)
    finally:
        task.close()


def test_residual_task_accepts_sensor_observations_and_motor_feedback():
    task = Task()
    try:
        obs = task.reset(97001)
        assert obs.shape == (42,)
        obs, _, _, _ = task.step(np.zeros(9))
        assert np.isfinite(obs).all()
    finally:
        task.close()
