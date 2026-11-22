"""Factory example: `panthera view --task examples.custom_task:make_task`."""

from panthera_lab.tasks.registry import TaskBundle
from panthera_lab.tasks.pingpong.arm_recovery import SingleArmTask


def make_task(*, force_controls=False, checkpoint=None):
    task = SingleArmTask(force_controls=force_controls, checkpoint=checkpoint)
    return TaskBundle(task=task, iteration=task.iteration, arm_only=True)
