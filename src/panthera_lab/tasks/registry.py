"""Explicit task factories; users can register a task without changing the viewer."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any
import torch
from panthera_lab.resources import resource
from panthera_lab.learning.ppo import ActorCritic


@dataclass
class TaskBundle:
    task: Any
    policy: Any = None
    iteration: int = 0
    arm_only: bool = True


_FACTORIES = {}


def register_task(name, factory):
    if name in _FACTORIES:
        raise ValueError(f"Task already registered: {name}")
    _FACTORIES[name] = factory


def arm(*, force_controls=False, checkpoint=None):
    from .pingpong.arm_recovery import SingleArmTask

    task = SingleArmTask(force_controls, checkpoint)
    return TaskBundle(task, iteration=task.iteration)


def residual(*, force_controls=False, checkpoint=None):
    from .pingpong.residual import Task

    checkpoint = checkpoint or resource("checkpoints", "pingpong", "residual_ppo.pt")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy = ActorCritic(42, 9, hidden=64)
    policy.load_state_dict(ck["model"])
    policy.eval()
    return TaskBundle(Task(force_controls), policy, int(ck["iteration"]), False)


register_task("pingpong-arm", arm)
register_task("pingpong-residual", residual)


def load_task(name, **kwargs):
    # module:function plugins are explicit trusted local Python code, not downloaded.
    if ":" in name:
        module, function = name.split(":", 1)
        factory = getattr(import_module(module), function)
    else:
        try:
            factory = _FACTORIES[name]
        except KeyError:
            raise ValueError(f"Unknown task {name}; available: {', '.join(_FACTORIES)}") from None
    return factory(**kwargs)
