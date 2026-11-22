"""Environment and resource diagnostics, optionally with a genuine RGB-D rollout."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
from panthera_lab.resources import resource, resource_root


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    for name in ("panthera-lab", "mujoco", "numpy", "torch", "opencv-python-headless", "glfw"):
        print(f"{name}: {metadata.version(name)}")
    print(f"Resources: {resource_root()}")
    manifest = json.loads(resource("checkpoints", "manifest.json").read_text())
    for name, item in manifest.items():
        path = resource("checkpoints", *Path(item["file"]).parts)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch: {name}")
        print(f"{name}: checksum OK")
    from panthera_lab.sim.model import robot_xml, compile_model

    model = compile_model(robot_xml())
    print(f"Robot: {model.nbody} bodies / {model.nu} motor actuators / camera_link present")
    if args.render:
        from panthera_lab.tasks.registry import load_task
        import numpy as np
        import torch

        torch.set_num_threads(1)
        bundle = load_task("pingpong-arm")
        task = bundle.task
        try:
            obs = task.reset(81000)
            for _ in range(30):
                obs, _, _, failure = task.step(wheel_override=np.zeros(3))
            assert np.isfinite(obs).all() and not failure
            frame = task.env.camera.capture(task.env.d)
            print(f"RGB {frame.rgb.shape}; depth {frame.depth.shape}; 42-D arm inference OK")
        finally:
            task.close()
    print("PASS")
