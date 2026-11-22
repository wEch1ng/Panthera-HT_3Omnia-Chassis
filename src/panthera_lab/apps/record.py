"""Record sensor-only streams with commands for future image-policy experiments."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from panthera_lab.tasks.registry import load_task


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/recordings/demo.npz"))
    args = parser.parse_args(argv)
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.output.exists():
        parser.error("Output exists; choose another path")
    torch.set_num_threads(1)
    task = load_task("pingpong-arm").task
    samples = {
        k: []
        for k in (
            "time_s",
            "rgb",
            "depth_m",
            "motor_position",
            "motor_velocity",
            "motor_torque",
            "arm_target",
            "wheel_command",
        )
    }
    try:
        task.reset(81000)
        for _ in range(round(args.seconds * 30)):
            packet = task.env.read_sensors()
            # Observation at t, applied low-level command over [t,t+dt].
            _, _, _, failed = task.step(wheel_override=np.zeros(3))
            values = [
                packet.time_s,
                packet.camera.rgb,
                packet.camera.depth,
                packet.motors.position,
                packet.motors.velocity,
                packet.motors.torque,
                task.target.copy(),
                np.zeros(3),
            ]
            for key, value in zip(samples, values):
                samples[key].append(value)
            if failed:
                break
    finally:
        task.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: np.asarray(value) for key, value in samples.items()})
    args.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                policy="arm_recovery",
                hz=30,
                frames=len(samples["time_s"]),
                joint_order="left,right,back,joint1..6",
                contains_sim_truth=False,
                depth_units="optical-axis metres",
            ),
            indent=2,
        )
        + "\n"
    )
    print(args.output)
