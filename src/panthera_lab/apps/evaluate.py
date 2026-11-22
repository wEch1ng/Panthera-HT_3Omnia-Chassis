"""Independent evaluation: ground truth stays in metrics, not actor inputs."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from panthera_lab.tasks.registry import load_task


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="pingpong-arm")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--seed", type=int, default=81000)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.seconds < 1:
        parser.error("Use >=1 episode and >=1 second")
    torch.set_num_threads(1)
    bundle = load_task(args.task, checkpoint=args.checkpoint)
    task = bundle.task
    rows = []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            obs = task.reset(seed)
            errors = []
            contacts = []
            failed = False
            for _ in range(round(args.seconds * 30)):
                with torch.no_grad():
                    action = (
                        None
                        if bundle.policy is None
                        else bundle.policy.actor(torch.tensor(obs)).numpy()
                    )
                obs, _, _, failed = task.step(
                    action, wheel_override=np.zeros(3) if bundle.arm_only else None
                )
                env = task.env
                errors.append(float(np.linalg.norm(env.error())))
                contacts.append(any(set(c.geom) == {env.ball, env.blade} for c in env.d.contact))
                dropped = (
                    errors[-1] > 0.085
                    or env.d.geom_xpos[env.ball, 2] < env.d.geom_xpos[env.blade, 2] - 0.02
                )
                if failed or dropped:
                    failed = True
                    break
            rows.append(
                dict(
                    seed=seed,
                    seconds=len(errors) / 30,
                    success=bool(not failed and max(errors[-30:]) < 0.02 and all(contacts[-30:])),
                    mean_error_mm=float(np.mean(errors) * 1000),
                    final_max_mm=max(errors[-30:]) * 1000,
                )
            )
            print(json.dumps(rows[-1]), flush=True)
    finally:
        task.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                task=args.task,
                checkpoint=str(args.checkpoint) if args.checkpoint else "bundled",
                iteration=bundle.iteration,
                results=rows,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"{sum(r['success'] for r in rows)}/{len(rows)} passed; {args.output}")
