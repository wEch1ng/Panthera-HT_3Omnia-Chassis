"""PPO entry point; run outputs never overwrite bundled demonstration weights."""

import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from panthera_lab.tasks.pingpong.residual import Task
from panthera_lab.learning.ppo import ActorCritic, compute_gae, ppo_update

torch.set_num_threads(1)


def worker(conn):
    task = Task()
    try:
        while True:
            op, arg = conn.recv()
            if op == "reset":
                conn.send(task.reset(arg))
            elif op == "step":
                obs, r, done, fail = task.step(arg)
                terminal = obs.copy()
                if done:
                    task.episode += 1
                    obs = task.reset(50000 + task.episode * 31 + int(task.rng.integers(10000)))
                conn.send((obs, r, done, fail, terminal))
            else:
                break
    finally:
        task.close()
        conn.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("outputs/runs/residual"))
    args = ap.parse_args(argv)
    from multiprocessing import get_context

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.workers < 1 or args.steps < 8 or args.iterations < 1 or (args.steps * args.workers) % 4:
        ap.error("workers/iterations >= 1, steps >= 8, and workers*steps divisible by 4 required")
    if (out / "last.pt").exists() and not args.resume:
        ap.error("Run already exists: use --resume or a different --output")
    torch.manual_seed(73)
    np.random.seed(73)
    ac = ActorCritic(42, 9, hidden=64, init_std=0.18)
    opt = torch.optim.Adam(ac.parameters(), lr=5e-5)
    start = 0
    if args.resume:
        ck = torch.load(out / "last.pt", weights_only=True)
        ac.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["iteration"]
    else:
        torch.save(dict(model=ac.state_dict(), iteration=0), out / "initial.pt")
    manifest = dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        observation="RGBD relative ball estimate and velocity, visibility/age, 9 motor position/velocity/torque, previous 9 actions",
        action="3 wheel speed residuals and 6 joint position residuals",
        algorithm="reference PPO / sensor-only actor and critic / sensor-only reward",
        dt=1 / 30,
        delay_frames=2,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ctx = get_context("spawn")
    pipes = []
    processes = []
    for i in range(args.workers):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=worker, args=(child,))
        p.start()
        child.close()
        pipes.append(parent)
        processes.append(p)
        parent.send(("reset", 700 + i))
    obs = np.array([p.recv() for p in pipes])
    begin = time.time()
    try:
        for iteration in range(start, start + args.iterations):
            buf = {k: [] for k in ["obs", "act", "logp", "rew", "val", "next", "done"]}
            failures = 0
            for t in range(args.steps):
                with torch.no_grad():
                    act, logp, val = ac.act(torch.tensor(obs))
                for p, a in zip(pipes, act.numpy()):
                    p.send(("step", a))
                results = [p.recv() for p in pipes]
                nextobs, reward, done, fail, terminal = map(np.array, zip(*results))
                failures += sum(fail)
                with torch.no_grad():
                    nv = ac.value(torch.tensor(terminal)).numpy()
                    nv[fail] = 0
                for k, v in zip(
                    buf, [obs, act.numpy(), logp.numpy(), reward, val.numpy(), nv, done]
                ):
                    buf[k].append(v)
                obs = nextobs
            b = {k: np.array(v) for k, v in buf.items()}
            adv, ret = compute_gae(b["rew"], b["val"], b["next"], b["done"])
            batch = {
                k: torch.tensor(v.reshape((-1,) + v.shape[2:]), dtype=torch.float32)
                for k, v in dict(
                    obs=b["obs"], act=b["act"], logp=b["logp"], adv=adv, ret=ret
                ).items()
            }
            metrics = ppo_update(ac, opt, batch, epochs=4, minibatches=4, ent_coef=0.001)
            row = dict(
                iteration=iteration + 1,
                mean_reward=float(b["rew"].mean()),
                failures=int(failures),
                elapsed_s=time.time() - begin,
                **metrics,
            )
            with (out / "log.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            torch.save(
                dict(
                    model=ac.state_dict(),
                    optimizer=opt.state_dict(),
                    iteration=iteration + 1,
                    obs_dim=42,
                    act_dim=9,
                ),
                out / "last.pt",
            )
            print(json.dumps(row), flush=True)
    finally:
        for p in pipes:
            try:
                p.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for p in processes:
            p.join(20)
            if p.is_alive():
                p.terminate()
                p.join(5)


if __name__ == "__main__":
    main()
