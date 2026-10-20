"""Fresh-trajectory, episodic MC Q regression; no TD targets or replay."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from functools import cache
from pathlib import Path

import jax
import numpy as np
import torch

import heart

from .model import (
    RETURN_SCALE,
    QNetwork,
    features,
    learner_action,
    legal_mask,
    select_actions,
)
from .policy import CHECKPOINT_FORMAT, load_network


def mc_returns(rewards):
    """Undiscounted returns through match termination, including this reward."""
    return np.cumsum(np.asarray(rewards, dtype=np.float32)[::-1])[::-1].copy()


def score_metrics(scores, players, deals):
    scores = np.asarray(scores)
    players = np.asarray(players)
    own = scores[np.arange(len(scores)), players]
    lower = (scores < own[:, None]).sum(axis=1)
    tied = (scores == own[:, None]).sum(axis=1)
    return {
        "matches": len(scores),
        "mean_penalty": float(own.mean()),
        "penalty_per_deal": float(own.sum() / np.asarray(deals).sum()),
        "mean_rank": float((1 + lower + (tied - 1) / 2).mean()),
        "joint_win_rate": float((lower == 0).mean()),
        "mean_deals": float(np.mean(deals)),
    }


@cache
def environment(player):
    env = heart.make_single_agent(controlled_player=player, opponents="medium")
    return jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))


def collect(network, device, rng, num_envs, player, epsilon, *, training=True):
    """Freeze the policy until every environment in this batch finishes."""
    reset, step = environment(player)
    key = jax.random.key(int(rng.integers(0, 2**32)))
    state, obs = reset(jax.random.split(key, num_envs))
    done = np.zeros(num_envs, dtype=np.bool_)
    trajectories = [[] for _ in range(num_envs)]
    total_reward = np.zeros(num_envs, dtype=np.float32)
    network.eval()
    for _ in range(heart.MAX_CLASSIC_DEALS * 14):
        active = np.flatnonzero(~done)
        host_obs = jax.device_get(obs)
        rows = [jax.tree.map(lambda x, index=i: x[index], host_obs) for i in active]
        x = np.stack([features(row.match) for row in rows])
        masks = np.stack([legal_mask(row.match) for row in rows])
        selected = select_actions(network, x, masks, device, rng, epsilon)
        actions = np.zeros(num_envs, dtype=np.int32)
        for i, row, action in zip(active, rows, selected):
            actions[i] = learner_action(row, int(action))
        # Finished rows stay terminal; only live rows contribute actions/rewards.
        state, obs, rewards, terminated, info = step(state, actions)
        rewards, terminated, invalid = jax.device_get(
            (rewards, terminated, info.core.invalid_action)
        )
        if np.any(invalid[active]):
            raise RuntimeError("environment rejected a masked learner action")
        total_reward[active] += rewards[active]
        if training:
            for row_index, i in enumerate(active):
                trajectories[i].append(
                    (x[row_index], int(selected[row_index]), float(rewards[i]))
                )
        done = np.asarray(terminated)
        if done.all():
            break
    else:
        raise RuntimeError(
            "match exceeded the classic-v0 bound; no partial targets used"
        )
    scores, deal_indices = jax.device_get(
        (state.match.match_scores, state.match.deal_index)
    )
    np.testing.assert_allclose(total_reward, -scores[:, player], atol=1e-5)
    samples = []
    for trajectory in trajectories:
        returns = mc_returns([row[2] for row in trajectory])
        samples.extend(
            (row[0], row[1], target / RETURN_SCALE)
            for row, target in zip(trajectory, returns)
        )
    return samples, np.asarray(scores), np.asarray(deal_indices) + 1


def update(network, optimizer, samples, batch_size, device, rng):
    """One shuffled regression epoch over newly completed trajectories."""
    network.train()
    losses = []
    for indices in np.array_split(
        rng.permutation(len(samples)), (len(samples) + batch_size - 1) // batch_size
    ):
        rows = [samples[i] for i in indices]
        x = torch.as_tensor(np.stack([r[0] for r in rows]), device=device)
        actions = torch.as_tensor([r[1] for r in rows], device=device)
        targets = torch.as_tensor(
            np.asarray([r[2] for r in rows], dtype=np.float32), device=device
        )
        prediction = network(x).gather(1, actions[:, None]).squeeze(1)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite MC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            network.parameters(), max_norm=5, error_if_nonfinite=True
        )
        optimizer.step()
        losses.append((float(loss.detach()), len(rows)))
    return sum(loss * n for loss, n in losses) / len(samples)


def evaluate(network, device, seed, num_envs):
    scores, players, deals = [], [], []
    for player in range(4):
        rng = np.random.default_rng(seed)
        _, s, d = collect(network, device, rng, num_envs, player, 0.0, training=False)
        scores.extend(s)
        players.extend([player] * num_envs)
        deals.extend(d)
    return score_metrics(scores, players, deals)


def save_checkpoint(path, network, optimizer, rng, iteration, best, args):
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "network": {"input_dim": network.input_dim, "hidden": network.hidden},
        "model": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": json.dumps(rng.bit_generator.state),
        "torch_rng": torch.get_rng_state(),
        "iteration": iteration,
        "best_mean_penalty": best,
        "config": vars(args) | {"output": str(args.output), "resume": str(args.resume)},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations", type=int, default=1000, help="total collection iterations"
    )
    parser.add_argument(
        "--num-envs", type=int, default=16, help="complete matches per iteration"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=1000000)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    for name in ("iterations", "num_envs", "batch_size", "hidden", "eval_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite")
    if not 0 <= args.epsilon <= 1:
        parser.error("--epsilon must be in [0, 1]")
    if not 0 <= args.seed < 2**32 or not 0 <= args.eval_seed < 2**32:
        parser.error("seeds must be in [0, 2**32)")
    if args.seed == args.eval_seed:
        parser.error("training and evaluation seeds must differ")
    if args.output is None:
        args.output = (
            args.resume.parent
            if args.resume
            else Path(".local_training/mc")
            / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )
    if args.resume and args.output.resolve() != args.resume.parent.resolve():
        parser.error("--resume must keep the checkpoint's output directory")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the GPU/PyTorch installation")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start, best = 0, float("inf")
    if args.resume:
        network, checkpoint = load_network(args.resume, device)
        for name in (
            "seed",
            "eval_seed",
            "num_envs",
            "hidden",
            "lr",
            "epsilon",
            "batch_size",
        ):
            if checkpoint["config"][name] != getattr(args, name):
                raise ValueError(f"resume must retain --{name.replace('_', '-')}")
        rng.bit_generator.state = json.loads(checkpoint["rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        start = checkpoint["iteration"]
        best = checkpoint["best_mean_penalty"]
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(f"output is not empty: {args.output}; use --resume")
        sample_env = heart.make_single_agent()
        _, obs = sample_env.reset(jax.random.key(0))
        network = QNetwork(len(features(jax.device_get(obs.match))), args.hidden).to(
            device
        )
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if start >= args.iterations:
        raise ValueError("--iterations must exceed the resumed iteration")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.jsonl").open("a") as log:

        def report(event, **values):
            line = json.dumps({"event": event, **values}, allow_nan=False)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        report(
            "start",
            config=vars(args)
            | {"output": str(args.output), "resume": str(args.resume)},
            torch=torch.__version__,
            jax=jax.__version__,
            resumed_iteration=start,
        )
        if not args.resume:
            metrics = evaluate(network, device, args.eval_seed, args.num_envs)
            best = metrics["mean_penalty"]
            report("evaluation", iteration=0, **metrics)
            save_checkpoint(
                args.output / "best.pt", network, optimizer, rng, 0, best, args
            )
            save_checkpoint(
                args.output / "last.pt", network, optimizer, rng, 0, best, args
            )
        for iteration in range(start + 1, args.iterations + 1):
            began = time.monotonic()
            player = (iteration - 1) % 4
            samples, scores, deals = collect(
                network, device, rng, args.num_envs, player, args.epsilon
            )
            loss = update(network, optimizer, samples, args.batch_size, device, rng)
            report(
                "train",
                iteration=iteration,
                total_matches=iteration * args.num_envs,
                learner_seat=player,
                decisions=len(samples),
                loss=loss,
                seconds=time.monotonic() - began,
                **score_metrics(scores, [player] * args.num_envs, deals),
            )
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                metrics = evaluate(network, device, args.eval_seed, args.num_envs)
                report("evaluation", iteration=iteration, **metrics)
                if metrics["mean_penalty"] < best:
                    best = metrics["mean_penalty"]
                    save_checkpoint(
                        args.output / "best.pt",
                        network,
                        optimizer,
                        rng,
                        iteration,
                        best,
                        args,
                    )
            save_checkpoint(
                args.output / "last.pt", network, optimizer, rng, iteration, best, args
            )


if __name__ == "__main__":
    main()
