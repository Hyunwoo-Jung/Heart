"""Evaluate an MC checkpoint against three arbitrary HEART SeatPolicy factories."""

from __future__ import annotations

import argparse
import json

import jax
import numpy as np
import torch

import heart
from heart.play import load_seat_policy

from .policy import make_policy
from .train import score_metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--opponents", nargs=3, default=["medium"] * 3)
    parser.add_argument(
        "--games", type=int, default=100, help="multiple of 4, rotating seats"
    )
    parser.add_argument("--seed", type=int, default=2000000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 4:
        parser.error("--games must be a positive multiple of 4")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    torch.set_num_threads(1)
    learner = make_policy(args.checkpoint, args.device)
    opponents = [load_seat_policy(spec) for spec in args.opponents]
    env = heart.make()
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    scores, players, deals = [], [], []
    for game in range(args.games):
        player = game % 4
        # Replay a shared deal seed across all four learner seats.
        key = jax.random.fold_in(jax.random.key(args.seed), game // 4)
        deal_key, policy_key = jax.random.split(key)
        state, _ = reset(deal_key)
        policies = opponents.copy()
        policies.insert(player, learner)
        for _ in range(heart.MAX_CLASSIC_CORE_STEPS):
            policy_key, action_key = jax.random.split(policy_key)
            actor = int(state.active_player)
            action = policies[actor](state, actor, action_key)
            state, _, _, terminated, info = step(state, action)
            if bool(info.invalid_action):
                raise RuntimeError(f"seat {actor} returned invalid action {action}")
            if bool(terminated):
                break
        else:
            raise RuntimeError("match exceeded classic-v0 event bound")
        scores.append(np.asarray(state.match_scores))
        players.append(player)
        deals.append(int(state.deal_index) + 1)
        print(
            json.dumps(
                {
                    "event": "match",
                    "game": game + 1,
                    "learner_seat": player,
                    "scores": scores[-1].tolist(),
                    "deals": deals[-1],
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "event": "summary",
                "checkpoint": args.checkpoint,
                "opponents": args.opponents,
                "seed": args.seed,
                **score_metrics(scores, players, deals),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
