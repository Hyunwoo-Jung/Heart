"""Balanced full-match evaluation shared by learned and search policies."""

import argparse
import json
import time
from itertools import permutations

import jax
import numpy as np

import heart
from heart.play import load_seat_policy


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


def seating_schedule(opponents):
    if len(opponents) != 3:
        raise ValueError("exactly three opponents are required")
    seats = ((True, ""), *((False, spec) for spec in opponents))
    return sorted(set(permutations(seats)))


def run_matches(policy, opponents, games, seed, *, report=print):
    layouts = seating_schedule(opponents)
    if games <= 0 or games % len(layouts):
        raise ValueError(
            f"games must be a positive multiple of {len(layouts)} for balanced seats"
        )
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    policies = {spec: load_seat_policy(spec) for spec in set(opponents)}
    env = heart.make()
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    scores, players, deals, latencies = [], [], [], []
    with jax.default_device(jax.devices("cpu")[0]):
        for game in range(games):
            layout = layouts[game % len(layouts)]
            player = next(i for i, (focal, _) in enumerate(layout) if focal)
            key = jax.random.fold_in(jax.random.key(seed), game // len(layouts))
            deal_key, policy_key = jax.random.split(key)
            state, _ = reset(deal_key)
            choices = [policy if focal else policies[spec] for focal, spec in layout]
            elapsed = []
            for _ in range(heart.MAX_CLASSIC_CORE_STEPS):
                policy_key, action_key = jax.random.split(policy_key)
                actor = int(state.active_player)
                began = time.perf_counter()
                action = choices[actor](state, actor, action_key)
                if actor == player:
                    elapsed.append(time.perf_counter() - began)
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
            latencies.extend(elapsed)
            report(
                json.dumps(
                    {
                        "event": "match",
                        "game": game + 1,
                        "deal_seed_index": game // len(layouts),
                        "learner_seat": player,
                        "seating": ["focal" if f else s for f, s in layout],
                        "scores": scores[-1].tolist(),
                        "deals": deals[-1],
                        "decision_seconds": elapsed,
                    }
                )
            )
    summary = {
        "event": "summary",
        "opponents": opponents,
        "seed": seed,
        "seatings_per_deal_seed": len(layouts),
        "decisions": len(latencies),
        "mean_decision_seconds": float(np.mean(latencies)),
        "p95_decision_seconds": float(np.percentile(latencies, 95)),
        **score_metrics(scores, players, deals),
    }
    report(json.dumps(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="HEART SeatPolicy factory")
    parser.add_argument("--opponents", nargs=3, default=["medium"] * 3)
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2000000)
    args = parser.parse_args(argv)
    policy = load_seat_policy(args.policy)
    run_matches(
        policy,
        args.opponents,
        args.games,
        args.seed,
        report=lambda line: print(line, flush=True),
    )


if __name__ == "__main__":
    main()
