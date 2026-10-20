"""Evaluate an MC checkpoint against balanced HEART SeatPolicy lineups."""

import argparse

import torch

from baselines.evaluate import run_matches

from .policy import make_policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--opponents", nargs=3, default=["medium"] * 3)
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2000000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    policy = make_policy(args.checkpoint, args.device)
    run_matches(
        policy,
        args.opponents,
        args.games,
        args.seed,
        report=lambda line: print(line, flush=True),
    )


if __name__ == "__main__":
    main()
