"""Run full matches with the GPU ISMCTS policy, logging search settings."""

import argparse
import json

import jax

from baselines.evaluate import run_matches

from .policy import make_policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exploration", type=float, default=2**0.5)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--opponents", nargs=3, default=["medium"] * 3)
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2000000)
    args = parser.parse_args(argv)
    policy = make_policy(
        args.iterations,
        batch_size=args.batch_size,
        exploration=args.exploration,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "event": "start",
                "config": vars(args),
                "jax": jax.__version__,
                "simulator_device": str(policy.search.simulator.device),
                "rollout": "medium",
                "horizon": "deal",
                "reward": "-effective_penalty/26",
            }
        ),
        flush=True,
    )
    run_matches(
        policy,
        args.opponents,
        args.games,
        args.seed,
        report=lambda line: print(line, flush=True),
    )


if __name__ == "__main__":
    main()
