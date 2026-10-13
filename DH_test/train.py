"""Self-play Monte Carlo Q learning for HEART, following Mortal's recipe.

Each rollout records legal-action-masked decisions, then labels them with the
remaining match return and final rank. The learner combines Q regression,
conservative Q regularization, and an auxiliary rank prediction objective.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import jax
import numpy as np
import torch
from torch import nn

import heart
from heart.classic import PASS
from DH_test.model import HeartQNetwork


def observation_features(obs) -> np.ndarray:
    """Encode only seat-visible information as (channels, 52 card slots)."""
    game = obs.game
    features = np.zeros((29, 52), dtype=np.float32)
    features[0] = np.asarray(game.hand, dtype=np.float32)
    history = np.asarray(game.trick_history).reshape(-1)
    history = history[history >= 0]
    features[1, history.astype(np.int64)] = 1.0
    trick = np.asarray(game.current_trick)
    trick = trick[trick >= 0]
    features[2, trick.astype(np.int64)] = 1.0
    features[3, np.asarray(obs.cards_passed)[np.asarray(obs.cards_passed) >= 0]] = 1.0
    features[4, np.asarray(obs.cards_received)[np.asarray(obs.cards_received) >= 0]] = 1.0

    scalars = [
        *((np.asarray(obs.match_scores, dtype=np.float32) / 100.0).tolist()),
        *((np.asarray(game.hand_sizes, dtype=np.float32) / 13.0).tolist()),
        *((np.eye(4, dtype=np.float32)[int(game.player)]).tolist()),
        *((np.eye(4, dtype=np.float32)[int(game.leader)]).tolist()),
        float(game.trick_index) / 12.0,
        float(game.trick_position) / 3.0,
        float(game.hearts_broken),
        float(obs.phase == PASS),
        *((np.eye(4, dtype=np.float32)[int(obs.pass_direction)]).tolist()),
    ]
    for channel, value in enumerate(scalars, start=5):
        features[channel, :] = value
    return features


def legal_mask(obs) -> np.ndarray:
    mask = np.zeros(338, dtype=np.bool_)
    if int(obs.phase) == PASS:
        mask[:286] = np.asarray(obs.pass_action_mask, dtype=np.bool_)
    else:
        mask[286:] = np.asarray(obs.play_action_mask, dtype=np.bool_)
    return mask


def action_to_engine(obs, action_id: int) -> int:
    if int(obs.phase) == PASS:
        return action_id
    return action_id - 286


def collect_match(reset, observe, step, key, network, device, replay, rule_mix: float, epsilon: float):
    network.eval()
    state, _ = reset(key)
    records = []
    rewards = []
    while not bool(state.terminated):
        actor = int(state.active_player)
        obs = observe(state, actor)
        features = observation_features(obs)
        mask = legal_mask(obs)
        valid_ids = np.flatnonzero(mask)
        if random.random() < rule_mix:
            key, policy_key = jax.random.split(key)
            if int(obs.phase) == PASS:
                rule_action = heart.make_rule_pass_policy("hard")(obs, policy_key)
            else:
                rule_action = heart.make_rule_policy("hard")(obs.game, policy_key)
            action = int(rule_action)
            action_id = action if int(obs.phase) == PASS else 286 + action
        elif random.random() < epsilon:
            action_id = int(random.choice(valid_ids.tolist()))
            action = action_to_engine(obs, action_id)
        else:
            with torch.inference_mode():
                q_values, _ = network(torch.as_tensor(features[None], device=device))
                q_values = q_values[0].float().cpu().numpy()
            action_id = int(valid_ids[np.argmax(q_values[valid_ids])])
            action = action_to_engine(obs, action_id)

        records.append((features, mask, action_id, actor))
        state, _, reward, _, _ = step(state, action)
        rewards.append(np.asarray(reward, dtype=np.float32))

    final_scores = np.asarray(state.match_scores, dtype=np.int32)
    # Hearts ranks the lowest cumulative penalty score first.
    ranks = final_scores.argsort(kind="stable").argsort(kind="stable")
    returns = np.zeros(4, dtype=np.float32)
    for (features, mask, action_id, actor), reward in zip(reversed(records), reversed(rewards)):
        returns += reward
        replay.append((features, mask, action_id, returns[actor].copy(), int(ranks[actor])))
    return final_scores


def train_step(network, optimizer, batch, device, cql_weight, rank_weight):
    network.train()
    features = torch.as_tensor(np.stack([row[0] for row in batch]), device=device)
    masks = torch.as_tensor(np.stack([row[1] for row in batch]), device=device)
    actions = torch.as_tensor([row[2] for row in batch], dtype=torch.long, device=device)
    targets = torch.as_tensor([row[3] for row in batch], dtype=torch.float32, device=device)
    ranks = torch.as_tensor([row[4] for row in batch], dtype=torch.long, device=device)

    q_all, rank_logits = network(features)
    legal_q = q_all.masked_fill(~masks, -1e9)
    q_taken = q_all.gather(1, actions[:, None]).squeeze(1)
    q_loss = 0.5 * nn.functional.mse_loss(q_taken, targets)
    cql_loss = (torch.logsumexp(legal_q, dim=-1) - q_taken).mean()
    rank_loss = nn.functional.cross_entropy(rank_logits, ranks)
    loss = q_loss + cql_weight * cql_loss + rank_weight * rank_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(network.parameters(), 5.0)
    optimizer.step()
    return {"loss": float(loss.detach()), "q": float(q_loss.detach()), "cql": float(cql_loss.detach()), "rank": float(rank_loss.detach())}


def evaluate(reset, observe, step, network, device, seed: int, games: int) -> dict[str, float]:
    """Play seat 0 against three hard rule policies on deterministic seeds."""
    network.eval()
    key = jax.random.key(seed)
    pass_policy = heart.make_rule_pass_policy("hard")
    play_policy = heart.make_rule_policy("hard")
    ranks, scores = [], []
    for _ in range(games):
        key, game_key = jax.random.split(key)
        state, _ = reset(game_key)
        while not bool(state.terminated):
            actor = int(state.active_player)
            obs = observe(state, actor)
            key, action_key = jax.random.split(key)
            if actor == 0:
                features = observation_features(obs)
                mask = legal_mask(obs)
                valid_ids = np.flatnonzero(mask)
                with torch.inference_mode():
                    q_values, _ = network(torch.as_tensor(features[None], device=device))
                    q_values = q_values[0].float().cpu().numpy()
                action_id = int(valid_ids[np.argmax(q_values[valid_ids])])
                action = action_to_engine(obs, action_id)
            elif int(obs.phase) == PASS:
                action = int(pass_policy(obs, action_key))
            else:
                action = int(play_policy(obs.game, action_key))
            state, _, _, _, _ = step(state, action)
        final_scores = np.asarray(state.match_scores, dtype=np.int32)
        final_ranks = final_scores.argsort(kind="stable").argsort(kind="stable")
        ranks.append(int(final_ranks[0]))
        scores.append(int(final_scores[0]))
    return {
        "avg_rank": float(np.mean(ranks)),
        "avg_score": float(np.mean(scores)),
        "first_rate": float(np.mean(np.asarray(ranks) == 0)),
        "last_rate": float(np.mean(np.asarray(ranks) == 3)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=250_000)
    parser.add_argument("--warmup-matches", type=int, default=100)
    parser.add_argument("--updates-per-match", type=int, default=100)
    parser.add_argument("--rule-mix", type=float, default=0.75)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cql-weight", type=float, default=0.1)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # 5 card planes + 24 global context planes.
    network = HeartQNetwork(input_channels=29).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.steps, 1), eta_min=args.lr * 0.1)
    replay = deque(maxlen=args.replay_capacity)
    env = heart.make("classic-v0")
    reset = jax.jit(env.reset)
    observe = jax.jit(env.observe)
    step = jax.jit(env.step)
    key = jax.random.key(args.seed)
    args.output = Path(__file__).parent / "checkpoints"
    args.output.mkdir(parents=True, exist_ok=True)
    total_updates = 0
    updates_this_run = 0
    matches = 0
    best_rank = float("inf")
    checkpoint_path = args.output / "best.pt"
    evaluation_seed = args.eval_seed if args.eval_seed is not None else args.seed + 876_543
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        network.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        total_updates = int(checkpoint.get("updates", 0))
        matches = int(checkpoint.get("matches", 0))
        best_rank = float(checkpoint["evaluation"]["avg_rank"])
        evaluation_seed = int(checkpoint.get("evaluation_seed", evaluation_seed))
        # A new invocation gets a fresh cosine segment while retaining AdamW
        # moments and the best model from the previous invocation.
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.steps, 1), eta_min=args.lr * 0.1
        )
        print(json.dumps({"loaded_best": str(checkpoint_path), "updates": total_updates, "best_avg_rank": best_rank, "evaluation_seed": evaluation_seed}))
    else:
        # Establish a comparable baseline so even a short first run has a
        # valid best checkpoint to continue from next time.
        evaluation = evaluate(reset, observe, step, network, device, evaluation_seed, args.eval_games)
        best_rank = evaluation["avg_rank"]
        torch.save({
            "model": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "updates": total_updates,
            "matches": matches,
            "evaluation": evaluation,
            "evaluation_seed": evaluation_seed,
        }, checkpoint_path)
        print(json.dumps({"initial_best": str(checkpoint_path), **evaluation}))
    while updates_this_run < args.steps:
        key, match_key = jax.random.split(key)
        scores = collect_match(reset, observe, step, match_key, network, device, replay, args.rule_mix, args.epsilon)
        matches += 1
        if matches < args.warmup_matches or len(replay) < args.batch_size:
            continue
        for _ in range(args.updates_per_match):
            stats = train_step(network, optimizer, random.sample(replay, args.batch_size), device, args.cql_weight, args.rank_weight)
            scheduler.step()
            updates_this_run += 1
            total_updates += 1
            if total_updates % 100 == 0:
                print(json.dumps({"updates": total_updates, "run_updates": updates_this_run, "matches": matches, "replay": len(replay), "scores": scores.tolist(), **stats}))
            if updates_this_run % args.eval_every == 0:
                evaluation = evaluate(reset, observe, step, network, device, evaluation_seed, args.eval_games)
                print(json.dumps({"evaluation": total_updates, "run_updates": updates_this_run, **evaluation}))
                if evaluation["avg_rank"] < best_rank:
                    best_rank = evaluation["avg_rank"]
                    torch.save({
                        "model": network.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "updates": total_updates,
                        "matches": matches,
                        "evaluation": evaluation,
                        "evaluation_seed": evaluation_seed,
                    }, checkpoint_path)
                    print(json.dumps({"saved_new_best": str(checkpoint_path), "updates": total_updates}))
            if updates_this_run >= args.steps:
                break


if __name__ == "__main__":
    main()
