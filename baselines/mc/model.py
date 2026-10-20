"""Shared private-observation encoding and masked Q action selection."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from heart import NUM_PASS_ACTIONS, PASS, PLAY

NUM_ACTIONS = NUM_PASS_ACTIONS + 52
RETURN_SCALE = 100.0
CARD_ONE_HOT = np.eye(53, dtype=np.float32)
PLAYER_ONE_HOT = np.eye(5, dtype=np.float32)


def features(obs) -> np.ndarray:
    """Encode ClassicObservation; preserve public trick order and winners."""
    game = obs.game
    passed = np.zeros(52, dtype=np.float32)
    received = np.zeros(52, dtype=np.float32)
    for plane, cards in ((passed, obs.cards_passed), (received, obs.cards_received)):
        cards = np.asarray(cards, dtype=np.int64)
        plane[cards[cards >= 0]] = 1
    return np.concatenate(
        [
            np.asarray(game.hand, dtype=np.float32),
            CARD_ONE_HOT[np.asarray(game.trick_history, dtype=np.int64) + 1].ravel(),
            CARD_ONE_HOT[np.asarray(game.current_trick, dtype=np.int64) + 1].ravel(),
            PLAYER_ONE_HOT[np.asarray(game.trick_winners, dtype=np.int64) + 1].ravel(),
            passed,
            received,
            np.asarray(obs.match_scores, dtype=np.float32) / 100,
            np.asarray(game.penalties, dtype=np.float32) / 26,
            np.asarray(game.hand_sizes, dtype=np.float32) / 13,
            PLAYER_ONE_HOT[int(game.player) + 1, 1:],
            PLAYER_ONE_HOT[int(game.leader) + 1, 1:],
            np.eye(4, dtype=np.float32)[int(obs.pass_direction)],
            np.asarray(
                [
                    int(obs.phase) == PASS,
                    int(obs.phase) == PLAY,
                    float(game.trick_index) / 13,
                    float(game.trick_position) / 4,
                    float(game.hearts_broken),
                    float(obs.deal_index) / 16,
                ],
                dtype=np.float32,
            ),
        ]
    )


def legal_mask(obs) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs.pass_action_mask), np.asarray(obs.play_action_mask)]
    ).astype(np.bool_)


def engine_action(obs, action: int) -> int:
    if not 0 <= action < NUM_ACTIONS or not legal_mask(obs)[action]:
        raise ValueError(f"illegal MC action {action}")
    return action if int(obs.phase) == PASS else action - NUM_PASS_ACTIONS


def learner_action(obs, action: int) -> int:
    """Translate the global card ID to the adapter's stable hand slot."""
    card = engine_action(obs.match, action)
    if int(obs.match.phase) == PASS:
        return card
    slots = np.flatnonzero(np.asarray(obs.play_hand_cards) == card)
    if len(slots) != 1 or not bool(obs.play_action_mask[slots[0]]):
        raise ValueError(f"card {card} has no legal learner slot")
    return int(slots[0])


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_ACTIONS),
        )

    def forward(self, x):
        return self.layers(x)


@torch.inference_mode()
def select_actions(network, x, masks, device, rng, epsilon=0.0):
    if not np.all(masks.any(axis=1)):
        raise ValueError("cannot act on an observation without legal actions")
    q = network(torch.as_tensor(x, dtype=torch.float32, device=device))
    if not bool(torch.isfinite(q).all()):
        raise FloatingPointError("non-finite Q prediction")
    valid = torch.as_tensor(masks, dtype=torch.bool, device=device)
    actions = q.masked_fill(~valid, -torch.inf).argmax(dim=1).cpu().numpy()
    if epsilon > 0:
        for i in np.flatnonzero(rng.random(len(actions)) < epsilon):
            actions[i] = rng.choice(np.flatnonzero(masks[i]))
    return actions
