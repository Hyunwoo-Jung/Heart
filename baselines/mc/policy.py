"""Load an MC checkpoint into the common HEART SeatPolicy interface."""

from __future__ import annotations

import jax
import numpy as np
import torch

from heart.classic import observe_classic

from .model import QNetwork, engine_action, features, legal_mask, select_actions

CHECKPOINT_FORMAT = "heart-mc-q-v1"


def load_network(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("expected a heart-mc-q-v1 checkpoint")
    network = QNetwork(**checkpoint["network"]).to(device)
    network.load_state_dict(checkpoint["model"])
    network.eval()
    return network, checkpoint


def make_policy(checkpoint_path: str, device: str = "cpu"):
    """Return choose(state, player, key) -> engine action, with no hidden input."""
    network, _ = load_network(checkpoint_path, device)
    observe = jax.jit(observe_classic)
    rng = np.random.default_rng(0)

    def choose(state, player: int, key) -> int:
        del key
        obs = jax.device_get(observe(state, player))
        action = select_actions(
            network,
            features(obs)[None],
            legal_mask(obs)[None],
            device,
            rng,
        )[0]
        return engine_action(obs, int(action))

    return choose
