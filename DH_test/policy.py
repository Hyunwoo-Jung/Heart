"""Load a DH_test checkpoint as a HEART ``SeatPolicy``."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import torch

from heart.classic import observe_classic
from .model import HeartQNetwork
from .train import action_to_engine, legal_mask, observation_features


def make_policy(checkpoint_path: str | Path, device: str = "cpu"):
    """Return a ``SeatPolicy`` callable for ``heart.play`` or ``--seat``."""
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    network = HeartQNetwork(input_channels=29).to(device)
    network.load_state_dict(checkpoint["model"])
    network.eval()

    def choose(state, player: int, key: jax.Array) -> int:
        del key  # The greedy Q policy is deterministic.
        observation = observe_classic(state, player)
        features = observation_features(observation)
        mask = legal_mask(observation)
        valid_ids = np.flatnonzero(mask)
        with torch.inference_mode():
            q_values, _ = network(torch.as_tensor(features[None], device=device))
            q_values = q_values[0].float().cpu().numpy()
        action_id = int(valid_ids[np.argmax(q_values[valid_ids])])
        return action_to_engine(observation, action_id)

    return choose
