"""Optional PyTorch baseline checks; also runnable with unittest alone."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest(
        "MC baseline tests require the optional torch dependency"
    ) from error

import jax
import numpy as np

import heart
from baselines.mc.model import (
    QNetwork,
    engine_action,
    features,
    learner_action,
    legal_mask,
    select_actions,
)
from baselines.mc.policy import load_network, make_policy
from baselines.mc.train import (
    collect,
    mc_returns,
    parse_args,
    save_checkpoint,
    score_metrics,
    update,
)


class MonteCarloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = heart.make_single_agent(controlled_player=2, opponents="medium")
        cls.state, cls.obs = jax.jit(cls.env.reset)(jax.random.key(19))
        cls.x = features(jax.device_get(cls.obs.match))

    def test_returns_cross_deal_boundaries_and_include_terminal_reward(self):
        np.testing.assert_array_equal(mc_returns([0, -5, 0, -9]), [-14, -14, -9, -9])
        self.assertEqual(len(mc_returns([])), 0)

    def test_history_order_and_public_penalties_are_preserved(self):
        obs = jax.device_get(self.obs.match)
        history = np.array(obs.game.trick_history)
        history[0] = [0, 1, 2, 3]
        first = obs._replace(game=obs.game._replace(trick_history=history))
        second = obs._replace(game=obs.game._replace(trick_history=history[:, ::-1]))
        self.assertFalse(np.array_equal(features(first), features(second)))
        changed = obs._replace(game=obs.game._replace(penalties=np.array([0, 1, 0, 0])))
        self.assertFalse(np.array_equal(features(obs), features(changed)))
        self.assertEqual(features(obs).dtype, np.float32)

    def test_masking_and_adapter_conversion(self):
        network = QNetwork(len(self.x), hidden=8)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.layers[-1].bias[300] = 10000
        rng = np.random.default_rng(1)
        masks = legal_mask(self.obs.match)[None]
        for epsilon in (0, 1):
            action = int(
                select_actions(network, self.x[None], masks, "cpu", rng, epsilon)[0]
            )
            self.assertTrue(masks[0, action])
            self.assertEqual(learner_action(self.obs, action), action)
        with self.assertRaises(ValueError):
            engine_action(self.obs.match, 300)
        with self.assertRaises(ValueError):
            select_actions(network, self.x[None], np.zeros_like(masks), "cpu", rng)
        _, obs, _, _, info = jax.jit(self.env.step)(self.state, np.int32(0))
        self.assertFalse(bool(info.core.invalid_action))
        self.assertEqual(int(obs.match.phase), heart.PLAY)
        for card in np.flatnonzero(np.asarray(obs.match.play_action_mask)):
            action = heart.NUM_PASS_ACTIONS + int(card)
            slot = learner_action(obs, action)
            self.assertEqual(int(obs.play_hand_cards[slot]), card)
            self.assertTrue(bool(obs.play_action_mask[slot]))
            self.assertEqual(engine_action(obs.match, action), card)

    def test_ties_have_average_ranks_and_joint_winners(self):
        result = score_metrics([[10, 10, 30, 40], [20, 10, 30, 40]], [0, 0], [5, 5])
        self.assertEqual(result["mean_rank"], 1.75)
        self.assertEqual(result["joint_win_rate"], 0.5)
        self.assertEqual(result["mean_penalty"], 15)
        self.assertEqual(result["penalty_per_deal"], 3)

    def test_complete_batched_matches_train_and_checkpoint_roundtrip(self):
        rng = np.random.default_rng(9)
        network = QNetwork(len(self.x), hidden=8)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        samples, scores, deals = collect(network, "cpu", rng, 2, 2, 0.2)
        self.assertTrue(np.all(scores.max(axis=1) >= 100))
        self.assertTrue(np.all(deals > 1))
        self.assertTrue(all(row[2] <= 0 for row in samples))
        first = next(network.parameters()).detach().clone()
        loss = update(network, optimizer, samples, 64, "cpu", rng)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(first, next(network.parameters())))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            args = argparse.Namespace(output=Path(directory), resume=None)
            save_checkpoint(path, network, optimizer, rng, 1, 50.0, args)
            loaded, checkpoint = load_network(path)
            torch.testing.assert_close(
                network(torch.tensor(self.x)), loaded(torch.tensor(self.x))
            )
            restored_rng = np.random.default_rng()
            restored_rng.bit_generator.state = json.loads(checkpoint["rng"])
            np.testing.assert_array_equal(
                rng.integers(100, size=8), restored_rng.integers(100, size=8)
            )
            policy = make_policy(str(path))
            action = policy(self.state.match, 2, jax.random.key(0))
            self.assertTrue(bool(self.obs.match.pass_action_mask[action]))
            self.assertEqual(checkpoint["iteration"], 1)

    def test_argument_validation(self):
        for argv in (["--epsilon", "2"], ["--num-envs", "0"], ["--lr", "nan"]):
            with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
