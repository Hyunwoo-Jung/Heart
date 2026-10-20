"""Public-information, rule replay, tree accounting, and rollout regressions."""

import json
import unittest
from collections import Counter
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import heart
from baselines.evaluate import run_matches, seating_schedule
from baselines.ismcts.belief import Belief
from baselines.ismcts.policy import make_policy
from baselines.ismcts.search import Edge, Node, Search
from baselines.ismcts.simulator import Simulator


class ISMCTSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = heart.make()
        cls.step = staticmethod(jax.jit(cls.env.step))
        cls.observe = staticmethod(jax.jit(cls.env.observe))
        cls.deal_step = staticmethod(jax.jit(heart.DealEnv().step))
        cls.initial, _ = cls.env.reset(jax.random.key(51))
        state = cls.initial
        cls.pass_states = []
        for _ in range(4):
            cls.pass_states.append(state)
            state, _, _, _, _ = cls.step(state, np.int32(0))
        cls.after_pass = state
        cls.snapshots = []
        rule = heart.make_rule_seat_policy("medium")
        for turn in range(52):
            if turn in (0, 2, 8, 24, 44, 48, 51):
                cls.snapshots.append(state)
            actor = int(state.active_player)
            action = rule(state, actor, jax.random.key(turn))
            state, _, _, _, _ = cls.step(state, action)

    def test_samples_replay_public_history_legally(self):
        template, _ = heart.DealEnv().reset(jax.random.key(0))
        rng = np.random.default_rng(4)
        for state in self.snapshots:
            obs = jax.device_get(self.observe(state, state.active_player))
            belief = Belief(obs)
            for hands in belief.sample(3, rng):
                np.testing.assert_array_equal(hands.sum(axis=1), obs.game.hand_sizes)
                self.assertTrue(np.all(hands.sum(axis=0) <= 1))
                np.testing.assert_array_equal(
                    hands[int(obs.game.player)], obs.game.hand
                )
                original = hands.copy()
                for actor, card in belief.plays:
                    original[actor, card] = True
                np.testing.assert_array_equal(original.sum(axis=1), np.full(4, 13))
                opener = np.int32(np.argmax(original[:, 0]))
                replay = template._replace(
                    hands=jnp.asarray(original), active_player=opener, leader=opener
                )
                for actor, card in belief.plays:
                    self.assertEqual(int(replay.active_player), actor)
                    replay, _, _, _, info = self.deal_step(replay, np.int32(card))
                    self.assertFalse(bool(info.invalid_action))
                np.testing.assert_array_equal(replay.hands, hands)
                np.testing.assert_array_equal(replay.penalties, obs.game.penalties)

    def test_passed_cards_remain_with_known_recipient(self):
        for player in range(4):
            obs = jax.device_get(self.observe(self.after_pass, player))
            samples = Belief(obs).sample(5, np.random.default_rng(player))
            for card in obs.cards_passed:
                self.assertTrue(samples[:, (player + 1) % 4, int(card)].all())

    def test_first_trick_point_discard_forbids_remaining_nonpoints(self):
        hands = np.zeros((4, 52), dtype=bool)
        for player, cards in enumerate(
            (range(13), range(39, 52), range(13, 26), range(26, 39))
        ):
            hands[player, list(cards)] = True
        game, _ = heart.DealEnv().reset(jax.random.key(0))
        game = game._replace(
            hands=jnp.asarray(hands), active_player=np.int32(0), leader=np.int32(0)
        )
        for card in (0, 39):
            game, _, _, _, info = self.deal_step(game, np.int32(card))
            self.assertFalse(bool(info.invalid_action))
        state = self.initial._replace(
            game=game,
            active_player=np.int32(2),
            phase=np.int8(heart.PLAY),
            pass_direction=np.int8(heart.PASS_HOLD),
            deal_index=np.int8(3),
        )
        samples = Belief(jax.device_get(self.observe(state, 2))).sample(
            20, np.random.default_rng(9)
        )
        nonpoints = (np.arange(52) < 39) & (np.arange(52) != 36)
        self.assertFalse(samples[:, 1, nonpoints].any())

    def test_impossible_observation_fails_instead_of_relaxing_constraints(self):
        obs = jax.device_get(self.observe(self.after_pass, 0))
        bad = obs._replace(
            cards_passed=np.array([int(np.flatnonzero(obs.game.hand)[0]), -1, -1])
        )
        with self.assertRaises(ValueError):
            Belief(bad)

    def test_medium_pass_exchange_matches_engine(self):
        sim = Simulator("cpu")
        for direction in (0, 1, 2):
            obs = jax.device_get(
                self.observe(
                    self.initial._replace(pass_direction=np.int8(direction)), 0
                )
            )
            hands = Belief(obs).sample(2, np.random.default_rng(15))
            states = sim.states(obs, hands)
            keys = sim.keys(2, np.random.default_rng(21))
            actions = np.array([5, 17], dtype=np.int32)
            after, received = sim.pass_cards(
                states,
                actions,
                keys,
                np.int32(0),
                np.int32(direction),
                obs.match_scores,
                obs.deal_index,
            )
            policy = jax.jit(heart.make_rule_pass_policy("medium"))
            for i in range(2):
                classic = self.initial._replace(
                    game=jax.tree.map(lambda x, index=i: x[index], states),
                    pass_direction=np.int8(direction),
                )
                for actor in range(4):
                    action = (
                        actions[i]
                        if actor == 0
                        else policy(
                            self.observe(classic, actor),
                            jax.random.fold_in(keys[i], actor),
                        )
                    )
                    classic, _, _, _, info = self.step(classic, action)
                    self.assertFalse(bool(info.invalid_action))
                np.testing.assert_array_equal(after.hands[i], classic.game.hands)
                np.testing.assert_array_equal(received[i], classic.cards_received[0])

    def test_hidden_state_changes_do_not_change_pass_search(self):
        policy = make_policy("12", batch_size=5, device="cpu")
        state = self.pass_states[3]
        hands = np.asarray(state.game.hands).copy()
        a = int(np.flatnonzero(hands[0] & (np.arange(52) != 0))[0])
        b = int(np.flatnonzero(hands[1] & (np.arange(52) != 0))[0])
        hands[0, [a, b]] = [False, True]
        hands[1, [a, b]] = [True, False]
        changed = state._replace(
            game=state.game._replace(hands=jnp.asarray(hands)),
            rng_key=jax.random.key(999),
            pass_cards=state.pass_cards.at[:3].set(-1),
        )
        for left, right in zip(
            jax.tree.leaves(self.observe(state, 3)),
            jax.tree.leaves(self.observe(changed, 3)),
        ):
            np.testing.assert_array_equal(left, right)
        first = policy(state, 3, jax.random.key(3))
        visits = policy.search.last_stats["root_visits"].copy()
        second = policy(changed, 3, jax.random.key(3))
        self.assertEqual(first, second)
        self.assertEqual(visits, policy.search.last_stats["root_visits"])
        self.assertEqual(sum(visits.values()), 12)

    def test_pass_and_play_search_use_exact_budget_and_legal_moves(self):
        search = Search(iterations=1024, batch_size=32, device="cpu")
        for state in (self.initial, self.snapshots[-2]):
            obs = self.observe(state, state.active_player)
            action = search.choose(obs, jax.random.key(7))
            mask = (
                obs.pass_action_mask
                if int(obs.phase) == heart.PASS
                else obs.play_action_mask
            )
            self.assertTrue(bool(mask[action]))
            self.assertEqual(search.last_stats["simulations"], 1024)
            self.assertEqual(sum(search.last_stats["root_visits"].values()), 1024)

    def test_gpu_unavailable_fails_without_cpu_fallback(self):
        with patch(
            "baselines.ismcts.simulator.jax.devices",
            side_effect=RuntimeError("GPU unavailable"),
        ) as devices:
            with self.assertRaisesRegex(RuntimeError, "gpu backend unavailable"):
                Simulator("gpu")
            devices.assert_called_once_with("gpu")

    def test_terminal_rollout_uses_moon_adjusted_score(self):
        sim = Simulator("cpu")
        obs = jax.device_get(self.observe(self.initial, 0))
        states = sim.states(obs, Belief(obs).sample(1, np.random.default_rng(0)))
        states = states._replace(
            terminated=jnp.array([True]),
            scores=jnp.array([[0, 26, 26, 26]], dtype=jnp.int16),
        )
        rewards = sim.rollout(states, sim.keys(1, np.random.default_rng(0)))
        np.testing.assert_array_equal(rewards, [[0, -1, -1, -1]])

    def test_tree_uses_acting_player_value_and_available_actions(self):
        node = Node(
            {
                2: Edge(visits=10, available=10, value=np.array([0, -10, 0, 0])),
                3: Edge(visits=10, available=10, value=np.array([-10, 0, 0, 0])),
            }
        )
        action, edge, expanded = node.select(
            np.array([2, 3]), 1, 0, np.random.default_rng(0)
        )
        self.assertEqual(action, 3)
        self.assertFalse(expanded)
        self.assertEqual(edge.pending, 1)
        before = node.children[3].available
        action, _, _ = node.select(np.array([2]), 1, 0, np.random.default_rng(0))
        self.assertEqual(action, 2)
        self.assertEqual(node.children[3].available, before)

    def test_balanced_opponent_seating_and_complete_arena(self):
        layouts = seating_schedule(["other-policy", "medium", "medium"])
        self.assertEqual(len(layouts), 12)
        pairs = Counter(
            (
                next(i for i, (f, _) in enumerate(layout) if f),
                next(i for i, (_, s) in enumerate(layout) if s == "other-policy"),
            )
            for layout in layouts
        )
        self.assertEqual(set(pairs.values()), {1})
        records = []
        summary = run_matches(
            heart.make_rule_seat_policy("medium"),
            ["medium"] * 3,
            4,
            1234,
            report=lambda line: records.append(json.loads(line)),
        )
        self.assertEqual(summary["matches"], 4)
        self.assertEqual({r["learner_seat"] for r in records[:-1]}, set(range(4)))
        self.assertTrue(all(max(r["scores"]) >= 100 for r in records[:-1]))


if __name__ == "__main__":
    unittest.main()
