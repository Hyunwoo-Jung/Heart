"""Batched SO-ISMCTS with availability counts and in-flight virtual losses."""

import time
from dataclasses import dataclass, field

import jax
import numpy as np

import heart

from .belief import Belief
from .simulator import Simulator


@dataclass
class Edge:
    visits: int = 0
    available: int = 0
    pending: int = 0
    value: np.ndarray = field(default_factory=lambda: np.zeros(4))
    outcomes: dict = field(default_factory=dict)


@dataclass
class Node:
    children: dict = field(default_factory=dict)

    def select(self, legal, player, exploration, rng):
        for action in legal:
            if int(action) in self.children:
                self.children[int(action)].available += 1
        untried = [int(a) for a in legal if int(a) not in self.children]
        expanded = bool(untried)
        if expanded:
            action = int(rng.choice(untried))
            self.children[action] = Edge(available=1)
        else:
            values = []
            for action in legal:
                edge = self.children[int(action)]
                visits = edge.visits + edge.pending
                values.append(
                    (edge.value[player] - edge.pending) / visits
                    + exploration * np.sqrt(np.log(max(edge.available, 1)) / visits)
                )
            maxima = np.flatnonzero(np.isclose(values, np.max(values)))
            action = int(legal[rng.choice(maxima)])
        edge = self.children[action]
        edge.pending += 1
        return action, edge, expanded


class Search:
    def __init__(
        self, iterations=1024, batch_size=32, exploration=2**0.5, device="gpu"
    ):
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise ValueError("iterations must be a positive integer")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if not np.isfinite(exploration) or exploration < 0:
            raise ValueError("exploration must be nonnegative and finite")
        self.iterations = iterations
        self.batch_size = min(batch_size, iterations)
        self.exploration = exploration
        self.simulator = Simulator(device)
        self.last_stats = {}

    def choose(self, observation, key):
        began = time.perf_counter()
        obs = jax.device_get(observation)
        player = int(obs.game.player)
        passing = int(obs.phase) == heart.PASS
        mask = obs.pass_action_mask if passing else obs.play_action_mask
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise ValueError("ISMCTS requires an active, nonterminal observation")
        rng = np.random.default_rng(
            np.asarray(jax.random.key_data(key), dtype=np.uint32)
        )
        belief = Belief(obs)
        root = Node()
        sim = self.simulator
        for offset in range(0, self.iterations, self.batch_size):
            count = min(self.batch_size, self.iterations - offset)
            state = sim.states(obs, belief.sample(count, rng))
            nodes = [root for _ in range(count)]
            paths = [[] for _ in range(count)]
            active = np.ones(count, dtype=bool)
            if passing:
                actions = []
                edges = []
                for i in range(count):
                    action, edge, expanded = root.select(
                        legal, player, self.exploration, rng
                    )
                    actions.append(action)
                    edges.append(edge)
                    paths[i].append(edge)
                    active[i] = not expanded
                state, received = sim.pass_cards(
                    state,
                    sim.put(np.asarray(actions, dtype=np.int32)),
                    sim.keys(count, rng),
                    sim.put(np.int32(player)),
                    sim.put(np.int32(obs.pass_direction)),
                    sim.put(obs.match_scores),
                    sim.put(obs.deal_index),
                )
                received, leaders = jax.device_get((received, state.leader))
                for i, edge in enumerate(edges):
                    # Only the observer's received cards and the public opener are visible.
                    outcome = (tuple(sorted(received[i].tolist())), int(leaders[i]))
                    nodes[i] = edge.outcomes.setdefault(outcome, Node())
            for _ in range(52):
                if not active.any():
                    break
                masks, actors, terminal = jax.device_get(sim.inspect(state))
                active &= ~terminal
                if not active.any():
                    break
                actions = np.zeros(count, dtype=np.int32)
                stepping = active.copy()
                for i in np.flatnonzero(active):
                    action, edge, expanded = nodes[i].select(
                        np.flatnonzero(masks[i]), int(actors[i]), self.exploration, rng
                    )
                    actions[i] = action
                    paths[i].append(edge)
                    nodes[i] = edge.outcomes.setdefault((), Node())
                    active[i] = not expanded
                state = sim.play(state, sim.put(actions), sim.put(stepping))
            rewards = np.asarray(
                jax.device_get(sim.rollout(state, sim.keys(count, rng)))
            )
            if not np.isfinite(rewards).all():
                raise FloatingPointError("non-finite ISMCTS rollout reward")
            for path, reward in zip(paths, rewards):
                for edge in path:
                    edge.pending -= 1
                    edge.visits += 1
                    edge.value += reward
        visits = {action: edge.visits for action, edge in root.children.items()}
        if sum(visits.values()) != self.iterations or any(
            e.pending for e in root.children.values()
        ):
            raise RuntimeError("ISMCTS simulation accounting mismatch")
        best = max(visits.values())
        candidates = [a for a, n in visits.items() if n == best]
        values = [root.children[a].value[player] / best for a in candidates]
        action = candidates[int(np.argmax(values))]
        self.last_stats = {
            "simulations": sum(visits.values()),
            "root_visits": visits,
            "root_actions": len(legal),
            "seconds": time.perf_counter() - began,
        }
        return int(action)
