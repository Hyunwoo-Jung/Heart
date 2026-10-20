"""Sample remaining hands using only one player's observation."""

from functools import cache

import numpy as np

from heart import PASS


def public_plays(game):
    """Recover actors from each public trick's winner and winning card."""
    plays = []
    for cards, winner in zip(game.trick_history, game.trick_winners):
        cards = np.asarray(cards, dtype=int)
        if np.all(cards == -1):
            continue
        if np.any(cards < 0) or not 0 <= int(winner) < 4:
            raise ValueError("incomplete public trick history")
        eligible = cards // 13 == cards[0] // 13
        offset = int(np.argmax(np.where(eligible, cards % 13, -1)))
        leader = (int(winner) - offset) % 4
        plays.extend(((leader + i) % 4, int(card)) for i, card in enumerate(cards))
    plays.extend(
        ((int(game.leader) + i) % 4, int(card))
        for i, card in enumerate(game.current_trick)
        if card >= 0
    )
    return plays


class Belief:
    """Uniform over assignments satisfying explicit public-card constraints.

    Counting completions before sampling avoids greedy allocation dead ends.
    No behavioral likelihood for earlier opponent choices is assumed.
    """

    def __init__(self, obs):
        self.player = int(obs.game.player)
        self.fixed = np.zeros((4, 52), dtype=bool)
        self.fixed[self.player] = np.asarray(obs.game.hand)
        self.plays = public_plays(obs.game)
        seen = [card for _, card in self.plays]
        if len(set(seen)) != len(seen) or np.any(self.fixed[:, seen]):
            raise ValueError("public cards overlap or repeat")
        sizes = np.asarray(obs.game.hand_sizes, dtype=int)
        counts = np.bincount([p for p, _ in self.plays], minlength=4)
        if not np.array_equal(sizes + counts, np.full(4, 13)):
            raise ValueError("hand sizes disagree with public history")
        if self.fixed[self.player].sum() != sizes[self.player]:
            raise ValueError("own hand size disagrees with observation")
        self.allowed = np.ones((4, 52), dtype=bool)
        suits = np.arange(52) // 13
        points = (suits == 3) | (np.arange(52) == 36)
        broken = False
        for i, (player, card) in enumerate(self.plays):
            if i == 0 and card != 0:
                raise ValueError("the public opening card must be 2C")
            leading = i % 4 == 0
            led_suit = self.plays[i - i % 4][1] // 13
            if not leading and card // 13 != led_suit:
                self.allowed[player, suits == led_suit] = False
            if leading and not broken and card // 13 == 3:
                self.allowed[player, suits != 3] = False
            if i < 4 and points[card]:
                self.allowed[player, ~points] = False
            # Later public cards must also obey facts established earlier.
            if not self.allowed[player, card]:
                raise ValueError("public play contradicts a known void")
            broken |= card // 13 == 3
        if bool(obs.game.hearts_broken) != broken:
            raise ValueError("hearts-broken flag disagrees with public history")
        if int(obs.phase) != PASS:
            direction = int(obs.pass_direction)
            if direction != 3:
                recipient = (self.player + (1, -1, 2)[direction]) % 4
                owners = {card: player for player, card in self.plays}
                for card in np.asarray(obs.cards_passed, dtype=int):
                    if card < 0:
                        continue
                    if card in owners:
                        if owners[card] != recipient:
                            raise ValueError(
                                "passed card was played by the wrong player"
                            )
                    else:
                        self.fixed[recipient, card] = True
        if np.any(self.fixed.sum(axis=0) > 1) or np.any(self.fixed & ~self.allowed):
            raise ValueError("known hands contradict public information")
        remaining = np.ones(52, dtype=bool)
        remaining[seen] = False
        remaining[self.fixed.any(axis=0)] = False
        self.players = tuple(p for p in range(4) if p != self.player)
        self.capacity = tuple(int(sizes[p] - self.fixed[p].sum()) for p in self.players)
        self.cards = tuple(
            sorted(
                np.flatnonzero(remaining).tolist(),
                key=lambda card: sum(self.allowed[p, card] for p in self.players),
            )
        )
        self.choices = tuple(
            tuple(i for i, p in enumerate(self.players) if self.allowed[p, card])
            for card in self.cards
        )
        if min(self.capacity) < 0 or sum(self.capacity) != len(self.cards):
            raise ValueError("known cards exceed hand capacities")

        @cache
        def completions(index, capacity):
            if index == len(self.cards):
                return int(not any(capacity))
            total = 0
            for choice in self.choices[index]:
                if capacity[choice] > 0:
                    after = list(capacity)
                    after[choice] -= 1
                    total += completions(index + 1, tuple(after))
            return total

        self.completions = completions
        if not completions(0, self.capacity):
            raise ValueError("no hidden hand assignment satisfies the observation")

    def sample(self, count, rng):
        hands = np.broadcast_to(self.fixed, (count, 4, 52)).copy()
        for row in hands:
            capacity = self.capacity
            for index, card in enumerate(self.cards):
                choices, weights, capacities = [], [], []
                for choice in self.choices[index]:
                    if capacity[choice] > 0:
                        after = list(capacity)
                        after[choice] -= 1
                        weight = self.completions(index + 1, tuple(after))
                        if weight:
                            choices.append(choice)
                            weights.append(weight)
                            capacities.append(tuple(after))
                probabilities = np.asarray(weights, dtype=np.float64)
                probabilities /= probabilities.sum()
                selected = int(rng.choice(len(choices), p=probabilities))
                row[self.players[choices[selected]], card] = True
                capacity = capacities[selected]
        return hands
