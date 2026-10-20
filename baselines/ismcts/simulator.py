"""JIT/vmap deal transitions and medium rollouts on the requested JAX device."""

import jax
import jax.numpy as jnp
import numpy as np

import heart
from heart.classic import PASS_COMBINATIONS, ClassicObservation
from heart.rules import legal_action_mask, observe, step_unchecked
from heart.types import State

PLAY_POLICY = heart.make_rule_policy("medium")
PASS_POLICY = heart.make_rule_pass_policy("medium")


def play_step(state, action, enabled):
    return jax.lax.cond(
        enabled & ~state.terminated,
        lambda _: step_unchecked(state, action)[0],
        lambda _: state,
        operand=None,
    )


def pass_step(state, action, key, player, direction, match_scores, deal_index):
    """Choose all hidden passes from pre-exchange hands, then exchange together."""
    selected = []
    for actor in range(4):
        game = state._replace(active_player=jnp.asarray(actor, dtype=jnp.int32))
        obs = ClassicObservation(
            game=observe(game, actor),
            match_scores=match_scores,
            deal_index=deal_index,
            phase=jnp.asarray(heart.PASS, dtype=jnp.int8),
            pass_direction=direction,
            cards_passed=jnp.full(3, -1, dtype=jnp.int8),
            cards_received=jnp.full(3, -1, dtype=jnp.int8),
            pass_action_mask=jnp.ones(286, dtype=bool),
            play_action_mask=jnp.zeros(52, dtype=bool),
        )
        choice = jnp.where(
            actor == player,
            action,
            PASS_POLICY(obs, jax.random.fold_in(key, actor)),
        )
        cards = jnp.nonzero(state.hands[actor], size=13)[0]
        selected.append(cards[PASS_COMBINATIONS[choice]])
    selected = jnp.stack(selected)
    recipients = (jnp.arange(4) + jnp.asarray([1, -1, 2])[direction]) % 4
    hands = state.hands.at[jnp.arange(4)[:, None], selected].set(False)
    hands = hands.at[recipients[:, None], selected].set(True)
    received = jnp.empty((4, 3), dtype=jnp.int32).at[recipients].set(selected)
    opener = jnp.argmax(hands[:, 0]).astype(jnp.int32)
    return state._replace(hands=hands, active_player=opener, leader=opener), received[
        player
    ]


def rollout(state, key):
    def advance(carry):
        game, rng_key = carry
        rng_key, action_key = jax.random.split(rng_key)
        action = PLAY_POLICY(observe(game, game.active_player), action_key)
        return step_unchecked(game, action)[0], rng_key

    state, _ = jax.lax.while_loop(
        lambda carry: ~carry[0].terminated, advance, (state, key)
    )
    return -state.scores.astype(jnp.float32) / 26.0


class Simulator:
    def __init__(self, device="gpu"):
        if device not in ("gpu", "cpu"):
            raise ValueError("device must be gpu or cpu")
        try:
            self.device = jax.devices(device)[0]
        except (RuntimeError, IndexError) as error:
            raise RuntimeError(
                f"JAX {device} backend unavailable; install a matching JAX runtime"
            ) from error
        self.play = jax.jit(jax.vmap(play_step))
        self.pass_cards = jax.jit(
            jax.vmap(pass_step, in_axes=(0, 0, 0, None, None, None, None))
        )
        self.rollout = jax.jit(jax.vmap(rollout))
        self.inspect = jax.jit(
            jax.vmap(lambda s: (legal_action_mask(s), s.active_player, s.terminated))
        )

    def put(self, value):
        return jax.device_put(value, self.device)

    def keys(self, count, rng):
        key = self.put(jax.random.key(int(rng.integers(0, 2**32))))
        return jax.random.split(key, count)

    def states(self, obs, hands):
        game = obs.game
        batch = len(hands)
        played = np.count_nonzero(np.asarray(game.trick_history) >= 0)
        played += np.count_nonzero(np.asarray(game.current_trick) >= 0)
        state = State(
            hands=hands,
            current_trick=np.asarray(game.current_trick, dtype=np.int8),
            trick_history=np.asarray(game.trick_history, dtype=np.int8),
            trick_winners=np.asarray(game.trick_winners, dtype=np.int8),
            penalties=np.asarray(game.penalties, dtype=np.int16),
            scores=np.asarray(game.scores, dtype=np.int16),
            moon_shooter=np.asarray(-1, dtype=np.int8),
            winner_mask=np.zeros(4, dtype=bool),
            leader=np.asarray(game.leader, dtype=np.int32),
            active_player=np.asarray(game.player, dtype=np.int32),
            trick_index=np.asarray(game.trick_index, dtype=np.int32),
            trick_position=np.asarray(game.trick_position, dtype=np.int32),
            hearts_broken=np.asarray(game.hearts_broken, dtype=bool),
            num_cards_played=np.asarray(played, dtype=np.int32),
            terminated=np.asarray(False),
        )
        state = State(
            hands,
            *(np.broadcast_to(x, (batch, *x.shape)).copy() for x in state[1:]),
        )
        if int(obs.phase) == heart.PASS:
            # The pre-pass leader depends on the sampled deal, not the real hands.
            state = state._replace(
                leader=np.argmax(hands[:, :, 0], axis=1).astype(np.int32)
            )
        return self.put(state)
