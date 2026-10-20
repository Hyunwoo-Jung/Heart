"""Connect ISMCTS to HEART without forwarding omniscient state to the search."""

import jax

from heart.classic import observe_classic

from .search import Search


def make_policy(iterations="1024", *, batch_size=32, exploration=2**0.5, device="gpu"):
    search = Search(int(iterations), batch_size, exploration, device)
    observe = jax.jit(observe_classic)

    def choose(state, player, key):
        return search.choose(observe(state, player), key)

    choose.search = search
    return choose
