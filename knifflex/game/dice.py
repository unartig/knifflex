from __future__ import annotations

from itertools import combinations_with_replacement
from typing import TypeAlias

import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, Scalar, UInt, jaxtyped

from knifflex.utils.utils import typechecker

DiceArray: TypeAlias = UInt[Array, "5"]

N_DICE = 5

# ALL_ROLLS[i]: the i-th sorted 5-dice combination, shape (252, 5)
ALL_ROLLS: Int[Array, f"{N_ROLLS} {N_DICE}"] = jnp.array(
    list(combinations_with_replacement(range(1, 7), N_DICE)),
    dtype=jnp.uint8,
)
N_ROLLS = len(ALL_ROLLS)
assert N_ROLLS == 252

N_MASKS = 32
KEEP_MASKS: Bool[Array, f"{N_MASKS} {N_DICE}"] = jnp.array(
    [[bool((m >> i) & 1) for i in range(5)] for m in range(32)],
    dtype=jnp.bool_,
)


_POW7: Int[Array, f"{N_DICE}"] = jnp.array([7**i for i in range(N_DICE)], dtype=jnp.uint16)

_ROLL_LOOKUP: Int[Array, "16807"] = (
    -jnp.ones(7**5, dtype=jnp.int16)
    .at[jnp.sum(ALL_ROLLS.astype(jnp.int16) * _POW7, axis=1)]
    .set(jnp.arange(N_ROLLS, dtype=jnp.int16))
)


@jaxtyped(typechecker=typechecker)
def dice_to_idx(dice: DiceArray) -> UInt[Scalar, ""]:
    """Sorted dice array -> roll index (0-251).  O(1)."""
    return _ROLL_LOOKUP[jnp.sum(jnp.sort(dice).astype(jnp.uint16) * _POW7)]


@jaxtyped(typechecker=typechecker)
def idx_to_dice(idx: UInt[Scalar, ""] | Int[Scalar, ""]) -> DiceArray:
    """Roll index -> sorted dice array (5,)."""
    return ALL_ROLLS[idx]
