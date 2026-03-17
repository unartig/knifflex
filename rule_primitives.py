from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import jaxtyped

from game import KniffelState, score_case, score_upper, score_full_house, score_three_of_a_kind, score_four_of_a_kind, score_small_straight, score_large_straight, score_faces, score_kniffel
from utils import typechecker

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int

# ------------------------------------------------------------
# Primitive IDs
# ------------------------------------------------------------


class StatePrimitive(IntEnum):
    COUNT_1 = 0
    COUNT_2 = 1
    COUNT_3 = 2
    COUNT_4 = 3
    COUNT_5 = 4
    COUNT_6 = 5

    SUM_1 = 6
    SUM_2 = 7
    SUM_3 = 8
    SUM_4 = 9
    SUM_5 = 10
    SUM_6 = 11

    HAS_2_KIND = 12
    HAS_3_KIND = 13
    HAS_4_KIND = 14
    HAS_5_KIND = 15

    ROLLS_LEFT = 16
    UNIQUE_FACES = 17
    MAX_COUNT = 18
    ROUND_PROGRESS = 19
    UPPER_BONUS_PROGRESS = 20

    SCORE_EINSEN = 21
    SCORE_ZWEIEN = 22
    SCORE_DREIEN = 23
    SCORE_VIEREN = 24
    SCORE_FUENFEN = 25
    SCORE_SECHSEN = 26
    SCORE_FULL_HOUSE = 27
    SCORE_DREIER_PASCH = 28
    SCORE_VIERER_PASCH = 29
    SCORE_KLEINE_STR = 30
    SCORE_GROSSE_STR = 31
    SCORE_AUGENZAHL = 32
    SCORE_KNIFFEL = 33

    # --- New primitives ---
    HAS_FULL_HOUSE   = 34  # bool: has both a pair and a triple
    HAS_TWO_PAIR     = 35  # bool: has two distinct pairs
    IS_LAST_ROLL     = 36  # bool: rolls_left == 0, must score next
    N_OPEN_CATEGORIES = 37  # how many scorecard slots remain (normalised)
    UPPER_BONUS_GAP  = 38  # points still needed to reach 63 (normalised)
    BEST_SCORE_AVAILABLE = 39  # max normalised score across all open categories


N_STATE_PRIMITIVES = len(StatePrimitive)


# ------------------------------------------------------------
# Existing primitive implementations
# ------------------------------------------------------------

def _count_face(face: int) -> Callable:
    @jaxtyped(typechecker=typechecker)
    def fn(state: KniffelState) -> Float[Array, ""]:
        return jnp.sum(state.dice == face).astype(jnp.float32) / 5.0

    return fn


def _sum_face(face: int) -> Callable:
    @jaxtyped(typechecker=typechecker)
    def fn(state: KniffelState) -> Float[Array, ""]:
        total = jnp.sum(jnp.where(state.dice == face, face, 0))
        return total.astype(jnp.float32) / (5.0 * face)

    return fn


def _has_n_of_kind(n: int) -> Callable:
    @jaxtyped(typechecker=typechecker)
    def fn(state: KniffelState) -> Float[Array, ""]:
        counts = jnp.bincount(state.dice, length=7)
        return (jnp.max(counts) >= n).astype(jnp.float32)

    return fn


@jaxtyped(typechecker=typechecker)
def rolls_left(state: KniffelState) -> Float[Array, ""]:
    return state.rolls_left.astype(jnp.float32).squeeze() / 3.0


@jaxtyped(typechecker=typechecker)
def unique_faces(state: KniffelState) -> Float[Array, ""]:
    return jnp.sum(jnp.bincount(state.dice, length=7)[1:] > 0).astype(jnp.float32) / 6.0


@jaxtyped(typechecker=typechecker)
def max_count(state: KniffelState) -> Float[Array, ""]:
    return jnp.max(jnp.bincount(state.dice, length=7)).astype(jnp.float32) / 5.0


@jaxtyped(typechecker=typechecker)
def round_progress(state: KniffelState) -> Float[Array, ""]:
    return state.round.astype(jnp.float32).squeeze() / 13.0


@jaxtyped(typechecker=typechecker)
def upper_bonus_progress(state: KniffelState) -> Float[Array, ""]:
    upper = jnp.sum(jnp.where(state.scorecard[:6] >= 0, state.scorecard[:6], 0))
    return jnp.clip(upper.astype(jnp.float32) / 63.0, 0.0, 1.0)


def _score_category(case_id: int, max_score: float) -> Callable:
    @jaxtyped(typechecker=typechecker)
    def fn(state: KniffelState) -> Float[Array, ""]:
        open_mask = state.scorecard[case_id] < 0
        raw = score_case(jnp.array(case_id), state.dice).astype(jnp.float32)
        return jnp.where(open_mask, raw / max_score, 0.0)

    return fn


# ------------------------------------------------------------
# New primitive implementations
# ------------------------------------------------------------

# Max scores per category — mirrors the existing _score_category normalisers.
_CATEGORY_MAX_SCORES = jnp.array(
    [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 25.0, 30.0, 30.0, 30.0, 40.0, 30.0, 50.0],
    dtype=jnp.float32,
)


@jaxtyped(typechecker=typechecker)
def has_full_house(state: KniffelState) -> Float[Array, ""]:
    """1.0 if the current dice contain both a pair and a triple, else 0.0."""
    counts = jnp.bincount(state.dice, length=7)
    has_three = jnp.any(counts == 3)
    has_two   = jnp.any(counts == 2)
    return (has_three & has_two).astype(jnp.float32)


@jaxtyped(typechecker=typechecker)
def has_two_pair(state: KniffelState) -> Float[Array, ""]:
    """1.0 if the current dice contain at least two distinct pairs, else 0.0."""
    counts = jnp.bincount(state.dice, length=7)
    n_pairs = jnp.sum(counts >= 2)
    return (n_pairs >= 2).astype(jnp.float32)


@jaxtyped(typechecker=typechecker)
def is_last_roll(state: KniffelState) -> Float[Array, ""]:
    """1.0 when rolls_left == 0 — the agent must score on this step."""
    return (state.rolls_left.squeeze() == 0).astype(jnp.float32)


@jaxtyped(typechecker=typechecker)
def n_open_categories(state: KniffelState) -> Float[Array, ""]:
    """Fraction of scorecard slots still open (0.0 = all filled, 1.0 = all open)."""
    open_count = jnp.sum(state.scorecard < 0).astype(jnp.float32)
    return open_count / 13.0


@jaxtyped(typechecker=typechecker)
def upper_bonus_gap(state: KniffelState) -> Float[Array, ""]:
    """How far the agent still is from the 63-point upper bonus, normalised to [0, 1].
    0.0 means the bonus is already secured; 1.0 means no upper points scored yet."""
    upper = jnp.sum(jnp.where(state.scorecard[:6] >= 0, state.scorecard[:6], 0))
    gap = jnp.clip(63.0 - upper.astype(jnp.float32), 0.0, 63.0)
    return gap / 63.0


@jaxtyped(typechecker=typechecker)
def best_score_available(state: KniffelState) -> Float[Array, ""]:
    """Normalised score of the best open category for the current dice.
    Returns 0.0 if all categories are filled."""
    dice = state.dice
    raw = jnp.array([
        score_upper(dice, 1), score_upper(dice, 2), score_upper(dice, 3),
        score_upper(dice, 4), score_upper(dice, 5), score_upper(dice, 6),
        score_full_house(dice), score_three_of_a_kind(dice), score_four_of_a_kind(dice),
        score_small_straight(dice), score_large_straight(dice), score_faces(dice),
        score_kniffel(dice),
    ], dtype=jnp.float32)
    open_mask = state.scorecard < 0
    return jnp.max(jnp.where(open_mask, raw / _CATEGORY_MAX_SCORES, 0.0))


# ------------------------------------------------------------
# Primitive table — order must match StatePrimitive IntEnum
# ------------------------------------------------------------

STATE_PRIMITIVES: list[Callable] = [
    # COUNT_1 .. COUNT_6
    *(_count_face(i) for i in range(1, 7)),
    # SUM_1 .. SUM_6
    *(_sum_face(i) for i in range(1, 7)),
    # HAS_2_KIND .. HAS_5_KIND
    *(_has_n_of_kind(i) for i in range(2, 6)),
    # Structural
    rolls_left,         # 16
    unique_faces,       # 17
    max_count,          # 18
    round_progress,     # 19
    upper_bonus_progress,  # 20
    # Per-category score signals
    _score_category(0, 5.0),    # 21  SCORE_EINSEN
    _score_category(1, 10.0),   # 22  SCORE_ZWEIEN
    _score_category(2, 15.0),   # 23  SCORE_DREIEN
    _score_category(3, 20.0),   # 24  SCORE_VIEREN
    _score_category(4, 25.0),   # 25  SCORE_FUENFEN
    _score_category(5, 30.0),   # 26  SCORE_SECHSEN
    _score_category(6, 25.0),   # 27  SCORE_FULL_HOUSE
    _score_category(7, 30.0),   # 28  SCORE_DREIER_PASCH
    _score_category(8, 30.0),   # 29  SCORE_VIERER_PASCH
    _score_category(9, 30.0),   # 30  SCORE_KLEINE_STR
    _score_category(10, 40.0),  # 31  SCORE_GROSSE_STR
    _score_category(11, 30.0),  # 32  SCORE_AUGENZAHL
    _score_category(12, 50.0),  # 33  SCORE_KNIFFEL
    # --- New ---
    has_full_house,         # 34  HAS_FULL_HOUSE
    has_two_pair,           # 35  HAS_TWO_PAIR
    is_last_roll,           # 36  IS_LAST_ROLL
    n_open_categories,      # 37  N_OPEN_CATEGORIES
    upper_bonus_gap,        # 38  UPPER_BONUS_GAP
    best_score_available,   # 39  BEST_SCORE_AVAILABLE
]

assert len(STATE_PRIMITIVES) == N_STATE_PRIMITIVES, (
    f"STATE_PRIMITIVES length {len(STATE_PRIMITIVES)} != N_STATE_PRIMITIVES {N_STATE_PRIMITIVES}"
)


@jaxtyped(typechecker=typechecker)
def run_state_primitive(
    primitive_id: int | Int[Array, ""],
    state: KniffelState,
) -> Float[Array, ""]:
    return jax.lax.switch(
        primitive_id,
        STATE_PRIMITIVES,
        state,
    )


# ------------------------------------------------------------
# Binary ops (unchanged)
# ------------------------------------------------------------

class BinaryOp(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    MAX = 3
    MIN = 4


N_BINARY_OPS = len(BinaryOp)


def add(a, b):
    return jnp.clip((a + b) / 2.0, 0.0, 1.0)


def sub(a, b):
    return jnp.clip(a - b, 0.0, 1.0)


def mul(a, b):
    return jnp.clip(a * b, 0.0, 1.0)


def max_op(a, b):
    return jnp.maximum(a, b)


def min_op(a, b):
    return jnp.minimum(a, b)


BINARY_OPS = [
    add,
    sub,
    mul,
    max_op,
    min_op,
]


def run_binary_op(op_id, a, b):
    return jax.lax.switch(
        op_id,
        BINARY_OPS,
        a,
        b,
    )
