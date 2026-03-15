from enum import IntEnum

import jax
import jax.numpy as jnp

from game import mask_to_reroll_idx


# ------------------------------------------------------------
# Atomic mask primitives  (produce Bool[5] keep-masks)
# ------------------------------------------------------------

class MaskPrimitive(IntEnum):
    FACE_1    = 0
    FACE_2    = 1
    FACE_3    = 2
    FACE_4    = 3
    FACE_5    = 4
    FACE_6    = 5
    COMMONEST = 6
    STRAIGHT  = 7
    ALL       = 8
    NONE      = 9
    # --- New ---
    PAIR      = 10   # keep highest-count pair
    TWO_PAIR  = 11   # keep both pairs when present
    HIGHEST   = 12   # keep only the highest face value
    KNIFFEL   = 13   # keep all dice matching most frequent face


N_MASK_PRIMITIVES = len(MaskPrimitive)


def _mask_face(face):
    def fn(state):
        return state.dice == face
    return fn


def mask_commonest(state):
    counts = jnp.bincount(state.dice, length=7)
    return state.dice == jnp.argmax(counts)


def mask_straight(state):
    present = (jnp.bincount(state.dice, length=7) > 0).astype(jnp.int32)
    faces   = present[1:]

    def scan_fn(carry, x):
        run = jnp.where(x, carry + 1, 0)
        return run, run

    _, runs = jax.lax.scan(scan_fn, 0, faces)
    end    = jnp.argmax(runs)
    length = runs[end]
    start  = end - length + 1
    face_idx = state.dice - 1
    return (face_idx >= start) & (face_idx <= end)


def mask_all(state):
    return jnp.ones(5, dtype=jnp.bool_)


def mask_none(state):
    return jnp.zeros(5, dtype=jnp.bool_)


def mask_pair(state):
    """Keep dice belonging to the highest-count pair (>=2); falls back to commonest."""
    counts   = jnp.bincount(state.dice, length=7)
    has_pair = counts >= 2
    any_pair = jnp.any(has_pair[1:])
    pair_face = jnp.argmax(jnp.arange(7) * has_pair)
    fallback  = jnp.argmax(counts)
    chosen    = jnp.where(any_pair, pair_face, fallback)
    return state.dice == chosen


def mask_two_pair(state):
    """Keep dice belonging to the two highest-faced pairs; degrades to one pair."""
    counts   = jnp.bincount(state.dice, length=7)
    has_pair = (counts >= 2).astype(jnp.int32)
    scored   = jnp.arange(7) * has_pair
    first    = jnp.argmax(scored)
    second   = jnp.argmax(scored.at[first].set(0))
    any_pair = jnp.any(has_pair[1:])
    return jnp.where(any_pair, (state.dice == first) | (state.dice == second), mask_commonest(state))


def mask_highest(state):
    """Keep only dice showing the highest face present."""
    return state.dice == jnp.max(state.dice)


def mask_kniffel(state):
    """Keep all dice matching the most frequent face (Kniffel hunt)."""
    counts = jnp.bincount(state.dice, length=7)
    return state.dice == jnp.argmax(counts)


MASK_PRIMITIVES = [
    _mask_face(1),   # 0
    _mask_face(2),   # 1
    _mask_face(3),   # 2
    _mask_face(4),   # 3
    _mask_face(5),   # 4
    _mask_face(6),   # 5
    mask_commonest,  # 6
    mask_straight,   # 7
    mask_all,        # 8
    mask_none,       # 9
    mask_pair,       # 10
    mask_two_pair,   # 11
    mask_highest,    # 12
    mask_kniffel,    # 13
]

assert len(MASK_PRIMITIVES) == N_MASK_PRIMITIVES


# ------------------------------------------------------------
# Boolean composition ops
# ------------------------------------------------------------

class MaskBoolOp(IntEnum):
    AND   = 0
    OR    = 1
    XOR   = 2
    NOT_A = 3   # complement of left  (right ignored)
    NOT_B = 4   # complement of right (left  ignored)


N_MASK_BOOL_OPS = len(MaskBoolOp)

_BOOL_OPS = [
    lambda a, b: a & b,
    lambda a, b: a | b,
    lambda a, b: a ^ b,
    lambda a, b: ~a,
    lambda a, b: ~b,
]


# ------------------------------------------------------------
# Public runners
# ------------------------------------------------------------

def run_mask_primitive(pid, state):
    """Evaluate one atomic mask → Bool[5]."""
    return jax.lax.switch(pid, MASK_PRIMITIVES, state)


def run_composite_mask(left_pid, right_pid, bool_op_id, state):
    """Compose two atomic masks with a boolean op → Bool[5].
    When left_pid == right_pid and op == OR this is identical to the atomic mask,
    so newly initialised leaves have zero overhead semantically."""
    mask_a = run_mask_primitive(left_pid,  state)
    mask_b = run_mask_primitive(right_pid, state)
    return jax.lax.switch(bool_op_id, _BOOL_OPS, mask_a, mask_b)


def reroll_mask_to_action(left_pid, right_pid, bool_op_id, state):
    """Composite (or atomic when left==right, op==OR) reroll leaf → action index."""
    mask = run_composite_mask(left_pid, right_pid, bool_op_id, state)
    return mask_to_reroll_idx(mask)
