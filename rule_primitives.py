from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import jaxtyped

from ev_table import (
    SCORE_TABLE,  # (252, 13) int32
    get_ev_table,
)
from game import N_ROLLS, ROLLS_TABLE, KniffelState
from utils import typechecker

if TYPE_CHECKING:
    from jaxtyping import Array, Float, Int

# ------------------------------------------------------------------ #
#  EV / mask tables                                                   #
# ------------------------------------------------------------------ #

EV_TABLE, _MASK_TABLE, _ROLLS = (jnp.asarray(arr) for arr in get_ev_table())
MASK_TABLE = _MASK_TABLE.astype(jnp.uint8)

_EV_MAX = jnp.array(
    [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 25.0, 30.0, 30.0, 30.0, 40.0, 30.0, 50.0],
    dtype=jnp.float32,
)

# ------------------------------------------------------------------ #
#  Primitive IDs                                                      #
#                                                                     #
#  36 signals — every one earns its place:                            #
#                                                                     #
#   0      MAX_COUNT         pair/three/four/five of a kind           #
#   1      ROLLS_LEFT        rolls remaining this turn                #
#   2      ROUND_PROGRESS    how far through the game (0→1)           #
#   3      UPPER_BONUS_GAP   how far from the 63-point bonus          #
#   4      N_OPEN_CATEGORIES fraction of slots still open             #
#   5      HAS_TWO_PAIR      useful for full house / pasch routing    #
#   6      UNIQUE_FACES      straight potential, independent of score  #
#   7      EV_GAP            dominance of best option over second     #
#   8      EV_UPPER_BEST     best normed EV in upper section          #
#   9      EV_LOWER_BEST     best normed EV in lower section          #
#   10-22  SCORE_CAT_0..12   score-now normed (zero more rolls)       #
#   23-35  EV_CAT_0..12      EV normed, rolls-aware                   #
# ------------------------------------------------------------------ #


class StatePrimitive(IntEnum):
    MAX_COUNT = 0
    ROLLS_LEFT = 1
    ROUND_PROGRESS = 2
    UPPER_BONUS_GAP = 3
    N_OPEN_CATEGORIES = 4
    HAS_TWO_PAIR = 5
    UNIQUE_FACES = 6
    EV_GAP = 7
    EV_UPPER_BEST = 8
    EV_LOWER_BEST = 9
    SCORE_CAT_0 = 10
    SCORE_CAT_1 = 11
    SCORE_CAT_2 = 12
    SCORE_CAT_3 = 13
    SCORE_CAT_4 = 14
    SCORE_CAT_5 = 15
    SCORE_CAT_6 = 16
    SCORE_CAT_7 = 17
    SCORE_CAT_8 = 18
    SCORE_CAT_9 = 19
    SCORE_CAT_10 = 20
    SCORE_CAT_11 = 21
    SCORE_CAT_12 = 22
    EV_CAT_0 = 23
    EV_CAT_1 = 24
    EV_CAT_2 = 25
    EV_CAT_3 = 26
    EV_CAT_4 = 27
    EV_CAT_5 = 28
    EV_CAT_6 = 29
    EV_CAT_7 = 30
    EV_CAT_8 = 31
    EV_CAT_9 = 32
    EV_CAT_10 = 33
    EV_CAT_11 = 34
    EV_CAT_12 = 35


N_STATE_PRIMITIVES = len(StatePrimitive)  # 36
assert N_STATE_PRIMITIVES == 36

# ------------------------------------------------------------------ #
#  Dice primitive table  (252 x 16)                                   #
#  Columns:                                                           #
#    0     max_count (normed)                                         #
#    1     unique_faces (normed)                                      #
#    2     has_two_pair                                               #
#    3-15  score_cat_0..12 (normed, score-now)                        #
# ------------------------------------------------------------------ #

_ROLLS_NP = np.array(ROLLS_TABLE)  # (252, 5)


def _build_dice_primitive_table() -> np.ndarray:
    rolls = _ROLLS_NP
    n = N_ROLLS
    cols = []

    # 0  max_count (normed 0..1)
    cols.append(np.array([np.bincount(rolls[i], minlength=7).max() / 5.0 for i in range(n)]))

    # 1  unique_faces (normed 0..1)
    cols.append(np.array([np.sum(np.bincount(rolls[i], minlength=7)[1:] > 0) / 6.0 for i in range(n)]))

    # 2  has_two_pair
    cols.append(np.array([float(np.sum(np.bincount(rolls[i], minlength=7) >= 2) >= 2) for i in range(n)]))

    # 3-15  score_cat_0..12 normed

    score_np = np.array(SCORE_TABLE, dtype=np.float32)
    max_scores = np.array([5, 10, 15, 20, 25, 30, 25, 30, 30, 30, 40, 30, 50], dtype=np.float32)
    for cat in range(13):
        cols.append(score_np[:, cat] / max_scores[cat])

    table = np.stack(cols, axis=1).astype(np.float32)  # (252, 16)
    assert table.shape == (252, 16), table.shape
    return table


DICE_PRIMITIVE_TABLE = jnp.array(_build_dice_primitive_table(), dtype=jnp.float32)


# ------------------------------------------------------------------ #
#  compute_all_primitives  →  Float[36]                               #
# ------------------------------------------------------------------ #


@jaxtyped(typechecker=typechecker)
def compute_all_primitives(state: KniffelState) -> Float[Array, 36]:
    """Compute all 36 primitives for the current state.

    Layout matches StatePrimitive exactly.
    """
    dice_feats = DICE_PRIMITIVE_TABLE[state.dice_idx]  # (16,)
    # cols: 0=max_count, 1=unique_faces, 2=has_two_pair, 3-15=score_cat_0..12

    rl = state.rolls_left.squeeze()
    open_mask = state.scorecard < 0  # (13,) bool

    # Rolls-aware EV, normalized
    ev_raw = EV_TABLE[state.dice_idx, rl]  # (13,) raw
    masked_ev = jnp.where(open_mask, ev_raw, 0.0)
    norm_ev = masked_ev / _EV_MAX  # (13,) in [0,1]

    # Scalar state features
    rl_norm = rl.astype(jnp.float32) / 2.0  # {0,1,2} → {0,.5,1}
    round_prog = state.round.astype(jnp.float32).squeeze() / 13.0
    upper_filled = jnp.sum(jnp.where(state.scorecard[:6] >= 0, state.scorecard[:6], 0)).astype(jnp.float32)
    upper_bonus_gap = jnp.clip((63.0 - upper_filled) / 63.0, 0.0, 1.0)
    n_open = jnp.sum(open_mask).astype(jnp.float32) / 13.0

    # EV summary
    best_ev = jnp.max(masked_ev)
    second_ev = jnp.max(jnp.where(masked_ev >= best_ev, 0.0, masked_ev))
    ev_gap = jnp.clip((best_ev - second_ev) / 50.0, 0.0, 1.0)
    ev_upper_best = jnp.max(norm_ev[:6])
    ev_lower_best = jnp.max(norm_ev[6:])

    return jnp.concatenate(
        [
            dice_feats[0:1],  # 0   max_count
            jnp.array([rl_norm]),  # 1   rolls_left
            jnp.array([round_prog]),  # 2   round_progress
            jnp.array([upper_bonus_gap]),  # 3   upper_bonus_gap
            jnp.array([n_open]),  # 4   n_open_categories
            dice_feats[2:3],  # 5   has_two_pair
            dice_feats[1:2],  # 6   unique_faces
            jnp.array([ev_gap, ev_upper_best, ev_lower_best]),  # 7-9
            dice_feats[3:16],  # 10-22 score_cat_0..12
            norm_ev,  # 23-35 ev_cat_0..12
        ]
    )  # total: 36


# ------------------------------------------------------------------ #
#  Binary ops                                                         #
# ------------------------------------------------------------------ #


class BinaryOp(IntEnum):
    ADD = 0  # (a+b)/2      are both signals moderately high?
    SUB = 1  # a-b+0.5      is a clearly higher than b? (centred at 0.5)
    MUL = 2  # a*b          are BOTH signals high? (strict AND)
    MAX = 3  # max(a,b)     is EITHER signal high? (OR)
    MIN = 4  # min(a,b)     is the weaker signal high?
    GT = 5  # a>b → 1/0    clean boolean comparison
    A = 6
    B = 7


N_BINARY_OPS = len(BinaryOp)  # 6


def add(a, b):
    return jnp.clip((a + b) / 2.0, 0.0, 1.0)


def sub(a, b):
    return jnp.clip(a - b + 0.5, 0.0, 1.0)


def mul(a, b):
    return jnp.clip(a * b, 0.0, 1.0)


def max_op(a, b):
    return jnp.maximum(a, b)


def min_op(a, b):
    return jnp.minimum(a, b)


def gt(a, b):
    return (a > b).astype(jnp.float32)
def a(a, b):
    return (a).astype(jnp.float32)
def b(a, b):
    return (b).astype(jnp.float32)



BINARY_OPS = [add, sub, mul, max_op, min_op, gt, a, b]


def run_binary_op(op_id, a, b):
    return jax.lax.switch(op_id, BINARY_OPS, a, b)
