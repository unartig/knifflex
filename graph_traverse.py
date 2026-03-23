import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Scalar

from game import KniffelState
from genome import N_LEAVES, DAGGenome
from rule_primitives import (
    _EV_MAX,
    EV_TABLE,
    MASK_TABLE,
    compute_all_primitives,
    run_binary_op,
)

# ------------------------------------------------------------------ #
#  Oracle                                                             #
# ------------------------------------------------------------------ #


def oracle_action(state: KniffelState, target_cat: Int[Scalar, ""]) -> Int[Scalar, ""]:
    rl = state.rolls_left.squeeze()
    idx = state.dice_idx
    open_mask = state.scorecard < 0

    # Safety: if target_cat is filled, fall back to first open slot
    safe_cat = jnp.where(
        open_mask[target_cat],
        target_cat,
        jnp.argmax(open_mask).astype(jnp.int32),  # punish for wrong move
    )

    ev_now = EV_TABLE[idx, 0, safe_cat]
    ev_here = EV_TABLE[idx, rl, safe_cat]

    def do_score(_):
        return (safe_cat + 32).astype(jnp.int32)

    def do_reroll(_):
        keep_mask_idx = MASK_TABLE[idx, rl - 1, target_cat].astype(jnp.int32)
        return keep_mask_idx

    return jax.lax.cond(
        (rl == 0) | (ev_here <= ev_now),
        do_score,
        do_reroll,
        operand=None,
    )


# ------------------------------------------------------------------
#  Leaf strategies
#
#  Each strategy is a pure function (signals, open_mask) -> cat_idx.
#  Signals are the 36-dim primitive vector.
#  open_mask is Bool[13] - filled categories must not be selected.
#
#  Strategy index matches LEAF_STRATEGY_NAMES in genome.py:
# 0  argmax_EV                    greedy baseline — always valid
# 1  argmax_SCORE_NOW             commit to what you have now
# 2  upper_bonus_chase            argmax EV_upper, but only if bonus still reachable
#                                 (upper_bonus_gap < 0.5), else argmax_EV
# 3  argmax_EV_lower              lower section focus
# 4  kniffel_or_best              cat 12 if EV_CAT_12 > 0.3, else argmax_EV
# 5  sacrifice                    argmin EV — dump the worst slot
# 6  score_now_upper_else_ev      take guaranteed upper points, else EV
# 7  argmax_EV_excluding_best     second-best EV — useful when best is unreachable
#                                 next turn due to scorecard interaction
#
#  Primitives layout reminder (from rule_primitives.py):
#    10-22  SCORE_CAT_0..12  score-now normed
#    23-35  EV_CAT_0..12     rolls-aware EV normed
# ------------------------------------------------------------------

_UPPER = jnp.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
_LOWER = jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float32)
_INF = jnp.float32(1e9)


_HIGH_VAR = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], dtype=jnp.bool_)
_SAFE_LOWER = jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=jnp.bool_)


def _apply_strategy(
    strategy_id: Int[Scalar, ""],
    signals: Float[jnp.ndarray, " 36"],
    open_mask: "jnp.ndarray",  # Bool[13]
) -> Int[Scalar, ""]:
    """Dispatch to one of 8 strategies. Returns category index 0..12."""
    bonus_reachable = signals[3] < 0.5  # upper_bonus_gap primitive

    ev_cats = signals[23:36]  # (13,) rolls-aware EV, normed
    score_cats = signals[10:23]  # (13,) score-now, normed

    # Mask helpers: -inf for filled slots so argmax never picks them
    ev_masked = jnp.where(open_mask, ev_cats, -_INF)
    score_masked = jnp.where(open_mask, score_cats, -_INF)

    # Section-masked versions
    ev_upper = jnp.where(open_mask & (_UPPER > 0), ev_cats, -_INF)
    ev_lower = jnp.where(open_mask & (_LOWER > 0), ev_cats, -_INF)

    # For argmin (sacrifice): +inf for filled so argmin never picks them
    ev_for_min = jnp.where(open_mask, ev_cats, _INF)
    bonus_reachable = signals[3] < 0.5  # upper_bonus_gap primitive
    ev_high_var = jnp.where(open_mask & _HIGH_VAR, ev_cats, -_INF)
    ev_safe_lower = jnp.where(open_mask & _SAFE_LOWER, ev_cats, -_INF)
    # bonus chase: upper EV but only if bonus still reachable, else fall back to argmax_EV
    ev_bonus_chase = jnp.where(bonus_reachable, ev_upper, ev_masked)

    strategies = [
        lambda: jnp.argmax(ev_masked).astype(jnp.int32),  # 0 argmax_EV — greedy baseline
        lambda: jnp.argmax(score_masked).astype(jnp.int32),  # 1 argmax_SCORE_NOW — commit to dice
        lambda: jnp.argmax(ev_bonus_chase).astype(jnp.int32),  # 2 bonus_chase_or_best — upper if reachable
        lambda: jnp.argmax(ev_upper).astype(jnp.int32),  # 3 argmax_EV_upper — always upper
        lambda: jnp.argmax(ev_lower).astype(jnp.int32),  # 4 argmax_EV_lower — always lower
        lambda: jnp.argmax(ev_high_var).astype(jnp.int32),  # 5 chase_high_var — straights/Kniffel
        lambda: jnp.argmax(ev_safe_lower).astype(jnp.int32),  # 6 safe_lower — FH/pasch/augenzahl
        lambda: jnp.argmin(ev_for_min).astype(jnp.int32),  # 7 sacrifice — dump worst slot
    ]

    assert len(strategies) == N_LEAVES
    return jax.lax.switch(strategy_id, strategies)


# ------------------------------------------------------------------ #
#  DAG traversal                                                      #
# ------------------------------------------------------------------ #


def traverse(genome: DAGGenome, state: KniffelState, max_depth: int) -> Int[Scalar, ""]:
    """Traverse the DAG to a strategy leaf, apply it, pass to oracle.

    1. Compute 36 primitives once.
    2. Walk the tree: each node evaluates binary_op(signal_L, signal_R)
       and branches on >= threshold.
    3. On reaching a leaf, apply its strategy to pick a category.
    4. If stuck (cycle/back-edge), fall back to strategy 0 (argmax_EV).
    5. Oracle converts category → reroll bitmask or score action.
    """
    signals = compute_all_primitives(state)  # (36,)
    open_mask = state.scorecard < 0  # (13,) bool

    def step(carry, _):
        node, done = carry
        safe = jnp.maximum(node, 0)

        rl_idx = genome.rules_left[safe].astype(jnp.int32)
        rr_idx = genome.rules_right[safe].astype(jnp.int32)
        bop = genome.binary_ops[safe]
        thresh = genome.thresholds[safe]

        val = run_binary_op(bop, signals[rl_idx], signals[rr_idx])
        go_r = val >= thresh

        lchild = genome.left[safe]
        rchild = genome.right[safe]
        next_node = jnp.where(go_r, rchild, lchild)

        is_stuck = (next_node == node) & (node >= 0)
        hit_leaf = next_node < 0
        new_done = done | hit_leaf | is_stuck

        return (jnp.where(done, node, next_node), new_done), None

    (final_node, did_reach_leaf), _ = jax.lax.scan(step, (jnp.int32(0), jnp.bool_(False)), None, length=max_depth)

    # Decode leaf id (negative encoding)
    leaf_id = jnp.where(final_node < 0, -final_node - 1, jnp.int32(0))
    strategy_id = jnp.where(did_reach_leaf, leaf_id, jnp.int32(0))  # fallback: argmax_EV

    target_cat = _apply_strategy(strategy_id, signals, open_mask)
    return oracle_action(state, target_cat)
