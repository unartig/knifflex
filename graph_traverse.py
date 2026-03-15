import jax
import jax.numpy as jnp
from jaxtyping import Int, Scalar

from game import KniffelState, score_case
from genome import DAGGenome
from mask_primitives import reroll_mask_to_action
from rule_primitives import run_binary_op, run_state_primitive


def traverse(genome: DAGGenome, state: KniffelState) -> Int[Scalar, ""]:
    must_score = state.rolls_left.squeeze() == 0
    has_score, has_reroll = genome.get_subtree_leaf_flags()

    def cond_fn(node):
        return node >= 0

    def body_fn(node):
        rule = genome.rules[node]
        threshold = genome.thresholds[node]
        binary_op = genome.binary_ops[node]

        left_rule = genome.rules[jnp.maximum(genome.left[node], 0)]
        right_rule = genome.rules[jnp.maximum(genome.right[node], 0)]

        signal = run_state_primitive(rule, state)
        left_signal = run_state_primitive(left_rule, state)
        right_signal = run_state_primitive(right_rule, state)

        enriched = run_binary_op(binary_op, signal, (left_signal + right_signal) / 2.0)
        go_right = enriched >= threshold

        preferred = jnp.where(go_right, genome.right[node], genome.left[node])
        fallback = jnp.where(go_right, genome.left[node], genome.right[node])

        # Gate: if must_score and preferred subtree has no score leaf, force fallback
        def child_has_score(child):
            from_leaf = (child < 0) & ~genome.leaf_is_reroll[-child - 1]
            from_node = jnp.where(child >= 0, has_score[child], False)
            return from_leaf | from_node

        preferred_ok = jnp.where(
            must_score,
            child_has_score(preferred),
            True,  # when can reroll, never force — let tree decide freely
        )

        return jnp.where(preferred_ok, preferred, fallback)

    node = jax.lax.while_loop(cond_fn, body_fn, 0)
    leaf_id = -node - 1
    return leaf_to_action(genome, leaf_id, state)


def best_open_score_action(state: KniffelState) -> Int[Scalar, ""]:
    scores = jax.vmap(lambda cid: score_case(cid, state.dice))(jnp.arange(13))
    open_mask = state.scorecard < 0  # True where still available
    masked = jnp.where(open_mask, scores, -1)  # -1 hides filled slots
    best_cat = jnp.argmax(masked).astype(jnp.int32)
    return best_cat + 32


def leaf_to_action(genome: DAGGenome, leaf_index: Int[Scalar, ""], state: KniffelState) -> Int[Scalar, ""]:

    is_reroll = genome.leaf_is_reroll[leaf_index]
    left_pid = genome.leaf_mask_left[leaf_index]
    right_pid = genome.leaf_mask_right[leaf_index]
    bool_op = genome.leaf_mask_op[leaf_index]
    score_cat = genome.leaf_score_cat[leaf_index]

    def do_reroll(_):
        reroll_action = reroll_mask_to_action(left_pid, right_pid, bool_op, state)

        can_reroll = state.rolls_left.squeeze() > 0
        return jnp.where(can_reroll, reroll_action, best_open_score_action(state))

    def do_score(_):
        requested_action = score_cat + 32
        slot_is_open = state.scorecard[score_cat] < 0
        return jnp.where(slot_is_open, requested_action, best_open_score_action(state))

    return jax.lax.cond(is_reroll, do_reroll, do_score, operand=None)

    is_reroll = genome.leaf_is_reroll[leaf_index]
    left_pid = genome.leaf_mask_left[leaf_index]
    right_pid = genome.leaf_mask_right[leaf_index]
    bool_op = genome.leaf_mask_op[leaf_index]
    score_cat = genome.leaf_score_cat[leaf_index]

    return jax.lax.cond(
        is_reroll,
        lambda _: reroll_mask_to_action(left_pid, right_pid, bool_op, state),
        lambda _: score_cat + 32,
        operand=None,
    )
