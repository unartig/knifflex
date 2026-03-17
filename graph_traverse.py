import jax
import jax.numpy as jnp
from jaxtyping import Bool, Int, Scalar

from game import KniffelState
from genome import DAGGenome
from mask_primitives import reroll_mask_to_action
from rule_primitives import run_binary_op, run_state_primitive


def traverse(genome: DAGGenome, state: KniffelState, max_depth: int) -> Int[Scalar, ""]:
    can_reroll = state.rolls_left.squeeze() > 0

    def is_leaf_viable(child_idx):
        """Returns True if the leaf is a valid action in the current state."""
        is_leaf = child_idx < 0
        leaf_id = jnp.where(is_leaf, -child_idx - 1, 0)

        is_reroll = genome.leaf_is_reroll[leaf_id]
        score_cat = genome.leaf_score_cat[leaf_id]

        # A reroll is viable if we have rolls left.
        # A score is viable if that scorecard slot is empty (-1).
        slot_open = state.scorecard[score_cat] < 0
        viability = jnp.where(is_reroll, can_reroll, slot_open)

        # If it's not a leaf, we return assume the sub-tree has a path
        return jnp.where(is_leaf, viability, jnp.bool_(True))

    def step(carry, _):
        node, done = carry
        safe = jnp.maximum(node, 0)

        # 1. Decision Logic
        rl = genome.rules_left[safe]
        rr = genome.rules_right[safe]
        bop = genome.binary_ops[safe]
        thresh = genome.thresholds[safe]

        signal = run_binary_op(bop, run_state_primitive(rl, state), run_state_primitive(rr, state))
        go_right = signal >= thresh

        lchild = genome.left[safe]
        rchild = genome.right[safe]

        preferred = jnp.where(go_right, rchild, lchild)
        other = jnp.where(go_right, lchild, rchild)

        # 2. Graceful Selection
        # If preferred is viable, take it. Else if other is viable, take it.
        # Otherwise, stay at current node (this signals a stall/failure).
        next_node = jnp.where(is_leaf_viable(preferred), preferred, jnp.where(is_leaf_viable(other), other, node))

        # 3. Termination
        # Finish if we hit a leaf (next_node < 0) or if we are stuck (next_node == node)
        is_stuck = (next_node == node) & (node >= 0)
        hit_leaf = next_node < 0

        new_done = done | hit_leaf | is_stuck
        return (jnp.where(done, node, next_node), new_done), None

    # Run the scan
    (final_node, _), _ = jax.lax.scan(step, (jnp.int32(0), jnp.bool_(False)), None, length=max_depth)

    # 4. Final Fallback Logic
    # If final_node is still >= 0, the traversal failed to find a valid leaf.
    def hard_fallback(_):
        # Pick the first open scorecard category as a last resort
        return jnp.argmax(state.scorecard < 0).astype(jnp.int32) + 32

    def execute_leaf(_):
        leaf_id = -final_node - 1
        return leaf_to_action(genome, leaf_id, state)

    return jax.lax.cond(final_node < 0, execute_leaf, hard_fallback, operand=None)


def leaf_to_action(genome: DAGGenome, leaf_index: Int[Scalar, ""], state: KniffelState) -> Int[Scalar, ""]:
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
