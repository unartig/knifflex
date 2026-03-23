import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Scalar, jaxtyped

from rule_primitives import N_BINARY_OPS, N_STATE_PRIMITIVES
from utils import typechecker

N_CATEGORIES = 13  # Kniffel scoring categories

# Leaves output a fixed strategy — the tree decides WHICH strategy
# to apply given the current game state.
N_LEAVES = 8

LEAF_STRATEGY_NAMES = [
    "argmax_EV",  # 0
    "argmax_SCORE_NOW",  # 1
    "bonus_chase_or_best",  # 2  (EV_upper if bonus reachable, else argmax_EV)
    "argmax_EV_upper",  # 3
    "argmax_EV_lower",  # 4
    "chase_high_var",  # 5  (straights / Kniffel)
    "safe_lower",  # 6  (FH / pasch / augenzahl)
    "sacrifice",  # 7  (dump worst slot)
]

assert len(LEAF_STRATEGY_NAMES) == N_LEAVES


@jaxtyped(typechecker=typechecker)
class DAGGenome(eqx.Module):
    """Strategy-leaf DAG genome.

    The DAG navigates 36 normalized state signals to one of N_LEAVES
    fixed strategy leaves. Each leaf selects a category via a
    deterministic rule. The oracle then handles reroll vs. score.

    Interpretability: read the tree as
      "IF ev_gap > 0.5 AND rolls_left > 0 THEN argmax_EV
       ELSE IF upper_bonus_gap < 0.3 THEN argmax_EV_upper
       ELSE argmax_SCORE_NOW"
    Every node is a readable binary decision over normalized signals.
    """

    thresholds: Float[Array, " N"]
    rules_left: Int[Array, " N"]  # signal index 0..35
    rules_right: Int[Array, " N"]  # signal index 0..35
    binary_ops: Int[Array, " N"]  # op index 0..5
    left: Int[Array, " N"]  # >=0 → node idx, <0 → leaf -(id+1)
    right: Int[Array, " N"]

    def get_active_mask(self) -> Bool[Array, " N"]:
        n = self.thresholds.shape[0]
        left_valid = (self.left >= 0) & (self.left < n)
        right_valid = (self.right >= 0) & (self.right < n)
        safe_left = jnp.where(left_valid, self.left, 0)
        safe_right = jnp.where(right_valid, self.right, 0)

        def step(reachable, _):
            l = jnp.zeros(n, jnp.bool_).at[safe_left].set(reachable & left_valid)
            r = jnp.zeros(n, jnp.bool_).at[safe_right].set(reachable & right_valid)
            return reachable | l | r, None

        mask, _ = jax.lax.scan(step, jnp.zeros(n, jnp.bool_).at[0].set(True), None, length=n)
        return mask

    @property
    def active_node_count(self) -> Int[Scalar, ""]:
        return jnp.sum(self.get_active_mask())

    @property
    def max_node_count(self) -> int:
        return self.thresholds.shape[0]


# ------------------------------------------------------------------ #
#  Utility                                                            #
# ------------------------------------------------------------------ #


def max_depth_from_nodes(n_nodes: int) -> int:
    return int(np.ceil(np.log2(n_nodes)) * 6)


def max_reachable_depth(genome: DAGGenome) -> int:
    """CPU-side DFS — only call outside the hot loop."""
    n = genome.rules_left.shape[0]
    left = np.array(genome.left)
    right = np.array(genome.right)
    depth = np.zeros(n, dtype=np.int32)
    for i in range(n - 1, -1, -1):
        l, r = left[i], right[i]
        ld = 0 if l < 0 else depth[l] + 1
        rd = 0 if r < 0 else depth[r] + 1
        depth[i] = max(ld, rd)
    return int(depth[0])


# ------------------------------------------------------------------ #
#  Initialisation — perfect binary tree                               #
# ------------------------------------------------------------------ #


def sample_forward_child(key, from_idx, n_nodes, temperature=3.0):
    """Sample a child index > from_idx, biased toward nearby nodes."""
    # Distances to all valid forward targets
    targets = jnp.arange(n_nodes)
    dist = targets - from_idx  # negative or zero = invalid
    valid = dist > 0  # strictly forward only

    # Exponential decay over distance — temperature controls how local
    log_w = -dist.astype(jnp.float32) / temperature
    log_w = jnp.where(valid, log_w, -jnp.inf)
    weights = jax.nn.softmax(log_w)  # (n_nodes,)

    return jr.choice(key, n_nodes, p=weights)


@jaxtyped(typechecker=typechecker)
def random_dag_genome(key: PRNGKeyArray, n_nodes: int, n_rules: int) -> DAGGenome:
    """Perfect binary tree init — all nodes reachable, all leaves covered.

    left[i]  = 2i+1  (internal) if in range, else a leaf strategy
    right[i] = 2i+2  (internal) if in range, else a leaf strategy

    Bottom-level leaf slots are distributed round-robin over N_LEAVES
    with a per-genome random offset, so all 8 strategies appear from
    generation 0 and individuals differ in leaf assignment.

    n_rules kept for API compat; nodes always index N_STATE_PRIMITIVES.
    """
    k1, k2, k3, k4, k5 = jr.split(key, 5)

    rules_left = jr.randint(k1, (n_nodes,), 0, N_STATE_PRIMITIVES).astype(jnp.int8)
    rules_right = jr.randint(k2, (n_nodes,), 0, N_STATE_PRIMITIVES).astype(jnp.int8)
    thresholds = jr.uniform(k3, (n_nodes,))
    binary_ops = jr.randint(k4, (n_nodes,), 0, N_BINARY_OPS).astype(jnp.int8)

    leaf_offset = jr.randint(k5, (), 0, N_LEAVES)
    indices = jnp.arange(n_nodes)

    def make_child(i, child_slot):
        leaf_id = (i + leaf_offset + child_slot) % N_LEAVES
        return jnp.where(
            child_slot < n_nodes,
            child_slot.astype(jnp.int32),
            -(leaf_id + 1).astype(jnp.int32),
        )

    left = jax.vmap(lambda i: make_child(i, 2 * i + 1))(indices).astype(jnp.int8)
    right = jax.vmap(lambda i: make_child(i, 2 * i + 2))(indices).astype(jnp.int8)

    return DAGGenome(
        rules_left=rules_left,
        rules_right=rules_right,
        thresholds=thresholds,
        binary_ops=binary_ops,
        left=left,
        right=right,
    )


# ------------------------------------------------------------------ #
#  Mutation                                                           #
# ------------------------------------------------------------------ #


@jaxtyped(typechecker=typechecker)
def mutate_genome(
    genome: DAGGenome,
    key: PRNGKeyArray,
    p_struct: Float[Scalar, ""],
    p_param: Float[Scalar, ""],
    p_inactive_mult: Float[Scalar, ""],
    sigma: Float[Scalar, ""],
) -> DAGGenome:
    n_nodes = genome.thresholds.shape[0]
    active = genome.get_active_mask()
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12 = jr.split(key, 12)

    struct_p = jnp.where(active, p_struct, p_struct * p_inactive_mult)
    param_p = jnp.where(active, p_param, p_param * p_inactive_mult)
    struct_mask = jr.bernoulli(k1, struct_p, (n_nodes,))
    param_mask = jr.bernoulli(k2, param_p, (n_nodes,))

    # Thresholds
    new_thresholds = jnp.clip(
        jnp.where(param_mask, genome.thresholds + jr.normal(k3, (n_nodes,)) * sigma, genome.thresholds),
        0.0,
        1.0,
    )

    # Rule pointers into the 36-signal space
    new_rl = jnp.where(
        param_mask,
        jr.randint(k4, (n_nodes,), 0, N_STATE_PRIMITIVES).astype(jnp.int8),
        genome.rules_left,
    )
    new_rr = jnp.where(
        param_mask,
        jr.randint(k5, (n_nodes,), 0, N_STATE_PRIMITIVES).astype(jnp.int8),
        genome.rules_right,
    )
    new_ops = jnp.where(
        struct_mask,
        jr.randint(k5, (n_nodes,), 0, N_BINARY_OPS).astype(jnp.int8),
        genome.binary_ops,
    )

    # Structure — left biased toward internal (80%), right toward leaves (80%)
    from_indices = jnp.arange(n_nodes)
    go_leaf_left = jr.bernoulli(k6, 0.8, (n_nodes,))
    go_leaf_right = jr.bernoulli(k7, 0.2, (n_nodes,))
    rand_internal_l = jax.vmap(lambda i, k: sample_forward_child(k, i, n_nodes, temperature=1.312))(
        from_indices, jr.split(k8, n_nodes)
    )
    rand_leaf_l = -(jr.randint(k9, (n_nodes,), 0, N_LEAVES) + 1)
    rand_internal_r = jax.vmap(lambda i, k: sample_forward_child(k, i, n_nodes, temperature=4.0))(
        from_indices, jr.split(k10, n_nodes)
    )
    rand_leaf_r = -(jr.randint(k11, (n_nodes,), 0, N_LEAVES) + 1)

    new_left = jnp.where(
        struct_mask,
        jnp.where(go_leaf_left, rand_leaf_l, rand_internal_l),
        genome.left,
    ).astype(jnp.int8)

    new_right = jnp.where(
        struct_mask,
        jnp.where(go_leaf_right, rand_leaf_r, rand_internal_r),
        genome.right,
    ).astype(jnp.int8)

    new_ops = jnp.where(
        struct_mask,
        jr.randint(k12, (n_nodes,), 0, N_BINARY_OPS).astype(jnp.int8),
        genome.binary_ops,
    )
    return DAGGenome(
        thresholds=new_thresholds,
        rules_left=new_rl,
        rules_right=new_rr,
        binary_ops=new_ops,
        left=new_left,
        right=new_right,
    )


# ------------------------------------------------------------------ #
#  Crossover                                                          #
# ------------------------------------------------------------------ #


def subtree_crossover(parent_a: DAGGenome, parent_b: DAGGenome, key: PRNGKeyArray) -> DAGGenome:
    k1, k2 = jr.split(key)
    n_nodes = parent_a.rules_left.shape[0]
    point_a = jr.randint(k1, (), 1, n_nodes)
    point_b = jr.randint(k2, (), 1, n_nodes)
    offset = point_a - point_b
    indices = jnp.arange(n_nodes)
    take_b = indices >= point_a

    def remap(conn_b, conn_a, mask):
        shifted = jnp.where(conn_b >= 0, conn_b + offset, conn_b)
        in_b_region = (shifted >= point_a) & (shifted < n_nodes)
        return jnp.where(mask, jnp.where(in_b_region, shifted, conn_a), conn_a).astype(conn_a.dtype)

    return DAGGenome(
        thresholds=jnp.where(take_b, parent_b.thresholds, parent_a.thresholds),
        rules_left=jnp.where(take_b, parent_b.rules_left, parent_a.rules_left),
        rules_right=jnp.where(take_b, parent_b.rules_right, parent_a.rules_right),
        binary_ops=jnp.where(take_b, parent_b.binary_ops, parent_a.binary_ops),
        left=remap(parent_b.left, parent_a.left, take_b),
        right=remap(parent_b.right, parent_a.right, take_b),
    )


@jaxtyped(typechecker=typechecker)
def single_point_crossover(parent_a: DAGGenome, parent_b: DAGGenome, key: PRNGKeyArray) -> DAGGenome:
    n_nodes = parent_a.rules_left.shape[0]
    node_mask = jr.bernoulli(key, 0.5, (n_nodes,))

    def mix(a, b):
        return jnp.where(node_mask, a, b)

    return DAGGenome(
        thresholds=mix(parent_a.thresholds, parent_b.thresholds),
        rules_left=mix(parent_a.rules_left, parent_b.rules_left),
        rules_right=mix(parent_a.rules_right, parent_b.rules_right),
        binary_ops=mix(parent_a.binary_ops, parent_b.binary_ops),
        left=mix(parent_a.left, parent_b.left),
        right=mix(parent_a.right, parent_b.right),
    )


@jaxtyped(typechecker=typechecker)
def crossover_genomes(
    key: PRNGKeyArray, parent_a: DAGGenome, parent_b: DAGGenome, p_ratio: Float[Scalar, ""]
) -> DAGGenome:
    """p_ratio kept for call-site compat; modes split 50/50."""
    k1, k2, k3 = jr.split(key, 3)
    use_subtree = jr.bernoulli(k1, 0.5)
    return jax.lax.cond(
        use_subtree,
        lambda k: subtree_crossover(parent_a, parent_b, k),
        lambda k: single_point_crossover(parent_a, parent_b, k),
        jnp.where(use_subtree, k2, k3),
    )
