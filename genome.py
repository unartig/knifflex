import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, Scalar, jaxtyped

from mask_primitives import N_MASK_BOOL_OPS, N_MASK_PRIMITIVES
from rule_primitives import N_BINARY_OPS, N_STATE_PRIMITIVES
from utils import typechecker

# ------------------------------------------------------------
# Leaf layout  (same count as before the composite explosion)
# ------------------------------------------------------------
# Reroll leaves : N_MASK_PRIMITIVES
# Score  leaves : N_STATE_PRIMITIVES
#
# Each reroll leaf carries THREE evolvable fields:
#   leaf_mask_left  — left  mask primitive id
#   leaf_mask_right — right mask primitive id
#   leaf_mask_op    — MaskBoolOp id
#
# At initialisation left==right==pid and op==OR, so every leaf degenerates to a
# plain atomic mask — zero behaviour change on day one.  Evolution discovers
# useful compositions by mutating these fields just like it mutates node rules.

N_REROLL_LEAVES = N_MASK_PRIMITIVES
N_SCORE_LEAVES = 13
N_LEAVES = N_REROLL_LEAVES + N_SCORE_LEAVES


@jaxtyped(typechecker=typechecker)
class DAGGenome(eqx.Module):
    # Decision-node arrays  (length N)
    thresholds: Float[Array, " N"]
    rules_left: Int[Array, " N"]
    rules_right: Int[Array, " N"]
    binary_ops: Int[Array, " N"]
    left: Int[Array, " N"]
    right: Int[Array, " N"]
    # Leaf descriptor arrays  (length L = N_LEAVES)
    leaf_is_reroll: Bool[Array, " L"]  # True → reroll, False → score
    leaf_mask_left: Int[Array, " L"]  # left  mask primitive id  (reroll only)
    leaf_mask_right: Int[Array, " L"]  # right mask primitive id  (reroll only)
    leaf_mask_op: Int[Array, " L"]  # MaskBoolOp id            (reroll only)
    leaf_score_cat: Int[Array, " L"]  # scoring category 0-12    (score only)

    def get_active_mask(self) -> Bool[Array, " N"]:
        n_nodes = self.rules_left.shape[0]

        def step(reachable, _):
            active_left = jnp.where(reachable & (self.left >= 0), self.left, n_nodes)
            active_right = jnp.where(reachable & (self.right >= 0), self.right, n_nodes)
            new_reachable = jnp.zeros(n_nodes + 1, dtype=jnp.bool_)
            new_reachable = new_reachable.at[active_left].set(True)
            new_reachable = new_reachable.at[active_right].set(True)
            new_reachable = new_reachable.at[0].set(True)
            return new_reachable[:n_nodes] | reachable, None

        init = jnp.zeros(n_nodes, dtype=jnp.bool_).at[0].set(True)
        mask, _ = jax.lax.scan(step, init, None, length=n_nodes)
        return mask

    @property
    def active_node_count(self) -> Int[Scalar, ""]:
        return jnp.sum(self.get_active_mask())

    @property
    def max_node_count(self) -> int:
        return self.thresholds.shape[0]


# ------------------------------------------------------------
# Leaf table construction
# ------------------------------------------------------------
def max_depth_from_nodes(n_nodes: int) -> int:
    # O(log N) expected, add generous margin for worst-case evolved chains
    # Empirically safe for random + mutated DAGs; verify on your population
    return int(np.ceil(np.log2(n_nodes)) * 6)


def max_reachable_depth(genome: DAGGenome) -> int:
    """CPU-side BFS/DFS — run once on a sample, not in the hot loop."""
    n = genome.rules_left.shape[0]
    left = np.array(genome.left)
    right = np.array(genome.right)

    depth = np.zeros(n, dtype=np.int32)
    for i in range(n):
        l, r = left[i], right[i]
        ld = 0 if l < 0 else depth[l] + 1
        rd = 0 if r < 0 else depth[r] + 1
        depth[i] = max(ld, rd)
    return int(depth[0])  # depth of root


@jaxtyped(typechecker=typechecker)
def make_leaves() -> tuple[
    Bool[Array, " L"],
    Int[Array, " L"],
    Int[Array, " L"],
    Int[Array, " L"],
    Int[Array, " L"],
]:
    is_reroll = [True] * N_REROLL_LEAVES + [False] * N_SCORE_LEAVES
    mask_left = list(range(N_REROLL_LEAVES)) + [0] * N_SCORE_LEAVES
    mask_right = list(range(N_REROLL_LEAVES)) + [0] * N_SCORE_LEAVES
    mask_op = [1] * N_REROLL_LEAVES + [0] * N_SCORE_LEAVES  # 1=OR; a|a == a
    score_cat = [0] * N_REROLL_LEAVES + list(range(N_SCORE_LEAVES))
    return (
        jnp.array(is_reroll, dtype=jnp.bool_),
        jnp.array(mask_left, dtype=jnp.int32),
        jnp.array(mask_right, dtype=jnp.int32),
        jnp.array(mask_op, dtype=jnp.int32),
        jnp.array(score_cat, dtype=jnp.int32),
    )


# ------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------


@jaxtyped(typechecker=typechecker)
def random_dag_genome(key: PRNGKeyArray, n_nodes: int, n_rules: int) -> DAGGenome:
    k1, k2, k3, k4, k5 = jr.split(key, 5)

    rules_left = jr.randint(k1, (n_nodes,), 0, n_rules)
    rules_right = jr.randint(k2, (n_nodes,), 0, n_rules)
    rules = jr.randint(k1, (n_nodes,), 0, n_rules)
    thresholds = jr.uniform(k3, (n_nodes,))
    binary_ops = jr.randint(k4, (n_nodes,), 0, N_BINARY_OPS)

    child_keys = jr.split(k5, n_nodes * 2)
    indices = jnp.arange(n_nodes)

    def random_child_v(key_i):
        key, i = key_i
        total = (n_nodes - i - 1) + N_LEAVES
        choice = jr.randint(key, (), 0, total)
        return jax.lax.cond(
            choice < (n_nodes - i - 1),
            lambda _: i + 1 + choice,
            lambda _: -(choice - (n_nodes - i - 1) + 1),
            operand=None,
        )

    left = jax.vmap(random_child_v)((child_keys[:n_nodes], indices))
    right = jax.vmap(random_child_v)((child_keys[n_nodes:], indices))

    leaf_is_reroll, leaf_mask_left, leaf_mask_right, leaf_mask_op, leaf_score_cat = make_leaves()

    return DAGGenome(
        rules_left=rules_left,
        rules_right=rules_right,
        thresholds=thresholds,
        binary_ops=binary_ops,
        left=left,
        right=right,
        leaf_is_reroll=leaf_is_reroll,
        leaf_mask_left=leaf_mask_left,
        leaf_mask_right=leaf_mask_right,
        leaf_mask_op=leaf_mask_op,
        leaf_score_cat=leaf_score_cat,
    )


# ------------------------------------------------------------
# Mutation
# ------------------------------------------------------------


@jaxtyped(typechecker=typechecker)
def mutate_thresholds(key: PRNGKeyArray, thresholds: Float[Array, " N"], sigma: float) -> Float[Array, " N"]:
    return jnp.clip(thresholds + jr.normal(key, thresholds.shape) * sigma, 0.0, 1.0)


@jaxtyped(typechecker=typechecker)
def mutate_rules(key: PRNGKeyArray, rules: Int[Array, " N"], n_rules: int, p: float) -> Int[Array, " N"]:
    mask = jr.bernoulli(key, p, rules.shape)
    return jnp.where(mask, jr.randint(key, rules.shape, 0, n_rules), rules)


@jaxtyped(typechecker=typechecker)
def mutate_binary_ops(key: PRNGKeyArray, ops: Int[Array, " N"], p: float) -> Int[Array, " N"]:
    mask = jr.bernoulli(key, p, ops.shape)
    return jnp.where(mask, jr.randint(key, ops.shape, 0, N_BINARY_OPS), ops)


@jaxtyped(typechecker=typechecker)
def mutate_connections(
    key: PRNGKeyArray, conn: Int[Array, " N"], n_nodes: int, n_leaves: int, p: float
) -> Int[Array, " N"]:
    k1, k2 = jr.split(key)
    should_mutate = jr.bernoulli(k1, p, conn.shape)
    indices = jnp.arange(conn.shape[0])
    mut_keys = jr.split(k2, conn.shape[0])

    def mutate_one(
        key: PRNGKeyArray, conn_val: Int[Scalar, ""], idx: Int[Scalar, ""], do_mutate: Bool[Scalar, ""]
    ) -> Int[Scalar, ""]:
        total = (n_nodes - idx - 1) + n_leaves
        choice = jr.randint(key, (), 0, total)
        new_target = jnp.where(
            choice < (n_nodes - idx - 1),
            idx + 1 + choice,
            -(choice - (n_nodes - idx - 1) + 1),
        )
        return jnp.where(do_mutate, new_target, conn_val)

    return jax.vmap(mutate_one)(mut_keys, conn, indices, should_mutate)


@jaxtyped(typechecker=typechecker)
def mutate_leaf_mask_field(key: PRNGKeyArray, field: Int[Array, " L"], n_vals: int, p: float) -> Int[Array, " L"]:
    mask = jr.bernoulli(key, p, field.shape)
    return jnp.where(mask, jr.randint(key, field.shape, 0, n_vals), field)


@jaxtyped(typechecker=typechecker)
def mutate_genome(
    dag: DAGGenome,
    key: PRNGKeyArray,
    sigma_thresh: float,
    p_connections: float,
    p_rules: float,
    p_binary_ops: float,
    p_mask_fields: float = 0.02,
) -> DAGGenome:
    k1, k2, k3, k4, k5, k6, k7, k8, k9 = jr.split(key, 9)
    return DAGGenome(
        thresholds=mutate_thresholds(k1, dag.thresholds, sigma_thresh),
        rules_left=mutate_rules(k2, dag.rules_left, N_STATE_PRIMITIVES, p_rules),
        rules_right=mutate_rules(k3, dag.rules_right, N_STATE_PRIMITIVES, p_rules),
        binary_ops=mutate_binary_ops(k4, dag.binary_ops, p_binary_ops),
        left=mutate_connections(k5, dag.left, dag.rules_left.shape[0], N_LEAVES, p_connections),
        right=mutate_connections(k6, dag.right, dag.rules_right.shape[0], N_LEAVES, p_connections),
        leaf_is_reroll=dag.leaf_is_reroll,
        leaf_mask_left=mutate_leaf_mask_field(k7, dag.leaf_mask_left, N_MASK_PRIMITIVES, p_mask_fields),
        leaf_mask_right=mutate_leaf_mask_field(k8, dag.leaf_mask_right, N_MASK_PRIMITIVES, p_mask_fields),
        leaf_mask_op=mutate_leaf_mask_field(k9, dag.leaf_mask_op, N_MASK_BOOL_OPS, p_mask_fields),
        leaf_score_cat=dag.leaf_score_cat,
    )


# ------------------------------------------------------------
# Crossover
# ------------------------------------------------------------


def subtree_crossover(parent_a: DAGGenome, parent_b: DAGGenome, key: PRNGKeyArray) -> DAGGenome:
    k1, k2 = jr.split(key)
    n_nodes = parent_a.rules_left.shape[0]
    point_a = jr.randint(k1, (), 1, n_nodes)
    point_b = jr.randint(k2, (), 1, n_nodes)
    offset = point_a - point_b
    indices = jnp.arange(n_nodes)
    take_from_b = indices >= point_a

    def remap_conn(conn_b, conn_a, take_mask):
        shifted = jnp.where(conn_b >= 0, conn_b + offset, conn_b)
        valid = (shifted > indices) & (shifted < n_nodes)
        return jnp.where(take_mask, jnp.where(valid, shifted, conn_a), conn_a)

    return DAGGenome(
        thresholds=jnp.where(take_from_b, parent_b.thresholds, parent_a.thresholds),
        rules_left=jnp.where(take_from_b, parent_b.rules_left, parent_a.rules_left),
        rules_right=jnp.where(take_from_b, parent_b.rules_right, parent_a.rules_right),
        binary_ops=jnp.where(take_from_b, parent_b.binary_ops, parent_a.binary_ops),
        left=remap_conn(parent_b.left, parent_a.left, take_from_b),
        right=remap_conn(parent_b.right, parent_a.right, take_from_b),
        leaf_is_reroll=parent_a.leaf_is_reroll,
        leaf_mask_left=parent_a.leaf_mask_left,
        leaf_mask_right=parent_a.leaf_mask_right,
        leaf_mask_op=parent_a.leaf_mask_op,
        leaf_score_cat=parent_a.leaf_score_cat,
    )


@jaxtyped(typechecker=typechecker)
def single_point_crossover(parent_a: DAGGenome, parent_b: DAGGenome, key: PRNGKeyArray) -> DAGGenome:
    k1, k2 = jr.split(key)
    n_nodes = parent_a.rules_left.shape[0]
    node_mask = jr.bernoulli(k1, 0.5, (n_nodes,))
    leaf_mask = jr.bernoulli(k2, 0.5, (N_LEAVES,))

    def mix_n(a, b):
        return jnp.where(node_mask, a, b)

    def mix_l(a, b):
        return jnp.where(leaf_mask, a, b)

    return DAGGenome(
        thresholds=mix_n(parent_a.thresholds, parent_b.thresholds),
        rules_left=mix_n(parent_a.rules_left, parent_b.rules_left),
        rules_right=mix_n(parent_a.rules_right, parent_b.rules_right),
        binary_ops=mix_n(parent_a.binary_ops, parent_b.binary_ops),
        left=mix_n(parent_a.left, parent_b.left),
        right=mix_n(parent_a.right, parent_b.right),
        leaf_is_reroll=parent_a.leaf_is_reroll,
        leaf_mask_left=mix_l(parent_a.leaf_mask_left, parent_b.leaf_mask_left),
        leaf_mask_right=mix_l(parent_a.leaf_mask_right, parent_b.leaf_mask_right),
        leaf_mask_op=mix_l(parent_a.leaf_mask_op, parent_b.leaf_mask_op),
        leaf_score_cat=parent_a.leaf_score_cat,
    )


@jaxtyped(typechecker=typechecker)
def crossover_genomes(key: PRNGKeyArray, parent_a: DAGGenome, parent_b: DAGGenome, p_ratio: float) -> DAGGenome:
    k1, k2, k3 = jr.split(key, 3)
    use_subtree = jr.bernoulli(k1, p_ratio)
    return jax.lax.cond(
        use_subtree,
        lambda k: subtree_crossover(parent_a, parent_b, k),
        lambda k: single_point_crossover(parent_a, parent_b, k),
        jnp.where(use_subtree, k2, k3),
    )
