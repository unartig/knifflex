import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, Shaped, jaxtyped
from tensorboardX import SummaryWriter

from game import KniffelState, reset, step
from genome import (
    DAGGenome,
    crossover_genomes,
    max_depth_from_nodes,
    max_reachable_depth,
    mutate_genome,
    random_dag_genome,
)
from graph_traverse import traverse
from rule_primitives import N_STATE_PRIMITIVES
from utils import typechecker

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------

key = jr.key(123)

K = 8  # Survival pressure — larger → more pressure
N_ISLANDS = 4
ISLAND_SIZE = 2**9  # individuals per island (512)
POP_SIZE = N_ISLANDS * ISLAND_SIZE  # 2048 total, same as before
EPISODES = 2048

N_NODES = 512
MAX_DEPTH = max_depth_from_nodes(N_NODES)
TOURNAMENT_SIZE = 4

ISLAND_MU = ISLAND_SIZE // K  # elites kept per island
ISLAND_CHILDREN = ISLAND_SIZE - ISLAND_MU

MIGRATE_EVERY = 25  # epochs between ring migrations
N_MIGRANTS = 4  # individuals exchanged per island pair

EPOCHS = int(1e4)

SIGMA_THRESHOLDS = 0.05  # magnitude of threshold mutations
P_MUTATE_RULES = 0.02  # probability of rule change
P_MUTATE_CONNS = 0.005  # probability of child pointer change
P_MUTATE_BINARY_OPS = 0.05  # probability binary op change
P_CROSSOVER = 0.33  # probability of crossover
P_CROSSOVER_RATIO = 0.5  # single vs subtree crossover


writer = SummaryWriter()

hparams = {
    "K": K,
    "N_ISLANDS": N_ISLANDS,
    "ISLAND_SIZE": ISLAND_SIZE,
    "POP_SIZE": POP_SIZE,
    "EPISODES": EPISODES,
    "N_NODES": N_NODES,
    "TOURNAMENT_SIZE": TOURNAMENT_SIZE,
    "P_MUTATE_RULES": P_MUTATE_RULES,
    "P_MUTATE_CONNS": P_MUTATE_CONNS,
    "P_CROSSOVER": P_CROSSOVER,
    "MIGRATE_EVERY": MIGRATE_EVERY,
    "N_MIGRANTS": N_MIGRANTS,
}
writer.add_hparams(hparams, {}, name="")

# ------------------------------------------------------------
# Policy  (island_size x episodes)
# ------------------------------------------------------------


def policy(population: DAGGenome, states: KniffelState):
    """Map a single island's population over a batch of episodes."""
    return jax.vmap(
        lambda dag, s_batch: jax.vmap(lambda s: traverse(dag, s, MAX_DEPTH))(s_batch),
        in_axes=(0, 0),
    )(population, states)


# ------------------------------------------------------------
# Vectorized episode batch  (works for any leading pop dimension)
# ------------------------------------------------------------


def run_batch(population: DAGGenome, keys: PRNGKeyArray):
    """
    keys : (pop, episodes) PRNGKey array
    Returns rewards (pop, episodes) and final KniffelState.
    """
    pop_size = population.rules_left.shape[0]
    states = jax.vmap(jax.vmap(reset))(keys)
    rewards = jnp.zeros((pop_size, EPISODES), dtype=jnp.int32)

    def body_fn(carry, _):
        states, rewards = carry
        actions = policy(population, states)
        new_states, r = jax.vmap(jax.vmap(step))(states, actions)
        return (new_states, rewards + r), None

    (final_states, rewards), _ = jax.lax.scan(body_fn, (states, rewards), None, length=39)
    return rewards, final_states


# ------------------------------------------------------------
# Island-level crossover
# ------------------------------------------------------------


def crossover_stage_island(
    parents_a: DAGGenome, parents_b: DAGGenome, key: PRNGKeyArray, p_cross: float, p_ratio: float
) -> DAGGenome:
    k1, k2 = jr.split(key)
    cross_keys = jr.split(k1, ISLAND_CHILDREN)
    children_crossed = jax.vmap(crossover_genomes, in_axes=(0, 0, 0, None))(cross_keys, parents_a, parents_b, p_ratio)
    do_cross = jr.bernoulli(k2, p_cross, (ISLAND_CHILDREN,))
    return jtu.tree_map(
        lambda crossed, p_a: jnp.where(do_cross[:, None] if crossed.ndim > 1 else do_cross, crossed, p_a),
        children_crossed,
        parents_a,
    )


# ------------------------------------------------------------
# Island-level evolution step  (no evaluation inside)
# ------------------------------------------------------------


@eqx.filter_jit
def evolve_island(
    population: DAGGenome,
    fitnesses: Float[Array, " ISLAND_SIZE"],
    key: PRNGKeyArray,
):
    k_sel, k_cross, k_mut, k_idx, k_idx1 = jr.split(key, 5)

    # Tournament selection → ISLAND_MU elites
    tournament_keys = jr.split(k_sel, ISLAND_MU)

    def tournament_select_one(k):
        idx = jr.choice(k, ISLAND_SIZE, (TOURNAMENT_SIZE,), replace=False)
        return idx[jnp.argmax(fitnesses[idx])]

    elite_idx = jax.vmap(tournament_select_one)(tournament_keys)
    elites = jtu.tree_map(lambda x: x[elite_idx], population)
    elite_fitnesses = fitnesses[elite_idx]

    # Parent selection from elites (diversity-guaranteed offset)
    idx_a = jr.randint(k_idx, (ISLAND_CHILDREN,), 0, ISLAND_MU)
    offset = jr.randint(k_idx1, (ISLAND_CHILDREN,), 1, ISLAND_MU)
    idx_b = (idx_a + offset) % ISLAND_MU

    parents_a = jtu.tree_map(lambda x: x[idx_a], elites)
    parents_b = jtu.tree_map(lambda x: x[idx_b], elites)

    # Crossover + mutation
    children = crossover_stage_island(parents_a, parents_b, k_cross, P_CROSSOVER, P_CROSSOVER_RATIO)

    mut_keys = jr.split(k_mut, ISLAND_CHILDREN)
    children = jax.vmap(
        lambda g, k: mutate_genome(g, k, SIGMA_THRESHOLDS, P_MUTATE_CONNS, P_MUTATE_RULES, P_MUTATE_BINARY_OPS)
    )(children, mut_keys)

    return children, elites, elite_fitnesses


# ------------------------------------------------------------
# Island-level child evaluation
# ------------------------------------------------------------


@eqx.filter_jit
def evaluate_island(children: DAGGenome, key: PRNGKeyArray):
    """Evaluate ISLAND_CHILDREN individuals on shared episode seeds."""
    ep_keys = jax.vmap(lambda j: jr.fold_in(key, j))(jnp.arange(EPISODES))
    ep_keys_broadcast = jnp.broadcast_to(ep_keys[None], (ISLAND_CHILDREN, EPISODES))
    rewards, states = run_batch(children, ep_keys_broadcast)
    return jnp.mean(rewards, axis=1), states


# ------------------------------------------------------------
# Migration  (ring topology — best leave, worst replaced)
# ------------------------------------------------------------


def migrate(
    pop: DAGGenome,
    fitnesses: Float[Array, "N_ISLANDS ISLAND_SIZE"],
) -> tuple[DAGGenome, Float[Array, "N_ISLANDS ISLAND_SIZE"]]:
    # Identify top / bottom N_MIGRANTS per island
    top_idx = jnp.argsort(fitnesses, axis=1)[:, -N_MIGRANTS:]  # (N_ISLANDS, N_MIGRANTS)
    worst_idx = jnp.argsort(fitnesses, axis=1)[:, :N_MIGRANTS]

    # Gather migrants then rotate one step (ring)
    migrants = jtu.tree_map(lambda x: x[jnp.arange(N_ISLANDS)[:, None], top_idx], pop)
    migrants = jtu.tree_map(lambda x: jnp.roll(x, 1, axis=0), migrants)

    migrant_fitnesses = fitnesses[jnp.arange(N_ISLANDS)[:, None], top_idx]
    migrant_fitnesses = jnp.roll(migrant_fitnesses, 1, axis=0)

    # Inject into worst slots
    pop = jtu.tree_map(
        lambda p, m: p.at[jnp.arange(N_ISLANDS)[:, None], worst_idx].set(m),
        pop,
        migrants,
    )
    fitnesses = fitnesses.at[jnp.arange(N_ISLANDS)[:, None], worst_idx].set(migrant_fitnesses)
    return pop, fitnesses


# ------------------------------------------------------------
# Diversity metric  (works on island-shaped population)
# ------------------------------------------------------------


@eqx.filter_jit
def population_diversity(population: DAGGenome) -> Float[Scalar, ""]:
    flat_rl = population.rules_left.reshape(-1, N_NODES)
    flat_rr = population.rules_right.reshape(-1, N_NODES)
    flat_lc = population.left.reshape(-1, N_NODES)

    def rule_diff(rules):
        mode = jnp.round(jnp.mean(rules.astype(jnp.float32), axis=0)).astype(jnp.int32)
        return jnp.mean(rules != mode[None, :])

    rule_div = rule_diff(flat_rl) + rule_diff(flat_rr)
    conn_div = jnp.mean(flat_lc != jnp.round(jnp.mean(flat_lc, axis=0))[None, :])
    return (rule_div + conn_div) / 2.0


# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------

# Population shape: (N_ISLANDS, ISLAND_SIZE, N_NODES)
all_keys = jr.split(key, N_ISLANDS * ISLAND_SIZE + 1)
pop_keys = all_keys[:-1].reshape(N_ISLANDS, ISLAND_SIZE)  # remove the , -1
key = all_keys[-1]

population = jax.vmap(
    lambda island_keys: jax.vmap(random_dag_genome, in_axes=(0, None, None))(island_keys, N_NODES, N_STATE_PRIMITIVES)
)(pop_keys)


# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------

pop = population
fitnesses = jnp.zeros((N_ISLANDS, ISLAND_SIZE))

for epoch in range(EPOCHS):
    key, k_island, k_eval, k_migrate = jr.split(key, 4)

    # Ring migration every MIGRATE_EVERY epochs
    if epoch % MIGRATE_EVERY == 0 and epoch > 0:
        pop, fitnesses = migrate(pop, fitnesses)

    island_keys = jr.split(k_island, N_ISLANDS)
    eval_keys = jr.split(k_eval, N_ISLANDS)

    # Evolve all islands in parallel
    children, elites, elite_fitnesses = jax.vmap(evolve_island)(pop, fitnesses, island_keys)

    # Evaluate only the new children
    child_fitnesses, states = jax.vmap(evaluate_island)(children, eval_keys)

    # Reassemble
    pop = jtu.tree_map(lambda e, c: jnp.concatenate([e, c], axis=1), elites, children)
    fitnesses = jnp.concatenate([elite_fitnesses, child_fitnesses], axis=1)

    # Best individual across all islands
    flat_fitnesses = fitnesses.reshape(-1)
    best_flat_idx = jnp.argmax(flat_fitnesses)
    best_island = best_flat_idx // ISLAND_SIZE
    best_local = best_flat_idx % ISLAND_SIZE
    best = jtu.tree_map(lambda x: x[best_island, best_local], pop)

    # Logging
    avg_fit = float(jnp.mean(flat_fitnesses))
    max_fit = float(jnp.max(flat_fitnesses))
    avg_rounds = float(jnp.mean(states.round))  # (N_ISLANDS, ISLAND_CHILDREN)
    diversity = float(population_diversity(pop))
    active_nodes = int(best.active_node_count)

    writer.add_scalar("Fitness/Average", avg_fit, epoch)
    writer.add_scalar("Fitness/Best", max_fit, epoch)
    writer.add_scalar("Stats/Average_Rounds", avg_rounds, epoch)
    writer.add_scalar("Stats/Diversity", diversity, epoch)
    writer.add_scalar("Genome/Best_Active_Nodes", active_nodes, epoch)

    print(
        f"[{epoch}] Fit: {avg_fit:.2f} | Best: {max_fit:.2f} | "
        f"Div: {diversity:.3f} | N-Nodes: {active_nodes} | Rounds: {avg_rounds:.2f}"
    )

    if epoch % 100 == 0:
        best_cpu = jax.device_get(best)
        actual_depth = max_reachable_depth(best_cpu)
        print("Actual Depth", actual_depth)
        writer.add_scalar("Genome/Best_Actual_Depth", actual_depth, epoch)

writer.close()
