import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, Shaped, jaxtyped
from tensorboardX import SummaryWriter

from game import KniffelState, reset, step
from genome import DAGGenome, crossover_genomes, mutate_genome, random_dag_genome
from graph_traverse import traverse
from rule_primitives import N_STATE_PRIMITIVES
from utils import typechecker

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------

key = jr.key(123)

K = 4  # Survival pressure, larger -> more pressure
POP_SIZE = 2**11
EPISODES = 512


N_NODES = 512
MU = POP_SIZE // K
LAMBDA = MU * K

NUM_CHILDREN = POP_SIZE - MU

EPOCHS = int(1e4)


SIGMA_THRESHOLDS = 0.1  # magnitude of threshold mutations
P_MUTATE_RULES = 0.0075  # probability of rule change
P_MUTATE_CONNS = 0.005  # probability of child pointer change
P_MUTATE_BINARY_OPS = 0.05  # probability binary op change
P_CROSSOVER = 0.33  # probability of parent crossover
P_CROSSOVER_RATIO = 0.5  # single vs subtree crossover


# ------------------------------------------------------------
# Policy (population x episodes)
# ------------------------------------------------------------
@jaxtyped(typechecker=typechecker)
def policy(population: DAGGenome, states: KniffelState) -> Int[Array, "POP_SIZE EPISODES"]:
    actions = jax.vmap(
        lambda dag, s_batch: jax.vmap(lambda s: traverse(dag, s))(s_batch),
        in_axes=(0, 0),
    )(population, states)

    return actions  # shape (POP_SIZE, EPISODES_PER_IND)


# ------------------------------------------------------------
# Vectorized Episode Batch
# ------------------------------------------------------------
@jaxtyped(typechecker=typechecker)
def run_batch(
    population: DAGGenome, keys: Shaped[PRNGKeyArray, "POP_SIZE EPISODES"]
) -> tuple[Int[Array, "POP_SIZE EPISODES"], KniffelState]:
    states = jax.vmap(jax.vmap(reset))(keys)

    rewards = jnp.zeros((POP_SIZE, EPISODES), dtype=jnp.int32)

    def cond_fn(carry):
        states, rewards = carry
        # Any state not done?
        return jnp.logical_not(jnp.all(states.done))

    def body_fn(carry):
        states, rewards = carry

        # Get actions for all states
        actions = policy(population, states)  # shape: (POP_SIZE, EPISODES_PER_IND)

        # Step all states
        new_states, r = jax.vmap(jax.vmap(step))(states, actions)

        rewards = rewards + r  # accumulate

        return new_states, rewards

    # Run until all states are done
    final_states, rewards = jax.lax.while_loop(cond_fn, body_fn, (states, rewards))

    return rewards, final_states  # shape: (POP_SIZE, EPISODES_PER_IND)


# ------------------------------------------------------------
# Population Fitness
# ------------------------------------------------------------


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def evaluate_population(population: DAGGenome, key: PRNGKeyArray) -> tuple[Float[Array, " POP_SIZE"], KniffelState]:

    keys = jr.split(key, POP_SIZE * EPISODES)
    keys = keys.reshape(POP_SIZE, EPISODES)

    rewards, states = run_batch(population, keys)

    fitnesses = jnp.mean(rewards, axis=1)

    return fitnesses, states


# ------------------------------------------------------------
# Evolution Step
# ------------------------------------------------------------


@jaxtyped(typechecker=typechecker)
def crossover_stage(
    parents_a: DAGGenome, parents_b: DAGGenome, key: PRNGKeyArray, p_cross: float, p_ratio: float
) -> DAGGenome:
    k1, k2 = jr.split(key)

    cross_keys = jr.split(k1, NUM_CHILDREN)
    children_crossed = jax.vmap(crossover_genomes, in_axes=(0, 0, 0, None))(cross_keys, parents_a, parents_b, p_ratio)

    do_cross = jr.bernoulli(k2, p_cross, (NUM_CHILDREN,))

    return jtu.tree_map(
        lambda crossed, p_a: jnp.where(do_cross[:, None] if crossed.ndim > 1 else do_cross, crossed, p_a),
        children_crossed,
        parents_a,
    )


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def evolve(
    population: DAGGenome, key: PRNGKeyArray
) -> tuple[DAGGenome, Float[Array, " POP_SIZE"], KniffelState, DAGGenome]:
    k_eval, k_cross, k_mut, k_idx = jr.split(key, 4)

    # 1. Evaluation
    fitnesses, states = evaluate_population(population, k_eval)
    elite_idx = jnp.argsort(fitnesses)[-MU:]
    elites = jtu.tree_map(lambda x: x[elite_idx], population)

    # 2. Parent Selection (Diversity Guaranteed)
    idx_a = jr.randint(k_idx, (NUM_CHILDREN,), 0, MU)
    offset = jr.randint(k_idx, (NUM_CHILDREN,), 1, MU)
    idx_b = (idx_a + offset) % MU

    parents_a = jtu.tree_map(lambda x: x[idx_a], elites)
    parents_b = jtu.tree_map(lambda x: x[idx_b], elites)

    # 3. Crossover with Probability
    children = crossover_stage(parents_a, parents_b, k_cross, P_CROSSOVER, P_CROSSOVER_RATIO)

    # 4. Mutation (using jnp.where instead of lax.cond)
    mut_keys = jr.split(k_mut, NUM_CHILDREN)
    children = jax.vmap(
        lambda g, k: mutate_genome(
            g,
            k,
            SIGMA_THRESHOLDS,
            P_MUTATE_CONNS,
            P_MUTATE_RULES,
            P_MUTATE_BINARY_OPS,
        )
    )(children, mut_keys)

    # Return everything needed for training and logging
    new_pop = jtu.tree_map(lambda e, c: jnp.concatenate([e, c], axis=0), elites, children)

    # Pass the best individual back for external analysis
    best_dag = jtu.tree_map(lambda x: x[elite_idx[-1]], population)

    return new_pop, fitnesses, states, best_dag


def population_diversity(population: DAGGenome) -> Float[Scalar, ""]:
    rules = population.rules  # (POP_SIZE, N_NODES)
    # Compare each individual to the mean rule (cheap approximation)
    mode_rules = jnp.round(jnp.mean(rules.astype(jnp.float32), axis=0)).astype(jnp.int32)
    diffs = jnp.mean(rules != mode_rules[None, :], axis=1)
    return jnp.mean(diffs)


# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------

keys = jr.split(key, POP_SIZE + 1)
pop_keys, key = keys[:-1], keys[-1]

population = jax.vmap(random_dag_genome, in_axes=(0, None, None))(
    pop_keys,
    N_NODES,
    N_STATE_PRIMITIVES,
)

writer = SummaryWriter()

hparams = {
    "K": K,
    "POP_SIZE": POP_SIZE,
    "EPISODES": EPISODES,
    "N_NODES": N_NODES,
    "P_MUTATE_RULES": P_MUTATE_RULES,
    "P_MUTATE_CONNS": P_MUTATE_CONNS,
    "P_CROSSOVER": P_CROSSOVER,
}
# Note: add_hparams often requires a dummy metric to display correctly in TB
writer.add_hparams(hparams, {}, name="")

# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------

pop = population

for epoch in range(EPOCHS):
    key, subkey = jr.split(key)

    pop, fitnesses, states, best = evolve(pop, subkey)

    avg_fit = float(jnp.mean(fitnesses))
    max_fit = float(jnp.max(fitnesses))
    avg_rounds = float(jnp.mean(states.round))
    diversity = float(population_diversity(pop))
    active_nodes = int(best.active_node_count)
    # TensorBoard Logging
    writer.add_scalar("Fitness/Average", avg_fit, epoch)
    writer.add_scalar("Fitness/Best", max_fit, epoch)
    writer.add_scalar("Stats/Average_Rounds", avg_rounds, epoch)
    writer.add_scalar("Stats/Diversity", diversity, epoch)
    writer.add_scalar("Genome/Best_Active_Nodes", active_nodes, epoch)

    # Keep the debug print for console visibility
    print(
        f"[{epoch}] Fit: {avg_fit:.2f} | Best: {max_fit:.2f} | Div: {diversity:.3f} | N-Nodes: {active_nodes} | Rounds: {avg_rounds:.2f}"
    )

writer.close()
