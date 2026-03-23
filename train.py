import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar
from tensorboardX import SummaryWriter

from game import KniffelState, reset, step
from genome import (
    DAGGenome,
    crossover_genomes,
    max_reachable_depth,
    mutate_genome,
    random_dag_genome,
)
from graph_traverse import traverse
from log import log_game
from rule_primitives import N_STATE_PRIMITIVES
from vizzies import FitnessRecord, dag_to_image, plot_fitness_history, plot_leaf_frequency, plot_signal_heatmap

# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------


# --- Infrastructure ---
SEED = 123
key = jr.key(SEED)
N_ISLANDS = 4
ISLAND_SIZE = 512  # Population per island
EPISODES = 1024  # Number of games for evaluation
EPOCHS = 10000  # Total training generations
MIGRATE_EVERY = 10  # Generations between island migrations
N_MIGRANTS = 16  # Number of top individuals to swap

# --- Genome Architecture ---
N_NODES = 16
MAX_DEPTH = N_NODES - 1  # Max steps in DAG traversal

# --- Evolutionary Pressure ---
SURVIVAL_RATIO = 0.25  # Keep top %
TOURNAMENT_SIZE = 3

# --- Crossover Schedule ---
P_CROSSOVER_START = 0.01  # low early — let structure develop
P_CROSSOVER_PEAK = 0.1  # peak at midpoint — mix diverse individuals
P_CROSSOVER_END = 0.01  # taper late — refine without disruption
CROSSOVER_PEAK_AT = 0.15  # fraction of EPOCHS where peak occurs

# --- Mutation Schedule (Start -> End) ---
# Structural changes (adding/removing logic) should decay
P_STRUCT_START = 0.1
P_STRUCT_END = 0.03

# Parameter changes (thresholds/rule indices) stay relatively steady
P_PARAM_START = 0.15
P_PARAM_END = 0.10

# Mutation noise (Sigma) for thresholds
SIGMA_START = 0.3
SIGMA_END = 0.02

# The "Innovation Driver": Multiplier for mutations on inactive nodes
INACTIVE_MUT_MULT = 10.0

REEVAL_EVERY = 1
TARGET_DIV = 0.1

writer = SummaryWriter()

hparams = {
    "N_ISLANDS": N_ISLANDS,
    "ISLAND_SIZE": ISLAND_SIZE,
    "EPISODES": EPISODES,
    "N_NODES": N_NODES,
    "TOURNAMENT_SIZE": TOURNAMENT_SIZE,
    "P_CROSSOVER_START": P_CROSSOVER_START,
    "P_CROSSOVER_PEAK": P_CROSSOVER_PEAK,
    "P_CROSSOVER_END": P_CROSSOVER_END,
    "CROSSOVER_PEAK_AT": CROSSOVER_PEAK_AT,
    "MIGRATE_EVERY": MIGRATE_EVERY,
    "N_MIGRANTS": N_MIGRANTS,
    "SURVIVAL_RATIO": SURVIVAL_RATIO,
    "INACTIVE_MUT_MULT": INACTIVE_MUT_MULT,
    "P_STRUCT_START": P_STRUCT_START,
    "SIGMA_START": SIGMA_START,
    "TARGET_DIV": TARGET_DIV,
}
writer.add_hparams(hparams, {}, name="")  # ty:ignore[invalid-argument-type]

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


N_ELITES = int(ISLAND_SIZE * SURVIVAL_RATIO)
N_CHILDREN = ISLAND_SIZE - N_ELITES


def _evaluate_one_island(pop: DAGGenome, key: PRNGKeyArray):
    pop_size = pop.rules_left.shape[0]
    ep_keys = jax.vmap(lambda j: jr.fold_in(key, j))(jnp.arange(EPISODES))
    ep_keys_bc = jnp.broadcast_to(ep_keys[None], (pop_size, EPISODES))
    rewards, states = run_batch(pop, ep_keys_bc)

    raw_fitness = jnp.mean(rewards, axis=1)  # (pop_size,)

    # Structural fingerprint: active rules_left
    fingerprints = jax.vmap(lambda g: g.rules_left * g.get_active_mask())(pop).astype(
        jnp.float32
    )  # (pop_size, N_NODES)

    # Each individual's mean hamming distance to the rest of the island
    # roll comparison is O(pop) not O(pop^2)
    dists = jnp.mean(
        jnp.stack(
            [
                jnp.mean(fingerprints != jnp.roll(fingerprints, i, axis=0), axis=1)
                for i in range(1, 9)  # 8 offsets as cheap approximation
            ]
        ),
        axis=0,
    )  # (pop_size,) — high = structurally unique

    # normalize diversity to match fitness scale
    dist_mean = jnp.mean(dists)
    dist_std = jnp.std(dists) + 1e-8

    fit_std = jnp.std(raw_fitness) + 1e-8
    current_div = jnp.mean(dists)  # proxy for island diversity right now
    alpha = jnp.clip(0.3 * (TARGET_DIV - current_div) / TARGET_DIV, 0.0, 0.5)

    # rescale: diversity bonus has same std as fitness, weighted by alpha
    diversity_bonus = ((dists - dist_mean) / dist_std) * fit_std * alpha

    return raw_fitness + diversity_bonus, jnp.std(rewards, axis=1), states


def _evolve_one_island(
    population: DAGGenome,
    fitnesses: Float[Array, " ISLAND_SIZE"],
    key: PRNGKeyArray,
    epoch: Int[Array, ""],
) -> tuple[DAGGenome, Float[Array, " N_ELITES"]]:
    k_sel, k_cross, k_mut = jr.split(key, 3)

    # Selection
    def tournament_select_one(k):
        idx = jr.choice(k, ISLAND_SIZE, (TOURNAMENT_SIZE,), replace=False)
        return idx[jnp.argmax(fitnesses[idx])]

    elite_idx = jax.vmap(tournament_select_one)(jr.split(k_sel, N_ELITES))
    elites = jtu.tree_map(lambda x: x[elite_idx], population)

    # Crossover + mutation → children
    t = epoch.astype(jnp.float32) / EPOCHS
    peak = jnp.float32(CROSSOVER_PEAK_AT)
    p_cross = jnp.where(
        t < peak,
        P_CROSSOVER_START + (P_CROSSOVER_PEAK - P_CROSSOVER_START) * (t / peak),
        P_CROSSOVER_PEAK + (P_CROSSOVER_END - P_CROSSOVER_PEAK) * ((t - peak) / (1.0 - peak)),
    )
    k_idx_a, k_idx_b = jr.split(k_cross)
    idx_a = jr.randint(k_idx_a, (N_CHILDREN,), 0, N_ELITES)
    idx_b = (idx_a + jr.randint(k_idx_b, (N_CHILDREN,), 1, N_ELITES)) % N_ELITES
    parents_a = jtu.tree_map(lambda x: x[idx_a], elites)
    parents_b = jtu.tree_map(lambda x: x[idx_b], elites)
    cross_keys = jr.split(k_cross, N_CHILDREN)
    children = jax.vmap(crossover_genomes, in_axes=(0, 0, 0, None))(
        cross_keys, parents_a, parents_b, jnp.float32(p_cross)
    )
    progress = jnp.clip(1.0 - (epoch.astype(jnp.float32) / EPOCHS), 0.0, 1.0)
    mut_keys = jr.split(k_mut, N_CHILDREN)
    children = jax.vmap(mutate_genome, in_axes=(0, 0, None, None, None, None))(
        children,
        mut_keys,
        P_STRUCT_END + (P_STRUCT_START - P_STRUCT_END) * progress,
        P_PARAM_END + (P_PARAM_START - P_PARAM_END) * progress,
        jnp.float32(INACTIVE_MUT_MULT),
        SIGMA_END + (SIGMA_START - SIGMA_END) * progress,
    )

    # Merge — no fitness carried forward, everything gets evaluated fresh
    new_pop = jtu.tree_map(lambda e, c: jnp.concatenate([e, c], axis=0), elites, children)
    return new_pop  # (ISLAND_SIZE, N_NODES)


@eqx.filter_jit
def train_step(
    pop: DAGGenome,
    fitnesses: Float[Array, "N_ISLANDS ISLAND_SIZE"],
    key: PRNGKeyArray,
    epoch: Int[Array, ""],
) -> tuple:
    k_evolve, k_eval = jr.split(key)
    island_ev_keys = jr.split(k_evolve, N_ISLANDS)
    island_eval_keys = jr.split(k_eval, N_ISLANDS)

    new_pop = jax.vmap(_evolve_one_island, in_axes=(0, 0, 0, None))(pop, fitnesses, island_ev_keys, epoch)

    new_fitnesses, fitness_stds, final_states = jax.vmap(_evaluate_one_island, in_axes=(0, 0))(
        new_pop, island_eval_keys
    )

    return new_pop, new_fitnesses, fitness_stds, final_states


# ------------------------------------------------------------
# Migration  (ring topology — best leave, worst replaced)
# ------------------------------------------------------------


def migrate(
    pop: DAGGenome,
    fitnesses: Float[Array, "N_ISLANDS ISLAND_SIZE"],
    wc_pop: DAGGenome,
    wc_fitnesses: Float[Array, " ISLAND_SIZE"],
) -> tuple[DAGGenome, Float[Array, "N_ISLANDS ISLAND_SIZE"]]:
    """Ring migration between main islands + wildcard donation.

    Main islands: top N_MIGRANTS rotate clockwise into worst N_MIGRANTS slots.
    Wildcard:     donates its top N_MIGRANTS into the *next* worst N_MIGRANTS slots,
                  so ring and wildcard migrants never overwrite each other.
    """
    sorted_idx = jnp.argsort(fitnesses, axis=1)  # ascending

    top_idx = sorted_idx[:, -N_MIGRANTS:]  # best  → will emigrate
    worst_idx = sorted_idx[:, :N_MIGRANTS]  # worst → ring targets
    wc_idx = sorted_idx[:, N_MIGRANTS : 2 * N_MIGRANTS]  # next worst → wildcard targets

    # ── Ring among main islands ──────────────────────────────────────
    ring_migrants = jtu.tree_map(lambda x: x[jnp.arange(N_ISLANDS)[:, None], top_idx], pop)
    ring_migrants = jtu.tree_map(lambda x: jnp.roll(x, 1, axis=0), ring_migrants)
    ring_fit = jnp.roll(fitnesses[jnp.arange(N_ISLANDS)[:, None], top_idx], 1, axis=0)

    pop = jtu.tree_map(
        lambda p, m: p.at[jnp.arange(N_ISLANDS)[:, None], worst_idx].set(m),
        pop,
        ring_migrants,
    )
    fitnesses = fitnesses.at[jnp.arange(N_ISLANDS)[:, None], worst_idx].set(ring_fit)

    # ── Wildcard donation to every main island ───────────────────────
    wc_top_idx = jnp.argsort(wc_fitnesses)[-N_MIGRANTS:]
    wc_migrants = jtu.tree_map(lambda x: x[wc_top_idx], wc_pop)
    wc_fit_vals = wc_fitnesses[wc_top_idx]

    wc_migrants_bc = jtu.tree_map(lambda x: jnp.broadcast_to(x[None], (N_ISLANDS,) + x.shape), wc_migrants)
    wc_fit_bc = jnp.broadcast_to(wc_fit_vals[None], (N_ISLANDS, N_MIGRANTS))

    pop = jtu.tree_map(
        lambda p, m: p.at[jnp.arange(N_ISLANDS)[:, None], wc_idx].set(m),
        pop,
        wc_migrants_bc,
    )
    fitnesses = fitnesses.at[jnp.arange(N_ISLANDS)[:, None], wc_idx].set(wc_fit_bc)

    return pop, fitnesses


# ------------------------------------------------------------
# Diversity metric  (works on island-shaped population)
# ------------------------------------------------------------


@eqx.filter_jit
def population_diversity(population: DAGGenome) -> Float[Scalar, ""]:
    # Only look at active nodes per individual
    active = jax.vmap(DAGGenome.get_active_mask)(
        jtu.tree_map(lambda x: x.reshape(-1, N_NODES), population)
    )  # (N_ISLANDS * ISLAND_SIZE, N_NODES)

    flat_rl = population.rules_left.reshape(-1, N_NODES)
    flat_rr = population.rules_right.reshape(-1, N_NODES)
    flat_lc = population.left.reshape(-1, N_NODES)

    # Zero out inactive nodes so they don't contribute
    rl_a = jnp.where(active, flat_rl, 0)
    rr_a = jnp.where(active, flat_rr, 0)
    lc_a = jnp.where(active, flat_lc, 0)

    # Pairwise Hamming on a random subsample (full pairwise is O(pop^2))
    # Compare each individual against a rolled version of itself
    def hamming(a, b):
        return jnp.mean(a != b, axis=1)  # (pop,) per-individual diff

    rl_div = jnp.mean(hamming(rl_a, jnp.roll(rl_a, 1, axis=0)))
    rr_div = jnp.mean(hamming(rr_a, jnp.roll(rr_a, 1, axis=0)))
    lc_div = jnp.mean(hamming(lc_a, jnp.roll(lc_a, 1, axis=0)))

    return (rl_div + rr_div + lc_div) / 3.0


# ------------------------------------------------------------
# Wildcard island
# ------------------------------------------------------------
# Lives outside the main pop array so train_step shape never changes.
# - Evolves independently each epoch (same train_step, N_ISLANDS=1 slice)
# - Skipped by ring migration entirely
# - Completely reseeded from scratch at every migration event
# - Its best individual can still be picked for the global best log


@eqx.filter_jit
def wildcard_step(
    wc_pop: DAGGenome,
    wc_fitnesses: Float[Array, " ISLAND_SIZE"],
    key: PRNGKeyArray,
    epoch: Int[Array, ""],
) -> tuple[DAGGenome, Float[Array, " ISLAND_SIZE"], KniffelState]:
    """Evolve + evaluate the single wildcard island."""
    k_evolve, k_eval = jr.split(key)
    new_pop = _evolve_one_island(wc_pop, wc_fitnesses, k_evolve, epoch)
    new_fitnesses, _, final_states = _evaluate_one_island(new_pop, k_eval)
    return new_pop, new_fitnesses, final_states


def reseed_wildcard(key: PRNGKeyArray) -> tuple[DAGGenome, Float[Array, " ISLAND_SIZE"]]:
    """Burn the wildcard island to the ground and start fresh."""
    wc_keys = jr.split(key, ISLAND_SIZE)
    wc_pop = jax.vmap(random_dag_genome, in_axes=(0, None, None))(wc_keys, N_NODES, N_STATE_PRIMITIVES)
    wc_fitnesses = jnp.zeros(ISLAND_SIZE)
    return wc_pop, wc_fitnesses


# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------

# Population shape: (N_ISLANDS, ISLAND_SIZE, N_NODES)
all_keys = jr.split(key, N_ISLANDS * ISLAND_SIZE + 1)
pop_keys = all_keys[:-1].reshape(N_ISLANDS, ISLAND_SIZE)
key = all_keys[-1]

population = jax.vmap(
    lambda island_keys: jax.vmap(random_dag_genome, in_axes=(0, None, None))(island_keys, N_NODES, N_STATE_PRIMITIVES)
)(pop_keys)

# Wildcard island — separate array, same size as a normal island
key, k_wc = jr.split(key)
wc_pop, wc_fitnesses = reseed_wildcard(k_wc)


# ------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------

pop = population
fitnesses = jnp.zeros((N_ISLANDS, ISLAND_SIZE))
history: list[FitnessRecord] = []

for epoch in range(EPOCHS):
    key, k_step, k_wc_step, k_reseed, k_log = jr.split(key, 5)

    if epoch % MIGRATE_EVERY == 0 and epoch > 0:
        pop, fitnesses = migrate(pop, fitnesses, wc_pop, wc_fitnesses)
        wc_pop, wc_fitnesses = reseed_wildcard(k_reseed)

    pop, fitnesses, fitness_stds, final_states = train_step(pop, fitnesses, k_step, jnp.int32(epoch))
    # key, k_step, k_wc_step, k_reseed, k_log = jr.split(key, 5)

    # # Migration: wildcard donates its best then reseeds from scratch
    # if epoch % MIGRATE_EVERY == 0 and epoch > 0:
    #     pop, fitnesses = migrate(pop, fitnesses, wc_pop, wc_fitnesses)
    #     wc_pop, wc_fitnesses = reseed_wildcard(k_reseed)

    # # Main islands — single fused GPU dispatch
    # pop, fitnesses, fitness_stds, final_states = train_step(pop, fitnesses, k_step, jnp.int32(epoch))

    # Wildcard — independent, always evolving toward next donation
    wc_pop, wc_fitnesses, wc_states = wildcard_step(wc_pop, wc_fitnesses, k_wc_step, jnp.int32(epoch))

    # --- Logging ---
    flat_fitnesses = fitnesses.reshape(-1)
    wc_best_fit = float(jnp.max(wc_fitnesses))
    avg_fit = float(jnp.mean(flat_fitnesses))
    max_fit = float(jnp.max(flat_fitnesses))
    avg_rounds = float(jnp.mean(final_states.round))

    diversity = float(population_diversity(pop))

    best_flat_idx = int(jnp.argmax(flat_fitnesses))
    best_island = best_flat_idx // ISLAND_SIZE
    best_local = best_flat_idx % ISLAND_SIZE
    best = jtu.tree_map(lambda x: x[best_island, best_local], pop)
    avg_std = float(jnp.mean(fitness_stds))
    max_std = float(jnp.max(fitness_stds))

    writer.add_scalar("Fitness/Average", avg_fit, epoch)
    writer.add_scalar("Fitness/Best", max_fit, epoch)
    writer.add_scalar("Fitness/Wildcard_Best", wc_best_fit, epoch)
    writer.add_scalar("Stats/Average_Rounds", avg_rounds, epoch)
    writer.add_scalar("Stats/Diversity", diversity, epoch)
    writer.add_scalar("Fitness/Avg_Std", avg_std, epoch)
    writer.add_scalar("Fitness/Max_Std", max_std, epoch)
    print(
        f"[{epoch}] Fit: {avg_fit:.2f} ±{avg_std:.1f} | Best: {max_fit:.2f} | "
        f"WC: {wc_best_fit:.2f} | Div: {diversity:.3f} | Rounds: {avg_rounds:.2f}"
    )

    if epoch % 10 == 0:
        active_nodes = int(best.active_node_count)
        best_cpu = jax.device_get(best)
        actual_depth = max_reachable_depth(best_cpu)
        print(
            f"Active Nodes {active_nodes} | Depth {actual_depth}",
        )
        writer.add_scalar("Genome/Best_Actual_Depth", actual_depth, epoch)
        writer.add_scalar("Genome/Best_Active_Nodes", active_nodes, epoch)

    # DAG image every 20 epochs
    if epoch % 20 == 0:
        best_jax = jax.device_put(best_cpu)
        log_game(lambda s: traverse(best_jax, s, MAX_DEPTH), k_log)  # lean by default

        img = dag_to_image(
            best_cpu,
            title=f"Best DAG — epoch {epoch}  fit={max_fit:.1f}",
        )
        # TensorBoard wants (C, H, W) uint8
        writer.add_image("Genome/Best_DAG", img.transpose(2, 0, 1), epoch)

writer.close()
