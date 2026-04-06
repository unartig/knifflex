import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tensorboardX import SummaryWriter

from cereal import save_genome
from game import KniffelState, reset, step
from log import log_game
from w_genome import WGenome, crossover_w, mutate_w, population_get, random_w_population, traverse_w

# ------------------------------------------------------------------ #
#  Hyperparameters                                                    #
# ------------------------------------------------------------------ #

SEED = 123
N_ISLANDS = 4
ISLAND_SIZE = 128
EPISODES = 1024
EPOCHS = 30_000
MIGRATE_EVERY = 20
N_MIGRANTS = 4

SURVIVAL_RATIO = 0.25
TOURNAMENT_SIZE = 3

# Mutation schedule
SIGMA_START = 0.3
SIGMA_END = 0.02
P_RESET_START = 0.02
P_RESET_END = 0.001

P_CROSSOVER = 0.5

N_ELITES = int(ISLAND_SIZE * SURVIVAL_RATIO)
N_CHILDREN = ISLAND_SIZE - N_ELITES

writer = SummaryWriter(comment="_w_genome")

# ------------------------------------------------------------------ #
#  Episode runner                                                     #
# ------------------------------------------------------------------ #


def policy(population: WGenome, states: KniffelState):
    return jax.vmap(
        lambda w, s_batch: jax.vmap(lambda s: traverse_w(w, s))(s_batch),
        in_axes=(0, 0),
    )(population, states)


def run_batch(population: WGenome, keys: PRNGKeyArray) -> tuple[Float[Array, ""], KniffelState]:
    pop_size = population.W.shape[0]
    states = jax.vmap(jax.vmap(reset))(keys)
    rewards = jnp.zeros((pop_size, EPISODES), dtype=jnp.int32)

    def body_fn(carry, _):
        states, rewards = carry
        actions = policy(population, states)
        new_states, r = jax.vmap(jax.vmap(step))(states, actions)
        return (new_states, rewards + r), None

    (final_states, rewards), _ = jax.lax.scan(body_fn, (states, rewards), None, length=39)
    return rewards, final_states


# ------------------------------------------------------------------ #
#  Island evolution                                                  #
# ------------------------------------------------------------------ #


def _evaluate_one_island(pop: WGenome, key: PRNGKeyArray):
    pop_size = pop.W.shape[0]
    ep_keys = jax.vmap(lambda j: jr.fold_in(key, j))(jnp.arange(EPISODES))
    ep_keys_bc = jnp.broadcast_to(ep_keys[None], (pop_size, EPISODES))
    rewards, states = run_batch(pop, ep_keys_bc)
    return jnp.mean(rewards, axis=1), jnp.std(rewards, axis=1), states


def _evolve_one_island(
    pop: WGenome,
    fitnesses: Float[Array, " ISLAND_SIZE"],
    key: PRNGKeyArray,
    epoch: Int[Array, ""],
) -> WGenome:
    k_sel, k_cross_mask, k_cross_op, k_mut = jr.split(key, 4)

    # 1. Selection
    def select_one(k):
        idx = jr.choice(k, ISLAND_SIZE, (TOURNAMENT_SIZE,), replace=False)
        return idx[jnp.argmax(fitnesses[idx])]

    elite_idx = jax.vmap(select_one)(jr.split(k_sel, N_ELITES))
    elites = jtu.tree_map(lambda x: x[elite_idx], pop)

    # 2. Prepare Parents
    idx_a = jr.randint(jr.split(k_cross_op)[0], (N_CHILDREN,), 0, N_ELITES)
    idx_b = (idx_a + jr.randint(jr.split(k_cross_op)[1], (N_CHILDREN,), 1, N_ELITES)) % N_ELITES
    pa = jtu.tree_map(lambda x: x[idx_a], elites)
    pb = jtu.tree_map(lambda x: x[idx_b], elites)

    # 3. Conditional Crossover
    crossed_children = jax.vmap(crossover_w)(pa, pb, jr.split(k_cross_op, N_CHILDREN))

    do_cross_mask = jr.bernoulli(k_cross_mask, P_CROSSOVER, (N_CHILDREN,))

    children = jtu.tree_map(
        lambda crossed, original: jnp.where(do_cross_mask[:, None, None], crossed, original), crossed_children, pa
    )

    # 4. Mutation (Applied to all children regardless of crossover)
    progress = jnp.clip(1.0 - epoch.astype(jnp.float32) / EPOCHS, 0.0, 1.0)
    sigma = SIGMA_END + (SIGMA_START - SIGMA_END) * progress
    p_reset = P_RESET_END + (P_RESET_START - P_RESET_END) * progress

    children = jax.vmap(mutate_w, in_axes=(0, 0, None, None))(children, jr.split(k_mut, N_CHILDREN), sigma, p_reset)

    return jtu.tree_map(
        lambda e, c: jnp.concatenate([e, c], axis=0),
        elites,
        children,
    )


@eqx.filter_jit()
def train_step(pop, fitnesses, key, epoch):
    def one_island(args):
        p, f, k = args
        new_p = _evolve_one_island(p, f, k, epoch)
        k_eval = jr.fold_in(k, 999)
        new_f, std, st = _evaluate_one_island(new_p, k_eval)
        return new_p, new_f, std, st

    keys = jr.split(key, N_ISLANDS)
    new_pop, new_fit, stds, states = jax.vmap(one_island)((pop, fitnesses, keys))
    return new_pop, new_fit, stds, states


# ------------------------------------------------------------------ #
#  Diversity                                                          #
# ------------------------------------------------------------------ #


@eqx.filter_jit()
def population_diversity(pop: WGenome) -> Float[Array, ""]:
    """Mean std of W entries across individuals — high means spread out."""
    flat = pop.W.reshape(pop.W.shape[0], -1)  # (pop, 182)
    return jnp.mean(jnp.std(flat, axis=0))


@eqx.filter_jit()
def island_diversities(pop: WGenome) -> Float[Array, " N_ISLANDS"]:
    """Per-island diversity over (N_ISLANDS, ISLAND_SIZE, 13, 14) pop."""
    return jax.vmap(population_diversity)(pop)


# ------------------------------------------------------------------ #
#  Wildcard                                                           #
# ------------------------------------------------------------------ #


def reseed_wildcard(key):
    return random_w_population(key, ISLAND_SIZE), jnp.zeros(ISLAND_SIZE)


@eqx.filter_jit()
def wildcard_step(wc_pop, wc_fit, key, epoch):
    k_ev, k_evo = jr.split(key)
    new_pop = _evolve_one_island(wc_pop, wc_fit, k_evo, epoch)
    new_fit, _, states = _evaluate_one_island(new_pop, k_ev)
    return new_pop, new_fit, states


def migrate(pop, fitnesses, wc_pop, wc_fitnesses):
    sorted_idx = jnp.argsort(fitnesses, axis=1)
    top_idx = sorted_idx[:, -N_MIGRANTS:]
    worst_idx = sorted_idx[:, :N_MIGRANTS]
    wc_idx = sorted_idx[:, N_MIGRANTS : 2 * N_MIGRANTS]

    ring = jtu.tree_map(lambda x: x[jnp.arange(N_ISLANDS)[:, None], top_idx], pop)
    ring = jtu.tree_map(lambda x: jnp.roll(x, 1, axis=0), ring)
    ring_fit = jnp.roll(fitnesses[jnp.arange(N_ISLANDS)[:, None], top_idx], 1, axis=0)

    pop = jtu.tree_map(lambda p, m: p.at[jnp.arange(N_ISLANDS)[:, None], worst_idx].set(m), pop, ring)
    fitnesses = fitnesses.at[jnp.arange(N_ISLANDS)[:, None], worst_idx].set(ring_fit)

    wc_top = jnp.argsort(wc_fitnesses)[-N_MIGRANTS:]
    wc_m = jtu.tree_map(lambda x: x[wc_top], wc_pop)
    wc_f = wc_fitnesses[wc_top]
    wc_m_bc = jtu.tree_map(lambda x: jnp.broadcast_to(x[None], (N_ISLANDS,) + x.shape), wc_m)
    wc_f_bc = jnp.broadcast_to(wc_f[None], (N_ISLANDS, N_MIGRANTS))

    pop = jtu.tree_map(lambda p, m: p.at[jnp.arange(N_ISLANDS)[:, None], wc_idx].set(m), pop, wc_m_bc)
    fitnesses = fitnesses.at[jnp.arange(N_ISLANDS)[:, None], wc_idx].set(wc_f_bc)
    return pop, fitnesses


# ------------------------------------------------------------------ #
#  Init                                                               #
# ------------------------------------------------------------------ #

key = jr.key(SEED)
key, k_pop, k_wc = jr.split(key, 3)

population = jax.vmap(random_w_population, in_axes=(0, None))(jr.split(k_pop, N_ISLANDS), ISLAND_SIZE)
wc_pop, wc_fitnesses = reseed_wildcard(k_wc)
fitnesses = jnp.zeros((N_ISLANDS, ISLAND_SIZE))

# ------------------------------------------------------------------ #
#  Loop                                                               #
# ------------------------------------------------------------------ #

for epoch in range(EPOCHS):
    key, k_step, k_wc_step, k_reseed = jr.split(key, 4)

    if epoch % MIGRATE_EVERY == 0 and epoch > 0:
        population, fitnesses = migrate(population, fitnesses, wc_pop, wc_fitnesses)
        wc_pop, wc_fitnesses = reseed_wildcard(k_reseed)

    # Island evolution
    population, fitnesses, stds, final_states = train_step(population, fitnesses, k_step, jnp.int32(epoch))
    wc_pop, wc_fitnesses, _ = wildcard_step(wc_pop, wc_fitnesses, k_wc_step, jnp.int32(epoch))

    # ── Metrics ──────────────────────────────────────────────────── #
    avg_fit = float(jnp.mean(fitnesses))
    max_fit = float(jnp.max(fitnesses))
    wc_best = float(jnp.max(wc_fitnesses))

    per_island_div = island_diversities(population)  # (N_ISLANDS,)
    global_div = float(jnp.mean(per_island_div))

    writer.add_scalar("Fitness/Average", avg_fit, epoch)
    writer.add_scalar("Fitness/Best", max_fit, epoch)
    writer.add_scalar("Fitness/WC_Best", wc_best, epoch)
    writer.add_scalar("Diversity/Global", global_div, epoch)
    for i, d in enumerate(per_island_div):
        writer.add_scalar(f"Diversity/Island_{i}", float(d), epoch)

    if epoch % 10 == 0:
        key, log_key = jr.split(key)
        best_flat = int(jnp.argmax(fitnesses.reshape(-1)))
        bi, bl = best_flat // ISLAND_SIZE, best_flat % ISLAND_SIZE
        best_genome = population_get(population, bi, bl)
        print(f"[{epoch:4d}] avg={avg_fit:.1f}  best={max_fit:.1f}  wc={wc_best:.1f}  div={global_div:.4f}")

        # Best island W as heatmap
        W_norm = (best_genome.W - best_genome.W.min()) / (best_genome.W.max() - best_genome.W.min() + 1e-8)
        writer.add_image("Genome/Best_W", W_norm[None], epoch)

    if epoch % 100 == 0:
        save_genome(best_genome, path=writer.logdir + f"/best_w{epoch}")
        log_game(lambda s: traverse_w(best_genome, s), log_key)


writer.close()
