import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar
from tensorboardX import SummaryWriter

from cereal import save_genome
from game import KniffelState, reset, step
from log import log_game
from w_genome import (
    DecompWGenome,
    FullWGenome,
    WGenome,
    traverse_w,
)

# ------------------------------------------------------------------ #
#  Hyperparameters                                                    #
# ------------------------------------------------------------------ #

SEED = 123

EPISODES = 2048
EPOCHS = 3_000

ES_POP = 256

ES_SIGMA_A_START = 0.05
ES_SIGMA_A_END = 0.001
ES_SIGMA_B_START = 0.05
ES_SIGMA_B_END = 0.001

ES_SIGMA_W = 0.1
ES_LR = 0.01


@eqx.filter_jit()
def population_diversity(pop: WGenome) -> Float[Array, ""]:
    """Mean std of W entries across individuals — high means spread out."""
    flat = pop.W.reshape(pop.W.shape[0], -1)  # (pop, 182)
    return jnp.mean(jnp.std(flat, axis=0))


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


@jax.jit
def es_step(genome: WGenome, key: PRNGKeyArray, epoch: Int[Scalar, ""]) -> tuple[WGenome, Float[Array, ""]]:
    if isinstance(genome, DecompWGenome):
        k_noise_a, k_noise_b, k_ep = jr.split(key, 3)
        half = ES_POP // 2

        noises_a = jr.normal(k_noise_a, (half, *genome.A.shape))
        noises_full_a = jnp.concatenate([noises_a, -noises_a], axis=0)

        noises_b = jr.normal(k_noise_b, (half, *genome.B.shape))
        noises_full_b = jnp.concatenate([noises_b, -noises_b], axis=0)

        progress = jnp.clip(1.0 - epoch.astype(jnp.float32) / EPOCHS, 0.0, 1.0)
        sigma_a = ES_SIGMA_A_END + (ES_SIGMA_A_START - ES_SIGMA_A_END) * progress
        sigma_b = ES_SIGMA_B_END + (ES_SIGMA_B_START - ES_SIGMA_B_END) * progress
        pop = jax.vmap(lambda n_a, n_b: DecompWGenome(A=genome.A + sigma_a * n_a, B=genome.B + sigma_b * n_b))(
            noises_full_a, noises_full_b
        )

        ep_keys = jax.vmap(lambda j: jr.fold_in(k_ep, j))(jnp.arange(EPISODES))
        ep_keys_bc = jnp.broadcast_to(ep_keys[None], (ES_POP, EPISODES))
        rewards, _ = run_batch(pop, ep_keys_bc)
        fitnesses = jnp.mean(rewards, axis=1)

        f_norm = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)

        grad_a = jnp.mean(f_norm[:, None, None] * noises_full_a, axis=0) / sigma_a
        grad_b = jnp.mean(f_norm[:, None, None] * noises_full_b, axis=0) / sigma_b

        new_genome = DecompWGenome(A=genome.A + ES_LR * grad_a, B=genome.B + ES_LR * grad_b)

    elif isinstance(genome, FullWGenome):
        k_noise, k_ep = jr.split(key, 2)
        half = ES_POP // 2

        noises = jr.normal(k_noise, (half, *genome.W.shape))
        noises_full = jnp.concatenate([noises, -noises], axis=0)

        pop = jax.vmap(lambda n: FullWGenome(W=genome.W + ES_SIGMA_W * n))(noises_full)

        ep_keys = jax.vmap(lambda j: jr.fold_in(k_ep, j))(jnp.arange(EPISODES))
        ep_keys_bc = jnp.broadcast_to(ep_keys[None], (ES_POP, EPISODES))
        rewards, _ = run_batch(pop, ep_keys_bc)
        fitnesses = jnp.mean(rewards, axis=1)

        f_norm = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        grad_w = jnp.mean(f_norm[:, None, None] * noises_full, axis=0) / ES_SIGMA_W

        new_genome = FullWGenome(W=genome.W + ES_LR * grad_w)

    return new_genome, fitnesses


writer = SummaryWriter(comment="_w_genome")
#
# ------------------------------------------------------------------ #
#  Init                                                               #
# ------------------------------------------------------------------ #

key = jr.key(SEED)
key, k_genome = jr.split(key, 2)

genome = WGenome.random(k_genome)

# ------------------------------------------------------------------ #
#  Loop                                                               #
# ------------------------------------------------------------------ #

for epoch in range(EPOCHS):
    key, k_step, k_wc_step, k_reseed = jr.split(key, 4)

    # Evolution
    genome, fitnesses = es_step(genome, k_step, epoch)

    # ── Metrics ──────────────────────────────────────────────────── #
    avg_fit = float(jnp.mean(fitnesses))
    max_fit = float(jnp.max(fitnesses))

    writer.add_scalar("Fitness/Best", max_fit, epoch)
    writer.add_scalar("Fitness/Average", avg_fit, epoch)

    if epoch % 10 == 0:
        key, log_key = jr.split(key)
        print(f"[{epoch:4d}] avg={avg_fit:.1f}  best={max_fit:.1f}")

        W_norm = (genome.W - genome.W.min()) / (genome.W.max() - genome.W.min() + 1e-8)
        writer.add_image("Genome/Best_W", W_norm[None], epoch)

    if epoch % 100 == 0:
        save_genome(genome, path=writer.logdir + f"/best_w{epoch}")
        log_game(lambda s: traverse_w(genome, s), log_key)


writer.close()
