import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, Shaped, jaxtyped
from tensorboardX import SummaryWriter

from cereal import save_genome
from game import KniffelState, reset, step
from log import log_game
from utils import typechecker
from w_genome import (
    DecompWGenome,
    FullWGenome,
    WGenome,
    genome_action,
)

SEED = 123

EPISODES = 512
EPOCHS = 3_000

ES_POP = 512

ES_SIGMA_UP = 3.0

ES_SIGMA_A = 2.0
ES_SIGMA_B = 2.0
ES_SIGMA_A_SCALE = 0.5
ES_SIGMA_B_SCALE = 0.5

ES_SIGMA_W = 0.1
ES_SIGMA_W_SCALE = 0.1

ES_LR_PEAK = 5e-1
ES_LR_MIN = 1e-2
WARMUP_STEPS = 5
SCHED_FRAC = 0.5


@jaxtyped(typechecker=typechecker)
def policy(population: WGenome, states: KniffelState) -> Int[Array, f"{ES_POP} {EPISODES}"]:
    return jax.vmap(
        lambda w, s_batch: jax.vmap(lambda s: genome_action(w, s))(s_batch),
        in_axes=(0, 0),
    )(population, states)


@jaxtyped(typechecker=typechecker)
def run_batch(
    population: WGenome, keys: Shaped[PRNGKeyArray, f"{ES_POP} {EPISODES}"]
) -> tuple[Int[Array, f"{ES_POP} {EPISODES}"], KniffelState]:
    pop_size = population.W.shape[0]
    states = jax.vmap(jax.vmap(reset))(keys)
    rewards = jnp.zeros((pop_size, EPISODES), dtype=jnp.int32)

    def body_fn(
        carry: tuple[KniffelState, Int[Array, f"{ES_POP} {EPISODES}"]], _: None
    ) -> tuple[tuple[KniffelState, Int[Array, f"{ES_POP} {EPISODES}"]], None]:
        states, rewards = carry
        actions = policy(population, states)
        new_states, r = jax.vmap(jax.vmap(step))(states, actions)
        return (new_states, rewards + r), None

    (final_states, rewards), _ = jax.lax.scan(body_fn, (states, rewards), None, length=39)
    return rewards, final_states


@jax.jit
@jaxtyped(typechecker=typechecker)
def es_step(
    genome: WGenome, opt_state: optax.OptState, key: PRNGKeyArray, epoch: Int[Scalar, ""]
) -> tuple[FullWGenome | DecompWGenome, optax.OptState, Float[Array, ""]]:
    progress = jnp.clip(1.0 - epoch.astype(jnp.float32) / EPOCHS, 0.0, 1.0)

    half = ES_POP // 2

    # Generate antithetic noise for all components
    @jaxtyped(typechecker=typechecker)
    def make_noise(k: PRNGKeyArray, shape: tuple[int, ...]) -> Float[Array, f"{ES_POP} *"]:
        n = jr.normal(k, (half, *shape))
        return jnp.concatenate([n, -n], axis=0)

    @jaxtyped(typechecker=typechecker)
    def get_grad(
        f_norm: Float[Array, f"{ES_POP}"], noise: Float[Array, f"{ES_POP} *"], sigma: float
    ) -> Float[Array, "*"]:
        expand = (slice(None),) + (None,) * (noise.ndim - 1)
        return jnp.mean(f_norm[expand] * noise, axis=0) / sigma

    key, k_ep = jr.split(key)
    ep_keys = jax.vmap(lambda j: jr.fold_in(k_ep, j))(jnp.arange(EPISODES))
    ep_keys_bc = jnp.tile(ep_keys[None], (ES_POP, 1))

    if isinstance(genome, DecompWGenome):
        k_a, k_b, k_as, k_bs, k_up = jr.split(key, 5)

        noise_a = make_noise(k_a, genome.A.shape)
        noise_b = make_noise(k_b, genome.B.shape)
        noise_as = make_noise(k_as, genome.A_scale.shape)
        noise_bs = make_noise(k_bs, genome.B_scale.shape)
        noise_up = make_noise(k_up, genome.bonus_uplift.shape)

        # Create population with perturbed weights AND scales
        pop = jax.vmap(
            lambda na, nb, nas, nbs, nup: DecompWGenome(
                A=genome.A + ES_SIGMA_A * na,
                B=genome.B + ES_SIGMA_B * nb,
                A_scale=genome.A_scale + ES_SIGMA_A_SCALE * nas,
                B_scale=genome.B_scale + ES_SIGMA_B_SCALE * nbs,
                bonus_uplift=genome.bonus_uplift + ES_SIGMA_UP * nup,
            )
        )(noise_a, noise_b, noise_as, noise_bs, noise_up)

        # Run episodes
        rewards, _ = run_batch(pop, ep_keys_bc)
        fitnesses = jnp.mean(rewards, axis=1)

        f_norm = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)

        grads = DecompWGenome(
            A=get_grad(f_norm, noise_a, ES_SIGMA_A),
            B=get_grad(f_norm, noise_b, ES_SIGMA_B),
            A_scale=get_grad(f_norm, noise_as, ES_SIGMA_A_SCALE),
            B_scale=get_grad(f_norm, noise_bs, ES_SIGMA_B_SCALE),
            bonus_uplift=get_grad(f_norm, noise_up, ES_SIGMA_UP),
        )

    elif isinstance(genome, FullWGenome):
        k_w, k_ws, k_up = jr.split(key, 3)

        noise_w = make_noise(k_w, genome.W.shape)
        noise_ws = make_noise(k_ws, genome.W_scale.shape)
        noise_up = make_noise(k_up, genome.bonus_uplift.shape)

        pop = jax.vmap(
            lambda nw, nws, nup: FullWGenome(
                _W=genome.W + ES_SIGMA_W * nw,
                _W_scale=genome.W_scale + ES_SIGMA_W_SCALE * nws,
                bonus_uplift=genome.bonus_uplift + ES_SIGMA_UP * nup,
            )
        )(noise_w, noise_ws, noise_up)

        rewards, _ = run_batch(pop, ep_keys_bc)
        fitnesses = jnp.mean(rewards, axis=1)
        f_norm = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)

        grads = FullWGenome(
            _W=get_grad(f_norm, noise_w, ES_SIGMA_W),
            _W_scale=get_grad(f_norm, noise_ws, ES_SIGMA_W_SCALE),
            bonus_uplift=get_grad(f_norm, noise_up, ES_SIGMA_UP),
        )

    grads_arrays = eqx.filter(grads, eqx.is_array)
    neg_grads = jax.tree.map(lambda g: -g, grads_arrays)

    updates, new_opt_state = optimizer.update(neg_grads, opt_state)
    new_genome = eqx.apply_updates(genome, updates)

    return new_genome, new_opt_state, fitnesses


writer = SummaryWriter(comment="_w_genome")

key = jr.key(SEED)
key, k_genome = jr.split(key, 2)

genome = WGenome.random(k_genome)
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=ES_LR_PEAK,
    warmup_steps=WARMUP_STEPS,
    decay_steps=int(EPOCHS * SCHED_FRAC),
    end_value=ES_LR_MIN,
)
optimizer = optax.adam(schedule)
opt_state = optimizer.init(eqx.filter(genome, eqx.is_array))

for epoch in range(EPOCHS):
    key, k_step, k_wc_step, k_reseed = jr.split(key, 4)

    # Evolution
    genome, opt_state, fitnesses = es_step(genome, opt_state, k_step, epoch)

    # Metrics
    avg_fit = float(jnp.mean(fitnesses))
    max_fit = float(jnp.max(fitnesses))

    writer.add_scalar("Fitness/Best", max_fit, epoch)
    writer.add_scalar("Fitness/Average", avg_fit, epoch)
    current_lr = float(jnp.float32(schedule(epoch)))
    writer.add_scalar("Train/LR", current_lr, epoch)
    writer.add_scalar("Genome/uplift", genome.bonus_uplift, epoch)

    if epoch % 10 == 0:
        key, log_key = jr.split(key)
        print(f"[{epoch:4d}] avg={avg_fit:.1f}  best={max_fit:.1f} w/ {genome.bonus_uplift:.4f}")

        W_norm = (genome.W - genome.W.min()) / (genome.W.max() - genome.W.min() + 1e-8)
        writer.add_image("Genome/Best_W", W_norm[None], epoch)
        W_scale_norm = (genome.W_scale - genome.W_scale.min()) / (genome.W_scale.max() - genome.W_scale.min() + 1e-8)
        writer.add_image("Genome/Best_W_scale", W_scale_norm[None], epoch)

    if epoch % 100 == 0:
        save_genome(genome, path=writer.logdir + f"/best_w{epoch}")
        log_game(lambda s: genome_action(genome, s), log_key)


writer.close()
