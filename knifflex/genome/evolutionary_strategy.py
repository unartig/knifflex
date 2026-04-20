import dataclasses
from datetime import datetime

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, Shaped, jaxtyped
from tensorboardX import SummaryWriter

from knifflex.game.game import KniffelState, reset, step
from knifflex.utils.log import log_game
from knifflex.utils.utils import typechecker

from .cereal import save_genome
from .w_genome import (
    GENOME_TYPE,
    WGenome,
)

SEED = 123

EPISODES = 1024
ES_POP = 1024
EPOCHS = 3_000

ES_SIGMA_UP = 1.0

ES_SIGMA_A = 0.3
ES_SIGMA_B = 0.3
ES_SIGMA_A_SCALE = 0.5
ES_SIGMA_B_SCALE = 0.5

ES_SIGMA_W = 0.1
ES_SIGMA_W_SCALE = 0.1

ES_LR_PEAK = 5e-1
ES_LR_MIN = 1e-2
WARMUP_STEPS = 5
SCHED_FRAC = 0.5

if GENOME_TYPE == "full":
    SIGMAS: dict[str, float] = {
        "raw_w": ES_SIGMA_W,
        "raw_w_scale": ES_SIGMA_W_SCALE,
        "raw_bonus_uplift": ES_SIGMA_UP,
    }
elif GENOME_TYPE == "decomp":
    SIGMAS: dict[str, float] = {
        "raw_a": ES_SIGMA_A,
        "raw_b": ES_SIGMA_B,
        "raw_a_scale": ES_SIGMA_A_SCALE,
        "raw_b_scale": ES_SIGMA_B_SCALE,
        "raw_bonus_uplift": ES_SIGMA_UP,
    }
else:
    raise ValueError(f"Unknown GENOME_TYPE: {GENOME_TYPE!r}  (expected 'full' or 'decomp')")


@jaxtyped(typechecker=typechecker)
def policy(population: WGenome, states: KniffelState) -> Int[Array, f"{ES_POP} {EPISODES}"]:
    return jax.vmap(
        lambda w, s_batch: jax.vmap(w.oracle_action)(s_batch),
        in_axes=(0, 0),
    )(population, states)


@jaxtyped(typechecker=typechecker)
def run_batch(
    population: WGenome, keys: Shaped[PRNGKeyArray, f"{ES_POP} {EPISODES}"]
) -> tuple[Int[Array, f"{ES_POP} {EPISODES}"], KniffelState]:
    pop_size = population.w.shape[0]
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
) -> tuple[WGenome, optax.OptState, Float[Array, ""]]:
    half = ES_POP // 2
    key, k_noise, k_ep = jr.split(key, 3)

    noises = genome.es_make_noises(k_noise, half)

    # perturb: pop[i] = genome + sigma * noise[i]
    pop = jax.vmap(lambda n: genome.es_perturb(n, SIGMAS))(noises)

    ep_keys = jnp.tile(jax.vmap(lambda j: jr.fold_in(k_ep, j))(jnp.arange(EPISODES))[None], (ES_POP, 1))
    rewards, _ = run_batch(pop, ep_keys)
    fitnesses = jnp.mean(rewards, axis=1)
    f_norm = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)

    # Grad: mean(f_norm * noise) / sigma, field-wise — zip leaves directly
    genome_arrays, _static = eqx.partition(genome, eqx.is_array)
    _genome_leaves, treedef = jax.tree.flatten(genome_arrays)  # treedef from genome, not noises
    noise_leaves, _ = jax.tree.flatten(eqx.filter(noises, eqx.is_array))
    field_names = [f.name for f in dataclasses.fields(genome)]

    grad_leaves = [
        jnp.mean(f_norm[(..., *([None] * (n.ndim - 1)))] * n, axis=0) / SIGMAS[name]
        for n, name in zip(noise_leaves, field_names, strict=True)
    ]

    neg_grads = jax.tree.unflatten(treedef, [-g for g in grad_leaves])
    updates, new_opt_state = optimizer.update(neg_grads, opt_state)
    return eqx.apply_updates(genome, updates), new_opt_state, fitnesses


date_str = datetime.now().isoformat().split(".")[0].replace("-", "_").replace(":", "_")
writer = SummaryWriter(f"data/runs/{date_str}", comment="_w_genome")

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

    raw_w = genome.raw_a @ genome.raw_b  # or genome.raw_w for FullWGenome
    raw_ws = genome.raw_a_scale @ genome.raw_b_scale

    writer.add_scalar("Fitness/Best", max_fit, epoch)
    writer.add_scalar("Fitness/Average", avg_fit, epoch)
    current_lr = float(jnp.float32(schedule(epoch)))
    writer.add_scalar("Train/LR", current_lr, epoch)
    writer.add_scalar("Genome/uplift", genome.bonus_uplift, epoch)

    writer.add_scalar("Weights/w_saturation", float(jnp.mean(jnp.abs(jnp.tanh(raw_w)) > 0.95)), epoch)
    writer.add_scalar("Weights/w_scale_saturation", float(jnp.mean(jnp.abs(jnp.tanh(raw_ws)) > 0.95)), epoch)
    writer.add_scalar("Weights/w_raw_absmax", float(jnp.max(jnp.abs(raw_w))), epoch)
    writer.add_scalar("Weights/w_raw_absmean", float(jnp.mean(jnp.abs(raw_w))), epoch)
    writer.add_scalar("Weights/w_out_absmax", float(jnp.max(jnp.abs(genome.w))), epoch)
    writer.add_scalar("Weights/w_scale_out_max", float(jnp.max(genome.w_scale)), epoch)
    writer.add_scalar("Weights/w_scale_out_min", float(jnp.min(genome.w_scale)), epoch)
    writer.add_scalar("Weights/raw_w_rms", float(jnp.sqrt(jnp.mean((genome.raw_a @ genome.raw_b) ** 2))), epoch)
    writer.add_scalar(
        "Weights/raw_ws_rms", float(jnp.sqrt(jnp.mean((genome.raw_a_scale @ genome.raw_b_scale) ** 2))), epoch
    )

    if epoch % 10 == 0:
        key, log_key = jr.split(key)
        print(f"[{epoch:4d}] avg={avg_fit:.1f}  best={max_fit:.1f} w/ {genome.bonus_uplift:.4f}")

        W_norm = (genome.w - genome.w.min()) / (genome.w.max() - genome.w.min() + 1e-8)
        writer.add_image("Genome/Best_W", W_norm[None], epoch)
        W_scale_norm = (genome.w_scale - genome.w_scale.min()) / (genome.w_scale.max() - genome.w_scale.min() + 1e-8)
        writer.add_image("Genome/Best_W_scale", W_scale_norm[None], epoch)

    if epoch % 100 == 0:
        save_genome(genome, path=writer.logdir + f"/best_w{epoch}")
        log_game(genome.oracle_action, log_key)


writer.close()
