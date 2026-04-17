from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, Shaped, jaxtyped

from knifflex.game.ev_table import get_ev_table
from knifflex.game.game import TRANSITION_TABLE
from knifflex.game.scoring import CAT_MAX, N_CATS
from knifflex.utils.utils import typechecker

if TYPE_CHECKING:
    from knifflex.game.game import KniffelState

EV_TABLE, _MASK_TABLE, _ROLLS = (jnp.asarray(arr) for arr in get_ev_table())
MASK_TABLE = _MASK_TABLE.astype(jnp.uint8)

GENOME_TYPE: Literal["full", "decomp"] = "decomp"
DECOMP_RANK: int = 1

# Context layout (16 dims):
#   [0:13]  normed_scores  — actual score / max_score if filled, else 0.0
#   [13]    upper_sum      — sum of upper scores / 63.0
#   [14]    bonus_dist     — clip((63 - upper_filled) / 63, 0, 1)
#   [15]    rolls_left     — rolls_left / 2.0
#   [16]    rounds_left    - rounds / 12.0
CTX_DIM = 17


@jaxtyped(typechecker=typechecker)
def build_context(state: KniffelState) -> Float[Array, f" {CTX_DIM}"]:
    scorecard = state.scorecard.astype(jnp.float32)
    filled_scores = jnp.where(scorecard >= 0, scorecard, 0.0)
    normed_scores = filled_scores / CAT_MAX

    upper_filled = jnp.sum(filled_scores[:6])
    upper_sum = upper_filled / 63.0
    bonus_dist = jnp.clip((63.0 - upper_filled) / 63.0, 0.0, 1.0)
    rolls_left_norm = state.rolls_left.squeeze().astype(jnp.float32) / 2.0
    rounds_left_norm = state.round.squeeze().astype(jnp.float32) / 12.0

    return jnp.concatenate([normed_scores, jnp.array([upper_sum, bonus_dist, rolls_left_norm, rounds_left_norm])])


@jaxtyped(typechecker=typechecker)
class FullWGenome(eqx.Module):
    _W: Float[Array, f"{N_CATS} {CTX_DIM}"]
    _W_scale: Float[Array, f"{N_CATS} {CTX_DIM}"]
    bonus_uplift: Float[Scalar, ""] = eqx.field(default_factory=lambda: jnp.array(0.0))

    @property
    def W(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return self._W

    @property
    def W_scale(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return self._W_scale

    @staticmethod
    def random(key: PRNGKeyArray, sigma: float = 0.1) -> FullWGenome:
        kw, kws = jr.split(key, 2)
        # QR on the taller dimension, then transpose to fit (13, 16)
        raw = jr.normal(kw, (CTX_DIM, N_CATS))  # (16, 13)
        Q, _ = jnp.linalg.qr(raw)  # (16, 13) orthonormal columns

        _W_scale = jr.normal(kws, (N_CATS, CTX_DIM)) * 0.001
        return FullWGenome(_W=Q.T * sigma, _W_scale=_W_scale)  # (13, 16)

    @staticmethod
    def crossover(pa: FullWGenome, pb: FullWGenome, key: PRNGKeyArray) -> FullWGenome:
        """Bernoulli blend per trainable leaf."""
        leaves_a, treedef = jax.tree.flatten(eqx.filter(pa, eqx.is_array))
        leaves_b, _ = jax.tree.flatten(eqx.filter(pb, eqx.is_array))

        n_leaves = len(leaves_a)
        keys = jr.split(key, n_leaves)

        new_leaves = [
            jnp.where(jr.bernoulli(k, 0.5, la.shape), la, lb)
            for la, lb, k in zip(leaves_a, leaves_b, keys, strict=True)
        ]
        return jax.tree.unflatten(treedef, new_leaves)


@jaxtyped(typechecker=typechecker)
class DecompWGenome(eqx.Module):
    A: Float[Array, f"{N_CATS} {DECOMP_RANK}"]
    B: Float[Array, f"{DECOMP_RANK} {CTX_DIM}"]

    A_scale: Float[Array, f"{N_CATS} {DECOMP_RANK}"]
    B_scale: Float[Array, f"{DECOMP_RANK} {CTX_DIM}"]
    bonus_uplift: Float[Scalar, ""] = eqx.field(default_factory=lambda: jnp.float32(0.0))

    @property
    def W(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return self.A @ self.B

    @property
    def W_scale(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return self.A_scale @ self.B_scale

    @staticmethod
    def random(key: PRNGKeyArray, sigma: float = 0.1) -> DecompWGenome:
        ka, kb, kas, kbs = jr.split(key, 4)

        A = jr.normal(ka, (N_CATS, DECOMP_RANK)) * sigma
        raw = jr.normal(kb, (CTX_DIM, DECOMP_RANK))  # tall matrix
        Q, _ = jnp.linalg.qr(raw)  # (16, 4) orthonormal columns
        B = Q.T * sigma  # (4, 16), scaled

        A_scale = jr.normal(kas, (N_CATS, DECOMP_RANK)) * sigma
        B_scale = jr.normal(kbs, (DECOMP_RANK, CTX_DIM)) * 0.001
        return DecompWGenome(A=A, B=B, A_scale=A_scale, B_scale=B_scale)

    @staticmethod
    def crossover(pa: DecompWGenome, pb: DecompWGenome, key: PRNGKeyArray) -> DecompWGenome:
        ka, kb, kup = jr.split(key, 3)
        # Row-wise crossover:
        # swaps entire latent profiles for categories (A)
        # and entire basis vectors for features (B)
        mask_A = jr.bernoulli(ka, 0.5, (N_CATS, 1))
        mask_B = jr.bernoulli(kb, 0.5, (DECOMP_RANK, 1))

        return DecompWGenome(
            A=jnp.where(mask_A, pa.A, pb.A),
            B=jnp.where(mask_B, pa.B, pb.B),
            A_scale=jnp.where(mask_A, pa.A_scale, pb.A_scale),
            B_scale=jnp.where(mask_B, pa.B_scale, pb.B_scale),
            bonus_uplift=jnp.where(jr.bernoulli(kup), pa.bonus_uplift, pb.bonus_uplift),
        )


if GENOME_TYPE == "full":
    WGenome = FullWGenome
elif GENOME_TYPE == "decomp":
    WGenome = DecompWGenome
else:
    raise ValueError(f"Unknown GENOME_TYPE: {GENOME_TYPE!r}  (expected 'full' or 'decomp')")

@jaxtyped(typechecker=typechecker)
def _mutate(
    genome: WGenome,
    key: PRNGKeyArray,
    sigma: Float[Scalar, ""],
    p_reset: Float[Scalar, ""],
) -> WGenome:
    arrays, treedef = jax.tree.flatten(eqx.filter(genome, eqx.is_array))
    n = len(arrays)
    keys = jr.split(key, 3 * n).reshape(n, 3)

    # Scalar leaves (bonus_uplift) get a fixed small absolute perturbation
    # rather than inheriting the weight sigma
    @jaxtyped(typechecker=typechecker)
    def leaf_sigma(leaf: Float[Array, "*"] | Float[Scalar, ""]) -> Float[Array, "*"] | Float[Scalar, ""]:
        return jnp.where(leaf.ndim == 0, jnp.float32(0.5), sigma)

    @jaxtyped(typechecker=typechecker)
    def _perturb(leaf: Float[Array, "*"], ks: Shaped[PRNGKeyArray, " 3"]) -> Float[Array, "*"]:
        k1, k2, k3 = ks
        s = leaf_sigma(leaf)
        noise = jr.normal(k1, leaf.shape) * s
        reset_mask = jr.bernoulli(k2, p_reset, leaf.shape)
        fresh = jr.normal(k3, leaf.shape) * jnp.where(leaf.ndim == 0, jnp.float32(1.0), jnp.float32(0.1))
        return jnp.where(reset_mask, fresh, leaf + noise)

    new_leaves = [_perturb(lf, ks) for lf, ks in zip(arrays, keys, strict=True)]
    return jax.tree.unflatten(treedef, new_leaves)


@jaxtyped(typechecker=typechecker)
def oracle_action(genome: WGenome, state: KniffelState) -> Int[Scalar, ""]:
    ctx = build_context(state)
    cat_weights = genome.W @ ctx
    cat_scales = jax.nn.softplus(genome.W_scale @ ctx)

    rl = state.rolls_left.squeeze().astype(jnp.int32)
    dice_idx = state.dice_idx.astype(jnp.int32)
    open_mask = state.scorecard < 0

    # Heuristic Bonus Uplift
    upper_filled = jnp.sum(jnp.where(state.scorecard[:6] >= 0, state.scorecard[:6], 0))
    bonus_rem = jnp.clip(63.0 - upper_filled, 0.0, 63.0)
    bonus_earned = upper_filled >= 63

    @jaxtyped(typechecker=typechecker)
    def compute_cat_scores(rolls_idx: Int[Scalar, ""]) -> Float[Array, f"ROLLS {N_CATS}"]:
        raw_evs = EV_TABLE[:, rolls_idx, :]  # (252, 13)
        uplift = jnp.where((raw_evs >= bonus_rem) & (jnp.arange(13) < 6) & (~bonus_earned), genome.bonus_uplift, 0.0)
        return (raw_evs + uplift) * cat_scales + cat_weights  # (252, 13)

    cat_scores_now = compute_cat_scores(jnp.int32(0))  # (252, 13)
    now_utilities = jnp.max(jnp.where(open_mask, cat_scores_now, -1e9), axis=1)  # (252,)
    val_now = now_utilities[dice_idx]
    score_cat = jnp.argmax(jnp.where(open_mask, cat_scores_now[dice_idx], -1e9))  # consistent

    # Calculate "Reroll" Utility
    # We look at the utilities of the outcome of the NEXT roll (rl - 1)
    # Note: EV_TABLE[dice_idx, rl] is the EV of having 'rl' rolls remaining.
    @jaxtyped(typechecker=typechecker)
    def compute_roll_utilities(rolls_idx: Int[Scalar, ""]) -> Float[Array, " ROLLS"]:
        return jnp.max(jnp.where(open_mask, compute_cat_scores(rolls_idx), -1e9), axis=1)

    future_utilities = compute_roll_utilities(rl)

    # Matrix Multiply: (32, 252) @ (252,) -> (32,)
    # This gives the expected max utility for every possible reroll mask
    expected_utils_per_mask = TRANSITION_TABLE[dice_idx] @ future_utilities

    best_mask_idx = jnp.argmax(expected_utils_per_mask)
    val_reroll = expected_utils_per_mask[best_mask_idx]

    # Final Decision
    # A reroll is only valid if rl > 0 AND the expected future value is better
    should_reroll = (rl > 0) & (val_reroll > val_now)

    return jax.lax.cond(
        should_reroll,
        lambda _: best_mask_idx.astype(jnp.int32),
        lambda _: (score_cat + 32).astype(jnp.int32),
        operand=None,
    )


@jaxtyped(typechecker=typechecker)
def population_get(population: WGenome, *indices: int) -> WGenome:
    genome = population
    for idx in indices:
        genome = jax.tree.map(lambda leaf: leaf[idx], genome)
    return genome


@jaxtyped(typechecker=typechecker)
def genome_action(genome: WGenome, state: KniffelState) -> Int[Scalar, ""]:
    return oracle_action(genome, state)


@jaxtyped(typechecker=typechecker)
def mutate_w(
    genome: WGenome,
    key: PRNGKeyArray,
    sigma: Float[Scalar, ""],
    p_reset: Float[Scalar, ""],
) -> WGenome:
    return _mutate(genome, key, sigma, p_reset)


@jaxtyped(typechecker=typechecker)
def crossover_w(pa: WGenome, pb: WGenome, key: PRNGKeyArray) -> WGenome:
    return WGenome.crossover(pa, pb, key)


@jaxtyped(typechecker=typechecker)
def random_w_population(key: PRNGKeyArray, pop_size: int, sigma: float = 0.1) -> WGenome:
    keys = jr.split(key, pop_size)
    return jax.vmap(WGenome.random, in_axes=(0, None))(keys, sigma)
