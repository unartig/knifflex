from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import TYPE_CHECKING, Literal, TypeVar

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

W_SCALE = 20
W_TMP = 200.0
W_SCALE_SCALE = 1.0
W_SCALE_TMP = 100.0


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


G = TypeVar("G", bound="WGenomeBase")


class WGenomeBase(eqx.Module):
    @property
    @abstractmethod
    def w(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]: ...

    @property
    @abstractmethod
    def w_scale(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]: ...

    @property
    @abstractmethod
    def bonus_uplift(self) -> Float[Scalar, ""]: ...

    @staticmethod
    @abstractmethod
    def random(key: PRNGKeyArray, sigma: float = 0.1) -> WGenomeBase: ...

    @staticmethod
    @abstractmethod
    def crossover(pa: G, pb: G, key: PRNGKeyArray) -> G: ...

    @jaxtyped(typechecker=typechecker)
    def get_scale_and_weight(
        self, ctx: Float[Array, f"{CTX_DIM}"]
    ) -> tuple[Float[Array, f"{N_CATS}"], Float[Array, f"{N_CATS}"]]:
        cat_weights = self.w @ ctx
        cat_scales = self.w_scale @ ctx
        return cat_weights, cat_scales

    @jaxtyped(typechecker=typechecker)
    def mutate(
        self,
        key: PRNGKeyArray,
        sigma: Float[Scalar, ""],
        p_reset: Float[Scalar, ""],
    ) -> G:
        arrays, static = eqx.partition(self, eqx.is_array)
        leaves, treedef = jax.tree.flatten(arrays)
        keys = jr.split(key, 3 * len(leaves)).reshape(len(leaves), 3)

        def _perturb(leaf: Float[Array, "*"], ks: Shaped[PRNGKeyArray, " 3"]) -> Float[Array, "*"]:
            k1, k2, k3 = ks
            s = jnp.where(leaf.ndim == 0, jnp.float32(0.5), sigma)
            noise = jr.normal(k1, leaf.shape) * s
            reset_val = jr.normal(k3, leaf.shape) * jnp.where(leaf.ndim == 0, jnp.float32(1.0), jnp.float32(0.1))
            return jnp.where(jr.bernoulli(k2, p_reset, leaf.shape), reset_val, leaf + noise)

        new_leaves = [_perturb(lf, ks) for lf, ks in zip(leaves, keys, strict=True)]
        new_arrays = jax.tree.unflatten(treedef, new_leaves)
        return eqx.combine(new_arrays, static)

    @jaxtyped(typechecker=typechecker)
    def es_perturb(self, noise: G, sigmas: dict[str, float]) -> G:
        """Return a new genome with noise added field-wise."""
        arrays, static = eqx.partition(self, eqx.is_array)
        noise_arrays, _ = eqx.partition(noise, eqx.is_array)
        leaves, treedef = jax.tree.flatten(arrays)
        noise_leaves, _ = jax.tree.flatten(noise_arrays)
        field_names = [f.name for f in dataclasses.fields(self)]

        perturbed_leaves = [w + sigmas[name] * n for w, n, name in zip(leaves, noise_leaves, field_names, strict=True)]
        return eqx.combine(jax.tree.unflatten(treedef, perturbed_leaves), static)

    @jaxtyped(typechecker=typechecker)
    def es_make_noises(self, key: PRNGKeyArray, half: int) -> G:
        """Antithetic noise with same structure as genome."""
        arrays, static = eqx.partition(self, eqx.is_array)
        leaves, treedef = jax.tree.flatten(arrays)
        keys = jr.split(key, len(leaves))
        noise_leaves = [
            jnp.concatenate([n := jr.normal(k, (half, *leaf.shape)), -n], axis=0)
            for k, leaf in zip(keys, leaves, strict=True)
        ]
        return eqx.combine(jax.tree.unflatten(treedef, noise_leaves), static)

    @jaxtyped(typechecker=typechecker)
    def oracle_action(self, state: KniffelState) -> Int[Scalar, ""]:
        ctx = build_context(state)
        cat_weights, cat_scales = self.get_scale_and_weight(ctx)

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
            uplift = jnp.where((raw_evs >= bonus_rem) & (jnp.arange(13) < 6) & (~bonus_earned), self.bonus_uplift, 0.0)
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
class FullWGenome(WGenomeBase):
    raw_w: Float[Array, f"{N_CATS} {CTX_DIM}"]
    raw_w_scale: Float[Array, f"{N_CATS} {CTX_DIM}"]
    raw_bonus_uplift: Float[Scalar, ""] = eqx.field(default_factory=lambda: jnp.float32(0.0))

    @property
    def w(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return W_SCALE * jnp.tanh(self.raw_w / W_TMP)

    @property
    def w_scale(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return jnp.exp(W_SCALE_SCALE * jnp.tanh(self.raw_w_scale / W_SCALE_TMP))

    @property
    def bonus_uplift(self) -> Float[Scalar, ""]:
        return self.raw_bonus_uplift

    @staticmethod
    def random(key: PRNGKeyArray, sigma: float = 0.1) -> FullWGenome:
        kw, kws = jr.split(key, 2)
        # QR on the taller dimension, then transpose to fit
        raw = jr.normal(kw, (CTX_DIM, N_CATS))
        Q, _ = jnp.linalg.qr(raw)

        _W_scale = jr.normal(kws, (N_CATS, CTX_DIM)) * 0.001
        return FullWGenome(raw_w=Q.T * sigma, raw_w_scale=_W_scale)

    @staticmethod
    def crossover(pa: FullWGenome, pb: FullWGenome, key: PRNGKeyArray) -> FullWGenome:
        leaves_a, treedef = jax.tree.flatten(eqx.filter(pa, eqx.is_array))
        leaves_b, _ = jax.tree.flatten(eqx.filter(pb, eqx.is_array))
        n_leaves = len(leaves_a)
        keys = jr.split(key, n_leaves)
        new_leaves = [
            jnp.where(jr.bernoulli(k, 0.5, la.shape), la, lb)
            for la, lb, k in zip(leaves_a, leaves_b, keys, strict=True)
        ]
        new_raw_fields = jax.tree.unflatten(treedef, new_leaves)
        return FullWGenome(**new_raw_fields)


@jaxtyped(typechecker=typechecker)
class DecompWGenome(WGenomeBase):
    raw_a: Float[Array, f"{N_CATS} {DECOMP_RANK}"]
    raw_b: Float[Array, f"{DECOMP_RANK} {CTX_DIM}"]
    raw_a_scale: Float[Array, f"{N_CATS} {DECOMP_RANK}"]
    raw_b_scale: Float[Array, f"{DECOMP_RANK} {CTX_DIM}"]
    raw_bonus_uplift: Float[Scalar, ""] = eqx.field(default_factory=lambda: jnp.float32(0.0))

    @property
    def w(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return W_SCALE * jnp.tanh(self.raw_a @ self.raw_b / W_TMP)

    @property
    def w_scale(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return jnp.exp(W_SCALE_SCALE * jnp.tanh(self.raw_a_scale @ self.raw_b_scale / W_SCALE_TMP))

    @property
    def bonus_uplift(self) -> Float[Scalar, ""]:
        return self.raw_bonus_uplift

    @staticmethod
    def random(key: PRNGKeyArray, sigma: float = 0.1) -> DecompWGenome:
        ka, kb, kas, kbs = jr.split(key, 4)

        A = jr.normal(ka, (N_CATS, DECOMP_RANK)) * sigma
        raw = jr.normal(kb, (CTX_DIM, DECOMP_RANK))  # tall matrix
        Q, _ = jnp.linalg.qr(raw)
        B = Q.T * sigma

        A_scale = jr.normal(kas, (N_CATS, DECOMP_RANK)) * sigma
        B_scale = jr.normal(kbs, (DECOMP_RANK, CTX_DIM)) * 0.001
        return DecompWGenome(raw_a=A, raw_b=B, raw_a_scale=A_scale, raw_b_scale=B_scale)

    @staticmethod
    def crossover(pa: DecompWGenome, pb: DecompWGenome, key: PRNGKeyArray) -> DecompWGenome:
        ka, kb, kup = jr.split(key, 3)
        # Row-wise crossover:
        # swaps entire latent profiles for categories (A)
        # and entire basis vectors for features (B)
        mask_A = jr.bernoulli(ka, 0.5, (N_CATS, 1))
        mask_B = jr.bernoulli(kb, 0.5, (DECOMP_RANK, 1))

        return DecompWGenome(
            raw_a=jnp.where(mask_A, pa.raw_a, pb.raw_a),
            raw_b=jnp.where(mask_B, pa.raw_b, pb.raw_b),
            raw_a_scale=jnp.where(mask_A, pa.raw_a_scale, pb.raw_a_scale),
            raw_b_scale=jnp.where(mask_B, pa.raw_b_scale, pb.raw_b_scale),
            raw_bonus_uplift=jnp.where(jr.bernoulli(kup), pa.bonus_uplift, pb.bonus_uplift),
        )


if GENOME_TYPE == "full":
    WGenome = FullWGenome
elif GENOME_TYPE == "decomp":
    WGenome = DecompWGenome
else:
    raise ValueError(f"Unknown GENOME_TYPE: {GENOME_TYPE!r}  (expected 'full' or 'decomp')")


@jaxtyped(typechecker=typechecker)
def population_get(population: WGenome, *indices: int) -> WGenome:
    genome = population
    for idx in indices:
        genome = jax.tree.map(lambda leaf: leaf[idx], genome)
    return genome


@jaxtyped(typechecker=typechecker)
def crossover_w(pa: WGenome, pb: WGenome, key: PRNGKeyArray) -> WGenome:
    return WGenome.crossover(pa, pb, key)


@jaxtyped(typechecker=typechecker)
def random_w_population(key: PRNGKeyArray, pop_size: int, sigma: float = 0.1) -> WGenome:
    keys = jr.split(key, pop_size)
    return jax.vmap(WGenome.random, in_axes=(0, None))(keys, sigma)
