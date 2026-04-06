from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, Int, PRNGKeyArray, Scalar, jaxtyped

from ev_table import get_ev_table
from game import TRANSITION_TABLE
from scoring import CASE_MAX
from utils import typechecker

if TYPE_CHECKING:
    from game import KniffelState

EV_TABLE, _MASK_TABLE, _ROLLS = (jnp.asarray(arr) for arr in get_ev_table())
MASK_TABLE = _MASK_TABLE.astype(jnp.uint8)

# ------------------------------------------------------------------ #
#  Config — change these two lines to switch implementations          #
# ------------------------------------------------------------------ #

GENOME_TYPE: Literal["full", "decomp"] = "full"
DECOMP_RANK: int = 4

# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

N_CATS = 13

# Context layout (16 dims):
#   [0:13]  normed_scores  — actual score / max_score if filled, else 0.0
#   [13]    upper_sum      — sum of upper scores / 63.0
#   [14]    bonus_dist     — clip((63 - upper_filled) / 63, 0, 1)
#   [15]    rolls_left     — rolls_left / 2.0
#   [16]    rounds_left    - rounds / 12.0
CTX_DIM = 17


# ------------------------------------------------------------------ #
#  Context builder                                                    #
# ------------------------------------------------------------------ #


def build_context(state: KniffelState) -> Float[Array, f" {CTX_DIM}"]:
    scorecard = state.scorecard.astype(jnp.float32)
    filled_scores = jnp.where(scorecard >= 0, scorecard, 0.0)
    normed_scores = filled_scores / CASE_MAX

    upper_filled = jnp.sum(filled_scores[:6])
    upper_sum = upper_filled / 63.0
    bonus_dist = jnp.clip((63.0 - upper_filled) / 63.0, 0.0, 1.0)
    rolls_left_norm = state.rolls_left.squeeze().astype(jnp.float32) / 2.0
    rounds_left_norm = state.round.squeeze().astype(jnp.float32) / 12.0

    return jnp.concatenate([normed_scores, jnp.array([upper_sum, bonus_dist, rolls_left_norm, rounds_left_norm])])


# ------------------------------------------------------------------ #
#  FullWGenome                                                        #
# ------------------------------------------------------------------ #


@jaxtyped(typechecker=typechecker)
class FullWGenome(eqx.Module):
    _W: Float[Array, f"{N_CATS} {CTX_DIM}"]

    @property
    def W(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return self._W

    @staticmethod
    def random(key: PRNGKeyArray, sigma: float = 0.1) -> FullWGenome:
        # QR on the taller dimension, then transpose to fit (13, 16)
        raw = jr.normal(key, (CTX_DIM, N_CATS))  # (16, 13)
        Q, _ = jnp.linalg.qr(raw)  # (16, 13) orthonormal columns
        return FullWGenome(_W=Q.T * sigma)  # (13, 16)

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


# ------------------------------------------------------------------ #
#  DecompWGenome                                                      #
# ------------------------------------------------------------------ #


@jaxtyped(typechecker=typechecker)
class DecompWGenome(eqx.Module):
    A: Float[Array, f"{N_CATS} {DECOMP_RANK}"]
    B: Float[Array, f"{DECOMP_RANK} {CTX_DIM}"]


    @staticmethod
    def empty() -> DecompWGenome:
        A = jnp.ones((N_CATS, DECOMP_RANK))
        B = jnp.ones((DECOMP_RANK, CTX_DIM))
        return DecompWGenome(A=A, B=B)

    @property
    def W(self) -> Float[Array, f"{N_CATS} {CTX_DIM}"]:
        return self.A @ self.B

    @staticmethod
    def random(key: PRNGKeyArray, sigma: float = 0.1) -> DecompWGenome:
        ka, kb = jr.split(key)

        A = jr.normal(ka, (N_CATS, DECOMP_RANK)) * sigma
        raw = jr.normal(kb, (CTX_DIM, DECOMP_RANK))  # tall matrix
        Q, _ = jnp.linalg.qr(raw)  # (16, 4) orthonormal columns
        B = Q.T * sigma  # (4, 16), scaled

        return DecompWGenome(A=A, B=B)

    @staticmethod
    def crossover(pa: DecompWGenome, pb: DecompWGenome, key: PRNGKeyArray) -> DecompWGenome:
        ka, kb = jr.split(key)
        # Row-wise crossover: swaps entire latent profiles for categories (A)
        # and entire basis vectors for features (B)
        mask_A = jr.bernoulli(ka, 0.5, (N_CATS, 1))
        mask_B = jr.bernoulli(kb, 0.5, (DECOMP_RANK, 1))

        return DecompWGenome(
            A=jnp.where(mask_A, pa.A, pb.A),
            B=jnp.where(mask_B, pa.B, pb.B),
        )


if GENOME_TYPE == "full":
    WGenome = FullWGenome
elif GENOME_TYPE == "decomp":
    WGenome = DecompWGenome
else:
    raise ValueError(f"Unknown GENOME_TYPE: {GENOME_TYPE!r}  (expected 'full' or 'decomp')")

# ------------------------------------------------------------------ #
#  Shared traverse / mutate / crossover                               #
# ------------------------------------------------------------------ #


def _traverse(genome: WGenome, state: KniffelState) -> Int[Scalar, ""]:
    ctx = build_context(state)
    cat_weights = genome.W @ ctx

    return oracle_action(state, cat_weights)


def _mutate(
    genome: WGenome,
    key: PRNGKeyArray,
    sigma: Float[Scalar, ""],
    p_reset: Float[Scalar, ""],
) -> WGenome:
    """Perturbs every trainable leaf."""
    leaves, treedef = jax.tree.flatten(
        eqx.filter(genome, eqx.is_array),
        is_leaf=lambda x: isinstance(x, jax.Array),
    )

    n_leaves = len(leaves)
    keys = jr.split(key, 3 * n_leaves).reshape(n_leaves, 3)

    def _perturb(leaf, ks: PRNGKeyArray):
        k1, k2, k3 = ks
        noise = jr.normal(k1, leaf.shape) * sigma
        reset_mask = jr.bernoulli(k2, p_reset, leaf.shape)
        fresh = jr.normal(k3, leaf.shape) * 0.1
        return jnp.where(reset_mask, fresh, leaf + noise)

    new_leaves = [_perturb(lf, ks) for lf, ks in zip(leaves, keys, strict=True)]
    return jax.tree.unflatten(treedef, new_leaves)


# ------------------------------------------------------------------ #
#  Shared helpers                                                     #
# ------------------------------------------------------------------ #


def oracle_action(state: KniffelState, cat_weights: Float[Scalar, " 13"]) -> Int[Scalar, ""]:
    rl = state.rolls_left.squeeze().astype(jnp.int32)
    dice_idx = state.dice_idx.astype(jnp.int32)
    open_mask = state.scorecard < 0

    # Heuristic Bonus Uplift
    upper_filled = jnp.sum(jnp.where(state.scorecard[:6] >= 0, state.scorecard[:6], 0))
    bonus_rem = jnp.clip(63.0 - upper_filled, 0.0, 63.0)
    bonus_earned = upper_filled >= 63

    def compute_roll_utilities(rolls_idx: Int[Array, ""]) -> Float[Array, ""]:
        """Computes the max possible weighted score for every roll (252) at a specific rolls_left."""
        raw_evs = EV_TABLE[:, rolls_idx, :]  # (252, 13)
        # Apply uplift to the first 6 categories
        uplift = jnp.where((raw_evs >= bonus_rem) & (jnp.arange(13) < 6) & (~bonus_earned), 35.0, 0.0)
        weighted_scores = (raw_evs + uplift) * cat_weights
        # Max over open categories for each roll
        return jnp.max(jnp.where(open_mask, weighted_scores, -1e9), axis=1)

    # Calculate "Score Now" Utility
    # We always need the current utility to decide whether to stop
    now_utilities = compute_roll_utilities(jnp.int32(0))
    val_now = now_utilities[dice_idx]
    # what to score if we stop
    score_cat = jnp.argmax(jnp.where(open_mask, (EV_TABLE[dice_idx, 0, :] + 0) * cat_weights, -1e9))

    # Calculate "Reroll" Utility
    # We look at the utilities of the outcome of the NEXT roll (rl - 1)
    # Note: EV_TABLE[dice_idx, rl] is the EV of having 'rl' rolls remaining.
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


# ------------------------------------------------------------------ #
#  Public API — dispatch on GENOME_TYPE                               #
# ------------------------------------------------------------------ #


def population_get(population: WGenome, *indices: int) -> WGenome:
    genome = population
    for idx in indices:
        genome = jax.tree.map(lambda leaf: leaf[idx], genome)
    return genome


def traverse_w(genome: WGenome, state: KniffelState) -> Int[Scalar, ""]:
    return _traverse(genome, state)


@jaxtyped(typechecker=typechecker)
def mutate_w(
    genome: WGenome,
    key: PRNGKeyArray,
    sigma: Float[Scalar, ""],
    p_reset: Float[Scalar, ""],
) -> WGenome:
    return _mutate(genome, key, sigma, p_reset)


def crossover_w(pa: WGenome, pb: WGenome, key: PRNGKeyArray) -> WGenome:
    return WGenome.crossover(pa, pb, key)


def random_w_population(key: PRNGKeyArray, pop_size: int) -> WGenome:
    keys = jr.split(key, pop_size)
    return jax.vmap(WGenome.random)(keys)
