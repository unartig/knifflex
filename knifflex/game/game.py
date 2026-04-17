from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Int, Int8, PRNGKeyArray, Scalar, UInt, jaxtyped

from knifflex.utils.utils import typechecker

from .dice import N_ROLLS, DiceArray, idx_to_dice
from .ev_table import FRESH_PROBS, TRANSITION_TABLE
from .scoring import CAT_NAMES, N_CATS, score_case

"""
Optimal: 245,870775 of max. 375
"""


ScoreCardArray: TypeAlias = Int8[Array, f"{N_CATS}"]


@jaxtyped(typechecker=typechecker)
def mask_to_reroll_idx(keep_mask: Bool[Array, "5"]) -> Int[Array, ""]:
    reroll = ~keep_mask
    bits = jnp.array([1, 2, 4, 8, 16], dtype=jnp.int32)
    return jnp.sum(reroll.astype(jnp.int32) * bits).astype(jnp.int32)


def action_to_str(action: int) -> str:
    action = int(action)
    if action < 32:
        mask = [1 - ((action >> i) & 1) for i in range(5)]
        return f"Reroll {mask}"
    idx = action - 32
    if 0 <= idx < len(CAT_NAMES):
        return f"Score {CAT_NAMES[idx]}"
    return "Invalid Action"


@jaxtyped(typechecker=typechecker)
class KniffelState(eqx.Module):
    dice_idx: UInt[Array, ""]  # single scalar index replacing the 5-element dice array
    key: PRNGKeyArray
    rolls_left: UInt[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([2], dtype=jnp.uint8))
    scorecard: ScoreCardArray = eqx.field(default_factory=lambda: -jnp.ones(13, dtype=jnp.int8))
    round: UInt[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([0], dtype=jnp.uint8))
    done: Bool[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([False]))

    @property
    def dice(self) -> DiceArray:
        """Reconstruct (5,) dice array"""
        return idx_to_dice(self.dice_idx)

    @property
    def size(self) -> int:
        return self.done.shape[0]


@jaxtyped(typechecker=typechecker)
def is_reroll(action: Int[Array, ""]) -> Bool[Array, ""]:
    return action < 32


@jaxtyped(typechecker=typechecker)
def step(state: KniffelState, action: Int[Scalar, ""]) -> tuple[KniffelState, Int[Scalar, ""]]:

    def do_nothing(_: None) -> tuple[KniffelState, Int[Scalar, ""]]:
        return state, jnp.int32(0)

    def do_real_step(_: None) -> tuple[KniffelState, Int[Scalar, ""]]:
        key, subkey = jr.split(state.key)

        def do_reroll(_: None) -> tuple[KniffelState, Int[Scalar, ""]]:
            valid = state.rolls_left > 0
            mask_idx = action & jnp.int32(0x1F)  # lower 5 bits
            probs = TRANSITION_TABLE[state.dice_idx, mask_idx]  # (252,)
            new_dice_idx = jr.choice(subkey, N_ROLLS, p=probs)
            return KniffelState(
                dice_idx=jnp.where(valid.squeeze(), new_dice_idx, state.dice_idx).astype(jnp.uint8),
                rolls_left=state.rolls_left - 1,
                round=state.round,
                scorecard=state.scorecard,
                done=state.done | ~valid,
                key=key,
            ), jnp.int32(0)

        def do_score(_: None) -> tuple[KniffelState, Int[Scalar, ""]]:
            case = action - 32
            valid = state.scorecard[case] < 0

            dice = idx_to_dice(state.dice_idx)  # one gather, only on score path
            case_score = score_case(case, dice)

            is_upper = case < 6
            upper_before = jnp.sum(jnp.where(state.scorecard[:6] >= 0, state.scorecard[:6], 0))
            upper_after = upper_before + jnp.where(is_upper & valid, case_score, 0)
            bonus_triggered = (upper_before < 63) & (upper_after >= 63)
            upper_bonus = jnp.where(bonus_triggered, jnp.int32(35), jnp.int32(0))

            reward = jnp.where(
                valid,
                case_score + upper_bonus,
                -(13 - state.round).squeeze(),
            )

            new_scorecard = state.scorecard.at[case].set(
                jnp.where(valid, case_score, state.scorecard[case]).astype(jnp.int8)
            )

            # Fresh roll for next round — sample from reroll-all distribution
            new_dice_idx = jr.choice(subkey, N_ROLLS, p=FRESH_PROBS)
            return KniffelState(
                dice_idx=new_dice_idx.astype(jnp.uint8),
                scorecard=new_scorecard,
                round=jnp.uint8(state.round + 1),
                done=jnp.bool_((state.round == 12) | ~valid),
                rolls_left=jnp.array([2], dtype=jnp.uint8),
                key=key,
            ), reward

        return jax.lax.cond(is_reroll(action), do_reroll, do_score, operand=None)

    return jax.lax.cond(state.done.squeeze(), do_nothing, do_real_step, None)


@jaxtyped(typechecker=typechecker)
def reset(key: PRNGKeyArray) -> KniffelState:
    key, subkey = jr.split(key)
    dice_idx = jr.choice(subkey, N_ROLLS, p=FRESH_PROBS).astype(jnp.uint8)
    return KniffelState(
        dice_idx=dice_idx,
        key=key,
    )
