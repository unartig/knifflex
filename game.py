from itertools import combinations_with_replacement, product

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Bool, Int, PRNGKeyArray, UInt, jaxtyped

from utils import DiceArray, ScoreCardArray, typechecker

"""
Optimal: 245,870775 of max. 375
"""
# ------------------------------------------------------------------ #
#  Roll catalogue  (252 sorted 5-dice combos)                        #
# ------------------------------------------------------------------ #

_ALL_ROLLS_NP: np.ndarray = np.array(list(combinations_with_replacement(range(1, 7), 5)), dtype=np.int32)  # (252, 5)
N_ROLLS: int = 252

ROLLS_TABLE: Int[Array, "252 5"] = jnp.array(_ALL_ROLLS_NP, dtype=jnp.int32)

# O(1) dice → index via base-7 encoding
_POW7: Int[Array, "5"] = jnp.array([7**i for i in range(5)], dtype=jnp.int32)
_ROLL_LOOKUP: Int[Array, "16807"] = (
    -jnp.ones(7**5, dtype=jnp.int32).at[jnp.sum(ROLLS_TABLE * _POW7, axis=1)].set(jnp.arange(N_ROLLS, dtype=jnp.int32))
)


def dice_to_idx(dice: DiceArray) -> Int[Array, ""]:
    """Sorted dice array → roll index (0-251).  O(1)."""
    return _ROLL_LOOKUP[jnp.sum(jnp.sort(dice).astype(jnp.int32) * _POW7)]


def idx_to_dice(idx: Int[Array, ""]) -> DiceArray:
    """Roll index → sorted dice array (5,)."""
    return ROLLS_TABLE[idx]


# ------------------------------------------------------------------ #
#  Transition distribution                                           #
# ------------------------------------------------------------------ #
#
#  TRANSITION_TABLE[roll_idx, mask_idx, next_roll_idx] — float32
#  Built by ev_table.get_transition_tensor() and cached to disk.
#
#  FRESH_PROBS[j] = P(roll j | reroll all 5 dice)
#  = TRANSITION_TABLE[0, 31, j]   (mask 31 = all bits set = reroll all)


def _load_transition() -> tuple:
    from ev_table import get_transition_tensor  # noqa: PLC0415 — deferred to avoid circular import

    T_np = get_transition_tensor()  # (252, 32, 252) float32
    T = jnp.array(T_np, dtype=jnp.float32)
    fresh = T[0, 0]  # (252,)
    return T, fresh


TRANSITION_TABLE, FRESH_PROBS = _load_transition()

# ------------------------------------------------------------------ #
#  Action helpers                                                     #
# ------------------------------------------------------------------ #

REROLL_LISTS = jnp.array(list(product([0, 1], repeat=5)), dtype=jnp.bool_)


@jaxtyped(typechecker=typechecker)
def mask_to_reroll_idx(keep_mask: Bool[Array, "5"]) -> Int[Array, ""]:
    reroll = ~keep_mask
    bits = jnp.array([1, 2, 4, 8, 16], dtype=jnp.int32)
    return jnp.sum(reroll.astype(jnp.int32) * bits).astype(jnp.int32)


def action_to_str(action: int) -> str:
    action = int(action)
    if action < 32:
        mask = [(action >> i) & 1 for i in range(5)]
        return f"Reroll {mask}"
    idx = action - 32
    if 0 <= idx < len(CASE_NAMES):
        return f"Kreuze {CASE_NAMES[idx]}"
    return "Invalid Action"


# ------------------------------------------------------------------ #
#  KniffelState                                                       #
# ------------------------------------------------------------------ #


@jaxtyped(typechecker=typechecker)
class KniffelState(eqx.Module):
    # Core field — single scalar index replacing the 5-element dice array
    dice_idx: UInt[Array, ""]
    key: PRNGKeyArray
    rolls_left: UInt[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([2], dtype=jnp.uint8))
    scorecard: ScoreCardArray = eqx.field(default_factory=lambda: -jnp.ones(13, dtype=jnp.int8))
    round: UInt[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([0], dtype=jnp.uint8))
    done: Bool[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([False]))
    last_round_bonus: Bool[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([False]))

    @property
    def dice(self) -> DiceArray:
        """Reconstruct (5,) dice array on demand — for display and gym compat."""
        return idx_to_dice(self.dice_idx)

    def as_obs(self) -> Int[Array, "20"]:
        return jnp.concat([self.dice, self.rolls_left, self.scorecard, self.round], axis=-1).astype(jnp.int32)

    @property
    def size(self) -> tuple:
        return self.done.shape[0]

# ------------------------------------------------------------------ #
#  Score functions — re-exported from scoring.py                     #
# ------------------------------------------------------------------ #
# Import everything so callers can still do `from game import score_case` etc.
from scoring import (  # noqa: E402
    CASE_NAMES,
    score_case,
    score_faces,  # noqa: F401
    score_four_of_a_kind,  # noqa: F401
    score_full_house,  # noqa: F401
    score_kniffel,  # noqa: F401
    score_large_straight,  # noqa: F401
    score_small_straight,  # noqa: F401
    score_three_of_a_kind,  # noqa: F401
    score_upper,  # noqa: F401
)


@jaxtyped(typechecker=typechecker)
def is_reroll(action: Int[Array, ""]) -> Bool[Array, ""]:
    return action < 32


# ------------------------------------------------------------------ #
#  step  — hot path, index-based, no dice arithmetic                 #
# ------------------------------------------------------------------ #


@jaxtyped(typechecker=typechecker)
def step(state: KniffelState, action: Int[Array, ""]) -> tuple[KniffelState, Int[Array, ""]]:

    def do_nothing(_):
        return state, jnp.int32(0)

    def do_real_step(_):
        key, subkey = jr.split(state.key)

        def do_reroll(_):
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

        def do_score(_):
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


# ------------------------------------------------------------------ #
#  reset                                                              #
# ------------------------------------------------------------------ #


@jaxtyped(typechecker=typechecker)
def reset(key: PRNGKeyArray, last_round_bonus: bool = False) -> KniffelState:
    key, subkey = jr.split(key)
    dice_idx = jr.choice(subkey, N_ROLLS, p=FRESH_PROBS).astype(jnp.uint8)
    return KniffelState(
        dice_idx=dice_idx,
        key=key,
        last_round_bonus=jnp.array([last_round_bonus]),
    )


# ------------------------------------------------------------------ #
#  KniffelGym                                                        #
# ------------------------------------------------------------------ #


class KniffelGym(gym.Env):
    def __init__(self) -> None:
        self.key = jr.PRNGKey(0)
        self.state = reset(self.key)

    def step(self, action: jnp.int32) -> tuple:
        self.state, reward = step(self.state, action)
        return self.state.as_obs(), float(reward), bool(self.state.done), False, {}

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple:
        if seed is not None:
            self.key = jr.PRNGKey(seed)
        self.state = reset(self.key)
        return self.state.as_obs(), {}
