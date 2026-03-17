from itertools import product

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import field
from jax.experimental import checkify
from jaxtyping import Array, Bool, Int, PRNGKeyArray, Scalar, ScalarLike, jaxtyped

from utils import DiceArray, ScoreCardArray, typechecker

CASE_NAMES = [
    "Einsen",
    "Zweien",
    "Dreien",
    "Vieren",
    "Fünfen",
    "Sechsen",
    "Full-House",
    "Dreier-Pasch",
    "Vierer-Pasch",
    "Kleine Straße",
    "Große Straße",
    "Augenzahl",
    "Kniffel",
]

REROLL_LISTS = jnp.array(list(product([0, 1], repeat=5)), dtype=jnp.bool)


@jaxtyped(typechecker=typechecker)
def mask_to_reroll_idx(keep_mask: Bool[Array, "5"]) -> Int[Array, ""]:
    reroll = ~keep_mask  # (5,) bool - True means reroll this die
    bits = jnp.array([1, 2, 4, 8, 16], dtype=jnp.int32)
    return jnp.sum(reroll.astype(jnp.int32) * bits).astype(jnp.int32)


@jaxtyped(typechecker=typechecker)
def new_dice_roll(key: PRNGKeyArray) -> DiceArray:
    return jnp.sort(jr.randint(key, (5,), 1, 7))


@jaxtyped(typechecker=typechecker)
class KniffelState(eqx.Module):
    dice: DiceArray  # (5,)
    key: PRNGKeyArray
    rolls_left: Int[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([2]).astype(jnp.int32))  # 0,1,2
    scorecard: ScoreCardArray = eqx.field(default_factory=lambda: -jnp.ones(13).astype(jnp.int32))  # -1 unused
    round: Int[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([0]).astype(jnp.int32))  # 0..12
    done: Bool[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([0]).astype(jnp.bool))

    last_round_bonus: Bool[Array, "1"] = eqx.field(default_factory=lambda: jnp.array([0], dtype=jnp.bool_))

    def as_obs(self) -> Int[Array, "20"]:
        return jnp.concat([self.dice, self.rolls_left, self.scorecard, self.round], axis=-1).astype(jnp.int32)

    @property
    def size(self) -> tuple[int]:
        return self.done.shape[0]

    def pretty_print(self) -> str:
        return pretty_print_state(self)


def pretty_print_state(state: KniffelState) -> str:  # noqa: PLR0915
    dice_faces = {
        1: ["┌────────┐", "│        │", "│   ●    │", "│        │", "└────────┘"],
        2: ["┌────────┐", "│ ●      │", "│        │", "│     ●  │", "└────────┘"],
        3: ["┌────────┐", "│ ●      │", "│   ●    │", "│     ●  │", "└────────┘"],
        4: ["┌────────┐", "│ ●   ●  │", "│        │", "│ ●   ●  │", "└────────┘"],
        5: ["┌────────┐", "│ ●   ●  │", "│   ●    │", "│ ●   ●  │", "└────────┘"],
        6: ["┌────────┐", "│ ●   ●  │", "│ ●   ●  │", "│ ●   ●  │", "└────────┘"],
    }

    categories = [
        "Ones",
        "Twos",
        "Threes",
        "Fours",
        "Fives",
        "Sixes",
        "Full House",
        "3 of a Kind",
        "4 of a Kind",
        "Small Straight",
        "Large Straight",
        "Chance",
        "Kniffel",
    ]

    lines = []

    lines.append("╔═══════════════════════════════════════════════════════════════╗")
    lines.append("║                        🎲 KNIFFEL 🎲                          ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    status = "FINISHED" if state.done else "IN PROGRESS"
    lines.append(
        f"║  Round: {int(state.round) + 1:2d}/13        Rolls Left: {int(state.rolls_left)}        Status: {status:11s} ║"
    )
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    lines.append("║  Current Dice:                                                ║")
    lines.append("║                                                               ║")

    dice_lines = [[] for _ in range(5)]
    for die in state.dice:
        face = dice_faces[int(die)]
        for i, line in enumerate(face):
            dice_lines[i].append(line)

    for line_parts in dice_lines:
        combined = "  ".join(line_parts)
        lines.append(f"║  {combined}   ║")

    lines.append("║                                                               ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")
    lines.append("║  Scorecard:                                                   ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")

    lines.append("║  ┌─────────────────────────────────────────────────────────┐  ║")
    lines.append("║  │ UPPER SECTION                                           │  ║")
    lines.append("║  ├─────────────────────────────────────────────────────────┤  ║")

    upper_total = 0
    for i in range(6):
        score = int(state.scorecard[i])
        score_str = "---" if score < 0 else f"{score:3d}"
        if score >= 0:
            upper_total += score
        lines.append(f"║  │ {categories[i]:15s}                                    {score_str:>4s} │  ║")

    lines.append("║  ├─────────────────────────────────────────────────────────┤  ║")
    bonus = 35 if upper_total >= 63 else 0
    lines.append(f"║  │ Upper Subtotal:                                 {upper_total:3d}     │  ║")
    lines.append(f"║  │ Bonus (if ≥ 63):                                 {bonus:2d}     │  ║")
    lines.append(f"║  │ Upper Total:                                    {upper_total + bonus:3d}     │  ║")
    lines.append("║  └─────────────────────────────────────────────────────────┘  ║")

    lines.append("║  ┌─────────────────────────────────────────────────────────┐  ║")
    lines.append("║  │ LOWER SECTION                                           │  ║")
    lines.append("║  ├─────────────────────────────────────────────────────────┤  ║")

    lower_total = 0
    for i in range(6, 13):
        score = int(state.scorecard[i])
        score_str = "---" if score < 0 else f"{score:3d}"
        if score >= 0:
            lower_total += score
        lines.append(f"║  │ {categories[i]:15s}                                    {score_str:>4s} │  ║")

    lines.append("║  ├─────────────────────────────────────────────────────────┤  ║")
    lines.append(f"║  │ Lower Total:                                        {lower_total:3d} │  ║")
    lines.append("║  └─────────────────────────────────────────────────────────┘  ║")

    grand_total = upper_total + bonus + lower_total
    lines.append("╠═══════════════════════════════════════════════════════════════╣")
    lines.append(f"║  GRAND TOTAL:                                          {grand_total:3d}    ║")
    lines.append("╚═══════════════════════════════════════════════════════════════╝")

    return "\n".join(lines)


def action_to_str(action: int | Int[Array, ""]) -> str:
    action = action.astype(int) if isinstance(action, jnp.ndarray) else int(action)

    if action < 32:
        mask = [(action >> i) & 1 for i in range(5)]
        return f"Reroll {mask}"

    idx = action - 32
    if 0 <= idx < len(CASE_NAMES):
        return f"Kreuze {CASE_NAMES[idx]}"

    return "Invalid Action"


@jaxtyped(typechecker=typechecker)
def score_upper(dice: DiceArray, face: Int[Array, ""] | int) -> Int[Array, ""]:
    return jnp.sum(dice == face) * face


@jaxtyped(typechecker=typechecker)
def score_full_house(wurf: DiceArray) -> Int[Array, ""]:
    counts = jnp.bincount(wurf, length=7)[1:]  # skip face 0
    has_three = jnp.any(counts == 3)
    has_two = jnp.any(counts == 2)
    return jnp.where(has_three & has_two, 25, 0)


@jaxtyped(typechecker=typechecker)
def score_three_of_a_kind(dice: DiceArray) -> Int[Array, ""]:
    return jnp.where(jnp.max(jnp.bincount(dice, length=7)) >= 3, jnp.sum(dice), 0)


@jaxtyped(typechecker=typechecker)
def score_four_of_a_kind(dice: DiceArray) -> Int[Array, ""]:
    return jnp.where(jnp.max(jnp.bincount(dice, length=7)) >= 4, jnp.sum(dice), 0)


@jaxtyped(typechecker=typechecker)
def score_small_straight(wurf: DiceArray) -> Int[Array, ""]:
    present = jnp.bincount(wurf, length=7)[1:] > 0  # shape (6,)
    # check 4 consecutive present
    has = jnp.any(jnp.stack([
        present[0] & present[1] & present[2] & present[3],
        present[1] & present[2] & present[3] & present[4],
        present[2] & present[3] & present[4] & present[5],
    ]))
    return jnp.where(has, 30, 0)


@jaxtyped(typechecker=typechecker)
def score_large_straight(wurf: DiceArray) -> Int[Array, ""]:
    present = jnp.bincount(wurf, length=7)[1:] > 0
    has = (present[0] & present[1] & present[2] & present[3] & present[4]) | \
          (present[1] & present[2] & present[3] & present[4] & present[5])
    return jnp.where(has, 40, 0)


@jaxtyped(typechecker=typechecker)
def score_faces(wurf: DiceArray) -> Int[Array, ""]:
    return jnp.sum(wurf)


@jaxtyped(typechecker=typechecker)
def score_kniffel(dice: DiceArray) -> Int[Array, ""]:
    return jnp.where(jnp.all(dice == dice[0]), 50, 0)


@jaxtyped(typechecker=typechecker)
def score_case(case_id: Int[Array, ""], dice: DiceArray) -> Int[Array, ""]:
    return jax.lax.switch(
        case_id,
        [
            lambda d: score_upper(d, 1),
            lambda d: score_upper(d, 2),
            lambda d: score_upper(d, 3),
            lambda d: score_upper(d, 4),
            lambda d: score_upper(d, 5),
            lambda d: score_upper(d, 6),
            score_full_house,
            score_three_of_a_kind,
            score_four_of_a_kind,
            score_small_straight,
            score_large_straight,
            score_faces,
            score_kniffel,
        ],
        dice,
    )


@jaxtyped(typechecker=typechecker)
def is_reroll(action: Int[Array, ""]) -> Bool[Array, ""]:
    return action < 32


@jaxtyped(typechecker=typechecker)
def step(state: KniffelState, action: Int[Array, ""]) -> tuple[KniffelState, Int[Array, ""]]:

    @jaxtyped(typechecker=typechecker)
    def do_nothing(_) -> tuple[KniffelState, Int[Array, ""]]:
        return state, jnp.int32(0)

    @jaxtyped(typechecker=typechecker)
    def do_real_step(_) -> tuple[KniffelState, Int[Array, ""]]:
        key, subkey = jr.split(state.key)

        @jaxtyped(typechecker=typechecker)
        def do_reroll(_) -> tuple[KniffelState, Int[Array, ""]]:
            # checkify.check(state.rolls_left > 0, "Reroll attempted with 0 rolls left!")
            valid = state.rolls_left > 0
            mask = jnp.array(
                [(action >> i) & 1 for i in range(5)],
                dtype=jnp.bool_,
            )

            new_dice = jnp.where(
                mask,
                new_dice_roll(subkey),
                state.dice,
            )

            return KniffelState(
                dice=jnp.sort(new_dice),
                rolls_left=state.rolls_left - 1,
                round=state.round,
                scorecard=state.scorecard,
                done=state.done | ~valid,
                key=key,
            ), jnp.int32(0)

        @jaxtyped(typechecker=typechecker)
        def do_score(_) -> tuple[KniffelState, Int[Array, ""]]:
            case = action - 32
            valid = state.scorecard[case] < 0

            # is_last_round = state.round == 12
            # last_round_bonus = jnp.where(
            #     state.last_round_bonus & valid & is_last_round,
            #     jnp.array(50),
            #     jnp.array(0),
            # )
            case_score = score_case(case, state.dice)

            is_upper = case < 6
            upper_before = jnp.sum(jnp.where(state.scorecard[:6] >= 0, state.scorecard[:6], 0))
            upper_after = upper_before + jnp.where(is_upper & valid, case_score, 0)
            bonus_triggered = (upper_before < 63) & (upper_after >= 63)
            upper_bonus = jnp.where(bonus_triggered, jnp.int32(35), jnp.int32(0))

            reward = jnp.where(
                valid,
                case_score + upper_bonus,  # + last_round_bonus,
                -(13 - state.round).squeeze(),  # punish early invalids
            )

            new_scorecard = state.scorecard.at[case].set(jnp.where(valid, case_score, state.scorecard[case]))

            return KniffelState(
                dice=new_dice_roll(subkey),
                rolls_left=jnp.int32([2]),
                scorecard=new_scorecard,
                round=jnp.int32(state.round + 1),
                done=jnp.bool_((state.round == 12) | ~valid),
                key=key,
            ), reward

        return jax.lax.cond(
            is_reroll(action),
            do_reroll,
            do_score,
            operand=None,
        )

    return jax.lax.cond(state.done.squeeze(), do_nothing, do_real_step, None)


@jaxtyped(typechecker=typechecker)
def reset(key: PRNGKeyArray, last_round_bonus: bool = False) -> KniffelState:
    key, subkey = jr.split(key)
    dice = new_dice_roll(subkey)

    return KniffelState(dice=jnp.sort(dice), key=key, last_round_bonus=jnp.array([last_round_bonus]))


@jaxtyped(typechecker=typechecker)
def get_action_mask(state: KniffelState) -> jnp.ndarray:
    mask = jnp.ones(45, dtype=jnp.bool_)  # 32 rerolls + 13 scores

    mask = mask.at[:32].set(state.rolls_left > 0)

    mask = mask.at[32:].set(state.scorecard < 0)

    return mask


class KniffelGym(gym.Env):
    def __init__(self) -> None:
        self.key = jr.PRNGKey(0)
        self.state = reset(self.key)

    def step(self, action: jnp.int32) -> tuple[DiceArray, float, bool, bool, dict]:
        self.state, reward = step(self.state, action)
        return (
            self.state.as_obs(),
            float(reward),
            bool(self.state.done),
            False,
            {},
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[DiceArray, dict]:
        if seed is not None:
            self.key = jr.PRNGKey(seed)
        self.state = reset(self.key)
        return self.state.as_obs(), {}
