from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray, Scalar, jaxtyped

from knifflex.game.game import KniffelState, idx_to_dice, reset, step
from knifflex.game.scoring import CAT_NAMES

from .utils import typechecker

if TYPE_CHECKING:
    from jaxtyping import Array, Int


@jaxtyped(typechecker=typechecker)
def get_action_mask(state: KniffelState) -> jnp.ndarray:
    mask = jnp.ones(45, dtype=jnp.bool_)
    mask = mask.at[:32].set(state.rolls_left > 0)
    mask = mask.at[32:].set(state.scorecard < 0)
    return mask


PolicyFn = Callable[[KniffelState], int]


def _dice_str(dice_idx: Int[Scalar, ""]) -> str:
    return " ".join(str(d) for d in np.array(idx_to_dice(dice_idx)))


def _reroll_arrow(action: Int[Scalar, ""], dice_idx: Int[Scalar, ""]) -> str:
    """Reroll bitmask + current dice arrow e.g. '[X 3 X X 5]'
    Kept dice show their value, rerolled dice show X.
    """
    dice = np.array(idx_to_dice(dice_idx))
    # action is a keep mask — bit set means KEEP
    parts = " ".join(str(dice[i]) if (action >> i) & 1 else "X" for i in range(5))
    return f"[{parts}]"


def _totals(scorecard: np.ndarray) -> tuple[int, int, int, int]:
    """Returns (upper, bonus, lower, grand)."""
    upper = int(np.sum(np.where(scorecard[:6] >= 0, scorecard[:6], 0)))
    lower = int(np.sum(np.where(scorecard[6:] >= 0, scorecard[6:], 0)))
    bonus = 35 if upper >= 63 else 0
    return upper, bonus, lower, upper + bonus + lower


def pretty_print_state(state: KniffelState) -> str:  # noqa: PLR0915
    dice_faces = {
        1: ["┌────────┐", "│        │", "│   ●    │", "│        │", "└────────┘"],
        2: ["┌────────┐", "│ ●      │", "│        │", "│     ●  │", "└────────┘"],
        3: ["┌────────┐", "│ ●      │", "│   ●    │", "│     ●  │", "└────────┘"],
        4: ["┌────────┐", "│ ●   ●  │", "│        │", "│ ●   ●  │", "└────────┘"],
        5: ["┌────────┐", "│ ●   ●  │", "│   ●    │", "│ ●   ●  │", "└────────┘"],
        6: ["┌────────┐", "│ ●   ●  │", "│ ●   ●  │", "│ ●   ●  │", "└────────┘"],
    }
    lines = []
    lines.append("╔═══════════════════════════════════════════════════════════════╗")
    lines.append("║                        🎲 KNIFFEL 🎲                          ║")
    lines.append("╠═══════════════════════════════════════════════════════════════╣")
    status = "FINISHED" if state.done else "IN PROGRESS"
    lines.append(
        f"║  Round: {int(state.round.squeeze()) + 1:2d}/13        Rolls Left: {int(state.rolls_left.squeeze())}"
        f"        Status: {status:11s} ║"
    )
    lines.append("╠═══════════════════════════════════════════════════════════════╣")
    lines.append("║  Current Dice:                                                ║")
    dice = np.array(state.dice)
    dice_lines = [[] for _ in range(5)]
    for die in dice:
        face = dice_faces[int(die)]
        for i, line in enumerate(face):
            dice_lines[i].append(line)
    for line_parts in dice_lines:
        combined = "  ".join(line_parts)
        lines.append(f"║  {combined}   ║")
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
        lines.append(f"║  │ {CAT_NAMES[i]:15s}                                    {score_str:>4s} │  ║")
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
        lines.append(f"║  │ {CAT_NAMES[i]:15s}                                    {score_str:>4s} │  ║")
    lines.append("║  ├─────────────────────────────────────────────────────────┤  ║")
    lines.append(f"║  │ Lower Total:                                        {lower_total:3d} │  ║")
    lines.append("║  └─────────────────────────────────────────────────────────┘  ║")
    grand_total = upper_total + bonus + lower_total
    lines.append("╠═══════════════════════════════════════════════════════════════╣")
    lines.append(f"║  GRAND TOTAL:                                          {grand_total:3d}    ║")
    lines.append("╚═══════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


PolicyFn = Callable[[KniffelState], int]


def _dice_str(dice_idx: Int[Array, ""]) -> str:
    return " ".join(str(d) for d in np.array(idx_to_dice(dice_idx)))


def _totals(scorecard: np.ndarray) -> tuple[int, int, int, int]:
    """Returns (upper, bonus, lower, grand)."""
    upper = int(np.sum(np.where(scorecard[:6] >= 0, scorecard[:6], 0)))
    lower = int(np.sum(np.where(scorecard[6:] >= 0, scorecard[6:], 0)))
    bonus = 35 if upper >= 63 else 0
    return upper, bonus, lower, upper + bonus + lower


def _log_lean(policy_fn: PolicyFn, key: PRNGKeyArray) -> None:
    """One line per round:
    Round  1 --- R1: 1 1 2 3 5  [X 1 X X 5]  R2: 3 1 4 6 5  [X X 4 X 5]  R3: 2 6 4 1 5  |  Fünfen  15

    Kept dice show their value in the arrow, rerolled dice show X.
    """
    state = reset(key)
    parts: list[str] = []
    lines: list[str] = []
    roll_num = 1

    while not bool(state.done):
        parts.append(f"R{roll_num}: {_dice_str(jnp.int32(state.dice_idx))}")

        prev_dice_idx = int(state.dice_idx)
        action = int(policy_fn(state))
        state, _ = step(state, jnp.int32(action))

        if action < 32:
            parts.append(_reroll_arrow(jnp.int32(action), jnp.int32(prev_dice_idx)))
            roll_num += 1
        else:
            cat = action - 32
            score = int(state.scorecard[cat])
            roll_chain = "  ".join(parts)
            lines.append(f"Round {len(lines) + 1:2d} --- {roll_chain}  |  {CAT_NAMES[cat]:<16} {score:3d}")
            parts = []
            roll_num = 1

    upper, bonus, lower, grand = _totals(np.array(state.scorecard))
    print("\n".join(lines))
    print("─" * 72)
    print(f"Total: {grand}  (upper: {upper}  bonus: {bonus}  lower: {lower})")


def _log_pretty(policy_fn: PolicyFn, key: PRNGKeyArray) -> None:
    """Print pretty_print_state after every action."""
    state = reset(key)
    print(pretty_print_state(state))

    while not bool(state.done):
        action = policy_fn(state)
        state, reward = step(state, jnp.int32(action))
        print(pretty_print_state(state))
        if action < 32:
            mask = [(action >> i) & 1 for i in range(5)]
            print(f"  ↳ reroll {mask}  (reward: {reward})")
        else:
            print(f"  ↳ scored {CAT_NAMES[action - 32]}  (reward: {reward})")


def log_game(
    policy_fn: PolicyFn,
    key: PRNGKeyArray,
    mode: Literal["lean", "fancy"] = "lean",
) -> None:
    """Play one game and log it.

    Parameters
    ----------
    policy_fn : callable (KniffelState) -> int action
    key       : JAX PRNGKey for the game
    mode      : "lean"  — one line per round (default)
                "fancy" — full pretty_print_state after every step
    """
    if mode == "lean":
        _log_lean(policy_fn, key)
    elif mode == "fancy":
        _log_pretty(policy_fn, key)
    else:
        raise ValueError(f"Unknown mode {mode!r} — choose 'lean' or 'fancy'")
