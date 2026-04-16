import os

import jax.numpy as jnp
import jax.random as jr

from game import KniffelState, action_to_str, reset, step
from log import get_action_mask, pretty_print_state
from scoring import CAT_NAMES
from w_genome import WGenome, genome_action


def clear() -> None:
    os.system("clear")


def render(state, show_help, genome=None) -> None:
    clear()
    print(pretty_print_state(state))

    if show_help and genome is not None and not bool(state.done):
        suggestion = int(genome_action(genome, state))
        print("\n\t💡 Suggestion:")
        print(f"\t   {suggestion}: {action_to_str(suggestion)}")


def selection_to_action(selected):
    action = 0
    for i, keep in enumerate(selected):
        if keep:
            action |= 1 << i
    return action


def choose_action(state):  # noqa: ANN201
    mask = get_action_mask(state)
    valid = [i for i, m in enumerate(mask) if m]

    print("\n\tAvailable actions:")
    for i in valid:
        print(f"{i:2d}: {action_to_str(i)}")

    while True:
        raw = input("\n\tAction (number, h=help, q=quit): ").strip()

        if raw == "q":
            return None
        if raw == "h":
            return "toggle_help"

        try:
            a = int(raw)
            if a in valid:
                return a
        except ValueError:
            pass

        print("Invalid input.")


def render_dice_selection(dice, selected) -> str:
    parts = []
    for i, d in enumerate(dice):
        if selected[i]:
            parts.append(f" {d} ")  # rerolled
        else:
            parts.append(f"[{d}]")  # kept
    return "\t\t" +" ".join(parts) + "\n"


def choose_reroll(state):
    dice = list(state.dice)
    selected = [True] * 5  # False = keep, True = reroll

    while True:
        os.system("clear")
        print(pretty_print_state(state))

        print("\n\tREROLL MODE")
        print("\tToggle dice with 1-5, ENTER to confirm, q to cancel\n\t x  keep\n\t[x] reroll\n")

        print(render_dice_selection(dice, selected))

        cmd = input("\t> ").strip()

        if cmd == "":
            return selection_to_action(selected)

        if cmd == "q":
            return None

        if cmd in ["1", "2", "3", "4", "5"]:
            i = int(cmd) - 1
            selected[i] = not selected[i]


def choose_score(state: KniffelState) -> int:
    print("\n\tSCORE MODE")

    for i, name in enumerate(CAT_NAMES):
        if state.scorecard[i] < 0:
            print(f"\t{i:2d}: {name}")
    print()

    while True:
        cmd = input("\t> ").strip()
        if cmd.isdigit():
            i = int(cmd)
            if 0 <= i < 13 and state.scorecard[i] < 0:
                return 32 + i


def play():
    key = jr.PRNGKey(0)
    state = reset(key)

    genome = WGenome.random(jr.PRNGKey(1))
    show_help = True

    while not bool(state.done):
        os.system("clear")

        print(pretty_print_state(state))

        if show_help:
            suggestion = int(genome_action(genome, state))
            print("\n\t💡 AI Suggestion:")
            print(f"\t   {action_to_str(suggestion)}")

        print("\n\tCommands: r=reroll, s=score, h=toggle help, q=quit")

        cmd = input("\t> ").strip()

        if cmd == "q":
            break

        if cmd == "h":
            show_help = not show_help
            continue

        if cmd == "r":
            action = choose_reroll(state)
            if action is not None:
                state, _ = step(state, jnp.int32(action))

        elif cmd == "s":
            action = choose_score(state)
            state, _ = step(state, jnp.int32(action))


if __name__ == "__main__":
    play()
