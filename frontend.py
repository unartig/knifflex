import curses
import jax.numpy as jnp
from scoring import CAT_NAMES

# Dice characters as requested
CHAR_KEPT = "●"
CHAR_HOVER = "○"
CHAR_REROLL = "X"


def get_dice_face(value, status):
    """
    status: 'kept', 'hover', or 'reroll'
    """
    dot = CHAR_KEPT if status == "kept" else (CHAR_HOVER if status == "hover" else CHAR_REROLL)

    # Simple layout mapping for dice faces 1-6
    # (Using the dot variable for the pips)
    faces = {
        1: ["         ", f"    {dot}    ", "         "],
        2: [f" {dot}       ", "         ", f"       {dot} "],
        3: [f" {dot}       ", f"    {dot}    ", f"       {dot} "],
        4: [f" {dot}     {dot} ", "         ", f" {dot}     {dot} "],
        5: [f" {dot}     {dot} ", f"    {dot}    ", f" {dot}     {dot} "],
        6: [f" {dot}     {dot} ", f" {dot}     {dot} ", f" {dot}     {dot} "],
    }

    body = faces[int(value)]
    return [f"┌─────────┐", f"│{body[0]}│", f"│{body[1]}│", f"│{body[2]}│", f"└─────────┘"]


def draw_screen(stdscr, state, cursor_pos, mode, reroll_mask):
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    # 1. Header
    stdscr.addstr(
        0, 2, f"🎲 KNIFFEL | Round: {int(state.round.squeeze()) + 1}/13 | Rolls Left: {int(state.rolls_left.squeeze())}", curses.A_BOLD
    )
    stdscr.addstr(1, 2, "─" * 60)

    # 2. Dice Section
    dice = state.dice
    for i in range(5):
        # Determine status for rendering
        if reroll_mask[i]:
            status = "reroll"
        elif mode == "reroll" and cursor_pos == i:
            status = "hover"
        else:
            status = "kept"

        face = get_dice_face(dice[i], status)
        start_x = 2 + (i * 12)

        # Highlight die border if hovering in reroll mode
        attr = curses.A_REVERSE if (mode == "reroll" and cursor_pos == i) else curses.A_NORMAL

        for row, line in enumerate(face):
            stdscr.addstr(3 + row, start_x, line, attr)

    # 3. Scorecard Section
    stdscr.addstr(9, 2, "SCORECARD", curses.A_UNDERLINE)
    for i, name in enumerate(CAT_NAMES):
        val = int(state.scorecard[i])
        score_str = "---" if val < 0 else f"{val:3d}"

        y = 10 + i
        if mode == "score" and cursor_pos == i:
            stdscr.addstr(y, 2, f"▶ {name.upper():<16} {score_str:>5}", curses.A_REVERSE)
        else:
            attr = curses.A_DIM if val >= 0 else curses.A_NORMAL
            stdscr.addstr(y, 2, f"  {name:<16} {score_str:>5}", attr)

    # 4. Controls Footer
    footer_y = 10 + len(CAT_NAMES) + 1
    controls = "WASD/HJKL: Move | Space: Toggle | Enter: Confirm | Q: Quit"
    stdscr.addstr(footer_y, 2, "─" * 60)
    stdscr.addstr(footer_y + 1, 2, controls, curses.A_DIM)

    stdscr.refresh()


def interactive_action(stdscr, state):
    # Determine if we should start in Reroll or Score mode
    mode = "reroll" if state.rolls_left > 0 else "score"
    cursor = 0
    reroll_mask = [False] * 5

    while True:
        draw_screen(stdscr, state, cursor, mode, reroll_mask)

        key = stdscr.getch()

        # Quit
        if key in [ord("q"), ord("Q")]:
            return None

        # Movement (HJKL / WASD)
        if key in [ord("h"), ord("a")]:  # Left
            if mode == "reroll":
                cursor = max(0, cursor - 1)
        elif key in [ord("l"), ord("d")]:  # Right
            if mode == "reroll":
                cursor = min(4, cursor + 1)
        elif key in [ord("k"), ord("w")]:  # Up
            if mode == "score":
                cursor = (cursor - 1) % 13
                while state.scorecard[cursor] >= 0:
                    cursor = (cursor - 1) % 13
        elif key in [ord("j"), ord("s")]:  # Down
            if mode == "score":
                cursor = (cursor + 1) % 13
                while state.scorecard[cursor] >= 0:
                    cursor = (cursor + 1) % 13

        # Mode Swapping (Tab or manual switch)
        elif key == 9:  # Tab
            mode = "score" if mode == "reroll" else "reroll"
            cursor = 0

        # Toggle / Select
        elif key == ord(" "):
            if mode == "reroll":
                reroll_mask[cursor] = not reroll_mask[cursor]

        # Confirm (Enter)
        elif key in [10, 13]:
            if mode == "reroll":
                action = 0
                for i, reroll in enumerate(reroll_mask):
                    if not reroll:
                        action |= 1 << i
                return action
            else:
                if state.scorecard[cursor] < 0:
                    return 32 + cursor


def play_curses(stdscr):
    # Setup curses colors and cursor
    curses.curs_set(0)

    from game import reset, step
    import jax.random as jr

    key = jr.PRNGKey(42)
    state = reset(key)

    while not bool(state.done):
        action = interactive_action(stdscr, state)
        if action is None:
            break

        state, _ = step(state, jnp.int32(action))


if __name__ == "__main__":
    curses.wrapper(play_curses)
