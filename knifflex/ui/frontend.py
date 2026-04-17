import os

import jax
import jax.numpy as jnp
import numpy as np

from knifflex.game.ev_table import get_ev_table
from knifflex.game.game import TRANSITION_TABLE, action_to_str, reset, step
from knifflex.game.scoring import CAT_NAMES
from knifflex.genome.w_genome import build_context, genome_action

try:
    import curses
except ImportError:
    if os.name == "nt":
        print(
            "It seems you're saying Hello from Windows, for the frontend to work properly install:\npip install windows-curses"
        )
    else:
        print("The frontend needs curses to work. Your system does currently not support it.")


# Load static tables
EV_TABLE, _, _ = (jnp.asarray(arr) for arr in get_ev_table())


class KniffelApp:
    def __init__(self, genome):
        self.genome = genome
        self.show_ai = True
        self.mode = "reroll"
        self.cursor = 0
        self.reroll_mask = [False] * 5

    def get_dice_face(self, value, status):
        dot = "●" if status == "kept" else ("○" if status == "hover" else "X")
        faces = {
            1: ["         ", f"    {dot}    ", "         "],
            2: [f" {dot}       ", "         ", f"       {dot} "],
            3: [f" {dot}       ", f"    {dot}    ", f"       {dot} "],
            4: [f" {dot}     {dot} ", "         ", f" {dot}     {dot} "],
            5: [f" {dot}     {dot} ", f"    {dot}    ", f" {dot}     {dot} "],
            6: [f" {dot}     {dot} ", f" {dot}     {dot} ", f" {dot}     {dot} "],
        }
        body = faces[int(value)]

        # Enhanced Hover Visibility: Use Double Lines for Hover, Tildes for Reroll
        if status in {"reroll_hover", "reroll"}:
            return ["~~~~~~~~~~~", f"~{body[0]}~", f"~{body[1]}~", f"~{body[2]}~", "~~~~~~~~~~~"]
        elif status == "hover":
            return ["┏━━━━━━━━━┓", f"┃{body[0]}┃", f"┃{body[1]}┃", f"┃{body[2]}┃", "┗━━━━━━━━━┛"]
        else:
            return ["┌─────────┐", f"│{body[0]}│", f"│{body[1]}│", f"│{body[2]}│", "└─────────┘"]

    def calculate_mask_idx(self):
        mask_idx = 0
        for i, reroll in enumerate(self.reroll_mask):
            if not reroll:
                mask_idx |= 1 << i
        return mask_idx

    def draw_ai_panels(self, stdscr, state, y_off):
        mask_idx = self.calculate_mask_idx()
        rl = int(state.rolls_left.squeeze())

        # 1. Math Projections
        probs = TRANSITION_TABLE[state.dice_idx, mask_idx]
        raw_evs = probs @ EV_TABLE[:, rl - 1, :]

        # 2. Genome Adjustments
        ctx = build_context(state)
        cat_weights = self.genome.W @ ctx
        cat_scales = jax.nn.softplus(self.genome.W_scale @ ctx)

        upper_filled = jnp.sum(jnp.where(state.scorecard[:6] >= 0, state.scorecard[:6], 0))
        bonus_rem = jnp.clip(63.0 - upper_filled, 0.0, 63.0)
        bonus_earned = upper_filled >= 63
        uplift = jnp.where(
            (raw_evs >= bonus_rem) & (jnp.arange(13) < 6) & (~bonus_earned), self.genome.bonus_uplift, 0.0
        )

        adj_evs = (raw_evs + uplift) * cat_scales + cat_weights

        # Find bests for highlighting (only consider open slots)
        open_mask = state.scorecard < 0
        best_raw_idx = jnp.argmax(jnp.where(open_mask, raw_evs, -1e9))
        best_adj_idx = jnp.argmax(jnp.where(open_mask, adj_evs, -1e9))

        # --- Col 2: EV TABLE ---
        x_ev = 30
        stdscr.addstr(y_off - 1, x_ev, " RAW EV │ ADJ. EV", curses.A_DIM)
        for i in range(13):
            y = y_off + i
            if not open_mask[i]:
                stdscr.addstr(y, x_ev, "   [LOCKED]   ", curses.A_DIM)
                continue

            # Highlighting Highest
            r_attr = curses.A_REVERSE if i == best_raw_idx else curses.A_NORMAL
            a_attr = curses.color_pair(2) | curses.A_BOLD if i == best_adj_idx else curses.A_NORMAL

            stdscr.addstr(y, x_ev, f" {float(raw_evs[i]):>6.1f} ", r_attr)
            stdscr.addstr(y, x_ev + 8, "│")
            stdscr.addstr(y, x_ev + 10, f"{float(adj_evs[i]):>7.1f}", a_attr)

        # --- Col 3: DECISION LOGIC ---
        x_logic = 52
        stdscr.addstr(y_off - 1, x_logic, "DECISION LOGIC", curses.A_UNDERLINE)

        # Calculate current utility vs reroll utility
        val_now = jnp.max(
            jnp.where(open_mask, (EV_TABLE[state.dice_idx, 0, :] + uplift) * cat_scales + cat_weights, -1e9)
        )
        val_reroll = adj_evs[best_adj_idx]
        diff = val_reroll - val_now

        stdscr.addstr(y_off + 1, x_logic, f"Current Val: {float(val_now):.2f}", curses.A_DIM)
        stdscr.addstr(y_off + 2, x_logic, f"Reroll  Val: {float(val_reroll):.2f}", curses.A_DIM)

        # Action recommendation
        rec_color = curses.color_pair(2) if diff > 0 and rl > 0 else curses.color_pair(3)
        rec_text = "SHOULD REROLL" if diff > 0 and rl > 0 else "SHOULD STAY"
        stdscr.addstr(y_off + 4, x_logic, "Given the current reroll selection:", rec_color | curses.A_BOLD)
        stdscr.addstr(y_off + 5, x_logic, f"» {rec_text}", rec_color | curses.A_BOLD)

        suggestion = int(genome_action(self.genome, state))
        stdscr.addstr(y_off + 7, x_logic, f"💡 AI SUGGESTS: {action_to_str(suggestion)}", curses.color_pair(2))
        stdscr.addstr(y_off + 8, x_logic, "1 reroll - 0 keep" if suggestion < 32 else "", curses.color_pair(2))

        # Strategy Focus
        focus = "Endgame" if int(state.round.squeeze()) > 9 else ("Bonus Hunt" if ctx[14] > 0.1 else "Scoring")
        stdscr.addstr(y_off + 10, x_logic, f"Mode: {focus}", curses.A_DIM)

        # Genome Bias
        bias_feat = ["Scores", "UpperSum", "BonusDist", "Rolls", "Round"][
            jnp.argmax(jnp.abs(self.genome.B.squeeze()[-5:]))
        ]
        stdscr.addstr(y_off + 12, x_logic, f"Bias: {bias_feat}", curses.A_DIM)

        # utilities = jnp.where(open_mask, cat_weights, -999)
        # top_indices = jnp.argsort(utilities)[-3:][::-1]
        # stdscr.addstr(y_off + 14, x_logic, "TOP CONSIDERATIONS:", curses.A_UNDERLINE)
        # for i, idx in enumerate(top_indices):
        #     name = CAT_NAMES[int(idx)]
        #     weight = float(utilities[int(idx)])
        #     # Create a small visual bar [#####.....]
        #     bar_len = max(0, min(10, int(weight * 5)))
        #     bar = "█" * bar_len + "░" * (10 - bar_len)
        #     stdscr.addstr(y_off + 15 + i, x_logic, f"{name[:8]:<8} {bar} {weight:4.1f}")

    def render(self, stdscr, state):
        stdscr.erase()
        stdscr.addstr(
            0,
            2,
            f"🎲 KNIFFEL | Round: {int(state.round.squeeze()) + 1}/13 | Rolls: {int(state.rolls_left.squeeze())}",
            curses.A_BOLD,
        )

        for i, val in enumerate(state.dice):
            is_hover = self.mode == "reroll" and self.cursor == i
            is_reroll = self.reroll_mask[i]

            if is_hover and is_reroll:
                status = "reroll_hover"
            elif is_hover:
                status = "hover"
            elif is_reroll:
                status = "reroll"
            else:
                status = "kept"

            face = self.get_dice_face(val, status)
            attr = curses.A_REVERSE if status in {"hover", "reroll_hover"} else curses.A_NORMAL
            for row, line in enumerate(face):
                stdscr.addstr(2 + row, 2 + (i * 12), line, attr)

        stdscr.addstr(8, 2, "SCORECARD", curses.A_UNDERLINE)
        for i, name in enumerate(CAT_NAMES):
            val = int(state.scorecard[i])
            y = 9 + i
            if self.mode == "score" and self.cursor == i:
                stdscr.addstr(y, 2, f"▶ {name.upper():<16} {('---' if val < 0 else val):>4}", curses.A_REVERSE)
            else:
                stdscr.addstr(
                    y,
                    2,
                    f"  {name:<16} {('---' if val < 0 else val):>4}",
                    curses.A_DIM if val >= 0 else curses.A_NORMAL,
                )

        if self.show_ai:
            self.draw_ai_panels(stdscr, state, 9)

        stdscr.addstr(23, 2, "WASD/HJKL: Move | SPACE: Toggle | ENTER: Confirm", curses.A_DIM)
        stdscr.addstr(24, 2, "TAB: Switch Mode | G: Toggle AI | Q: Quit", curses.A_DIM)
        stdscr.refresh()

    def play(self, stdscr, state):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.curs_set(0)

        while not bool(state.done):
            self.render(stdscr, state)
            key = stdscr.getch()
            if key in [ord("q"), ord("Q")]:
                break
            if key in [ord("g"), ord("G")]:
                self.show_ai = not self.show_ai
            if key == 9:  # TAB
                self.mode = "score" if self.mode == "reroll" else "reroll"
                self.cursor = 0

            if key in [ord("h"), ord("a")]:
                if self.mode == "reroll":
                    self.cursor = max(0, self.cursor - 1)
            elif key in [ord("l"), ord("d")]:
                if self.mode == "reroll":
                    self.cursor = min(4, self.cursor + 1)
            elif key in [ord("k"), ord("w")]:
                if self.mode == "score":
                    self.cursor = (self.cursor - 1) % 13
                    while state.scorecard[self.cursor] >= 0:
                        self.cursor = (self.cursor - 1) % 13
            elif key in [ord("j"), ord("s")] and self.mode == "score":
                self.cursor = (self.cursor + 1) % 13
                while state.scorecard[self.cursor] >= 0:
                    self.cursor = (self.cursor + 1) % 13

            if key == ord(" ") and self.mode == "reroll":
                self.reroll_mask[self.cursor] = not self.reroll_mask[self.cursor]

            if key in [10, 13]:  # ENTER
                if self.mode == "reroll":
                    action = self.calculate_mask_idx()
                    state, _ = step(state, jnp.int32(action))
                    self.reroll_mask = [False] * 5
                else:
                    action = 32 + self.cursor
                    state, _ = step(state, jnp.int32(action))
                    self.mode = "reroll"
                    self.cursor = 0


def run_game(genome):
    key = jax.random.PRNGKey(np.random.randint(0, 1000))
    state = reset(key)
    app = KniffelApp(genome)
    curses.wrapper(app.play, state)


if __name__ == "__main__":
    # Example initialization
    from knifflex.genome.cereal import load_genome

    genome = load_genome("data/runs/best.npz")
    run_game(genome)
