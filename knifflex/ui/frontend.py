import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from knifflex.game.ev_table import get_ev_table
from knifflex.game.game import TRANSITION_TABLE, action_to_str, reset, step
from knifflex.game.scoring import CAT_NAMES, N_CATS
from knifflex.genome.w_genome import build_context

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

# ---------------------------------------------------------------------------
# Inspector color pairs (initialised in play())
#   4 = strong negative  (blue, dim)
#   5 = weak negative    (normal)
#   6 = near-zero        (dim)
#   7 = weak positive    (cyan)
#   8 = strong positive  (yellow, bold)
# ---------------------------------------------------------------------------
INSP_PAIRS = {
    "neg_strong": 4,
    "neg_weak": 5,
    "zero": 6,
    "pos_weak": 7,
    "pos_strong": 8,
}

CTX_NAMES = [
    "sc:ones",
    "sc:twos",
    "sc:threes",
    "sc:fours",
    "sc:fives",
    "sc:sixes",
    "sc:fhouse",
    "sc:3kind",
    "sc:4kind",
    "sc:sstr",
    "sc:lstr",
    "sc:chance",
    "sc:kniffel",
    "upper_sum",
    "bonus_dist",
    "rolls_left",
    "rounds_left",
]

# Scorecard layout constants
N_UPPER = 6  # Ones … Sixes
N_LOWER = 7  # Full House … Kniffel
BONUS_THRESHOLD = 63

# Rows in the scorecard widget (including separators and summary lines):
#   header
#   6 upper cats
#   upper-total separator + row
#   bonus separator + row
#   lower separator label
#   7 lower cats
#   lower-total separator + row
#   grand-total separator + row
# = 1 + 6 + 2 + 2 + 1 + 7 + 2 + 2 = 23 rows
SCORE_WIDGET_HEIGHT = 23


def _val_to_pair(v: float, vmax: float) -> int:
    """Map a signed float to one of 5 inspector color pairs."""
    if vmax < 1e-9:
        return INSP_PAIRS["zero"]
    t = v / vmax  # in [-1, 1]
    if t < -0.4:
        return INSP_PAIRS["neg_strong"]
    if t < -0.1:
        return INSP_PAIRS["neg_weak"]
    if t < 0.1:
        return INSP_PAIRS["zero"]
    if t < 0.4:
        return INSP_PAIRS["pos_weak"]
    return INSP_PAIRS["pos_strong"]


def _val_to_attr(v: float, vmax: float) -> int:
    pair = _val_to_pair(v, vmax)
    bold = pair in (INSP_PAIRS["neg_strong"], INSP_PAIRS["pos_strong"])
    return curses.color_pair(pair) | (curses.A_BOLD if bold else curses.A_NORMAL)


# ---------------------------------------------------------------------------
# Data classes / named tuples for passing computed data between helpers
# ---------------------------------------------------------------------------


class AIContext:
    """All pre-computed values needed by the AI panel."""

    __slots__ = (
        "ctx",
        "cat_weights",
        "cat_scales",
        "upper_filled",
        "bonus_rem",
        "bonus_earned",
        "open_mask",
        "rl",
    )

    def __init__(self, genome, state):
        self.rl = int(state.rolls_left.squeeze())
        self.open_mask = state.scorecard < 0
        self.ctx = build_context(state)
        self.cat_weights, self.cat_scales = genome.get_scale_and_weight(self.ctx)
        self.upper_filled = jnp.sum(jnp.where(state.scorecard[:6] >= 0, state.scorecard[:6], 0))
        self.bonus_rem = jnp.clip(63.0 - self.upper_filled, 0.0, 63.0)
        self.bonus_earned = self.upper_filled >= 63


class ColumnData:
    """EV data for one display column."""

    __slots__ = ("label", "raw_ev", "adj_ev", "best_idx", "available")

    def __init__(self, label, raw_ev, adj_ev, best_idx, available):
        self.label = label
        self.raw_ev = raw_ev
        self.adj_ev = adj_ev
        self.best_idx = best_idx
        self.available = available


class DecisionData:
    """AI decision summary for the right-hand panel."""

    __slots__ = ("val_now", "val_reroll", "should_reroll", "suggestion", "mode", "bias", "ai_mask_idx")

    def __init__(self, val_now, val_reroll, should_reroll, suggestion, mode, bias, ai_mask_idx):
        self.val_now = val_now
        self.val_reroll = val_reroll
        self.should_reroll = should_reroll
        self.suggestion = suggestion
        self.mode = mode
        self.bias = bias
        self.ai_mask_idx = ai_mask_idx


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def _compute_adj_ev_with_uplift(genome, raw_ev, ai_ctx: AIContext):
    """Apply bonus uplift (genome-aware) + scale/weight."""
    uplift = jnp.where(
        (raw_ev >= ai_ctx.bonus_rem) & (jnp.arange(13) < 6) & (~ai_ctx.bonus_earned),
        genome.bonus_uplift,
        0.0,
    )
    return (raw_ev + uplift) * ai_ctx.cat_scales + ai_ctx.cat_weights


def compute_column_data(genome, state, ai_ctx: AIContext, mask_idx: int) -> list[ColumnData]:
    """
    Compute the three display columns (score-now, reroll→1, reroll→2) for the
    user's currently selected reroll mask.

    Returns a list of three ColumnData objects.
    """
    probs = TRANSITION_TABLE[state.dice_idx, mask_idx]  # (252,)

    def make_column(label, rolls_remaining, available):
        if rolls_remaining == 0:
            raw = EV_TABLE[state.dice_idx, 0, :]
        else:
            raw = probs @ EV_TABLE[:, rolls_remaining - 1, :]
        adj = _compute_adj_ev_with_uplift(genome, raw, ai_ctx)
        best = int(jnp.argmax(jnp.where(ai_ctx.open_mask, adj, -1e9)))
        return ColumnData(label, raw, adj, best, available)

    return [
        make_column("SCORE NOW", 0, True),
        make_column("REROLL →1", 1, ai_ctx.rl >= 1),
        make_column("REROLL →2", 2, ai_ctx.rl >= 2),
    ]


def compute_decision_data(genome, state, ai_ctx: AIContext) -> DecisionData:
    """
    Compute AI decision metrics mirroring oracle_action exactly.

    val_now   = best adj EV scoreable right now (no reroll)
    val_reroll = E[best adj EV after optimal reroll] under AI's chosen mask
    Both are expected utilities, comparable on the same scale.
    """
    rl = ai_ctx.rl

    # Score-now: best adj EV for current dice (rolls_remaining = 0)
    raw_now = EV_TABLE[state.dice_idx, 0, :]
    adj_now = _compute_adj_ev_with_uplift(genome, raw_now, ai_ctx)
    val_now = float(jnp.max(jnp.where(ai_ctx.open_mask, adj_now, -1e9)))

    # Future utilities: for every possible dice outcome, what's the best adj EV
    # with (rl-1) rolls remaining — exactly what oracle_action uses.
    ai_mask_idx = 0
    val_reroll = val_now  # fallback: no improvement possible
    if rl > 0:
        all_raw = EV_TABLE[:, rl - 1, :]  # (252, 13)
        uplift = jnp.where(
            (all_raw >= ai_ctx.bonus_rem) & (jnp.arange(13) < 6) & (~ai_ctx.bonus_earned),
            genome.bonus_uplift,
            0.0,
        )
        all_adj = (all_raw + uplift) * ai_ctx.cat_scales + ai_ctx.cat_weights  # (252, 13)
        future_utils = jnp.max(jnp.where(ai_ctx.open_mask, all_adj, -1e9), axis=1)  # (252,)

        # E[future utility] for each of the 32 keep-masks
        expected = TRANSITION_TABLE[state.dice_idx] @ future_utils  # (32,)
        best_mask = int(jnp.argmax(expected))
        val_reroll = float(expected[best_mask])
        ai_mask_idx = best_mask

    should_reroll = rl > 0 and val_reroll > val_now
    suggestion = int(genome.oracle_action(state))

    mode = "Endgame" if int(state.round.squeeze()) > 9 else "Bonus Hunt" if float(ai_ctx.ctx[14]) > 0.1 else "Scoring"
    bias_feat = ["Scores", "UpperSum", "BonusDist", "Rolls", "Round"][
        int(jnp.argmax(jnp.abs(genome.raw_b.squeeze()[-5:])))
    ]

    return DecisionData(val_now, val_reroll, should_reroll, suggestion, mode, bias_feat, ai_mask_idx)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

COL_W = 20  # width of each EV column
CAT_NAME_W = 10  # width of the category name prefix column

def draw_ev_columns(stdscr, columns: list[ColumnData], open_mask, y_off: int, x_start: int):
    """
    Render category names followed by three EV columns.  Each column shows:
      adj_ev  (bright, primary)   raw_ev  (dimmed, secondary)
    """
    # Layout per row:
    #   x_start                        : category name (CAT_NAME_W chars)
    #   x_start + CAT_NAME_W + c*COL_W :
    #     +0  : marker ("▶ " or "  ")
    #     +2  : adj value (6 chars)
    #     +10 : raw value (6 chars)
    ADJ_X = 2
    RAW_X = 10
    x_cols = x_start + CAT_NAME_W

    # Header rows
    stdscr.addstr(y_off - 2, x_start, "─" * CAT_NAME_W, curses.A_DIM)
    stdscr.addstr(y_off - 1, x_start, " " * CAT_NAME_W, curses.A_DIM)
    for c, col in enumerate(columns):
        x = x_cols + c * COL_W
        attr = curses.A_NORMAL if col.available else curses.A_DIM
        stdscr.addstr(y_off - 2, x, "─" * (COL_W - 1), attr)
        stdscr.addstr(y_off - 1, x, f"{col.label:<{ADJ_X + 6}}", attr | curses.A_UNDERLINE)
        stdscr.addstr(y_off - 1, x + RAW_X, f"{'raw':>6}  ", curses.A_DIM)

    for i in range(13):
        y = y_off + i
        name = CAT_NAMES[i][: CAT_NAME_W - 1]
        if not open_mask[i]:
            stdscr.addstr(y, x_start, f"{name:<{CAT_NAME_W}}", curses.A_DIM)
            stdscr.addstr(y, x_cols, "[LOCKED]" + " " * (COL_W * 3 - 8), curses.A_DIM)
            continue

        stdscr.addstr(y, x_start, f"{name:<{CAT_NAME_W}}", curses.A_DIM)
        for c, col in enumerate(columns):
            x = x_cols + c * COL_W
            adj_val = float(col.adj_ev[i])
            raw_val = float(col.raw_ev[i])
            is_best = i == col.best_idx

            if not col.available:
                stdscr.addstr(y, x, f"  {adj_val:>6.1f}  {'':>6}  ", curses.A_DIM)
            elif is_best:
                stdscr.addstr(y, x, f"> {adj_val:>6.1f}  ", curses.color_pair(2) | curses.A_BOLD)
                stdscr.addstr(y, x + RAW_X, f"{raw_val:>6.1f}  ", curses.A_DIM)
            else:
                stdscr.addstr(y, x, f"  {adj_val:>6.1f}  ", curses.A_NORMAL)
                stdscr.addstr(y, x + RAW_X, f"{raw_val:>6.1f}  ", curses.A_DIM)

    # Legend below the EV table
    legend_y = y_off + 13
    stdscr.addstr(legend_y, x_start, "> best  ", curses.color_pair(2) | curses.A_BOLD)
    stdscr.addstr(legend_y, x_start + 8, "adj = weighted EV   raw = expected score", curses.A_DIM)


def draw_decision_panel(stdscr, decision: DecisionData, y_off: int, x: int):
    """Render the right-hand decision logic panel."""
    stdscr.addstr(y_off - 1, x, "DECISION LOGIC", curses.A_UNDERLINE)

    stdscr.addstr(y_off + 1, x, f"Score now:      {decision.val_now:>7.2f}", curses.A_DIM)
    stdscr.addstr(y_off + 2, x, f"E[reroll→best]: {decision.val_reroll:>7.2f}", curses.A_DIM)

    rec_color = curses.color_pair(2) if decision.should_reroll else curses.color_pair(3)
    rec_text = "SHOULD REROLL" if decision.should_reroll else "SHOULD STAY"
    stdscr.addstr(y_off + 5, x, rec_text, rec_color | curses.A_BOLD)

    stdscr.addstr(y_off + 10, x, f"Mode: {decision.mode}", curses.A_DIM)
    stdscr.addstr(y_off + 11, x, f"Bias: {decision.bias}", curses.A_DIM)


# ---------------------------------------------------------------------------
# Inspector views
# ---------------------------------------------------------------------------


def _safe_addstr(stdscr, y, x, text, attr=curses.A_NORMAL):
    """addstr that silently ignores out-of-bounds writes."""
    with contextlib.suppress(curses.error):
        stdscr.addstr(y, x, text, attr)


def draw_inspector_w(stdscr, genome, view: str, row_cursor: int):
    """
    Full-screen inspector.  view in {'w', 'wscale', 'ctx', 'row'}.
    row_cursor selects the highlighted category in 'row' view.
    """
    max_y, max_x = stdscr.getmaxyx()
    stdscr.erase()

    views = [("W", "w"), ("W_scale", "wscale"), ("ctx", "ctx"), ("row profile", "row")]
    header = "  GENOME INSPECTOR   "
    for label, key in views:
        marker = "▶ " if key == view else "  "
        header += f"{marker}{label}  "
    _safe_addstr(stdscr, 0, 0, header[: max_x - 1], curses.A_BOLD)
    _safe_addstr(stdscr, 1, 0, "[ / ]: switch view   hjkl: scroll row   I: close", curses.A_DIM)

    if view == "w":
        _draw_insp_matrix(stdscr, genome, "w", row_cursor, max_y, max_x)
    elif view == "wscale":
        _draw_insp_matrix(stdscr, genome, "wscale", row_cursor, max_y, max_x)
    elif view == "ctx":
        _draw_insp_ctx(stdscr, genome, max_y, max_x)
    elif view == "row":
        _draw_insp_row(stdscr, genome, row_cursor, max_y, max_x)

    stdscr.refresh()


def _draw_insp_matrix(stdscr, genome, which: str, row_cursor: int, max_y: int, max_x: int):
    """Render the W or W_scale matrix as a colour heatmap."""
    mat = np.array(genome.w if which == "w" else genome.w_scale)  # (13, 17)
    vmax = float(np.max(np.abs(mat))) if which == "w" else float(np.max(mat))

    CELL_W = 7
    ROW_LABEL_W = 14

    y = 2
    _safe_addstr(stdscr, y, 0, f"{'':>{ROW_LABEL_W}}", curses.A_DIM)
    for ci, cname in enumerate(CTX_NAMES):
        x = ROW_LABEL_W + ci * CELL_W
        _safe_addstr(stdscr, y, x, f"{cname[: CELL_W - 1]:^{CELL_W}}", curses.A_DIM)

    for ri in range(N_CATS):
        y = 3 + ri
        if y >= max_y - 2:
            break
        is_sel = ri == row_cursor
        row_attr = curses.A_REVERSE if is_sel else curses.A_NORMAL
        label = CAT_NAMES[ri]
        _safe_addstr(stdscr, y, 0, f"{label:>{ROW_LABEL_W}}", row_attr)

        for ci in range(len(CTX_NAMES)):
            x = ROW_LABEL_W + ci * CELL_W
            v = float(mat[ri, ci])
            if which == "w":
                attr = _val_to_attr(v, vmax)
                cell = f"{v:+.1f}".center(CELL_W)
            else:
                attr = _val_to_attr(v - 1.0, max(vmax - 1.0, 1e-9))
                cell = f"{v:.3f}".center(CELL_W)
            _safe_addstr(stdscr, y, x, cell, attr)

    legend_y = max_y - 2
    _safe_addstr(stdscr, legend_y, 0, "  ◀ neg_strong   neg_weak   ~zero   pos_weak   pos_strong ▶", curses.A_DIM)
    for i, (label, pair) in enumerate(
        [
            ("■ neg", INSP_PAIRS["neg_strong"]),
            ("■ neg", INSP_PAIRS["neg_weak"]),
            ("■ zer", INSP_PAIRS["zero"]),
            ("■ pos", INSP_PAIRS["pos_weak"]),
            ("■ pos", INSP_PAIRS["pos_strong"]),
        ]
    ):
        _safe_addstr(stdscr, legend_y, 2 + i * 12, label, curses.color_pair(pair))


def _draw_insp_ctx(stdscr, genome, max_y: int, max_x: int):
    """Show the context vector layout with example annotation."""
    _safe_addstr(stdscr, 2, 0, "Context vector — 17 dims passed to genome at each decision", curses.A_DIM)
    _safe_addstr(stdscr, 3, 0, f"  {'DIM':<4} {'NAME':<14} {'RANGE':<14} {'MEANING'}", curses.A_UNDERLINE)

    rows = [
        ("0–12", "sc:<cat>", "[0, 1]", "filled score / cat_max, else 0"),
        ("13", "upper_sum", "[0, 1]", "Σ upper scores / 63"),
        ("14", "bonus_dist", "[0, 1]", "clip((63 − upper_filled) / 63, 0, 1)"),
        ("15", "rolls_left", "{0, 0.5, 1}", "rolls_left / 2"),
        ("16", "rounds_left", "[0, 1]", "round / 12"),
    ]
    for i, (dim, name, rng, meaning) in enumerate(rows):
        y = 4 + i * 2
        if y >= max_y - 2:
            break
        _safe_addstr(stdscr, y, 2, f"{dim:<4} {name:<14} {rng:<14} {meaning}", curses.A_NORMAL)

    y = 4 + len(rows) * 2 + 1
    _safe_addstr(stdscr, y, 0, "B vector  (context feature directions, rank-1 decomp):", curses.A_UNDERLINE)
    b = np.array(genome.raw_b).squeeze()
    vmax = float(np.max(np.abs(b))) + 1e-9
    BAR_W = 30
    for ci, (name, val) in enumerate(zip(CTX_NAMES, b)):
        vy = y + 2 + ci
        if vy >= max_y - 1:
            break
        filled = int(abs(float(val)) / vmax * BAR_W)
        bar = ("█" * filled).ljust(BAR_W)
        attr = _val_to_attr(float(val), vmax)
        _safe_addstr(stdscr, vy, 2, f"{name:<14}", curses.A_DIM)
        _safe_addstr(stdscr, vy, 16, bar, attr)
        _safe_addstr(stdscr, vy, 16 + BAR_W + 1, f"{float(val):+.3f}", attr)


def _draw_insp_row(stdscr, genome, row_cursor: int, max_y: int, max_x: int):
    """Show the weight profile for the selected category."""
    cat_name = CAT_NAMES[row_cursor]
    w_row = np.array(genome.w)[row_cursor]
    ws_row = np.array(genome.w_scale)[row_cursor]
    vmax_w = float(np.max(np.abs(w_row))) + 1e-9

    _safe_addstr(stdscr, 2, 0, f"Category: {cat_name}   (hjkl to change)", curses.A_BOLD)
    _safe_addstr(stdscr, 3, 0, f"  {'CTX DIM':<14} {'W (weight)':>12}  {'bar':^32}  {'W_scale':>8}", curses.A_UNDERLINE)

    BAR_W = 30
    for ci, (name, wv, wsv) in enumerate(zip(CTX_NAMES, w_row, ws_row)):
        y = 4 + ci
        if y >= max_y - 2:
            break
        filled = int(abs(wv) / vmax_w * BAR_W)
        bar = ("▌" if wv < 0 else "▐") * filled
        bar = bar.ljust(BAR_W) if wv >= 0 else bar.rjust(BAR_W)
        attr = _val_to_attr(wv, vmax_w)
        _safe_addstr(stdscr, y, 2, f"{name:<14}", curses.A_DIM)
        _safe_addstr(stdscr, y, 16, f"{wv:>+8.2f}", attr)
        _safe_addstr(stdscr, y, 25, f"  {bar}  ", attr)
        _safe_addstr(stdscr, y, 25 + BAR_W + 4, f"{wsv:>6.3f}×", curses.A_DIM)


def draw_all_masks(stdscr, genome, state, y, x, selected_idx, best_idx):
    ai_ctx = AIContext(genome, state)
    rl = ai_ctx.rl

    if rl == 0:
        return

    all_raw = EV_TABLE[:, rl - 1, :]
    uplift = jnp.where(
        (all_raw >= ai_ctx.bonus_rem) & (jnp.arange(13) < 6) & (~ai_ctx.bonus_earned),
        genome.bonus_uplift,
        0.0,
    )
    all_adj = (all_raw + uplift) * ai_ctx.cat_scales + ai_ctx.cat_weights

    # Bug fix: mask out already-filled categories so best_cat_idx only considers open slots
    all_adj_open = jnp.where(ai_ctx.open_mask, all_adj, -1e9)
    expected_per_cat = TRANSITION_TABLE[state.dice_idx] @ all_adj_open

    best_cat_idx = jnp.argmax(expected_per_cat, axis=1)  # (32,)
    best_cat_val = jnp.max(expected_per_cat, axis=1)  # (32,)

    stdscr.addstr(y, x, "ALL MASKS", curses.A_UNDERLINE)

    for i in range(32):
        row = y + 1 + i
        if row >= curses.LINES - 1:
            break

        mask = [1 - ((i >> b) & 1) for b in range(5)]
        ev = float(best_cat_val[i])
        cat = CAT_NAMES[int(best_cat_idx[i])]

        if i == best_idx and i == selected_idx:
            attr = curses.color_pair(2) | curses.A_BOLD | curses.A_REVERSE
        elif i == best_idx:
            attr = curses.color_pair(2) | curses.A_BOLD
        elif i == selected_idx:
            attr = curses.A_REVERSE
        else:
            attr = curses.A_NORMAL

        stdscr.addstr(row, x, f"{mask} | {cat:<15} | {ev:6.2f}", attr)


INSPECTOR_VIEWS = ["w", "wscale", "ctx", "row"]


# ---------------------------------------------------------------------------
# Scorecard rendering
# ---------------------------------------------------------------------------

_SC_W = 26  # inner width of scorecard box (excluding border chars)


def _sc_row(name: str, value: int | None, selected: bool, dim: bool) -> tuple[str, int]:
    """Format one scorecard category row. Returns (text, attr)."""
    val_str = "---" if value is None else str(value)
    text = f" {name:<16}    {val_str:>4} "
    if selected:
        return (">" + text[1:], curses.A_REVERSE)
    return (text, curses.A_DIM if dim else curses.A_NORMAL)


def _sc_sep(label: str = "") -> str:
    """A horizontal separator row, optionally labelled."""
    if label:
        pad = _SC_W - len(label) - 2
        return f"├─ {label} {'─' * (pad - 1)}┤"
    return f"├{'─' * _SC_W}┤"


def draw_scorecard(stdscr, state, mode: str, cursor: int, y_start: int, x: int):
    """
    Draw a bordered scorecard with upper/lower subtotals, bonus indicator,
    and grand total.
    """
    sc = state.scorecard  # shape (13,)

    # Derived totals
    upper_vals = [int(sc[i]) if sc[i] >= 0 else None for i in range(N_UPPER)]
    lower_vals = [int(sc[i]) if sc[i] >= 0 else None for i in range(N_UPPER, 13)]

    upper_sum = sum(v for v in upper_vals if v is not None)
    lower_sum = sum(v for v in lower_vals if v is not None)
    bonus = 35 if upper_sum >= BONUS_THRESHOLD else 0
    total = upper_sum + bonus + lower_sum

    bonus_progress = f"{min(upper_sum, BONUS_THRESHOLD)}/{BONUS_THRESHOLD}"
    bonus_str = f"+35 ✓" if bonus else f"({bonus_progress})"

    y = y_start

    def put(text, attr=curses.A_NORMAL):
        nonlocal y
        with contextlib.suppress(curses.error):
            stdscr.addstr(y, x, text, attr)
        y += 1

    # Top border + header
    put(f"┌{'─' * _SC_W}┐")
    put(f"│{'SCORECARD':^{_SC_W}}│", curses.A_BOLD)
    put(f"├{'─' * _SC_W}┤")

    # Upper section
    for i in range(N_UPPER):
        selected = mode == "score" and cursor == i
        dim = upper_vals[i] is not None and not selected
        text, attr = _sc_row(CAT_NAMES[i], upper_vals[i], selected, dim)
        put(f"│{text}│", attr)

    # Upper subtotal + bonus
    put(_sc_sep("UPPER"))
    put(f"│ {'Subtotal':<16}    {upper_sum:>4} │", curses.A_DIM)
    bonus_attr = curses.color_pair(2) | curses.A_BOLD if bonus else curses.A_DIM
    put(f"│ {'Bonus':<16}  {bonus_str:>4} │", bonus_attr)

    # Lower section
    put(_sc_sep("LOWER"))
    for i in range(N_LOWER):
        cat_idx = N_UPPER + i
        selected = mode == "score" and cursor == cat_idx
        dim = lower_vals[i] is not None and not selected
        text, attr = _sc_row(CAT_NAMES[cat_idx], lower_vals[i], selected, dim)
        put(f"│{text}│", attr)

    # Lower subtotal + grand total
    put(_sc_sep())
    put(f"│ {'Lower total':<16}    {lower_sum:>4} │", curses.A_DIM)
    put(_sc_sep())
    put(f"│ {'TOTAL':<16}    {total:>4} │", curses.A_BOLD)
    put(f"└{'─' * _SC_W}┘")

    return y  # return the row after the last drawn line


class KniffelApp:
    def __init__(self, genome):
        self.genome = genome
        self.show_ai = True
        self.mode = "reroll"
        self.cursor = 0
        self.reroll_mask = [False] * 5
        # Inspector state
        self.inspector = False
        self.insp_view = "w"
        self.insp_row = 0

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

    def draw_ai_panels(self, stdscr, state, y_off: int):
        ai_ctx = AIContext(self.genome, state)
        mask_idx = self.calculate_mask_idx()
        columns = compute_column_data(self.genome, state, ai_ctx, mask_idx)
        decision = compute_decision_data(self.genome, state, ai_ctx)

        x_cols = 40
        x_logic = x_cols + 3 * COL_W + 2

        draw_ev_columns(stdscr, columns, ai_ctx.open_mask, y_off, x_cols)
        draw_decision_panel(stdscr, decision, y_off, x_logic)
        draw_all_masks(stdscr, self.genome, state, y_off - 2, x_logic + 30, mask_idx, decision.ai_mask_idx)

    def render(self, stdscr, state):
        stdscr.erase()
        stdscr.addstr(
            0,
            2,
            f"🎲 KNIFFEL | Round: {int(state.round.squeeze()) + 1}/13 | Rolls: {int(state.rolls_left.squeeze())}",
            curses.A_BOLD,
        )

        # Dice row (rows 2-6)
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

        # Scorecard with border — starts at row 8, returns the row after the closing border
        _scorecard_end = draw_scorecard(stdscr, state, self.mode, self.cursor, y_start=8, x=2)

        # AI panels sit directly below the scorecard
        if self.show_ai:
            ai_y = 11  # one blank line gap
            self.draw_ai_panels(stdscr, state, ai_y)

        # Help bar at the very bottom
        help_y = curses.LINES - 2
        stdscr.addstr(help_y, 2, "WASD/HJKL: Move | SPACE: Toggle | ENTER: Confirm", curses.A_DIM)
        stdscr.addstr(
            help_y + 1,
            2,
            "TAB: Switch Mode (Dice/Scorecard)| G: Toggle AI | I: Genome Inspector | Q: Quit",
            curses.A_DIM,
        )
        stdscr.refresh()

    def play(self, stdscr, state):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        # Inspector palette
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)  # neg_strong
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)  # neg_weak
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_BLACK)  # zero (dim)
        curses.init_pair(7, curses.COLOR_CYAN, curses.COLOR_BLACK)  # pos_weak
        curses.init_pair(8, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # pos_strong
        curses.curs_set(0)

        while not bool(state.done):
            if self.inspector:
                draw_inspector_w(stdscr, self.genome, self.insp_view, self.insp_row)
            else:
                self.render(stdscr, state)

            key = stdscr.getch()

            # --- Inspector mode input ---
            if self.inspector:
                if key in [ord("i"), ord("I")]:
                    self.inspector = False
                elif key in [ord("[")]:
                    idx = INSPECTOR_VIEWS.index(self.insp_view)
                    self.insp_view = INSPECTOR_VIEWS[(idx - 1) % len(INSPECTOR_VIEWS)]
                elif key in [ord("]")]:
                    idx = INSPECTOR_VIEWS.index(self.insp_view)
                    self.insp_view = INSPECTOR_VIEWS[(idx + 1) % len(INSPECTOR_VIEWS)]
                elif key in [ord("k"), ord("w")]:
                    self.insp_row = max(0, self.insp_row - 1)
                elif key in [ord("j"), ord("s")]:
                    from knifflex.game.scoring import N_CATS

                    self.insp_row = min(N_CATS - 1, self.insp_row + 1)
                continue

            # --- Normal game input ---
            if key in [ord("q"), ord("Q")]:
                break
            if key in [ord("i"), ord("I")]:
                self.inspector = True
                continue
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
    from knifflex.genome.cereal import load_genome

    genome = load_genome("data/runs/best.npz")
    run_game(genome)
