from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import PRNGKeyArray

from cereal import load_genome
from ev_table import get_ev_table
from scoring import CAT_NAMES

if TYPE_CHECKING:
    from game import KniffelState

# ------------------------------------------------------------------ #
#  EV data (loaded once at import)                                    #
# ------------------------------------------------------------------ #

_ev, _mask, _rolls = get_ev_table()
EV = np.array(_ev, dtype=np.float32)  # (252, 3, 13)
ROLLS = np.array(_rolls, dtype=np.int8)  # (252, 5)

EV_CEILING = EV.max(axis=0)  # (3, 13) — best possible EV per (rl, cat)
EV_FLOOR = EV.min(axis=0)  # (3, 13)

N_CATS = 13

# ------------------------------------------------------------------ #
#  Input unpacking — accepts KniffelState or raw (13,) array          #
# ------------------------------------------------------------------ #


# def _unpack(state_or_scorecard) -> tuple[np.ndarray, int, np.ndarray | None]:
#     """Return (scorecard_np, rolls_left, dice_np_or_None).

#     Accepts either a KniffelState (duck-typed by presence of .scorecard)
#     or a plain (13,) NumPy array / list.
#     """
#     if hasattr(state_or_scorecard, "scorecard"):
#         s    = state_or_scorecard
#         sc   = np.array(s.scorecard, dtype=np.int32)
#         rl   = int(np.array(s.rolls_left).squeeze())
#         dice = np.array(s.dice, dtype=np.int32)
#         return sc, rl, dice
#     sc = np.asarray(state_or_scorecard, dtype=np.int32)
#     assert sc.shape == (N_CATS,), f"Expected (13,) scorecard, got {sc.shape}"
#     return sc, 2, None   # default: start of turn, no dice


def _get_W(genome) -> np.ndarray:
    """Full (13, 15) matrix from any genome type."""
    return np.array(genome.W, dtype=np.float32)


# ------------------------------------------------------------------ #
#  Context builder (mirrors w_genome.build_context in NumPy)          #
# ------------------------------------------------------------------ #


def _build_ctx(sc: np.ndarray, rl: int) -> np.ndarray:
    open_flags = (sc < 0).astype(np.float32)
    scored = np.where(sc >= 0, sc, 0).astype(np.float32)
    upper_sum = scored[:6].sum() / 63.0
    return np.concatenate([open_flags, [upper_sum, rl / 2.0]]).astype(np.float32)


# ------------------------------------------------------------------ #
#  Analysis functions                                                 #
# ------------------------------------------------------------------ #


def _category_priorities(W: np.ndarray, sc: np.ndarray, rl: int) -> dict:
    """Genome-weighted category ranking for the current state."""
    open_mask = sc < 0
    ctx = _build_ctx(sc, rl)
    cat_weights = W @ ctx  # (13,)
    multipliers = np.maximum(0.0, 1.0 + cat_weights)  # (13,)
    ev_mean = EV[:, rl, :].mean(axis=0)  # (13,)  avg over all 252 rolls
    ev_ceil = EV_CEILING[rl]  # (13,)  best possible roll
    adjusted = ev_mean * multipliers  # (13,)

    open_idx = np.where(open_mask)[0]
    rank_order = open_idx[np.argsort(adjusted[open_idx])[::-1]]

    return {
        "open_mask": open_mask,
        "cat_weights": cat_weights,  # raw logit, signed
        "multipliers": multipliers,  # applied to EV, ≥ 0
        "ev_mean": ev_mean,
        "ev_ceil": ev_ceil,
        "adjusted": adjusted,
        "rank_order": rank_order,  # open cats sorted best-first
        "rl": rl,
    }


def _dice_ev(sc: np.ndarray, rl: int, dice: np.ndarray | None) -> dict | None:
    """EV of every open category for the *actual* current dice roll."""
    if dice is None:
        return None

    from ev_table import _ROLL_TO_IDX  # noqa: PLC0415 — deferred, not always needed

    sorted_dice = tuple(sorted(dice.tolist()))
    roll_idx = _ROLL_TO_IDX.get(sorted_dice)
    if roll_idx is None:
        return None

    open_mask = sc < 0
    ev_now = EV[roll_idx, rl, :]  # (13,) EV for THIS exact roll
    open_idx = np.where(open_mask)[0]
    rank_by_ev = open_idx[np.argsort(ev_now[open_idx])[::-1]]

    return {
        "dice": dice,
        "roll_idx": roll_idx,
        "ev_now": ev_now,
        "rank_by_ev": rank_by_ev,  # open cats ranked by raw EV (no genome weighting)
    }


def _upper_bonus_analysis(W: np.ndarray, sc: np.ndarray, rl: int) -> dict:
    """How aggressively does the genome chase the 63-pt upper bonus threshold?

    Sweeps upper_sum from 0 → 1 while holding everything else constant,
    tracking how the genome's mean weight on upper vs lower cats evolves.
    The urgency slope tells you whether the genome learned the threshold effect.
    """
    open_mask = sc < 0
    upper_open = open_mask[:6]
    steps = 21
    progress = np.linspace(0.0, 1.0, steps)
    upper_w = np.zeros(steps)
    lower_w = np.zeros(steps)

    for i, p in enumerate(progress):
        ctx = _build_ctx(sc, rl).copy()
        ctx[13] = p  # override upper_sum slot
        wts = W @ ctx
        if upper_open.any():
            upper_w[i] = wts[:6][upper_open].mean()
        if open_mask[6:].any():
            lower_w[i] = wts[6:][open_mask[6:]].mean()

    mid = slice(int(steps * 0.5), int(steps * 0.95))
    urgency = float(np.polyfit(progress[mid], upper_w[mid], 1)[0]) if upper_open.any() else 0.0

    scored = np.where(sc[:6] >= 0, sc[:6], 0)
    current_upper = int(scored.sum())
    needed = max(0, 63 - current_upper)
    remaining = int(upper_open.sum())

    return {
        "progress": progress,
        "upper_weight_mean": upper_w,
        "lower_weight_mean": lower_w,
        "urgency": urgency,
        "current_upper": current_upper,
        "needed": needed,
        "remaining_upper": remaining,
        "bonus_reachable": (remaining > 0) and (needed <= remaining * 5),
    }


def _rolls_left_sensitivity(W: np.ndarray, sc: np.ndarray) -> dict:
    """Does the genome shift its category ranking based on rolls remaining?

    Returns a Spearman ρ between the rl=0 and rl=2 orderings over open cats.
    ρ ≈ 1: commits to same target regardless of rolls left.
    ρ ≈ -1: completely reverses priorities depending on rolls.
    """
    open_idx = np.where(sc < 0)[0]
    orderings = {}
    for rl in [0, 1, 2]:
        orderings[rl] = np.argsort(W @ _build_ctx(sc, rl))[::-1].tolist()

    if len(open_idx) < 2:
        rho = 1.0
    else:

        def rank_vec(order):
            ov = [c for c in order if c in open_idx]
            return np.array([ov.index(c) for c in open_idx], dtype=float)

        r0, r2 = rank_vec(orderings[0]), rank_vec(orderings[2])
        n = len(open_idx)
        d2 = ((r0 - r2) ** 2).sum()
        rho = float(1 - 6 * d2 / (n * (n**2 - 1))) if n > 2 else 1.0

    return {"orderings": orderings, "rank_stability": rho}


def _latent_factors(genome) -> dict | None:
    """Interpret rank-r latent factors from DecompWGenome.

    Each factor r encodes:
      B[r]  — which context dimensions drive it (sensitivity profile)
      A[:,r] — which categories respond to it (loading vector)

    Returns None for FullWGenome.
    """
    if not (hasattr(genome, "A") and hasattr(genome, "B")):
        return None

    A = np.array(genome.A, dtype=np.float32)  # (13, rank)
    B = np.array(genome.B, dtype=np.float32)  # (rank, 15)
    rank = A.shape[1]
    ctx_names = [*CAT_NAMES, "upper_sum", "rolls_left"]

    factors = []
    for r in range(rank):
        b, a = B[r], A[:, r]
        frob = float(np.linalg.norm(np.outer(a, b), "fro"))
        factors.append(
            {
                "index": r,
                "b_vec": b,
                "a_vec": a,
                "top_ctx": [(ctx_names[i], float(b[i])) for i in np.argsort(np.abs(b))[::-1][:3]],
                "top_cats_pos": [(CAT_NAMES[i], float(a[i])) for i in np.argsort(a)[::-1][:3]],
                "top_cats_neg": [(CAT_NAMES[i], float(a[i])) for i in np.argsort(a)[:3]],
                "frobenius": frob,
            }
        )

    factors.sort(key=lambda f: f["frobenius"], reverse=True)
    total = sum(f["frobenius"] for f in factors) or 1.0
    for f in factors:
        f["explained_pct"] = f["frobenius"] / total * 100

    return {"rank": rank, "factors": factors}


def _context_column_sensitivity(W: np.ndarray) -> dict:
    """Rank all 15 context dims by how much they can swing cat_weights (L2 col norm)."""
    ctx_names = [*CAT_NAMES, "upper_sum", "rolls_left"]
    col_norms = np.linalg.norm(W, axis=0)  # (15,)
    return {"ctx_names": ctx_names, "col_norms": col_norms, "order": np.argsort(col_norms)[::-1]}


# ------------------------------------------------------------------ #
#  Report                                                             #
# ------------------------------------------------------------------ #


@dataclass
class InterpretationReport:
    scorecard: np.ndarray
    rolls_left: int
    dice: np.ndarray | None  # None when constructed from raw scorecard
    W: np.ndarray
    priorities: dict
    dice_ev: dict | None  # None when no dice info available
    upper: dict
    rl_sens: dict
    ctx_sens: dict
    factors: dict | None  # None for FullWGenome

    def summary(self) -> str:
        p = self.priorities
        u = self.upper
        rs = self.rl_sens

        open_count = int(p["open_mask"].sum())
        scored_count = N_CATS - open_count
        top3_names = [CAT_NAMES[c] for c in p["rank_order"][:3]]

        print(f"\n{'─' * 60}")
        state_line = (
            f"  Round {scored_count + 1}/13  |"
            f"  rolls_left={self.rolls_left}  |"
            f"  dice={sorted(self.dice.tolist()) if self.dice is not None else None}"
        )

        rl_line = urg_line = ""
        if self.rolls_left == 2:
            rl_line = (
                "Commits early — barely shifts target with rolls remaining.\n"
                if rs["rank_stability"] > 0.85
                else "Moderately adjusts targets based on rolls remaining.\n"
                if rs["rank_stability"] > 0.4
                else "Strongly shifts strategy depending on rolls remaining.\n"
            )
            urg_line = (
                "Aggressively chases upper bonus as threshold nears.\n"
                if u["urgency"] > 0.3
                else "Mild urgency around upper bonus threshold.\n"
                if u["urgency"] > 0.05
                else "Little sensitivity to upper bonus threshold.\n"
            )

        return "  ".join(
            [
                state_line,
                f"\nTop genome priorities: {', '.join(top3_names)}.\n",
                rl_line,
                urg_line,
            ]
        )

    # ── Section printers ───────────────────────────────────────────── #

    def print_priority_table(self) -> None:
        p = self.priorities
        de = self.dice_ev
        rl = self.rolls_left

        extra_hdr = f"  {'EV_dice':>8}" if de else ""
        print(f"\n── Category priorities  (rolls_left={rl}) {'─' * 20}")
        print(
            f"  {'#':<3} {'Category':<18} {'Status':<7} "
            f"{'EV_mean':>8} {'EV_ceil':>8} {'Genome×':>8} {'Adj_EV':>8} {'Weight':>8}" + extra_hdr
        )
        print("  " + "─" * (72 + (10 if de else 0)))

        for rank, cat in enumerate(p["rank_order"]):
            sc_val = int(self.scorecard[cat])
            status = "open" if sc_val < 0 else f"{sc_val:>6}"
            dice_col = f"  {de['ev_now'][cat]:>8.1f}" if de else ""
            print(
                f"  {rank + 1:<3} {CAT_NAMES[cat]:<18} {status:<7} "
                f"{p['ev_mean'][cat]:>8.1f} {p['ev_ceil'][cat]:>8.1f} "
                f"{p['multipliers'][cat]:>8.3f} {p['adjusted'][cat]:>8.1f} "
                f"{p['cat_weights'][cat]:>+8.3f}" + dice_col
            )

        if de is not None:
            best3 = ", ".join(CAT_NAMES[c] for c in de["rank_by_ev"][:3])
            print(f"\n  Dice {sorted(de['dice'].tolist())} — best raw EV: {best3}")

    def print_upper_bonus(self) -> None:
        u = self.upper
        print(f"\n── Upper bonus pressure {'─' * 34}")
        print(
            f"  Current : {u['current_upper']}/63   needed: {u['needed']}   "
            f"open upper slots: {u['remaining_upper']}   "
            f"reachable: {'yes' if u['bonus_reachable'] else 'no'}"
        )
        print(
            f"  Urgency slope: {u['urgency']:+.3f}  (how much genome shifts toward upper cats as threshold approaches)"
        )

        uw = u["upper_weight_mean"]
        lo, hi = uw.min(), uw.max()
        chars = " ▁▂▃▄▅▆▇█"
        bar = "".join(chars[int((v - lo) / (hi - lo + 1e-8) * 8)] for v in uw)
        print(f"\n  Upper weight  0%{'─' * 18}100%")
        print(f"  lo={lo:.2f} {bar} hi={hi:.2f}")

    def print_rolls_left_sensitivity(self) -> None:
        rs = self.rl_sens
        open_idx = np.where(self.scorecard < 0)[0]
        print(f"\n── Rolls-left sensitivity  (Spearman rho rl=0↔2: {rs['rank_stability']:+.3f}) {'─' * 10}")
        print(f"  {'Category':<18}  rl=0  rl=1  rl=2")
        print("  " + "─" * 42)
        for cat in open_idx:
            ranks = [[c for c in rs["orderings"][rl] if c in open_idx].index(cat) + 1 for rl in [0, 1, 2]]
            flag = " *" if max(ranks) - min(ranks) >= 3 else ""
            print(f"  {CAT_NAMES[cat]:<18}  {ranks[0]:>4}  {ranks[1]:>4}  {ranks[2]:>4}{flag}")
        print("  (* = shifts ≥3 ranks between rl=0 and rl=2)")

    def print_context_sensitivity(self) -> None:
        cs = self.ctx_sens
        print(f"\n── Context dimension influence  (‖W col‖₂) {'─' * 15}")
        peak = cs["col_norms"].max()
        for i in cs["order"]:
            bar = "█" * int(cs["col_norms"][i] / peak * 28)
            print(f"  {cs['ctx_names'][i]:<18}  {bar:<28}  {cs['col_norms'][i]:.4f}")

    def print_latent_factors(self) -> None:
        if self.factors is None:
            print(f"\n── Latent factors: n/a (FullWGenome) {'─' * 22}")
            return
        print(f"\n── Latent factors  rank={self.factors['rank']} {'─' * 30}")
        for f in self.factors["factors"]:
            print(f"\n  Factor {f['index'] + 1}  [{f['explained_pct']:.1f}% of ‖W‖_F]")
            print("    Context drivers:")
            for name, val in f["top_ctx"]:
                print(f"      {'↑' if val > 0 else '↓'} {name:<18} ({val:+.3f})")
            boosted = ", ".join(f"{n}({v:+.2f})" for n, v in f["top_cats_pos"] if v > 0.01)
            suppressed = ", ".join(f"{n}({v:+.2f})" for n, v in f["top_cats_neg"] if v < -0.01)
            if boosted:
                print(f"    Boosts:     {boosted}")
            if suppressed:
                print(f"    Suppresses: {suppressed}")

    def print_full(self) -> None:
        print("=" * 60)
        print("  GENOME STRATEGY INTERPRETATION")
        print("=" * 60)
        print()
        print(self.summary())
        self.print_priority_table()
        self.print_upper_bonus()
        self.print_rolls_left_sensitivity()
        self.print_context_sensitivity()
        self.print_latent_factors()
        print()


# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #


def interpret(genome: WGenome, state: KniffelState) -> InterpretationReport:
    sc, rl, dice = np.asarray(state.scorecard), int(state.rolls_left.squeeze()), np.asarray(state.dice)
    W = _get_W(genome)

    return InterpretationReport(
        scorecard=sc,
        rolls_left=rl,
        dice=dice,
        W=W,
        priorities=_category_priorities(W, sc, rl),
        dice_ev=_dice_ev(sc, rl, dice),
        upper=_upper_bonus_analysis(W, sc, rl),
        rl_sens=_rolls_left_sensitivity(W, sc),
        ctx_sens=_context_column_sensitivity(W),
        factors=_latent_factors(genome),
    )


def compare(genome_a: WGenome, genome_b: WGenome, state: KniffelState, labels: tuple[str, ...] = ("A", "B")) -> None:
    """Side-by-side priority comparison of two genomes on the same state."""
    sc = np.asarray(state.scorecard)
    ra = interpret(genome_a, state)
    rb = interpret(genome_b, state)
    pa, pb = ra.priorities, rb.priorities
    open_idx = np.where(sc < 0)[0]
    oa = {c: r for r, c in enumerate(pa["rank_order"])}
    ob = {c: r for r, c in enumerate(pb["rank_order"])}

    print(f"\n── Compare: {labels[0]} vs {labels[1]}  (rl={state.rolls_left}) {'─' * 25}")
    print(f"  {'Category':<18}  {labels[0]:>10}  {labels[1]:>10}  Δweight")
    print("  " + "─" * 55)
    for cat in open_idx:
        dw = pa["cat_weights"][cat] - pb["cat_weights"][cat]
        arrow = "↑" if dw > 0.05 else ("↓" if dw < -0.05 else "≈")
        print(f"  {CAT_NAMES[cat]:<18}  {oa.get(cat, -1) + 1:>10}  {ob.get(cat, -1) + 1:>10}  {arrow} {dw:+.3f}")


# ------------------------------------------------------------------ #
#  __main__ — load genome, play a full game, print per-turn analysis  #
# ------------------------------------------------------------------ #


def _fmt_scorecard(sc: np.ndarray) -> str:
    upper_total = int(np.where(sc[:6] >= 0, sc[:6], 0).sum())
    lines = [f"  Scorecard  (upper: {upper_total}/63)"]
    for i, name in enumerate(CAT_NAMES):
        val = int(sc[i])
        filled = f"{val:>4}" if val >= 0 else "  --"
        sep = "\n ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌" if i == 5 else ""
        lines.append(f"    {name:<18} {filled}{sep}")
    lines.append(f"  Total: {int(np.where(sc >= 0, sc, 0).sum())}")
    return "\n".join(lines)


def _play_game(genome: WGenome, key: PRNGKeyArray) -> None:
    state = reset(key)
    total = 0

    print("\n" + "=" * 60)
    print(f"  GAME  (key={key})")
    print("=" * 60)

    while not bool(state.done):
        # Full interpretation at the start of each fresh turn
        report = interpret(genome, state)
        print(report.summary())
        report.print_priority_table()

        # Genome picks action
        action = int(genome_action(genome, state))
        if action < 32:
            kept = [int(d) for i, d in enumerate(state.dice.tolist()) if (action >> i) & 1]
            print(f"\n  → Reroll  (keep {kept})")
        else:
            cat = action - 32
            print(f"\n  → Score   {CAT_NAMES[cat]}")

        state, reward = step(state, jnp.array(action))
        total += int(reward)

        if action >= 32:
            print(f"\n{_fmt_scorecard(np.array(state.scorecard, dtype=np.int32))}")
            print(f"  Reward: {int(reward):+d}   Running total: {total}")

    print("\n" + "=" * 60)
    print(f"  GAME OVER — Final score: {total}")
    print("=" * 60)
    print(_fmt_scorecard(np.array(state.scorecard, dtype=np.int32)))

    # Post-game: static genome analysis (no dice, no state)
    fresh_key, key = jr.split(key)
    fresh = reset(fresh_key)
    print("\n── Post-game genome character (fresh scorecard) ───────────")
    end_report = interpret(genome, fresh)
    end_report.print_context_sensitivity()
    if end_report.factors:
        end_report.print_latent_factors()


if __name__ == "__main__":
    from game import reset, step
    from w_genome import (
        WGenome,  # assumes load_genome(path) -> genome
        genome_action,
    )

    p = "runs/Mar28_19-46-41_tinkpad_w_genome/best_w15200.npz"

    genome = load_genome(p)

    key = jr.key(123)

    report = interpret(genome, reset(key))
    report.print_full()

    _play_game(genome, key)
