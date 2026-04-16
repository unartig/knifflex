from __future__ import annotations

from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

from dice import ALL_ROLLS, KEEP_MASKS, N_ROLLS
from scoring import score_case

_ROLLS_NP = np.array(ALL_ROLLS)  # (252, 5)
_MASKS_NP = np.array(KEEP_MASKS)  # (32, 5)

_score_vmapped = jax.vmap(
    jax.vmap(score_case, in_axes=(0, None)),  # inner: all 13 cats, one roll
    in_axes=(None, 0),  # outer: all 252 rolls
)
_cats_jnp = jnp.arange(13, dtype=jnp.int32)
_ROLL_TO_IDX = {tuple(r): i for i, r in enumerate(_ROLLS_NP)}

_SCORE_TABLE = np.array(_score_vmapped(_cats_jnp, ALL_ROLLS), dtype=np.int32)  # (252, 13)

# print(summarize_array("score table", SCORE_TABLE))

# ---------------------------------------------------------------------------
# Transition tensor  T[roll_i, mask, roll_j]  (252 x 32 x 252)
#     T[i, m, j] = P(land on roll j | start at roll i, keep-mask m)
#
#     For a given (roll_i, mask):
#       - kept dice = roll_i[keep_mask[m]]           (sorted, fixed)
#       - free dice = 5 - popcount(mask)  =: k
#       - each free die is uniform on {1..6}
#       - T[i, m, j] = #{ways to get roll_j from kept+free} / 6^k
#
#     We compute this exactly using multinomial counting.
# ---------------------------------------------------------------------------


def get_transition_tensor() -> np.ndarray:
    if Path("transition_tensor.npz").exists():
        T = np.load("transition_tensor.npz")["transition_tensor"]
        return T

    print("Building transition tensor (252 dice x 32 reroll masks x 252 dice)")
    T = np.zeros((N_ROLLS, 32, N_ROLLS), dtype=np.float32)

    for i in trange(N_ROLLS, desc="Building transition tensor"):
        roll = _ROLLS_NP[i]
        for m_idx, keep in enumerate(KEEP_MASKS):
            kept = np.sort(roll[keep])
            n_free = int((~keep).sum())

            if n_free == 0:
                T[i, m_idx, i] = 1.0
                continue

            denom = 6**n_free
            for free_vals in product(range(1, 7), repeat=n_free):
                new_roll = tuple(np.sort(np.concatenate([kept, list(free_vals)])))
                j = _ROLL_TO_IDX.get(new_roll)
                if j is not None:
                    T[i, m_idx, j] += 1.0 / denom

    # Sanity check: each row should sum to ~1
    row_sums = T.sum(axis=2)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"Row sum error: min={row_sums.min():.6f} max={row_sums.max():.6f}"

    np.savez_compressed(
        "transition_tensor.npz",
        transition_tensor=T,
    )
    return T


# T = get_transition_tensor()
# print(summarize_array("transition tensor", T))

# ---------------------------------------------------------------------------
#  Backwards induction
#     EV shape: (252, 3, 13)   dim-1: rolls_left index (0=must score, 1, 2)
#     mask shape: (252, 2, 13)  dim-1: rolls_left index (0 -> rl=1, 1 -> rl=2)
# ---------------------------------------------------------------------------


def get_ev_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if Path("ev_table.npz").exists():
        _data = np.load("ev_table.npz")

        ev = _data["ev_table"]  # (252, 3, 13)  float32
        best_mask = _data["best_mask"]  # (252, 2, 13)  int32
        return ev, best_mask, _ROLLS_NP

    T = get_transition_tensor()
    ev = np.zeros((N_ROLLS, 3, 13), dtype=np.float32)
    best_mask = np.zeros((N_ROLLS, 2, 13), dtype=np.int32)

    print("Computing EV for rolls_left=0")
    # rolls_left = 0
    ev[:, 0, :] = _SCORE_TABLE.astype(np.float32)

    # rolls_left = 1
    # For each mask m: expected_ev1[roll, m, cat] = T[roll, m, :] @ ev[:, 0, cat]
    # Shape: (252, 32, 13)  via einsum
    print("Computing EV for rolls_left=1")
    expected_after_reroll = np.einsum("imj,jc->imc", T, ev[:, 0, :])  # (252, 32, 13)
    assert expected_after_reroll.shape == (252, 32, 13), f"Expected (252, 32, 13), got {expected_after_reroll.shape}"

    # max over masks
    best_mask[:, 0, :] = expected_after_reroll.argmax(axis=1)  # (252, 13)
    ev[:, 1, :] = expected_after_reroll.max(axis=1)  # (252, 13)

    # rolls_left = 2
    print("Computing EV for rolls_left=2")
    expected_after_reroll2 = np.einsum("imj,jc->imc", T, ev[:, 1, :])  # (252, 32, 13)
    assert expected_after_reroll2.shape == (252, 32, 13), f"Expected (252, 32, 13), got {expected_after_reroll2.shape}"

    best_mask[:, 1, :] = expected_after_reroll2.argmax(axis=1)  # (252, 13)
    ev[:, 2, :] = expected_after_reroll2.max(axis=1)  # (252, 13)

    # Sanity checks
    # Already-kniffel roll should have EV=50 at rolls_left=0
    five_sixes = _ROLL_TO_IDX[(6, 6, 6, 6, 6)]
    assert ev[five_sixes, 0, 12] == 50.0, "5x6 Kniffel score should be 50"
    # Large straight (cat 10) at rolls_left=0 on [2,3,4,5,6]
    lstr = _ROLL_TO_IDX[(2, 3, 4, 5, 6)]
    assert ev[lstr, 0, 10] == 40.0, "Large straight score should be 40"

    np.savez_compressed(
        "ev_table.npz",
        ev_table=ev,
        best_mask=best_mask,
    )

    return ev, best_mask, _ROLLS_NP


# ev, best_mask = get_ev_table(T)
# print(summarize_array("best_mask", best_mask))
# print(summarize_array("ev_table", ev))

# print("\n── Spot-check: [1,1,1,6,6] ──")
# roll_idx = ROLL_TO_IDX[(1, 1, 1, 6, 6)]
# for rl in range(3):
#     best_cats = sorted(range(13), key=lambda c: ev[roll_idx, rl, c], reverse=True)[:3]
#     top = [(CASE_NAMES[c], f"{ev[roll_idx, rl, c]:.2f}") for c in best_cats]
#     print(f"  rolls_left={rl}: " + "  |  ".join(f"{n}={v}" for n, v in top))

# print("\n── Spot-check: [3,4,5,6,6] ──")
# roll_idx2 = ROLL_TO_IDX[(3, 4, 5, 6, 6)]
# for rl in range(3):
#     best_cats = sorted(range(13), key=lambda c: ev[roll_idx2, rl, c], reverse=True)[:3]
#     top = [(CASE_NAMES[c], f"{ev[roll_idx2, rl, c]:.2f}") for c in best_cats]
#     print(f"  rolls_left={rl}: " + "  |  ".join(f"{n}={v}" for n, v in top))


def _load_transition() -> tuple:
    from ev_table import get_transition_tensor  # noqa: PLC0415

    T = jnp.array(get_transition_tensor(), dtype=jnp.float32)  # (252, 32, 252)
    fresh = T[0, 0]  # (252,) - reroll-all distribution
    return T, fresh


TRANSITION_TABLE, FRESH_PROBS = _load_transition()
