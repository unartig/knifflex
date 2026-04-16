import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, Int8, Scalar, jaxtyped

from dice import DiceArray
from utils import typechecker

CAT_NAMES = [
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

CAT_MAX = jnp.array(
    [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 25.0, 30.0, 30.0, 30.0, 40.0, 30.0, 50.0],
    dtype=jnp.float32,
)

N_CATS = len(CAT_NAMES)


@jaxtyped(typechecker=typechecker)
def score_upper(dice: DiceArray, face: Int[Array, ""] | int) -> Int8[Scalar, ""]:
    return (jnp.sum(dice == face) * face).astype(jnp.int8)


@jaxtyped(typechecker=typechecker)
def score_full_house(wurf: DiceArray) -> Int8[Scalar, ""]:
    counts = jnp.bincount(wurf, length=7)[1:]
    return (jnp.where(jnp.any(counts == 3) & jnp.any(counts == 2), 25, 0)).astype(jnp.int8)


@jaxtyped(typechecker=typechecker)
def score_three_of_a_kind(dice: DiceArray) -> Int8[Scalar, ""]:
    return (jnp.where(jnp.max(jnp.bincount(dice, length=7)) >= 3, jnp.sum(dice), 0)).astype(jnp.int8)


@jaxtyped(typechecker=typechecker)
def score_four_of_a_kind(dice: DiceArray) -> Int8[Scalar, ""]:
    return (jnp.where(jnp.max(jnp.bincount(dice, length=7)) >= 4, jnp.sum(dice), 0)).astype(jnp.int8)


@jaxtyped(typechecker=typechecker)
def score_small_straight(wurf: DiceArray) -> Int8[Scalar, ""]:
    present = jnp.bincount(wurf, length=7)[1:] > 0
    has = jnp.any(
        jnp.stack(
            [
                present[0] & present[1] & present[2] & present[3],
                present[1] & present[2] & present[3] & present[4],
                present[2] & present[3] & present[4] & present[5],
            ]
        )
    )
    return (jnp.where(has, 30, 0)).astype(jnp.int8)


@jaxtyped(typechecker=typechecker)
def score_large_straight(wurf: DiceArray) -> Int8[Scalar, ""]:
    present = jnp.bincount(wurf, length=7)[1:] > 0
    has = (present[0] & present[1] & present[2] & present[3] & present[4]) | (
        present[1] & present[2] & present[3] & present[4] & present[5]
    )
    return (jnp.where(has, 40, 0)).astype(jnp.int8)


@jaxtyped(typechecker=typechecker)
def score_faces(wurf: DiceArray) -> Int8[Scalar, ""]:
    return (jnp.sum(wurf)).astype(jnp.int8)


@jaxtyped(typechecker=typechecker)
def score_kniffel(dice: DiceArray) -> Int8[Scalar, ""]:
    return (jnp.where(jnp.all(dice == dice[0]), 50, 0)).astype(jnp.int8)


@jaxtyped(typechecker=typechecker)
def score_case(case_id: Int[Array, ""], dice: DiceArray) -> Int8[Scalar, ""]:
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
