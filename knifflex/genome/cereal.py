from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .w_genome import DECOMP_RANK, GENOME_TYPE, DecompWGenome, FullWGenome, WGenome


def save_genome(genome: DecompWGenome | FullWGenome, path: str | Path) -> None:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    if isinstance(genome, FullWGenome):
        arrays = {
            "W": np.asarray(genome._W),
            "W_scale": np.asarray(genome._W_scale),
            "bonus_uplift": np.asarray(genome.bonus_uplift),
        }
        gtype = "full"
        rank = 0  # unused for full, stored for completeness
    elif isinstance(genome, DecompWGenome):
        arrays = {
            "A": np.asarray(genome.A),
            "B": np.asarray(genome.B),
            "A_scale": np.asarray(genome.A_scale),
            "B_scale": np.asarray(genome.B_scale),
            "bonus_uplift": np.asarray(genome.bonus_uplift),
        }
        gtype = "decomp"
        rank = genome.A.shape[1]
    else:
        raise TypeError(f"Unknown genome type: {type(genome)}")

    np.savez(
        path,
        **arrays,
        genome_type=np.array(gtype),
        decomp_rank=np.array(rank, dtype=np.int32),
    )


def load_genome(path: str | Path) -> WGenome:
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    data = np.load(path, allow_pickle=False)
    gtype = str(data["genome_type"])
    rank = int(data["decomp_rank"])

    if gtype != GENOME_TYPE:
        raise ValueError(
            f"Saved genome_type={gtype!r} does not match current "
            f"GENOME_TYPE={GENOME_TYPE!r}.  Update the config at the top of "
            "w_genome.py before loading."
        )

    if gtype == "full":
        return FullWGenome(
            _W=jnp.asarray(data["W"]),
            _W_scale=jnp.asarray(data["W_scale"]),
            bonus_uplift=jnp.asarray(data["bonus_uplift"]),
        )

    if gtype == "decomp":
        if rank != DECOMP_RANK:
            raise ValueError(
                f"Saved decomp_rank={rank} does not match current "
                f"DECOMP_RANK={DECOMP_RANK}.  Update the config at the top of "
                "w_genome.py before loading."
            )
        return DecompWGenome(
            A=jnp.asarray(data["A"]),
            B=jnp.asarray(data["B"]),
            A_scale=jnp.asarray(data["A_scale"]),
            B_scale=jnp.asarray(data["B_scale"]),
            bonus_uplift=jnp.asarray(data["bonus_uplift"]),
        )

    raise ValueError(f"Unrecognised genome_type in file: {gtype!r}")
