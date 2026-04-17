from collections.abc import Callable
from functools import wraps
from time import time

import numpy as np
from beartype import beartype as typechecker  # noqa: F401
from jaxtyping import Array


def timing(f: Callable) -> Callable:
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__!r} took: {(te - ts):2.2f} sec")
        return result

    return wrap


def summarize_array(name: str, arr: Array | np.ndarray) -> str:
    return f"{name}: {arr.shape}  dtype={arr.dtype}  ~{arr.nbytes / 1024:.0f} KB"
