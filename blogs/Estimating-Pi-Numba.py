import numpy as np
import numba as nb
import math


# Params for array construction
shape = (40_000, 40_000)
dtype = np.float32

@nb.jit
def circle_filter(val: float, row: int, col: int, nrows: int, ncols: int) -> float:
    x = (2. * row / nrows) - 1.
    y = (2. * col / ncols) - 1.
    if ((x ** 2 + y ** 2) <= 1) and val >= 0.5:
        return 1.
    return math.nan

@nb.jit
def square_filter(val: float) -> float:
    if val >= 0.5:
        return 1.
    return math.nan

@nb.jit(parallel=True)
def filter_func(out, vals, nrows: int, ncols: int, iscircle: bool) -> int:
    n, m = out.shape
    for i in nb.prange(n):
        for j in nb.prange(m):
            if iscircle:
                out[i, j] = circle_filter(vals[i, j], i, j, nrows, ncols)
            else:
                out[i, j] = square_filter(vals[i, j])
    return 0


@profile
def numpy_rand():
    rng = np.random.default_rng()
    return rng.random(shape, dtype=dtype)

rand_data = numpy_rand()

circle = np.empty(shape, dtype)
square = np.empty(shape, dtype)

@profile
def numba_computations():
    filter_func(circle, rand_data, shape[0], shape[1], True)
    filter_func(square, rand_data, shape[0], shape[1], False)

numba_computations()

@profile
def numpy_reductions():
    area_circle = np.nansum(circle)
    area_square = np.nansum(square)
    print(f"PI estimation: {4 * area_circle / area_square}")

numpy_reductions()
