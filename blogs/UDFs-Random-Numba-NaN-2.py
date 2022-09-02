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
def circle_fun(out, vals, nrows: int, ncols: int) -> int:
    n = out.shape[0]
    m = out.shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = circle_filter(vals[i, j], i, j, nrows, ncols)
    return 0


@profile
def numpy_rand():
    return np.random.random_sample(shape).astype(dtype)

rand_data = numpy_rand()

circle = np.empty(shape, dtype)

@profile
def numba_computations():
    circle_fun(circle, rand_data, *shape)

numba_computations()

@profile
def numpy_reductions():
    area_circle = np.nansum(circle)
    area_square = np.nansum(rand_data)
    print(f"PI value: {4 * area_circle / area_square}")

numpy_reductions()
