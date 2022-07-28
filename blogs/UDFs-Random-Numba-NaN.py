import numpy as np
import numba as nb
import math


# Params for array construction
shape = (20_000, 20_000)
dtype = np.float32

@nb.jit
def circle_fun(out, vals) -> int:
    n = out.shape[0]
    m = out.shape[1]
    for i in range(n):
        for j in range(m):
            x = (2. * i / n) - 1.
            y = (2. * j / m) - 1.
            if ((x ** 2 + y ** 2) <= 1) and vals[i, j] >= 0.5:
                out[i, j] = 1.
            else:
                out[i, j] = math.nan
    return 0

@nb.jit
def circle_filter(val: float, row: int, col: int, nrows: int, ncols: int) -> float:
    x = (2. * row / nrows) - 1.
    y = (2. * col / ncols) - 1.
    if ((x ** 2 + y ** 2) <= 1) and val >= 0.5:
        return 1.
    return math.nan

@nb.jit
def circle_lib_fun(out, vals, nrows: int, ncols: int) -> int:
    n = out.shape[0]
    m = out.shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = circle_filter(vals[i, j], i, j, nrows, ncols)
    return 0

@nb.jit
def ellipse_filter(val: float, row: int, col: int, nrows: int, ncols: int) -> float:
    x = (2. * row / nrows) - 1.
    y = (2. * col / ncols) - 1.
    a = 1  # semi-major axis
    b = 0.5  # semi-minor axis
    if (((x ** 2 / a ** 2) + (y ** 2 / b ** 2)) <= 1) and val >= 0.5:
        return 1.
    return math.nan

@nb.jit
def ellipse_lib_fun(out, vals, nrows: int, ncols: int) -> int:
    n = out.window_shape[0]
    m = out.window_shape[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = ellipse_filter(vals[i, j], i, j, nrows, ncols)
    return 0

@profile
def numpy_rand():
    return np.random.random_sample(shape).astype(dtype)

rand_data = numpy_rand()

circle = np.empty(shape, dtype)
circle_lib = np.empty(shape, dtype)
ellipse_lib = np.empty(shape, dtype)

@profile
def numba_computations():
    circle_fun(circle, rand_data)
    circle_lib_fun(circle_lib, rand_data, *shape)
    circle_lib_fun(ellipse_lib, rand_data, *shape)

numba_computations()

@profile
def numpy_reductions():
    area_circle = np.nansum(circle)
    area_circle_lib = np.nansum(circle_lib)
    print(f"Diff in circle area: {(area_circle - area_circle_lib) / area_circle}")
    area_ellipse_lib = np.nansum(ellipse_lib)
    print(f"Circle/ellipse area: {area_ellipse_lib / area_circle}")
    area_square = np.nansum(rand_data)
    print(f"PI value: {4 * area_circle / area_square}")

numpy_reductions()