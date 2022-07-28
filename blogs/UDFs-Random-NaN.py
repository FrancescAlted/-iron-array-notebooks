import numpy as np
import iarray as ia
from iarray import udf
import math


# Params for array construction
shape = (20_000, 20_000)
ia.set_config_defaults(dtype=np.float32, fp_mantissa_bits=15, favor=ia.Favor.SPEED)#, btune=False, clevel=0)


@udf.jit
def circle_fun(out: udf.Array(udf.float32, 2), vals: udf.Array(udf.float32, 2), nrows: udf.int64, ncols: udf.int64) -> udf.int32:
    n = out.window_shape[0]
    m = out.window_shape[1]
    row_start = out.window_start[0]
    col_start = out.window_start[1]
    for i in range(n):
        for j in range(m):
            x = (2. * (row_start + i) / nrows) - 1.
            y = (2. * (col_start + j) / ncols) - 1.
            if ((x ** 2 + y ** 2) <= 1) and vals[i, j] >= 0.5:
                out[i, j] = 1.
            else:
                out[i, j] = math.nan
    return 0

@udf.scalar()
def circle_filter(val: udf.float32, row: udf.int64, col: udf.int64, nrows: udf.int64, ncols: udf.int64) -> udf.float32:
    x = (2. * row / nrows) - 1.
    y = (2. * col / ncols) - 1.
    if ((x ** 2 + y ** 2) <= 1) and val >= 0.5:
        return 1.
    return math.nan

@udf.jit
def circle_lib_fun(out: udf.Array(udf.float32, 2), vals: udf.Array(udf.float32, 2), nrows: udf.int64, ncols: udf.int64) -> udf.int32:
    n = out.window_shape[0]
    m = out.window_shape[1]
    row_start = out.window_start[0]
    col_start = out.window_start[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = ulib.circle_filter(vals[i, j], row_start + i, col_start + j, nrows, ncols)
    return 0

@udf.scalar()
def ellipse_filter(val: udf.float32, row: udf.int64, col: udf.int64, nrows: udf.int64, ncols: udf.int64) -> udf.float32:
    x = (2. * row / nrows) - 1.
    y = (2. * col / ncols) - 1.
    a = 1  # semi-major axis
    b = 0.5  # semi-minor axis
    if (((x ** 2 / a ** 2) + (y ** 2 / b ** 2)) <= 1) and val >= 0.5:
        return 1.
    return math.nan


@udf.jit
def ellipse_lib_fun(out: udf.Array(udf.float32, 2), vals: udf.Array(udf.float32, 2), nrows: udf.int64, ncols: udf.int64) -> udf.int32:
    n = out.window_shape[0]
    m = out.window_shape[1]
    row_start = out.window_start[0]
    col_start = out.window_start[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = ulib.ellipse_filter(vals[i, j], row_start + i, col_start + j, nrows, ncols)
    return 0


@profile
def iarray_rand():
    return ia.random.random_sample(shape)

rand_data = iarray_rand()

@profile
def iarray_computations():
    global circle, circle_lib, ellipse_lib
    expr = ia.expr_from_udf(circle_fun, [rand_data], list(shape))
    circle = expr.eval()
    expr = ia.expr_from_udf(circle_lib_fun, [rand_data], list(shape))
    circle_lib = expr.eval()
    expr = ia.expr_from_udf(ellipse_lib_fun, [rand_data], list(shape))
    ellipse_lib = expr.eval()

iarray_computations()

@profile
def iarray_reductions():
    area_circle = ia.nansum(circle, axis=(1,0))
    area_circle_lib = ia.nansum(circle_lib, axis=(1,0))
    print(f"Diff in circle area: {(area_circle - area_circle_lib) / area_circle}")
    area_ellipse_lib = ia.nansum(ellipse_lib, axis=(1,0))
    print(f"Circle/ellipse area: {area_ellipse_lib / area_circle}")
    area_square = ia.nansum(rand_data, axis=(1,0))
    print(f"PI value: {4 * area_circle / area_square}")

iarray_reductions()
