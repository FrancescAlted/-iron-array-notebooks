import numpy as np
import iarray as ia
from iarray import udf
import math


# Params for array construction
shape = (40_000, 40_000)
ia.set_config_defaults(dtype=np.float32, fp_mantissa_bits=15)#, favor=ia.Favor.SPEED)#, btune=False, clevel=0)


@udf.scalar()
def circle_filter(val: udf.float32, row: udf.int64, col: udf.int64, nrows: udf.int64, ncols: udf.int64) -> udf.float32:
    x = (2. * row / nrows) - 1.
    y = (2. * col / ncols) - 1.
    if ((x ** 2 + y ** 2) <= 1) and val >= 0.5:
        return 1.
    return math.nan


@udf.scalar()
def square_filter(val: udf.float32) -> udf.float32:
    if val >= 0.5:
        return 1.
    return math.nan


@udf.jit()
def filter_func(out: udf.Array(udf.float32, 2), vals: udf.Array(udf.float32, 2),
                nrows: udf.int64, ncols: udf.int64, iscircle: udf.bool) -> udf.int32:
    n = out.window_shape[0]
    m = out.window_shape[1]
    row_start = out.window_start[0]
    col_start = out.window_start[1]
    for i in range(n):
        for j in range(m):
            if iscircle:
                out[i, j] = ulib.circle_filter(vals[i, j], row_start + i, col_start + j, nrows, ncols)
            else:
                out[i, j] = ulib.square_filter(vals[i, j])
    return 0

@profile
def iarray_rand():
    return ia.random.random_sample(shape)

rand_data = iarray_rand()

@profile
def iarray_computations():
    global circle, square
    expr = ia.expr_from_udf(filter_func, [rand_data], [shape[0], shape[1], True])
    circle = expr.eval()
    expr = ia.expr_from_udf(filter_func, [rand_data], [shape[0], shape[1], False])
    square = expr.eval()

iarray_computations()

@profile
def iarray_reductions():
    area_circle = ia.nansum(circle)
    area_square = ia.nansum(square)
    print(f"PI estimate: {4 * area_circle / area_square}")

iarray_reductions()
