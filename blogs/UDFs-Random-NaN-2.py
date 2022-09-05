import numpy as np
import iarray as ia
from iarray import udf
import math


# Params for array construction
shape = (50_000, 50_000)
ia.set_config_defaults(dtype=np.float32, favor=ia.Favor.SPEED)#, btune=False, clevel=0)


@udf.scalar()
def circle_filter(val: udf.float32, row: udf.int64, col: udf.int64, nrows: udf.int64, ncols: udf.int64) -> udf.float32:
    x = (2. * row / nrows) - 1.
    y = (2. * col / ncols) - 1.
    if ((x ** 2 + y ** 2) <= 1) and val >= 0.5:
        return 1.
    return math.nan

@udf.jit
def circle_fun(out: udf.Array(udf.float32, 2), vals: udf.Array(udf.float32, 2), nrows: udf.int64, ncols: udf.int64) -> udf.int32:
    n = out.window_shape[0]
    m = out.window_shape[1]
    row_start = out.window_start[0]
    col_start = out.window_start[1]
    for i in range(n):
        for j in range(m):
            out[i, j] = ulib.circle_filter(vals[i, j], row_start + i, col_start + j, nrows, ncols)
    return 0


@profile
def iarray_rand():
    return ia.random.random_sample(shape)

rand_data = iarray_rand()

@profile
def iarray_computations():
    global circle
    expr = ia.expr_from_udf(circle_fun, [rand_data], list(shape))
    circle = expr.eval()

iarray_computations()

@profile
def iarray_reductions():
    area_circle = ia.nansum(circle, axis=(1,0))
    area_square = ia.nansum(rand_data, axis=(1,0))
    print(f"PI value: {4 * area_circle / area_square}")

iarray_reductions()
