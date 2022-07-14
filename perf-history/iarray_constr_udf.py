import iarray as ia
from time import time
import sys
from iarray.udf import jit, Array, float64, int64


@jit
def tri(out: Array(float64, 2), k: int64) -> int:
    n = out.window_shape[0]
    m = out.window_shape[1]
    row_start = out.window_start[0]
    col_start = out.window_start[1]
    for i in range(n):
        for j in range(m):
            if (row_start + i) >= (col_start + j - k):
                out[i, j] = 1
            else:
                out[i, j] = 0
    return 0


shape = (20 * 1024, 20 * 1024)
urlpath = None if str(sys.argv[1]) == "memory" else "array.iarr"
ia.remove_urlpath(urlpath)


@profile
def iarray_udf_constr():
    expr = ia.expr_from_udf(tri, [], [3], shape=shape)
    return expr.eval()


with ia.config(urlpath=urlpath):
    t0 = time()
    ia_out = iarray_udf_constr()
    t = time() - t0
print("udf time ->", round(t, 3))
ia.remove_urlpath(urlpath)
