import iarray as ia
from time import time
import sys
import os
import math
from iarray.udf import jit, Array, float32


@jit()
def mean(out: Array(float32, 3),
         p1: Array(float32, 3),
         p2: Array(float32, 3),
         p3: Array(float32, 3)) -> int:
    l = p1.window_shape[0]
    m = p1.window_shape[1]
    n = p1.window_shape[2]

    for i in range(l):
        for j in range(m):
            for k in range(n):
                value = p1[i, j, k] + p2[i, j, k] + p3[i, j, k]
                out[i, j, k] = value / 3

    return 0


@jit()
def trans(out: Array(float32, 3),
          p1: Array(float32, 3),
          p2: Array(float32, 3),
          p3: Array(float32, 3)) -> int:
    l = p1.window_shape[0]
    m = p1.window_shape[1]
    n = p1.window_shape[2]

    for i in range(l):
        for j in range(m):
            for k in range(n):
                value = math.sin(p1[i, j, k]) * math.sin(p2[i, j, k]) + math.cos(p2[i, j, k])
                value *= math.tan(p1[i, j, k])
                value += math.cosh(p3[i, j, k]) * 2
                out[i, j, k] = value

    return 0


open_method = sys.argv[1]
if str(open_method) == "load":
    t0 = time()
    precip1 = ia.load("../tutorials/precip1.iarr")
    precip2 = ia.load("../tutorials/precip2.iarr")
    precip3 = ia.load("../tutorials/precip3.iarr")
    t = time() - t0
    urlpath = None
else:
    cmd = 'vmtouch -e ../tutorials/precip1.iarr ../tutorials/precip2.iarr ../tutorials/precip3.iarr'
    os.system(cmd)
    t0 = time()
    precip1 = ia.open("../tutorials/precip1.iarr")
    precip2 = ia.open("../tutorials/precip2.iarr")
    precip3 = ia.open("../tutorials/precip3.iarr")
    t = time() - t0
    urlpath = str(sys.argv[2]) + "-3m.iarr"
print(str(open_method), " time ->", round(t, 3))

precip1.info
print("cratio:", round(precip1.cratio, 3))


@profile
def iarray_expr_udf(expr):
    return expr.eval()


@profile
def iarray_expr_lazy(expr):
    return expr.eval()


ia.remove_urlpath(urlpath)
if len(sys.argv) == 3:
    with ia.config(urlpath=urlpath):
        if sys.argv[2] == "mean":
            expr = ia.expr_from_udf(mean, [precip1, precip2, precip3])
        else:
            expr = ia.expr_from_udf(trans, [precip1, precip2, precip3])

        t0 = time()
        val = iarray_expr_udf(expr)
        t = time() - t0
    print(str(expr), " time ->", round(t, 3))
else:  # Lazy expression
    with ia.config(urlpath=urlpath):
        if sys.argv[2] == "mean":
            t0 = time()
            expr = (precip1 + precip2 + precip3) / 3
            val = iarray_expr_lazy(expr)
            t = time() - t0
        else:
            t0 = time()
            expr = ia.tan(precip1) * (ia.sin(precip1) * ia.sin(precip2) + ia.cos(precip2)) + ia.sqrt(
                precip3) * 2
            val = iarray_expr_lazy(expr)
            t = time() - t0
    print(str(sys.argv[2]), " time ->", round(t, 3))

ia.remove_urlpath(urlpath)
