import iarray as ia
from time import time
import sys
import os


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
    urlpath = str(sys.argv[2])+ "-3m.iarr"
print(str(open_method), " time ->", round(t, 3))

precip1.info
print("cratio:", round(precip1.cratio, 3))


@profile
def iarray_expr_eval(expr):
    expr_val = expr.eval()
    return expr_val


if sys.argv[2] == "mean":
    expr = (precip1 + precip2 + precip3) / 3
else:
    expr = (
            ia.tan(precip1) * (ia.sin(precip1) * ia.sin(precip2) + ia.cos(precip2)) + ia.sqrt(precip3) * 2
    )

ia.remove_urlpath(urlpath)
with ia.config(urlpath=urlpath):
    t0 = time()
    val = iarray_expr_eval(expr)
    t = time() - t0
print(str(expr), " time ->", round(t, 3))
ia.remove_urlpath(urlpath)
