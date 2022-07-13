import iarray as ia
from time import time
import sys
import os


open_method = sys.argv[1]
if str(open_method) == "load":
    t0 = time()
    ia_precip = ia.load("../tutorials/precip-3m.iarr")
    t = time() - t0
else:
    cmd = 'vmtouch -e ../tutorials/precip-3m.iarr'
    os.system(cmd)
    t0 = time()
    ia_precip = ia.open("../tutorials/precip-3m.iarr")
    t = time() - t0
print(str(open_method), " time ->", round(t, 3))



print("cratio:", round(ia_precip.cratio, 3))


@profile
def iarray_red(op):
    return getattr(ia, op)(ia_precip, axis=(3, 0))

@profile
def iarray_red_os(op, oneshot):
    return getattr(ia, op)(ia_precip, axis=(3, 0), oneshot=oneshot)

op = sys.argv[2]
if len(sys.argv) == 4:
    t0 = time()
    ia_reduc = iarray_red_os(op, sys.argv[3])
    t = time() - t0
else:
    t0 = time()
    ia_reduc = iarray_red(op)
    t = time() - t0

print(str(op), "time ->", round(t, 3))
