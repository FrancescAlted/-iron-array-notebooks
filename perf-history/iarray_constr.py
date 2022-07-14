import iarray as ia
from time import time
import numpy as np
import sys


func = str(sys.argv[2])

shape = (20 * 1024, 20 * 1024)
size = np.prod(shape)
chunks = (4 * 1024, 4 * 1024)
blocks = (256, 256)
ia.reset_config_defaults()

dtype = getattr(np, sys.argv[3])
urlpath = None if str(sys.argv[1]) == "memory" else "array.iarr"
ia.remove_urlpath(urlpath)
if func == "ones":
    @profile
    def iarray_ones_constr():
        return getattr(ia, func)(shape, chunks=chunks, blocks=blocks, dtype=dtype, urlpath=urlpath)

    t0 = time()
    ia_out = iarray_ones_constr()
    t = time() - t0
elif func == "full":
    @profile
    def iarray_full_constr():
        return ia.full(shape, fill_value, chunks=chunks, blocks=blocks, dtype=dtype, urlpath=urlpath)
    if dtype in [np.int32, np.int64]:
        fill_value = 5
    else:
        fill_value = 3.14
    t0 = time()
    ia_out = iarray_full_constr()
    t = time() - t0
elif func == "arange":
    @profile
    def iarray_arange_constr():
        return ia.arange(shape, 0, 1, chunks=chunks, blocks=blocks, dtype=dtype, urlpath=urlpath)

    t0 = time()
    ia_out = iarray_arange_constr()
    t = time() - t0
elif func == "linspace":
    @profile
    def iarray_linspace_constr():
        return ia.linspace(shape, 0, 5, chunks=chunks, blocks=blocks, dtype=dtype, urlpath=urlpath)
    t0 = time()
    ia_out = iarray_linspace_constr()
    t = time() - t0
elif func == "standard_normal":
    @profile
    def iarray_standard_normal(fp_mantissa_bits):
        return ia.random.standard_normal(shape=shape, chunks=chunks, blocks=blocks, dtype=dtype,
                                         fp_mantissa_bits=fp_mantissa_bits, urlpath=urlpath)

    t0 = time()
    ia_out = iarray_standard_normal(int(sys.argv[4]))
    t = time() - t0
else:
    @profile
    def iarray_poisson():
        return ia.random.poisson(shape=shape, lam=3, chunks=chunks, blocks=blocks, dtype=dtype, urlpath=urlpath)
    t0 = time()
    ia_out = iarray_poisson()
    t = time() - t0
print(func, " time ->", round(t, 3))
ia.remove_urlpath(urlpath)
