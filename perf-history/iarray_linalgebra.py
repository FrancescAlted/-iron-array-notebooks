import iarray as ia
from time import time
import sys
import numpy as np
import os

ia.set_config_defaults(dtype=np.float64)

func = str(sys.argv[2])
shape = (100_000, 25000, 1000)
amshape = (shape[0], shape[1])
bmshape = (shape[1], shape[2])

# Obtain optimal chunk and block shapes
t0 = time()
mparams = ia.matmul_params(amshape, bmshape)
amchunks, amblocks, bmchunks, bmblocks = mparams
t = time() - t0
print("matmul_params time ->", round(t, 3))


filename = "../tutorials/arr-gemm.iarr"
cmd = 'vmtouch -e ' + filename 
if not os.path.exists(filename):
    t0 = time()
    ia.set_config_defaults(btune=False)
    am = ia.random.normal(amshape, 3, 2, chunks=amchunks, blocks=amblocks, urlpath=filename, fp_mantissa_bits=20)
    t = time() - t0
    print("random.normal time ->", round(t, 3))
os.system(cmd)

open_method = str(sys.argv[1])
if open_method == "open":    
    t0 = time()
    am = ia.open(filename)
    t = time() - t0
    print("open time ->", round(t, 3))        
    urlpathb = "biarray.iarr"
    res_urlpath = "res.iarr"
else:
    t0 = time()
    am = ia.load(filename)
    t = time() - t0
    print("load time ->", round(t, 3))
    urlpathb = None
    res_urlpath = None

print(am.info)


ia.remove_urlpath(res_urlpath)
func = str(sys.argv[2])
if func == "matmul":
    ia.remove_urlpath(urlpathb)
    w = np.ones(bmshape)
    t0 = time()
    bm = ia.numpy2iarray(w, chunks=bmchunks, blocks=bmblocks, urlpath=urlpathb)
    t = time() - t0
    print("numpy2iarray time ->", round(t, 3))
    print(bm.info)

    if open_method == "open":
        os.system(cmd)

    @profile
    def iarray_matmul(a, b):
        return ia.matmul(a, b, urlpath=res_urlpath)

    t0 = time()
    iacm_opt = iarray_matmul(am, bm)
    t = time() - t0
    ia.remove_urlpath(urlpathb)
    print(func, " time ->", round(t, 3))
else:
    if open_method == "open":
        os.system(cmd)
    # @profile
    def iarray_transpose(array, path):
        return array.transpose(urlpath=path)
    @profile
    def iarray_copy(array, path):
        return array.copy(urlpath=path)

    if res_urlpath is not None:
        apath ="a.iarr"
    else:
        apath = None
    t0 = time()
    a_opt = iarray_transpose(am, apath)
    t = time() - t0
    print("am ", func, " time ->", round(t, 3))
    
    t0 = time()
    acopy = iarray_copy(a_opt, apath)
    t = time() - t0
    print("a copy time ->", round(t, 3))
    ia.remove_urlpath(apath)

ia.remove_urlpath(res_urlpath)

