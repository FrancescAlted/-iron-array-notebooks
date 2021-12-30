from time import time
import os

t0 = time()
import iarray as ia

# chunks = (360, 128, 1440)
# blocks = (8, 8, 720)
# ia.set_config_defaults(chunks=chunks, blocks=blocks)

t = time() - t0
print("iarray import time ->", round(t, 3))
# ia_precip = ia.open("precip-3m.iarr")
# chunks = ia_precip.chunks[1:]
# blocks = ia_precip.blocks[1:]

# print("-->", ia_precip.info)

# precip1 = ia_precip[0].copy(chunks=chunks, blocks=blocks)
# precip2 = ia_precip[1].copy(chunks=chunks, blocks=blocks)
# precip3 = ia_precip[2].copy(chunks=chunks, blocks=blocks)
# print("1 -->", precip1.info)

# precip1 = ia_precip[0].copy()
# precip2 = ia_precip[1].copy()
# precip3 = ia_precip[2].copy()
# print("1 -->", precip1.info)

# ia.save("precip1.iarr", precip1)
# ia.save("precip2.iarr", precip2)
# ia.save("precip3.iarr", precip3)

# @profile
def iarray_open_data():
    precip1 = ia.open("../tutorials/precip1.iarr")
    precip2 = ia.open("../tutorials/precip2.iarr")
    precip3 = ia.open("../tutorials/precip3.iarr")
    return precip1, precip2, precip3

cmd = 'vmtouch -e ../tutorials/precip1.iarr ../tutorials/precip2.iarr ../tutorials/precip3.iarr'
os.system(cmd)
t0 = time()
precip1, precip2, precip3 = iarray_open_data()
t = time() - t0
print("open time ->", round(t, 3))

chunks = precip1.chunks
blocks = precip1.blocks

# cfg = ia.Config(nthreads=14, chunks=precip1.chunks, blocks=precip1.blocks, clevel=1, codec=ia.Codec.ZSTD, filters=[ia.Filter.BITSHUFFLE])
# cfg = ia.Config(chunks=precip1.chunks, blocks=precip1.blocks)


@profile
def iarray_mean_disk(expr):
    with ia.config(urlpath="mean-3m.iarr"):
        expr_val = expr.eval(mode="w")
    return expr_val

os.system(cmd)
t0 = time()
mean_expr = (precip1 + precip2 + precip3) / 3
t = time() - t0
print("mean expr time ->", round(t, 3))

if os.path.exists("mean-3m.iarr"): ia.remove_urlpath("mean-3m.iarr")
t0 = time()
mean_disk = iarray_mean_disk(mean_expr)
t = time() - t0
print("mean eval time ->", round(t, 3))
mean_disk.info


@profile
def iarray_trans_disk(expr):
    with ia.config(urlpath="trans-3m.iarr"):
        expr_val = expr.eval(mode="w")
    return expr_val

os.system(cmd)
trans_expr = (
    ia.tan(precip1) * (ia.sin(precip1) * ia.sin(precip2) + ia.cos(precip2)) + ia.sqrt(precip3) * 2
)
if os.path.exists("trans-3m.iarr"): ia.remove_urlpath("trans-3m.iarr")
t0 = time()
trans_val = iarray_trans_disk(trans_expr)
t = time() - t0
print("trans eval time ->", round(t, 3))
