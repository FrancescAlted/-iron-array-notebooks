import iarray as ia
from time import time


# contiguous = False
# nthreads = 21

# Some experiments lead to these chunk and block shapes
# chunks = (128, 256, 1440)
# blocks = (16, 16, 180)
# chunks = (180, 256, 1440)
# block = (12, 16, 180)
# chunks = (360, 128, 1440)
# blocks = (8, 8, 720)
# ia.set_config_defaults(chunks=chunks, blocks=blocks)

# cfg = ia.Config(contiguous=contiguous, chunks=chunks, blocks=blocks)
# cfg = ia.Config(nthreads=nthreads, store=store, clevel=1, codec=ia.Codec.ZSTD, filters=[ia.Filter.BITSHUFFLE])
# cfg = ia.Config(store=store, clevel=1, codec=ia.Codec.ZSTD, filters=[ia.Filter.BITSHUFFLE])
# cfg = ia.Config(nthreads=0, store=store, clevel=1, codec=ia.Codec.ZSTD, filters=[ia.Filter.BITSHUFFLE])
# cfg = ia.Config(contiguous=contiguous)
# print("cfg:", cfg)

# t0 = time()
# ia_precip = ia.open("precip-3m.iarr")
# precip1 = ia_precip[0].copy(cfg=cfg)
# precip2 = ia_precip[1].copy(cfg=cfg)
# precip3 = ia_precip[2].copy(cfg=cfg)
# t = time() - t0
# print("load time ->", round(t, 3))

# t0 = time()
# ia.save("precip1.iarr", precip1)
# ia.save("precip2.iarr", precip2)
# ia.save("precip3.iarr", precip3)
# t = time() - t0
# print("save time ->", round(t, 3))

t0 = time()
precip1 = ia.load("../tutorials/precip1.iarr")
precip2 = ia.load("../tutorials/precip2.iarr")
precip3 = ia.load("../tutorials/precip3.iarr")
t = time() - t0
print("load time ->", round(t, 3))

# t0 = time()
# precip1 = ia.open("precip1.iarr").copy()
# precip2 = ia.open("precip2.iarr").copy()
# precip3 = ia.open("precip3.iarr").copy()
# t = time() - t0
# print("open/copy time ->", round(t, 3))

precip1.info
print("cratio:", round(precip1.cratio, 3))


@profile
def iarray_mean_memory(expr):
    expr_val = expr.eval()
    return expr_val


mean_expr = (precip1 + precip2 + precip3) / 3
t0 = time()
mean_val = iarray_mean_memory(mean_expr)
t = time() - t0
print("mean time ->", round(t, 3))


@profile
def iarray_trans_memory(expr):
    expr_val = expr.eval()
    return expr_val


trans_expr = (
    ia.tan(precip1) * (ia.sin(precip1) * ia.sin(precip2) + ia.cos(precip2)) + ia.sqrt(precip3) * 2
)
t0 = time()
trans_val = iarray_trans_memory(trans_expr)
t = time() - t0
print("mean time ->", round(t, 3))
