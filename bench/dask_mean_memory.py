from functools import reduce
import numpy as np
import iarray as ia
import dask
import dask.array as da
import zarr


# Use a previous computation for getting metadata (shape, chunks...)
precip_disk = ia.open("../tutorials/precip-3m.iarr")
shape = precip_disk.shape[1:]
dtype = np.float32
clevel = ia.Config().clevel
compressor = zarr.Blosc(clevel=1, cname="zstd", shuffle=zarr.Blosc.BITSHUFFLE)

#zprecip1 = precip_disk[0]
#zprecip2 = precip_disk[1]
#zprecip3 = precip_disk[2]
#chunks = precip_disk.chunkshape[1:]

zprecip1 = zarr.open("precip1-op.zarr")
zprecip2 = zarr.open("precip2-op.zarr")
zprecip3 = zarr.open("precip3-op.zarr")
chunks = zprecip1.chunks

zarr_precip1 = zarr.create(shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)
zarr_precip2 = zarr.create(shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)
zarr_precip3 = zarr.create(shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)

#zprecip1.copyto(zarr_precip1)
#zprecip2.copyto(zarr_precip2)
#zprecip3.copyto(zarr_precip3)

zarr_precip1[:] = zprecip1
zarr_precip2[:] = zprecip2
zarr_precip3[:] = zprecip3

precip1 = da.from_zarr(zarr_precip1)
precip2 = da.from_zarr(zarr_precip2)
precip3 = da.from_zarr(zarr_precip3)


@profile
def dask_mean_memory(expr):
    with dask.config.set(scheduler="threads"):
        expr_val = zarr.create(
            shape=shape,
            #chunks=chunks,
            dtype=dtype,
            compressor=compressor,
        )
        da.to_zarr(expr, expr_val)
    return expr_val

mean_expr = (precip1 + precip2 + precip3) / 3
mean_memory = dask_mean_memory(mean_expr)
print(mean_memory.info)

@profile
def dask_trans_memory(expr):
    with dask.config.set(scheduler="threads"):
        expr_val = zarr.create(
            shape=shape,
            #chunks=chunks,
            dtype=dtype,
            compressor=compressor,
        )
        da.to_zarr(expr, expr_val)
    return expr_val

trans_expr = np.tan(precip1) * (np.sin(precip1) * np.sin(precip2) + np.cos(precip2)) + np.sqrt(precip3) * 2
trans_memory = dask_trans_memory(trans_expr)
