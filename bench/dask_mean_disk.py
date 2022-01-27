from functools import reduce
import numpy as np
import iarray as ia
import dask
import dask.array as da
import zarr
import os

cmd = 'vmtouch -e precip1-op.zarr precip2-op.zarr precip3-op.zarr'
os.system(cmd)
zprecip1 = zarr.open("precip1-op.zarr")
zprecip2 = zarr.open("precip2-op.zarr")
zprecip3 = zarr.open("precip3-op.zarr")
shape = zprecip1.shape
dtype = zprecip1.dtype

precip1 = da.from_zarr(zprecip1)
precip2 = da.from_zarr(zprecip2)
precip3 = da.from_zarr(zprecip3)


@profile
def dask_mean_disk(expr):
    with dask.config.set(scheduler="threads", num_workers=None):
        expr_val = zarr.open(
            "mean-3m.zarr",
            "w",
            shape=shape,
            dtype=dtype,
        )
        da.to_zarr(expr, expr_val)
    return expr_val

os.system(cmd)
mean_expr = (precip1 + precip2 + precip3) / 3
mean_disk = dask_mean_disk(mean_expr)

@profile
def dask_trans_disk(expr):
    with dask.config.set(scheduler="threads", num_workers=None):
        expr_val = zarr.open(
            "trans-3m.zarr",
            "w",
            shape=shape,
            dtype=dtype,
        )
        da.to_zarr(expr, expr_val)
    return expr_val

os.system(cmd)
trans_expr = np.tan(precip1) * (np.sin(precip1) * np.sin(precip2) + np.cos(precip2)) + np.sqrt(precip3) * 2
trans_disk = dask_trans_disk(trans_expr)
