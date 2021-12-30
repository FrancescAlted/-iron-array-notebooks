import iarray as ia

precip_opt = ia.open("../tutorials/precip-3m-optimal.iarr")

@profile
def iarray_reduc_memory(array):
    reduc = ia.mean(array, axis=(3, 2, 0))
    return reduc

precip = precip_opt.copy()
mean_disk = iarray_reduc_memory(precip)
