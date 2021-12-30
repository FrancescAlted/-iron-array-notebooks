import numpy as np
import iarray as ia

ia_precip = ia.open("../tutorials/precip-3m.iarr")

precip1 = ia_precip[0].data
precip2 = ia_precip[1].data
precip3 = ia_precip[2].data

@profile
def numpy_mean_memory():
    expr_val = (precip1 + precip2 + precip3) / 3
    return expr_val

mean_val = numpy_mean_memory()

@profile
def numpy_trans_memory():
    expr_val = np.tan(precip1) * (np.sin(precip1) * np.sin(precip2) + np.cos(precip2)) + np.sqrt(precip3) * 2
    return expr_val

trans_val = numpy_trans_memory()
