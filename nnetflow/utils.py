import cupy as cp
import numpy as np 


def is_cuda_available():
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False