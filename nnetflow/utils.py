import cupy as cp
import numpy as np 


def is_cuda_available() -> bool:
    """this function is used to check if there is a gpu """
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False
