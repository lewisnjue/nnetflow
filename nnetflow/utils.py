import numpy as np 
try:
    import cupy as cp 
except Exception as e: 
    print(f"Cannot improt cupy:: you cant use GPU the error is :: {e}")



def is_cuda_available() -> bool:
    """this function is used to check if there is a gpu """
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except cp.cuda.runtime.CUDARuntimeError:
        return False
    except Exception as e: 
        print(e)
        return False
