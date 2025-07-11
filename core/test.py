import numpy as np
import matmul

B, T, C, OC = 1, 2, 3, 2
inp = np.array([1,2,3,4,5,6], dtype=np.float32)
weight = np.array([1,0,-1,0,1,2], dtype=np.float32)
bias = np.array([0.5,-0.5], dtype=np.float32)
out = np.zeros((B*T*OC,), dtype=np.float32)

matmul.matmul_forward_cpu(inp, weight, bias, B, T, C, OC, out)
print(out)