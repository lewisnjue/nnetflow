import numpy as np 


# xavier initialization or glorot initialization 
""" 
normal distribution: 
    mean = 0 
    variance  = 1  / (fan_avg) # where fan_avg = (fan_in + fan_out) / 2 

uniform distribution: 
    -r to r where r = sqrt(3/fan_avg) 

if you replace fan_avg with fan_in you get lecun initialization 


for relu activation function we use he initialization or kaiming initialization , for SELU we use lecun initialization and the most recommended 
is with normal distribution 

INITIALIZATION   ACTIVATION FUNCTION    VARIANCE(NORMAL)
------------------------------------------------------- 
 xavier/glorot  none,softmax,tanh, sigmoid         1/(fan_avg)
 he/kaming           relu, leaky relu      2/(fan_in)
    lecun         selu                  1/(fan_in)

"""
"""
inputs_  = np.random.randn(3,5) # shape (batch_size, fan_in) 

layer_1 = np.random.randn(5,4) * 1 / 8

layer_2 = np.random.randn(4,2) * 1 / 3


out = inputs_.dot(layer_1).dot(layer_2) 

print(inputs_.var())
print(inputs_.dot(layer_1).var()) 
print(inputs_.dot(layer_1).dot(layer_2).var()) 
print(out.var())

"""
def he_normal(shape):
    fan_in = shape[0]
    stddev = np.sqrt(2 / fan_in)
    return np.random.normal(0, 1, shape) * stddev  # Scale standard normal by stddev

def he_uniform(shape):
    fan_in  = shape[0]
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, shape)  # Directly sample in scaled range


def xavier_normal(shape):
    fan_in, fan_out = shape[0],shape[1] 
    fan_avg = (fan_in + fan_out)/2 
    stddev = np.sqrt(1/fan_avg)
    return np.random.normal(0, 1, shape) * stddev  # Scale standard normal by stddev  

def xavier_uniform(shape): 
    fan_in, fan_out = shape[0],shape[1] 
    fan_avg = (fan_in + fan_out)/2 
    limit = np.sqrt(3 / fan_avg)
    return np.random.uniform(-limit, limit, shape)  # Directly sample in scaled range