import numpy as np
import scipy.special as sp 
#activation functions

def RELU(x: np.ndarray) -> np.ndarray: 
    """
    """

    return np.where(x > 0, x, 0) 


def LEAKY_RELU(x:np.ndarray,alpha:float=0.01) -> np.ndarray:
    """
    gradients never die :) they can go into comma but never die, 
    we also have other variance of leaky relu like parametric relu and randomized relu where alpha is learead in parametric relu  
    and in randomized relu apha is randomly sampled from a range , you can impemnt this i you like 
    """
    return np.where(x >0,x, alpha * x) 


""" 
all activations we have suffer because they  dont have a smooth curve and the gradient change abruptly at 0 , this is not good for optimization we want smooth curve 

"""


def ELU(x:np.ndarray,alpha:float=1.0) -> np.ndarray: 
    """
    experimentally better than relu and leaky relu but computationally expensive because of exp operation 
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))



def SELU(x:np.ndarray, lambda_:float=1.0507, alpha:float=1.67326) -> np.ndarray:
    """
    scaled exponential linear unit 
    self normalizing neural networks 
    https://arxiv.org/abs/1706.02515
    """
    return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))


def GELU(x:np.ndarray) -> np.ndarray:
    """
    gaussian error linear unit 
    https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + sp.erf(x / np.sqrt(2))) 

def SWISH(x:np.ndarray, beta:float=1.0) -> np.ndarray:
    """
    swish activation function 
    https://arxiv.org/abs/1710.05941
    """
    return x * (1 / (1 + np.exp(-beta * x)))