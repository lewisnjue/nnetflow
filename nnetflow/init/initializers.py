
""" initializers for layers weights and bias  """
from nnetflow.engine import Tensor 
from nnetflow.init.calculate_gain import _calculate_gain 
from nnetflow.device import get_array_module
import numpy as np 
from typing import Literal


def xavier_uniform(wights:Tensor,gain:float=1.0)->None: 
    """ 
    Initialize the weights tensor using xavier uniform initialization 
    Args: 
        weights: the weights tensor to initialize 
        gain: the gain factor to use (default is 1.0) 
    Returns: 
        None 
    """ 
    if not isinstance(wights,Tensor): 
        raise TypeError("weights must be a Tensor") 
    if wights.data.ndim <2: 
        raise ValueError("weights tensor must have at least 2 dimensions") 
    fan_in = wights.data.shape[0]
    fan_out = wights.data.shape[1]  
    xp = get_array_module()
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))  # np.sqrt for constant
    wights.data = xp.random.uniform(-limit, limit, size=wights.data.shape)


def xavier_normal(weights:Tensor,gain:float=1.0)->None: 
    """ 
    Initialize the weights tensor using xavier normal initialization 
    Args: 
        weights: the weights tensor to initialize 
        gain: the gain factor to use (default is 1.0) 
    Returns: 
        None 
    """ 
    if not isinstance(weights,Tensor): 
        raise TypeError("weights must be a Tensor") 
    if weights.data.ndim <2: 
        raise ValueError("weights tensor must have at least 2 dimensions") 
    fan_in = weights.data.shape[0]
    fan_out = weights.data.shape[1]  
    xp = get_array_module()
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))  # np.sqrt for constant
    weights.data = xp.random.normal(0.0, std, size=weights.data.shape)



_non_linearity  = Literal['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu', 'gelu'] 



def He_uniform(weights:Tensor,mode:Literal['fan_in','fan_out','fan_avg']='fan_in', nonlinearity:_non_linearity = 'relu')->None:  
    """ 
    initalize the weight tensor using He uniform initialization
    Args: 
        weights: the weights tensor to initialize 
        mode: the mode to use ('fan_in', 'fan_out', 'fan_avg') 
        nonlinearity: the non-linearity function to use (default is 'relu') 
    Returns: 
        None 
    """
    gain = _calculate_gain(nonlinearity) 
    if not isinstance(weights,Tensor): 
        raise TypeError("weights must be a Tensor") 
    if weights.data.ndim <2: 
        raise ValueError("weights tensor must have at least 2 dimensions") 
    fan_in = weights.data.shape[0] 
    fan_out = weights.data.shape[1] 
    if mode == 'fan_in': 
        fan = fan_in 
    elif mode == 'fan_out': 
        fan = fan_out 
    elif mode == 'fan_avg': 
        fan = (fan_in + fan_out) / 2 
    else: 
        raise ValueError("mode must be 'fan_in', 'fan_out', or 'fan_avg'") 
    
    xp = get_array_module()
    limit = gain * np.sqrt(3.0 / fan)  # np.sqrt for constant
    weights.data = xp.random.uniform(-limit, limit, size=weights.data.shape) 

def He_normal(weights:Tensor,mode:Literal['fan_in','fan_out','fan_avg']='fan_in', nonlinearity:_non_linearity = 'relu')->None:  
    """ 
    initalize the weight tensor using He normal initialization
    Args: 
        weights: the weights tensor to initialize 
        mode: the mode to use ('fan_in', 'fan_out', 'fan_avg') 
        nonlinearity: the non-linearity function to use (default is 'relu') 
    Returns: 
        None 
    """
    gain = _calculate_gain(nonlinearity) 
    if not isinstance(weights,Tensor): 
        raise TypeError("weights must be a Tensor") 
    if weights.data.ndim <2: 
        raise ValueError("weights tensor must have at least 2 dimensions") 
    fan_in = weights.data.shape[0] 
    fan_out = weights.data.shape[1] 
    if mode == 'fan_in': 
        fan = fan_in 
    elif mode == 'fan_out': 
        fan = fan_out 
    elif mode == 'fan_avg': 
        fan = (fan_in + fan_out) / 2 
    else: 
        raise ValueError("mode must be 'fan_in', 'fan_out', or 'fan_avg'") 
    
    xp = get_array_module()
    std = gain * np.sqrt(2.0 / fan)  # np.sqrt for constant
    weights.data = xp.random.normal(0.0, std, size=weights.data.shape) 
