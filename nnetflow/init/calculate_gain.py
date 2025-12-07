from typing import Literal 


ActivationType = Literal['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu', 'gelu'] 


def _calculate_gain(activation: ActivationType, param: float = None) -> float:
    """Return the recommended gain value for the given nonlinearity function.
    
    Args:
        activation (ActivationType): The nonlinearity function.
        param (float, optional): Optional parameter for certain activations.
        
    Returns:
        float: The recommended gain value.
    """
    if activation == 'relu':
        return 2.0 ** 0.5
    elif activation == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        else:
            negative_slope = param
        return (2.0 / (1 + negative_slope ** 2)) ** 0.5
    elif activation == 'tanh':
        return 5.0 / 3
    elif activation == 'sigmoid':
        return 1.0
    elif activation == 'elu':
        return 1.5505188080679277
    elif activation == 'selu':
        return 1.0
    elif activation == 'gelu':
        return 1.0
    else:
        raise ValueError(f"Unsupported activation type: {activation}")