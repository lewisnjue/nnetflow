import numpy as np 
from nnetflow.engine import Tensor 

class Linear:
    def __init__(self,in_features:int, out_features:int) -> None: 
        self.in_features = in_features 
        self.out_features = out_features 
        _weight = np.random.randn(in_features, out_features) * np.sqrt(2. / in_features) 
        _bias = np.zeros((1, out_features)) 
        self.weight = Tensor(_weight, requires_grad=True)
        self.bias = Tensor(_bias, requires_grad=True)
    
    def __call__(self,x:Tensor) -> Tensor:
        assert x.shape[-1] == self.in_features, f"Input feature size mismatch, expected {self.in_features}, got {x.shape[-1]}"
        assert len(x.shape) == 2, f"Input tensor must be 2D, got {len(x.shape)}D"  
        # x : (batch_size, in_features) 
        # weight : (in_features, out_features) 
        # x @ weight (batch_size,in_features) @ (in_features, out_features) = (batch_size, out_features)
        return x @ self.weight + self.bias 
    

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features})" 
    
    def __str__(self):
        return self.__repr__()
    

class Flatten:
    def __init__(self) -> None:
        pass 
    
    def __call__(self,x:Tensor) -> Tensor:
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
    
    def __repr__(self) -> str:
        return "Flatten()"
    
    def __str__(self) -> str:
        return self.__repr__()
