from .engine import Tensor 
from .module import Module
from .utils import is_cuda_available

ALLOWED_INITS = {"xavier_uniform", "xavier_normal", "he_uniform", "he_normal"}

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, device: str = 'cuda' if is_cuda_available() else 'cpu', init: str = 'xavier_uniform') -> None:
        """Fully-connected linear layer.

        Parameters
        ----------
        in_features: int
            Number of input features.
        out_features: int
            Number of output features.
        device: str
            'cpu' or 'cuda'. Defaults to CUDA if available.
        init: str
            Weight initialization scheme: one of 'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal'.
        """
        super().__init__()
        # Use proper weight initialization
        if init == 'xavier_uniform':
            self.weight = Tensor.xavier_uniform((in_features, out_features), device=device)
        elif init == 'xavier_normal':
            self.weight = Tensor.xavier_normal((in_features, out_features), device=device)
        elif init == 'he_uniform':
            self.weight = Tensor.he_uniform((in_features, out_features), device=device)
        elif init == 'he_normal':
            self.weight = Tensor.he_normal((in_features, out_features), device=device)
        else:
            raise ValueError(f"Unknown initialization scheme: {init}. Choose from: {sorted(ALLOWED_INITS)}")
        
        self.bias = Tensor.zeros((out_features,), device=device)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Linear(in_features={self.weight.shape[0]}, out_features={self.weight.shape[1]})"
    def __str__(self) -> str:
        return f"Linear Layer: {self.weight.shape[0]} -> {self.weight.shape[1]} units"
    
    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]
