from .engine import Tensor 
from .module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Tensor.zeros((out_features, in_features))
        self.bias = Tensor.zeros((out_features,))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T + self.bias

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Linear(in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]})"
    def __str__(self) -> str:
        return f"Linear Layer: {self.weight.shape[1]} -> {self.weight.shape[0]} units"
    
    def parameters(self) -> list[Tensor]:
        return [self.weight, self.bias]
