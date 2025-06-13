from .engine import Tensor 
from .module import Module


class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = Tensor.ones((in_features, out_features))
        self.bias = Tensor.zeros((out_features,))

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
