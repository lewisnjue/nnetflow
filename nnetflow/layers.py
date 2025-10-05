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


class Dropout(Module):
    def __init__(self, p: float = 0.5, device: str = 'cpu') -> None:
        super().__init__()
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        self.p = p
        self.device = device
        self._mask = None

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        if not training or self.p == 0.0:
            return x
        xp = __import__('numpy') if self.device == 'cpu' else __import__('cupy')
        mask = (xp.random.rand(*x.shape) >= self.p).astype(x.data.dtype) / (1.0 - self.p)
        self._mask = Tensor(mask, device=self.device, require_grad=False)
        return x * self._mask

    def parameters(self) -> list[Tensor]:
        return []


class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, device: str = 'cpu') -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.device = device
        self.gamma = Tensor.ones((num_features,), device=device)
        self.beta = Tensor.zeros((num_features,), device=device)
        self.running_mean = Tensor.zeros((num_features,), device=device, require_grad=False)
        self.running_var = Tensor.ones((num_features,), device=device, require_grad=False)

    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        # x shape: (N, C)
        if training:
            mean = x.mean(axis=0)
            var = (x - mean).pow(2).mean(axis=0)
            # Update running stats (no grad)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        x_hat = (x - mean) / (var + Tensor(self.eps, device=x.device, require_grad=False)).pow(0.5)
        return self.gamma * x_hat + self.beta

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]
