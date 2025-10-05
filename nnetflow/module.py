from typing import List, Literal, Optional, Any
from .engine import Tensor
import pickle

class Module:
    """Base class for neural network modules.

    Handles parameter and submodule registration, forward invocation,
    and serialization of parameter tensors.
    """
    def __init__(self) -> None:
        self._parameters: List[Tensor] = []
        self._modules: List[Module] = []

    def register_parameter(self, param: Tensor) -> None:
        self._parameters.append(param)

    def register_module(self, module: 'Module') -> None:
        self._modules.append(module)

    def params(self) -> List[Tensor]:
        """Return a flat list of all parameters for this module tree."""
        params = list(self._parameters)
        for m in self._modules:
            params.extend(m.params())
        return params

    def add_parameter(self, name: str, param: Tensor) -> None:
        """Attach a parameter tensor and register it automatically."""
        setattr(self, name, param)
        self.register_parameter(param)

    def add_module(self, name: str, module: 'Module') -> None:
        """Attach a child module and register it automatically."""
        setattr(self, name, module)
        self.register_module(module)

    def zero_grad(self) -> None:
        """Zero gradients of all parameters if they track gradients."""
        for p in self.params():
            if hasattr(p, 'zero_grad'):
                p.zero_grad()

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Tensor):
            self.register_parameter(value)
        elif isinstance(value, Module):
            self.register_module(value)
        super().__setattr__(name, value)
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Forward method must be implemented in subclasses.")
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def parameters(self) -> List[Tensor]:
        """Alias for params()."""
        return self.params()
    
    def save(self, path: str) -> None:
        """Save parameter arrays to a pickle file.

        Only parameter data is saved for portability; architecture code is not serialized.
        """
        # Save only parameter data for portability
        state = {name: getattr(self, name).numpy() for name in dir(self) if isinstance(getattr(self, name), Tensor)}
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> 'Module':
        """Load parameters from a pickle file into a new instance of cls.

        Note: This creates a blank instance; user code must re-attach parameters as needed.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__init__()
        for name, arr in state.items():
            setattr(obj, name, Tensor(arr))
        return obj
