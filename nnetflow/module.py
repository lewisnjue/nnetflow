import os
import pickle
import numpy as np
import numpy.typing as npt
from typing import Union, List, Tuple
import nnetflow
from nnetflow.engine import Tensor 

class Module:
    def __init__(self):
        """
        Base Module class. 
        """
        self.training = True

    def to(self, dtype: npt.DTypeLike) -> 'Module':
        """
        Casts all Tensors in this module (and sub-modules) to the specific dtype.
        Modifies Tensors in-place for memory efficiency.
        """
        def cast_fn(x):
            if hasattr(x, 'data'): 
                if x.data.dtype != dtype:
                    x.data = x.data.astype(dtype)
                
                if x.grad is not None and x.grad.dtype != dtype:
                    is_integer = np.issubdtype(dtype, np.integer)
                    target_grad_dtype = np.float64 if is_integer else dtype
                    x.grad = x.grad.astype(target_grad_dtype)
                return x
            
            if isinstance(x, np.ndarray):
                return x.astype(dtype)
                
            return x

        self.apply(cast_fn)
        return self

    def apply(self, fn): 
        """
        Applies a function `fn` to all Tensor attributes of this module and its sub-modules.
        """
        for name, value in vars(self).items():
            if hasattr(value, 'data'):
                setattr(self, name, fn(value))
            
            elif isinstance(value, Module):
                value.apply(fn)
            
            elif isinstance(value, (list, tuple)):
                new_list = []
                for item in value:
                    if isinstance(item, Module):
                        item.apply(fn)
                        new_list.append(item)
                    elif hasattr(item, 'data'): 
                         new_list.append(fn(item))
                    else:
                        new_list.append(item)
                
                if isinstance(value, list):
                    setattr(self, name, new_list)
                else:
                    setattr(self, name, tuple(new_list))
        return self

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def state_dict(self, prefix="") -> dict:
        state = {}
        for name, value in vars(self).items():
            if hasattr(value, 'data'):
                key = f"{prefix}{name}"
                state[key] = value.data
            elif isinstance(value, Module):
                child_state = value.state_dict(prefix=f"{prefix}{name}.")
                state.update(child_state)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        child_state = item.state_dict(prefix=f"{prefix}{name}.{i}.")
                        state.update(child_state)
                    if isinstance(item, Tensor):
                        key = f"{prefix}{name}.{i}"
                        state[key] = item.data
        return state
    
    def load(self, filepath: str, weights_only=True) -> None:
        """
        Loads the model from a file.
        Args:
            filepath: Path to load the file from (e.g., 'model.pkl')
            weights_only:
                If True: Loads only the dictionary of numbers (Safer, recommended).
                If False: Unpickles the entire object (Includes architecture).
        """
        path = str(filepath)
        with open(path, 'rb') as f:
            if weights_only:
                state_dict = pickle.load(f)
                self.load_state_dict(state_dict)
                print(f"Loaded weights from {path}")
            else:
                loaded_model = pickle.load(f)
                self.__dict__.update(loaded_model.__dict__)
                print(f"Loaded full model from {path}")

    def load_state_dict(self, state_dict: dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            parts = name.split('.')
            obj = self
            try:
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                target_tensor = getattr(obj, parts[-1])
                target_tensor.data = param
            except AttributeError:
                pass

    def parameters(self):
        params = []
        for name, value in vars(self).items():
            if hasattr(value, 'data'):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def train(self) -> None:
        self.training = True
        for v in vars(self).values():
            if isinstance(v, Module): v.train()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module): item.train()

    def eval(self) -> None:
        self.training = False
        for v in vars(self).values():
            if isinstance(v, Module): v.eval()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module): item.eval()


    def save(self, filepath: str, weights_only=True) -> None:
            """
            Saves the model to a file.
            Args:
                filepath: Path to save the file (e.g., 'model.pkl')
                weights_only:
                    If True: Saves only the dictionary of numbers (Safer, recommended).
                    If False: Pickles the entire object (Includes architecture).
            """
            path = str(filepath)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'wb') as f:
                if weights_only:
                    pickle.dump(self.state_dict(), f)
                    print(f"Saved weights to {path}")
                else:
                    pickle.dump(self, f)
                    print(f"Saved full model to {path}")
