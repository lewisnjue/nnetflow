from nnetflow.engine import Tensor
from typing import List
from nnetflow.module import Module
class SGD(Module):
    """Stochastic Gradient Descent optimizer with optional momentum."""
    def __init__(self, params: List[Tensor],
     lr: float = 0.01, momentum: float = 0.0,
     nesterov=False,use_max_norm:bool = False,
     r: float = 1.0,grad_clip:bool = False,clip_value: float = 1.0) -> None:
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov= nesterov 
        self.use_max_norm = use_max_norm 
        self.r = r 
        self.grad_clip = grad_clip
        self.clip_value = clip_value
        self.velocities = [Tensor.zeros_like(p) for p in params]
    

    def forward(self, *args, **kwargs):
        return None

    def step(self):
        for i, param in enumerate(self.params):
            if not param.requires_grad:
                continue
            if param.grad is None:
                continue
            grad = param.grad
            if self.grad_clip: 
                grad_norm = np.linalg.norm(grad.data) 
                if grad_norm > self.clip_value: 
                    grad.data = grad.data * (self.clip_value / grad_norm)
            if self.nesterov and self.momentum > 0:
                prev_velocity = self.velocities[i].data.copy()
                self.velocities[i].data = (
                    self.momentum * self.velocities[i].data
                    - self.lr * grad
                )
                param.data += (
                    -self.momentum * prev_velocity
                    + (1 + self.momentum) * self.velocities[i].data
                )
            elif self.momentum > 0:
                self.velocities[i].data = (
                    self.momentum * self.velocities[i].data
                    - self.lr * grad
                )
                param.data += self.velocities[i].data
            else:
                param.data -= self.lr * grad

            if self.use_max_norm: 
                norm = np.linalg.norm(param.data) # L2 norm 
                if norm > self.r: 
                    param.data = param.data * (self.r / norm)

    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for param in self.params:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()
    
    def state_dict(self, prefix=""):
        state = {}
        state[f"{prefix}.lr"] = self.lr 
        state[f"{prefix}.momentum"] = self.momentum
        state[f"{prefix}.nesterov"] = self.nesterov 
        state[f"{prefix}.use_max_norm"] = self.use_max_norm
        state[f"{prefix}.r"] = self.r
        state[f"{prefix}.grad_clip"] = self.grad_clip
        state[f"{prefix}.clip_value"] = self.clip_value
        for i, velocity in enumerate(self.velocities):
            state[f"{prefix}.velocity.{i}"] = velocity.data
        return state

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.momentum})"
    def __str__(self) -> str:
        return f"SGD Optimizer: lr={self.lr}, momentum={self.momentum}"


class Adagrad(Module): 
    """Adagrad optimizer.""" 
    def __init__(self,params:List[Tensor],lr:float=0.01,
    eps:float=1e-8,use_max_norm:bool = False,r: float = 1.0,grad_clip:bool = False,clip_value: float = 1.0) -> None:
        self.params = params
        self.lr = lr
        self.eps = eps
        self.accumulators = [Tensor.zeros_like(p) for p in params]
        self.use_max_norm = use_max_norm
        self.r = r
        self.grad_clip = grad_clip 
        self.clip_value = clip_value 
    
    def forward(self, *args, **kwargs):
        return None 
    
    def step(self) -> None: 
        for i, parm in enumerate(self.params): 
            if not parm.requires_grad: 
                continue 
            if parm.grad is None: 
                continue 
            if self.grad_clip:
                grad_norm = np.linalg.norm(parm.grad.data) 
                if grad_norm > self.clip_value: 
                    parm.grad.data = parm.grad.data * (self.clip_value / grad_norm)
            self.accumulators[i].data += parm.grad ** 2 
            adjusted_lr = self.lr / (self.accumulators[i].data ** 0.5 + self.eps) 
            parm.data -= adjusted_lr * parm.grad 

            if self.use_max_norm: 
                norm = np.linalg.norm(parm.data) # L2 norm 
                if norm > self.r: 
                    parm.data = parm.data * (self.r / norm)
    
    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for param in self.params:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()
    
    def state_dict(self, prefix=""):
        state = {}
        state[f"{prefix}.lr"] = self.lr 
        state[f"{prefix}.eps"] = self.eps
        state[f"{prefix}.grad_clip"] = self.grad_clip
        state[f"{prefix}.clip_value"] = self.clip_value
        state[f"{prefix}.use_max_norm"] = self.use_max_norm
        state[f"{prefix}.r"] = self.r
        for i, acc in enumerate(self.accumulators):
            state[f"{prefix}.accumulator.{i}"] = acc.data
        return state
    


    def __repr__(self) -> str:
        return f"Adagrad(lr={self.lr}, eps={self.eps})"
    def __str__(self) -> str:
        return f"Adagrad Optimizer: lr={self.lr}, eps={self.eps}"

class RMSProp(Module): 
    """ RMSProp optimizer.""" 
    def __init__(self, params: List[Tensor], lr: float = 0.01,
     beta: float = 0.9, eps: float = 1e-8, use_max_norm: bool = False,
      r: float = 1.0,grad_clip:bool = False,clip_value: float = 1.0) -> None:
            grad = parm.grad
        self.params = params
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.squared_avg = [Tensor.zeros_like(p) for p in params]
        self.use_max_norm = use_max_norm
        self.r = r
        self.grad_clip = grad_clip
        self.clip_value = clip_value
    
    def forward(self, *args, **kwargs):
        return None
    
    def step(self) -> None: 
        for i, param in enumerate(self.params): 
            if not param.requires_grad: 
                continue 
            if param.grad is None: 
                continue 
            if self.grad_clip:
                grad_norm = np.linalg.norm(param.grad.data) 
                if grad_norm > self.clip_value: 
                    param.grad.data = param.grad.data * (self.clip_value / grad_norm)
            self.squared_avg[i].data = self.beta * self.squared_avg[i].data + (1 - self.beta) * (param.grad ** 2) 
            adjusted_lr = self.lr / (self.squared_avg[i].data ** 0.5 + self.eps) 
            param.data -= adjusted_lr * param.grad
            if self.use_max_norm: 
                norm = np.linalg.norm(param.data) # L2 norm 
                if norm > self.r: 
                    param.data = param.data * (self.r / norm)

    def state_dict(self, prefix=""):
        state = {}
        state[f"{prefix}.lr"] = self.lr 
        state[f"{prefix}.beta"] = self.beta
        state[f"{prefix}.eps"] = self.eps
        state[f"{prefix}.use_max_norm"] = self.use_max_norm
        state[f"{prefix}.r"] = self.r
        state[f"{prefix}.grad_clip"] = self.grad_clip
        state[f"{prefix}.clip_value"] = self.clip_value 
        for i, sq_avg in enumerate(self.squared_avg):
            state[f"{prefix}.squared_avg.{i}"] = sq_avg.data
        return state

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.lr}, beta={self.beta}, eps={self.eps})"
    def __str__(self) -> str:
        return f"RMSProp Optimizer: lr={self.lr}, beta={self.beta}, eps={self.eps}"

class Adam(Module):
    """
    Adam optimizer.
    link to the paper: https://arxiv.org/abs/1412.6980
    """
    def __init__(self, params: List[Tensor], lr: float = 0.001, 
    beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
     use_max_norm: bool = False, r: float = 1.0,grad_clip:bool = False,clip_value: float = 1.0) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.use_max_norm = use_max_norm
        self.r = r
        self.grad_clip = grad_clip
        self.clip_value = clip_value
        self.m = [Tensor.zeros_like(p) for p in params]
        self.v = [Tensor.zeros_like(p) for p in params]
        self.t = 0
    
    def forward(self, *args, **kwargs):
        return None 
    

    def state_dict(self, prefix=""):
        state = {}
        state[f"{prefix}.lr"] = self.lr 
        state[f"{prefix}.beta1"] = self.beta1
        state[f"{prefix}.beta2"] = self.beta2
        state[f"{prefix}.eps"] = self.eps
        state[f"{prefix}.t"] = self.t
        state[f"{prefix}.grad_clip"] = self.grad_clip
        state[f"{prefix}.clip_value"] = self.clip_value
        for i, (m_i, v_i) in enumerate(zip(self.m, self.v)):
            state[f"{prefix}.m.{i}"] = m_i.data
            state[f"{prefix}.v.{i}"] = v_i.data
        return state
    
    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for param in self.params:
            if hasattr(param, 'zero_grad'):
                param.zero_grad() 
    
    def __str__(self) -> str:
        return f"Adam Optimizer: lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}" 
    
    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})" 

    def step(self) -> None:
        """Apply one optimization step to all parameters with bias correction."""
        self.t += 1
        for i, param in enumerate(self.params):
            if param.requires_grad is False:
                continue 
            if param.grad is None:
                continue
            if self.grad_clip:
                grad_norm = np.linalg.norm(param.grad.data) 
                if grad_norm > self.clip_value: 
                    param.grad.data = param.grad.data * (self.clip_value / grad_norm)
            self.m[i].data = self.beta1 * self.m[i].data + (1 - self.beta1) * param.grad
            self.v[i].data = self.beta2 * self.v[i].data + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[i].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[i].data / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)
            if self.use_max_norm: 
                norm = np.linalg.norm(param.data) # L2 norm 
                if norm > self.r: 
                    param.data = param.data * (self.r / norm)

    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for param in self.params:
            if hasattr(param, 'zero_grad'):
                param.zero_grad()
    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})"
    def __str__(self) -> str:
        return f"Adam Optimizer: lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}"