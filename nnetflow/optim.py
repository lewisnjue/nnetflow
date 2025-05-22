from .engine import Tensor as tensor 

"""add optimizers here"""

class SGD:
    def __init__(self,params,lr=0.01):
        self.params = params
        self.lr = lr
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad
    def zero_grad(self):
        for p in self.params:
            p.zero_grad() 
        



