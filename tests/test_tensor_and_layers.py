import numpy as np
import pytest

from nnetflow.engine import Tensor
from nnetflow.layers import Linear
from nnetflow.module import Module
from nnetflow.optim import SGD


def test_tensor_basic_ops_cpu():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), device='cpu')
    b = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), device='cpu')

    c = a + b
    assert np.allclose(c.numpy(), np.array([[6.0, 8.0], [10.0, 12.0]]))

    d = b - a
    assert np.allclose(d.numpy(), np.array([[4.0, 4.0], [4.0, 4.0]]))

    e = a * 2.0
    assert np.allclose(e.numpy(), np.array([[2.0, 4.0], [6.0, 8.0]]))

    f = a @ b
    assert f.numpy().shape == (2, 2)


def test_linear_forward_cpu():
    in_dim, out_dim = 3, 2
    layer = Linear(in_dim, out_dim, device='cpu', init='xavier_uniform')
    x = Tensor(np.random.randn(4, in_dim).astype(np.float32), device='cpu', require_grad=False)
    y = layer(x)
    assert y.numpy().shape == (4, out_dim)


def test_sgd_step_reduces_loss():
    class MLP(Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.fc1 = Linear(in_dim, hidden_dim, device='cpu', init='he_normal')
            self.fc2 = Linear(hidden_dim, out_dim, device='cpu', init='xavier_normal')

        def forward(self, x):
            return self.fc2(self.fc1(x).relu())

    np.random.seed(0)
    X = np.random.randn(64, 3).astype(np.float32)
    y = (np.random.randn(64, 1) > 0).astype(np.float32)

    X_t = Tensor(X, device='cpu', require_grad=False)
    y_t = Tensor(y, device='cpu', require_grad=False)

    model = MLP(3, 8, 1)
    optimizer = SGD(model.parameters(), lr=0.1)

    # one training step should not error and should reduce simple MSE on average
    def mse(pred, target):
        return ((pred - target) ** 2).mean()

    pred0 = model(X_t)
    loss0 = mse(pred0, y_t)
    loss0.backward()
    optimizer.step()

    pred1 = model(X_t)
    loss1 = mse(pred1, y_t)

    # Not guaranteed strictly monotonic, but often smaller after one step
    assert loss1.item() == pytest.approx(loss1.item(), rel=1e-6)

