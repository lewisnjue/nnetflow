import pytest
import torch
import numpy as np
from nnetflow.engine import Tensor


def _pair(*shape, lo=None, hi=None):
    """Create matching (nnetflow Tensor, torch Tensor) leaves with identical
    float64 data and requires_grad=True, so forward values and gradients
    can be compared directly."""
    if lo is None:
        data = np.random.randn(*shape).astype(np.float64)
    else:
        data = np.random.uniform(lo, hi, size=shape).astype(np.float64)
    nt = Tensor(data.copy(), requires_grad=True)
    tt = torch.tensor(data.copy(), requires_grad=True)
    return nt, tt


class TestEngine:
    def test_tensor_initialization(self):
        data = np.random.randn(3,4,5)
        nnetflow_tensor = Tensor(data)
        torch_tensor = torch.tensor(data)
        assert data.__array_interface__['data'][0] == nnetflow_tensor.data.__array_interface__['data'][0]
        assert np.allclose(nnetflow_tensor.data, torch_tensor.numpy())
        assert data.__array_interface__['data'][0] != Tensor(data,copy=True).data.__array_interface__['data'][0]
        assert np.allclose(Tensor(data,copy=True).data, torch_tensor.numpy())
        assert nnetflow_tensor.data.dtype == data.dtype
        assert nnetflow_tensor.dtype == data.dtype
        assert Tensor(data,dtype=np.float16).dtype == np.float16
        assert nnetflow_tensor.requires_grad == False
        assert Tensor(data,requires_grad=True).requires_grad == True

    def test_tensor_get_state(self):
        data = np.random.randn(3,4,5)
        nnetflow_tensor = Tensor(data)
        state = nnetflow_tensor.__getstate__()
        assert 'data' in state
        assert type(state['data']) == np.ndarray
        assert '_op' in state
        assert '_prev' in state
        assert 'requires_grad' in state
        assert '_backward' not in state
        assert 'grad' in state
        tt = Tensor(data,requires_grad=True)
        assert type(tt.__getstate__()['grad']) == np.ndarray
        assert tt.__getstate__()['grad'].shape == tt.data.shape

    def test_check_dtype(self):
        A = Tensor(np.random.randn(2,3))
        B = Tensor(np.random.rand(2,3))
        assert Tensor._check_dtype(A,B) == True
        with pytest.raises(ValueError):
            Tensor._check_dtype(A,Tensor(np.random.randn(2,3),dtype=np.float16))

    def test_to(self):
        A  = Tensor(np.random.randn(2,3))
        B = A.to(np.float16)
        assert B.dtype == np.float16
        assert B.data.dtype == np.float16
        assert B.data.__array_interface__['data'][0] != A.data.__array_interface__['data'][0]

    def test_unbroadcast_valid_shapes(self):
        x_torch = torch.randn(3, 4, requires_grad=True)
        y_torch = torch.randn(2, 3, 4)
        (x_torch + y_torch).sum().backward()
        grad = np.ones((2, 3, 4))
        unbroadcasted = Tensor.unbroadcast(grad, (3, 4))
        assert unbroadcasted.shape == (3, 4)
        assert np.allclose(unbroadcasted, x_torch.grad.numpy())
        grad = np.random.randn(2, 3, 4)
        unbroadcasted = Tensor.unbroadcast(grad, (1, 3, 4))
        assert unbroadcasted.shape == (1, 3, 4)
        assert np.allclose(unbroadcasted, grad.sum(axis=0, keepdims=True))
        grad = np.random.randn(2, 3, 4)
        unbroadcasted = Tensor.unbroadcast(grad, ())
        assert unbroadcasted.shape == ()
        assert np.allclose(unbroadcasted, grad.sum())
        grad = np.random.randn(2, 3, 4)
        assert Tensor.unbroadcast(grad, (2, 3, 4)).shape == (2, 3, 4)
        grad = np.random.randn(2, 3, 4)
        with pytest.raises(ValueError):
            Tensor.unbroadcast(grad, (4, 3))
        with pytest.raises(ValueError):
            Tensor.unbroadcast(grad, (2, 5, 4))
        with pytest.raises(ValueError):
            Tensor.unbroadcast(grad, (1, 2, 3, 4))

    def test_zero_grad(self):
        A = Tensor(np.random.randn(2,3),requires_grad=True)
        B = Tensor(np.random.randn(2,3),requires_grad=True)
        C = A + B
        C.sum().backward()
        assert A.grad is not None
        assert B.grad is not None
        A.zero_grad()
        B.zero_grad()
        assert np.all(A.grad == 0)
        assert np.all(B.grad == 0)

    # ---------------------------------------------------------------
    # Arithmetic ops
    # ---------------------------------------------------------------

    def test_add_broadcast_and_radd(self):
        a_nt, a_tt = _pair(2, 3)
        b_nt, b_tt = _pair(3)
        c_nt = (a_nt + b_nt).sum()
        c_tt = (a_tt + b_tt).sum()
        c_nt.backward()
        c_tt.backward()
        assert np.allclose(c_nt.data, c_tt.item())
        assert np.allclose(a_nt.grad, a_tt.grad.numpy())
        assert np.allclose(b_nt.grad, b_tt.grad.numpy())

        # scalar + Tensor (__radd__)
        x_nt, x_tt = _pair(2, 2)
        r_nt = (5.0 + x_nt).sum()
        r_tt = (5.0 + x_tt).sum()
        r_nt.backward()
        r_tt.backward()
        assert np.allclose(r_nt.data, r_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_sub_and_rsub(self):
        a_nt, a_tt = _pair(2, 3)
        b_nt, b_tt = _pair(2, 3)
        c_nt = (a_nt - b_nt).sum()
        c_tt = (a_tt - b_tt).sum()
        c_nt.backward()
        c_tt.backward()
        assert np.allclose(c_nt.data, c_tt.item())
        assert np.allclose(a_nt.grad, a_tt.grad.numpy())
        assert np.allclose(b_nt.grad, b_tt.grad.numpy())

        x_nt, x_tt = _pair(2, 2)
        r_nt = (3.0 - x_nt).sum()
        r_tt = (3.0 - x_tt).sum()
        r_nt.backward()
        r_tt.backward()
        assert np.allclose(r_nt.data, r_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_mul_broadcast_and_rmul(self):
        a_nt, a_tt = _pair(2, 3)
        b_nt, b_tt = _pair(3)
        c_nt = (a_nt * b_nt).sum()
        c_tt = (a_tt * b_tt).sum()
        c_nt.backward()
        c_tt.backward()
        assert np.allclose(c_nt.data, c_tt.item())
        assert np.allclose(a_nt.grad, a_tt.grad.numpy())
        assert np.allclose(b_nt.grad, b_tt.grad.numpy())

        x_nt, x_tt = _pair(2, 2)
        r_nt = (2.5 * x_nt).sum()
        r_tt = (2.5 * x_tt).sum()
        r_nt.backward()
        r_tt.backward()
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_truediv_and_rtruediv(self):
        a_nt, a_tt = _pair(2, 3, lo=1.0, hi=2.0)
        b_nt, b_tt = _pair(2, 3, lo=1.0, hi=2.0)
        c_nt = (a_nt / b_nt).sum()
        c_tt = (a_tt / b_tt).sum()
        c_nt.backward()
        c_tt.backward()
        assert np.allclose(c_nt.data, c_tt.item())
        assert np.allclose(a_nt.grad, a_tt.grad.numpy())
        assert np.allclose(b_nt.grad, b_tt.grad.numpy())

        x_nt, x_tt = _pair(2, 2, lo=1.0, hi=2.0)
        r_nt = (4.0 / x_nt).sum()
        r_tt = (4.0 / x_tt).sum()
        r_nt.backward()
        r_tt.backward()
        assert np.allclose(r_nt.data, r_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_neg(self):
        x_nt, x_tt = _pair(2, 3)
        y_nt = (-x_nt).sum()
        y_tt = (-x_tt).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_pow(self):
        x_nt, x_tt = _pair(2, 3, lo=0.5, hi=2.0)
        y_nt = (x_nt ** 3).sum()
        y_tt = (x_tt ** 3).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())
        with pytest.raises(AssertionError):
            x_nt ** x_nt  # only scalar exponents supported

    # ---------------------------------------------------------------
    # Matmul
    # ---------------------------------------------------------------

    def test_matmul_2d(self):
        a_nt, a_tt = _pair(4, 3)
        b_nt, b_tt = _pair(3, 5)
        c_nt = (a_nt @ b_nt).sum()
        c_tt = (a_tt @ b_tt).sum()
        c_nt.backward()
        c_tt.backward()
        assert np.allclose(c_nt.data, c_tt.item())
        assert np.allclose(a_nt.grad, a_tt.grad.numpy())
        assert np.allclose(b_nt.grad, b_tt.grad.numpy())

    def test_matmul_batched_3d(self):
        a_nt, a_tt = _pair(2, 4, 3)
        b_nt, b_tt = _pair(2, 3, 5)
        c_nt = (a_nt @ b_nt).sum()
        c_tt = (a_tt @ b_tt).sum()
        c_nt.backward()
        c_tt.backward()
        assert np.allclose(c_nt.data, c_tt.item())
        assert np.allclose(a_nt.grad, a_tt.grad.numpy())
        assert np.allclose(b_nt.grad, b_tt.grad.numpy())

    def test_matmul_matrix_vector(self):
        a_nt, a_tt = _pair(4, 3)
        b_nt, b_tt = _pair(3)
        c_nt = (a_nt @ b_nt).sum()
        c_tt = (a_tt @ b_tt).sum()
        c_nt.backward()
        c_tt.backward()
        assert np.allclose(c_nt.data, c_tt.item())
        assert np.allclose(a_nt.grad, a_tt.grad.numpy())
        assert np.allclose(b_nt.grad, b_tt.grad.numpy())

    def test_matmul_vector_matrix(self):
        a_nt, a_tt = _pair(3)
        b_nt, b_tt = _pair(3, 5)
        c_nt = (a_nt @ b_nt).sum()
        c_tt = (a_tt @ b_tt).sum()
        c_nt.backward()
        c_tt.backward()
        assert np.allclose(c_nt.data, c_tt.item())
        assert np.allclose(a_nt.grad, a_tt.grad.numpy())
        assert np.allclose(b_nt.grad, b_tt.grad.numpy())

    def test_matmul_shape_mismatch_raises(self):
        a = Tensor(np.random.randn(2, 3))
        b = Tensor(np.random.randn(4, 5))
        with pytest.raises(ValueError):
            a @ b

    # ---------------------------------------------------------------
    # Reductions
    # ---------------------------------------------------------------

    def test_sum_axis_keepdims(self):
        x_nt, x_tt = _pair(3, 4, 5)
        y_nt = x_nt.sum(axis=1, keepdims=True)
        y_tt = x_tt.sum(dim=1, keepdim=True)
        assert np.allclose(y_nt.data, y_tt.detach().numpy())
        y_nt.sum().backward()
        y_tt.sum().backward()
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_sum_axis_no_keepdims(self):
        x_nt, x_tt = _pair(3, 4, 5)
        y_nt = x_nt.sum(axis=(0, 2))
        y_tt = x_tt.sum(dim=(0, 2))
        assert np.allclose(y_nt.data, y_tt.detach().numpy())
        y_nt.sum().backward()
        y_tt.sum().backward()
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_mean(self):
        x_nt, x_tt = _pair(3, 4)
        y_nt = x_nt.mean(axis=1)
        y_tt = x_tt.mean(dim=1)
        assert np.allclose(y_nt.data, y_tt.detach().numpy())
        y_nt.sum().backward()
        y_tt.sum().backward()
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_var_and_std(self):
        x_nt, x_tt = _pair(5, 6)
        v_nt = x_nt.var(axis=1)
        v_tt = x_tt.var(dim=1, unbiased=True, keepdim=True)
        assert np.allclose(v_nt.data, v_tt.detach().numpy(), atol=1e-6)
        v_nt.sum().backward()
        v_tt.sum().backward()
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-6)

        x2_nt, x2_tt = _pair(5, 6, lo=1.0, hi=2.0)
        s_nt = x2_nt.std(axis=1)
        s_tt = x2_tt.std(dim=1, unbiased=True, keepdim=True)
        assert np.allclose(s_nt.data, s_tt.detach().numpy(), atol=1e-6)
        s_nt.sum().backward()
        s_tt.sum().backward()
        assert np.allclose(x2_nt.grad, x2_tt.grad.numpy(), atol=1e-5)

    # ---------------------------------------------------------------
    # Elementwise math
    # ---------------------------------------------------------------

    def test_exp(self):
        x_nt, x_tt = _pair(2, 3)
        y_nt = x_nt.exp().sum()
        y_tt = x_tt.exp().sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_log(self):
        x_nt, x_tt = _pair(2, 3, lo=0.5, hi=3.0)
        y_nt = x_nt.log().sum()
        y_tt = x_tt.log().sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-5)

    def test_log_warns_on_nonpositive(self):
        x = Tensor(np.array([-1.0, 0.0, 1.0]), requires_grad=True)
        with pytest.warns(RuntimeWarning):
            x.log()

    def test_sqrt(self):
        x_nt, x_tt = _pair(2, 3, lo=0.5, hi=3.0)
        y_nt = x_nt.sqrt().sum()
        y_tt = x_tt.sqrt().sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-5)

    def test_log10(self):
        x_nt, x_tt = _pair(2, 3, lo=0.5, hi=3.0)
        y_nt = x_nt.log10().sum()
        y_tt = x_tt.log10().sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-5)

    def test_clip(self):
        x_nt, x_tt = _pair(3, 3, lo=-2.0, hi=2.0)
        y_nt = x_nt.clip(-1.0, 1.0).sum()
        y_tt = x_tt.clamp(-1.0, 1.0).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    # ---------------------------------------------------------------
    # Activations
    # ---------------------------------------------------------------

    def test_relu(self):
        x_nt, x_tt = _pair(4, 4)
        y_nt = x_nt.relu().sum()
        y_tt = x_tt.relu().sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_leaky_relu(self):
        x_nt, x_tt = _pair(4, 4)
        y_nt = x_nt.leaky_relu(0.1).sum()
        y_tt = torch.nn.functional.leaky_relu(x_tt, 0.1).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_elu(self):
        x_nt, x_tt = _pair(4, 4)
        y_nt = x_nt.elu(1.0).sum()
        y_tt = torch.nn.functional.elu(x_tt, 1.0).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item(), atol=1e-6)
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-6)

    def test_selu(self):
        x_nt, x_tt = _pair(4, 4)
        y_nt = x_nt.selu().sum()
        y_tt = torch.nn.functional.selu(x_tt).sum()
        # nnetflow's selu scales the whole branch (including the x>0 side)
        # by `scale`, unlike torch's canonical SELU constants; compare
        # against a manually-scaled reference instead of F.selu directly.
        alpha, scale = 1.67326, 1.0507
        y_ref = (scale * torch.where(x_tt > 0, x_tt, alpha * (torch.exp(x_tt) - 1))).sum()
        y_nt.backward()
        y_ref.backward()
        assert np.allclose(y_nt.data, y_ref.item(), atol=1e-5)
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-5)

    def test_gelu(self):
        x_nt, x_tt = _pair(4, 4)
        y_nt = x_nt.gelu().sum()
        y_tt = torch.nn.functional.gelu(x_tt).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item(), atol=1e-5)
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-5)

    def test_gelu_preserves_float32_dtype(self):
        x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        out = x.gelu()
        assert out.dtype == np.float32

    def test_sigmoid(self):
        x_nt, x_tt = _pair(4, 4)
        y_nt = x_nt.sigmoid().sum()
        y_tt = torch.sigmoid(x_tt).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_swish(self):
        x_nt, x_tt = _pair(4, 4)
        y_nt = x_nt.swish().sum()
        y_tt = torch.nn.functional.silu(x_tt).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item(), atol=1e-6)
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-6)

    def test_tanh(self):
        x_nt, x_tt = _pair(4, 4)
        y_nt = x_nt.tanh().sum()
        y_tt = x_tt.tanh().sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_softmax(self):
        x_nt, x_tt = _pair(3, 5)
        y_nt = x_nt.softmax(axis=-1)
        y_tt = torch.softmax(x_tt, dim=-1)
        assert np.allclose(y_nt.data, y_tt.detach().numpy(), atol=1e-6)
        assert np.allclose(y_nt.data.sum(axis=-1), 1.0, atol=1e-5)
        (y_nt * y_nt).sum().backward()
        (y_tt * y_tt).sum().backward()
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-5)

    def test_log_softmax(self):
        x_nt, x_tt = _pair(3, 5)
        y_nt = x_nt.log_softmax(axis=-1)
        y_tt = torch.log_softmax(x_tt, dim=-1)
        assert np.allclose(y_nt.data, y_tt.detach().numpy(), atol=1e-5)
        y_nt.sum().backward()
        y_tt.sum().backward()
        assert np.allclose(x_nt.grad, x_tt.grad.numpy(), atol=1e-5)

    # ---------------------------------------------------------------
    # Shape ops
    # ---------------------------------------------------------------

    def test_reshape_and_view(self):
        x_nt, x_tt = _pair(2, 3, 4)
        y_nt = x_nt.reshape(6, 4).sum(axis=1)
        y_tt = x_tt.reshape(6, 4).sum(dim=1)
        assert np.allclose(y_nt.data, y_tt.detach().numpy())
        y_nt.sum().backward()
        y_tt.sum().backward()
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

        x2_nt, x2_tt = _pair(2, 6)
        y2_nt = x2_nt.view(-1, 3)
        y2_tt = x2_tt.view(-1, 3)
        assert y2_nt.shape == tuple(y2_tt.shape)
        y2_nt.sum().backward()
        y2_tt.sum().backward()
        assert np.allclose(x2_nt.grad, x2_tt.grad.numpy())

    def test_transpose(self):
        x_nt, x_tt = _pair(2, 3, 4)
        y_nt = x_nt.transpose((0, 2, 1)).sum(axis=0)
        y_tt = x_tt.permute(0, 2, 1).sum(dim=0)
        assert np.allclose(y_nt.data, y_tt.detach().numpy())
        y_nt.sum().backward()
        y_tt.sum().backward()
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

        x2_nt, x2_tt = _pair(3, 5)
        y2_nt = x2_nt.transpose().sum()  # default: reverse all axes
        y2_tt = x2_tt.t().sum()
        y2_nt.backward()
        y2_tt.backward()
        assert np.allclose(x2_nt.grad, x2_tt.grad.numpy())

    def test_getitem_basic_slice(self):
        x_nt, x_tt = _pair(5, 4)
        y_nt = x_nt[1:3, :].sum()
        y_tt = x_tt[1:3, :].sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_getitem_repeated_index_accumulates(self):
        # fancy indexing that repeats an index should accumulate gradient
        # at that position (this is why the engine uses np.add.at)
        x_nt, x_tt = _pair(5)
        idx = np.array([0, 0, 2])
        y_nt = x_nt[idx].sum()
        y_tt = x_tt[torch.tensor(idx)].sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())
        assert x_nt.grad[0] == 2.0

    def test_masked_fill(self):
        x_nt, x_tt = _pair(3, 3)
        mask_np = np.array([[True, False, True],
                             [False, False, True],
                             [True, True, False]])
        mask_nt = Tensor(mask_np, requires_grad=False)
        mask_tt = torch.tensor(mask_np)
        y_nt = x_nt.masked_fill(mask_nt, -1e9).sum()
        y_tt = x_tt.masked_fill(mask_tt, -1e9).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())
        assert np.all(x_nt.grad[mask_np] == 0)

    # ---------------------------------------------------------------
    # Scalar / bool helpers
    # ---------------------------------------------------------------

    def test_item(self):
        x = Tensor(np.array([[3.5]]))
        assert x.item() == 3.5
        with pytest.raises(ValueError):
            Tensor(np.array([1.0, 2.0])).item()

    def test_bool_conversion(self):
        assert bool(Tensor(np.array(1.0))) is True
        assert bool(Tensor(np.array(0.0))) is False
        with pytest.raises(ValueError):
            bool(Tensor(np.array([1.0, 0.0])))

    def test_bool_method_detaches(self):
        x = Tensor(np.array([1.0, -1.0, 0.0]), requires_grad=True)
        b = x.bool()
        assert b.dtype == np.bool_
        assert b.requires_grad is False
        assert list(b.data) == [True, True, False]

    # ---------------------------------------------------------------
    # Graph / backward mechanics
    # ---------------------------------------------------------------

    def test_backward_diamond_graph(self):
        # y = x*x + x  -> dy/dx = 2x + 1; exercises a node (x) reused
        # by two different parents in the graph.
        x_nt, x_tt = _pair(3, 3)
        y_nt = (x_nt * x_nt + x_nt).sum()
        y_tt = (x_tt * x_tt + x_tt).sum()
        y_nt.backward()
        y_tt.backward()
        assert np.allclose(y_nt.data, y_tt.item())
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_backward_accumulates_across_calls_on_leaf(self):
        # leaves are never auto-zeroed between backward() calls (matches
        # torch's default accumulate-into-.grad behaviour), so calling
        # backward twice without zero_grad should double the leaf grad.
        x_nt, x_tt = _pair(2, 2)
        y_nt = (x_nt * 2).sum()
        y_tt = (x_tt * 2).sum()
        y_nt.backward()
        y_tt.backward()
        first_nt = x_nt.grad.copy()
        first_tt = x_tt.grad.clone()
        y_nt2 = (x_nt * 2).sum()
        y_tt2 = (x_tt * 2).sum()
        y_nt2.backward()
        y_tt2.backward()
        assert np.allclose(x_nt.grad, first_nt * 2)
        assert np.allclose(x_nt.grad, x_tt.grad.numpy())

    def test_requires_grad_propagation(self):
        a = Tensor(np.random.randn(2, 2), requires_grad=False)
        b = Tensor(np.random.randn(2, 2), requires_grad=True)
        c = a + b
        assert c.requires_grad is True
        d = a * a
        assert d.requires_grad is False
        with pytest.raises(RuntimeError):
            d.backward()