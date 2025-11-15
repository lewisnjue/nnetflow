"""Tests for RNN layer handling sequential data."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from nnetflow.engine import Tensor
from nnetflow.layers import RNN


RTOL = 1e-5
ATOL = 1e-6


class TestRNNLayer:
    """Test suite for the simple RNN layer."""

    def test_rnn_forward_shapes_return_last(self):
        """RNN returns correct shape when return_sequence=False."""
        B, T, F, H = 4, 7, 5, 3
        x = Tensor.randn(B, T, F, requires_grad=False)

        rnn = RNN(n_neurons=H, return_sequence=False)
        out = rnn(x)

        assert out.shape == (B, H)
        assert out.requires_grad

    def test_rnn_forward_shapes_return_sequence(self):
        """RNN returns correct shape when return_sequence=True."""
        B, T, F, H = 2, 5, 4, 6
        x = Tensor.randn(B, T, F, requires_grad=False)

        rnn = RNN(n_neurons=H, return_sequence=True)
        out = rnn(x)

        assert out.shape == (B, T, H)
        assert out.requires_grad

    @pytest.mark.parametrize("B, T, F, H", [
        (1, 3, 4, 2),
        (2, 5, 3, 4),
        (4, 7, 5, 3),
    ])
    def test_rnn_forward_matches_pytorch_last(self, B, T, F, H):
        """Compare RNN forward pass (last state) with PyTorch nn.RNN."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Random input
        np_x = np.random.randn(B, T, F).astype(np.float64)
        x = Tensor(np_x, requires_grad=True)

        # Random weights in a canonical layout
        W_ih = np.random.randn(H, F).astype(np.float64)
        W_hh = np.random.randn(H, H).astype(np.float64)
        b_h = np.zeros((H,), dtype=np.float64)

        # Our RNN expects Wxh shape (F, H) and Whh shape (H, H)
        rnn = RNN(n_neurons=H, return_sequence=False)
        # Trigger lazy init to set input_size
        _ = rnn(x)
        rnn.Wxh.data = W_ih.T.copy()
        rnn.Whh.data = W_hh.T.copy()
        rnn.bh.data = b_h.reshape(1, H).copy()

        # Re-run forward with set weights
        y = rnn(x)

        # PyTorch reference RNN
        torch_rnn = nn.RNN(
            input_size=F,
            hidden_size=H,
            nonlinearity="tanh",
            batch_first=True,
            bias=True,
        )
        torch_rnn = torch_rnn.double()
        with torch.no_grad():
            torch_rnn.weight_ih_l0.copy_(torch.tensor(W_ih, dtype=torch.float64))
            torch_rnn.weight_hh_l0.copy_(torch.tensor(W_hh, dtype=torch.float64))
            torch_rnn.bias_ih_l0.zero_()
            torch_rnn.bias_hh_l0.zero_()

        torch_x = torch.tensor(np_x, dtype=torch.float64, requires_grad=True)
        out_seq, h_n = torch_rnn(torch_x)
        torch_last = h_n[-1]  # (B, H)

        np.testing.assert_allclose(y.data, torch_last.detach().numpy(), rtol=RTOL, atol=ATOL)

    def test_rnn_backward_parameters(self):
        """Gradients for RNN parameters are non-None and have correct shapes."""
        np.random.seed(0)
        B, T, F, H = 3, 4, 5, 6
        x = Tensor.randn(B, T, F, requires_grad=False)

        rnn = RNN(n_neurons=H, return_sequence=False)
        y = rnn(x)
        loss = (y ** 2).mean()
        loss.backward()

        params = rnn.parameters()
        assert len(params) == 3
        Wxh, Whh, bh = params

        assert Wxh.grad is not None
        assert Whh.grad is not None
        assert bh.grad is not None

        assert Wxh.grad.shape == Wxh.data.shape
        assert Whh.grad.shape == Whh.data.shape
        assert bh.grad.shape == bh.data.shape

    def test_rnn_input_feature_mismatch_raises(self):
        """RNN raises assertion if feature dimension changes after init."""
        B, T, F, H = 2, 3, 4, 5
        x1 = Tensor.randn(B, T, F, requires_grad=False)
        x2 = Tensor.randn(B, T, F + 1, requires_grad=False)

        rnn = RNN(n_neurons=H, return_sequence=False)
        _ = rnn(x1)

        with pytest.raises(AssertionError, match="RNN expected input feature size"):
            rnn(x2)
