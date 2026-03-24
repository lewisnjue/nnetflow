"""Tests for LSTM layer against PyTorch reference implementation."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from nnetflow.engine import Tensor
from nnetflow.layers import LSTM


RTOL = 1e-5
ATOL = 1e-6


def _set_same_weights(ours: LSTM, torch_lstm: nn.LSTM, F: int, H: int) -> None:
    """Initialize our LSTM and PyTorch LSTM with exactly the same weights."""
    np.random.seed(123)
    torch.manual_seed(123)

    # Gate weights in our layout
    W_xi = np.random.randn(F, H).astype(np.float64)
    W_xf = np.random.randn(F, H).astype(np.float64)
    W_xg = np.random.randn(F, H).astype(np.float64)
    W_xo = np.random.randn(F, H).astype(np.float64)

    W_hi = np.random.randn(H, H).astype(np.float64)
    W_hf = np.random.randn(H, H).astype(np.float64)
    W_hg = np.random.randn(H, H).astype(np.float64)
    W_ho = np.random.randn(H, H).astype(np.float64)

    b_i = np.random.randn(1, H).astype(np.float64)
    b_f = np.random.randn(1, H).astype(np.float64)
    b_g = np.random.randn(1, H).astype(np.float64)
    b_o = np.random.randn(1, H).astype(np.float64)

    # Set our weights (our layout is (in, hidden) and (hidden, hidden))
    ours.w_xi.data = W_xi.copy()
    ours.w_xf.data = W_xf.copy()
    ours.w_xg.data = W_xg.copy()
    ours.w_xo.data = W_xo.copy()

    ours.w_hi.data = W_hi.copy()
    ours.w_hf.data = W_hf.copy()
    ours.w_hg.data = W_hg.copy()
    ours.w_ho.data = W_ho.copy()

    ours.b_i.data = b_i.copy()
    ours.b_f.data = b_f.copy()
    ours.b_g.data = b_g.copy()
    ours.b_o.data = b_o.copy()

    # PyTorch expects gate order: i, f, g, o and weight layout (hidden, in/hidden)
    w_ih = np.concatenate([W_xi.T, W_xf.T, W_xg.T, W_xo.T], axis=0)
    w_hh = np.concatenate([W_hi.T, W_hf.T, W_hg.T, W_ho.T], axis=0)
    b_ih = np.concatenate([b_i.reshape(-1), b_f.reshape(-1), b_g.reshape(-1), b_o.reshape(-1)], axis=0)
    b_hh = np.zeros((4 * H,), dtype=np.float64)

    with torch.no_grad():
        torch_lstm.weight_ih_l0.copy_(torch.tensor(w_ih, dtype=torch.float64))
        torch_lstm.weight_hh_l0.copy_(torch.tensor(w_hh, dtype=torch.float64))
        torch_lstm.bias_ih_l0.copy_(torch.tensor(b_ih, dtype=torch.float64))
        torch_lstm.bias_hh_l0.copy_(torch.tensor(b_hh, dtype=torch.float64))


@pytest.mark.parametrize("B,T,F,H", [(2, 4, 3, 5), (3, 6, 4, 2)])
def test_lstm_forward_last_matches_pytorch(B: int, T: int, F: int, H: int) -> None:
    """Compare final hidden/cell states with PyTorch using identical weights."""
    np.random.seed(42)
    torch.manual_seed(42)

    np_x = np.random.randn(B, T, F).astype(np.float64)
    x = Tensor(np_x, requires_grad=False, dtype=np.float64)

    ours = LSTM(return_sequence=False, hidden_size=H, dtype=np.float64)
    # Trigger lazy initialization once so parameters exist
    _ = ours(x)

    torch_lstm = nn.LSTM(input_size=F, hidden_size=H, batch_first=True, bias=True).double()
    _set_same_weights(ours, torch_lstm, F=F, H=H)

    ours_h, ours_c = ours(x)

    torch_x = torch.tensor(np_x, dtype=torch.float64)
    torch_out_seq, (torch_h_n, torch_c_n) = torch_lstm(torch_x)
    torch_last_h = torch_h_n[-1]  # (B, H)
    torch_last_c = torch_c_n[-1]  # (B, H)

    np.testing.assert_allclose(ours_h.data, torch_last_h.detach().numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ours_c.data, torch_last_c.detach().numpy(), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("B,T,F,H", [(2, 5, 4, 3)])
def test_lstm_forward_sequence_matches_pytorch(B: int, T: int, F: int, H: int) -> None:
    """Compare full hidden-state sequence and final cell state to PyTorch."""
    np.random.seed(7)
    torch.manual_seed(7)

    np_x = np.random.randn(B, T, F).astype(np.float64)
    x = Tensor(np_x, requires_grad=False, dtype=np.float64)

    ours = LSTM(return_sequence=True, hidden_size=H, dtype=np.float64)
    _ = ours(x)  # lazy init

    torch_lstm = nn.LSTM(input_size=F, hidden_size=H, batch_first=True, bias=True).double()
    _set_same_weights(ours, torch_lstm, F=F, H=H)

    ours_seq, ours_c = ours(x)

    torch_x = torch.tensor(np_x, dtype=torch.float64)
    torch_seq, (torch_h_n, torch_c_n) = torch_lstm(torch_x)

    np.testing.assert_allclose(ours_seq.data, torch_seq.detach().numpy(), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(ours_c.data, torch_c_n[-1].detach().numpy(), rtol=RTOL, atol=ATOL)

