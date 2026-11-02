"""PyTorch implementation of a 2D Convolutional LSTM (ConvLSTM) Cell."""

from typing import Tuple
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """A single 2D Convolutional Long Short-Term Memory cell.
    
    Equations:
        i_t = sigmoid(W_xi * x_t + W_hi * h_{t-1} + b_i)
        f_t = sigmoid(W_xf * x_t + W_hf * h_{t-1} + b_f)
        o_t = sigmoid(W_xo * x_t + W_ho * h_{t-1} + b_o)
        g_t = tanh(W_xg * x_t + W_hg * h_{t-1} + b_g)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # 4 gates: input, forget, output, candidate
        self.conv = nn.Conv2d(
            in_channels=self.in_channels + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single time step.
        
        Args:
            x: Input tensor of shape (Batch, in_channels, Height, Width)
            state: Tuple of (h, c) each of shape (Batch, hidden_dim, Height, Width)
            
        Returns:
            Tuple of updated (h_next, c_next)
        """
        h_prev, c_prev = state

        # Concatenate along channel dimension
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        # Split into 4 gates: i, f, o, g
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes hidden state h and cell state c with zeros."""
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )

