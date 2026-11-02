"""Multi-Layer Spatiotemporal ConvLSTM Forecasting Network."""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn

from models.convlstm_cell import ConvLSTMCell


class ConvLSTMForecaster(nn.Module):
    """Stacked Multi-Layer Convolutional LSTM for Wildfire Progression Forecasting.
    
    Architecture:
        Input: (Batch, Seq_Len=5, In_Channels=6, H=128, W=128)
        Stacked ConvLSTM layers (e.g. 6 -> 32 -> 64)
        Autoregressive / multi-step decoder generating (Batch, Pred_Len=3, Out_Channels=1, H, W)
        Separate Fire-Presence Probability Head & Fire-Intensity Head.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_dims: List[int] = [32, 64],
        out_channels: int = 1,
        kernel_size: int = 3,
        pred_len: int = 3,
        dropout: float = 0.15,
        use_intensity_head: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.out_channels = out_channels
        self.pred_len = pred_len
        self.use_intensity_head = use_intensity_head
        self.num_layers = len(hidden_dims)

        # Encoder ConvLSTM layers
        self.encoder_cells = nn.ModuleList()
        current_dim = in_channels
        for h_dim in hidden_dims:
            self.encoder_cells.append(
                ConvLSTMCell(in_channels=current_dim, hidden_dim=h_dim, kernel_size=kernel_size)
            )
            current_dim = h_dim

        # Decoder ConvLSTM cell (takes previous step's prediction features)
        self.decoder_cell = ConvLSTMCell(
            in_channels=hidden_dims[-1],
            hidden_dim=hidden_dims[-1],
            kernel_size=kernel_size
        )

        self.dropout = nn.Dropout2d(p=dropout)

        # Presence Head: per-pixel fire probability
        self.presence_head = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Optional Intensity Head: continuous fire intensity / FRP estimation
        if self.use_intensity_head:
            self.intensity_head = nn.Sequential(
                nn.Conv2d(hidden_dims[-1], 32, kernel_size=3, padding=1),
                nn.GroupNorm(4, 32),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(32, out_channels, kernel_size=1),
                nn.ReLU()  # Intensity is non-negative
            )

    def forward(
        self,
        x: torch.Tensor,
        pred_steps: Optional[int] = None,
        return_intensity: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor (Batch, Seq_Len, Channels, Height, Width)
            pred_steps: Number of forecast horizons to generate (defaults to self.pred_len)
            return_intensity: If True and intensity head is enabled, returns (presence, intensity).
            
        Returns:
            presence_predictions: (Batch, Pred_Len, Out_Channels=1, Height, Width)
            [optional] intensity_predictions: (Batch, Pred_Len, Out_Channels=1, Height, Width)
        """
        if pred_steps is None:
            pred_steps = self.pred_len

        b, seq_len, _, h, w = x.shape
        device = x.device

        # 1. Initialize hidden states for all encoder layers
        states = [cell.init_hidden(b, h, w, device) for cell in self.encoder_cells]

        # 2. Sequential Encoding over historical time steps (t = 0 ... seq_len - 1)
        for t in range(seq_len):
            x_t = x[:, t]  # (B, C, H, W)
            for l, cell in enumerate(self.encoder_cells):
                h_l, c_l = cell(x_t, states[l])
                h_l = self.dropout(h_l)
                states[l] = (h_l, c_l)
                x_t = h_l  # Input to next layer

        # Top-layer state will initialize decoder
        decoder_state = states[-1]
        decoder_input = decoder_state[0]  # Start with final encoder top hidden state

        # 3. Decoding future steps
        presence_outputs = []
        intensity_outputs = []

        for _ in range(pred_steps):
            h_dec, c_dec = self.decoder_cell(decoder_input, decoder_state)
            h_dec = self.dropout(h_dec)
            decoder_state = (h_dec, c_dec)

            # Heads
            presence_frame = self.presence_head(h_dec)
            presence_outputs.append(presence_frame)

            if self.use_intensity_head:
                intensity_frame = self.intensity_head(h_dec)
                intensity_outputs.append(intensity_frame)

            # Autoregressive feedback for next decoder step
            decoder_input = h_dec

        # Stack across time: (Batch, Pred_Len, Channels, Height, Width)
        presence = torch.stack(presence_outputs, dim=1)

        if return_intensity and self.use_intensity_head:
            intensity = torch.stack(intensity_outputs, dim=1)
            return presence, intensity

        return presence

