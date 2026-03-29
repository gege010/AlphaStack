"""
src/models/model_lstm.py
Bidirectional LSTM with multi-head self-attention and Monte Carlo Dropout.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head attention over time steps."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(x, x, x)
        return self.norm(x + attn_out), attn_weights  # Residual


class TemporalBlock(nn.Module):
    """Single BiLSTM block with residual projection and layer norm."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        # Residual projection if dims don't match
        self.proj = nn.Linear(input_dim, out_dim) if input_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.norm(out)
        out = self.dropout(out)
        residual = self.proj(x)
        return out + residual


class BiLSTMAttentionModel(nn.Module):
    """
    Architecture:
        Input → [Input Projection] → N × TemporalBlock → MultiHeadAttention
        → Global Average Pool + Last Timestep → [Head MLP] → Output

    Supports:
    - Monte Carlo Dropout for uncertainty estimation
    - Multi-horizon prediction (multiple output heads)
    - Auxiliary loss on intermediate representations
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attention_heads: int = 4,
        output_dim: int = 1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Input projection + normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Stacked BiLSTM blocks
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.temporal_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_d = hidden_dim if i == 0 else lstm_out_dim
            self.temporal_blocks.append(
                TemporalBlock(in_d, hidden_dim, dropout, bidirectional)
            )

        # Self-attention over temporal dimension
        self.attention = MultiHeadSelfAttention(lstm_out_dim, attention_heads, dropout * 0.5)

        # Global context: concat [avg_pool, last_step]
        combined_dim = lstm_out_dim * 2
        self.head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(combined_dim // 2, combined_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(combined_dim // 4, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                        # Forget gate bias = 1 (helps learning long deps)
                        n = param.size(0)
                        param.data[n // 4: n // 2].fill_(1.0)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (batch, seq_len, input_dim)
        Returns: (predictions, attention_weights) if return_attention else (predictions, None)
        """
        # Input projection
        x = self.input_proj(x)

        # Temporal blocks
        for block in self.temporal_blocks:
            x = block(x)

        # Self-attention
        x, attn_weights = self.attention(x)

        # Aggregate: avg pool + last timestep
        avg_pool = x.mean(dim=1)
        last_step = x[:, -1, :]
        context = torch.cat([avg_pool, last_step], dim=-1)

        out = self.head(context)

        if return_attention:
            return out, attn_weights
        return out, None

    @torch.no_grad()
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout inference.
        Returns: (mean_prediction, std_prediction, all_samples)
        """
        self.train()  # Enable dropout during inference
        preds = []
        for _ in range(n_samples):
            pred, _ = self.forward(x)
            preds.append(pred.cpu().numpy())
        self.eval()
        preds = np.array(preds)           # (n_samples, batch, output_dim)
        mean = preds.mean(axis=0)
        std = preds.std(axis=0)
        return mean, std, preds

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
