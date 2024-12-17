# src/models/components/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for cross-modal fusion.
    Allows the model to jointly attend to information from different spaces.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize multi-head attention.

        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim, bias=False)
        self.k_linear = nn.Linear(dim, dim, bias=False)
        self.v_linear = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        """Initialize attention weights using Kaiming initialization."""
        for linear in [self.q_linear, self.k_linear, self.v_linear, self.out]:
            kaiming_normal_(linear.weight, mode='fan_out', nonlinearity='relu')
            # If there's a bias, initialize to zero
            if linear.bias is not None:
                constant_(linear.bias, 0)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.

        Args:
            q: Query tensor, shape [batch_size, seq_len_q, dim]
            k: Key tensor, shape [batch_size, seq_len_k, dim]
            v: Value tensor, shape [batch_size, seq_len_v, dim]
            mask: Optional attention mask

        Returns:
            out: The transformed output [batch_size, seq_len_q, dim]
            attn: The attention weights [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)

        # Project and reshape to [batch_size, seq_len, num_heads, head_dim]
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len_q, head_dim]

        # Reshape and project back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out = self.out(out)
        out = self.out_dropout(out)

        return out, attn
