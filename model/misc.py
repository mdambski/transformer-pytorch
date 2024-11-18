import torch
from torch import nn as nn, Tensor
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    """
    Generation layer.
    """

    def __init__(self, d_model: int, vocab: int):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: Tensor) -> Tensor:
        return log_softmax(self.proj(x), dim=-1)


class LayerNorm(nn.Module):
    """
    Normalization layer.

    Implements layer normalization, a technique that normalizes summed inputs
    to neurons within a layer, stabilizing and reducing training time.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_feed_forward: int, dropout: float = 0.1) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_feed_forward)
        self.w_2 = nn.Linear(d_feed_forward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.dropout(self.w_1(x).relu()))
