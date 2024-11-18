from torch import nn as nn, Tensor

from model.attention import MultiHeadedAttention
from model.misc import LayerNorm, SublayerConnection, PositionwiseFeedForward
from model.utils import clones


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(
        self,
        size: int,
        self_attention: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ) -> None:
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attention
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Follow Figure 1 (left) for connections."""
        attn_layer, ff_layer = self.sublayers

        def calculate_attention(x_in: Tensor) -> Tensor:
            return self.self_attn(x_in, x_in, x_in, mask)

        x = attn_layer(x, calculate_attention)
        out = ff_layer(x, self.feed_forward)

        return out


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: EncoderLayer, layer_count: int) -> None:
        super(Encoder, self).__init__()
        self.layers = clones(layer, layer_count)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
