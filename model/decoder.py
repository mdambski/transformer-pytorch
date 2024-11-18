from torch import nn as nn, Tensor

from model.attention import MultiHeadedAttention
from model.misc import LayerNorm, SublayerConnection, PositionwiseFeedForward
from model.utils import clones


class DecoderLayer(nn.Module):
    """Decoder layer is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        src_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(
        self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        out = self.sublayer[2](x, self.feed_forward)
        return out


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer: DecoderLayer, layer_count: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, layer_count)
        self.norm = LayerNorm(layer.size)

    def forward(
        self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
