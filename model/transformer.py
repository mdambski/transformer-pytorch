from torch import nn, Tensor

from model.misc import Generator
from model.decoder import Decoder
from model.encoder import Encoder


class Transformer(nn.Module):
    """Standard Encoder-Decoder architecture."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: nn.Sequential,
        tgt_embedding: nn.Sequential,
        generator: Generator,
    ):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.generator = generator

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor):
        """Take in and process masked src and tgt sequences."""
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(enc_out, src_mask, tgt, tgt_mask)
        return dec_out

    def decode(
        self,
        encoded_src: Tensor,
        src_mask: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
    ):
        x = self.tgt_embedding(tgt)
        out = self.decoder(x, encoded_src, src_mask, tgt_mask)
        return out

    def encode(self, src: Tensor, src_mask: Tensor):
        x = self.src_embedding(src)
        out = self.encoder(x, src_mask)
        return out
