import torch.nn as nn
import copy

from model.attention import MultiHeadedAttention
from model.decoder import Decoder, DecoderLayer
from model.embeddings import Embeddings, PositionalEmbeddings
from model.encoder import Encoder, EncoderLayer
from model.misc import Generator, PositionwiseFeedForward
from model.transformer import Transformer


def make_model(
    src_vocab: int,
    tgt_vocab: int,
    layers_num: int = 6,
    d_model: int = 512,
    d_feed_forward: int = 2048,
    head_size: int = 8,
    dropout: float = 0.1,
) -> Transformer:
    """
    Builds a Transformer model based on the given hyper-parameters.

    This function constructs a Transformer model consisting of an encoder,
    a decoder, embeddings (with positional encoding), multi-headed attention,
    and position-wise feed-forward layers. The model's weights are initialized
    using the Xavier uniform initialization method for improved training stability.

    Parameters
    ----------
    src_vocab : int
        Size of the source vocabulary (number of tokens).
    tgt_vocab : int
        Size of the target vocabulary (number of tokens).
    layers_num : int, optional
        Number of layers in both the encoder and decoder stacks (default is 6).
    d_model : int, optional
        Dimensionality of the model's embeddings and hidden states (default is 512).
    d_feed_forward : int, optional
        Dimensionality of the feed-forward network's inner layer (default is 2048).
    head_size : int, optional
        Number of attention heads in the multi-headed attention mechanism (default is 8).
    dropout : float, optional
        Dropout rate applied in various parts of the model (default is 0.1).

    Returns
    -------
    Transformer
        An instance of the Transformer model configured with the specified hyper-parameters.
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(head_size, d_model)
    ff = PositionwiseFeedForward(d_model, d_feed_forward, dropout)
    position = PositionalEmbeddings(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), layers_num),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), layers_num),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
