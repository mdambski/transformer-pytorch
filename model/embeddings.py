import math

import torch
from torch import nn as nn, Tensor


class Embeddings(nn.Module):
    def __init__(self, model_dimensions: int, vocab: int):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab, model_dimensions)
        self.model_dimensions = model_dimensions

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x) * math.sqrt(self.model_dimensions)


class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, dropout, max_len=5000):
        super(PositionalEmbeddings, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
