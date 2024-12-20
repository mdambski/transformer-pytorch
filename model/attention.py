import math
from typing import Optional

import torch
from torch import Tensor, nn as nn

from model.utils import clones


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple[Tensor, Tensor]:
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size: int, d_model: int, dropout: float = 0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()

        if d_model % head_size != 0:
            raise ValueError("Incompatible hyper-parameters: d_model and head_size")

        self.d_k = d_model // head_size

        self.head_size = head_size
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attn: Optional[Tensor] = None

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)

        n_batches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(n_batches, -1, self.head_size, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(n_batches, -1, self.head_size * self.d_k)
        )

        del query
        del key
        del value

        return self.linears[-1](x)
