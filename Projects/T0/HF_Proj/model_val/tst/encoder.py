import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tst.multiHeadAttention import MultiHeadAttention, MultiHeadAttentionChunk, MultiHeadAttentionWindow
from tst.positionwiseFeedForward import PositionwiseFeedForward
from tst.utilities import TrainingOptions


class Encoder(nn.Module):
    # """Encoder block from Attention is All You Need.
    #
    # Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    # Residual sum and normalization are applied at each step.
    #
    # Parameters
    # ----------
    # d_model:
    #     Dimension of the input vector.
    # q:
    #     Dimension of all query matrix.
    # v:
    #     Dimension of all value matrix.
    # h:
    #     Number of heads.
    # attention_size:
    #     Number of backward elements to apply attention.
    #     Deactivated if ``None``. Default is ``None``.
    # dropout:
    #     Dropout probability after each MHA or PFF block.
    #     Default is ``0.3``.
    # chunk_mode:
    #     Swict between different MultiHeadAttention blocks.
    #     One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    # """

    def __init__(self,
                 tOpt):
        """Initialize the Encoder block"""
        super().__init__()

        self._d_model = tOpt.d_model
        d_input = tOpt.d_input
        d_model = tOpt.d_model
        d_output = tOpt.d_output
        q = tOpt.query
        v = tOpt.value
        h = tOpt.heads
        N = tOpt.N_stack
        attention_size = tOpt.attention_size
        window_size = tOpt.window
        padding = tOpt.padding
        dropout = tOpt.dropout
        chunk_mode = tOpt.chunk_mode
        pe = tOpt.pe

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        elif chunk_mode is None:
            MHA = MultiHeadAttention
        else:
            raise NameError(
                f'chunk_mode "{chunk_mode}" not understood. Must be one of {", ".join(chunk_mode_modules.keys())} or None.')

        self._selfAttention = MHA(d_model, q, v, h, attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dropout(x)
        x = self._layerNorm1(x + residual)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dropout(x)
        x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map

if __name__ == '__main__':
    encoder = Encoder(1)