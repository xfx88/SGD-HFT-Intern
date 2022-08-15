from typing import Optional
import torch
import torch.nn as nn
from torch.nn.functional import gelu

from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE

from tst.utilities import TrainingOptions

class TransformerEncoder(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is ``24``.
    """

    def __init__(self,
                 tOpt,
                 pe_period = None):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = tOpt.d_model
        self.d_input = tOpt.d_input
        self.d_model = tOpt.d_model
        self.d_output = tOpt.d_output
        N = tOpt.N_stack
        pe = tOpt.pe

        self.layers_encoding = nn.ModuleList([Encoder(tOpt) for _ in range(N)])
        # self.layers_decoding = nn.ModuleList([Decoder(tOpt) for _ in range(N)])

        self._embedding = nn.Linear(self.d_input, self.d_model)
        self.upsample = nn.Linear(self.d_model, 512)
        self.relu = nn.ReLU()

        self.downsample = nn.Linear(512, self.d_output)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor):
        K = x.shape[1]

        # Embedding module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        decoding = encoding

        if self._generate_PE is not None:
            positional_encoding = self._generate_PE(K, self._d_model)
            positional_encoding = positional_encoding.to(decoding.device)
            encoding.add_(positional_encoding)

        for layer in self.layers_decoding:
            decoding = layer(decoding, encoding)

        to_output = self.upsample(decoding)
        to_output = self.relu(to_output)
        # Output module
        y = self.downsample(to_output)

        return y


class TransformerCLS(nn.Module):
    """Transformer model from Attention is All You Need.

    A classic transformer model adapted for sequential data.
    Embedding has been replaced with a fully connected layer,
    the last layer softmax is now a sigmoid.

    Attributes
    ----------
    layers_encoding: :py:class:`list` of :class:`Encoder.Encoder`
        stack of Encoder layers.
    layers_decoding: :py:class:`list` of :class:`Decoder.Decoder`
        stack of Decoder layers.

    Parameters
    ----------
    d_input:
        Model input dimension.
    d_model:
        Dimension of the input vector.
    d_output:
        Model output dimension.
    q:
        Dimension of queries and keys.
    v:
        Dimension of values.
    h:
        Number of heads.
    N:
        Number of encoder and decoder layers to stack.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Switch between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    pe:
        Type of positional encoding to add.
        Must be one of ``'original'``, ``'regular'`` or ``None``. Default is ``None``.
    pe_period:
        If using the ``'regular'` pe, then we can define the period. Default is ``24``.
    """

    def __init__(self,
                 tOpt,
                 pe_period = None):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_model = tOpt.d_model
        self.d_input = tOpt.d_input
        self.d_model = tOpt.d_model
        self.d_output = tOpt.d_output
        N = tOpt.N_stack
        pe = tOpt.pe

        self.layers_encoding = nn.ModuleList([Encoder(tOpt) for _ in range(N)])

        self._embedding = nn.Linear(self.d_input, self.d_model)
        self.upsample = nn.Linear(self.d_model, 512)
        self.relu = nn.ReLU()

        # self.downsample1 = nn.Linear(512, self.d_output // 3)
        self.downsample2 = nn.Linear(512, self.d_output // 3)
        # self.downsample3 = nn.Linear(512, self.d_output // 3)

        pe_functions = {
            'original': generate_original_PE,
            'regular': generate_regular_PE,
        }

        if pe in pe_functions.keys():
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
        else:
            raise NameError(
                f'PE "{pe}" not understood. Must be one of {", ".join(pe_functions.keys())} or None.')

        self.name = 'transformer'

    def forward(self, x: torch.Tensor):
        K = x.shape[1]

        # Embedding module
        encoding = self._embedding(x)

        # Add position encoding
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(K, self._d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)
            encoding.add_(positional_encoding)

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        # decoding = encoding

        # if self._generate_PE is not None:
        #     positional_encoding = self._generate_PE(K, self._d_model)
        #     positional_encoding = positional_encoding.to(decoding.device)
        #     encoding.add_(positional_encoding)

        # for layer in self.layers_decoding:
        #     decoding = layer(decoding, encoding)

        to_output = self.upsample(encoding)
        to_output = self.relu(to_output)
        # Output module
        # p2 = self.downsample1(to_output)
        p5 = self.downsample2(to_output)
        # p18 = self.downsample3(to_output)

        return p5