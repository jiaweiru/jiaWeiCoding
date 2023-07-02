import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionMask(nn.Module):
    """
    Time attention with causal_mask
    Freq attention without mask
    """

    def __init__(self, causal):
        super(AttentionMask, self).__init__()
        self.causal = causal

    def lower_triangular_mask(self, shape):
        """

        Parameters
        ----------
        shape : a tuple of ints
        Returns
        -------
        a square Boolean tensor with the lower triangle being False
        """
        row_index = torch.cumsum(torch.ones(size=shape), dim=-2)
        col_index = torch.cumsum(torch.ones(size=shape), dim=-1)
        return torch.lt(
            row_index, col_index
        )  # lower triangle:True, upper triangle:False

    def merge_masks(self, x, y):
        if x is None:
            return y
        if y is None:
            return x
        return torch.logical_and(x, y)

    def forward(self, inp):
        # input (bs, L, ...)
        max_seq_len = inp.shape[1]
        if self.causal:
            causal_mask = self.lower_triangular_mask(
                [max_seq_len, max_seq_len]
            )  # (L, l)
            return causal_mask
        else:
            return torch.zeros(size=(max_seq_len, max_seq_len), dtype=torch.float32)


class PositionalEncoding(nn.Module):
    """This class implements the absolute sinusoidal positional encoding function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).
    Example
    -------
    >>> a = torch.rand((8, 120, 512))
    >>> enc = PositionalEncoding(input_size=a.shape[-1])
    >>> b = enc(a)
    >>> b.shape
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float() * -(math.log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        return self.pe[:, : x.size(1)].clone().detach()


class MultiHeadAttentionEncoder(nn.Module):
    """
    MHA encoder without attention mask
    """

    def __init__(self, d_model, d_ff, n_heads, is_pe):
        super(MultiHeadAttentionEncoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        self.MHA = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.n_heads, bias=False
        )
        self.norm_1 = nn.LayerNorm([self.d_model], eps=1e-6)
        self.fc_1 = nn.Conv1d(self.d_model, self.d_ff, 1)
        self.act = nn.ReLU()
        self.fc_2 = nn.Conv1d(self.d_ff, self.d_model, 1)
        self.norm_2 = nn.LayerNorm([self.d_model], eps=1e-6)

        self.is_pe = is_pe
        if self.is_pe:
            self.pe = PositionalEncoding(input_size=self.d_model)

    def forward(self, x):
        # [B, C, F, T]
        B, C, F, T = x.size()

        out = x.permute(0, 3, 2, 1).reshape(B * T, F, C)
        # [B * T, F, C]

        if self.is_pe:
            out = out + self.pe(out)
            # PositionalEncoding

        out = out.permute(1, 0, 2).contiguous()
        # [F, B * T, C]

        res, _ = self.MHA(out, out, out, need_weights=False)
        # [F, B * T, C]
        out = torch.add(out, res).permute(1, 0, 2).contiguous()
        out = self.norm_1(out)
        # [B * T, F, C]

        res = self.fc_1(out.permute(0, 2, 1).contiguous())
        res = self.act(res)
        res = self.fc_2(res).permute(0, 2, 1).contiguous()

        out = torch.add(out, res)
        out = self.norm_2(out)
        out = out.reshape(B, T, F, C).permute(0, 3, 2, 1).contiguous()

        # out = torch.add(out, x)

        return out
