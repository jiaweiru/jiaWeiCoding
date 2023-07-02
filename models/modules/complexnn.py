import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttentionEncoder


class ComplexMHA(nn.Module):
    """
    Adapt MHA to complex network,
    """

    def __init__(self, d_model, d_ff, n_heads, is_pe):
        super(ComplexMHA, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.is_pe = is_pe
        self.r_MHA = MultiHeadAttentionEncoder(
            self.d_model // 2, self.d_ff // 2, self.n_heads, self.is_pe
        )
        self.i_MHA = MultiHeadAttentionEncoder(
            self.d_model // 2, self.d_ff // 2, self.n_heads, self.is_pe
        )

    def forward(self, x):
        real, imag = torch.chunk(x, 2, dim=1)
        # print(real.shape)

        real2real = self.r_MHA(real)
        real2imag = self.i_MHA(real)
        imag2real = self.r_MHA(imag)
        imag2imag = self.i_MHA(imag)

        real = torch.sub(real2real, imag2imag)
        imag = torch.add(real2imag, imag2real)

        out = torch.cat([real, imag], dim=1)

        # out = torch.add(out, x)

        return out


class cPReLU(nn.Module):
    def __init__(self, complex_axis=1):
        super(cPReLU, self).__init__()
        self.r_prelu = nn.PReLU()
        self.i_prelu = nn.PReLU()
        self.complex_axis = complex_axis

    def forward(self, inputs):
        real, imag = torch.chunk(inputs, 2, self.complex_axis)
        real = self.r_prelu(real)
        imag = self.i_prelu(imag)
        return torch.cat([real, imag], self.complex_axis)


class NaiveComplexGRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        projection_dim=None,
        bidirectional=False,
        batch_first=False,
    ):
        super(NaiveComplexGRU, self).__init__()

        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.real_lstm = nn.GRU(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.imag_lstm = nn.GRU(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=False,
        )
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        if projection_dim is not None:
            self.projection_dim = projection_dim // 2
            self.r_trans = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
            self.i_trans = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs, list):
            real, imag = inputs
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        # print(real_out.shape,imag_out.shape)
        return [real_out, imag_out]


class NaiveComplexLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        projection_dim=None,
        bidirectional=False,
        batch_first=False,
    ):
        super(NaiveComplexLSTM, self).__init__()

        self.input_dim = input_size // 2
        self.rnn_units = hidden_size // 2
        self.real_lstm = nn.LSTM(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.imag_lstm = nn.LSTM(
            self.input_dim,
            self.rnn_units,
            num_layers=1,
            bidirectional=bidirectional,
            batch_first=False,
        )
        if bidirectional:
            bidirectional = 2
        else:
            bidirectional = 1
        if projection_dim is not None:
            self.projection_dim = projection_dim // 2
            self.r_trans = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
            self.i_trans = nn.Linear(
                self.rnn_units * bidirectional, self.projection_dim
            )
        else:
            self.projection_dim = None

    def forward(self, inputs):
        if isinstance(inputs, list):
            real, imag = inputs
        elif isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, -1)
        r2r_out = self.real_lstm(real)[0]
        r2i_out = self.imag_lstm(real)[0]
        i2r_out = self.real_lstm(imag)[0]
        i2i_out = self.imag_lstm(imag)[0]
        real_out = r2r_out - i2i_out
        imag_out = i2r_out + r2i_out
        if self.projection_dim is not None:
            real_out = self.r_trans(real_out)
            imag_out = self.i_trans(imag_out)
        # print(real_out.shape,imag_out.shape)
        return [real_out, imag_out]

    def flatten_parameters(self):
        # may work with multi-gpu
        self.imag_lstm.flatten_parameters()
        self.real_lstm.flatten_parameters()


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=1,
        groups=1,
        causal=True,
        complex_axis=1,
    ):
        """
        in_channels: real+imag
        out_channels: real+imag
        kernel_size : input [B,C,D,T] kernel size in [D,T]
        padding : input [B,C,D,T] padding in [D,T]
        causal: if causal, will padding time dimension's left side,
                otherwise both
        """
        super(ComplexConv2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.causal = causal
        self.groups = groups
        self.dilation = dilation
        self.complex_axis = complex_axis
        self.real_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=[self.padding[0], 0],
            dilation=self.dilation,
            groups=self.groups,
        )
        self.imag_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=[self.padding[0], 0],
            dilation=self.dilation,
            groups=self.groups,
        )

        nn.init.normal_(self.real_conv.weight.data, std=0.05)
        nn.init.normal_(self.imag_conv.weight.data, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.0)
        nn.init.constant_(self.imag_conv.bias, 0.0)

    def forward(self, inputs):
        # if causal padding the back side, else two sides
        if self.padding[1] != 0 and self.causal:
            inputs = F.pad(inputs, [self.padding[1], 0, 0, 0])
        else:
            inputs = F.pad(inputs, [self.padding[1], self.padding[1], 0, 0])

        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(
                real,
            )
            imag2imag = self.imag_conv(
                imag,
            )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        output_padding=(0, 0),
        causal=False,
        complex_axis=1,
        groups=1,
    ):
        """
        in_channels: real+imag
        out_channels: real+imag
        """
        super(ComplexConvTranspose2d, self).__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

        self.real_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
        )
        self.imag_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size,
            self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
        )
        self.complex_axis = complex_axis

        nn.init.normal_(self.real_conv.weight, std=0.05)
        nn.init.normal_(self.imag_conv.weight, std=0.05)
        nn.init.constant_(self.real_conv.bias, 0.0)
        nn.init.constant_(self.imag_conv.bias, 0.0)

    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            real, imag = torch.chunk(inputs, 2, self.complex_axis)
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            real = inputs[0]
            imag = inputs[1]
        if self.complex_axis == 0:
            real = self.real_conv(inputs)
            imag = self.imag_conv(inputs)
            real2real, imag2real = torch.chunk(real, 2, self.complex_axis)
            real2imag, imag2imag = torch.chunk(imag, 2, self.complex_axis)

        else:
            if isinstance(inputs, torch.Tensor):
                real, imag = torch.chunk(inputs, 2, self.complex_axis)

            real2real = self.real_conv(
                real,
            )
            imag2imag = self.imag_conv(
                imag,
            )

            real2imag = self.imag_conv(real)
            imag2real = self.real_conv(imag)

        real = real2real - imag2imag
        imag = real2imag + imag2real
        out = torch.cat([real, imag], self.complex_axis)

        return out


class NaiveComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features // 2, **kwargs)
        self.bn_im = nn.BatchNorm2d(num_features=num_features // 2, **kwargs)

    def forward(self, x):
        batch, channels, f, t = x.size()
        real = self.bn_re(x[:, : channels // 2, :, :])
        imag = self.bn_im(x[:, channels // 2 :, :, :])
        output = torch.cat((real, imag), dim=1)
        return output
