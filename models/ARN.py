import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    GroupVectorQuantizer,
    ResidualVectorQuantizer,
    ConvSTFT,
    ConviSTFT,
    MultiHeadAttentionEncoder,
)


EPSILON = torch.finfo(torch.float32).eps


class ConvEncoder(nn.Module):
    """
    Real-valued Conv layers in encoder
    Causal version
    """

    def __init__(
        self, kernel_size, kernel_stride, kernel_num, layer_output, causal=True
    ):
        super(ConvEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_num = (2,) + kernel_num

        self.num_layers = len(self.kernel_size)

        self.layer_output = layer_output
        self.causal = causal

        self.module_list = nn.ModuleList()
        for idx in range(self.num_layers):
            if idx == 0:
                self.module_list.append(
                    nn.Sequential(
                        nn.ConstantPad2d((0, 0, 1, 0), value=0.0),
                        nn.Conv2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx + 1],
                            kernel_size=(
                                self.kernel_size[idx],
                                2 if self.causal else 3,
                            ),
                            stride=(self.kernel_stride[idx], 1),
                            padding=(self.kernel_size[idx] // 2 - 1, 1),
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx + 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.module_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx + 1],
                            kernel_size=(
                                self.kernel_size[idx],
                                2 if self.causal else 3,
                            ),
                            stride=(self.kernel_stride[idx], 1),
                            padding=(self.kernel_size[idx] // 2, 1),
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx + 1]),
                        nn.PReLU(),
                    )
                )

    def forward(self, x):
        """
        Input shape:[B, C, F, T]
        Output shape:[B, C', F', T]
        """
        out = x
        if self.layer_output:
            out_list = []

        for layer in self.module_list:
            out = layer(out)
            if self.causal:
                out = out[..., :-1]
            # print(out.shape)

            if self.layer_output:
                out_list.append(out)

        if self.layer_output:
            return out, out_list

        return out


class ConvDecoder(nn.Module):
    """
    TransConv layers in decoder
    """

    def __init__(self, kernel_size, kernel_stride, kernel_num, layer_output, causal):
        super(ConvDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_num = (2,) + kernel_num

        self.num_layers = len(self.kernel_size)

        self.layer_output = layer_output
        self.causal = causal

        self.module_list = nn.ModuleList()
        for idx in range(self.num_layers, 0, -1):
            # idx from num_layers to 1
            if idx != 1:
                self.module_list.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx - 1],
                            kernel_size=(
                                self.kernel_size[idx - 1],
                                2 if self.causal else 3,
                            ),
                            stride=(self.kernel_stride[idx - 1], 1),
                            padding=(
                                self.kernel_size[idx - 1] // 2,
                                0 if self.causal else 1,
                            ),
                            output_padding=(self.kernel_stride[idx - 1] - 1, 0),
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]),
                        nn.PReLU(),
                    )
                )
            else:
                self.module_list.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx - 1],
                            kernel_size=(
                                self.kernel_size[idx - 1],
                                2 if self.causal else 3,
                            ),
                            stride=(self.kernel_stride[idx - 1], 1),
                            padding=(
                                self.kernel_size[idx - 1] // 2 - 1,
                                0 if self.causal else 1,
                            ),
                            output_padding=(self.kernel_stride[idx - 1] - 1, 0),
                        ),
                    )
                )

    def forward(self, x):
        """
        Input shape:[B, C', F', T]
        Output shape:[B, C, F, T]
        """
        out = x
        if self.layer_output:
            out_list = [out]

        for idx, layer in enumerate(self.module_list):
            out = layer(out)
            if self.causal:
                out = out[..., :-1]
            if idx != self.num_layers - 1:
                if self.layer_output:
                    out_list.append(out)
            else:
                out = out[:, :, :-1, :]
            # print(out.shape)

        if self.layer_output:
            return out, out_list

        return out


class TemporalRNN(nn.Module):
    """
    LSTM or GRU for temporal filtering
    Causal version
    """

    def __init__(self, input_size, rnn_layers, rnn_units, rnn_type, causal):
        super(TemporalRNN, self).__init__()

        self.input_size = input_size
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.bidirectional = not causal

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                self.input_size,
                self.rnn_units,
                self.rnn_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                self.input_size,
                self.rnn_units,
                self.rnn_layers,
                batch_first=True,
                bidirectional=self.bidirectional,
            )

        # Project layer
        self.proj = nn.Linear(
            self.rnn_units * 2 if self.bidirectional else self.rnn_units,
            self.input_size,
        )

    def forward(self, x):
        """
        Input shape:[B, C, F, T]
        Output shape:[B, C, F, T]
        """
        batch_size, channels, freq_bins, frames = x.size()
        out = x.permute(0, 2, 3, 1).contiguous()
        # reshape to [B, F, T, C]
        out = out.reshape(batch_size * freq_bins, frames, -1)
        out, _ = self.rnn(out)
        out = self.proj(out)
        out = out.reshape(batch_size, freq_bins, frames, channels)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out


class FrequencyAttention(nn.Module):
    """
    MHSA+FFN encoder
    """

    def __init__(self, num_layers, input_size, ff_size, n_heads, is_pe):
        super(FrequencyAttention, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.ff_size = ff_size
        self.n_heads = n_heads
        self.is_pe = is_pe

        attn_list = []
        for _ in range(self.num_layers):
            attn_list.append(
                MultiHeadAttentionEncoder(
                    self.input_size, self.ff_size, self.n_heads, self.is_pe
                )
            )
        self.module_list = nn.ModuleList(attn_list)

    def forward(self, x):
        """
        Input shape:[B, C, F, T]
        Output shape:[B, C, F, T]
        """
        out = x
        for layer in self.module_list:
            out = layer(out)

        return out


def power_law(x, alpha=0.5):
    """
    Input shape:[B, 2, F, T]
    Output shape:[B, 2, F, T]

    Note that mag_comp is processed mag, not the compressed mag
    """
    real = x[:, 0, :, :]
    imag = x[:, 1, :, :]
    mag = torch.sqrt(real**2 + imag**2 + EPSILON)
    phase = torch.atan2(imag + EPSILON, real)
    mag_comp = torch.pow(mag, alpha)
    real = mag_comp * torch.cos(phase)
    imag = mag_comp * torch.sin(phase)

    out = torch.stack([real, imag], dim=1)
    return out, mag_comp, mag


def log_law(x, inverse=False):
    """
    Input shape:[B, 2, F, T]
    Output shape:[B, 2, F, T]
    """
    real = x[:, 0, :, :]
    imag = x[:, 1, :, :]
    mag = torch.sqrt(real**2 + imag**2 + EPSILON)
    phase = torch.atan2(imag + EPSILON, real)
    if not inverse:
        mag_comp = torch.log1p(mag)
    else:
        mag_comp = torch.expm1(mag)
    real = mag_comp * torch.cos(phase)
    imag = mag_comp * torch.sin(phase)

    out = torch.stack([real, imag], dim=1)
    return out, mag_comp, mag


class ARN(nn.Module):
    """
    Vector Quantised Attention-Recurrent Network for Neural Audio Coding (Real-valued version).

    The bitrate can be adjusted by simply adjusting the number of groups and the number of bits per codebook.
    """

    def __init__(
        self,
        win_len=1764,
        win_inc=441,
        fft_len=2048,
        win_type="hann",
        # fft params
        kernel_size=None,
        kernel_stride=None,
        kernel_num=None,
        rnn_layers=2,
        rnn_units=128,
        rnn_type="LSTM",
        # rnn params
        attn_layers=2,
        n_heads=8,
        # attn params
        vq="GVQ",
        project_dim=64,
        n_quantizers=48,
        bit_per_cbk=10,
        # VQ params
        comp_law="power-law",
        alpha=0.5,
        # power-law factor
        causal=True,
    ):
        super(ARN, self).__init__()

        self.causal = causal

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type
        self.stft = ConvSTFT(
            self.win_len, self.win_inc, self.fft_len, self.win_type, "complex"
        )
        self.istft = ConviSTFT(
            self.win_len, self.win_inc, self.fft_len, self.win_type, "complex"
        )

        if kernel_size is None:
            kernel_size = (7, 7, 7, 7)
        if kernel_stride is None:
            kernel_stride = (1, 4, 4, 4)
        if kernel_num is None:
            # kernel_num = [16, 32, 64, 64]
            kernel_num = (16, 32, 48, 48)
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_num = kernel_num
        self.conv_encoder = ConvEncoder(
            self.kernel_size, self.kernel_stride, self.kernel_num, True, self.causal
        )
        self.conv_decoder = ConvDecoder(
            self.kernel_size, self.kernel_stride, self.kernel_num, True, self.causal
        )

        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.rnn_type = rnn_type
        self.embedding_dim = self.fft_len // 2
        for stride in self.kernel_stride:
            self.embedding_dim //= stride
        self.embedding_dim *= self.kernel_num[-1]
        # rnn_input_size = self.embedding_dim
        rnn_input_size = self.kernel_num[-1]

        # print(rnn_input_size)
        self.rnn_encoder = TemporalRNN(
            rnn_input_size, self.rnn_layers, self.rnn_units, self.rnn_type, self.causal
        )
        self.rnn_decoder = TemporalRNN(
            rnn_input_size, self.rnn_layers, self.rnn_units, self.rnn_type, self.causal
        )

        self.attn_layers = attn_layers
        self.n_heads = n_heads
        attn_input_size = self.kernel_num[-1]
        self.attn_encoder = FrequencyAttention(
            self.attn_layers,
            attn_input_size,
            attn_input_size * 4,
            self.n_heads,
            is_pe=True,
        )
        self.attn_decoder = FrequencyAttention(
            self.attn_layers,
            attn_input_size,
            attn_input_size * 4,
            self.n_heads,
            is_pe=True,
        )
        # 4x in FFN

        assert vq in ["GVQ", "RVQ", "MPRVQ"], "Only support GVQ or RVQ"
        self.vq = vq
        self.n_quantizers = n_quantizers
        self.bit_per_cbk = bit_per_cbk
        self.project_dim = project_dim
        if vq == "GVQ":
            self.vector_quantizer = GroupVectorQuantizer(
                self.embedding_dim,
                self.n_quantizers,
                self.bit_per_cbk,
                self.project_dim,
            )
        elif vq == "RVQ":
            self.vector_quantizer = ResidualVectorQuantizer(
                self.embedding_dim,
                self.project_dim,
                self.n_quantizers,
                self.bit_per_cbk,
            )

        assert comp_law in [
            "power-law",
            "log-law",
        ], "Only support power-law and log-law"
        self.comp_law = comp_law
        if self.comp_law == "power-law":
            self.alpha = alpha
        elif self.comp_law == "log-law":
            pass

    def encode(self, x):
        specs = self.stft(x)
        real = specs[:, : self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1 :]
        specs = torch.stack([real, imag], dim=1)

        if self.comp_law == "power-law":
            specs_comp, mags_comp, _ = power_law(specs, self.alpha)
        elif self.comp_law == "log-law":
            specs_comp, mags_comp, _ = log_law(specs)
        else:
            specs_comp, mags_comp = None, None

        out, _ = self.conv_encoder(specs_comp)

        res = self.attn_encoder(out)
        out = torch.add(out, res)

        res = self.rnn_encoder(out)
        feature = torch.add(out, res)

        feature = feature.reshape(feature.shape[0], -1, feature.shape[-1])
        indices = self.vector_quantizer.encode(feature)

        return indices

    def decode(self, indices):
        quantized = self.vector_quantizer.decode(indices)
        quantized = quantized.reshape(
            quantized.shape[0], self.kernel_num[-1], -1, quantized.shape[-1]
        )

        res = self.rnn_decoder(quantized)
        out = torch.add(quantized, res)

        res = self.attn_decoder(out)
        out = torch.add(out, res)

        est_specs_comp, _ = self.conv_decoder(out)

        if self.comp_law == "power-law":
            est_specs, _, _ = power_law(est_specs_comp, 1.0 / self.alpha)
        elif self.comp_law == "log-law":
            est_specs, _, _ = log_law(est_specs_comp, inverse=True)
        else:
            est_specs = None

        out_wav = self.istft(
            est_specs.reshape(est_specs.shape[0], -1, est_specs.shape[-1])
        )
        out_wav = torch.clamp_(out_wav, -1, 1)

        return out_wav

    def forward(self, x):
        """
        Input shape:[B, 1, waveforms]
        Output shape:[B, 1, waveforms]
        The shape of tensor keep [B, (C), F, T] in forward flow.
        """

        specs = self.stft(x)
        real = specs[:, : self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1 :]
        specs = torch.stack([real, imag], dim=1)

        if self.comp_law == "power-law":
            specs_comp, mags_comp, _ = power_law(specs, self.alpha)
        elif self.comp_law == "log-law":
            specs_comp, mags_comp, _ = log_law(specs)
        else:
            specs_comp, mags_comp = None, None

        encoder_feature = []
        decoder_feature = []

        out, out_list = self.conv_encoder(specs_comp)
        encoder_feature += out_list

        res = self.attn_encoder(out)
        out = torch.add(out, res)
        encoder_feature.append(out)

        res = self.rnn_encoder(out)
        feature = torch.add(out, res)

        feature = feature.reshape(feature.shape[0], -1, feature.shape[-1])
        quantized, vq_input, vq_output_detach = self.vector_quantizer(feature)
        quantized = quantized.reshape(
            quantized.shape[0], self.kernel_num[-1], -1, quantized.shape[-1]
        )

        res = self.rnn_decoder(quantized)
        out = torch.add(quantized, res)
        decoder_feature.append(out)

        res = self.attn_decoder(out)
        out = torch.add(out, res)

        est_specs_comp, out_list = self.conv_decoder(out)
        decoder_feature += out_list

        if self.comp_law == "power-law":
            est_specs, _, est_mags_comp = power_law(est_specs_comp, 1.0 / self.alpha)
        elif self.comp_law == "log-law":
            est_specs, _, est_mags_comp = log_law(est_specs_comp, inverse=True)
        else:
            est_specs, est_mags_comp = None, None

        out_wav = self.istft(
            est_specs.reshape(est_specs.shape[0], -1, est_specs.shape[-1])
        )
        out_wav = torch.clamp_(out_wav, -1, 1)

        # for consistency constraints
        re_specs = self.stft(out_wav)
        real = re_specs[:, : self.fft_len // 2 + 1]
        imag = re_specs[:, self.fft_len // 2 + 1 :]
        re_specs = torch.stack([real, imag], dim=1)

        if self.comp_law == "power-law":
            est_specs_comp, est_mags_comp, _ = power_law(re_specs, self.alpha)
        elif self.comp_law == "log-law":
            est_specs_comp, est_mags_comp, _ = log_law(re_specs)
        else:
            est_specs_comp, est_mags_comp = None, None

        est_specs_comp = torch.cat(
            (est_specs_comp[:, 0, :, :], est_specs_comp[:, 1, :, :]), dim=1
        )
        specs_comp = torch.cat((specs_comp[:, 0, :, :], specs_comp[:, 1, :, :]), dim=1)

        return {
            "specs_comp": specs_comp,
            "est_specs_comp": est_specs_comp,
            "mags_comp": mags_comp,
            "est_mags_comp": est_mags_comp,
            "est_wav": out_wav,
            "raw_wav": x,
            "vq_input": vq_input,
            "vq_output_detach": vq_output_detach,
            "encoder_feature": encoder_feature,
            "decoder_feature": decoder_feature,
        }
