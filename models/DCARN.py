import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    VectorQuantizerEMA1D,
    ComplexConv2d,
    NaiveComplexBatchNorm2d,
    cPReLU,
    ComplexConvTranspose2d,
    NaiveComplexLSTM,
    NaiveComplexGRU,
    ComplexMHA,
    ConvSTFT,
    ConviSTFT,
)


class ComplexConvEncoder(nn.Module):
    """
    Complex Conv layers in encoder
    Causal version

    The use of naive complex batchnorm instead of complex batchnorm because of its computational complexity.
    Real-valued batchnorm in DCCRN and Uformer can also be used and there should be almost no difference in performance.
    """

    def __init__(self, kernel_size, kernel_stride, kernel_num, layer_output):
        super(ComplexConvEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_num = (2,) + kernel_num

        self.num_layers = len(self.kernel_size)

        self.module_list = nn.ModuleList()
        for idx in range(self.num_layers):
            if idx == 0:
                self.module_list.append(
                    nn.Sequential(
                        nn.ConstantPad2d((0, 0, 1, 0), value=0.0),
                        ComplexConv2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx + 1],
                            kernel_size=(self.kernel_size[idx], 2),
                            stride=(self.kernel_stride[idx], 1),
                            padding=(self.kernel_size[idx] // 2 - 1, 1),
                        ),
                        NaiveComplexBatchNorm2d(self.kernel_num[idx + 1]),
                        cPReLU(),
                    )
                )
            else:
                self.module_list.append(
                    nn.Sequential(
                        ComplexConv2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx + 1],
                            kernel_size=(self.kernel_size[idx], 2),
                            stride=(self.kernel_stride[idx], 1),
                            padding=(self.kernel_size[idx] // 2, 1),
                        ),
                        NaiveComplexBatchNorm2d(self.kernel_num[idx + 1]),
                        cPReLU(),
                    )
                )

        self.layer_output = layer_output

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
            # print(out.shape)

            if self.layer_output:
                out_list.append(out)

        if self.layer_output:
            return out, out_list

        return out


class ComplexConvDecoder(nn.Module):
    """
    Complex TransConv layers in decoder
    """

    def __init__(self, kernel_size, kernel_stride, kernel_num, layer_output):
        super(ComplexConvDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_num = (2,) + kernel_num

        self.num_layers = len(self.kernel_size)

        self.module_list = nn.ModuleList()
        for idx in range(self.num_layers, 0, -1):
            # idx from num_layers to 1
            if idx != 1:
                self.module_list.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size[idx - 1], 2),
                            stride=(self.kernel_stride[idx - 1], 1),
                            padding=(self.kernel_size[idx - 1] // 2, 0),
                            output_padding=(self.kernel_stride[idx - 1] - 1, 0),
                        ),
                        NaiveComplexBatchNorm2d(self.kernel_num[idx - 1]),
                        cPReLU(),
                    )
                )
            else:
                self.module_list.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size[idx - 1], 2),
                            stride=(self.kernel_stride[idx - 1], 1),
                            padding=(self.kernel_size[idx - 1] // 2 - 1, 0),
                            output_padding=(self.kernel_stride[idx - 1] - 1, 0),
                        ),
                    )
                )

        self.layer_output = layer_output

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


class ComplexRNN(nn.Module):
    """
    Complex LSTM or GRU
    Causal version
    """

    def __init__(
        self,
        input_size,
        rnn_layers,
        rnn_units,
        rnn_type,
    ):
        super(ComplexRNN, self).__init__()
        self.input_size = input_size

        if rnn_type == "LSTM":
            self.rnn = NaiveComplexLSTM
        elif rnn_type == "GRU":
            self.rnn = NaiveComplexGRU

        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units

        self.bidirectional = False
        # causal rnn

        rnn_list = []
        for idx in range(self.rnn_layers):
            rnn_list.append(
                self.rnn(
                    input_size=self.input_size if idx == 0 else self.rnn_units,
                    hidden_size=self.rnn_units,
                    bidirectional=self.bidirectional,
                    batch_first=False,
                    projection_dim=self.input_size
                    if idx == self.rnn_layers - 1
                    else None,
                )
            )
        self.module = nn.Sequential(*rnn_list)

    def forward(self, x):
        """
        Input shape:[B, C, F, T]
        Output shape:[B, C, F, T]
        """
        batch_size, channels, freq_bins, frames = x.size()
        out = x.permute(3, 0, 1, 2).contiguous()
        # reshape to [T, B, C, F]
        r_rnn_in = out[:, :, : channels // 2, :].reshape(frames, batch_size, -1)
        i_rnn_in = out[:, :, channels // 2 :, :].reshape(frames, batch_size, -1)
        r_rnn_in, i_rnn_in = self.module([r_rnn_in, i_rnn_in])
        r_rnn_in = r_rnn_in.reshape(frames, batch_size, channels // 2, freq_bins)
        i_rnn_in = i_rnn_in.reshape(frames, batch_size, channels // 2, freq_bins)

        out = torch.cat([r_rnn_in, i_rnn_in], dim=2).permute(1, 2, 3, 0).contiguous()

        return out


class ComplexAttention(nn.Module):
    """
    Complex MHA+FFN encoder
    """

    def __init__(self, num_layers, input_size, ff_size, n_heads, is_pe):
        super(ComplexAttention, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.ff_size = ff_size
        self.n_heads = n_heads
        self.is_pe = is_pe

        attn_list = []
        for idx in range(self.num_layers):
            attn_list.append(
                ComplexMHA(self.input_size, self.ff_size, self.n_heads, self.is_pe)
            )
        self.module = nn.Sequential(*attn_list)

    def forward(self, x):
        """
        Input shape:[B, C, F, T]
        Output shape:[B, C, F, T]
        """
        out = self.module(x)

        return out


class GroupVectorQuantizer(nn.Module):
    """
    Group VectorQuantizer by using VectorQuantizerEMA1D.

    Flatten the features of each frame, then group these features and perform vector quantization.
    """

    def __init__(self, embedding_dim, num_groups, vq_byte, channels):
        super(GroupVectorQuantizer, self).__init__()
        self.channels = channels
        self.fbins = embedding_dim // channels
        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        assert embedding_dim % num_groups == 0, (
            f"Can't group by the setting: embedding_dim={embedding_dim}, "
            f"num_groups={num_groups} "
        )

        self.sub_dim = embedding_dim // num_groups
        self.num_embeddings = 2**vq_byte
        # for each codebook

        self.codebooks = nn.ModuleList()
        for idx in range(self.num_groups):
            self.codebooks.append(
                VectorQuantizerEMA1D(self.sub_dim, self.num_embeddings)
            )

    def encode(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        indices_list = []

        for idx, s in enumerate(torch.chunk(x, self.num_groups, dim=1)):
            indices = self.codebooks[idx].encode(s)
            # [B, T,]
            indices_list.append(indices)

        return indices_list

    def decode(self, indices_list):
        s_q_list = []

        for idx, indices in enumerate(indices_list):
            s_q = self.codebooks[idx].decode(indices)
            s_q_list.append(s_q)

        x_q = torch.cat(s_q_list, dim=1)
        x_q = x_q.reshape(x_q.shape[0], self.channels, self.fbins, x_q.shape[-1])

        return x_q

    def forward(self, x):
        """
        Input shape:[B, C, F, T]
        Output shape:[B, C, F, T], [B, C, F, T](with no gradient)
        """
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])

        s_q_list = []
        s_q_detach_list = []
        for idx, s in enumerate(torch.chunk(x, self.num_groups, dim=1)):
            # [B, sub_dim, T]
            s_q, s_q_detach = self.codebooks[idx](s)
            s_q_list.append(s_q)
            s_q_detach_list.append(s_q_detach)
        x_q = torch.cat(s_q_list, dim=1)
        x_q = x_q.reshape(x_q.shape[0], self.channels, self.fbins, x_q.shape[-1])

        x_q_detach = torch.cat(s_q_detach_list, dim=1)
        x_q_detach = x_q_detach.reshape(
            x_q_detach.shape[0], self.channels, self.fbins, x_q_detach.shape[-1]
        )

        return x_q, x_q_detach


def power_law(x, alpha=0.5):
    """
    Input shape:[B, 2, F, T]
    Output shape:[B, 2, F, T]

    Note that mag_comp is processed mag, not the compressed mag
    """
    real = x[:, 0, :, :]
    imag = x[:, 1, :, :]
    mag = torch.sqrt(real**2 + imag**2 + 1e-8)
    phase = torch.atan2(imag, real)
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
    mag = torch.sqrt(real**2 + imag**2 + 1e-8)
    phase = torch.atan2(imag, real)
    if not inverse:
        mag_comp = torch.log1p(mag)
    else:
        mag_comp = torch.expm1(mag)
    real = mag_comp * torch.cos(phase)
    imag = mag_comp * torch.sin(phase)

    out = torch.stack([real, imag], dim=1)
    return out, mag_comp, mag


class DCARN(nn.Module):
    """
    Vector Quantised Attention-Recurrent Network for Neural Audio Coding.
    Default setting for audio at 44.1kHz, 812k trainable parameters.

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
        # TODO 512->128 or 1024
        rnn_type="LSTM",
        # rnn params
        attn_layers=2,
        n_heads=8,
        # attn params
        # groups=64,
        groups=48,
        bit_per_cbk=10,
        # VQ params
        comp_law="power-law",
        alpha=0.5,
        # power-law factor
    ):
        super(DCARN, self).__init__()

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
        self.conv_encoder = ComplexConvEncoder(
            self.kernel_size, self.kernel_stride, self.kernel_num, True
        )
        self.conv_decoder = ComplexConvDecoder(
            self.kernel_size, self.kernel_stride, self.kernel_num, True
        )

        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.rnn_type = rnn_type
        rnn_input_size = self.fft_len // 2
        for stride in self.kernel_stride:
            rnn_input_size //= stride
        rnn_input_size *= self.kernel_num[-1]
        # print(rnn_input_size)
        self.rnn_encoder = ComplexRNN(
            rnn_input_size, self.rnn_layers, self.rnn_units, self.rnn_type
        )
        self.rnn_decoder = ComplexRNN(
            rnn_input_size, self.rnn_layers, self.rnn_units, self.rnn_type
        )

        self.attn_layers = attn_layers
        self.n_heads = n_heads
        attn_input_size = self.kernel_num[-1]
        self.attn_encoder = ComplexAttention(
            self.attn_layers,
            attn_input_size,
            attn_input_size * 4,
            self.n_heads,
            is_pe=True,
        )
        self.attn_decoder = ComplexAttention(
            self.attn_layers,
            attn_input_size,
            attn_input_size * 4,
            self.n_heads,
            is_pe=True,
        )
        # 4x in FFN

        self.groups = groups
        self.embedding_dim = rnn_input_size
        self.bit_per_cbk = bit_per_cbk
        self.vector_quantizer = GroupVectorQuantizer(
            self.embedding_dim, self.groups, self.bit_per_cbk, self.kernel_num[-1]
        )

        self.comp_law = comp_law
        assert self.comp_law in [
            "power-law",
            "log-law",
        ], "Only support power-law and log-law"
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

        indices = self.vector_quantizer.encode(feature)

        return indices

    def decode(self, indices):
        quantized = self.vector_quantizer.decode(indices)

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
        quantized, quantized_detach = self.vector_quantizer(feature)

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
            "feature": feature,
            "quantized_feature": quantized_detach,
            "encoder_feature": encoder_feature,
            "decoder_feature": decoder_feature,
        }
