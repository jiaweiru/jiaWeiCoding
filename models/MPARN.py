import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (VectorQuantizerEMA1D, ConvSTFT, ConviSTFT, MultiHeadAttentionEncoder)


EPSILON = torch.finfo(torch.float32).eps

class ConvEncoder(nn.Module):
    """
    MagEncoder
    """

    def __init__(
            self,
            kernel_size,
            kernel_stride,
            kernel_num,
            first_channel
    ):
        super(ConvEncoder, self).__init__()

        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_num = (first_channel,) + kernel_num

        self.num_layers = len(self.kernel_size)

        self.module_list = nn.ModuleList()
        for idx in range(self.num_layers):
            if idx == 0:
                self.module_list.append(
                    nn.Sequential(
                        nn.ConstantPad2d((0, 0, 1, 0), value=0.),
                        nn.Conv2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx + 1],
                            kernel_size=(self.kernel_size[idx], 2),
                            stride=(self.kernel_stride[idx], 1),
                            padding=(self.kernel_size[idx] // 2 - 1, 1)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx + 1]),
                        nn.PReLU()
                    )
                )
            else:
                self.module_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx + 1],
                            kernel_size=(self.kernel_size[idx], 2),
                            stride=(self.kernel_stride[idx], 1),
                            padding=(self.kernel_size[idx] // 2, 1)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx + 1]),
                        nn.PReLU()
                    )
                )

    def forward(self, x):
        """
        Input shape:[B, C, F, T]
        Output shape:[B, C', F', T]
        """
        out = x

        for layer in self.module_list:
            out = layer(out)
            out = out[..., :-1]
            # print(out.shape)

        return out


class ConvDecoder(nn.Module):
    """
    TransConv layers in decoder
    """

    def __init__(
            self,
            kernel_size,
            kernel_stride,
            kernel_num,
            first_channel
    ):
        super(ConvDecoder, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.kernel_num = (first_channel,) + kernel_num

        self.num_layers = len(self.kernel_size)

        self.module_list = nn.ModuleList()
        for idx in range(self.num_layers, 0, -1):
            # idx from num_layers to 1
            if idx != 1:
                self.module_list.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            self.kernel_num[idx],
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size[idx - 1], 2),
                            stride=(self.kernel_stride[idx - 1], 1),
                            padding=(self.kernel_size[idx - 1] // 2, 0),
                            output_padding=(self.kernel_stride[idx - 1] - 1, 0)
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
                            kernel_size=(self.kernel_size[idx - 1], 2),
                            stride=(self.kernel_stride[idx - 1], 1),
                            padding=(self.kernel_size[idx - 1] // 2 - 1, 0),
                            output_padding=(self.kernel_stride[idx - 1] - 1, 0)
                        ),
                    )
                )

    def forward(self, x):
        """
        Input shape:[B, C', F', T]
        Output shape:[B, C, F, T]
        """
        out = x

        for idx, layer in enumerate(self.module_list):
            out = layer(out)
            out = out[..., :-1]
            if idx == self.num_layers - 1:
                out = out[:, :, :-1, :]
            # print(out.shape)[]

        return out
    

class TemporalRNN(nn.Module):
    """
    LSTM or GRU for temporal filtering
    Causal version
    """

    def __init__(
            self,
            input_size,
            rnn_layers,
            rnn_units,
            rnn_type,
    ):
        super(TemporalRNN, self).__init__()

        self.input_size = input_size
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.bidirectional = False

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.rnn_units, self.rnn_layers, batch_first=True, bidirectional=self.bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.rnn_units, self.rnn_layers, batch_first=True, bidirectional=self.bidirectional)

        # Project layer
        self.proj = nn.Linear(self.rnn_units * 2 if self.bidirectional else self.rnn_units, self.input_size)

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

        out = out + x

        return out


class FrequencyAttention(nn.Module):
    """
    MHSA+FFN encoder
    """

    def __init__(
            self,
            num_layers,
            input_size,
            ff_size,
            n_heads,
            is_pe
    ):
        super(FrequencyAttention, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.ff_size = ff_size
        self.n_heads = n_heads
        self.is_pe = is_pe

        attn_list = []
        for idx in range(self.num_layers):
            attn_list.append(MultiHeadAttentionEncoder(self.input_size, self.ff_size, self.n_heads, self.is_pe))
        self.module = nn.Sequential(*attn_list)

    def forward(self, x):
        """
        Input shape:[B, C, F, T]
        Output shape:[B, C, F, T]
        """
        out = self.module(x)
        out = out + x

        return out


class GroupVectorQuantizer(nn.Module):
    """
    Group VectorQuantizer by using VectorQuantizerEMA1D.

    Flatten the features of each frame, then group these features and perform vector quantization.
    """

    def __init__(
            self,
            embedding_dim,
            num_groups,
            vq_byte,
            channels

    ):
        super(GroupVectorQuantizer, self).__init__()
        self.channels = channels
        self.fbins = embedding_dim // channels
        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        assert embedding_dim % num_groups == 0, f'Can\'t group by the setting: embedding_dim={embedding_dim}, ' \
                                                f'num_groups={num_groups} '

        self.sub_dim = embedding_dim // num_groups
        self.num_embeddings = 2 ** vq_byte
        # for each codebook

        self.codebooks = nn.ModuleList()
        for idx in range(self.num_groups):
            self.codebooks.append(VectorQuantizerEMA1D(self.sub_dim, self.num_embeddings))

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
        x_q_detach = x_q_detach.reshape(x_q_detach.shape[0], self.channels, self.fbins, x_q_detach.shape[-1])

        return x_q, x_q_detach


def power_law(x, alpha=0.5):
    """
    Input shape:[B, 2, F, T]
    Output shape:[B, 2, F, T]

    Note that mag_comp is processed mag, not the compressed mag
    """
    real = x[:, 0, :, :]
    imag = x[:, 1, :, :]
    mag = torch.sqrt(real ** 2 + imag ** 2 + EPSILON)
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
    mag = torch.sqrt(real ** 2 + imag ** 2 + EPSILON)
    phase = torch.atan2(imag + EPSILON, real)
    if not inverse:
        mag_comp = torch.log1p(mag)
    else:
        mag_comp = torch.expm1(mag)
    real = mag_comp * torch.cos(phase)
    imag = mag_comp * torch.sin(phase)

    out = torch.stack([real, imag], dim=1)
    return out, mag_comp, mag


class MPARN(nn.Module):
    """
    Magnitude Phase Attention-Recurrent Network for Neural Audio Coding (Real-valued version).

    The bitrate can be adjusted by simply adjusting the number of groups and the number of bits per codebook.
    """

    def __init__(
            self,
            win_len=1764,
            win_inc=441,
            fft_len=2048,
            win_type='hann',
            # fft params
            # groups=64,
            mag_groups=48,
            phase_groups=48,
            bit_per_cbk=10,
            # VQ params
            comp_law='power-law',
            alpha=0.5,
            # power-law factor
    ):
        super(MPARN, self).__init__()

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex')
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, 'complex')
        
        # Mag encoder and decoder params
        self.mag_kernel_size = (7, 7, 7, 7)
        self.mag_kernel_stride = (1, 4, 4, 4)
        self.mag_kernel_num = (16, 32, 64, 96)
        
        self.mag_rnn_layers = 2
        self.mag_rnn_units = 64
        self.mag_rnn_type = 'LSTM'
        mag_rnn_input_size = self.mag_kernel_num[-1]
        
        self.mag_attn_layers = 2
        self.mag_n_heads = 8
        mag_attn_input_size = self.mag_kernel_num[-1]
        
        self.mag_encoder = nn.Sequential(
            ConvEncoder(self.mag_kernel_size, self.mag_kernel_stride, self.mag_kernel_num, 1),
            # FrequencyAttention(self.mag_attn_layers, mag_attn_input_size, mag_attn_input_size * 4, self.mag_n_heads, is_pe=True),
            # TemporalRNN(mag_rnn_input_size, self.mag_rnn_layers, self.mag_rnn_units, self.mag_rnn_type)
        )
        self.mag_decoder = nn.Sequential(
            # TemporalRNN(mag_rnn_input_size, self.mag_rnn_layers, self.mag_rnn_units, self.mag_rnn_type),
            # FrequencyAttention(self.mag_attn_layers, mag_attn_input_size, mag_attn_input_size * 4, self.mag_n_heads, is_pe=True),
            ConvDecoder(self.mag_kernel_size, self.mag_kernel_stride, self.mag_kernel_num, 1)
        )

        self.mag_embedding_dim = self.fft_len // 2
        for stride in self.mag_kernel_stride:
            self.mag_embedding_dim //= stride
        self.mag_embedding_dim *= self.mag_kernel_num[-1]
        
        # Phase encoder and decoder params
        self.phase_kernel_size = (7, 7, 7, 7)
        self.phase_kernel_stride = (1, 4, 4, 4)
        self.phase_kernel_num = (16, 32, 64, 96)
        
        self.phase_rnn_layers = 2
        self.phase_rnn_units = 64
        self.phase_rnn_type = 'LSTM'
        phase_rnn_input_size = self.phase_kernel_num[-1]
        
        self.phase_attn_layers = 2
        self.phase_n_heads = 8
        phase_attn_input_size = self.phase_kernel_num[-1]
        
        self.phase_encoder = nn.Sequential(
            ConvEncoder(self.phase_kernel_size, self.phase_kernel_stride, self.phase_kernel_num, 2),
            FrequencyAttention(self.phase_attn_layers, phase_attn_input_size, phase_attn_input_size * 4, self.phase_n_heads, is_pe=True),
            TemporalRNN(phase_rnn_input_size, self.phase_rnn_layers, self.phase_rnn_units, self.phase_rnn_type)
        )
        self.phase_decoder = nn.Sequential(
            TemporalRNN(phase_rnn_input_size, self.phase_rnn_layers, self.phase_rnn_units, self.phase_rnn_type),
            FrequencyAttention(self.phase_attn_layers, phase_attn_input_size, phase_attn_input_size * 4, self.phase_n_heads, is_pe=True),
            ConvDecoder(self.phase_kernel_size, self.phase_kernel_stride, self.phase_kernel_num, 2)
        )

        self.phase_embedding_dim = self.fft_len // 2
        for stride in self.phase_kernel_stride:
            self.phase_embedding_dim //= stride
        self.phase_embedding_dim *= self.phase_kernel_num[-1]
        
        self.mag_groups = mag_groups
        self.phase_groups = phase_groups
        self.bit_per_cbk = bit_per_cbk
        self.mag_vector_quantizer = GroupVectorQuantizer(self.mag_embedding_dim, self.mag_groups, self.bit_per_cbk, self.mag_kernel_num[-1])
        self.phase_vector_quantizer = GroupVectorQuantizer(self.phase_embedding_dim, self.phase_groups, self.bit_per_cbk, self.phase_kernel_num[-1])

        self.comp_law = comp_law
        assert self.comp_law in ['power-law', 'log-law'], "Only support power-law and log-law"
        if self.comp_law == 'power-law':
            self.alpha = alpha
        elif self.comp_law == 'log-law':
            pass
    
    def encode(self, x):

        specs = self.stft(x)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        specs = torch.stack([real, imag], dim=1)

        if self.comp_law == 'power-law':
            specs_comp, mags_comp, _ = power_law(specs, self.alpha)
        elif self.comp_law == 'log-law':
            specs_comp, mags_comp, _ = log_law(specs)
        else:
            specs_comp, mags_comp = None, None
        
        mags_feature = self.mag_encoder(mags_comp.unsqueeze(1))
        phases_feature = self.phase_encoder(specs_comp)

        mag_indices = self.mag_vector_quantizer.encode(mags_feature)
        phase_indices = self.phase_vector_quantizer.encode(phases_feature)
        
        return mag_indices + phase_indices
    
    def decode(self, indices):
        
        mag_indices = indices[:self.mag_groups]
        phase_indices = indices[-self.phase_groups:]
        
        mags_quantized = self.mag_vector_quantizer.decode(mag_indices)
        phases_quantized = self.phase_vector_quantizer.decode(phase_indices)

        est_mags_comp = self.mag_decoder(mags_quantized)
        est_phases = self.phase_decoder(phases_quantized)
        est_phases = est_phases / (torch.sqrt(torch.abs(est_phases[:, 0]) ** 2 + torch.abs(est_phases[:, 1]) ** 2 + EPSILON) + EPSILON).unsqueeze(1)
        
        est_specs_comp = est_mags_comp * est_phases

        if self.comp_law == 'power-law':
            est_specs, _, _ = power_law(est_specs_comp, 1. / self.alpha)
        elif self.comp_law == 'log-law':
            est_specs, _, _ = log_law(est_specs_comp, inverse=True)
        else:
            est_specs = None

        out_wav = self.istft(est_specs.reshape(est_specs.shape[0], -1, est_specs.shape[-1]))
        out_wav = torch.clamp_(out_wav, -1, 1)

        return out_wav

    def forward(self, x):
        """
        Input shape:[B, 1, waveforms]
        Output shape:[B, 1, waveforms]
        The shape of tensor keep [B, (C), F, T] in forward flow.
        """

        specs = self.stft(x)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        specs = torch.stack([real, imag], dim=1)

        if self.comp_law == 'power-law':
            specs_comp, mags_comp, _ = power_law(specs, self.alpha)
        elif self.comp_law == 'log-law':
            specs_comp, mags_comp, _ = log_law(specs)
        else:
            specs_comp, mags_comp = None, None

        mags_feature = self.mag_encoder(mags_comp.unsqueeze(1))
        mags_quantized, mags_quantized_detach = self.mag_vector_quantizer(mags_feature)
        est_mags_comp = self.mag_decoder(mags_quantized)
        
        phases_feature = self.phase_encoder(specs_comp)
        phases_quantized, phases_quantized_detach = self.phase_vector_quantizer(phases_feature)
        est_phases = self.phase_decoder(phases_quantized)
        est_phases = est_phases / (torch.sqrt(torch.abs(est_phases[:, 0]) ** 2 + torch.abs(est_phases[:, 1]) ** 2 + EPSILON) + EPSILON).unsqueeze(1)

        est_specs_comp = est_mags_comp * est_phases
        
        if self.comp_law == 'power-law':
            est_specs, _, est_mags_comp = power_law(est_specs_comp, 1. / self.alpha)
        elif self.comp_law == 'log-law':
            est_specs, _, est_mags_comp = log_law(est_specs_comp, inverse=True)
        else:
            est_specs, est_mags_comp = None, None

        out_wav = self.istft(est_specs.reshape(est_specs.shape[0], -1, est_specs.shape[-1]))
        out_wav = torch.clamp_(out_wav, -1, 1)

        # for consistency constraints
        re_specs = self.stft(out_wav)
        real = re_specs[:, :self.fft_len // 2 + 1]
        imag = re_specs[:, self.fft_len // 2 + 1:]
        re_specs = torch.stack([real, imag], dim=1)

        if self.comp_law == 'power-law':
            est_specs_comp, est_mags_comp, _ = power_law(re_specs, self.alpha)
        elif self.comp_law == 'log-law':
            est_specs_comp, est_mags_comp, _ = log_law(re_specs)
        else:
            est_specs_comp, est_mags_comp = None, None

        est_specs_comp = torch.cat((est_specs_comp[:, 0, :, :], est_specs_comp[:, 1, :, :]), dim=1)
        specs_comp = torch.cat((specs_comp[:, 0, :, :], specs_comp[:, 1, :, :]), dim=1)
        
        return {"specs_comp": specs_comp, "est_specs_comp": est_specs_comp, "mags_comp": mags_comp, "est_mags_comp": est_mags_comp, "est_wav": out_wav, "raw_wav": x, "mags_feature": mags_feature, "mags_quantized_feature": mags_quantized_detach,  "phases_feature": phases_feature, "phases_quantized_feature": phases_quantized_detach}
