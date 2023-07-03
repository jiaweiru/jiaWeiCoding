import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvSTFT, ConviSTFT, GroupVectorQuantizer


EPSILON = torch.finfo(torch.float32).eps


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(7, 2), stride=(4, 1), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(7, 2), stride=(4, 1), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 2), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)),
            nn.BatchNorm2d(256),
            nn.PReLU(),
        )
        self.Module_list = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])

    def forward(self, x):
        for layer in self.Module_list:
            x = layer(x)
            x = x[..., :-1]
            # print(x.shape)
        return x


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=(5, 2),
                stride=(2, 1),
                padding=(2, 0),
                output_padding=(1, 0),
            ),
            nn.BatchNorm2d(128),
            nn.PReLU(),
        )
        deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=(5, 2),
                stride=(2, 1),
                padding=(1, 0),
                output_padding=(0, 0),
            ),
            nn.BatchNorm2d(64),
            nn.PReLU(),
        )
        deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=(5, 2),
                stride=(2, 1),
                padding=(2, 0),
                output_padding=(1, 0),
            ),
            nn.BatchNorm2d(32),
            nn.PReLU(),
        )
        deconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                32,
                16,
                kernel_size=(7, 2),
                stride=(4, 1),
                padding=(3, 0),
                output_padding=(3, 0),
            ),
            nn.BatchNorm2d(16),
            nn.PReLU(),
        )
        deconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                16,
                2,
                kernel_size=(7, 2),
                stride=(4, 1),
                padding=(2, 0),
                output_padding=(2, 0),
            ),
            nn.BatchNorm2d(2),
            nn.PReLU(),
        )
        self.Module_list = nn.ModuleList([deconv1, deconv2, deconv3, deconv4, deconv5])

    def forward(self, x):
        for layer in self.Module_list:
            x = layer(x)
            x = x[..., :-1]
            # print(x.shape)
        return x


class GroupedGRU(nn.Module):
    def __init__(self, hidden_size=1024, groups=2):
        super(GroupedGRU, self).__init__()

        hidden_size_t = hidden_size // groups

        self.gru_list1 = nn.ModuleList(
            [
                nn.GRU(hidden_size_t, hidden_size_t, 1, batch_first=True)
                for i in range(groups)
            ]
        )
        self.gru_list2 = nn.ModuleList(
            [
                nn.GRU(hidden_size_t, hidden_size_t, 1, batch_first=True)
                for i in range(groups)
            ]
        )

        self.groups = groups

    def forward(self, x):
        # [B, C, 1, T]
        out = x
        out = out.transpose(1, 3).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack(
            [self.gru_list1[i](out[i])[0] for i in range(self.groups)], dim=-1
        )
        out = torch.flatten(out, start_dim=-2, end_dim=-1)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat(
            [self.gru_list2[i](out[i])[0] for i in range(self.groups)], dim=-1
        )

        out = out.view(out.size(0), out.size(1), -1, x.size(1)).contiguous()
        out = out.transpose(1, 3).contiguous()

        out = out + x
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation, causal
    ):
        super(DepthwiseSeparableConv, self).__init__()

        self.causal = causal
        self.padding = padding if self.causal else padding // 2

        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.act = nn.PReLU()
        self.BN = nn.BatchNorm1d(in_channels)
        self.pointwise_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        if self.causal:
            x = x[..., : -self.padding]
        x = self.act(x)
        x = self.BN(x)
        x = self.pointwise_conv(x)
        return x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, causal):
        super(TemporalBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1
            ),
            nn.PReLU(num_parameters=1),
            nn.BatchNorm1d(num_features=out_channels),
            DepthwiseSeparableConv(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) * dilation,
                dilation=dilation,
                causal=causal,
            ),
        )

    def forward(self, input):
        x = self.net(input)
        return x + input


class TCM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        init_dilation=2,
        num_layers=4,
        causal=True,
    ):  ##num_layers=6
        super(TCM, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = init_dilation**i
            # in_channels = in_channels if i == 0 else out_channels

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    causal=causal,
                )
            ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # [B, C, 1, T]
        x = x.squeeze(dim=2)
        x = self.net(x)
        x = x.unsqueeze(dim=2)
        return x


def power_law(x, alpha=0.5):
    """
    Input shape:[B, 2, F, T]
    Output shape:[B, 2, F, T]
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


class TFNet(nn.Module):
    """
    TFNet backbone in "END-TO-END NEURAL SPEECH CODING FOR REAL-TIME COMMUNICATIONS".
    Due to the lack of details in the paper for the model description, here is only an approximate replication.
    """

    def __init__(
        self,
        win_len=320,
        win_inc=80,
        fft_len=320,
        win_type="hann",
        groups=3,
        bit_per_cbk=10,
        # VQ params
        alpha=0.5,
        # power-law factor
    ):
        super(TFNet, self).__init__()

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

        self.conv_encoder = ConvEncoder()
        self.conv_decoder = ConvDecoder()

        # Temporal filtering
        self.tf_encoder = nn.Sequential(
            TCM(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                init_dilation=2,
                num_layers=4,
            ),
            GroupedGRU(hidden_size=256, groups=2),
            nn.Conv2d(in_channels=256, out_channels=120, kernel_size=(1, 1)),
        )
        self.tf_decoder = nn.Sequential(
            nn.Conv2d(in_channels=120, out_channels=256, kernel_size=(1, 1)),
            TCM(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                init_dilation=2,
                num_layers=4,
            ),
            GroupedGRU(hidden_size=256, groups=2),
            TCM(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                init_dilation=2,
                num_layers=4,
            ),
        )

        # 6kbps setting in paper
        self.groups = groups
        self.embedding_dim = 120
        self.bit_per_cbk = bit_per_cbk
        self.vector_quantizer = GroupVectorQuantizer(
            self.embedding_dim, self.groups, self.bit_per_cbk, 120
        )

        self.alpha = alpha

    def encode(self, x):
        specs = self.stft(x)
        real = specs[:, : self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1 :]
        specs = torch.stack([real, imag], dim=1)

        specs_comp, _, _ = power_law(specs, self.alpha)

        out = self.conv_encoder(specs_comp)

        feature = self.tf_encoder(out)

        indices = self.vector_quantizer.encode(feature)

        return indices

    def decode(self, indices):
        quantized = self.vector_quantizer.decode(indices)

        out = self.tf_decoder(quantized)

        est_specs_comp = self.conv_decoder(out)

        est_specs, _, _ = power_law(est_specs_comp, 1.0 / self.alpha)

        out_wav = self.istft(
            est_specs.reshape(est_specs.shape[0], -1, est_specs.shape[-1])
        )
        out_wav = torch.clamp_(out_wav, -1, 1)

        return out_wav

    def forward(self, x):
        """
        Input shape:[B, 1, waveforms]
        Output shape:[B, 1, waveforms]
        """
        specs = self.stft(x)
        real = specs[:, : self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1 :]
        specs = torch.stack([real, imag], dim=1)

        specs_comp, mags_comp, _ = power_law(specs, self.alpha)

        out = self.conv_encoder(specs_comp)

        feature = self.tf_encoder(out)

        quantized, vq_input, vq_output_detach = self.vector_quantizer(feature)

        out = self.tf_decoder(quantized)

        est_specs_comp = self.conv_decoder(out)

        est_specs, _, est_mags_comp = power_law(est_specs_comp, 1.0 / self.alpha)

        out_wav = self.istft(
            est_specs.reshape(est_specs.shape[0], -1, est_specs.shape[-1])
        )
        out_wav = torch.clamp_(out_wav, -1, 1)

        # for consistency constraints
        re_specs = self.stft(out_wav)
        real = re_specs[:, : self.fft_len // 2 + 1]
        imag = re_specs[:, self.fft_len // 2 + 1 :]
        re_specs = torch.stack([real, imag], dim=1)

        est_specs_comp, est_mags_comp, _ = power_law(re_specs, self.alpha)

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
        }
