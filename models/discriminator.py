"""discriminator.py"""

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# Encodec MSTFTD
def get_2d_padding(kernel_size, dilation=(1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


class DiscriminatorSTFT(nn.Module):
    def __init__(
        self, n_fft=1024, hop_length=256, win_length=1024, dilations=[1, 2, 4]
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=True,
            center=True,
            power=None,
        )  # use center=True, not False in Encodec

        self.convs = nn.ModuleList()
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=(3, 9), padding=get_2d_padding((3, 9))),
                nn.LeakyReLU(0.2),
            )
        )
        for d in dilations:
            self.convs.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            32,
                            32,
                            kernel_size=(3, 9),
                            stride=(1, 2),
                            dilation=(d, 1),
                            padding=get_2d_padding((3, 9), (d, 1)),
                        ),
                    ),
                    nn.LeakyReLU(0.2),
                )
            )
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(3, 3), padding=get_2d_padding((3, 3))),
                nn.LeakyReLU(0.2),
            )
        )
        self.conv_post = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(3, 3), padding=get_2d_padding((3, 3))),
        )

    def forward(self, x):
        fmap = []
        z = self.spec_transform(x)
        z = torch.cat([z.real, z.imag], dim=1)
        z = z.permute(0, 1, 3, 2).contiguous()
        for layer in self.convs:
            z = layer(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    def __init__(
        self,
        n_ffts=[1024, 2048, 512],
        hop_lengths=[256, 512, 128],
        win_lengths=[1024, 2048, 512],
    ):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorSTFT(
                    n_fft=n_ffts[i],
                    win_length=win_lengths[i],
                    hop_length=hop_lengths[i],
                )
                for i in range(len(n_ffts))
            ]
        )
        self.num_discriminators = len(self.discriminators)

    def forward(self, x):
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
