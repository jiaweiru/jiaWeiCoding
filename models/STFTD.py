import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 2), stride=(2, 2), padding=(0, 0)),
                                   nn.InstanceNorm2d(8),
                                   nn.LeakyReLU(0.01))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 2), stride=(2, 2), padding=(1, 0)),
                                   nn.InstanceNorm2d(8),
                                   nn.LeakyReLU(0.01))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 2), stride=(2, 2), padding=(1, 0)),
                                   nn.InstanceNorm2d(8),
                                   nn.LeakyReLU(0.01))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 2), stride=(2, 2), padding=(1, 0)),
                                   nn.InstanceNorm2d(8),
                                   nn.LeakyReLU(0.01))
        self.fc = nn.Conv2d(in_channels=160, out_channels=1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, input):
        fm = []
        real = input[:, :161]
        imag = input[:, 161:]
        input = torch.stack([real, imag], dim=1)
        
        x = self.conv1(input)
        fm.append(x)
        
        x = self.conv2(x)
        fm.append(x)
        
        x = self.conv3(x)
        fm.append(x)
        
        x = self.conv4(x)
        fm.append(x)
        
        x = x.reshape(x.shape[0], -1, 1, x.shape[-1])
        x = self.fc(x)
        
        logits = self.pool(x.reshape(x.shape[0], x.shape[-1])).squeeze(1)
        
        return {"logits": logits, 
                "fm": fm}