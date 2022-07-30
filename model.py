import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from deconv2d import DeConv2d

dw = config.downscale_factor
fc = 9  # Feature channels


class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.conv = nn.Conv2d(1, fc, 5, 1, 2)
        self.downscale = nn.Conv2d(fc, 3, dw, dw)

    def forward(self, x):
        x = self.conv(x)
        x = self.downscale(x)
        return torch.tanh(x)


class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        self.conv = nn.Conv2d(3, fc, 5, 1, 2)
        self.deconv = DeConv2d(fc, 1, (dw, dw), 512)

    def forward(self, x):
        x = self.conv(x)
        x = self.deconv(x)
        return x
