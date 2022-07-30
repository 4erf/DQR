import torch.nn as nn
import torch.nn.functional as F
import config
from deconv2d import DeConv2d

dw = config.downscale_factor

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, dw, dw)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x)


class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        self.deconv = DeConv2d(3, 1, (dw, dw))

    def forward(self, x):
        x = self.deconv(x)
        return nn.Sigmoid()(x)
