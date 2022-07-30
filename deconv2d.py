import torch
from torch import nn


class DeConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: (int, int), hidden_size: int):
        super(DeConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        oC, iC, (kH, kW) = self.out_channels, self.in_channels, self.kernel_size
        self.inverse_kernels = nn.ModuleList([nn.Sequential(
            nn.Linear(iC, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, kH * kW),
        ) for _ in range(oC)])

    def forward(self, batches):
        n, _, iH, iW = batches.size()
        oC, iC, (kH, kW) = self.out_channels, self.in_channels, self.kernel_size
        oH, oW = kH * iH, kW * iW

        # shape: (n, iC, iH, iW)
        in_channels = batches.transpose(1, 2).transpose(2, 3)
        # shape: (n, iH, iW, iC)

        out_channels = []  # shape: [oC](n, oH, oW)
        for i in range(oC):
            # shape: (n, iH, iW, iC)
            linear_in = in_channels.reshape(-1, iC)
            # shape: (n * iH * iW, iC)
            linear_out = self.inverse_kernels[i](linear_in)
            # shape: (n * iH * iW, kH * kW)
            aligned = linear_out.view(n, iH, iW, kH, kW)
            # shape: (n, iH, iW, kH, kW)
            out_channel = aligned.transpose(2, 3).reshape(n, oH, oW)
            # shape: (n, oH, oW)
            out_channels.append(out_channel)

        # Obtain final output
        result = torch.stack(out_channels)
        # shape: (oC, n, oH, oW)
        result = result.transpose(0, 1)
        # shape: (n, oC, oH, oW)

        return result
