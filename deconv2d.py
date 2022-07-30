import torch
from torch import nn


class DeConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: (int, int)):
        super(DeConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        oC, iC, (kH, kW) = self.out_channels, self.in_channels, self.kernel_size
        self.inverse_kernels = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 2 ** (kH * kW)),
            nn.ReLU(),
            nn.Linear(2 ** (kH * kW), 2 ** (kH * kW)),
            nn.ReLU(),
            nn.Linear(2 ** (kH * kW), kH * kW),
        ) for _ in range(oC * iC)])

    def forward(self, batches):
        n, _, iH, iW = batches.size()
        oC, iC, (kH, kW) = self.out_channels, self.in_channels, self.kernel_size
        oH, oW = kH * iH, kW * iW

        # shape: (n, iC, iH, iW)
        in_channels = batches.transpose(0, 1)
        # shape: (iC, n, iH, iW)

        out_channels = []  # shape: [oC](n, oH, oW)
        for i in range(oC):
            outputs = []  # shape: [iC](n, oH, oW)
            for j, channel in enumerate(in_channels):
                # shape: (n, iH, iW)
                linear_in = channel.view(-1, 1)
                # shape: (n * iH * iW, 1)
                linear_out = self.inverse_kernels[i * oC + j](linear_in)
                # shape: (n * iH * iW, kH * kW)
                out_squared = linear_out.view(-1, kH, kW)
                # shape: (n * iH * iW, kH, kW)
                aligned = out_squared.view(-1, iH, iW, kH, kW)
                # shape: (n, iH, iW, kH, kW)
                output = aligned.transpose(2, 3).reshape(-1, oH, oW)
                # shape: (n, oH, oW)
                outputs.append(output)

            # Aggregate outputs into a single out channel
            out_channel = torch.sum(torch.stack(outputs), dim=0)
            out_channels.append(out_channel)

        # Obtain final output
        result = torch.stack(out_channels)
        # shape: (oC, n, oH, oW)
        result = result.transpose(0, 1)
        # shape: (n, oC, oH, oW)

        return result
