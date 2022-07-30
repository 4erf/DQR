import torch
from torch import nn


class DeConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: (int, int)):
        super(DeConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        oC, iC, (kH, kW) = self.out_channels, self.in_channels, self.kernel_size
        self.weights = nn.init.uniform_(torch.zeros((oC, iC, kH, kW)), -1, 1)
        self.biases = torch.zeros((oC, iC, kH, kW))
        self.weights = nn.Parameter(self.weights.requires_grad_(True))
        self.biases = nn.Parameter(self.biases.requires_grad_(True))

    def _apply(self, fn):
        # Auto send tensors to model device
        super(DeConv2d, self)._apply(fn)
        self.weights = fn(self.weights)
        self.biases = fn(self.biases)
        return self

    def reshape_param(self, param, n, iH, iW):
        # shape: (oC, iC, kH, kW)
        repeated = param.repeat(1, 1, iH, iW)
        # shape: (oC, iC, oH, oW)
        expanded = repeated.unsqueeze(0).expand(n, -1, -1, -1, -1)
        return expanded

    def forward(self, batches):
        n, _, iH, iW = batches.size()
        oC, iC, (kH, kW) = self.out_channels, self.in_channels, self.kernel_size
        oH, oW = kH * iH, kW * iW

        # Reshape inputs
        # shape: (n, iC, iH, iW)
        repeated = batches.repeat_interleave(kW, 3, output_size=oW).repeat_interleave(kH, 2, output_size=oH)
        # shape: (n, iC, oH, oW)
        expanded = repeated.unsqueeze(1).expand(-1, oC, -1, -1, -1)
        # shape: (n, oC, iC, oH, oW)

        # Reshape weights & biases
        weights = self.reshape_param(self.weights, n, iH, iW)
        biases = self.reshape_param(self.biases, n, iH, iW)

        # Apply deconvolution
        result = expanded * weights + biases
        summed = result.sum(2)

        return summed
