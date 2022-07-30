import numpy as np
import torch
import config
from torch.utils.data import Dataset

dw = config.downscale_factor


class BitStringDataset(Dataset):
    def __init__(self, len_: int, size: float | int):
        self.len = len_
        self.size = int(size)
        self.seed = np.random.randint(0, np.iinfo(np.int64).max - self.size)

    def __getitem__(self, index: int):
        generator = torch.Generator()
        generator.manual_seed(self.seed + index)
        dim = (np.ceil(np.ceil(np.sqrt(self.len)) / dw) * dw).astype(int)
        string = torch.round(torch.rand(self.len, generator=generator))
        padded = torch.cat((string, torch.zeros(dim ** 2 - self.len)))
        return torch.reshape(padded, (1, dim, dim))

    def __len__(self):
        return self.size


class BitStringDatasetGPU(Dataset):
    def __init__(self, len_: int, size: float | int, device):
        self.len = len_
        self.size = int(size)
        self.seed = np.random.randint(0, np.iinfo(np.int64).max - self.size)
        self.device = device
        self.dim = (np.ceil(np.ceil(np.sqrt(self.len)) / dw) * dw).astype(int)
        self.zeros = torch.zeros(self.dim ** 2 - self.len, device=self.device)
        self.data = []
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)
        for _ in range(self.size):
            string = torch.round(torch.rand(self.len, generator=self.generator, device=self.device))
            padded = torch.cat((string, self.zeros))
            self.data.append(torch.reshape(padded, (1, self.dim, self.dim)))

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return self.size
