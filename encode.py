import sys
import cv2
import torch
import numpy as np
import config
from model import Encode
from bitstring import BitArray

dw = config.downscale_factor

# Get torch backend
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get binary file
filename = sys.argv[1]

with open(filename, mode='rb') as file:
    bit_string = BitArray(file)
    bit_string = [int(bit) for bit in bit_string]
    bit_string = torch.tensor(bit_string, dtype=torch.float32, device=device)

    dim = (np.ceil(np.ceil(np.sqrt(len(bit_string))) / dw) * dw).astype(int)
    zeros = torch.zeros(dim ** 2 - len(bit_string), device=device)
    padded = torch.cat((bit_string, zeros))

    bit_string = torch.reshape(padded, (1, dim, dim))

    encoder = Encode().to(device)

    checkpoint = torch.load('models/encoder.pth.tar')
    encoder.load_state_dict(checkpoint)

    encoder.eval()

    image = encoder(bit_string)

    image = image.permute((1, 2, 0))
    image = image.detach().cpu().numpy() * 128 + 128
    cv2.imwrite("output/encoded.png", image)
