import sys
import cv2
import torch
import numpy as np
import config
from model import Encode
from bitstring import BitArray

dw = config.downscale_factor
dim = (np.ceil(np.ceil(np.sqrt(config.bit_string_len)) / dw) * dw).astype(int)

# Get torch backend
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get binary file
filename = sys.argv[1]

with open(filename, mode='rb') as file:
    bit_string = BitArray(file)
    bit_string = [int(bit) for bit in bit_string]
    bit_string = torch.tensor(bit_string[:dim ** 2], dtype=torch.float32).to(device)
    bit_string = torch.reshape(bit_string, (1, dim, dim))

    encoder = Encode().to(device)

    checkpoint = torch.load('models/encoder.pth.tar')
    encoder.load_state_dict(checkpoint)

    encoder.eval()

    image = encoder(bit_string)

    image = image.permute((1, 2, 0))
    image = image.detach().cpu().numpy() * 256
    cv2.imwrite("output/encoded.png", image)
