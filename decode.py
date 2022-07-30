import sys
import cv2
import torch
from model import Decode
from bitstring import BitArray

# Get torch backend
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get image file
filename = sys.argv[1]

image = cv2.imread(filename)
image = torch.tensor(image, dtype=torch.float32).to(device)
image = image.permute((2, 0, 1)).unsqueeze(0)
image = image / 256

decoder = Decode().to(device)

checkpoint = torch.load('models/decoder.pth.tar')
decoder.load_state_dict(checkpoint)

decoder.eval()

bit_string = decoder(image)
bit_string = (bit_string > 0.5).int()
bit_string = bit_string.flatten().detach().cpu().numpy().tolist()
bit_string = ''.join([str(b) for b in bit_string])

with open('output/decoded.bin', 'wb') as file:
    bit_string = BitArray(bin=bit_string)
    bit_string.tofile(file)
