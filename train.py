import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import BitStringDataset, BitStringDatasetGPU
from model import Encode, Decode

# Get torch backend
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(loss_func):
    train_dataset = BitStringDatasetGPU(config.bit_string_len, config.train_dataset_size, device)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        drop_last=True
    )
    encoder = Encode()
    decoder = Decode()

    encoder.to(device)
    decoder.to(device)
    encoder.train()
    decoder.train()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
    )

    loss_list = []

    try:
        for epoch in range(config.epochs):
            total_loss = []
            running_loss = 0.0
            for i, bit_string in enumerate(train_loader):
                # Send data to device
                bit_string = bit_string.to(device)
                # Reset gradients
                optimizer.zero_grad()
                # Forward pass
                encoded = encoder(bit_string)
                decoded = decoder(encoded)
                # Calculating loss
                loss = loss_func(decoded, bit_string)
                # Backward pass
                loss.backward()
                # Optimize the weights
                optimizer.step()

                total_loss.append(loss.item())
                running_loss += loss.item()

                freq = 1000
                if i % freq == freq - 1:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / freq:.3f}')
                    running_loss = 0.0
            loss_list.append(sum(total_loss) / len(total_loss))
            print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
                100. * (epoch + 1) / config.epochs, loss_list[-1]))
    except KeyboardInterrupt:
        test(encoder, decoder, loss_func)
        sys.exit(0)

    return encoder, decoder


def test(encoder, decoder, loss_func):
    test_dataset = BitStringDatasetGPU(config.bit_string_len, config.test_dataset_size, device)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=True,
        drop_last=True
    )

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        total_loss = []
        correct = 0
        for batch_idx, bit_string in enumerate(test_loader):
            bit_string = bit_string.to(device)
            _, _, w, h = bit_string.size()

            encoded = encoder(bit_string)
            decoded = decoder(encoded)

            pred = (decoded > 0.5).float()
            correct += pred.eq(bit_string).all(3).all(2).all(1).sum().item()

            loss = loss_func(pred, bit_string)
            total_loss.append(loss.item())

        print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
            sum(total_loss) / len(total_loss),
            correct / len(test_loader) / config.batch_size * 100)
        )

    torch.save(encoder.state_dict(), './models/encoder.pth.tar')
    torch.save(decoder.state_dict(), './models/decoder.pth.tar')


if __name__ == '__main__':
    loss_func = nn.L1Loss()

    # Training
    model = train(loss_func)

    # Test
    test(*model, loss_func)
