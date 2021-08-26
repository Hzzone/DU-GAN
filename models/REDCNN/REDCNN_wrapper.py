import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=96, num_layers=10, kernel_size=5, padding=0):
        super(Generator, self).__init__()
        encoder = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        ]
        decoder = [
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding)
        ]
        for _ in range(num_layers):
            encoder.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            )
            decoder.append(
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
            )
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)

    def forward(self, x: torch.Tensor):
        residuals = []
        for block in self.encoder:
            residuals.append(x)
            x = F.relu(block(x), inplace=True)
        for residual, block in zip(residuals[::-1], self.decoder[::-1]):
            x = F.relu(block(x) + residual, inplace=True)
        return x
