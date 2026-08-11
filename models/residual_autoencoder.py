
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1
            ),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.block(x)

        out = out + identity

        out = self.relu(out)

        return out
    



class ResidualAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()


        self.encoder = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            ResidualBlock(32),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            ResidualBlock(64),

            nn.MaxPool2d(2),
        )




        self.bottleneck = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            ResidualBlock(128),
        )    



        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(inplace=True),

            ResidualBlock(64),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=2,
                stride=2
            ),

            nn.ReLU(inplace=True),

            ResidualBlock(32),

            nn.Conv2d(
                32,
                3,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid(),
        )

    
    def forward(self, x):

        x = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(x)

        return x
    

