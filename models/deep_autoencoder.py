
import torch
import torch.nn as nn



class DeepAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()
        

        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
               in_channels=32,
               out_channels=64,
               kernel_size=3,
               padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
               in_channels=64,
               out_channels=64,
               kernel_size=3,
               padding=1
            ),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )
        

        self.bottleneck = nn.Sequential(

            nn.Conv2d(
               in_channels=64,
               out_channels=128,
               kernel_size=3,
               padding=1
            ), 

            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
               in_channels=128,
               out_channels=64,
               kernel_size=2,
               stride=2
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
               in_channels=64,
               out_channels=64,
               kernel_size=3,
               padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
               in_channels=64,
               out_channels=64,
               kernel_size=3,
               padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
               in_channels=64,
               out_channels=32,
               kernel_size=2,
               stride=2
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
               in_channels=32,
               out_channels=32,
               kernel_size=3,
               padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
               in_channels=32,
               out_channels=32,
               kernel_size=3,
               padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
               in_channels=32,
               out_channels=3,
               kernel_size=3,
               padding=1
            ),

            nn.Sigmoid()
        )
    


    def forward(self, x):

        x = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(x)

        return x








        




















      







