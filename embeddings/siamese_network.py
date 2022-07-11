from torch import nn
import torch.nn.functional as f
import torch


# Define the Convolutional Autoencoder

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.network = None
        self.define_network()

    def define_network(self):
        self.network = nn.Sequential(
            nn.Linear()
        )

    def forward(self, x):
        return self.network(x)
