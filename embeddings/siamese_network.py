from torch import nn
import torch.nn.functional as f


# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.t_conv1(x))
        x = f.sigmoid(self.t_conv2(x))

        return x


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.network = None
        self.define_network()

    def define_network(self):
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            nn.Dropout2d(p=0.2),

            nn.Flatten(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10),

            nn.Softmax(),
        )

    def forward(self, x):
        return self.network(x)
