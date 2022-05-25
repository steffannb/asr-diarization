import torch
import torchvision


def get_dataloaders(batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/',
                                   train=True,
                                   download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,)
                                       )
                                   ])
                                   ),
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/',
                                   train=False,
                                   download=True,

                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),

        batch_size=batch_size_test,
        shuffle=True,
        pin_memory=True,
    )

    return train_loader, test_loader
