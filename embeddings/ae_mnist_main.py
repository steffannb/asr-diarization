import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from autoencoder import AE, AEPlusDNN
import os



def get_dataloaders(batch_size_train, batch_size_test):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        '/files/', train=True, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True
    )

    test_dataset = torchvision.datasets.MNIST(
        '/files/', train=False, transform=transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False
    )
    return train_loader, test_loader


def main():
    batch_size_train = 512
    batch_size_test = 10
    epochs = 2
    learning_rate = 1e-3
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    train_loader, test_loader = get_dataloaders(batch_size_train, batch_size_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AE().to(device)
    # model = AEPlusDNN().to(device)
    # model.load_state_dict(torch.load('weights'), strict=False)
    # model.eval()
    # model.load_state_dict(torch.load('weights_2'), strict=False)
    # model.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            # batch_features = batch_features.view(-1, 784).to(device)
            batch_features = batch_features.to(device)
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

    # torch.save(model.state_dict(), 'weights_3')

    # exit(0)
    test_examples = None

    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features[0]
            # test_examples = batch_features.view(-1, 784).to(device)
            test_examples = batch_features.to(device)
            reconstruction = model(test_examples)
            break

    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].cpu().numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].cpu().numpy().reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

if __name__ == '__main__':
    main()


# model_pretrained = AE().to(device)
#     model_pretrained.load_state_dict(torch.load('weights'))
#     model_pretrained.eval()
#
#     model = AEPlusDNN().to(device)
#     pretrained_dict = model_pretrained.state_dict()
#     model_dict = model.state_dict()
#
#     # 1. filter out unnecessary keys
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dict)
#     # 3. load the new state dict
#     model.load_state_dict(model_dict)
#
#     model.eval()