import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from autoencoder import AE, AEPlusDNN
import os
import torchaudio
import librosa
from torchvision import transforms
from torch import Tensor
from AudioDataset import AudioDataset


def get_dataloaders(batch_size_train, batch_size_test):

    train_dataset = AudioDataset('../audio-new/')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=False
    )

    test_dataset = AudioDataset('../audio-new/')
    # test_dataset = torchvision.datasets.MNIST(
    #     '/files/', train=False, transform=transform, download=True
    # )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size_test, shuffle=False
    )
    return train_loader, test_loader


def main():
    batch_size_train = 32
    batch_size_test = 10
    epochs = 10
    learning_rate = 1e-3
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    train_loader, test_loader = get_dataloaders(batch_size_train, batch_size_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE().to(device)

    # CODE TO SAVE AND LOAD WEIGHTS

    # model = AEPlusDNN().to(device)
    # model.load_state_dict(torch.load('weights'), strict=False)
    # model.eval()
    # model.load_state_dict(torch.load('weights_2'), strict=False)
    # model.eval()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            batch_features = batch_features.to(device)

            optimizer.zero_grad()

            outputs = model(batch_features)

            train_loss = criterion(outputs, batch_features)

            train_loss.backward()

            optimizer.step()

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

            # Create function for melspectogram back to wav

            break

    with torch.no_grad():
        number = 1
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            abc = test_examples[index].cpu().numpy().reshape(40, 80) # 1200
            plt.imshow(librosa.power_to_db(abc), origin="lower", aspect="auto")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            abcd = reconstruction[index].cpu().numpy().reshape(40, 80)
            plt.imshow(librosa.power_to_db(abcd), origin="lower", aspect="auto")
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


if __name__ == '__main__':
    # convertMP3ToWav()
    main()


# def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
#     fig, axs = plt.subplots(1, 1)
#     axs.set_title(title or "Spectrogram (db)")
#     axs.set_ylabel(ylabel)
#     axs.set_xlabel("frame")
#     im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
#     if xmax:
#         axs.set_xlim((0, xmax))
#     fig.colorbar(im, ax=axs)
#     plt.show(block=False)
#
# def toMFCC():
#     waveform, sample_rate = torchaudio.load('../audio/abjxc.wav', normalize=True)
#     # transform = MelSpectrogram(sample_rate)
#     # mel_specgram = transform(waveform)  # (channel, n_mels, time)
#     # return mel_specgram
#     n_fft = 2048
#     win_length = None
#     hop_length = 512
#     n_mels = 256
#     n_mfcc = 256
#
#     mfcc_transform = MFCC(
#         sample_rate=sample_rate,
#         n_mfcc=n_mfcc,
#         melkwargs={
#             "n_fft": n_fft,
#             "n_mels": n_mels,
#             "hop_length": hop_length,
#             "mel_scale": "htk",
#         },
#     )
#
#     mfcc = mfcc_transform(waveform)
#     print(mfcc.squeeze(dim=1)).transpose(1, 2)
#     #plot_spectrogram(mfcc[0])




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