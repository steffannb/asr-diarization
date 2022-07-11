import scipy.io.wavfile
import torch
from torch.utils.data.dataset import Dataset
import os
import torchaudio
from torchaudio.transforms import MFCC
import numpy as np
import torch.nn.functional as tf


class AudioDataset(Dataset):

    def __init__(self, path_to_audio_files, transform=None):
        self.path_to_audio_files = path_to_audio_files
        self.audio_files = os.listdir(path_to_audio_files)
        self.transform = transform

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_name = os.path.join(self.path_to_audio_files, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(file_name, normalize=True)
        waveform = torch.narrow(waveform, 1, 0, (16000)-1)

        # print(waveform.size())
        # print(sample_rate)
        # print(torch.narrow(waveform, 1, 0, 16000*60).size())

        if self.transform:
            waveform = self.transform(waveform)

        waveform = tf.normalize(waveform, p=10.0, dim=1)

        n_mfcc = 40

        mfcc_transform = MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc
        )

        mfcc = mfcc_transform(waveform)

        squeezed = mfcc.squeeze(dim=1)
        #print(squeezed.size())
        return squeezed


