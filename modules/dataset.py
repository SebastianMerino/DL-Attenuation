import os
import numpy as np
import scipy as sp
import torch
from torch.utils.data import Dataset
from pathlib import Path


def get_sld(rf, fs, freq_L, freq_H, deltaF, blocksize, overlap_pc):
    freq_C = (freq_L+freq_H)/2
    # Wavelength size
    c0 = 1540
    wl = c0/freq_C
    dz = 1/fs*c0/2
    
    nFFT = int(fs/deltaF)
    
    wz = int(blocksize*wl*(1-overlap_pc)/dz ); # Between windows
    nz = 2*int(blocksize*wl/dz /2 ); # Window size

    f,t,spgram = sp.signal.spectrogram(rf, fs=fs, window=('tukey', 0.25), nperseg=nz, noverlap=nz-wz, nfft=nFFT, axis=0)
    spgram.shape
    rang = np.squeeze((f > freq_L) & (f < freq_H))
    Slocal = spgram[rang,:,:]
    Slocal = np.moveaxis(Slocal, [1, 2], [2, 1])
    Slocal.shape
    z_ACS = c0*t[:-1]/2 *100
    sld = -np.diff(np.log(Slocal), axis=1)
    return z_ACS,sld

class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.input_folder = Path(data_folder)/'input'
        self.output_folder = Path(data_folder)/'output'
        # self.transform_input = transforms.Normalize([0,0],[0.1172,0.1172])
        # self.transform_output = transforms.Lambda(lambda t: (t * 2) - 1)
        self.input_file_list = sorted(os.listdir(self.input_folder))
        self.output_file_list = sorted(os.listdir(self.output_folder))

    def __len__(self):
        return len(self.input_file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.input_folder, self.input_file_list[idx])
        x = np.load(file_path)
        x = torch.Tensor(x)

        file_path = os.path.join(self.output_folder, self.output_file_list[idx])
        y = np.load(file_path)
        y = torch.Tensor(y)
        y = y.unsqueeze(0)

        # x = self.transform_input(x)
        # y = self.transform_output(y)
        return x, y