import os
import numpy as np
import scipy as sp
import torch
from torch.utils.data import Dataset
from torchvision import transforms
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
        data_folder = Path(data_folder)
        self.input_folder = data_folder/'input'
        self.output_folder = data_folder/'output'
        self.data_file_list = sorted(os.listdir(self.input_folder))

    def __len__(self):
        return len(self.data_file_list)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.data_file_list[idx])
        output_path = os.path.join(self.output_folder, self.data_file_list[idx])
        sld = np.load(input_path)
        att_ideal = np.expand_dims(np.load(output_path),axis=0)

        input_transforms = transforms.Compose([
            torch.Tensor,
            transforms.Normalize(0,1.5)
        ])

        output_transforms = transforms.Compose([
            torch.Tensor,
            transforms.Normalize(1,1)
        ])

        x = input_transforms(sld)
        y = output_transforms(att_ideal)

        return x,y