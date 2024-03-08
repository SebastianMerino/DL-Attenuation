"""
Script to set up training and testing data
"""
import os
import random
import argparse
from pathlib import Path
from modules.dataset import *
from scipy.io import loadmat
import numpy as np

# data_folder = Path(r'C:\Users\sebas\Documents\MATLAB\DataProCiencia\DeepLearning')
# percentage_test = 0.2
parser = argparse.ArgumentParser(description='Sets up training and testing data')
parser.add_argument('data_folder', type=str, help='parent folder where data is stored')
parser.add_argument('percentage', type=int, help='percentage of data for testing')
args = parser.parse_args()
data_folder = Path(args.data_folder)
percentage_test = args.percentage/100

mat_folder = data_folder/'raw'
train_folder = data_folder/'train'
test_folder = data_folder/'test'

(train_folder/'input').mkdir(parents=True, exist_ok=True)
(train_folder/'output').mkdir(parents=True, exist_ok=True)
(test_folder/'input').mkdir(parents=True, exist_ok=True)
(test_folder/'output').mkdir(parents=True, exist_ok=True)

mat_file_list = sorted(os.listdir(mat_folder))
num_testing_files = round(len(mat_file_list) * percentage_test)

# Use random.sample to pick the specified percentage of elements
random.seed(14)
testing_files = random.sample(mat_file_list, num_testing_files)

id = 0
for file in mat_file_list:
    file_path = mat_folder/file
    data_dict = loadmat(file_path)

    x = data_dict['x'].squeeze()
    z = data_dict['z'].squeeze()
    x_offset = x[0]*100
    x = 100*x
    z = 100*z

    rf = data_dict['rf']
    fs = data_dict['fs']
    c0 = data_dict['sos_mean']

    blocksize = 7.5
    freq_L = 2e6; freq_H = 10e6 
    deltaF = 100e3
    overlap_pc      = 0.8
    # (z_ACS,sld) = get_sld(rf, fs, freq_L, freq_H, deltaF, blocksize, overlap_pc)
    (z_ACS,Slocal) = get_spectra(rf, fs, freq_L, freq_H, deltaF, blocksize, overlap_pc)
    logS = 10*np.log10(Slocal)

    alpha_back = data_dict['alpha_mean'][0,0]
    alpha_inc = data_dict['alpha_mean'][0,1]
    cx = data_dict['center_meters'][0,0] * 100 + x_offset
    cz = data_dict['center_meters'][0,1] * 100
    rx = data_dict['radius_meters'][0,0] * 100
    rz = data_dict['radius_meters'][0,1] * 100

    xv, zv = np.meshgrid(x, z_ACS, indexing='xy')
    inc = ((xv - cx)/rx)**2 + ((zv - cz)/rz)**2 < 1

    att_ideal = np.ones(xv.shape)*alpha_back
    att_ideal[inc] = alpha_inc

    if file in testing_files:
        np.save(test_folder/'input'/f'{id:05d}',logS)
        np.save(test_folder/'output'/f'{id:05d}',att_ideal)
    else:
        np.save(train_folder/'input'/f'{id:05d}',logS)
        np.save(train_folder/'output'/f'{id:05d}',att_ideal)
    id += 1
