# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:29:50 2022

@author: Administrator
"""

# %% cd F:\Github\SSVEP_algorithms\programs

from timeit import timeit
from utils import *
import trca

import matplotlib.pyplot as plt

import scipy.io as io

import mne
from mne.filter import filter_data

from time import perf_counter

# %% 导入数据（60Hz私有数据集）
chan_path = r'G:\SSVEP\Preprocessed Data\SSVEP：60\chan_info.mat'
chan_info = io.loadmat(chan_path)['chan_info'].tolist()
# tar_chan_names = ['O1 ','OZ ','O2 ']

tar_chan_names = ['PZ ','PO5','PO3','POZ','PO4','PO6','O1 ','OZ ','O2 ']

# tar_chan_names = ['PZ ','P1 ','P2 ','P3 ','P4 ','P5 ','P6 ','P7 ','P8 ',
#                   'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
#                   'OZ ','O1 ','O2 ','CB1','CB2']

# tar_chan_names = ['CP1','CP2','CP3','CP4','CP5','CP6','TP7','TP8',
#                   'PZ ','P1 ','P2 ','P3 ','P4 ','P5 ','P6 ','P7 ','P8 ',
#                   'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
#                   'OZ ','O1 ','O2 ','CB1','CB2']

tar_chan_indices = [chan_info.index(name) for name in tar_chan_names]
# tar_chan_indices = [x for x in range(62)]
del chan_path

sub_id = 9
n_points = 500
data_path = r'G:\SSVEP\Preprocessed Data\SSVEP：60\Sub%d.mat' % sub_id
data = io.loadmat(data_path)['data'][:,:,tar_chan_indices,1140:1140+n_points]
del data_path


# %% 导入数据（Benchmark公开数据集）
sub_id = '09'
n_points = 500
data_path = r'G:\SSVEP\Bench\S%s.mat' % sub_id
data = io.loadmat(data_path)['data'].swapaxes(1,3).swapaxes(0,2)
f_data = np.zeros((5,40,6,64,1500))
for nb in range(5):
    for ne in range(40):
        f_data[nb,ne,...] = filter_data(data[ne,...], sfreq=250, l_freq=8*(nb+1), h_freq=90,
                                        l_trans_bandwidth=2, h_trans_bandwidth=2, n_jobs=8)

# %% 创建正余弦参考模板
n_points = 250
sfreq = 250

stim_para = io.loadmat(r'G:\SSVEP\Bench\Freq_Phase.mat')
freqs = stim_para['freqs'].squeeze()
phases = stim_para['phases'].squeeze()
del stim_para

n_harmonics = 5

Y = np.zeros((len(freqs),2*n_harmonics,n_points))
for nf,freq in enumerate(freqs):
    Y[nf,...] = sine_template(freq=freq, phase=phases[nf], n_points=n_points,
                              n_harmonics=n_harmonics, sfreq=sfreq).T


# %% 算法调试模块
Nk = 1
temp = f_data[0]
ed = trca.augmented_events(40, 5)

total_Q = einsum('etcp,ethp->ech', temp,temp)

Xsum = np.sum(temp, axis=1)
total_S = einsum('ecp,ehp->ech', Xsum,Xsum)

w = np.zeros((40,Nk,64))
for ne in range(40):
    idx = str(ne)
    temp_Q = np.sum(total_Q[ed[idx][0]:ed[idx][1],...], axis=0)  # (Nc,Nc)
    temp_S = np.sum(total_S[ed[idx][0]:ed[idx][1],...], axis=0)  # (Nc,Nc)
    w[ne,...] = trca.solve_gep(temp_S, temp_Q, Nk)

# %% 测试分类准确率模块
st = perf_counter()

repeat = 5
acc1 = np.zeros((repeat))
acc2 = np.zeros((repeat))

n_trials = data.shape[1]
n_train = 3
n_test = n_trials - n_train
rand_order = np.arange(n_trials)
for nrep in range(repeat):
    np.random.shuffle(rand_order)
    
    train_data = f_data[0][:,rand_order[:n_train],:,160:160+250]
    test_data = f_data[0][:,rand_order[-n_test:],:,160:160+250]
    
    rou, erou = trca.msetrca(train_data, test_data, d=5)
    _, acc1[nrep] = acc_compute(rou)
    _, acc2[nrep] = acc_compute(erou)

print(acc1.mean())
print(acc2.mean())

et = perf_counter()
print(et-st)

# %%
test_x = f_data[0,:,:,:,160:160+250]
u,v = trca.sctrca_compute_V2(test_x, Y)
# %%
