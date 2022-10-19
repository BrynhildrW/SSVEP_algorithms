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
ratio1 = np.zeros((17,2,9,9))
ratio2 = np.zeros((17,2,9,9))
ratio3 = np.zeros((17,2,9,9))

for npeo in range(17):
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
    
    n_points = 1000
    data_path = r'G:\SSVEP\Preprocessed Data\SSVEP：60\Sub%d.mat' % (npeo+1)
    data = io.loadmat(data_path)['normal'][:,:,tar_chan_indices,140:1140+n_points]
    del data_path
    
    rest = data[...,:1000]
    task = data[...,1000:]
    
    cov_task = np.einsum('etcp,ethp->etch', task, task)
    cov_rest_task = np.einsum('etcp,ethp->etch', rest, task)
    
    cov_mean_task = np.einsum('ecp,ehp->ech', task.mean(axis=1), task.mean(axis=1))
    cov_mean_rest_task = np.einsum('ecp,ehp->ech', rest.mean(axis=1), task.mean(axis=1))
    
    # mean of covariance
    ratio1[npeo] = cov_task.mean(axis=1) / cov_rest_task.mean(axis=1)
    
    # covariance of mean
    ratio2[npeo] = cov_mean_task / cov_mean_rest_task
    
    # mean of ratio
    ratio3[npeo] = np.mean(cov_task / cov_rest_task, axis=1)
    
    print('finish sub' + str(npeo+1))
    


# %% 修正数据集（Benchmark公开数据集）
# indices = ['0' + str(i+1) for i in range(9)]
# indices += [str(i+10) for i in range(26)]

# for npeo,idx in enumerate(indices):
#     data_path = r'G:\SSVEP\Bench\S%s.mat' % idx
#     data = io.loadmat(data_path)['data'].swapaxes(1,3).swapaxes(0,2)  # (Ne,Nt,Nc,Np)
#     f_data = np.zeros((5,40,6,64,1500))
#     for nb in range(5):
#         for ne in range(40):
#             f_data[nb,ne,...] = filter_data(data[ne,...], sfreq=250, l_freq=8*(nb+1), h_freq=90,
#                                         l_trans_bandwidth=2, h_trans_bandwidth=2, n_jobs=8)
#     save_path = r'G:\SSVEP\Preprocessed Data\Benchmark\S%s.mat' % idx
#     io.savemat(save_path, {'data':f_data,
#                            'chan_info':chan_info})


# %% 导入数据（Benchmark公开数据集）
sub_id = '09'
n_points = 500
data_path = r'G:\SSVEP\Preprocessed Data\Benchmark\S%s.mat' % sub_id
data = io.loadmat(data_path)['data']
chan_info = io.loadmat(data_path)['chan_info'].tolist()


# %% 创建正余弦参考模板
n_points = 500
sfreq = 1000

# stim_para = io.loadmat(r'G:\SSVEP\Bench\Freq_Phase.mat')
# freqs = stim_para['freqs'].squeeze()
freqs = [60,60]
# phases = stim_para['phases'].squeeze()
phases = [0,1]
# del stim_para

n_harmonics = 1

Y = np.zeros((len(freqs),2*n_harmonics,n_points))
for nf,freq in enumerate(freqs):
    Y[nf,...] = sine_template(freq=freq, phase=phases[nf], n_points=n_points,
                              n_harmonics=n_harmonics, sfreq=sfreq).T


# %% 算法调试模块
X = data[:,:80,...]
n_train = 80
n_chans = 9
n_events = 2

Xmean = X.mean(axis=1)  # (Ne,Nc,Np)
Xhat = np.concatenate((Xmean,Y), axis=1)  # (Ne,Nc+2Nh,Np)
S = einsum('ecp,ehp->ech', Xhat,Xhat)  # (Ne,Nc+2Nh,Nc+2Nh)


# block variance matrix Q: blkdiag(Q1,Q2)
Q1 = np.zeros_like(S)

# u @ Q1 @ u^T: variace of filtered EEG
Q1[:,:n_chans,:n_chans] = einsum('etcp,ethp->ech', X,X)/n_train

# v @ Q2 @ v^T: variance of filtered sine-cosine template
Q1[:,n_chans:,n_chans:] = S[:,-2*n_harmonics:,-2*n_harmonics:]


# block covariance matrix S: covariance of [X.T,Y.T].T
Q2 = np.zeros_like(S)

# u @ Q11 @ u^T: variace of filtered EEG
Q2[:,:n_chans,:n_chans] = einsum('etcp,ethp->ech', X,X)/n_train

# v @ Q2 @ v^T: variance of filtered sine-cosine template
Q2[:,n_chans:,n_chans:] = S[:,-2*n_harmonics:,-2*n_harmonics:]

# u @ Q12 @ v^T: covariance of filtered EEG & filtered template
# Q12 = Q21^T
for ne in range(n_events):
    temp = einsum('tcp,hp->ch', X[ne],Y[ne])/n_train
    Q2[ne,:n_chans,n_chans:] = temp
    Q2[ne,n_chans:,:n_chans] = temp.T
del temp


w11 = trca.solve_gep(S[0], Q1[0], Nk=1)
w12 = trca.solve_gep(S[1], Q1[1], Nk=1)

w21 = trca.solve_gep(S[0], Q2[0], Nk=1)
w22 = trca.solve_gep(S[0], Q2[0], Nk=1)

# u1,v1 = trca.sctrca_compute_V2(X, Y)
# u2,v2 = trca.sctrca_compute_sp(X, Y)


# %% 测试分类准确率模块
st = perf_counter()

repeat = 5
acc1 = np.zeros((repeat))
acc2 = np.zeros((repeat))

n_trials = data.shape[2]
n_train = 3
n_test = n_trials - n_train
rand_order = np.arange(n_trials)
for nrep in range(repeat):
    np.random.shuffle(rand_order)
    
    train_data = data[0][:,rand_order[:n_train],:,160:160+250]
    test_data = data[0][:,rand_order[-n_test:],:,160:160+250]
    
    rou, erou = trca.scetrca(train_data, Y, test_data)
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
