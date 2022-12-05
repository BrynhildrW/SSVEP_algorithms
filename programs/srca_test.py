# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

SRCA demo

update: 2022/12/5

"""

# %% load in modules
import utils
import special
from special import DSP
import trca
from trca import TRCA
import cca
import srca_cpu

import numpy as np

import scipy.linalg as sLA

from sklearn import linear_model
from itertools import combinations

from time import perf_counter


# %% SRCA test
import scipy.io as io
import matplotlib.pyplot as plt

data_path = r'E:\SSVEP\Preprocessed Data\SSVEP：60\Sub7.mat'
dataset = io.loadmat(data_path)['normal']
n_events = dataset.shape[0]
n_trials = dataset.shape[1]
rest_phase, task_phase = [0,1000], [1140,1640]

chan_info_path = r'E:\SSVEP\Preprocessed Data\62_chan_info.mat'
chan_info = io.loadmat(chan_info_path)['chan_info'].tolist()
del data_path, chan_info_path

target_chans = ['PZ ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1 ', 'OZ ', 'O2 ']
target_idx = [chan_info.index(tc) for tc in target_chans]

n_train = 40
rand_order = np.arange(n_trials)

np.random.shuffle(rand_order)
train_data = dataset[:,rand_order[:n_train],...]
test_data = dataset[:,rand_order[n_train:],...]

srca_model_chans = [[],[]]
esrca_model_chans = []

for ne in range(n_events):
    for tc in target_chans:
        model = srca_cpu.SRCA(
            train_data=train_data[ne],
            rest_phase=rest_phase,
            task_phase=task_phase,
            chan_info=chan_info,
            tar_chan=tc,
            tar_func='SNR',
            opt_method='Recursion',
            chan_num_limit=10,
            regression='MSE'
        )
        model.prepare()
        model.train()
        srca_model_chans[ne].append(model.srca_model)
print('Finish SRCA training!')

for tc in target_chans:
    model = srca_cpu.ESRCA(
        train_data=train_data,
        rest_phase=rest_phase,
        task_phase=task_phase,
        chan_info=chan_info,
        tar_chan=tc,
        tar_func='SNR',
        opt_method='Recursion',
        chan_num_limit=10,
        regression='MSE'
    )
    model.prepare()
    model.train()
    esrca_model_chans.append(model.srca_model)
print('Finish eSRCA training!')

# %%
srca_train = np.zeros((n_events, n_train, len(target_chans), task_phase[1]-task_phase[0]))
srca_test = np.zeros((n_events, n_trials-n_train, len(target_chans), task_phase[1]-task_phase[0]))
for ne in range(n_events):
    for nc,tc in enumerate(target_chans):
        srca_train[ne,:,nc,:] = srca_cpu.apply_SRCA(
            rest_data=train_data[ne,...,rest_phase[0]:rest_phase[1]],
            task_data=train_data[ne,...,task_phase[0]:task_phase[1]],
            target_chan=tc,
            model_chans=srca_model_chans[ne][nc],
            chan_info=chan_info
        )
        srca_test[ne,:,nc,:] = srca_cpu.apply_SRCA(
            rest_data=test_data[ne,...,rest_phase[0]:rest_phase[1]],
            task_data=test_data[ne,...,task_phase[0]:task_phase[1]],
            target_chan=tc,
            model_chans=srca_model_chans[ne][nc],
            chan_info=chan_info
        )


esrca_train = np.zeros((n_events, n_train, len(target_chans), task_phase[1]-task_phase[0]))
esrca_test = np.zeros((n_events, n_trials-n_train, len(target_chans), task_phase[1]-task_phase[0]))
for nc,tc in enumerate(target_chans):
    esrca_train[...,nc,:] = srca_cpu.apply_ESRCA(
        rest_data=train_data[...,rest_phase[0]:rest_phase[1]],
        task_data=train_data[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=esrca_model_chans[nc],
        chan_info=chan_info
    )
    esrca_test[...,nc,:] = srca_cpu.apply_ESRCA(
        rest_data=test_data[...,rest_phase[0]:rest_phase[1]],
        task_data=test_data[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=esrca_model_chans[nc],
        chan_info=chan_info
    )


# %%
snr_train_origin = np.zeros((n_events, len(target_chans), task_phase[1]-task_phase[0]))
snr_train_srca = np.zeros_like(snr_origin)
snr_test_origin = np.zeros_like(snr_origin)
snr_test_srca = np.zeros_like(snr_origin)
for nc,tc in enumerate(target_chans):
    tar_idx = chan_info.index(tc)
    snr_train_origin[:,nc,:] = srca_cpu.snr_sequence(train_data[...,tar_idx,task_phase[0]:task_phase[1]]).squeeze()
    snr_test_origin[:,nc,:] = srca_cpu.snr_sequence(test_data[...,tar_idx,task_phase[0]:task_phase[1]]).squeeze()
    snr_train_srca[:,nc,:] = srca_cpu.snr_sequence(srca_train[...,nc,:]).squeeze()
    snr_test_srca[:,nc,:] = srca_cpu.snr_sequence(srca_test[...,nc,:]).squeeze()

# %%
event_idx = 0
chan_idx = 3
# plt.plot(snr_train_origin[event_idx,chan_idx,:])
# plt.plot(snr_train_srca[event_idx,chan_idx,:])

plt.plot(snr_test_origin[event_idx,chan_idx,:])
plt.plot(snr_test_srca[event_idx,chan_idx,:])

# %% Acc for SRCA processed data
rou, erou = trca.etrca(
    train_data=train_data[...,target_idx,1140:1640],
    avg_template=train_data[...,target_idx,1140:1640].mean(axis=1),
    test_data=test_data[...,target_idx,1140:1640]
)
print('TRCA accuracy for original data: {}'.format(str(utils.acc_compute(rou))))
print('eTRCA accuracy for original data: {}'.format(str(utils.acc_compute(erou))))

rou, erou = trca.etrca(
    train_data=esrca_train,
    avg_template=esrca_train.mean(axis=1),
    test_data=esrca_test
)
print('TRCA accuracy for eSRCA processed data: {}'.format(str(utils.acc_compute(rou))))
print('eTRCA accuracy for eSRCA processed data: {}'.format(str(utils.acc_compute(erou))))

# %%