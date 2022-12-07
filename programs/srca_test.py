# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

SRCA demo

update: 2022/12/5

"""

# %% load in modules
# cd D:\Software\Github\SSVEP_algorithms\programs
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


# %% Load in data
import scipy.io as io
import matplotlib.pyplot as plt

data_path = r'E:\SSVEP\Preprocessed Data\SSVEP：60\Sub17.mat'
dataset = io.loadmat(data_path)['normal']
n_events = dataset.shape[0]
n_trials = dataset.shape[1]
rest_phase, task_phase = [0,1000], [1140,1340]

chan_info_path = r'E:\SSVEP\Preprocessed Data\62_chan_info.mat'
chan_info = io.loadmat(chan_info_path)['chan_info'].tolist()
del data_path, chan_info_path

target_chans = ['O1 ','OZ ','O2 ']  # 3 channels' group | fastest but useless

# target_chans = ['PZ ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1 ', 'OZ ', 'O2 ']

# target_chans = ['PZ ','P1 ','P2 ','P3 ','P4 ','P5 ','P6 ','P7 ','P8 ',
#                 'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
#                 'OZ ','O1 ','O2 ','CB1','CB2']  # 21 channels' group | best

# target_chans = ['CPZ','CP1','CP2','CP3','CP4','CP5','CP6','TP7','TP8',
#                 'PZ ','P1 ','P2 ','P3 ','P4 ','P5 ','P6 ','P7 ','P8 ',
#                 'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
#                 'OZ ','O1 ','O2 ','CB1','CB2']  # 30 channels' group

target_idx = [chan_info.index(tc) for tc in target_chans]

n_train = 30
n_test = n_trials - n_train
rand_order = np.arange(n_trials)

np.random.shuffle(rand_order)
train_data = dataset[:,rand_order[:n_train],...]
test_data = dataset[:,rand_order[n_train:],...]

# baseline
trca_classifier = trca.TRCA().fit(train_data=train_data[...,target_idx,task_phase[0]:task_phase[1]])
rou, erou = trca_classifier.predict(test_data[...,target_idx,task_phase[0]:task_phase[1]])
print('TRCA accuracy for original data: {}'.format(str(utils.acc_compute(rou))))
print('eTRCA accuracy for original data: {}'.format(str(utils.acc_compute(erou))))

# %% SRCA test
srca_model_chans = [[] for ne in range(n_events)]

# train models
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

# apply models into data
srca_train = np.zeros((n_events, n_train, len(target_chans), task_phase[1]-task_phase[0]))
srca_test = np.zeros((n_events, n_test, len(target_chans), task_phase[1]-task_phase[0]))
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

# classification accuracy
trca_classifier = trca.TRCA().fit(train_data=srca_train)
rou, erou = trca_classifier.predict(srca_test)
print('TRCA accuracy for SRCA data: {}'.format(str(utils.acc_compute(rou))))
print('eTRCA accuracy for SRCA data: {}'.format(str(utils.acc_compute(erou))))

# %% eSRCA test
esrca_model_chans = []

# train models
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

# apply models into data
esrca_train = np.zeros((n_events, n_train, len(target_chans), task_phase[1]-task_phase[0]))
esrca_test = np.zeros((n_events, n_test, len(target_chans), task_phase[1]-task_phase[0]))
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

# classification accuracy
trca_classifier = trca.TRCA().fit(train_data=esrca_train)
rou, erou = trca_classifier.predict(esrca_test)
print('TRCA accuracy for eSRCA data: {}'.format(str(utils.acc_compute(rou))))
print('eTRCA accuracy for eSRCA data: {}'.format(str(utils.acc_compute(erou))))


# %% tdSRCA test
tdsrca_model_chans = [[] for ne in range(n_events)]

for ne in range(n_events):
    for tc in target_chans:
        model = srca_cpu.TdSRCA(
            train_data=train_data[ne],
            rest_phase=rest_phase,
            task_phase=task_phase,
            chan_info=chan_info,
            tar_chan=tc,
            tar_chan_list=target_chans,
            tar_func='TRCA-val',
            opt_method='Recursion',
            chan_num_limit=10,
            regression='MSE'
        )
        model.prepare()
        model.train()
        tdsrca_model_chans[ne].append(model.srca_model)
print('Finish tdSRCA training!')

# apply models into data
tdsrca_train = np.zeros((n_events, n_train, len(target_chans), task_phase[1]-task_phase[0]))
tdsrca_test = np.zeros((n_events, n_test, len(target_chans), task_phase[1]-task_phase[0]))
for ne in range(n_events):
    for nc,tc in enumerate(target_chans):
        tdsrca_train[ne,:,nc,:] = srca_cpu.apply_SRCA(
            rest_data=train_data[ne,...,rest_phase[0]:rest_phase[1]],
            task_data=train_data[ne,...,task_phase[0]:task_phase[1]],
            target_chan=tc,
            model_chans=tdsrca_model_chans[ne][nc],
            chan_info=chan_info
        )
        tdsrca_test[ne,:,nc,:] = srca_cpu.apply_SRCA(
            rest_data=test_data[ne,...,rest_phase[0]:rest_phase[1]],
            task_data=test_data[ne,...,task_phase[0]:task_phase[1]],
            target_chan=tc,
            model_chans=tdsrca_model_chans[ne][nc],
            chan_info=chan_info
        )

# classification accuracy
trca_classifier = trca.TRCA().fit(train_data=tdsrca_train)
rou, erou = trca_classifier.predict(tdsrca_test)
print('TRCA accuracy for tdSRCA data: {}'.format(str(utils.acc_compute(rou))))
print('eTRCA accuracy for tdSRCA data: {}'.format(str(utils.acc_compute(erou))))


# %% target value test
# snr_train_origin = np.zeros((n_events, len(target_chans), task_phase[1]-task_phase[0]))
# snr_train_srca = np.zeros_like(snr_origin)
# snr_test_origin = np.zeros_like(snr_origin)
# snr_test_srca = np.zeros_like(snr_origin)
# for nc,tc in enumerate(target_chans):
#     tar_idx = chan_info.index(tc)
#     snr_train_origin[:,nc,:] = srca_cpu.snr_sequence(train_data[...,tar_idx,task_phase[0]:task_phase[1]]).squeeze()
#     snr_test_origin[:,nc,:] = srca_cpu.snr_sequence(test_data[...,tar_idx,task_phase[0]:task_phase[1]]).squeeze()
#     snr_train_srca[:,nc,:] = srca_cpu.snr_sequence(srca_train[...,nc,:]).squeeze()
#     snr_test_srca[:,nc,:] = srca_cpu.snr_sequence(srca_test[...,nc,:]).squeeze()

# event_idx = 0
# chan_idx = 3
# # plt.plot(snr_train_origin[event_idx,chan_idx,:])
# # plt.plot(snr_train_srca[event_idx,chan_idx,:])

# plt.plot(snr_test_origin[event_idx,chan_idx,:])
# plt.plot(snr_test_srca[event_idx,chan_idx,:])

