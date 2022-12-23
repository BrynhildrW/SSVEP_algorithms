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
from srca_cpu import (SRCA, ESRCA, MultiESRCA)

import numpy as np

import scipy.linalg as sLA

from sklearn import linear_model
from itertools import combinations

from time import perf_counter

from copy import deepcopy

import scipy.io as io
import matplotlib.pyplot as plt

# %% Load in data
import scipy.io as io
import matplotlib.pyplot as plt

data_path = r'E:\SSVEP\Preprocessed Data\SSVEP：60\Sub9.mat'
dataset = io.loadmat(data_path)['normal']
n_events = dataset.shape[0]
n_trials = dataset.shape[1]
rest_phase, task_phase = [0,1000], [1140,1340]

chan_info_path = r'E:\SSVEP\Preprocessed Data\62_chan_info.mat'
chan_info = io.loadmat(chan_info_path)['chan_info'].tolist()
del data_path, chan_info_path

# target_chans = ['O1 ','OZ ','O2 ']  # 3 channels' group | fastest but useless

target_chans = ['PZ ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1 ', 'OZ ', 'O2 ']

# target_chans = ['PZ ','P1 ','P2 ','P3 ','P4 ','P5 ','P6 ','P7 ','P8 ',
#                 'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
#                 'OZ ','O1 ','O2 ','CB1','CB2']  # 21 channels' group | best

# target_chans = ['CPZ','CP1','CP2','CP3','CP4','CP5','CP6','TP7','TP8',
#                 'PZ ','P1 ','P2 ','P3 ','P4 ','P5 ','P6 ','P7 ','P8 ',
#                 'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
#                 'OZ ','O1 ','O2 ','CB1','CB2']  # 30 channels' group

target_idx = [chan_info.index(tc) for tc in target_chans]

n_train = 40
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

dsp_classifier = special.DSP().fit(train_data=train_data[...,target_idx,task_phase[0]:task_phase[1]])
rou = dsp_classifier.predict(test_data[...,target_idx,task_phase[0]:task_phase[1]])
print('DSP-M1 accuracy for original data: {}'.format(str(utils.acc_compute(rou))))


# %% SRCA test
srca_model_chans = [[] for ne in range(n_events)]

kwargs = {'n_components':1,
          'ratio':None}

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
            regression='MSE',
            kwargs=kwargs
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

dsp_classifier = special.DSP().fit(train_data=srca_train)
rou = dsp_classifier.predict(srca_test)
print('DSP-M1 accuracy for SRCA data: {}'.format(str(utils.acc_compute(rou))))


# %% eSRCA test
esrca_model_chans = []
kwargs = {'n_components':1,
          'ratio':None}

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
        regression='MSE',
        kwargs=kwargs
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

dsp_classifier = special.DSP().fit(train_data=esrca_train)
rou = dsp_classifier.predict(esrca_test)
print('DSP-M1 accuracy for eSRCA data: {}'.format(str(utils.acc_compute(rou))))


# %% td-SRCA test
tdsrca_model_chans = [[] for ne in range(n_events)]
kwargs = {'n_components':1,
          'ratio':None}

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
            regression='MSE',
            kwargs=kwargs
        )
        model.prepare()
        model.train()
        tdsrca_model_chans[ne].append(model.srca_model)
print('Finish td-SRCA training!')

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
print('TRCA accuracy for td-SRCA data: {}'.format(str(utils.acc_compute(rou))))
print('eTRCA accuracy for td-SRCA data: {}'.format(str(utils.acc_compute(erou))))

dsp_classifier = special.DSP().fit(train_data=tdsrca_train)
rou = dsp_classifier.predict(tdsrca_test)
print('DSP-M1 accuracy for td-SRCA data: {}'.format(str(utils.acc_compute(rou))))


# %% td-eSRCA test
tdesrca_model_chans = []
kwargs = {'n_components':1,
          'ratio':None}

# train models
for tc in target_chans:
    model = srca_cpu.TdESRCA(
        train_data=train_data,
        rest_phase=rest_phase,
        task_phase=task_phase,
        chan_info=chan_info,
        tar_chan=tc,
        tar_chan_list=target_chans,
        tar_func='DSP-val',
        opt_method='Recursion',
        chan_num_limit=5,
        regression='MSE',
        kwargs=kwargs
    )
    model.prepare()
    model.train()
    tdesrca_model_chans.append(model.srca_model)
print('Finish td-eSRCA training!')

# apply models into data
tdesrca_train = np.zeros((n_events, n_train, len(target_chans), task_phase[1]-task_phase[0]))
tdesrca_test = np.zeros((n_events, n_test, len(target_chans), task_phase[1]-task_phase[0]))
for nc,tc in enumerate(target_chans):
    tdesrca_train[...,nc,:] = srca_cpu.apply_ESRCA(
        rest_data=train_data[...,rest_phase[0]:rest_phase[1]],
        task_data=train_data[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=tdesrca_model_chans[nc],
        chan_info=chan_info
    )
    tdesrca_test[...,nc,:] = srca_cpu.apply_ESRCA(
        rest_data=test_data[...,rest_phase[0]:rest_phase[1]],
        task_data=test_data[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=tdesrca_model_chans[nc],
        chan_info=chan_info
    )

# classification accuracy
trca_classifier = trca.TRCA().fit(train_data=tdesrca_train)
rou, erou = trca_classifier.predict(tdesrca_test)
print('TRCA accuracy for td-eSRCA data: {}'.format(str(utils.acc_compute(rou))))
print('eTRCA accuracy for td-eSRCA data: {}'.format(str(utils.acc_compute(erou))))

dsp_classifier = special.DSP().fit(train_data=tdesrca_train)
rou = dsp_classifier.predict(tdesrca_test)
print('DSP-M1 accuracy for td-eSRCA data: {}'.format(str(utils.acc_compute(rou))))


# %% Multi-eSRCA test
kwargs = {'n_components':1,
          'ratio':None,
          'n_repeat':5,
          'n_train':30}

model = srca_cpu.MultiESRCA(
    train_data=train_data,
    rest_phase=rest_phase,
    task_phase=task_phase,
    chan_info=chan_info,
    tar_chan_list=target_chans,
    tar_func='DSP-val',
    opt_method='Recursion',
    chan_num_limit=5,
    regression='MSE',
    kwargs=kwargs)

model.prepare()
model.train()
multiesrca_model_chans = model.multiesrca_model

# apply models into data
multiesrca_train = np.zeros((n_events, n_train, len(target_chans), task_phase[1]-task_phase[0]))
multiesrca_test = np.zeros((n_events, n_test, len(target_chans), task_phase[1]-task_phase[0]))
for nc,tc in enumerate(target_chans):
    multiesrca_train[...,nc,:] = srca_cpu.apply_ESRCA(
        rest_data=train_data[...,rest_phase[0]:rest_phase[1]],
        task_data=train_data[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=multiesrca_model_chans[nc],
        chan_info=chan_info
    )
    multiesrca_test[...,nc,:] = srca_cpu.apply_ESRCA(
        rest_data=test_data[...,rest_phase[0]:rest_phase[1]],
        task_data=test_data[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=multiesrca_model_chans[nc],
        chan_info=chan_info
    )

# classification accuracy
trca_classifier = trca.TRCA().fit(train_data=multiesrca_train)
rou, erou = trca_classifier.predict(multiesrca_test)
print('TRCA accuracy for Multi-eSRCA data: {}'.format(str(utils.acc_compute(rou))))
print('eTRCA accuracy for Multi-eSRCA data: {}'.format(str(utils.acc_compute(erou))))

dsp_classifier = special.DSP().fit(train_data=multiesrca_train)
rou = dsp_classifier.predict(multiesrca_test)
print('DSP-M1 accuracy for Multi-eSRCA data: {}'.format(str(utils.acc_compute(rou))))

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

# %% BIG LOOP YEAR
chan_info_path = r'E:\SSVEP\Preprocessed Data\62_chan_info.mat'
chan_info = io.loadmat(chan_info_path)['chan_info'].tolist()
del chan_info_path

target_chans = ['O1 ','OZ ','O2 ']  # 3 channels' group | fastest but useless

# target_chans = ['PZ ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1 ', 'OZ ', 'O2 ']

# target_chans = ['PZ ','P1 ','P2 ','P3 ','P4 ','P5 ','P6 ','P7 ','P8 ',
#                 'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
#                 'OZ ','O1 ','O2 ','CB1','CB2']  # 21 channels' group | best

# target_chans = ['CPZ','CP1','CP2','CP3','CP4','CP5','CP6','TP7','TP8',
#                 'PZ ','P1 ','P2 ','P3 ','P4 ','P5 ','P6 ','P7 ','P8 ',
#                 'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
#                 'OZ ','O1 ','O2 ','CB1','CB2']  # 30 channels' group

target_indices = [chan_info.index(tc) for tc in target_chans]

sub_list = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
length_list = [200,300,400,500]
sample_list = [20,40]
cv_times = 5

for nsub,sub_id in enumerate(sub_list):
    data_path = r'E:\SSVEP\Preprocessed Data\SSVEP：60\Sub%d.mat' %(sub_id)
    dataset = io.loadmat(data_path)['normal']
    n_events = dataset.shape[0]
    n_trials = dataset.shape[1]
    
    # initialization for results saving
    log_file = {'Data length':[],
                'Training samples':[],
                'SRCA model':[],
                'eSRCA model':[],
                'Multi-eSRCA model':[]}
    acc_trca = np.zeros((len(length_list), len(sample_list), cv_times))
    acc_etrca = np.zeros_like(acc_trca)
    acc_dsp = np.zeros_like(acc_trca)
    acc_srca_trca = np.zeros_like(acc_trca)
    acc_srca_etrca = np.zeros_like(acc_trca)
    acc_srca_dsp = np.zeros_like(acc_trca)
    acc_esrca_trca = np.zeros_like(acc_trca)
    acc_esrca_etrca = np.zeros_like(acc_trca)
    acc_esrca_dsp = np.zeros_like(acc_trca)
    acc_mesrca_trca = np.zeros_like(acc_trca)
    acc_mesrca_etrca = np.zeros_like(acc_trca)
    acc_mesrca_dsp = np.zeros_like(acc_trca)
    
    # begin loop
    for nlen,data_length in enumerate(length_list):
        rest_phase, task_phase = [0,1000], [1140,1140+data_length]
        for nsam,sample in enumerate(sample_list):
            n_train = sample
            n_test = n_trials - n_train
            rand_order = np.arange(n_trials)
            
            kwargs = {'n_components':1,
                      'ratio':None,
                      'n_repeat':5,
                      'n_train':int(0.75*n_train)}
            
            for nrep in range(cv_times):
                np.random.shuffle(rand_order)
                train_data = dataset[:,rand_order[:n_train],...]
                test_data = dataset[:,rand_order[-n_test:],...]

                # update log files
                log_file['Data length'].append(data_length)
                log_file['Training samples'].append(rand_order[:n_train])

                #***************************************************************************#
                # baseline
                trca_classifier = trca.TRCA()
                trca_classifier.fit(train_data=train_data[...,target_indices,task_phase[0]:task_phase[1]])
                rou, erou = trca_classifier.predict(test_data[...,target_indices,task_phase[0]:task_phase[1]])
                acc_trca[nlen,nsam,nrep] = utils.acc_compute(rou)
                acc_etrca[nlen,nsam,nrep] = utils.acc_compute(erou)

                dsp_classifier = special.DSP()
                dsp_classifier.fit(train_data=train_data[...,target_indices,task_phase[0]:task_phase[1]])
                drou = dsp_classifier.predict(test_data[...,target_indices,task_phase[0]:task_phase[1]])
                acc_dsp[nlen,nsam,nrep] = utils.acc_compute(drou)

                #***************************************************************************#
                # SRCA
                srca_model_chans = [[] for ne in range(n_events)]
                for ne in range(n_events):
                    for tc in target_chans:
                        model = SRCA(
                            train_data=train_data[ne],
                            rest_phase=rest_phase,
                            task_phase=task_phase,
                            chan_info=chan_info,
                            tar_chan=tc,
                            tar_func='SNR',
                            opt_method='Recursion',
                            chan_num_limit=10,
                            regression='MSE',
                            kwargs=kwargs
                        )
                        model.prepare()
                        model.train()
                        srca_model_chans[ne].append(model.srca_model)
                print('Finish SRCA training!')
                
                log_file['SRCA model'].append(srca_model_chans)

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
                trca_classifier = trca.TRCA().fit(train_data=srca_train)
                rou, erou = trca_classifier.predict(srca_test)
                acc_srca_trca[nlen,nsam,nrep] = utils.acc_compute(rou)
                acc_srca_etrca[nlen,nsam,nrep] = utils.acc_compute(erou)

                dsp_classifier = special.DSP().fit(train_data=srca_train)
                drou = dsp_classifier.predict(srca_test)
                acc_srca_dsp[nlen,nsam,nrep] = utils.acc_compute(drou)
                
                #***************************************************************************#
                # eSRCA
                esrca_model_chans = []
                for tc in target_chans:
                    model = ESRCA(
                        train_data=train_data,
                        rest_phase=rest_phase,
                        task_phase=task_phase,
                        chan_info=chan_info,
                        tar_chan=tc,
                        tar_func='SNR',
                        opt_method='Recursion',
                        chan_num_limit=10,
                        regression='MSE',
                        kwargs=kwargs
                    )
                    model.prepare()
                    model.train()
                    esrca_model_chans.append(model.srca_model)
                print('Finish eSRCA training!')

                log_file['eSRCA model'].append(esrca_model_chans)

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
                trca_classifier = trca.TRCA().fit(train_data=esrca_train)
                rou, erou = trca_classifier.predict(esrca_test)
                acc_esrca_trca[nlen,nsam,nrep] = utils.acc_compute(rou)
                acc_esrca_etrca[nlen,nsam,nrep] = utils.acc_compute(erou)

                dsp_classifier = special.DSP().fit(train_data=esrca_train)
                drou = dsp_classifier.predict(esrca_test)
                acc_esrca_dsp[nlen,nsam,nrep] = utils.acc_compute(drou)

                #***************************************************************************#
                # Multi-eSRCA
                model = MultiESRCA(
                    train_data=train_data,
                    rest_phase=rest_phase,
                    task_phase=task_phase,
                    chan_info=chan_info,
                    tar_chan_list=target_chans,
                    tar_func='DSP-val',
                    opt_method='Recursion',
                    chan_num_limit=10,
                    regression='MSE',
                    kwargs=kwargs
                )
                model.prepare()
                model.train()
                multiesrca_model_chans = model.multiesrca_model
                print('Finish Multi-eSRCA training!')
                
                log_file['Multi-eSRCA model'].append(multiesrca_model_chans)

                multiesrca_train = np.zeros((n_events, n_train, len(target_chans), task_phase[1]-task_phase[0]))
                multiesrca_test = np.zeros((n_events, n_test, len(target_chans), task_phase[1]-task_phase[0]))
                for nc,tc in enumerate(target_chans):
                    multiesrca_train[...,nc,:] = srca_cpu.apply_ESRCA(
                        rest_data=train_data[...,rest_phase[0]:rest_phase[1]],
                        task_data=train_data[...,task_phase[0]:task_phase[1]],
                        target_chan=tc,
                        model_chans=multiesrca_model_chans[nc],
                        chan_info=chan_info
                    )
                    multiesrca_test[...,nc,:] = srca_cpu.apply_ESRCA(
                        rest_data=test_data[...,rest_phase[0]:rest_phase[1]],
                        task_data=test_data[...,task_phase[0]:task_phase[1]],
                        target_chan=tc,
                        model_chans=multiesrca_model_chans[nc],
                        chan_info=chan_info
                    )
                trca_classifier = trca.TRCA().fit(train_data=multiesrca_train)
                rou, erou = trca_classifier.predict(multiesrca_test)
                acc_mesrca_trca[nlen,nsam,nrep] = utils.acc_compute(rou)
                acc_mesrca_etrca[nlen,nsam,nrep] = utils.acc_compute(erou)

                dsp_classifier = special.DSP().fit(train_data=multiesrca_train)
                drou = dsp_classifier.predict(multiesrca_test)
                acc_mesrca_dsp[nlen,nsam,nrep] = utils.acc_compute(drou)
    # save results
    result_path = r'E:\SSVEP\Results\20221208\Acc_Sub%d.mat' %(sub_id)
    io.savemat(result_path, {'TRCA':acc_trca, 'eTRCA':acc_etrca, 'DSP':acc_dsp,
                             'SRCA-TRCA':acc_srca_trca, 'SRCA-eTRCA':acc_srca_etrca, 'SRCA-DSP':acc_srca_dsp,
                             'eSRCA-TRCA':acc_esrca_trca, 'eSRCA-eTRCA':acc_esrca_etrca, 'eSRCA-DSP':acc_esrca_dsp,
                             'Multi-eSRCA-TRCA':acc_mesrca_trca, 'Multi-eSRCA-eTRCA':acc_mesrca_etrca, 'Multi-eSRCA-DSP':acc_esrca_dsp,
                             'Log file':log_file})
