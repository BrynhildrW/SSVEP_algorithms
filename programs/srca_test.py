# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

SRCA demo

update: 2022/12/5

"""

# %% load in modules
# cd F:\Github\SSVEP_algorithms\programs
import utils
import special
import trca

import srca_cpu
from srca_cpu import (SRCA, ESRCA, MultiESRCA)

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

import scipy.io as io
import matplotlib.pyplot as plt

# %% Load in data
import scipy.io as io
import matplotlib.pyplot as plt

data_path = r'D:\SSVEP\Preprocessed Data\SSVEP：60\Sub9.mat'
dataset = io.loadmat(data_path)['normal']
n_events = dataset.shape[0]
rest_phase, task_phase = [0,1000], [1140,1340]

chan_info_path = r'D:\SSVEP\Preprocessed Data\62_chan_info.mat'
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
X_total, y_total = utils.reshape_dataset(dataset)
event_type = np.unique(y_total)
n_trials = len(y_total)


# %% baseline
n_train = 40*2
n_repeat = 1
sss = StratifiedShuffleSplit(
    n_splits=n_repeat,
    test_size=1-n_train/n_trials,
    random_state=1
)

for nrep, (train_index,test_index) in enumerate(sss.split(X_total, y_total)):
    X_train, X_test = X_total[train_index], X_total[test_index]
    y_train, y_test = y_total[train_index], y_total[test_index]
    train_samples = np.array([np.sum(y_train==et) for et in event_type])
    test_samples = np.array([np.sum(y_test==et) for et in event_type])

    # baseline
    trca_classifier = trca.TRCA().fit(
        X_train=X_train[:, target_idx, task_phase[0]:task_phase[1]],
        y_train=y_train
    )
    _, trca_predict, _, etrca_predict = trca_classifier.predict(
        X_test=X_test[:, target_idx, task_phase[0]:task_phase[1]],
        y_test=y_test
    )
    acc_1 = utils.acc_compute(trca_predict, y_test)
    acc_2 = utils.acc_compute(etrca_predict, y_test)
    print('TRCA accuracy for original data: {}'.format(str(acc_1.mean())))
    print('eTRCA accuracy for original data: {}'.format(str(acc_2.mean())))

    dsp_classifier = special.DSP().fit(
        X_train=X_train[:, target_idx, task_phase[0]:task_phase[1]],
        y_train=y_train
    )
    _, dsp_predict = dsp_classifier.predict(
        X_test=X_test[:, target_idx, task_phase[0]:task_phase[1]],
        y_test=y_test
    )
    acc_3 = utils.acc_compute(dsp_predict, y_test)
    print('DSP-M1 accuracy for original data: {}'.format(str(acc_3.mean())))


# %% SRCA test
srca_model_chans = [[] for ne in range(n_events)]

# train models
for ne in range(n_events):
    for tc in target_chans:
        model = srca_cpu.SRCA(
            X_train=X_train[y_train==ne],
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
srca_X_train = np.zeros((len(y_train), len(target_chans), task_phase[1]-task_phase[0]))
srca_y_train = np.ones((len(y_train)))
srca_X_test = np.zeros((len(y_test), len(target_chans), task_phase[1]-task_phase[0]))
srca_y_test = np.ones((len(y_test)))

trial_idx = 0
for ne in range(n_events):
    train_trials = train_samples[ne]
    ytr = event_type[ne]
    for nc,tc in enumerate(target_chans):
        srca_X_train[trial_idx:trial_idx+train_trials, nc, :] = srca_cpu.apply_SRCA(
            rest_data=X_train[y_train==ytr][...,rest_phase[0]:rest_phase[1]],
            task_data=X_train[y_train==ytr][...,task_phase[0]:task_phase[1]],
            target_chan=tc,
            model_chans=srca_model_chans[ytr][nc],
            chan_info=chan_info
        )
    srca_y_train[trial_idx:trial_idx+train_trials] *= ytr
    trial_idx += train_trials

trial_idx = 0
for ne in range(n_events):
    test_trials = test_samples[ne]
    yte = event_type[ne]
    for nc,tc in enumerate(target_chans):
        srca_X_test[trial_idx:trial_idx+test_trials, nc, :] = srca_cpu.apply_SRCA(
            rest_data=X_test[y_test==yte][...,rest_phase[0]:rest_phase[1]],
            task_data=X_test[y_test==yte][...,task_phase[0]:task_phase[1]],
            target_chan=tc,
            model_chans=srca_model_chans[yte][nc],
            chan_info=chan_info
        )
    srca_y_test[trial_idx:trial_idx+test_trials] *= yte
    trial_idx += test_trials

# classification accuracy
trca_classifier = trca.TRCA().fit(
    X_train=srca_X_train,
    y_train=srca_y_train
)
_, trca_predict, _, etrca_predict = trca_classifier.predict(
    X_test=srca_X_test,
    y_test=srca_y_test
)
acc_1_srca = utils.acc_compute(trca_predict, srca_y_test)
acc_2_srca = utils.acc_compute(etrca_predict, srca_y_test)
print('TRCA accuracy for SRCA data: {}'.format(str(acc_1_srca.mean())))
print('eTRCA accuracy for SRCA data: {}'.format(str(acc_2_srca.mean())))

dsp_classifier = special.DSP().fit(
    X_train=srca_X_train,
    y_train=srca_y_train
)
_, dsp_predict = dsp_classifier.predict(
    X_test=srca_X_test,
    y_test=srca_y_test
)
acc_3_srca = utils.acc_compute(dsp_predict, srca_y_test)
print('DSP-M1 accuracy for SRCA data: {}'.format(str(acc_3_srca.mean())))


# %% eSRCA test
esrca_model_chans = []

# train models
for tc in target_chans:
    model = srca_cpu.ESRCA(
        X_train=X_train,
        y_train=y_train,
        rest_phase=rest_phase,
        task_phase=task_phase,
        chan_info=chan_info,
        tar_chan=tc,
        tar_func='SNR',
        opt_method='Recursion',
        chan_num_limit=10,
        regression='MSE',
    )
    model.prepare()
    model.train()
    esrca_model_chans.append(model.srca_model)
print('Finish eSRCA training!')

# apply models into data
esrca_X_train = np.zeros((len(y_train), len(target_chans), task_phase[1]-task_phase[0]))
esrca_X_test = np.zeros((len(y_test), len(target_chans),task_phase[1]-task_phase[0]))

for nc,tc in enumerate(target_chans):
    esrca_X_train[:,nc,:] = srca_cpu.apply_SRCA(
        rest_data=X_train[...,rest_phase[0]:rest_phase[1]],
        task_data=X_train[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=esrca_model_chans[nc],
        chan_info=chan_info
    )
    esrca_X_test[...,nc,:] = srca_cpu.apply_SRCA(
        rest_data=X_test[...,rest_phase[0]:rest_phase[1]],
        task_data=X_test[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=esrca_model_chans[nc],
        chan_info=chan_info
    )

# classification accuracy
trca_classifier = trca.TRCA().fit(
    X_train=esrca_X_train,
    y_train=y_train
)
_, trca_predict, _, etrca_predict = trca_classifier.predict(
    X_test=esrca_X_test,
    y_test=y_test
)
acc_1_esrca = utils.acc_compute(trca_predict, y_test)
acc_2_esrca = utils.acc_compute(etrca_predict, y_test)
print('TRCA accuracy for eSRCA data: {}'.format(str(acc_1_esrca.mean())))
print('eTRCA accuracy for eSRCA data: {}'.format(str(acc_2_esrca.mean())))

dsp_classifier = special.DSP().fit(
    X_train=esrca_X_train,
    y_train=y_train
)
_, dsp_predict = dsp_classifier.predict(
    X_test=esrca_X_test,
    y_test=y_test
)
acc_3_esrca = utils.acc_compute(dsp_predict, y_test)
print('DSP-M1 accuracy for eSRCA data: {}'.format(str(acc_3_esrca.mean())))


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
    X_train=X_train,
    y_train=y_train,
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

# apply models into data
multiesrca_X_train = np.zeros((len(y_train), len(target_chans), task_phase[1]-task_phase[0]))
multiesrca_X_test = np.zeros((len(y_test), len(target_chans), task_phase[1]-task_phase[0]))
for nc,tc in enumerate(target_chans):
    multiesrca_X_train[:,nc,:] = srca_cpu.apply_SRCA(
        rest_data=X_train[...,rest_phase[0]:rest_phase[1]],
        task_data=X_train[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=multiesrca_model_chans[nc],
        chan_info=chan_info
    )
    multiesrca_X_test[:,nc,:] = srca_cpu.apply_SRCA(
        rest_data=X_test[...,rest_phase[0]:rest_phase[1]],
        task_data=X_test[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=multiesrca_model_chans[nc],
        chan_info=chan_info
    )

# classification accuracy
trca_classifier = trca.TRCA().fit(
    X_train=multiesrca_X_train,
    y_train=y_train
)
_, trca_predict, _, etrca_predict = trca_classifier.predict(
    X_test=multiesrca_X_test,
    y_test=y_test
)
acc_1_mesrca = utils.acc_compute(trca_predict, y_test)
acc_2_mesrca = utils.acc_compute(etrca_predict, y_test)
print('TRCA accuracy for Multi-eSRCA data: {}'.format(str(acc_1_mesrca.mean())))
print('eTRCA accuracy for Multi-eSRCA data: {}'.format(str(acc_2_mesrca.mean())))

dsp_classifier = special.DSP().fit(
    X_train=multiesrca_X_train,
    y_train=y_train
)
_, dsp_predict = dsp_classifier.predict(
    X_test=multiesrca_X_test,
    y_test=y_test
)
acc_3_mesrca = utils.acc_compute(dsp_predict, y_test)
print('DSP-M1 accuracy for Multi-eSRCA data: {}'.format(str(acc_3_mesrca.mean())))

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


# %% Circle & Ring
chan_info_path = r'D:\SSVEP\Preprocessed Data\62_chan_info.mat'
chan_info = io.loadmat(chan_info_path)['chan_info'].tolist()
del chan_info_path

# target_chans = ['O1 ','OZ ','O2 ']  # 3 channels' group | fastest but useless

target_chans = ['PZ ', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'O1 ', 'OZ ', 'O2 ']

# target_chans = ['PZ ','P1 ','P2 ','P3 ','P4 ','P5 ','P6 ','P7 ','P8 ',
#                 'POZ','PO3','PO4','PO5','PO6','PO7','PO8',
#                 'OZ ','O1 ','O2 ','CB1','CB2']  # 21 channels' group | best


target_indices = [chan_info.index(tc) for tc in target_chans]

train_length = [100,200,300,400,500]
train_sample = [30,60]
n_repeat = 5

data_path = r'D:\SSVEP\Preprocessed Data\Ring & Circle：60\20230221-sjs\ring_1.5.mat'
eeg = io.loadmat(data_path)
X, y = eeg['narrow'], eeg['trial_info'].squeeze()
# for cirlce, Ne=0 is one more than Ne=1, so delete the first trial will be fine
# X = np.delete(X, [-1,-2], axis=0)
# y = np.delete(y, [-1,-2], axis=0)
total_trials = X.shape[0]

# %%
n_events = 2
# n_trials = dataset.shape[1]
    
# initialization for results saving
log_file = {'Data length':[],
            'Training samples':[],
            'SRCA model':[],
            'eSRCA model':[],
            'Multi-eSRCA model':[]}
acc_trca = np.zeros((len(train_length), len(train_sample), n_repeat))
acc_etrca = np.zeros_like(acc_trca)
acc_dsp = np.zeros_like(acc_trca)
# acc_srca_trca = np.zeros_like(acc_trca)
# acc_srca_etrca = np.zeros_like(acc_trca)
# acc_srca_dsp = np.zeros_like(acc_trca)
acc_esrca_trca = np.zeros_like(acc_trca)
acc_esrca_etrca = np.zeros_like(acc_trca)
acc_esrca_dsp = np.zeros_like(acc_trca)
acc_mesrca_trca = np.zeros_like(acc_trca)
acc_mesrca_etrca = np.zeros_like(acc_trca)
acc_mesrca_dsp = np.zeros_like(acc_trca)

# begin loop
# loop in data length: 0.2->0.5, d=0.1s
for nlen, data_length in enumerate(train_length):
    rest_phase, task_phase = [0,1000], [1140,1140+data_length]
    X_total, y_total = X[...,:task_phase[-1]], y
    
    # loop in samples: 16,32 (each)
    for nsam, sample in enumerate(train_sample):
        # cross-validation conditions
        sss = StratifiedShuffleSplit(
            n_splits=n_repeat,
            test_size=1-sample/total_trials,
            random_state=0
        )
        kwargs = {'n_components':1,
                  'ratio':None,
                  'n_repeat':5,
                  'n_train':0.75*nsam}

        # loop in cross-validation
        for nrep, (train_index, test_index) in enumerate(sss.split(X_total, y_total)):
            X_train, X_test = X_total[train_index], X_total[test_index]
            y_train, y_test = y_total[train_index], y_total[test_index]

            # update log files
            log_file['Data length'].append(data_length)
            log_file['Training samples'].append(train_index)

            #***************************************************************************#
            # baseline
            trca_classifier = trca.TRCA().fit(
                X_train=X_train[...,target_indices,task_phase[0]:task_phase[1]],
                y_train=y_train
            )
            _, y_trca, _, y_etrca = trca_classifier.predict(
                X_test=X_test[...,target_indices,task_phase[0]:task_phase[1]],
                y_test=y_test
            )
            acc_trca[nlen,nsam,nrep] = utils.acc_compute(y_trca, y_test)
            acc_etrca[nlen,nsam,nrep] = utils.acc_compute(y_etrca, y_test)

            dsp_classifier = special.DSP().fit(
                X_train=X_train[...,target_indices,task_phase[0]:task_phase[1]],
                y_train=y_train
            )
            _, y_dsp = dsp_classifier.predict(
                X_test=X_test[...,target_indices,task_phase[0]:task_phase[1]],
                y_test=y_test
            )
            acc_dsp[nlen,nsam,nrep] = utils.acc_compute(y_dsp, y_test)
            
            #***************************************************************************#


            #***************************************************************************#
            # eSRCA
            esrca_model_chans = []
            for tc in target_chans:
                model = ESRCA(
                    train_data=X_train_reshape,
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

            esrca_train = np.zeros((X_train_reshape.shape[0], X_train_reshape.shape[1], len(target_chans), task_phase[1]-task_phase[0]))
            esrca_test = np.zeros((X_test_reshape.shape[0], X_test_reshape.shape[1], len(target_chans), task_phase[1]-task_phase[0]))
            for nc,tc in enumerate(target_chans):
                esrca_train[...,nc,:] = srca_cpu.apply_ESRCA(
                    rest_data=X_train_reshape[...,rest_phase[0]:rest_phase[1]],
                    task_data=X_train_reshape[...,task_phase[0]:task_phase[1]],
                    target_chan=tc,
                    model_chans=esrca_model_chans[nc],
                    chan_info=chan_info
                )
                esrca_test[...,nc,:] = srca_cpu.apply_ESRCA(
                    rest_data=X_test_reshape[...,rest_phase[0]:rest_phase[1]],
                    task_data=X_test_reshape[...,task_phase[0]:task_phase[1]],
                    target_chan=tc,
                    model_chans=esrca_model_chans[nc],
                    chan_info=chan_info
                )
            X_esrca_train, y_esrca_train = utils.reshape_dataset(esrca_train)
            X_esrca_test, y_esrca_test = utils.reshape_dataset(esrca_test)
            trca_classifier = trca.TRCA().fit(
                X_train=X_esrca_train,
                y_train=y_esrca_train
            )
            _, y_trca, _, y_etrca = trca_classifier.predict(
                X_test=X_esrca_test,
                y_test=y_esrca_test
            )
            acc_esrca_trca[nlen,nsam,nrep] = utils.acc_compute(y_trca, y_esrca_test)
            acc_esrca_etrca[nlen,nsam,nrep] = utils.acc_compute(y_etrca, y_esrca_test)

            dsp_classifier = special.DSP().fit(
                X_train=X_esrca_train,
                y_train=y_esrca_train
            )
            _, y_dsp = dsp_classifier.predict(
                X_test=X_esrca_test,
                y_test=y_esrca_test
            )
            acc_esrca_dsp[nlen,nsam,nrep] = utils.acc_compute(y_dsp, y_esrca_test)

            #***************************************************************************#
            # Multi-eSRCA
            model = MultiESRCA(
                train_data=X_train_reshape,
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

            multiesrca_train = np.zeros((X_train_reshape.shape[0], X_train_reshape.shape[1], len(target_chans), task_phase[1]-task_phase[0]))
            multiesrca_test = np.zeros((X_test_reshape.shape[0], X_test_reshape.shape[1], len(target_chans), task_phase[1]-task_phase[0]))
            for nc,tc in enumerate(target_chans):
                multiesrca_train[...,nc,:] = srca_cpu.apply_ESRCA(
                    rest_data=X_train_reshape[...,rest_phase[0]:rest_phase[1]],
                    task_data=X_train_reshape[...,task_phase[0]:task_phase[1]],
                    target_chan=tc,
                    model_chans=multiesrca_model_chans[nc],
                    chan_info=chan_info
                )
                multiesrca_test[...,nc,:] = srca_cpu.apply_ESRCA(
                    rest_data=X_test_reshape[...,rest_phase[0]:rest_phase[1]],
                    task_data=X_test_reshape[...,task_phase[0]:task_phase[1]],
                    target_chan=tc,
                    model_chans=multiesrca_model_chans[nc],
                    chan_info=chan_info
                )
            X_multiesrca_train, y_multiesrca_train = utils.reshape_dataset(multiesrca_train)
            X_multiesrca_test, y_multiesrca_test = utils.reshape_dataset(multiesrca_test)
            trca_classifier = trca.TRCA().fit(
                X_train=X_multiesrca_train,
                y_train=y_multiesrca_train
            )
            _, y_trca, _, y_etrca = trca_classifier.predict(
                X_test=X_multiesrca_test,
                y_test=y_multiesrca_test
            )
            acc_mesrca_trca[nlen,nsam,nrep] = utils.acc_compute(y_trca, y_multiesrca_test)
            acc_mesrca_etrca[nlen,nsam,nrep] = utils.acc_compute(y_etrca, y_multiesrca_test)

            dsp_classifier = special.DSP().fit(
                X_train=X_multiesrca_train,
                y_train=y_multiesrca_train
            )
            _, y_dsp = dsp_classifier.predict(
                X_test=X_multiesrca_test,
                y_test=y_multiesrca_test
            )
            acc_mesrca_dsp[nlen,nsam,nrep] = utils.acc_compute(y_dsp, y_multiesrca_test)
        print('Finish training sample: {}'.format(str(sample)))
    print('Finish data length: {}'.format(str(data_length)))

# save results
result_path = r'D:\SSVEP\Results\20230226\20230219-wqy_circle(normal).mat'
io.savemat(result_path, {'TRCA':acc_trca, 'eTRCA':acc_etrca, 'DSP':acc_dsp,
                         'eSRCA-TRCA':acc_esrca_trca, 'eSRCA-eTRCA':acc_esrca_etrca, 'eSRCA-DSP':acc_esrca_dsp,
                         'Multi-eSRCA-TRCA':acc_mesrca_trca, 'Multi-eSRCA-eTRCA':acc_mesrca_etrca, 'Multi-eSRCA-DSP':acc_esrca_dsp,
                         'Log file':log_file})
