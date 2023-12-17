# load modules
import os
import string
import sys

import numpy as np
import numpy.matlib
import math
import scipy.io as sio
import warnings
from scipy import signal
from sklearn.cross_decomposition import CCA
warnings.filterwarnings('default')
# update logging
# 2022.12.13  trca_matrix() trca_matrix_sc() update numpy.matlab.repmat to np.tile (same results)
# 2022.12.13  Refine the annotation of code
# 2023.1.6 acc_calculate() label_target/predict 'int' np.ones((nTrials, 1),int)
"""
base
"""
def corr2(a, b):
    """ Solving for the two-dimensional correlation coefficient
    :param a: vector
    :param b: vector

    :return: correlation coefficient
    """
    a = a - np.sum(a) / np.size(a)
    b = b - np.sum(b) / np.size(b)
    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r

def acc_calculate(predict):
    """ Calculate accuracy
    :param predict:  (n_trial,n_event)

    :return: acc
    """
    [nTrials, nEvents] = predict.shape
    label_target = np.ones((nTrials, 1),int) * np.arange(0, nEvents, 1, int)
    logical_right = (label_target == predict)
    acc_num = np.sum(logical_right != 0)
    acc = acc_num / nTrials / nEvents
    return acc

def cal_CCA(X, Y):
    """ CCA count
    :param X: (num of sample points, num of channels1 )
    :param Y: (num of sample points, num of channels2 )

    :return:
    """

    #  Center the variables
    X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))
    if Y.ndim == 1:
        Ynew = Y.reshape(Y.shape[0], 1)  # reshape（num of sample points）to（num of sample points * 1）
        del Y
        Y = Ynew
        Y = Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1))
    else:
        Y = Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1))

    #  Calculate corr_cca
    P_U = Y @ np.linalg.inv(Y.T @ Y) @ Y.T
    b_U = (np.linalg.inv(X.T @ X)) @ (X.T @ P_U @ X)
    [eig_value_U, eig_U] = np.linalg.eig(b_U)  # Calculate U
    U = eig_U[:, np.argmax(eig_value_U)]
    P_V = X @ np.linalg.inv(X.T @ X) @ X.T
    b_V = (np.linalg.inv(Y.T @ Y)) @ (Y.T @ P_V @ Y)
    [eig_value_V, eig_V] = np.linalg.eig(b_V)  # Calculate V
    V = eig_V[:, np.argmax(eig_value_V)]

    corr = np.corrcoef(U.T @ X.T, V.T @ Y.T)
    corr_cca = corr[0, 1]
    return U.real, V.real, corr_cca.real

"""
load data
"""
class PreProcessing():
    """Adopted from OrionHH
    load data for Benchmark dataset
    """
    CHANNELS = [
        'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3',
        'F1','FZ','F2','F4','F6','F8','FT7','FC5',
        'FC3','FC1','FCZ','FC2','FC4','FC6','FC8','T7',
        'C5','C3','C1','CZ','C2','C4','C6','T8',
        'M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4',
        'CP6','TP8','M2','P7','P5','P3','P1','PZ',
        'P2','P4','P6','P8','PO7','PO5','PO3','POZ',
        'PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
    ] # M1: 33. M2: 43.

    def __init__(self, filepath,  t_begin, t_end, n_classes=40, fs_down=250, chans=None, num_filter=1):

        self.filepath = filepath
        self.fs_down = fs_down
        self.t_begin = t_begin
        self.t_end = t_end
        self.chans = chans
        self.n_classes = n_classes
        self.num_filter = num_filter

    def load_data(self):
        '''
        Application: load data and selected channels by chans.
        :param chans: list | None
        :return: raw_data: 4-D, numpy
            n_chans, n_samples, n_classes, n_trials
        :return: event: 2-D, numpy
            event[0, :]: label
            event[1, :]: latency
        '''
        raw_mat = sio.loadmat(self.filepath)
        raw_data = raw_mat['data']  # (64, 1500, 40, 6)
        # begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)

        idx_loc = list()
        if isinstance(self.chans, list):
            for _, char_value in enumerate(self.chans):
                idx_loc.append(self.CHANNELS.index(char_value.upper()))

        raw_data = raw_data[idx_loc, : , : , :] if idx_loc else raw_data

        self.raw_fs = 250  # .mat sampling rate

        return raw_data

    def resample_data(self, raw_data):
        '''
        :param raw_data: from method load_data.
        :return: raw_data_resampled, 4-D, numpy
            n_chans, n_samples, n_classes, n_trials
        '''
        if self.raw_fs > self.fs_down:
            raw_data_resampled = signal.resample(raw_data, round(self.fs_down*raw_data.shape[1]/self.raw_fs), axis=1)
        elif self.raw_fs < self.fs_down:
            warnings.warn('You are up-sampling, no recommended')
            raw_data_resampled = signal.resample(raw_data, round(self.fs_down*raw_data.shape[1]/self.raw_fs), axis=1)
        else:
            raw_data_resampled = raw_data

        return raw_data_resampled

    def _get_iir_sos_band(self, w_pass, w_stop):
        '''
        Get second-order sections (like 'ba') of Chebyshev type I filter.
        :param w_pass: list, 2 elements
        :param w_stop: list, 2 elements
        :return: sos_system
            i.e the filter coefficients.
        '''
        if len(w_pass) != 2 or len(w_stop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if w_pass[0] > w_pass[1] or w_stop[0] > w_stop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

        if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')

        wp = [2 * w_pass[0] / self.fs_down, 2 * w_pass[1] / self.fs_down]
        ws = [2 * w_stop[0] / self.fs_down, 2 * w_stop[1] / self.fs_down]
        gpass = 4
        gstop = 30  # dB

        N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
        sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')

        return sos_system

    def filtered_data_iir(self, w_pass_2d, w_stop_2d, data):
        '''
        filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data: 4-d, numpy, from method load_data or resample_data.
            n_chans * n_samples * n_classes * n_trials
        :return: filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans * n_samples * n_classes * n_trials.
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.num_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)
        sos_system = dict()
        filtered_data = dict()
        for idx_filter in range(self.num_filter):
            sos_system['filter'+str(idx_filter+1)] = self._get_iir_sos_band(w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                                                                            w_stop=[w_stop_2d[0, idx_filter],
                                                                                    w_stop_2d[1, idx_filter]])
            filter_data = signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=1)
            filtered_data['bank'+str(idx_filter+1)] = filter_data[:,begin_point:end_point,:,:]

        return filtered_data

    def filtered_data_iir111(self, w_pass_2d, w_stop_2d, data):
        '''
        filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data: 4-d, numpy, from method load_data or resample_data.
            n_chans * n_samples * n_classes * n_trials
        :return: filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans * n_samples * n_classes * n_trials.
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.num_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)
        data = data[:,begin_point:end_point,:,:]

        sos_system = dict()
        filtered_data = dict()
        for idx_filter in range(self.num_filter):
            sos_system['filter'+str(idx_filter+1)] = self._get_iir_sos_band(w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                                                                            w_stop=[w_stop_2d[0, idx_filter],
                                                                                    w_stop_2d[1, idx_filter]])
            filter_data = signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=1)
            filtered_data['bank'+str(idx_filter+1)] = filter_data

        return filtered_data

class PreProcessing_BETA():
    """Adopted from OrionHH
       load data for BETA dataset
    """
    CHANNELS = [
        'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3',
        'F1','FZ','F2','F4','F6','F8','FT7','FC5',
        'FC3','FC1','FCZ','FC2','FC4','FC6','FC8','T7',
        'C5','C3','C1','CZ','C2','C4','C6','T8',
        'M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4',
        'CP6','TP8','M2','P7','P5','P3','P1','PZ',
        'P2','P4','P6','P8','PO7','PO5','PO3','POZ',
        'PO4','PO6','PO8','CB1','O1','OZ','O2','CB2'
    ] # M1: 33. M2: 43.

    def __init__(self, filepath,  t_begin, t_end, n_classes=40, fs_down=250, chans=None, num_filter=1):

        self.filepath = filepath
        self.fs_down = fs_down
        self.t_begin = t_begin
        self.t_end = t_end
        self.chans = chans
        self.n_classes = n_classes
        self.num_filter = num_filter

    def load_data(self):
        '''
        Application: load data and selected channels by chans.
        :param chans: list | None
        :return: raw_data: 4-D, numpy
            n_chans, n_samples, n_classes, n_trials
        :return: event: 2-D, numpy
            event[0, :]: label
            event[1, :]: latency
        '''
        raw_mat = sio.loadmat(self.filepath)
        raw_data11 = raw_mat['data']  # (64, 750, 4, 40)
        data = raw_data11[0,0]['EEG']
        raw_data = np.transpose(data,[0,1,3,2])

        idx_loc = list()
        if isinstance(self.chans, list):
            for _, char_value in enumerate(self.chans):
                idx_loc.append(self.CHANNELS.index(char_value.upper()))

        raw_data = raw_data[idx_loc, : , : , :] if idx_loc else raw_data

        self.raw_fs = 250  # .mat sampling rate

        return raw_data

    def resample_data(self, raw_data):
        '''
        :param raw_data: from method load_data.
        :return: raw_data_resampled, 4-D, numpy
            n_chans, n_samples, n_classes, n_trials
        '''
        if self.raw_fs > self.fs_down:
            raw_data_resampled = signal.resample(raw_data, round(self.fs_down*raw_data.shape[1]/self.raw_fs), axis=1)
        elif self.raw_fs < self.fs_down:
            warnings.warn('You are up-sampling, no recommended')
            raw_data_resampled = signal.resample(raw_data, round(self.fs_down*raw_data.shape[1]/self.raw_fs), axis=1)
        else:
            raw_data_resampled = raw_data

        return raw_data_resampled

    def _get_iir_sos_band(self, w_pass, w_stop):
        '''
        Get second-order sections (like 'ba') of Chebyshev type I filter.
        :param w_pass: list, 2 elements
        :param w_stop: list, 2 elements
        :return: sos_system
            i.e the filter coefficients.
        '''
        if len(w_pass) != 2 or len(w_stop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if w_pass[0] > w_pass[1] or w_stop[0] > w_stop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

        if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')

        wp = [2 * w_pass[0] / self.fs_down, 2 * w_pass[1] / self.fs_down]
        ws = [2 * w_stop[0] / self.fs_down, 2 * w_stop[1] / self.fs_down]
        gpass = 4
        gstop = 30  # dB

        N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
        sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')

        return sos_system

    def filtered_data_iir(self, w_pass_2d, w_stop_2d, data):
        '''
        filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data: 4-d, numpy, from method load_data or resample_data.
            n_chans * n_samples * n_classes * n_trials
        :return: filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans, n_samples, n_classes, n_trials.
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.num_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)
        sos_system = dict()
        filtered_data = dict()
        for idx_filter in range(self.num_filter):
            sos_system['filter'+str(idx_filter+1)] = self._get_iir_sos_band(w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                                                                            w_stop=[w_stop_2d[0, idx_filter],
                                                                                    w_stop_2d[1, idx_filter]])
            filter_data = signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=1)
            filtered_data['bank'+str(idx_filter+1)] = filter_data[:,begin_point:end_point,:,:]

        return filtered_data

    def filtered_data_iir111(self, w_pass_2d, w_stop_2d, data):
        '''
        filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data: 4-d, numpy, from method load_data or resample_data.
            n_chans * n_samples * n_classes * n_trials
        :return: filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans, n_samples, n_classes, n_trials.
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.num_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)
        data = data[:,begin_point:end_point,:,:]

        sos_system = dict()
        filtered_data = dict()
        for idx_filter in range(self.num_filter):
            sos_system['filter'+str(idx_filter+1)] = self._get_iir_sos_band(w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                                                                            w_stop=[w_stop_2d[0, idx_filter],
                                                                                    w_stop_2d[1, idx_filter]])
            filter_data = signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=1)
            filtered_data['bank'+str(idx_filter+1)] = filter_data

        return filtered_data

"""
Algorithm
"""
## sCCA
def sCCA_test(test_data, f_list, fs, Nh):
    """ Canonical Correlation Analysis
        predict of singe block
    :param testData: test_data of multi trials
           ndarray(n_channels, n_sample_points, n_trials(equals to n_events))
    :param f_list: the all frequency of reference signal
    :param fs: the sample frequency
    :param Nh: the number of harmonics

    :return: predict of singe block
           ndarray(n_events, n_classes)
    """
    [nChannels, nTimes, nEvents] = test_data.shape
    rr = np.zeros((nEvents,nEvents))
    for m in range(nEvents):  # the m-th test data
        r = np.zeros(nEvents)
        for n in range(nEvents):  # the n-th train model
            test = test_data[:, :, m]
            f = f_list[n]
            #  Generate reference signal Yf
            Ts = 1 / fs
            n_list = np.arange(nTimes) * Ts
            Yf = np.zeros((nTimes, Nh * 2))
            for iNh in range(Nh):
                y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n_list)
                Yf[:, iNh * 2] = y_sin
                y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n_list)
                Yf[:, iNh * 2 + 1] = y_cos
            # CCA model
            _, _, corr_cca = cal_CCA(test.T, Yf)
            r[n] = np.abs(corr_cca)
        rr[m,:] = r
    return rr

## eCCA
def SS_eCCA_test(Data, mean_temp, fs, f, Nh):
    """
    :param Data: test_data of singe trial
           ndarray(n_channels, n_sample_points)
    :param mean_temp: Average template
           ndarray(n_channels, n_sample_points)
    :param fs: the sample frequency
    :param f:  the frequency of reference signal
    :param Nh: the number of harmonics

    :return: corr_ecca
    """

    Data = np.transpose(Data,[1,0])  # num of sample points * num of channels
    mean_temp = np.transpose(mean_temp, [1,0])
    #  Generate reference signal Yf
    Ts = 1 / fs
    nTimes = Data.shape[0]
    n = np.arange(nTimes) * Ts
    Yf = np.zeros((nTimes, Nh * 2))
    for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n)
        Yf[:, iNh * 2] = y_sin
        y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n)
        Yf[:, iNh * 2 + 1] = y_cos

    w1, w4, l1 = cal_CCA(Data, mean_temp)
    w2, w5, l2 =  cal_CCA(Data, Yf)
    w3, _, l3 =  cal_CCA(mean_temp, Yf)
    if l1 <=0:
        w4 = -w4
    if l2 <=0:
        w5 = -w5
    r1 = np.corrcoef(w2.T @ Data.T, w5.T @ Yf.T)
    r2 = np.corrcoef(w1.T @ Data.T, w1.T @ mean_temp.T)
    r3 = np.corrcoef(w2.T @ Data.T, w2.T @ mean_temp.T)
    r4 = np.corrcoef(w3.T @ Data.T, w3.T @ mean_temp.T)
    r5 = np.corrcoef(w1.T @ mean_temp.T, w4.T @ mean_temp.T)
    rr1 = r1[0, 1]
    rr2 = r2[0, 1]
    rr3 = r3[0, 1]
    rr4 = r4[0, 1]
    rr5 = r5[0, 1]

    corr_ecca = ( np.sign(rr1)* (rr1)**2 + np.sign(rr2)* (rr2)**2 +
                np.sign(rr3)* (rr3)**2 + np.sign(rr4)* (rr4)**2 + np.sign(rr5)* (rr5)**2 )

    return corr_ecca

def eCCA_test(test_data, mean_temp ,fs , f_list ,Nh):
    """ predict of singe block
    :param test_data: test_data of multi trials
           ndarray(n_channels, n_sample_points, n_trials(equals to n_events))
    :param mean_temp: Average template of all classes
           ndarray(n_channels, n_sample_points, n_events)
    :param fs: the sample frequency
    :param f_list: the all frequency of reference signal
    :param Nh: the number of harmonics

    :return: predict of singe block
           ndarray(n_events, n_classes)
    """

    [nChannels, nTimes, nEvents] = test_data.shape
    rr = np.zeros((nEvents,nEvents))
    for m in range(nEvents):  # the m-th test data
        r = np.zeros(nEvents)
        for n in range(nEvents):  # the n-th train model
            test = test_data[:, :, m]
            train = mean_temp[:, :, n]
            r[n] = SS_eCCA_test(test, train, fs=fs, f=f_list[n], Nh=Nh)
        rr[m,:] = r
    return rr

## TRCA
def trca_matrix(data):
    """ Task-related component analysis (TRCA)
    :param data: Multi-trial EEG signals under the same task
           ndarray(n_channels, n_sample_points, n_trials)

    :return: w: spatial filter
           ndarray(n_channels, 1)
    """

    X = data
    # X_mean = X.mean(axis=1, keepdims=True)
    # X = X - X_mean
    nChans = X.shape[0]
    nTimes = X.shape[1]
    nTrial = X.shape[2]
    #  solve S
    S = np.zeros((nChans, nChans))
    for i in range(nTrial):
        for j in range(nTrial):
            if (i != j):
                x_i = X[:, :, i]
                x_j = X[:, :, j]
                S = S + np.dot(x_i, (x_j.T))
    #  solve Q (It does not matter whether the mean is removed or not)
    X1 = X.reshape([nChans, nTimes * nTrial], order='F')
    # a = numpy.matlib.repmat(np.mean(X1, 1), X1.shape[1], 1)
    a  = np.tile(np.mean(X1, 1), (X1.shape[1], 1))  # repeat mean value of each channel(np.mean(X1, 1) is row vector)
    X1 = X1 - a.T
    Q = X1 @ X1.T
    #  get eigenvector
    b = np.dot(np.linalg.inv(Q), S)
    [eig_value, eig_w] = np.linalg.eig(b)  # in matlab：a/b = inv(a)*b

    # Descending order
    eig_w = eig_w[:, eig_value.argsort()[::-1]]  # return indices in ascending order and reverse
    eig_value.sort()
    eig_value = eig_value[::-1]  # sort in descending

    w = eig_w[:, 0]
    return w.real

def TRCA_train(trainData):
    """ Get TRCA spatial filters and average templates for all classes
    :param trainData: training data of all events
            ndarray(n_channels, n_sample_points, n_events, n_trials)

    :return: w: （n_channels, n_events）
             mean_temp （n_channels, n_sample_points, n_events）
    """

    [nChannels, nTimes, nEvents, nTrials] = trainData.shape
    # get w
    w = np.zeros((nChannels, nEvents))
    for i in range(nEvents):
        w_data = trainData[:, :, i, :]
        w1 = trca_matrix(w_data)
        w[:, i] = w1
    # get mean temps
    mean_temp = np.zeros((nChannels, nTimes, nEvents))
    mean_temp = np.mean(trainData, -1)
    return w, mean_temp

def TRCA_test(testData, w, mean_temp, ensemble):
    """ predict of singe block
    :param testData: test_data of multi trials
           ndarray(n_channels, n_sample_points, n_trials(equals to n_events))
    :param w: Spatial Filters
           ndarray(n_channels, n_events)
    :param mean_temp: Average template
           ndarray(n_channels, n_sample_points, n_events)
    :param ensemble: bool

    :return: predict of singe block
           ndarray(n_events,n_classes)
    """
    [nChannels, nTimes, nEvents] = testData.shape
    rr = np.zeros((nEvents, nEvents))
    for m in range(nEvents):  # the m-th test data
        test = testData[:, :, m]
        # Calculate the vector of correlation coefficients
        r = np.zeros(nEvents)
        for n in range(nEvents):  # the n-th train model
            train = mean_temp[:, :, n]
            if ensemble is True:
                r[n] = corr2(train.T @ w, test.T @ w)
            else:
                r[n] = corr2(train.T @ w[:, n], test.T @ w[:, n])
        rr[m, :] = r
    return rr

## TDCA
def get_P(f_list, Nh, sTime, sfreq):
    """ Get the projection matrix P for all classes
    :param f_list: the frequency of all events
    :param Nh: number of harmonics
    :param sTime: signal duration
    :param sfreq: sampling rate

    :return: P: the projection matrix P for all classes
             ndarray(n_Times, n_Times, n_Events)
    """
    nEvent = f_list.shape[0]
    P = np.zeros((int(sTime * sfreq), int(sTime * sfreq), nEvent))
    for iievent in range(nEvent):
        #  Generate reference signal Yf
        f = f_list[iievent]
        nTime = int(sTime * sfreq)
        Ts = 1 / sfreq
        n = np.arange(nTime) * Ts
        Yf = np.zeros((nTime, 2 * Nh))
        for iNh in range(Nh):
            y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n)
            Yf[:, iNh * 2] = y_sin
            y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n)
            Yf[:, iNh * 2 + 1] = y_cos
        q, _ = np.linalg.qr(Yf, mode='reduced')
        P[:, :, iievent] = q @ q.T

    return P

def tdca_matrix(data, Nk):
    """ Task-discriminant component analysis (TDCA)
    :param data: training data of all events
           ndarray(n_channels * (l + 1), 2*n_points,  n_events, n_trials)
    :param Nk: the number of subspaces
           int

    :return: w: Spatial Filters
           ndarray(n_channels * (l + 1), Nk)
    """

    X_aug_2 = np.transpose(data, [2, 3, 0, 1])  # nEvents, nTrials, nChannels * (l + 1), 2*npoints
    [n_events, n_trials, _, _] = X_aug_2.shape
    # get Sb
    class_center = X_aug_2.mean(axis=1)  # # nEvents , nChannels * (l + 1), 2*npoints
    total_center = class_center.mean(axis=0,keepdims=True)  # 1, nChannels * (l + 1), 2*npoints
    Hb = class_center - total_center   # Broadcasting in numpy
    Sb = np.einsum('ecp, ehp->ch', Hb, Hb)
    Sb /= n_events
    # get Sw
    class_center = np.expand_dims(class_center, 1)  # nEvents , 1, nChannels * (l + 1), 2*npoints
    Hw = X_aug_2 - np.tile(class_center, [1, n_trials, 1, 1]) # nEvents , nTrials, nChannels * (l + 1), 2*npoints
    Sw = np.einsum('etcp, ethp->ch', Hw, Hw)
    Sw /= (n_events * n_trials)
    Sw = 0.001 * np.eye(Hw.shape[2]) + Sw  # regularization
    #  get eigenvector
    b = np.dot(np.linalg.inv(Sw), Sb)
    [eig_value, eig_w] = np.linalg.eig(b)

    # Descending order
    eig_w = eig_w[:, eig_value.argsort()[::-1]]  # return indices in ascending order and reverse
    eig_value.sort()
    eig_value = eig_value[::-1]  # sort in descending
    w = eig_w[:, :Nk]

    return w

def TDCA_train(trainData, P, l, Nk):
    """ Get TDCA spatial filters and average templates for all classes
    :param trainData: training data of all events
           ndarray(n_channels, (n_sample_points + l), n_events, n_trials)
    :param P: projection matrix for all classes
           ndarray(n_sample_points, n_sample_points, n_events)
    :param l: delay point
    :param Nk: the number of subspaces

    :return: w: Spatial Filters
           ndarray(n_channels * (l + 1), Nk)
             mean_temp: average templates
           ndarray(Nk, 2 * n_sample_points, n_events)

    """

    [nChannels, nTimes, nEvents, nTrials] = trainData.shape
    npoints = nTimes - l

    data_aug_2 = np.zeros((nChannels * (l + 1), 2 * npoints, nEvents, nTrials))
    for ievent in range(nEvents):
        dat = trainData[:, :, ievent, :]
        # first
        dat_aug_1 = np.zeros((nChannels * (l + 1), npoints, nTrials))
        for il in range(l + 1):
            dat_aug_1[il * (nChannels):(il + 1) * nChannels, :, :] = dat[:, il:(il + npoints), :]
        # second
        dat_p = np.zeros_like((dat_aug_1))
        for itrial in range(nTrials):
            dat_p[:, :, itrial] = dat_aug_1[:, :, itrial] @ P[:, :, ievent]  # projection
        dat_aug_2 = np.concatenate((dat_aug_1, dat_p), axis=1, out=None)
        #
        data_aug_2[..., ievent, :] = dat_aug_2

    # get w
    w = tdca_matrix(data_aug_2, Nk=Nk)
    # get mean temps   Nk * 2 Num of sample points * num of events
    mean_tem = np.zeros((Nk, npoints * 2, nEvents))
    mean_data = np.mean(data_aug_2, -1)
    for i in range((nEvents)):
        mean_tem[:, :, i] = w.T @ mean_data[:, :, i]

    return w, mean_tem

def TDCA_test(testData, w, mean_temp, P, l):
    """ predict of singe block
    :param testData: test_data of multi trials
           ndarray(n_channels, n_sample_points, n_trials(equals to n_events)
    :param w: Spatial Filters
           ndarray(n_channels * (l + 1), Nk)
    :param mean_temp: Average template
           ndarray(Nk, 2 * n_sample_points , n_events)
    :param P:  projection matrix for all classes
           ndarray(n_sample_points, n_sample_points, n_events)
    :param l:  delay point

    :return: predict of singe block
           ndarray(n_events,n_classes)
    """

    [nChannels, nTimes, nEvents] = testData.shape
    rr = np.zeros((nEvents, nEvents))
    for m in range(nEvents):  # the m-th test data
        test = testData[:, :, m]
        # first
        test_aug_1 = np.zeros((nChannels * (l + 1), nTimes))
        aug_zero = np.zeros((nChannels, l))  # Splice 0 matrix
        test = np.concatenate((test, aug_zero), axis=1, out=None)  # nChannels, nTimes + l
        for il in range(l + 1):
            test_aug_1[il * (nChannels):(il + 1) * nChannels, :] = test[:, il:(il + nTimes)]
        # Calculate the vector of correlation coefficients
        r = np.zeros(nEvents)
        for n in range(nEvents):  # the n-th train model
            # second
            dat_p = test_aug_1 @ P[:, :, n]
            test_aug_2 = np.concatenate((test_aug_1, dat_p), axis=1, out=None)
            # slove rr
            train = mean_temp[:, :, n]
            r[n] = corr2(train, w.T @ test_aug_2)
        rr[m, :] = r
    return rr

## msCCA
def ms_eCCA_spatialFilter(mean_temp_all , iEvent , nTemplates,  fs  , f_list , phi_list ,Nh ):
    """ spatialFilter for multi-stimulus extended canonical correlation analysis (ms-eCCA)
    adopted from https://github.com/edwin465/SSVEP-MSCCA-MSTRCA
    :param mean_temp_all: ndarray(n_channels, n_times, n_events)
    :param iEvent:        the i-th event for the selection of neighboring frequencies
    :param nTemplates:    the number of neighboring frequencies
    :param fs:            the sample rate
    :param f_list:        the all frequency of reference signal
    :param phi_list:      the all phase of reference signal
    :param Nh:            the number of harmonics

    :return:  U, V        Spatial Filters
    """

    mean_temp_all = np.transpose(mean_temp_all, [1,0,2])

    [nTimes, nChannels, nEvents] = mean_temp_all.shape
    d0 = nTemplates/2
    d1 = nEvents

    n = iEvent + 1
    if n <= d0:
        template_st = 1
        template_ed = nTemplates
    elif (n > d0) & (n < (d1 - d0 + 1)):
        template_st = n - d0
        template_ed = n + (nTemplates - d0 - 1)
    else:
        template_st = (d1 - nTemplates + 1)
        template_ed = d1
    template_st = int(template_st-1)
    template_ed = int(template_ed)

    #  Concatenation of the templates (or sine-cosine references)
    mscca_ref = np.zeros((nTemplates *nTimes, 2 * Nh) )
    mscca_template = np.zeros((nTemplates *nTimes,  nChannels))

    index = 0
    for j in range(template_st, template_ed, 1):
        # sine-cosine references
        f = f_list[j]
        phi = phi_list[j]
        Ts = 1/fs
        n = np.arange(nTimes)*Ts
        Yf = np.zeros((nTimes , Nh*2))
        for iNh in range(Nh):
            y_sin = np.sin(2*np.pi*f*(iNh+1)*n+ (iNh+1)*np.pi*phi)
            Yf[:,iNh*2] = y_sin
            y_cos = np.cos(2*np.pi*f*(iNh+1)*n+ (iNh+1)*np.pi*phi)
            Yf[:,iNh*2+1] = y_cos
        mscca_ref[index * nTimes: (index+1) * nTimes , :] = Yf
        # templates
        ss = mean_temp_all[:, :, j]
        # ss = ss - np.tile(np.mean(ss, 0), (ss.shape[0], 1))
        mscca_template[index * nTimes:(index + 1) * nTimes, :] = ss
        index = index + 1

    # calculate U and V
    U, V, r = cal_CCA(mscca_template, mscca_ref)

    if r < 0:   # Symbol Control
        V = -V

    return U, V

def SS_mseCCA_test(test_data_S, mean_temp ,u ,v , fs  , f , phi ,Nh):
    """
    :param test_data_S: test_data of singe trial
           ndarray(n_channels, n_sample_points)
    :param mean_temp: Average template
           ndarray(n_channels, n_sample_points)
    :param u: spatial Filter for EEG
           ndarray(n_channels,)
    :param v: spatial Filter for Yf
           ndarray(2n_harmonics,)
    :param fs: the sample frequency
    :param f:  the frequency of reference signal
    :param phi: the phase of reference signal
    :param Nh: the number of harmonics

    :return: correlation coefficient
    """

    # calculate corr_ms_eCCA
    nTimes = test_data_S.shape[-1]

    Ts = 1/fs
    n = np.arange(nTimes)*Ts
    Yf_cca = np.zeros((nTimes , Nh*2))
    for iNh in range(Nh):
        y_sin = np.sin(2*np.pi*f*(iNh+1)*n+(iNh+1)*np.pi*phi)
        Yf_cca[:,iNh*2] = y_sin
        y_cos = np.cos(2*np.pi*f*(iNh+1)*n+(iNh+1)*np.pi*phi)
        Yf_cca[:,iNh*2+1] = y_cos

    rr1 = np.corrcoef(u.T @ test_data_S, v.T @ Yf_cca.T)
    rr2 = np.corrcoef(u.T @ test_data_S, u.T @ mean_temp)
    rr1_abs = rr1[0, 1]
    rr2_abs = rr2[0, 1]

    corr_ms_eCCA = np.sign(rr1_abs) * (rr1_abs) ** 2 + np.sign(rr2_abs) * (rr2_abs)** 2
    return corr_ms_eCCA

def mseCCA_test(test_data, mean_temp ,U ,V , fs  , f_list , phi_list ,Nh):
    """
    :param testData: test_data of multi trials
           ndarray(n_channels, n_sample_points, n_trials(equals to n_events))
    :param mean_temp: Average template of all classes
           ndarray（n_channels, n_sample_points, n_events）
    :param U: spatial Filter for EEG of all classes
           ndarray（n_channels, n_events）
    :param V: spatial Filter for Yf of all classes
           ndarray(2 * n_harmonics, n_events)
    :param fs: the sample frequency
    :param f_list: the all frequency of reference signal
    :param phi_list: the all phase of reference signal
    :param Nh: the number of harmonics

    :return: predict of singe block
           ndarray(n_events,n_classes)
    """

    [nChannels, nTimes, nEvents] = test_data.shape
    rr = np.zeros((nEvents,nEvents))
    for m in range(nEvents):  # the m-th test data
        test = test_data[:, :, m]
        # Calculate the vector of correlation coefficients
        r = np.zeros(nEvents)
        for n in range(nEvents):  # the n-th train model
            train = mean_temp[:, :, n]
            r[n] = SS_mseCCA_test(test, train ,u = U[:,n] ,v=V[:,n] , fs=fs  , f=f_list[n] , phi=phi_list[n] ,Nh=Nh)
        rr[m,:] = r
    return rr

## msTRCA
def ms_eTRCA_spatialFilter(data_all , iEvent , nTemplates):
    """ spatialFilter for multi-stimulus task-related component analysis (ms-TRCA)
    :param data_all: training data of all events
            ndarray(n_channels, n_sample_points, n_events, n_trials)
    :param iEvent: the i-th event
    :param nTemplates:  the number of neighborhood frequency

    :return: w: spatial filter
           ndarray(n_channels, 1)
    """

    [nChannels,nTimes, nEvents,n_trials] = data_all.shape
    d0 = nTemplates/2
    d1 = nEvents

    n = iEvent + 1
    if n <= d0:
        template_st = 1
        template_ed = nTemplates
    elif (n > d0) & (n < (d1 - d0 + 1)):
        template_st = n - d0
        template_ed = n + (nTemplates - d0 - 1)
    else:
        template_st = (d1 - nTemplates + 1)
        template_ed = d1
    template_st = int(template_st-1)
    template_ed = int(template_ed)

    #  Concatenation of the data
    mstrca_X1 = []
    mstrca_X2 = []
    for j in range(template_st, template_ed, 1):
        trca_X1 = np.zeros((nChannels,nTimes))
        trca_X2 = []
        for i in range(n_trials):
        # data
            ss = data_all[:, :, j, i]
            trca_X1 = trca_X1+ ss
            trca_X2.append(ss.T)
        trca_X2_np = np.concatenate(trca_X2)
        mstrca_X1.append(trca_X1)
        mstrca_X2.append(trca_X2_np.T)
    mstrca_X1_np = np.concatenate(mstrca_X1,1)
    mstrca_X2_np = np.concatenate(mstrca_X2,1)
    if n_trials == 1:
        S = np.zeros((nChannels,nChannels)) # At this point the spatial filter fails
    else:
        S = mstrca_X1_np @ mstrca_X1_np.T - mstrca_X2_np @ mstrca_X2_np.T
    Q = mstrca_X2_np @ mstrca_X2_np.T

    #  get eigenvector
    b = np.dot(np.linalg.inv(Q), S)
    [eig_value, eig_w] = np.linalg.eig(b)  # in matlab：a/b = inv(a)*b

    # Descending order
    eig_w = eig_w[:, eig_value.argsort()[::-1]]  # return indices in ascending order and reverse
    eig_value.sort()
    eig_value = eig_value[::-1]  # sort in descending

    w = eig_w[:, 0]
    return w

## msCCA+mseTRCA
def SS_mseCCA_msTRCA_test(test_data_S, mean_temp ,W, u ,v , fs  , f , phi ,Nh):
    """ msCCA+mseTRCA test
    :param test_data_S: test_data of singe trial
           ndarray(n_channels, n_sample_points)
    :param mean_temp: Average template
           ndarray(n_channels, n_sample_points)
    :param W: eTRCA spatial Filter
           ndarray(n_channels, n_events)
    :param u: msCCA spatial Filter for EEG
           ndarray(n_channels,)
    :param v: msCCA spatial Filter for Yf
           ndarray(2n_harmonics,)
    :param fs: the sample frequency
    :param f: the frequency of reference signal
    :param phi: the phase of reference signal
    :param Nh: the number of harmonics

    :return: correlation coefficient
    """

    # calculate corr_ms_eCCA
    nTimes = test_data_S.shape[-1]

    Ts = 1/fs
    n = np.arange(nTimes)*Ts
    Yf_cca = np.zeros((nTimes , Nh*2))
    for iNh in range(Nh):
        y_sin = np.sin(2*np.pi*f*(iNh+1)*n+(iNh+1)*np.pi*phi)
        Yf_cca[:,iNh*2] = y_sin
        y_cos = np.cos(2*np.pi*f*(iNh+1)*n+(iNh+1)*np.pi*phi)
        Yf_cca[:,iNh*2+1] = y_cos

    rr1 = np.corrcoef(u.T @ test_data_S, v.T @ Yf_cca.T)
    rr2 = corr2(W.T @ test_data_S, W.T @ mean_temp)
    rr1_abs = rr1[0,1]
    rr2_abs = rr2

    corr_ms_eCCA_TRCA = np.sign(rr1_abs) * (rr1_abs) ** 2 + np.sign(rr2_abs) * (rr2_abs)** 2
    return corr_ms_eCCA_TRCA

def mseCCA_msTRCA_test(test_data, mean_temp ,W, U ,V , fs  , f_list , phi_list ,Nh):
    """
    :param testData: test_data of multi trials
           ndarray(n_channels, n_sample_points, n_trials(equals to n_events))
    :param mean_temp: Average template of all classes
           ndarray（n_channels, n_sample_points, n_events）
    :param W: eTRCA spatial Filter
           ndarray(n_channels, n_events)
    :param U: msCCA spatial Filter for EEG of all classes
           ndarray(n_channels, n_events)
    :param V: msCCA spatial Filter for Yf of all classes
           ndarray(2 * n_harmonics, n_events)
    :param fs: the sample frequency
    :param f_list: the all frequency of reference signal
    :param phi_list: the all phase of reference signal
    :param Nh: the number of harmonics

    :return: predict of singe block
           ndarray(n_events,n_classes)
    """

    [nChannels, nTimes, nEvents] = test_data.shape
    rr = np.zeros((nEvents,nEvents))
    for m in range(nEvents):  # the m-th test data
        test = test_data[:, :, m]
        # Calculate the vector of correlation coefficients
        r = np.zeros(nEvents)
        for n in range(nEvents):  # the n-th train model
            train = mean_temp[:, :, n]
            r[n] = SS_mseCCA_msTRCA_test(test, train ,W = W,u = U[:,n] ,v=V[:,n] , fs=fs  , f=f_list[n] , phi=phi_list[n] ,Nh=Nh)
        rr[m,:] = r
    return rr

## TRCA-R
def trca_r_matrix(X1, Y1):
    """Task-related component analysis with Sine-cosine signal.
    :param   X (ndarray): (n_channels, n_times, n_trials ). Training dataset.
    :param    Y (ndarray): (n_times, 2*n_harmonics). Sine-cosine template.

    :return:  w (ndarray): (n_chans). Eigenvector refering to the largest eigenvalue.
    """
    X = X1.transpose(2,0,1)  #  n_trials, n_channels, n_times
    Y = Y1.transpose(1,0)    #  2*n_harmonics, n_times
    # basic information
    n_train = X.shape[0]
    n_chans = X.shape[1]
    n_points = X.shape[-1]

    # Q: inter-channel covariance
    Q = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for ntr in range(n_train):
        Q += X[ntr, ...] @ X[ntr, ...].T

    # P: orthogonal projection matrix
    P1 = Y.T @ np.linalg.solve((Y @ Y.T), Y)  # (Np,Np)

    # S: projected inter-channels' inter-trial covariance
    X_mean = X.mean(axis=0)  # (Nc,Np)
    S = X_mean @ P1 @ P1.T @ X_mean.T

    # GEPs
    e_val, e_vec = np.linalg.eig(np.linalg.solve(Q, S))
    w_index = np.argmax(e_val)
    w = e_vec[:, [w_index]].T
    return w.real

def TRCA_R_train(trainDataAll, Fre_list , fs  , Nh):
    """ Get TRCA-R spatial filters and average templates for all classes
    :param trainData: training data of all events
            ndarray (n_channels, n_sample_points, n_events, n_trials)
    :param Fre_list: the frequency of all class
    :param fs: the sample frequency
    :param Nh: the number of harmonics

    :return: w: (n_channels, n_events)
             mean_temp (n_channels, n_sample_points, n_events)
    """
    [nchannels, nTimes, nevents ,ntrials]= trainDataAll.shape
    w = np.zeros((nchannels,nevents))
    for iEvent in range(nevents):
        f = Fre_list[iEvent]
        train_Data = trainDataAll[:,:,iEvent,:] # n_channels * n_times * n_trials
        #  Generate reference signal Yf
        Ts = 1/fs
        n = np.arange(nTimes)*Ts
        Yf = np.zeros((nTimes , Nh*2))
        for iNh in range(Nh):
            y_sin = np.sin(2*np.pi*f*(iNh+1)*n)
            Yf[:,iNh*2] = y_sin
            y_cos = np.cos(2*np.pi*f*(iNh+1)*n)
            Yf[:,iNh*2+1] = y_cos

        # get w
        w0 = trca_r_matrix(train_Data,Yf )
        w[:, iEvent] = w0

    mean_temp = np.mean(trainDataAll,-1)

    return w, mean_temp

## scTRCA
def trca_matrix_sc(data):
    """ Similarity Constraints Task-related component analysis (scTRCA)
    :param data: Multi-trial EEG signals under the same task
           ndarray(n_channels, n_sample_points, n_trials)

    :return: S Q
           ndarray(n_channels, n_channels)
    """

    X = data
    # X_mean = X.mean(axis=1, keepdims=True)
    # X = X - X_mean
    nChans = X.shape[0]
    nTimes = X.shape[1]
    nTrial = X.shape[2]
    #  solve S
    S = np.zeros((nChans, nChans))
    for i in range(nTrial):
        for j in range(nTrial):
            if (i != j):
                x_i = X[:, :, i]
                x_j = X[:, :, j]
                S = S + np.dot(x_i, (x_j.T))
    #  solve Q (It does not matter whether the mean is removed or not)
    X1 = X.reshape([nChans, nTimes * nTrial], order='F')
    # a = numpy.matlib.repmat(np.mean(X1, 1), X1.shape[1], 1)
    a  = np.tile(np.mean(X1, 1), (X1.shape[1], 1))  # repeat mean value of each channel(np.mean(X1, 1) is row vector)
    X1 = X1 - a.T
    Q = X1 @ X1.T

    return S, Q

def scTRCA_train(trainDataAll, Fre_list , fs  , Nh):
    """ Get scTRCA spatial filters and average templates for all classes
    :param trainDataAll: training data of all events
            ndarray(n_channels, n_sample_points, n_events, n_trials)
    :param Fre_list: the all frequency of all class
    :param fs: the sample frequency
    :param Nh: the number of harmonics

    :return: w: ((n_channels + 2Nh), n_events)
             mean_temp (n_channels, n_sample_points, n_events)
    """
    [nchannels, ntimes, nevents ,ntrials]= trainDataAll.shape
    w = np.zeros(((nchannels+2*Nh),nevents))
    for iEvent in range(nevents):

        f = Fre_list[iEvent]
        train_Data = trainDataAll[:,:,iEvent,:]
        #  Generate reference signal Yf
        Ts = 1/fs
        n = np.arange(ntimes)*Ts
        Yf = np.zeros((ntimes , Nh*2))
        for iNh in range(Nh):
            y_sin = np.sin(2*np.pi*f*(iNh+1)*n)
            Yf[:,iNh*2] = y_sin
            y_cos = np.cos(2*np.pi*f*(iNh+1)*n)
            Yf[:,iNh*2+1] = y_cos    # Yf : Nt * 2Nh

        S11, Q1 = trca_matrix_sc(train_Data)
        if ntrials==1:
            S11 = S11
        else:
            S11 = S11 / ((ntrials)*(ntrials-1))
        Q1 =  Q1 / (ntrials)

        S = np.zeros((nchannels+2*Nh,nchannels+2*Nh))
        # cal S
        S12 = np.zeros((nchannels, 2*Nh))
        for itrial in range((ntrials)):
            S12 = S12 + train_Data[:,:,itrial] @ Yf
        S12 = S12 / (ntrials)
        S21 = S12.T
        S22 = Yf.T @ Yf
        S[0:nchannels,0:nchannels] = S11
        S[nchannels:, 0:nchannels] = S21
        S[0:nchannels,nchannels:] = S12
        S[nchannels:, nchannels:] = S22

        # cal Q
        Q = np.zeros((nchannels+2*Nh,nchannels+2*Nh))
        Q2 = S22
        Q[0:nchannels,0:nchannels] = Q1
        Q[nchannels:, nchannels:] = Q2

        #  get eigenvector
        b = np.dot(np.linalg.inv(Q), S)
        [eig_value, eig_w] = np.linalg.eig(b)  # in matlab：a/b = inv(a)*b
        # Descending order
        eig_w = eig_w[:, eig_value.argsort()[::-1]]  # return indices in ascending order and reverse
        eig_value.sort()
        eig_value = eig_value[::-1]  # sort in descending
        w1 = eig_w[:, 0]
        w[:, iEvent] = w1.real

    # get average templates for all classes
    mean_temp = np.mean(trainDataAll, -1)
    return w, mean_temp

def scTRCA_test(testData, w, mean_temp, ensemble, Fre_list , fs  , Nh):
    """
    :param testData: test_data of multi trials
           ndarray(n_channels, n_sample_points, n_trials(equals to n_events))
    :param w: Spatial Filters
           ndarray（n_channels, n_events）
    :param mean_temp: Average template
           ndarray（n_channels, n_sample_points, n_events）
    :param ensemble: bool
    :param Fre_list: the frequency of all class
    :param fs: the sample frequency
    :param Nh: the number of harmonics

    :return: predict of singe block
           ndarray(n_events,n_classes)
    """
    [nChannels, nTimes, nEvents] = testData.shape

    U = w[:nChannels, :]
    V = w[nChannels:, :]

    rr = np.zeros((nEvents, nEvents))
    for m in range(nEvents):   # the m-th test data
        r = np.zeros(nEvents)
        for n in range(nEvents):   # the n-th train model
            test = testData[:, :, m]
            train = mean_temp[:, :, n]
            #  Generate reference signal Yf
            f = Fre_list[n]
            Ts = 1 / fs
            nYf = np.arange(nTimes) * Ts
            Yf = np.zeros((nTimes, Nh * 2))
            for iNh in range(Nh):
                y_sin = np.sin(2 * np.pi * f * (iNh + 1) * nYf)
                Yf[:, iNh * 2] = y_sin
                y_cos = np.cos(2 * np.pi * f * (iNh + 1) * nYf)
                Yf[:, iNh * 2 + 1] = y_cos  # Yf : Nt * 2Nh

            if ensemble is True:
                r1 = corr2(train.T @ U, test.T @ U)
                r2 = corr2(Yf @ V, test.T @ U)
                r[n] = np.sign(r1) * (r1 ** 2) + np.sign(r2) * (r2 ** 2)
            else:
                r1 = corr2(train.T @ U[:, n], test.T @ U[:, n])
                r2 = corr2(test.T @ U[:, n], Yf @ V[:, n])
                r[n] = np.sign(r1) * (r1) ** 2 + np.sign(r2) * (r2) ** 2
        rr[m, :] = r
    return rr