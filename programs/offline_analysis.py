# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:27:48 2024

Re-write of offline_acc_120.py

@author: Brynhildr
"""

# %%
# cd D:\BaiduSyncdisk\程序\SSVEP_algorithms\programs
import utils
import trca
# import dsp

import os

import numpy as np
from numpy import ndarray
from scipy.fftpack import fft
from math import log

from scipy import signal

import mne
from mne.io import concatenate_raws
# from mne.filter import filter_data

from typing import Optional, List, Dict, Tuple, Union

import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

import pandas as pd


# %% Preprocessing functions
class PreProcessing(object):
    """Preprocessing for P300-SSVEP-mVEP paradigm."""

    CHANNELS = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
        'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
        'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FC8', 'T7',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
        'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
        'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
        'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]  # M1: 33. M2: 43.
    # TDMA order: 4-1-5-2-6-3
    TDMA = {'4': np.array([[True, False, False, False, False, False]]),
            '1': np.array([[False, True, False, False, False, False]]),
            '5': np.array([[False, False, True, False, False, False]]),
            '2': np.array([[False, False, False, True, False, False]]),
            '6': np.array([[False, False, False, False, True, False]]),
            '3': np.array([[False, False, False, False, False, True]])}
    FREQUENCY = [10.2, 16.6, 15.4, 13.0, 14.2,
                 12.6, 15.0, 13.8, 10.6, 17.0,
                 16.2, 13.4, 17.4, 11.0, 12.2,
                 14.6, 15.8, 11.8, 17.8, 11.4]
    PHASE = [np.mod((freq - 10.2) / 0.4 * 0.35, 2) for freq in FREQUENCY]

    def __init__(
            self,
            n_events: int = 36,
            n_rounds: int = 6,
            n_tdma: int = 6,
            tmin: float = -0.4,
            tmax: float = 1,
            t_ssvep: float = 0.7,
            sfreq: Union[int, float] = 1000,
            chan_info: Optional[List[str]] = None):
        """Config basic settings.

        Args:
            t_begin (float): Unknown.
            t_end (float): Unknown.
            n_events (int): The number of SSVEP classes.
                Defaults to 36.
            n_rounds (int): The number of repetitions of stimulation in a single trial.
                Defaults to 6.
            n_tdma (int): The number of P300-mVEP classes.
                Defaults to 6.
            tmin (float): See details in mne.Epoch().
            tmax (float): See details in mne.Epoch().
            t_ssvep (float): Time (s) of SSVEP stimulus for each round. Defaults to 0.7.
            sfreq (float): Sampling rate of preprocessed data. Defaults to 1000.
            chan_info (List[str], optional): Names of total channels.
        """
        self.n_events = n_events
        self.n_rounds = n_rounds
        self.n_tdma = n_tdma
        self.tmin = tmin
        self.tmax = tmax
        self.t_ssvep = t_ssvep
        self.sfreq = sfreq
        self.chan_info = chan_info

        # default number of commands
        self.n_chars = int(self.n_events * self.n_tdma)

        # total time of a complete trial for SSVEP data analysis
        # see details in extract_ssvep_trial()
        self.t_ssvep_trial = 1 + self.n_rounds * self.t_ssvep + 0.14 + 0.025

    def select_chans(self, chan_list: List[str]):
        """Select useful channels and convert them to indices.

        Args:
            chan_list (List[str]): Names of channels to be used.
        """
        self.chan_idx = []
        for cl in chan_list:
            self.chan_idx.append(self.CHANNELS.index(cl.upper()))

    def load_data(self, file_path: str):
        """Load multiple .cnt files and construct mne.Epoch() objects.
        Args
            file_path (str): The path of the folder to which the .cnt files belongs.
        """
        # load .cnt files into mne.Raw objects
        raw_cnts = []
        for cnt_file in os.listdir(file_path):
            data_path = os.path.join(file_path, cnt_file)

            raw_cnt = mne.io.read_raw_cnt(
                input_fname=data_path,
                eog=['HEO', 'VEO'],
                ecg=['EKG'],
                emg=['EMG'],
                preload=True,
                verbose=False
            )
            raw_cnt.filter(
                l_freq=0.1,
                h_freq=90,
                method='fir',
                picks='eeg',
                n_jobs='cuda'
            )  # remove slow drifts
            raw_cnts.append(raw_cnt)
        self.raw = concatenate_raws(raw_cnts)
        # layout picked channels' names
        self.picks = mne.pick_types(
            info=self.raw.info,
            emg=False,
            eeg=True,
            stim=False,
            eog=False
        )
        self.n_chans = len(self.picks)
        # cunstom mapping for extracting events (big & small labels)
        mapping = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}
        for char_idx in range(1, self.n_chars + 1):
            mapping[str(char_idx + 12)] = char_idx + 12
        self.events, self.events_id = mne.events_from_annotations(
            raw=self.raw,
            event_id=mapping
        )
        # check the integrity of label information
        self.n_labels = self.events.shape[0]  # total number of labels in .cnt files
        self.n_trial_labels = int(self.n_tdma * self.n_rounds + 1)  # small & big labels
        if np.mod(self.n_labels, self.n_trial_labels) != 0:
            raise Exception('Warning: Missing labels in some trials!')
        # re-arange continuous information of latencies & labels
        self.latencies = np.reshape(
            a=self.events[:, 0],
            newshape=(-1, self.n_trial_labels),
            order='C'
        )  # (n_trials, n_trial_labels)
        self.labels = np.reshape(
            a=self.events[:, 2],
            newshape=(-1, self.n_trial_labels),
            order='C'
        )  # (n_trials, n_trial_labels)
        self.locs = np.reshape(
            a=np.arange(self.n_labels),
            newshape=(-1, self.n_trial_labels),
            order='C'
        )  # (n_trials, n_trial_labels)
        self.n_trials = self.labels.shape[0]  # normally, n_trials = n_chars
        self.labels_char = self.labels[:, 0]  # big labels only

    def extract_ssvep_round(self) -> Tuple[ndarray, ndarray]:
        """
        Extract SSVEP data in rounds form: (n_trials, n_rounds, n_chans, n_points).
            n_points = tmax - tmin
            Usually, n_trials should be equal to n_chars.

        Returns:
            X_ssvep_round (ndarray): (n_trials, n_rounds, n_chans, n_points).
            y_ssvep_round (ndarray): (n_trials,). Labels for X_ssvep_round.
        """
        # extract SSVEP labels from P300-SSVEP-mVEP (character) labels
        y_ssvep_round = np.zeros((self.n_trials))
        for ncl, nl in enumerate(self.labels[:, 0]):
            temp_label = np.mod(nl - 12, self.n_events)
            # 1, n_events+1, 2*n_events+1,... belong to 1st type;
            # 2, n_events+2, 2*n_events+2,... belong to 2nd type, etc
            if temp_label == 0:  # np.mod(n_events, n_events) = 0
                y_ssvep_round[ncl] = self.n_events  # correct the '0th' type for SSVEP
            else:
                y_ssvep_round[ncl] = temp_label
        # cunstom mapping for extracting events at round-level
        mapping_ssvep = {'4': 4}  # SSVEP stimuli always start at label 4
        events, events_id = mne.events_from_annotations(
            raw=self.raw,
            event_id=mapping_ssvep
        )

        # extract X_ssvep_round from raw data
        X_ssvep_round = mne.Epochs(
            raw=self.raw,
            events=events,
            event_id=events_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
            picks=self.picks,
            preload=True,
            reject_by_annotation=False
        ).get_data() * 1e6  # (n_trials*n_rounds, n_chans, n_points)
        # decim = int(1000 / self.sfreq)
        # reshape X_ssvep_round into (n_trials, n_rounds, n_chans, n_points)
        loc_ssvep_round = np.reshape(
            a=np.arange(0, X_ssvep_round.shape[0]),
            newshape=(-1, self.n_rounds),
            order='C'
        )  # (n_trials, n_rounds)
        X_ssvep_round = X_ssvep_round[loc_ssvep_round, :, :]  # (..., n_chans, n_points)
        return X_ssvep_round, y_ssvep_round

    def extract_ssvep_trial(self) -> Tuple[ndarray, ndarray]:
        """
        Extract SSVEP data in trials form: (n_trials, n_chans, n_points).
            n_points_2 = 1 + 0.14 + t_ssvep * n_rounds + 0.025
            (1) 1 means starting from the character label (big label);
            (2) 0.14 is the commonly used visual pathway delay;
            (3) 0.025 is designed for TDCA-like algorithms.

        Returns:
            X_ssvep_trial (ndarray): (n_trials, n_chans, n_points).
            y_ssvep_trial (ndarray): (n_trials,). Labels for X_ssvep_trial.
        """
        # actually the same with y_ssvep_round
        y_ssvep_trial = np.zeros((self.n_trials))
        for ncl, nl in enumerate(self.labels[:, 0]):
            temp_label = np.mod(nl - 12, self.n_events)
            # 1, n_events+1, 2*n_events+1,... belong to 1st type;
            # 2, n_events+2, 2*n_events+2,... belong to 2nd type, etc
            if temp_label == 0:  # np.mod(n_events, n_events) = 0
                y_ssvep_trial[ncl] = self.n_events  # correct the '0th' type for SSVEP
            else:
                y_ssvep_trial[ncl] = temp_label

        # cunstom mapping for extracting events at trial-level
        mapping_ssvep = {}  # config trial (character) labels
        for char_idx in range(1, self.n_chars + 1):
            mapping_ssvep[str(char_idx + 12)] = char_idx + 12
        events, events_id = mne.events_from_annotations(
            raw=self.raw,
            event_id=mapping_ssvep
        )

        # extract X_ssvep_trial based on the character labels
        X_ssvep_trial = mne.Epochs(
            raw=self.raw,
            events=events,
            event_id=events_id,
            tmin=0,
            tmax=self.t_ssvep_trial,
            picks=self.picks,
            baseline=None,
            preload=True,
            reject_by_annotation=False
        ).get_data() * 1e6  # (n_trials, n_chans, n_points)
        return X_ssvep_trial, y_ssvep_trial

    def extract_p300_mvep(self) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Extract P300-mVEP data in rounds form: (n_trials, n_rounds, n_chans, n_points).
            n_points = tmax - tmin
            Usually, n_trials should be equal to n_chars.

        Returns:
            X_p300_mvep (ndarray): (n_trials, n_rounds, n_chans, n_points).
            y_p300_mvep (ndarray): (n_trials,). Labels for X_p300_mvep.
            y_tnt (ndarray): (n_trials, n_rounds).
                Target & non-target labels for X_p300_mvep (dtype=Bool).
        """
        # config P300-mVEP label according to TDMA design
        # label 1-20 belongs to 1st P300-mVEP type; 21-40 belongs to 2nd type, etc
        y_p300_mvep = np.zeros_like(self.labels_char)
        for nlc, lc in enumerate(self.labels_char):
            # 13 to n_events+12 belong to 1st P300-mVEP type;
            # n_events+13 to 2*n_events+12 belong to 2nd P300-mVEP type, etc
            y_p300_mvep[nlc] = (lc - 13) // self.n_events + 1
            # note that: n_events // n_events = 1 not 0, so use lc-13 to start from 0

        # config locations of P300-mVEP signal: (n_trials, n_rounds * n_tdma)
        locs_p300_mvep = np.delete(self.locs, 0, axis=1)  # delete big labels' location

        # config target & non-target ('tnt' for short) labels for P300-mVEP
        y_tnt = np.zeros_like(locs_p300_mvep) == 1  # initialization
        for ny, ypm in enumerate(y_p300_mvep):
            # tnt labels (TDMA[str(ypm)], 1 round) | repeat n_rounds times
            y_tnt[ny] = np.tile(A=self.TDMA[str(ypm)], reps=self.n_rounds)

        # extract X_p300_mvep based on events & locs_p300_mvep
        X_p300_mvep = mne.Epochs(
            raw=self.raw,
            events=self.events,
            event_id=self.events_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=(-0.2, 0),
            picks=self.picks,
            preload=True,
            reject_by_annotation=False,
            detrend=0
        )
        X_p300_mvep, _ = mne.set_eeg_reference(X_p300_mvep, ref_channels=['M1'])
        X_p300_mvep = X_p300_mvep.get_data()
        X_p300_mvep = X_p300_mvep[locs_p300_mvep, :, :]  # (..., n_chans, n_points)
        return X_p300_mvep, y_p300_mvep, y_tnt

    def get_iir_sos_band(
            self,
            w_pass: List[float],
            w_stop: List[float]):
        """Get second-order sections (like 'ba') of Chebyshev type I filter.
            Author: Jin Han
            Email: jinhan@tju.edu.cn

        Args:
            w_pass (List[float, float]): Passband edge frequencies (Hz).
            w_stop (List[float, float]): Stopband edge frequencies (Hz).

        Returns:
            sos_system (ndarray): Second-order sections representation of the IIR filter.
        """
        # check error input
        if len(w_pass) != 2 or len(w_stop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if w_pass[0] > w_pass[1] or w_stop[0] > w_stop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

        if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')
        # default filter parameters, ANCESTRAL! IMPORTANT!
        # normalized from 0 to 1, 1 is the Nyquist frequency
        wp = [2 * w_pass[0] / self.sfreq, 2 * w_pass[1] / self.sfreq]  # [low edge, high edge]
        ws = [2 * w_stop[0] / self.sfreq, 2 * w_stop[1] / self.sfreq]
        gpass = 4
        gstop = 30
        N, wn = signal.cheb1ord(wp=wp, ws=ws, gpass=gpass, gstop=gstop)  # order, 3dB frequency
        sos_system = signal.cheby1(N=N, rp=0.5, Wn=wn, btype='bandpass', output='sos')
        return sos_system

    def filtered_data_iir(
            self,
            w_pass_2d: ndarray,
            w_stop_2d: ndarray,
            data: ndarray) -> Dict[str, ndarray]:
        """Filter data by IIR method.

        Args:
            w_pass_2d, w_stop_2d (ndarray): (2,n). n is the number of sub-filters.
                e.g.
                w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],
                                      [70, 70, 70, 70, 70, 70, 70]])
                w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],
                                      [72, 72, 72, 72, 72, 72, 72]])
                bandwidth: [5-70], [14-70], ..., etc.
            data (ndarray): (..., n_points).
        Returns: Dict[str, ndarray]
            filtered_data (dict): {'bank1': ndarray, 'bank2': ndarray, ...}
        """
        # check error input
        self.n_filter = w_stop_2d.shape[1]
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')

        if self.n_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        filtered_data = {}
        for nf in range(self.n_filter):
            sos_system = self.get_iir_sos_band(
                w_pass=[w_pass_2d[0, nf], w_pass_2d[1, nf]],
                w_stop=[w_stop_2d[0, nf], w_stop_2d[1, nf]]
            )
            filtered_data['bank' + str(nf + 1)] = signal.sosfiltfilt(
                sos=sos_system,
                x=data,
                axis=-1
            )
        return filtered_data


# %% Preprocessing (save data into .pkl files) | Given Up, comsuming too much RAM
freqs = [10.2, 16.6, 15.4, 13.0, 14.2,
         12.6, 15.0, 13.8, 10.6, 17.0,
         16.2, 13.4, 17.4, 11.0, 12.2,
         14.6, 15.8, 11.8, 17.8, 11.4]

phases = [np.mod((freq - 10.2) / 0.4 * 0.35, 2) for freq in freqs]


test_period = ['1. 前测', '2. 一测', '3. 二测', '4. 后测']
mission = ['offline', 'online']

ntp = 2

dir_path = r'D:\BaiduSyncdisk\Exp Data\Exp\3. SSVEP-P300-mVEP\%s' % (test_period[ntp])
sub_name = [name[-3:] for name in os.listdir(dir_path)]

mi = mission[0]

sub_path = [os.path.join(dir_path, sn, mi) for sn in os.listdir(dir_path)]
total_path = dict(zip(sub_name, sub_path))

sub_list = list(total_path.keys())

# total_path = {}
# for nsub in range(len(total_path)):
#     curret_path[sub_list[nsub]] = total_path[sub_list[nsub]]
# curret_path[sub_list[nsub]] = total_path[sub_list[nsub]]

time_ssvep = 0.7
tmin = -0.4
tmax = 0.14 + time_ssvep + 0.025

sn = sub_name[0]
curret_path = total_path[sn]

pre_eeg = PreProcessing(
    n_events=20,
    n_rounds=5,
    n_tdma=6,
    tmin=tmin,
    tmax=tmax,
    t_ssvep=time_ssvep,
    sfreq=1000
)
pre_eeg.load_data(file_path=curret_path)

X_ssvep_round, y_ssvep_round = pre_eeg.extract_ssvep_round()
X_ssvep_trial, y_ssvep_trial = pre_eeg.extract_ssvep_trial()
X_p300_mvep, y_p300_mvep, y_tnt = pre_eeg.extract_p300_mvep()


# save raw SSVEP data
save_file_path = r'D:\BaiduSyncdisk\Datasets\Preprocessed Data\mVEP-P300-SSVEP\%s\%s\SSVEP\Raw' % (test_period[ntp], mi)
if not os.path.lexists(save_file_path):
    os.makedirs(save_file_path)

file_name_round = sn + '_round.pkl'
save_path_round = os.path.join(save_file_path, file_name_round)
with open(save_path_round, 'wb') as f:
    pickle.dump({'X': X_ssvep_round,
                 'y': y_ssvep_round,
                 'freqs': freqs,
                 'phases': phases}, f)

file_name_trial = sn + '_trial.pkl'
save_path_trial = os.path.join(save_file_path, file_name_trial)
with open(save_path_trial, 'wb') as f:
    pickle.dump({'X': X_ssvep_trial,
                 'y': y_ssvep_trial,
                 'freqs': freqs,
                 'phases': phases}, f)

# save Base SSVEP data
save_file_path = r'D:\BaiduSyncdisk\Datasets\Preprocessed Data\mVEP-P300-SSVEP\%s\%s\SSVEP\Base' % (test_period[ntp], mi)
if not os.path.lexists(save_file_path):
    os.makedirs(save_file_path)

base_X_ssvep_round = pre_eeg.filtered_data_iir(
    w_pass_2d=np.array([[8], [19]]),
    w_stop_2d=np.array([[6], [21]]),
    data=X_ssvep_round
)
base_X_ssvep_round = base_X_ssvep_round['bank1']

file_name_round = sn + '_round.pkl'
save_path_round = os.path.join(save_file_path, file_name_round)
with open(save_path_round, 'wb') as f:
    pickle.dump({'X': base_X_ssvep_round,
                 'y': y_ssvep_round,
                 'freqs': freqs,
                 'phases': phases}, f)

base_X_ssvep_trial = pre_eeg.filtered_data_iir(
    w_pass_2d=np.array([[8], [19]]),
    w_stop_2d=np.array([[6], [21]]),
    data=X_ssvep_trial
)
base_X_ssvep_trial = base_X_ssvep_trial['bank1']

file_name_trial = sn + '_trial.pkl'
save_path_trial = os.path.join(save_file_path, file_name_trial)
with open(save_path_trial, 'wb') as f:
    pickle.dump({'X': base_X_ssvep_trial,
                 'y': y_ssvep_trial,
                 'freqs': freqs,
                 'phases': phases}, f)


# save FB SSVEP data
save_file_path = r'D:\BaiduSyncdisk\Datasets\Preprocessed Data\mVEP-P300-SSVEP\%s\%s\SSVEP\Filter Bank' % (test_period[ntp], mi)
if not os.path.lexists(save_file_path):
    os.makedirs(save_file_path)

w_pass_2d = np.array([[8, 18, 28, 38, 48, 58],
                      [72, 72, 72, 72, 72, 72]])  # 70
w_stop_2d = np.array([[6, 16, 26, 36, 46, 56],
                      [74, 74, 74, 74, 74, 74]])  # 72

fb_X_ssvep_round = pre_eeg.filtered_data_iir(
    w_pass_2d=w_pass_2d,
    w_stop_2d=w_stop_2d,
    data=X_ssvep_round
)

file_name_round = sn + '_round.pkl'
save_path_round = os.path.join(save_file_path, file_name_round)
with open(save_path_round, 'wb') as f:
    pickle.dump({'X': fb_X_ssvep_round,
                 'y': y_ssvep_round,
                 'freqs': freqs,
                 'phases': phases,
                 'wp': w_pass_2d,
                 'ws': w_stop_2d}, f)

fb_X_ssvep_trial = pre_eeg.filtered_data_iir(
    w_pass_2d=w_pass_2d,
    w_stop_2d=w_stop_2d,
    data=X_ssvep_trial
)

file_name_trial = sn + '_trial.pkl'
save_path_trial = os.path.join(save_file_path, file_name_trial)
with open(save_path_trial, 'wb') as f:
    pickle.dump({'X': fb_X_ssvep_trial,
                 'y': y_ssvep_trial,
                 'freqs': freqs,
                 'phases': phases,
                 'wp': w_pass_2d,
                 'ws': w_stop_2d}, f)


# save raw P300-mVEP data
save_file_path = r'D:\BaiduSyncdisk\Datasets\Preprocessed Data\mVEP-P300-SSVEP\%s\%s\P300-mVEP\Raw' % (test_period[ntp], mi)
if not os.path.lexists(save_file_path):
    os.makedirs(save_file_path)

file_name = sn + '.pkl'
save_path = os.path.join(save_file_path, file_name)
with open(save_path, 'wb') as f:
    pickle.dump({'X': X_p300_mvep,
                 'y': y_p300_mvep,
                 'y_tnt': y_tnt}, f)


# save Base P300-mVEP data
save_file_path = r'D:\BaiduSyncdisk\Datasets\Preprocessed Data\mVEP-P300-SSVEP\%s\%s\P300-mVEP\Base' % (test_period[ntp], mi)
if not os.path.lexists(save_file_path):
    os.makedirs(save_file_path)

w_pass_p3 = np.array([[0.5], [10]])  # 70
w_stop_p3 = np.array([[0.1], [12]])  # 72

base_X_p300_mvep = pre_eeg.filtered_data_iir(
    w_pass_2d=w_pass_p3,
    w_stop_2d=w_stop_p3,
    data=X_p300_mvep
)

file_name = sn + '.pkl'
save_path = os.path.join(save_file_path, file_name)
with open(save_path, 'wb') as f:
    pickle.dump({'X': base_X_p300_mvep,
                 'y': y_p300_mvep,
                 'y_tnt': y_tnt}, f)


# %% load in data for training
test_period = ['1. 前测', '2. 一测', '3. 二测', '4. 后测']
mission = ['offline', 'online']

ntp = 0  # 1st test period

dir_path = r'D:\BaiduSyncdisk\Exp Data\Exp\3. SSVEP-P300-mVEP\%s' % (test_period[ntp])
sub_name = [name[-3:] for name in os.listdir(dir_path)]

mi = mission[0]  # 0 for offline, 1 for online

sub_path = [os.path.join(dir_path, sn, mi) for sn in os.listdir(dir_path)]
total_path = dict(zip(sub_name, sub_path))

# default parameters
time_ssvep = 0.7
tmin = -0.4
tmax = 0.14 + time_ssvep + 0.025

sn = sub_name[5]
curret_path = total_path[sn]

pre_eeg = PreProcessing(
    n_events=20,
    n_rounds=5,
    n_tdma=6,
    tmin=tmin,
    tmax=tmax,
    t_ssvep=time_ssvep,
    sfreq=1000
)
pre_eeg.load_data(file_path=curret_path)

# extract data into ndarray
# X_ssvep_round, y_ssvep_round = pre_eeg.extract_ssvep_round()
X_train, y_train = pre_eeg.extract_ssvep_trial()

# preprocessing
X_train = pre_eeg.filtered_data_iir(
    w_pass_2d=np.array([[8], [90]]),
    w_stop_2d=np.array([[6], [92]]),
    data=X_train
)
X_train = X_train['bank1']

# %% load in data for testing
test_period = ['1. 前测', '2. 一测', '3. 二测', '4. 后测']
mission = ['offline', 'online']

ntp = 0  # 1st test period

dir_path = r'D:\BaiduSyncdisk\Exp Data\Exp\3. SSVEP-P300-mVEP\%s' % (test_period[ntp])
sub_name = [name[-3:] for name in os.listdir(dir_path)]

mi = mission[1]  # 0 for offline, 1 for online

sub_path = [os.path.join(dir_path, sn, mi) for sn in os.listdir(dir_path)]
total_path = dict(zip(sub_name, sub_path))

# default parameters
time_ssvep = 0.7
tmin = -0.4
tmax = 0.14 + time_ssvep + 0.025

curret_path = total_path[sn]

pre_eeg = PreProcessing(
    n_events=20,
    n_rounds=5,
    n_tdma=6,
    tmin=tmin,
    tmax=tmax,
    t_ssvep=time_ssvep,
    sfreq=1000
)
pre_eeg.load_data(file_path=curret_path)

# extract data into ndarray
# X_ssvep_round, y_ssvep_round = pre_eeg.extract_ssvep_round()
X_test, y_test = pre_eeg.extract_ssvep_trial()

# preprocessing
X_test = pre_eeg.filtered_data_iir(
    w_pass_2d=np.array([[8], [19]]),
    w_stop_2d=np.array([[6], [21]]),
    data=X_test
)
X_test = X_test['bank1']

# %% classification
c30 = ['CPZ', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
       'PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
       'POZ', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
       'OZ', 'O1', 'O2', 'CB1', 'CB2']
chan_indices = [pre_eeg.CHANNELS.index(ch) for ch in c30]

sine_template = np.zeros((20, 2, 3500))
for ne in range(20):
    sine_template[ne] = utils.sine_template(
        freq=pre_eeg.FREQUENCY[ne],
        phase=pre_eeg.PHASE[ne],
        n_points=sine_template.shape[-1],
        n_harmonics=1,
        srate=1000
    )

classifier = trca.TRCA()
classifier.fit(
    X_train=X_train[:, chan_indices, 1140:1140 + 3500],
    y_train=y_train
)
y_trca, y_etrca = classifier.predict(X_test=X_test[:, chan_indices, 1140:1140 + 3500])
acc_trca = utils.acc_compute(
    y_true=y_test,
    y_pred=y_trca
)
acc_etrca = utils.acc_compute(
    y_true=y_test,
    y_pred=y_etrca
)

# %% SNR in frequency-domain
c30 = ['CPZ', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8',
       'PZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
       'POZ', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8',
       'OZ', 'O1', 'O2', 'CB1', 'CB2']
chan_indices = [pre_eeg.CHANNELS.index(ch) for ch in c30]


def freqDomian(signal, fs, nfft):
    mf = fft(signal, nfft)
    freqAns = abs(mf) * 2 / nfft
    f = np.arange(0, len(mf), 1) * fs / len(mf)
    return f, freqAns


classifier = trca.TRCA()
classifier.fit(
    X_train=X_train[:, chan_indices, 1140:1140 + 3500],
    y_train=y_train
)
wX = classifier.training_model['wX']

nfft = 3500
f, freqAns = freqDomian(
    signal=X_train[y_train == 2][:, 61, 1140:1140 + 3500].mean(axis=0),
    fs=1000,
    nfft=nfft
)
# X_train[y_train==19][:, 61, 1140:1140 + 3500].mean(axis=0)
# wX[1,0,:]


def freqSNR(f, freqAns):
    snr = np.zeros_like(freqAns)
    for i in range(315):
        if i < 3:
            snr[i] = 20 * log(freqAns[i] / np.sum(freqAns[:5]), 10)
        else:
            snr[i] = 20 * log(freqAns[i] / np.sum(freqAns[i - 2:i + 3]), 10)
    return snr


snr = freqSNR(f, freqAns)

plt.plot(f[10:200], freqAns[10:200])

# %% reload results
FREQUENCY = [10.2, 16.6, 15.4, 13.0, 14.2,
             12.6, 15.0, 13.8, 10.6, 17.0,
             16.2, 13.4, 17.4, 11.0, 12.2,
             14.6, 15.8, 11.8, 17.8, 11.4]

# load in results
data_path = r'D:\BaiduSyncdisk\Results\20240529\pkls\tsnr_round_raw.pkl'
with open(data_path, 'rb') as f:
    results = pickle.load(f)

total_snr = results['tsnr']
del f, results, data_path

stim_freqs = [np.round(10.2 + i * 0.4, 1) for i in range(20)]
reshape_order = [stim_freqs.index(i) for i in FREQUENCY]


def variability_compute(x):
    return np.std(x) / np.mean(x)


# %% plot figures: lineplot
xticks = np.arange(20)
xticks_freq = [str(sf) for sf in stim_freqs]

fig = plt.figure(figsize=(12, 25))
gs = GridSpec(4, 1)
sns.set_style('whitegrid')

titles = ['Pretest', '7 - 8th Day', '14 - 15th Day', 'Postest']

for nrow in range(4):
    for ncol in range(1):
        fig_idx = ncol + 1 * nrow
        ax = fig.add_subplot(gs[nrow:nrow + 1, ncol:ncol + 1])
        ax.set_title(titles[fig_idx], fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        for rnd in range(5):
            ax.plot(total_snr[fig_idx][:, rnd, reshape_order].mean(axis=0),
                    label='Round {}'.format(rnd + 1))
        ax.set_xlabel('Stimulus frequency/Hz', fontsize=18)
        ax.set_xticks(xticks, xticks_freq, rotation=45)
        ax.set_ylabel('SNR in time domain', fontsize=18)
        ax.set_ylim(2200, 5100)
        if ncol == 0 and nrow == 3:
            ax.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0, fontsize=16)
        else:
            ax.legend([], [], frameon=False)

plt.tight_layout()
# plt.show()

save_path = r'D:\BaiduSyncdisk\Results\20240529\Figs\3.png'
plt.savefig(save_path, dpi=600)
plt.close()

# %% plot figures: barplot
total_snr_mean = np.stack([ts.mean(axis=0) for ts in total_snr])[:, :, reshape_order]

time_period = ['Pretest', '7 - 8th Day', '14 - 15th Day', 'Postest']

time_series, round_series, snr_series = [], [], []
for ntp, tp in enumerate(time_period):
    for rnd in range(5):
        for ne in range(20):
            snr_series.append(total_snr_mean[ntp, rnd, ne])
            round_series.append('Round' + str(rnd + 1))
            time_series.append(tp)
df = pd.DataFrame({'Period': time_series,
                   'Round': round_series,
                   'SNR': snr_series})
del snr_series, round_series, time_series, tp, rnd, ne

fig = plt.figure(figsize=(12, 6))
gs = GridSpec(1, 1)
sns.set_style('whitegrid')

ax = fig.add_subplot(gs[:, :])
ax.set_title('SNR values across different periods', fontsize=20)
ax.tick_params(axis='both', labelsize=16)
ax = sns.barplot(data=df, x='Period', y='SNR', hue='Round', palette='pastel')
ax.set_xlabel('Experimental period', fontsize=18)
ax.set_ylabel('SNR in time domain', fontsize=18)
ax.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0, fontsize=16)

plt.tight_layout()
plt.show()

save_path = r'D:\BaiduSyncdisk\Results\20240529\Figs\4.png'
plt.savefig(save_path, dpi=600)
plt.close()

# %% variability across periods
# var on rounds
period_series, freq_series, var_series = [], [], []
for ntp, tp in enumerate(time_period):
    temp = total_snr[ntp][:, :, reshape_order]
    n_subjects = temp.shape[0]
    for nsub in range(n_subjects):
        for ne in range(20):
            period_series.append(tp)
            freq_series.append(stim_freqs[ne])
            var_series.append(variability_compute(temp[nsub, :, ne]))
df_var_rounds = pd.DataFrame({'Period': period_series,
                              'Freqs': freq_series,
                              'Variability': var_series})
del period_series, freq_series, var_series

period_series, round_series, var_series = [], [], []
for ntp, tp in enumerate(time_period):
    temp = total_snr[ntp][:, :, reshape_order]
    n_subjects = temp.shape[0]
    for nsub in range(n_subjects):
        for rnd in range(5):
            period_series.append(tp)
            round_series.append('Round' + str(rnd + 1))
            var_series.append(variability_compute(temp[nsub, rnd, :]))
df_var_freqs = pd.DataFrame({'Period': period_series,
                             'Round': round_series,
                             'Variability': var_series})
del period_series, round_series, var_series

fig = plt.figure(figsize=(14, 6))
gs = GridSpec(1, 2)
sns.set_style('whitegrid')

ax = fig.add_subplot(gs[:, :1])
ax.set_title('SNR variability across rounds', fontsize=20)
ax.tick_params(axis='both', labelsize=16)
ax = sns.barplot(data=df_var_rounds, x='Period', y='Variability', palette='pastel', saturation=0.75)
ax.set_xlabel('Experimental period', fontsize=18)
ax.set_ylabel('Variability values', fontsize=18)
# ax.legend(bbox_to_anchor=(1.02, 0), loc=3, borderaxespad=0, fontsize=16)

ax = fig.add_subplot(gs[:, 1:])
ax.set_title('SNR variability across frequencies', fontsize=20)
ax.tick_params(axis='both', labelsize=16)
ax = sns.barplot(data=df_var_freqs, x='Period', y='Variability', palette='pastel', saturation=0.75)
ax.set_xlabel('Experimental period', fontsize=18)
ax.set_ylabel('Variability values', fontsize=18)

plt.tight_layout()
plt.show()

save_path = r'D:\BaiduSyncdisk\Results\20240529\Figs\5.png'
plt.savefig(save_path, dpi=600)
plt.close()

# %% test
# calculation
# load in data
test_period = ['1. 前测', '2. 一测', '3. 二测', '4. 后测']
mission = ['offline', 'online']

cp3 = ['F3', 'Fz', 'F4', 'T7', 'C3', 'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4',
       'P8', 'PO7', 'PO8', 'OZ']
n_chans = len(cp3)


dir_path = r'D:\BaiduSyncdisk\Exp Data\Exp\3. SSVEP-P300-mVEP\%s' % (test_period[-1])
sub_name = [name[-3:] for name in os.listdir(dir_path)]

mi = mission[0]

sub_path = [os.path.join(dir_path, sn, mi) for sn in os.listdir(dir_path)]
total_path = dict(zip(sub_name, sub_path))

sub_list = list(total_path.keys())
wave_tar = np.zeros((len(sub_name), 2, n_chans, 750))  # subjects, 2, n_chans, n_points

# default parameters
time_ssvep = 0.7
tmin = -0.4
tmax = 0.14 + time_ssvep + 0.025

pre_eeg = PreProcessing(
    n_events=20,
    n_rounds=5,
    n_tdma=6,
    tmin=tmin,
    tmax=tmax,
    t_ssvep=time_ssvep,
    sfreq=1000
)

for nsub, sub in enumerate(sub_name):
    curret_path = total_path[sub]
    if curret_path not in [r'D:\BaiduSyncdisk\Exp Data\Exp\3. SSVEP-P300-mVEP\3. 二测\20240414-S01\offline',
                           r'D:\BaiduSyncdisk\Exp Data\Exp\3. SSVEP-P300-mVEP\3. 二测\20240414-S20\offline']:
        pre_eeg.load_data(file_path=curret_path)
        X, y, y_tnt = pre_eeg.extract_p300_mvep()

        # preprocessing
        X = pre_eeg.filtered_data_iir(
            w_pass_2d=np.array([[0.5], [10]]),
            w_stop_2d=np.array([[0.1], [12]]),
            data=X
        )['bank1']

        # extract useful channels &n time range
        chan_indices = [pre_eeg.CHANNELS.index(ch.upper()) for ch in cp3]
        X = X[:, :, chan_indices, 450:1200]  # (n_trials, n_chans, n_points)
        X_tar, X_nontar = X[y_tnt, :, :], X[~y_tnt, :, :]

        wave_tar[nsub, 0, :, :] = X_tar.mean(axis=0)
        wave_tar[nsub, 1, :, :] = X_nontar.mean(axis=0)
        # wave_tar = np.delete(wave_tar, [9, 14], axis=0)

# save results
save_path = r'D:\BaiduSyncdisk\Results\20240529\pkls\wave_p300_dsp_{}.pkl'.format(test_period[-1])
with open(save_path, 'wb') as f:
    pickle.dump({'wave': wave_tar}, f)
