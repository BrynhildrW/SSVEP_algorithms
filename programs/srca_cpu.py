# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Spatial Regression Component Analysis (SRCA) series.

Supported objects
1. SRCA: single channel & single-event
    Target functions (1-D): SNR, pCORR (1-D)
    Optimization methods: Traversal, Recursion, Mix

2. ESRCA: single channel & multi-event (Ensemble-SRCA)
    Target functions (1-D): SNR, FS, pCORR
    Optimization methods: Traversal, Recursion, Mix

3. MultiSRCA: multi-channel & single-event (Multi-channel SRCA)
    Target functions (2-D):
    Optimization methods: Traversal, Recursion, Mix
    Combination optimization methods: IBI(Item-by-item)

4. MtuliESRCA: multi-channel & multi-event (Multi-channel ensemble-SRCA)
    Target functions (2-D): DSP-val, TDCA-val
    Optimization methods: Traversal, Recursion, Mix
    Combination optimization methods: IBI

update: 2023/12/7

"""

# basic modules
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any

import utils

from dsp import DSP, TDCA
import trca
import cca

import numpy as np
from numpy import ndarray

import scipy.linalg as sLA

from sklearn import linear_model
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import combinations, chain

from time import perf_counter
from copy import deepcopy


# %% 1-D target functions | single channel
def snr_sequence(
    X_train: ndarray,
    y_train: ndarray,
    kwargs: Optional[dict] = None) -> ndarray:
    """Signal-to-Noise ratio (sequence) in time domain.

    Args:
        X_train (ndarray): (Ne*Nt,Np). Input data.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        kwargs (dict, optional): {'event_type': ndarray, (Ne,)}

    Returns:
        snr (ndarray): (Ne,Np). SNR sequence in time domain.
            n_events could be 1.
    """
    # basic information
    try:
        event_type = kwargs['event_type']
    except KeyError:
        event_type = np.unique(y_train)
    n_events = len(event_type)
    n_points = X_train.shape[-1]

    # compute SNR in time domain
    signal_power = np.zeros((n_events, n_points))
    noise_power = np.zeros_like(signal_power)
    for ne,et in enumerate(event_type):
        pure_signal = X_train[y_train==et].mean(axis=0, keepdims=True)  # (1,Np)
        signal_power[ne,:] = pure_signal**2  # (1,Np)
        noise_signal = X_train[y_train==et] - pure_signal  # (Nt,Np)
        noise_power[ne,:] = (noise_signal**2).mean(axis=0, keepdims=True)  # (1,Np)
    return signal_power/noise_power


def fs_sequence(
    X_train: ndarray,
    y_train: ndarray,
    kwargs: Optional[dict] = None) -> ndarray:
    """Fisher Score (sequence) in time domain.
 
    Args:
        X_train (ndarray): (Ne*Nt,Np). Input data.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        kwargs (dict, optional): {'event_type': ndarray, (Ne,)}

    Returns:
        fs (ndarray): (1,Np). Fisher-Score sequence.
    """
    # basic information
    try:
        event_type = kwargs['event_type']
    except KeyError:
        event_type = np.unique(y_train)

    # compute FS in time domain
    dataset = [X_train[y_train==et] for et in event_type]  # (event1, event2, ...)
    return utils.fisher_score(dataset)


# %% 2-D target functions | multiple channels, single event
# Target function values of ms-TRCA | single event
def trca_val(X_train, y_train, kwargs):
    """Target function values of eTRCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        kwargs (dict, optional): 
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns:
        coef (float)
    """
    pass


def trca_acc(dataset, kwargs):
    """Accuracy calculated by TRCA. (Deprecated)

    Args:
        dataset (ndarray): (n_events, n_trials, n_chans, n_points).
        (Below are contained in kwargs)
        n_train (int): Number of training samples. Must be less than n_trials.
        n_repeat (int, optional): Number of Monte-Carlo cross-validation.
            Defaults to 10.
        n_components (int, optional): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'. Defaults to 1.
        ratio (float, optional): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None' when n_component is not 'None'.

    Returns:
        acc (float)
    """
    pass


# %% 2-D target functions | multiple channels, multiple events
def dsp_val(
    X_train: ndarray,
    y_train: ndarray,
    kwargs: dict) -> float:
    """Target function values of DSP.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        kwargs (dict): {'n_components':int}

    Returns:
        coef (float): (w @ Sb @ w.T) / (w @ Sw @ w.T)
    """
    dsp_model = DSP(n_components=kwargs['n_components'])
    dsp_model.fit(
        X_train=X_train,
        y_train=y_train
    )
    w = dsp_model.training_model['w']
    Sb, Sw = dsp_model.training_model['Sb'], dsp_model.training_model['Sw']
    return (w @ Sb @ w.T)/(w @ Sw @ w.T)


def dsp_acc(X_train, y_train, kwargs):
    """Accuracy calculated by DSP-M1.

    Args:
        X_train (ndarray): (train_trials, n_chans, n_points). Training dataset.
        y_train (ndarray): (train_trials,). Labels for X_train.
        (Below are contained in kwargs)
        n_train (int): Number of training samples. Must be less than n_trials.
        n_repeat (int, optional): Number of Monte-Carlo cross-validation.
            Defaults to 10.
        n_components (int, optional): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'. Defaults to 1.
        ratio (float, optional): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None' when n_component is not 'None'.

    Returns:
        acc (float)
    """
    # basic information
    # n_repeat = kwargs['n_repeat']
    # n_train = kwargs['n_train']
    # n_components = kwargs['n_components']
    # ratio = kwargs['ratio']

    # # cross-validation
    # sss = StratifiedShuffleSplit(
    #     n_splits=kwargs['n_repeat'],
    #     test_size=1-kwargs['n_train']/len(y_train),
    #     random_state=0
    # )
    # acc = np.zeros((kwargs['n_repeat']))
    # for nrep, (train_index, test_index) in enumerate(sss.split(X_train, y_train)):
    #     X_part_train, X_part_test = X_train[train_index], X_train[test_index]
    #     y_part_train, y_part_test = y_train[train_index], y_train[test_index]

    #     dsp_model = DSP(
    #         n_components=kwargs['n_components'],
    #         ratio=kwargs['ratio']
    #     )
    #     dsp_model.fit(
    #         X_train=X_part_train,
    #         y_train=y_part_train
    #     )
    #     _, y_dsp = dsp_model.predict(
    #         X_test=X_part_test,
    #         y_test=y_part_test
    #     )
    #     acc[nrep] = utils.acc_compute(y_dsp, y_part_test)
    pass


def tdca_val(
    X_train: ndarray,
    y_train: ndarray,
    kwargs: dict) -> float:
    """Target function values of DSP.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        kwargs (dict): {'projection': ndarray, (Ne,Np,Np),
                        'extra_length': int,
                        'extra_data': ndarray, (Ne*Nt,Nc,m)
                        'n_components':int}

    Returns:
        coef (float): (w @ Sb @ w.T) / (w @ Sw @ w.T)
    """
    tdca_model = TDCA(n_components=kwargs['n_components'])
    tdca_model.fit(
        X_train=X_train,
        X_extra=kwargs['extra_data'],
        y_train=y_train,
        projection=kwargs['projection']
    )
    w = tdca_model.training_model['w']  # (Nk,(1+m)*Nc)
    Sb, Sw = tdca_model.training_model['Sb'], tdca_model.training_model['Sw']
    return np.mean(w @ Sb @ w.T)/np.mean(w @ Sw @ w.T)


# %% SRCA operation
def mse_regression(
    rs_model: ndarray,
    rs_target: ndarray,
    ts_model: ndarray) -> ndarray:
    """Linear regression for task-state target channel based on
        Mean squared error (Frobenius Norm).

    Args:
        rs_model (ndarray): (Nt,Nc,Np). Rest-state data of model channels.
        rs_target (ndarray): (Nt,Np). Rest-state data of target channel.
        ts_model (ndarray): (Nt,Nc,Np). Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (Nt,Np). Estimation of task-state data of target channel.
    """
    n_trials = ts_model.shape[0]  # Nt
    n_points = ts_model.shape[-1]  # Np
    ts_target_estimate = np.zeros((n_trials, n_points))  # (Nt,Np)
    for ntr in range(n_trials):
        projection = rs_target[ntr] @ rs_model[ntr].T @ sLA.inv(rs_model[ntr] @ rs_model[ntr].T)
        ts_target_estimate[ntr] = projection @ ts_model[ntr]
    return ts_target_estimate


def linear_regression(
    rs_model: ndarray,
    rs_target: ndarray,
    ts_model: ndarray) -> ndarray:
    """Argmin function based on Ordinary Least Squares (sklearn).

    Args:
        rs_model (ndarray): (Nt,Nc,Np). Rest-state data of model channels.
        rs_target (ndarray): (Nt,Np). Rest-state data of target channel.
        ts_model (ndarray): (Nt,Nc,Np). Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (Nt,Np). Estimation of task-state data of target channel.
    """
    n_trials = ts_model.shape[0]  # Nt
    n_points = ts_model.shape[-1]  # Np
    ts_target_estimate = np.zeros((n_trials, n_points))  # (Nt,Np)
    for ntr in range(n_trials):
        L = linear_model.LinearRegression().fit(
            X=rs_model[ntr].T,
            y=rs_target[ntr]
        )
        ts_target_estimate[ntr] = L.coef_@ts_model + L.intercept_
    return ts_target_estimate


def ridge(
    rs_model: ndarray,
    rs_target: ndarray,
    ts_model: ndarray) -> ndarray:
    """Argmin function based on Ridge regression (sklearn).

    Args:
        rs_model (ndarray): (Nt,Nc,Np). Rest-state data of model channels.
        rs_target (ndarray): (Nt,Np). Rest-state data of target channel.
        ts_model (ndarray): (Nt,Nc,Np). Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (Nt,Np). Estimation of task-state data of target channel.
    """
    n_trials = ts_model.shape[0]  # Nt
    n_points = ts_model.shape[-1]  # Np
    ts_target_estimate = np.zeros((n_trials, n_points))  # (Nt,Np)
    for ntr in range(n_trials):
        L = linear_model.Ridge().fit(
            X=rs_model[ntr].T,
            y=rs_target[ntr]
        )
        ts_target_estimate[ntr,:] = L.coef_@ts_model + L.intercept_
    return ts_target_estimate


def lasso(
    rs_model: ndarray,
    rs_target: ndarray,
    ts_model: ndarray) -> ndarray:
    """Argmin function based on Lasso regression (sklearn).

    Args:
        rs_model (ndarray): (Nt,Nc,Np). Rest-state data of model channels.
        rs_target (ndarray): (Nt,Np). Rest-state data of target channel.
        ts_model (ndarray): (Nt,Nc,Np). Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (Nt,Np). Estimation of task-state data of target channel.
    """
    n_trials = ts_model.shape[0]  # Nt
    n_points = ts_model.shape[-1]  # Np
    ts_target_estimate = np.zeros((n_trials, n_points))  # (Nt,Np)
    for ntr in range(n_trials):
        L = linear_model.Lasso().fit(rs_model[ntr].T, rs_target[ntr])
        ts_target_estimate[ntr,:] = L.coef_@ts_model + L.intercept_
    return ts_target_estimate


def elastic_net(
    rs_model: ndarray,
    rs_target: ndarray,
    ts_model: ndarray) -> ndarray:
    """Argmin function based on Elastic-Net regression (sklearn).

    Args:
        rs_model (ndarray): (Nt,Nc,Np). Rest-state data of model channels.
        rs_target (ndarray): (Nt,Np). Rest-state data of target channel.
        ts_model (ndarray): (Nt,Nc,Np). Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (Nt,Np). Estimation of task-state data of target channel.
    """
    n_trials = ts_model.shape[0]  # Nt
    n_points = ts_model.shape[-1]  # Np
    ts_target_estimate = np.zeros((n_trials, n_points))  # (Nt,Np)
    for ntr in range(n_trials):
        L = linear_model.ElasticNet().fit(rs_model[ntr].T, rs_target[ntr])
        ts_target_estimate[ntr,:] = L.coef_@ts_model + L.intercept_
    return ts_target_estimate


regressions = {
    'MSE':mse_regression,
    'OLS':linear_regression,
    'RI':ridge,
    'LA':lasso,
    'EN':elastic_net
}


def srca_process(
    rs_model: ndarray,
    rs_target: ndarray,
    ts_model: ndarray,
    ts_target: ndarray,
    regression: str = 'MSE') -> ndarray:
    """Main process of SRCA algorithm.

    Args:
        rs_model (ndarray): (Nt,Nc,Np). Rest-state data of model channels.
        rs_target (ndarray): (Nt,Np). Rest-state data of target channel.
        ts_model (ndarray): (Nt,Nc,Np). Task-state data of model channels.
        ts_target (ndarray): (Nt,Np). Task-state data of target channel.
        regression (str, optional): 'MSE', 'OLS', 'RI', 'LA' and 'EN'.

    Returns:
        ts_target_extraction (ndarray): (Nt,Np). SRCA processed task-state data of target channel.
    """
    ts_target_estimation = regressions[regression](
        rs_model=rs_model,
        rs_target=rs_target,
        ts_model=ts_model
    )
    return ts_target - ts_target_estimation


# %% Main classes
class BasicPearsonCorr(object):
    """Verify the basic function of dynamic spatial filtering."""
    def __init__(self):
        pass


    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        """Construct averaging templates.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        self.X_train = X_train
        self.y_train = y_train
        
        self.event_type = np.unique(self.y_train)
        self.n_events = len(self.event_type)
        self.n_chans = self.X_train.shape[-2]
        self.n_points = self.X_train.shape[-1]
        
        self.avg_template = np.zeros((self.n_events, self.n_chans, self.n_points))
        for ne,et in enumerate(self.event_type):
            self.avg_template[ne] = self.X_train[self.y_train==et].mean(axis=0)
        return self


    def transform(self,
        X_test: ndarray) -> Tuple:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return: Tuple
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of DSP.
        """
        n_test = X_test.shape[0]
        rho = np.zeros((n_test, self.n_events))
        for nte in range(n_test):
            temp = X_test[nte]
            for nem in range(self.n_events):
                rho[nte,nem] = utils.pearson_corr(
                    X=temp,
                    Y=self.avg_template[nem]
                )
        return rho


    def predict(self,
        X_test: ndarray) -> ndarray:
        """Using Pearson's correlation coefficients to predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        self.rho = self.transform(X_test)
        self.predict = self.event_type[np.argmax(self.rho, axis=-1)]
        return self.predict


class SRCA(object):
    """Spatial Regression Component Analysis for single-channel, single-event optimization.
    Target functions (1-D):
        (1) SNR (mean) in time domain
    """
    tar_functions = {'SNR':snr_sequence}
    opt_methods = ['Traversal', 'Recursion', 'Mix']


    def __init__(self,
        X_train: ndarray,
        y_train: ndarray,
        rest_phase: List,
        task_phase: List,
        chan_info: List,
        tar_chan: str,
        tar_func: str = 'SNR',
        opt_method: str = 'Recursion',
        regression: str = 'MSE',
        traversal_limit: Optional[int] = None,
        chan_num_limit: Optional[int] = None,
        kwargs: Optional[dict] = None):
        """Load in settings.

        Args:
            X_train (ndarray): (Nt,Nc,Np). Training dataset of 1 category.
            y_train (ndarray): (Nt,). Labels for X_train.
            rest_phase (list): [st,ed]. The start and end point of rest-state data.
            task_phase (list): [st,ed]. The start and end point of task-state data.
            chan_info (list): Names of all channels.
            tar_chan (str): Name of target channel.
            tar_func (str): 'SNR'.
            opt_method (str): 'Traversal', 'Recursion' or 'Mix'.
            regression (str): Regression method used in SRCA process.
                'MSE','OLS','RI','LA' or 'EN'. Defaults to 'MSE'.
            traversal_limit (int, optional): The maximum number of channels to be traversed.
                Defaults to None.
            chan_num_limit (int, optional): The maximum number of channels used in SRCA model.
                Defaults to None.
            kwargs (dict, optional): {'event_type':ndarray (Ne,)}
        """
        # basic information
        self.rest_data = X_train[...,rest_phase[0]:rest_phase[1]]
        self.task_data = X_train[...,task_phase[0]:task_phase[1]]
        self.n_chans = X_train.shape[-2]
        self.y_train = y_train
        self.chan_info = chan_info
        self.tar_chan = tar_chan
        self.tar_func = tar_func
        self.opt_method = opt_method
        self.traversal_limit = traversal_limit
        self.chan_num_limit = chan_num_limit
        self.regression = regression
        if not kwargs:
            self.kwargs = {'event_type':np.array([1])}
        else:
            self.kwargs = kwargs


    def prepare(self):
        """Initialization for training."""
        # pick up target data for both state.
        self.tar_index = self.chan_info.index(self.tar_chan)
        self.rest_target = self.rest_data[:,self.tar_index,:]
        self.task_target = self.task_data[:,self.tar_index,:]
        self.alter_indices = np.delete(np.arange(self.n_chans), self.tar_index)

        # model initialization
        self.init_value = np.mean(self.tar_functions[self.tar_func](
            X_train=self.task_target,
            y_train=self.y_train,
            kwargs=self.kwargs
        ))
        self.model_indices, self.value_change = [], [self.init_value]
        return self


    def check_traversal(self):
        """Check input while optimization method is 'Traversal'."""
        assert 0<self.traversal_limit<len(self.chan_info), 'Setting inappropriate channels for traversal!'
        self.traversal_combi = [c for c in combinations(self.alter_indices, self.traversal_limit)]


    def check_recursion(self):
        """Check input while optimization method is 'Recursion'."""
        assert 2<=self.chan_num_limit<self.n_chans, 'Setting inappropriate channels for recursion!'


    def srca_unit(self,
        chans_indices: List) -> float:
        """Compute updated target function values of SRCA-processed data.

        Args:
            chans_indices (list or tuple): Indices of channels to be used in SRCA model.

        Returns:
            srca_tar_value (float): Target function values of the SRCA-processed data.
        """
        srca_target = srca_process(
            rs_model=self.rest_data[:,chans_indices,:],
            rs_target=self.rest_target,
            ts_model=self.task_data[:,chans_indices,:],
            ts_target=self.task_target,
            regression=self.regression
        )
        srca_tar_value = np.mean(self.tar_functions[self.tar_func](
            X_train=srca_target,
            y_train=self.y_train,
            kwargs=self.kwargs
        ))
        return srca_tar_value


    def step_forward(self):
        """Add one channel respectively and pick the best one."""
        self.recursion_combi = [self.model_indices+[x] for x in self.alter_indices if x not in self.model_indices]
        results = list(map(self.srca_unit, self.recursion_combi))

        # update temp model information
        best_value = max(results)
        self.value_change.append(best_value)
        self.model_indices = self.recursion_combi[results.index(best_value)]


    def check_progress_1(self):
        """Check if there's any improvement after step_forward()."""
        if self.value_change[-1] < self.value_change[-2]:  # worse after adding a new channel
            del self.model_indices[-1]  # abandon the results after step_forward()
            self.continue_training = False


    def back_and_forth(self):
        """Delete one & add one respectively, then pick the best combination."""
        # combinations(self.model_indices[:-1], len(self.model_indices)-2):
        # self.model_indices[:-1] | every channel will be evaluated except for the last one added
        # len(self.model_indices)-2 | delete one respectively, pick up the rest
        remain_combi = [list(c)+[self.model_indices[-1]] for c in combinations(self.model_indices[:-1], len(self.model_indices)-2)]
        # if x not in self.model_indices: add up a new channel that has not been used before
        add_combi = [[x] for x in self.alter_indices if x not in self.model_indices]
        self.recursion_combi = [rc+ac for rc in remain_combi for ac in add_combi]
        results = list(map(self.srca_unit, self.recursion_combi))

        self.only_add = False  # in this part, the number of channles doesn't increase for sure
        best_value = max(results)
        self.value_change.append(best_value)
        self.uncheck_model_indices = self.recursion_combi[results.index(best_value)]


    def check_progress_2(self):
        """Check if there's any improvement after back_and_forth()."""
        if self.value_change[-1] < self.value_change[-2]:  # worse after stepwise operation
            del self.uncheck_model_indices  # abandon the results after back_and_forth()
            self.continue_training = False
            return self
        # better or remain after stepwise operation
        self.model_indices = self.uncheck_model_indices  # update channel group


    def check_limit(self):
        """Check whether the number of channels has met the limitation."""
        if hasattr(self, 'chan_num_limit') and (len(self.model_indices)==self.chan_num_limit):
            self.continue_training = False


    def traversal(self):
        """Directly traverse each channel group to train SRCA model."""
        self.check_traversal()
        self.results = list(map(self.srca_unit, self.traversal_combi))
        model_index = self.results.index(max(self.results))
        self.value_change.append(self.results[model_index])
        self.model_indices = list(self.traversal_combi[model_index])


    def recursion(self):
        """Use stepwise recursion to train SRCA model."""
        self.only_add = True  # only for first 2 channels
        self.continue_training = True
        self.step_forward()
        self.step_forward()
        while self.continue_training:
            self.step_forward()  # now has 3 channels in model
            self.check_progress_1()
            self.back_and_forth()
            self.check_progress_2()
            self.check_limit()


    def mix_operation(self):
        """'Traversal' for first several channels and then 'Recursion'."""
        self.only_add = (self.traversal_limit==2)  # >2: False; ==2: True
        self.continue_training = True
        self.traversal()
        while self.continue_training:
            self.step_forward()
            self.check_progress_1()
            self.back_and_forth()
            self.check_progress_2()
            self.check_limit()


    operations = dict(zip(opt_methods, [traversal, recursion, mix_operation]))


    def train(self):
        """Total training process."""
        self.operations[self.opt_method](self)
        self.srca_model = [self.chan_info[x] for x in self.model_indices]
        return self


class ESRCA(SRCA):
    """ensemble-SRCA for single-channel, multi-event optimization.
    Target functions (1-D):
        (1) SNR (mean) in time domain
        (2) Fisher score (mean) | only for 2categories
    """
    tar_functions = {'SNR':snr_sequence,
                     'FS':fs_sequence}


class TdSRCA(SRCA):
    """Intermediate process of MultiSRCA
        (i) multi-channel (2-D) target function
        (ii) optimization on single channel
        (iii) optimization on single event
    Target functions (2-D):
        (1) TRCA target function value
    """
    tar_functions = {'TRCA-val':trca_val}
    opt_methods = ['Traversal', 'Recursion', 'Mix']


    def __init__(self,
        X_train: ndarray,
        y_train: ndarray,
        rest_phase: List,
        task_phase: List,
        chan_info: List,
        tar_chan: str,
        tar_chan_list: List,
        tar_func: str = 'TRCA-val',
        opt_method: str = 'Recursion',
        regression: str = 'MSE',
        allow_target_group: bool = False,
        traversal_limit: Optional[int] = None,
        chan_num_limit: Optional[int] = None,
        kwargs: Optional[dict] = None):
        """Load in settings.

        Args:
            X_train (ndarray): (Nt,Nc,Np). Training dataset of 1 category.
            y_train (ndarray): (Nt,). Labels for X_train.
            rest_phase (list): [st,ed]. The start and end point of rest-state data.
            task_phase (list): [st,ed]. The start and end point of task-state data.
            chan_info (list): Names of all channels.
            tar_chan (str): Names of present target channel.
            tar_chan_list (List[str]): Names of all target channels.
            tar_func (str): 'TRCA-val'.
            opt_method (str): 'Traversal', 'Recursion' or 'Mix'.
            regression (str): Regression method used in SRCA process.
                'MSE','OLS','RI','LA' or 'EN'. Defaults to 'MSE'.
            allow_target_group (bool): Allow channels from target group to be estimation
                channels or not.
            traversal_limit (int, optional): The maximum number of channels to be traversed.
                Defaults to None.
            chan_num_limit (int, optional): The maximum number of channels used in tdSRCA model.
                Defaults to None.
            kwargs (dict, optional): {'event_type':ndarray (Ne,),
                                      'n_components':int}
        """
        super().__init__(
            X_train=X_train,
            y_train=y_train,
            rest_phase=rest_phase,
            task_phase=task_phase,
            chan_info=chan_info,
            tar_chan=tar_chan,
            tar_func=tar_func,
            opt_method=opt_method,
            regression=regression,
            traversal_limit=traversal_limit,
            chan_num_limit=chan_num_limit
        )

        # check extra input
        assert set(tar_chan_list) <= set(chan_info), 'Unknown target channel!'
        self.tar_chan_list = tar_chan_list
        self.allow_target_group = allow_target_group
        self.kwargs = kwargs


    def prepare(self):
        """Initialization for training."""
        # pick up target data for both state
        self.tar_index = self.chan_info.index(self.tar_chan)
        self.rest_target = self.rest_data[:,self.tar_index,:]  # (Nt,Np)
        self.task_target = self.task_data[:,self.tar_index,:]  # (Nt,Np)

        # config target group data
        self.tar_indices = [self.chan_info.index(tcl) for tcl in self.tar_chan_list]
        self.target_group = self.task_data[:,self.tar_indices,:]  # (Nt,Nc,Np)
        if self.allow_target_group:  # allowed to use target group channels
            self.alter_indices = np.delete(np.arange(self.n_chans), self.tar_index)
        else:  # not allowed to use target group channels
            self.alter_indices = np.delete(np.arange(self.n_chans), self.tar_indices)

        # model initialization
        try:
            self.init_value = np.mean(self.tar_functions[self.tar_func](
                X_train=self.target_group,
                y_train=self.y_train,
                kwargs=self.kwargs
            ))
        except KeyError:
            raise Exception('Check your kwargs parameters according to the target function!')
        self.model_indices, self.value_change = [], [self.init_value]


    def srca_unit(self,
        chans_indices: List) -> float:
        """Compute updated target function values of tdSRCA-processed data.

        Args:
            chans_indices (list or tuple): Indices of channels to be used in model.

        Returns:
            tdsrca_tar_value (float): Target function values of the tdSRCA-processed data.
        """
        tdsrca_target = srca_process(
            rs_model=self.rest_data[:,chans_indices,:],
            rs_target=self.rest_target,
            ts_model=self.task_data[:,chans_indices,:],
            ts_target=self.task_target,
            regression=self.regression
        )
        update_target_group = deepcopy(self.target_group)
        update_target_group[:,self.tar_chan_list.index(self.tar_chan),:] = tdsrca_target
        tdsrca_tar_value = np.mean(self.tar_functions[self.tar_func](
            X_train=self.update_target_group,
            y_train=self.y_train,
            kwargs=self.kwargs
        ))
        return tdsrca_tar_value


class TdESRCA(ESRCA):
    """Intermediate process of MultiESRCA
        (i) multi-channel (2-D) target function
        (ii) optimization on single channel
        (iii) optimization on multiple event
    Target functions (2-D):
        (1) DSP target function value
        (2) DSP classification accuracy
    """
    tar_functions = {'DSP-val':dsp_val,
                     'TDCA-val':tdca_val}
    opt_methods = ['Traversal', 'Recursion', 'Mix']


    def __init__(self,
        X_train: ndarray,
        y_train: ndarray,
        rest_phase: List,
        task_phase: List,
        chan_info: List,
        tar_chan: str,
        tar_chan_list: List,
        tar_func: str = 'TRCA-val',
        opt_method: str = 'Recursion',
        regression: str = 'MSE',
        allow_target_group: bool = False,
        traversal_limit: Optional[int] = None,
        chan_num_limit: Optional[int] = None,
        kwargs: Optional[dict] = None):
        """Load in settings.

        Args:
            X_train (ndarray): (Nt,Nc,Np). Training dataset of 1 category.
            y_train (ndarray): (Nt,). Labels for X_train.
            rest_phase (list): [st,ed]. The start and end point of rest-state data.
            task_phase (list): [st,ed]. The start and end point of task-state data.
            chan_info (list): Names of all channels.
            tar_chan (str): Names of present target channel.
            tar_chan_list (List[str]): Names of all target channels.
            tar_func (str): 'TRCA-val'.
            opt_method (str): 'Traversal', 'Recursion' or 'Mix'.
            regression (str, optional): Regression method used in SRCA process.
                'MSE','OLS','RI','LA' or 'EN'. Defaults to 'MSE'.
            allow_target_group (bool): Allow channels from target group to be estimation
                channels or not.
            traversal_limit (int, optional): The maximum number of channels to be traversed.
                Defaults to None.
            chan_num_limit (int, optional): The maximum number of channels used in tdSRCA model.
                Defaults to None.
            kwargs (dict, optional): {'event_type':ndarray (Ne,),
                                      'n_components':int}
        """
        super().__init__(
            X_train=X_train,
            y_train=y_train,
            rest_phase=rest_phase,
            task_phase=task_phase,
            chan_info=chan_info,
            tar_chan=tar_chan,
            tar_func=tar_func,
            opt_method=opt_method,
            regression=regression,
            traversal_limit=traversal_limit,
            chan_num_limit=chan_num_limit
        )

        # check extra input
        assert set(tar_chan_list) <= set(chan_info), 'Unknown target channel!'
        self.tar_chan_list = tar_chan_list
        self.allow_target_group = allow_target_group
        self.kwargs = kwargs


    def prepare(self):
        """Initialization for training."""
        # pick up target data for both state
        self.tar_index = self.chan_info.index(self.tar_chan)
        self.rest_target = self.rest_data[:,self.tar_index,:]  # (Nt,Np)
        self.task_target = self.task_data[:,self.tar_index,:]  # (Nt,Np)

        # config target group data
        self.tar_indices = [self.chan_info.index(tcl) for tcl in self.tar_chan_list]
        self.target_group = self.task_data[:,self.tar_indices,:]  # (Nt,Nc,Np)
        if self.allow_target_group:  # allowed to use target group channels
            self.alter_indices = np.delete(np.arange(self.n_chans), self.tar_index)
        else:  # not allowed to use target group channels
            self.alter_indices = np.delete(np.arange(self.n_chans), self.tar_indices)

        # model initialization
        self.init_value = np.mean(self.tar_functions[self.tar_func](
            X_train=self.target_group,
            y_train=self.y_train,
            kwargs=self.kwargs))
        self.model_indices, self.value_change = [], [self.init_value]
        return self


    def srca_unit(self,
        chans_indices: Tuple) -> float:
        """Compute updated target function values of td-eSRCA-processed data.

        Args:
            chans_indices (Tuple): Indices of channels to be used in model.

        Returns:
            tdesrca_tar_value (float): Target function values of the tdSRCA-processed data.
        """
        tdesrca_target = srca_process(
            rs_model=self.rest_data[:,chans_indices,:],
            rs_target=self.rest_target,
            ts_model=self.task_data[:,chans_indices,:],
            ts_target=self.task_target,
            regression=self.regression
        )
        update_target_group = deepcopy(self.target_group)
        update_target_group[:,self.tar_chan_list.index(self.tar_chan),:] = tdesrca_target
        tdesrca_tar_value = np.mean(self.tar_functions[self.tar_func](
            X_train=update_target_group,
            y_train=self.y_train,
            kwargs=self.kwargs
        ))
        return tdesrca_tar_value


class MultiSRCA(SRCA):
    """Spatial Regression Component Analysis for multi-channel, single-event optimization.
    Target functions (1-D):
        (1) TRCA target function value
    """
    pass


class MultiESRCA(object):
    """Spatial Regression Component Analysis for multi-channel, multi-event optimization.
    Target functions (2-D):
        (1) TRCA classification accuracy
        (2) DSP target value
        (3) DSP classification accuracy
    """
    tar_functions = {'DSP-val':dsp_val,
                     'TDCA-val':tdca_val}


    def __init__(self,
        X_train: ndarray,
        y_train: ndarray,
        rest_phase: List,
        task_phase: List,
        chan_info: List,
        tar_chan_list: List,
        tar_func: str = 'DSP-val',
        opt_method: str = 'Recursion',
        regression: str = 'MSE',
        allow_target_group: bool = False,
        traversal_limit: Optional[int] = None,
        chan_num_limit: Optional[int] = None,
        kwargs: Optional[dict] = None):
        """Load in settings.

        Args:
            X_train (ndarray): (Nt,Nc,Np). Training dataset of 1 category.
            y_train (ndarray): (Nt,). Labels for X_train.
            rest_phase (list): [st,ed]. The start and end point of rest-state data.
            task_phase (list): [st,ed]. The start and end point of task-state data.
            chan_info (list): Names of all channels.
            tar_chan_list (List[str]): Names of all target channels.
            tar_func (str): 'DSP-val' or 'TDCA-val'.
            opt_method (str): 'Traversal', 'Recursion' or 'Mix'.
            regression (str, optional): Regression method used in SRCA process.
                'MSE','OLS','RI','LA' or 'EN'. Defaults to 'MSE'.
            allow_target_group (bool): Use target channel as estimation channel.
            traversal_limit (int, optional): The maximum number of channels to be traversed.
                Defaults to None.
            chan_num_limit (int, optional): The maximum number of channels used in tdSRCA model.
                Defaults to None.
            kwargs (dict, optional): {'event_type':ndarray (Ne,),
                                      'n_components':int,
                                      (Below items exist when tar_func='TDCA-val')
                                      'projection':ndarray (Ne,Np,Np),
                                      'extra_length':int,
                                      'extra_data':ndarray (Ne*Nt,Nc,m)}
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.train_data = deepcopy(self.X_train)
        self.rest_phase = rest_phase
        self.task_phase = task_phase
        self.rest_data = X_train[...,self.rest_phase[0]:self.rest_phase[1]]
        self.task_data = X_train[...,self.task_phase[0]:self.task_phase[1]]

        # load in settings
        self.n_chans = X_train.shape[-2]
        self.chan_info = chan_info
        self.tar_func = tar_func
        self.opt_method = opt_method
        self.traversal_limit = traversal_limit
        self.chan_num_limit = chan_num_limit
        self.regression = regression

        # check extra input
        assert set(tar_chan_list) <= set(chan_info), 'Unknown target channel!'
        self.tar_chan_list = tar_chan_list

        if set(self.tar_chan_list) == set(self.chan_info) and allow_target_group == False:
            raise Exception('No available estimation channels!')
        self.allow_tar_group = allow_target_group

        self.kwargs = kwargs


    def prepare(self):
        """Initialization for training."""
        # config target group data
        self.tar_indices = [self.chan_info.index(tcl) for tcl in self.tar_chan_list]
        self.target_group = self.task_data[:,self.tar_indices,:]  # (Ne*Nt,Nc,Np)
        if self.allow_target_group:
            self.alter_indices = deepcopy(self.chan_info)
        else:
            self.alter_indices = np.delete(np.arange(self.n_chans), self.tar_indices)

        # model initialization
        self.init_value = np.mean(self.tar_functions[self.tar_func](
            X_train=self.target_group,
            y_train=self.y_train,
            kwargs=self.kwargs
        ))
        self.model_indices, self.value_change = [], [self.init_value]


    def check_progress_3(self):
        """Check if there's any improvement after TdESRCA for one channel."""
        if self.value_change[-1] < max(self.value_change):  # worse after td-eSRCA
            self.multiesrca_model[-1] = []  # abandon the results of the last-round training
            return False
        return True


    def train(self):
        self.multiesrca_model = []
        # print('Start. Initial value: {}'.format(str(self.init_value)))
        for nc,tc in enumerate(self.tar_chan_list):  # tc: str
            tar_index = self.chan_info.index(tc)

            # train one channel each loop
            td_model = TdESRCA(
                X_train=self.train_data,
                y_train=self.y_train,
                rest_phase=self.rest_phase,
                task_phase=self.task_phase,
                chan_info=self.chan_info,
                tar_chan=tc,
                tar_chan_list=self.tar_chan_list,
                tar_func=self.tar_func,
                opt_method=self.opt_method,
                regression=self.regression,
                allow_target_group=self.allow_target_group,
                traversal_limit=self.traversal_limit,
                chan_num_limit=self.chan_num_limit,
                kwargs=self.kwargs
            )
            td_model.prepare()
            td_model.train()
            self.multiesrca_model.append(td_model.srca_model)
            self.value_change.append(td_model.value_change[-1])

            # check whether the training results of Multi-eSRCA should be kept
            update_data = self.check_progress_3()
            if update_data:  # update the channel trained before
                self.train_data[:,tar_index,self.task_phase[0]:self.task_phase[1]] = apply_SRCA(
                    rest_data=self.train_data[...,self.rest_phase[0]:self.rest_phase[1]],
                    task_data=self.train_data[...,self.task_phase[0]:self.task_phase[1]],
                    target_chan=tc,
                    model_chans=self.multiesrca_model[nc],
                    chan_info=self.chan_info
                )
                # print('Training data update! Current target channel: {}'.format(tc))
                # print('Current value: {}'.format(str(self.value_change[-1])))


# %% Terminal function
def apply_SRCA(
    rest_data: ndarray,
    task_data: ndarray,
    target_chan: str,
    model_chans: List,
    chan_info: List,
    regression: str = 'MSE') -> ndarray:
    """Apply SRCA model to EEG data.

    Args:
        rest_data (ndarray): (Nt,Nc,Np). Rest-state data of all channels.
        task_data (ndarray): (Nt,Nc,Np). Task-state data of all channels.
        target_chan (str): Name of target channel.
        model_chans (List[str]): Names of model channels.
        chan_info (List[str]): Names of all channels.
        regression (str): Regression method used in SRCA process.
            'MSE','OLS','RI','LA' or 'EN'. Defaults to 'MSE'.

    Returns:
        srca_extraction (ndarray): (Nt,Np).
    """
    target_idx = chan_info.index(target_chan)
    if not model_chans:
        return task_data[...,target_idx,:]
    model_indices = [chan_info.index(x) for x in model_chans]

    # main process of SRCA
    srca_extraction = srca_process(
        rs_model=rest_data[:,model_indices,:],
        rs_target=rest_data[:,target_idx,:],
        ts_model=task_data[:,model_indices,:],
        ts_target=task_data[:,target_idx,:],
        regression=regression
    )
    return srca_extraction


def apply_channel_selection(
    task_data: ndarray,
    target_group: List,
    srca_model: List,
    chan_info: List) -> ndarray:
    """Select channels according to SRCA model.

    Args:
        task_data (ndarray): (Nt,Nc,Np). Task-state data of all channels.
        target_group (List[str]): Names of channels in target group.
        srca_model (List[List[str]]): Names of model channels of each target channels.
        chan_info (List[str]): Names of all channels.

    Returns:
        srca_augmentation (ndarray): (Nt,Nc(new),Np).
    """
    channel_groups = set(list(chain(*srca_model)) + target_group)
    aug_indices = [chan_info.index(cg) for cg in channel_groups]
    return task_data[:,aug_indices,:]