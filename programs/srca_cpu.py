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

3. TDSRCA: single channel & single-event (Two-dimensional SRCA)
    Target functions (2-D): TRCA coef, TRCA eval
    Optimization methods: Traversal, Recursion, Mix

4. MCSRCA: multi-channel & single-event (Future Version)
    Target functions (2-D): DSP coef, 
    Optimization methods: Traversal, Recursion, Mix
    Combination optimization methods: SA(Simulated annealing), IBI(Item-by-item)

5. MCESRCA: multi-channel & multi-event (Future Version)
    Target functions:
    Optimization methods: Traversal, Recursion, Mix
    Combination optimization methods: SA, IBI

update: 2022/11/30

"""

# %% basic modules
import utils
import special
import trca
import cca

from special import DSP

import numpy as np

import scipy.linalg as sLA

from sklearn import linear_model
from itertools import combinations

from time import perf_counter

# %% 1-D target functions | single channel
# SNR (mean) in time domain
def snr_sequence(train_data, *args, **kwargs):
    """Signal-to-Noise ratio (sequence) in time domain.

    Args:
        train_data (ndarray): (..., n_trials, n_points). Input data.

    Returns:
        snr (ndarray): (..., 1, n_points). SNR sequence in time domain. 
    """
    pure_signal = train_data.mean(axis=-2, keepdims=True)  # (..., 1, n_points)
    signal_power = pure_signal**2  # (..., 1, n_points)
    noise_power = ((train_data-pure_signal)**2).mean(axis=-2, keepdims=True)  # (..., 1, n_points)
    snr = signal_power / noise_power  # (..., 1, n_points)
    return snr


# Fisher score (mean) in time domain
def fs_sequence(train_data, *args, **kwargs):
    """Fisher Score (sequence) in time domain.
 
    Args:
        train_data (ndarray): (n_events, n_trials, n_points). Data array.

    Returns:
        fs (ndarray): (1, n_points). Fisher-Score sequence.
    """
    n_events = train_data.shape[0]  # Ne
    dataset = [train_data[ne] for ne in range(n_events)]  # (event1, event2, ...)
    return utils.fisher_score(dataset)


# %% 2-D target functions | multiple channels
# Target function values of DSP
def dsp_val(train_data, n_components=1, ratio=None):
    """f(w)=(w @ S_b @ w.T)/(w @ S_w @ w.T).

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None' when n_component is not 'None'.

    Returns:
        coef (float): f(w)
    """
    dsp_model = DSP(
        n_components=n_components,
        ratio=ratio
    ).fit(train_data=train_data)
    return (dsp_model.w@dsp_model.Sb@dsp_model.w.T)/(dsp_model.w@dsp_model.Sw@dsp_model.w.T)


# Accuracy of DSP
def dsp_acc(dataset, n_train, n_repeat=10, n_components=1, ratio=None):
    """Accuracy calculated by DSP-M1.

    Args:
        dataset (ndarray): (n_events, n_trials, n_chans, n_points).
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
    n_trials = dataset.shape[1]  # Nt

    # cross-validation
    rand_order = np.arange(n_trials)
    acc = np.zeros((n_repeat))
    for nrep in range(n_repeat):
        np.random.shuffle(rand_order)
        train_data = dataset[:,rand_order[:n_train],...]
        test_data = dataset[:,rand_order[n_train:],...]
        dsp_model = DSP(
            n_components=n_components,
            ratio=ratio
        ).fit(train_data=train_data)
        acc[nrep] = utils.acc_compute(dsp_model.predict(test_data=test_data))
    return acc.mean()


# %% SRCA operation
def mse_regression(rs_model, rs_target, ts_model):
    """Linear regression for task-state target channel based on
        Mean squared error (Frobenius Norm).

        Args:
        rs_model (ndarray): (n_trials, n_chans, n_points).
            Rest-state data of model channels.
        rs_target (ndarray): (n_trials, n_points).
            Rest-state data of target channel.
        ts_model (ndarray): (n_trials, n_chans, n_points).
            Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (n_trials, n_points).
            Estimation of task-state data of target channel.
    """
    n_trials = ts_model.shape[0]  # Nt
    n_points = ts_model.shape[-1]  # Np
    ts_target_estimate = np.zeros((n_trials, n_points))  # (Nt,Np)
    for ntr in range(n_trials):
        projection = rs_target[ntr] @ rs_model[ntr].T @ sLA.inv(rs_model[ntr] @ rs_model[ntr].T)
        ts_target_estimate[ntr] = projection @ ts_model[ntr]
    return ts_target_estimate


def linear_regression(rs_model, rs_target, ts_model):
    """Argmin function based on Ordinary Least Squares (sklearn).

        Args:
        rs_model (ndarray): (n_trials, n_chans, n_points).
            Rest-state data of model channels.
        rs_target (ndarray): (n_trials, n_points).
            Rest-state data of target channel.
        ts_model (ndarray): (n_trials, n_chans, n_points).
            Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (n_trials, n_points).
            Estimation of task-state data of target channel.
    """
    n_trials = ts_model.shape[0]  # Nt
    n_points = ts_model.shape[-1]  # Np
    ts_target_estimate = np.zeros((n_trials, n_points))  # (Nt,Np)
    for ntr in range(n_trials):
        L = linear_model.LinearRegression().fit(rs_model[ntr].T, rs_target[ntr])
        ts_target_estimate[ntr] = L.coef_@ts_model + L.intercept_
    return ts_target_estimate


def ridge(rs_model, rs_target, ts_model):
    """Argmin function based on Ridge regression (sklearn).

        Args:
        rs_model (ndarray): (n_trials, n_chans, n_points).
            Rest-state data of model channels.
        rs_target (ndarray): (n_trials, n_points).
            Rest-state data of target channel.
        ts_model (ndarray): (n_trials, n_chans, n_points).
            Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (n_trials, n_points).
            Estimation of task-state data of target channel.
    """
    n_trials = ts_model.shape[0]  # Nt
    n_points = ts_model.shape[-1]  # Np
    ts_target_estimate = np.zeros((n_trials, n_points))  # (Nt,Np)
    for ntr in range(n_trials):
        L = linear_model.Ridge().fit(rs_model[ntr].T, rs_target[ntr])
        ts_target_estimate[ntr,:] = L.coef_@ts_model + L.intercept_
    return ts_target_estimate


def lasso(rs_model, rs_target, ts_model):
    """Argmin function based on Lasso regression (sklearn).

    Args:
        rs_model (ndarray): (n_trials, n_chans, n_points).
            Rest-state data of model channels.
        rs_target (ndarray): (n_trials, n_points).
            Rest-state data of target channel.
        ts_model (ndarray): (n_trials, n_chans, n_points).
            Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (n_trials, n_points).
            Estimation of task-state data of target channel.
    """
    n_trials = ts_model.shape[0]  # Nt
    n_points = ts_model.shape[-1]  # Np
    ts_target_estimate = np.zeros((n_trials, n_points))  # (Nt,Np)
    for ntr in range(n_trials):
        L = linear_model.Lasso().fit(rs_model[ntr].T, rs_target[ntr])
        ts_target_estimate[ntr,:] = L.coef_@ts_model + L.intercept_
    return ts_target_estimate


def elastic_net(rs_model, rs_target, ts_model):
    """Argmin function based on Elastic-Net regression (sklearn).

        Args:
        rs_model (ndarray): (n_trials, n_chans, n_points).
            Rest-state data of model channels.
        rs_target (ndarray): (n_trials, n_points).
            Rest-state data of target channel.
        ts_model (ndarray): (n_trials, n_chans, n_points).
            Task-state data of model channels.

    Returns:
        ts_target_estimate (ndarray): (n_trials, n_points).
            Estimation of task-state data of target channel.
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


def srca_process(rs_model, rs_target, ts_model, ts_target, regression='MSE'):
    """Main process of SRCA algorithm.

    Args:
        rs_model (ndarray): (n_trials, n_chans, n_points).
            Rest-state data of model channels.
        rs_target (ndarray): (n_trials, n_points).
            Rest-state data of target channel.
        ts_model (ndarray): (n_trials, n_chans, n_points).
            Task-state data of model channels.
        ts_target (ndarray): (n_trials, n_points).
            Task-state data of target channel.
        regression (str, optional): 'MSE', 'OLS', 'RI', 'LA' and 'EN'.
            Defaults to 'MSE'.

    Returns:
        ts_target_extraction (ndarray): (n_trials, n_points). SRCA processed task-state data of target channel.
    """
    ts_target_estimation = regressions[regression](
        rs_model=rs_model,
        rs_target=rs_target,
        ts_model=ts_model
    )
    return ts_target - ts_target_estimation


def esrca_process(rs_model, rs_target, ts_model, ts_target, regression='MSE'):
    """Main process of ensemble SRCA algorithm.

    Args:
        rs_model (ndarray): (n_events, n_trials, n_chans, n_points).
            Rest-state data of model channels.
        rs_target (ndarray): (n_events, n_trials, n_points).
            Rest-state data of target channel.
        ts_model (ndarray): (n_events, n_trials, n_chans, n_points).
            Task-state data of model channels.
        ts_target (ndarray): (n_events, n_trials, n_points).
            Task-state data of target channel.
        regression (str, optional): 'MSE', 'OLS', 'RI', 'LA' and 'EN'.
            Defaults to 'MSE'.

    Returns:
        ts_target_extraction (ndarray): (n_trials, n_points). SRCA processed task-state data of target channel.
    """
    # basic information
    n_events = ts_model.shape[0]

    # repeat SRCA process on n_events axis
    ts_target_estimation = np.zeros_like(ts_target)
    for ne in range(n_events):
        ts_target_estimation[ne] = regressions[regression](
            rs_model=rs_model[ne],
            rs_target=rs_target[ne],
            ts_model=ts_model[ne]
        )
    return ts_target - ts_target_estimation


# %% Main classes
class SRCA(object):
    """Spatial Regression Component Analysis for single-channel, single-event optimization.
    Target functions (1-D):
        (1) SNR (mean) in time domain
    """
    tar_functions = {'SNR':snr_sequence}
    opt_methods = ['Traversal', 'Recursion', 'Mix']


    def __init__(self, train_data, rest_phase, task_phase, chan_info, tar_chan, tar_func,
                 opt_method, traversal_limit=None, chan_num_limit=None, regression='MSE'):
        """Load in settings.

        Args:
            train_data (ndarray): (n_train, n_chans, n_points). Training dataset.
            rest_phase (list): [st,ed]. The start and end point of rest-state data.
            task_phase (list): [st,ed]. The start and end point of task-state data.
            chan_info (list): Names of all channels.
            tar_chan (str): Name of target channel.
            tar_func (str): 'SNR'.
            opt_method (str): 'Traversal', 'Recursion' or 'Mix'.
            traversal_limit (int, optional): The maximum number of channels to be traversed.
                Defaults to None.
            chan_num_limit (int, optional): The maximum number of channels used in SRCA model.
                Defaults to None.
            regression (str, optional): Regression method used in SRCA process. Defaults to 'MSE'.
        """
        # basic information
        self.rest_data = train_data[...,rest_phase[0]:rest_phase[1]]
        self.task_data = train_data[...,task_phase[0]:task_phase[1]]
        self.n_chans = train_data.shape[-2]
        self.chan_info = chan_info
        self.tar_chan = tar_chan
        self.tar_func = tar_func
        self.opt_method = opt_method
        self.traversal_limit = traversal_limit
        self.chan_num_limit = chan_num_limit
        self.regression = regression
        # print('Load in data...Complete')


    def prepare(self):
        """Initialization for training."""
        # pick up target data for both state.
        tar_index = self.chan_info.index(self.tar_chan)
        self.rest_target = self.rest_data[...,tar_index,:]
        self.task_target = self.task_data[...,tar_index,:]
        self.alter_indices = np.delete(np.arange(self.n_chans), tar_index)

        # model initialization
        self.init_value = np.mean(self.tar_functions[self.tar_func](self.task_target))
        self.model_indices, self.value_change = [], [self.init_value]
        # print('Prepare for training...Complete!')
        return self


    def check_traversal(self):
        """Check input while optimization method is 'Traversal'."""
        assert 0<self.traversal_limit<len(self.chan_info), 'Setting inappropriate channels for traversal!'
        self.traversal_combi = [c for c in combinations(self.alter_indices, self.traversal_limit)]


    def check_recursion(self):
        """Check input while optimization method is 'Recursion'."""
        assert 2<=self.chan_num_limit<self.n_chans, 'Setting inappropriate channels for recursion!'


    def srca_unit(self, chans_indices):
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
        srca_tar_value = np.mean(self.tar_functions[self.tar_func](srca_target))
        return srca_tar_value


    def step_forward(self):
        """Add one channel respectively and pick the best one."""
        self.recursion_combi = [self.model_indices+[x] for x in self.alter_indices if x not in self.model_indices]
        results = list(map(self.srca_unit, self.recursion_combi))

        # update temp model information
        best_value = max(results)
        self.value_change.append(best_value)
        self.model_indices = self.recursion_combi[results.index(best_value)]


    def back_and_forth(self):
        """Delete one & add one respectively, then pick the best combination."""
        # combinations(self.model_indices[:-1], len(self.model_indices)-2):
        # self.model_indices[:-1] | every channel will be evaluated except for the last one added
        # len(self.model_indices)-2 | delete one respectively, pick up the rest
        remain_combi = [list(c)+[self.model_indices[-1]] for c in combinations(self.model_indices[:-1], len(self.model_indices)-2)]
        # if x not in self.model_indices:
        # add up a new channel that has not been used before
        add_combi = [[x] for x in self.alter_indices if x not in self.model_indices]
        self.recursion_combi = [rc+ac for rc in remain_combi for ac in add_combi]
        results = list(map(self.srca_unit, self.recursion_combi))

        self.only_add = False  # in this part, the number of channles doesn't increase for sure
        best_value = max(results)
        self.value_change.append(best_value)
        self.uncheck_model_indices = self.recursion_combi[results.index(best_value)]


    def check_progress_1(self):
        """Check if there's any improvement after step_forward()."""
        if self.value_change[-1] < self.value_change[-2]:  # no improvement after adding a new channel
            del self.model_indices[-1]  # abandon the results after step_forward()
            self.continue_training = False


    def check_progress_2(self):
        """Check if there's any improvement after back_and_forth()."""
        if self.value_change[-1] < self.value_change[-2]:  # no improvement after stepwise operation
            del self.uncheck_model_indices  # abandon the results after back_and_forth()
            self.continue_training = False
            return
        # still has improvement
        self.model_indices = self.uncheck_model_indices  # update channel group


    def check_limit(self):
        """Check whether the number of channels has met the limitation."""
        if hasattr(self, 'chan_num_limit') and (len(self.model_indices)==self.chan_num_limit):
            self.continue_training = False


    # optimization method 1
    def traversal(self):
        """Directly traverse each channel group to train SRCA model."""
        self.results = list(map(self.srca_unit, self.traversal_combi))
        model_index = self.results.index(max(self.results))
        self.value_change.append(self.results[model_index])
        self.model_indices = list(self.traversal_combi[model_index])


    # optimization method 2
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


    # optimization method 3
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
        # print('Target function: {} \nOptimization method: {} \nStart training...'.format(self.tar_func, self.opt_method))
        start_time = perf_counter()
        self.operations[self.opt_method](self)
        self.srca_model = [self.chan_info[x] for x in self.model_indices]
        end_time = perf_counter()
        # print('Training Complete! Running time: {}s'.format(str(end_time-start_time)))
        return self


class ESRCA(SRCA):
    """ensemble-SRCA for single-channel, multi-event optimization.
    Target functions (1-D):
        (1) SNR (mean) in time domain
        (2) Fisher score (mean) | only for 2categories
    """
    tar_functions = {'SNR':snr_sequence,
                     'FS':fs_sequence}


    def srca_unit(self, chans_indices):
        """Compute updated target function values of eSRCA-processed data.

        Args:
            chans_indices (list or tuple): Indices of channels to be used in SRCA model.

        Returns:
            esrca_tar_value (float): Target function values of the SRCA-processed data.
        """
        esrca_target = esrca_process(
            rs_model=self.rest_data[...,chans_indices,:],
            rs_target=self.rest_target,
            ts_model=self.task_data[...,chans_indices,:],
            ts_target=self.task_target,
            regression=self.regression
        )
        esrca_tar_value = np.mean(self.tar_functions[self.tar_func](esrca_target))
        return esrca_tar_value


class TDSRCA(SRCA):
    pass


class TDESRCA(ESRCA):
    pass


class MultiSRCA(TDSRCA):
    pass


class MultiESRCA(TDESRCA):
    pass


# %% Terminal function
def apply_SRCA(rest_data, task_data, target_chan, model_chans, chan_info, regression='MSE'):
    """Apply SRCA model to EEG data.

    Args:
        rest_data (ndarray): (n_trials, n_chans, n_points). Rest-state data of all channels.
        task_data (ndarray): (n_trials, n_chans, n_points). Task-state data of all channels.
        target_chan (str): Name of target channel.
        model_chans (list of str): Names of model channels.
        chan_info (list of str): Names of all channels.
        regression (str, optional): Regression method used in SRCA process. Defaults to 'MSE'.

    Returns:
        srca_extraction (ndarray): (n_trials, n_points).
    """
    # config model & target data
    target_idx = chan_info.index(target_chan)
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


def apply_ESRCA(rest_data, task_data, target_chan, model_chans, chan_info, regression='MSE'):
    """Apply eSRCA model to EEG data.

    Args:
        rest_data (ndarray): (n_events, n_trials, n_chans, n_points). Rest-state data of all channels.
        task_data (ndarray): (n_events, n_trials, n_chans, n_points). Task-state data of all channels.
        target_chan (str): Name of target channel.
        model_chans (list of str): Names of model channels.
        chan_info (list of str): Names of all channels.
        regression (str, optional): Regression method used in SRCA process. Defaults to 'MSE'.

    Returns:
        srca_extraction (ndarray): (n_events, n_trials, n_points).
    """
    # config model & target data
    target_idx = chan_info.index(target_chan)
    model_indices = [chan_info.index(x) for x in model_chans]

    # main process of eSRCA
    srca_extraction = esrca_process(
        rs_model=rest_data[...,model_indices,:],
        rs_target=rest_data[...,target_idx,:],
        ts_model=task_data[...,model_indices,:],
        ts_target=task_data[...,target_idx,:],
        regression=regression
    )
    return srca_extraction


def apply_TDSRCA():
    pass



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
        model = SRCA(
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
    model = ESRCA(
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
        srca_train[ne,:,nc,:] = apply_SRCA(
            rest_data=train_data[ne,...,rest_phase[0]:rest_phase[1]],
            task_data=train_data[ne,...,task_phase[0]:task_phase[1]],
            target_chan=tc,
            model_chans=srca_model_chans[ne][nc],
            chan_info=chan_info
        )
        srca_test[ne,:,nc,:] = apply_SRCA(
            rest_data=test_data[ne,...,rest_phase[0]:rest_phase[1]],
            task_data=test_data[ne,...,task_phase[0]:task_phase[1]],
            target_chan=tc,
            model_chans=srca_model_chans[ne][nc],
            chan_info=chan_info
        )


esrca_train = np.zeros((n_events, n_train, len(target_chans), task_phase[1]-task_phase[0]))
esrca_test = np.zeros((n_events, n_trials-n_train, len(target_chans), task_phase[1]-task_phase[0]))
for nc,tc in enumerate(target_chans):
    esrca_train[...,nc,:] = apply_ESRCA(
        rest_data=train_data[...,rest_phase[0]:rest_phase[1]],
        task_data=train_data[...,task_phase[0]:task_phase[1]],
        target_chan=tc,
        model_chans=esrca_model_chans[nc],
        chan_info=chan_info
    )
    esrca_test[...,nc,:] = apply_ESRCA(
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
    snr_train_origin[:,nc,:] = snr_sequence(train_data[...,tar_idx,task_phase[0]:task_phase[1]]).squeeze()
    snr_test_origin[:,nc,:] = snr_sequence(test_data[...,tar_idx,task_phase[0]:task_phase[1]]).squeeze()
    snr_train_srca[:,nc,:] = snr_sequence(srca_train[...,nc,:]).squeeze()
    snr_test_srca[:,nc,:] = snr_sequence(srca_test[...,nc,:]).squeeze()

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
