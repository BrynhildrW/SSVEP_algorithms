# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Task-state Channel Augmentation (TCA) series.

Supported objects
1. TCA: multiple channels & single event
    Target functions: TRCA-val
    Optimization methods: Traversal, Recursion, Mix

2. ETCA: multiple channels & multiple events (Ensemble-TCA)
    Target functions: DSP-val, DSP-acc
    Optimization methods: Traversal, Recursion, Mix

"""

# %% basic modules
from typing import Optional, List, Any, Dict, Tuple

import utils

import trca
import dsp

import numpy as np
from numpy import ndarray

import scipy.linalg as sLA

from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from itertools import combinations, chain

from time import perf_counter
from copy import deepcopy


# %% Target functions of TCA
def _trca_coef(
        X: ndarray,
        y: ndarray,
        kwargs: Dict[str, Any]) -> float:
    """Target function values of TRCA.

    Args:
        X (ndarray): (Nt,Nc,Np). Single-event training dataset. Nt>=2.
        y (ndarray): (Nt,). Labels for X_train.
        kwargs:
            n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns:
        coef (float): (w @ S @ w.T) / (w @ Q @ w.T)
    """
    # train TRCA model
    trca_model = trca.TRCA(ensemble=False, n_components=kwargs['n_components'])
    trca_model.fit(X_train=X, y_train=y)

    # compute target function value
    w = trca_model.training_model['w'].squeeze()  # (Nk,Nc)
    Q, S = trca_model.training_model['Q'], trca_model.training_model['S']
    return np.mean(w @ S @ w.T) / np.mean(w @ Q @ w.T)


def _trca_corr(
        X: ndarray,
        y: ndarray,
        kwargs: Dict[str, Any]) -> float:
    """Pearson correlation coefficient of w @ X_test and w @ X_train.mean().

    Args:
        X (ndarray): (Nt,Nc,Np). Single-event training dataset. Nt>=2.
        y (ndarray): (Nt,). Labels for X_train.
        kwargs:
            n_splits (int): Number of splits for StratifiedKFold.
            n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns:
        coef (float).
    """
    coef = 0
    skf = StratifiedKFold(n_splits=kwargs['n_splits'])
    for train_index, valid_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_valid = X[valid_index]  # (Ne(1)*Nte,Nc,Np)

        # train TRCA spatial filters & templates
        classifier = trca.TRCA(ensemble=False, n_components=kwargs['n_components'])
        classifier.fit(X_train=X_train, y_train=y_train)
        w, wX = classifier.training_model['w'][0], classifier.training_model['wX'][0]

        # compute correlation coefficients
        X_temp = np.einsum('kc,ncp->nkp', w, X_valid)  # (Nte,Nk,Np)
        coef += np.sum(utils.pearson_corr(X=wX, Y=X_temp, parallel=True))
    return coef / kwargs['n_splits']


def _dsp_coef(
        X: ndarray,
        y: ndarray,
        kwargs: Dict[str, Any]) -> float:
    """Target function values of DSP.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X_train.
        kwargs:
            n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns:
        coef (float): (w @ Sb @ w.T) / (w @ Sw @ w.T)
    """
    dsp_model = dsp.DSP(n_components=kwargs['n_components'])
    dsp_model.fit(X_train=X, y_train=y)
    w = dsp_model.training_model['w']  # (Ne,Nk,Nc)
    Sb, Sw = dsp_model.training_model['Sb'], dsp_model.training_model['Sw']
    return np.mean(w @ Sb @ w.T) / np.mean(w @ Sw @ w.T)


def _dsp_corr(
        X: ndarray,
        y: ndarray,
        kwargs: Dict[str, Any]) -> float:
    """Pearson correlation coefficient of w @ X_test and w @ X_train.mean().

    Args:
        X (ndarray): (Nt,Nc,Np). Single-event training dataset. Nt>=2.
        y (ndarray): (Nt,). Labels for X_train.
        kwargs:
            n_splits (int): Number of splits for StratifiedKFold.
            n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns:
        coef (float).
    """
    pass


def _dsp_acc(
        X: ndarray,
        y: ndarray,
        kwargs: Dict[str, Any]) -> float:
    """DSP classification accuracy.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X.
        kwargs:
            n_splits (int): Number of splits for StratifiedKFold.
            n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns:
        acc (float).
    """
    skf = StratifiedKFold(n_splits=kwargs['n_splits'])
    acc = 0
    for train_index, valid_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_valid, y_valid = X[valid_index], y[valid_index]

        classifier = dsp.DSP(n_components=kwargs['n_components'])
        classifier.fit(X_train=X_train, y_train=y_train)
        y_dsp = classifier.predict(X_test=X_valid)
        acc += utils.acc_compute(y_pred=y_dsp, y_true=y_valid)
    return acc / kwargs['n_splits']


def _trca_acc(
        X: ndarray,
        y: ndarray,
        kwargs: Dict[str, Any]) -> float:
    """TRCA classification accuracy (OvR strategy).

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X.
        kwargs:
            n_splits (int): Number of splits for StratifiedKFold.
            n_components (int): Number of eigenvectors picked as filters. Nk.
            target_event (int): Label of target event.

    Returns:
        acc (float).
    """
    skf = StratifiedKFold(n_splits=kwargs['n_splits'])
    acc = 0
    for train_index, valid_index in skf.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_valid, y_valid = X[valid_index], y[valid_index]
        classifier = trca.TRCA(
            standard=kwargs['standard'],
            ensemble=kwargs['ensemble'],
            n_components=kwargs['n_components']
        )
        classifier = classifier.fit(X_train=X_train, y_train=y_train)
        y_standard, y_ensemble = classifier.predict(X_test=X_valid)
        if kwargs['standard']:
            acc += utils.acc_compute(y_pred=y_standard, y_true=y_valid)
        if kwargs['ensemble']:
            acc += utils.acc_compute(y_pred=y_ensemble, y_true=y_valid)
    return acc / kwargs['n_splits']


# %% Main TCA classes
class TCA(object):
    """Task-state Channel Augmentation for mutliple-channel, single-event optimization.
    Target functions (1-D):
        TRCA target function value.
    """
    tar_functions = {'TRCA-coef': _trca_coef,
                     'TRCA-corr': _trca_corr}
    iterative_methods = ['Forward', 'Forward-Stepwise']

    def __init__(
        self,
        X_train: ndarray,
        y_train: ndarray,
        chan_info: List[str],
        init_chan_list: List[str],
        tar_func: str = 'TRCA-coef',
        iterative_method: str = 'Stepwise',
        chan_num_limit: Optional[int] = None,
        kwargs: Dict[str, Any] = None
    ):
        """Config basic settings.

        Args:
            X_train (ndarray): (Nt,Nc,Np). Single-event training dataset.
            y_train (ndarray): (Nt*Ne,). Labels for X_train.
            chan_info (List[str]): Names of all channels.
            init_chan_list (List[str]): Names of initial channels.
            tar_func (str): 'TRCA-val'.
            iterative_method (str): 'Forward' or 'Stepwise'.
            chan_num_limit (int, optional): The maximum number of channels used in SRCA model.
                Defaults to None.
            kwargs: Dict[str, Any]
                n_components (int): Number of eigenvectors picked as filters. Nk.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.n_chans = X_train.shape[-2]  # Nc
        self.chan_info = chan_info
        self.tar_func = tar_func
        self.iterative_method = iterative_method
        self.chan_num_limit = chan_num_limit
        self.kwargs = kwargs
        assert set(init_chan_list) <= set(self.chan_info), 'Unknown channel!'
        self.init_chan_list = init_chan_list

    def prepare(self):
        """Initialization for training."""
        # config target group data
        self.init_indices = [self.chan_info.index(icl) for icl in self.init_chan_list]
        self.alter_indices = np.delete(np.arange(self.n_chans), self.init_indices)

        # model initialization
        self.init_value = np.mean(
            self.tar_functions[self.tar_func](
                X=self.X_train[:, self.init_indices, :],
                y=self.y_train,
                kwargs=self.kwargs
            ))
        self.chan_indices = deepcopy(self.init_indices)
        self.value_change = [self.init_value]

    def check_recursion(self):
        """Check input while optimization method is 'Recursion'."""
        warning_msg = 'Setting inappropriate channels for recursion!'
        assert 2 <= self.chan_num_limit < self.n_chans, warning_msg

    def tca_unit(
        self,
        channel_indices: List[int]
    ) -> float:
        """Compute updated target function values of TCA-processed data.

        Args:
            chans_indices (List[int]): Indices of channels to be used in TCA model.

        Returns:
            tca_coef (float): Target function values of the TCA-processed data.
        """
        tca_coef = np.mean(
            self.tar_functions[self.tar_func](
                X=self.X_train[:, channel_indices, :],
                y=self.y_train,
                kwargs=self.kwargs
            ))
        return tca_coef

    def step_forward(self):
        """Add one channel respectively and pick the best one."""
        self.recursion_combi = [self.chan_indices + [x]
                                for x in self.alter_indices if x not in self.chan_indices]
        results = list(map(self.tca_unit, self.recursion_combi))

        # update target function value & channel group (temporally)
        best_value = max(results)
        self.value_change.append(best_value)
        self.chan_indices = self.recursion_combi[results.index(best_value)]

    def check_forward(self):
        """Check if there's any improvement after step_forward()."""
        if self.value_change[-1] - np.max(self.value_change[:-1]) < 0.001:  # worse
            # the last channel is the latest added one, just delete it.
            del self.chan_indices[-1]  # abandon the results after step_forward()
            self.continue_forward = False

    def back_and_forth(self):
        """Delete one & add one respectively, then pick the best combination."""
        # drop out an existed channel except the latest one added in
        temp_combi = combinations(self.chan_indices[:-1], len(self.chan_indices)-2)
        remain_combi = [list(tc) + [self.chan_indices[-1]] for tc in temp_combi]

        # add up a new channel that has not been used (self.chan_indices) before
        add_combi = [[x] for x in self.alter_indices if x not in self.chan_indices]
        self.recursion_combi = [rc + ac for rc in remain_combi for ac in add_combi]
        results = list(map(self.tca_unit, self.recursion_combi))

        # update target function value & channel group (temporally)
        best_value = max(results)
        self.value_change.append(best_value)
        self.uncheck_chan_indices = self.recursion_combi[results.index(best_value)]

    def check_stepwise(self):
        """Check if there's any improvement after back_and_forth()."""
        if self.value_change[-1] - np.max(self.value_change[:-1]) < 0.001:  # worse
            del self.uncheck_chan_indices  # abandon the results after back_and_forth()
            self.continue_stepwise = False
        else:  # better or remain after stepwise operation
            self.chan_indices = deepcopy(self.uncheck_chan_indices)  # update channel group
            self.alter_indices = [nc for nc in np.arange(self.n_chans)
                                  if nc not in self.chan_indices]

    def check_limit(self):
        """Check whether the number of channels has met the limitation."""
        if self.chan_num_limit is not None and len(self.chan_indices) == self.chan_num_limit:
            self.continue_forward = False
            self.continue_stepwise = False
            self.continue_training = False

    def iterate_forward(self):
        """Use forward method to train TCA model."""
        self.continue_forward = True
        n_forward = 0
        while self.continue_forward:
            n_forward += 1
            self.step_forward()
            self.check_forward()
            self.check_limit()
            if self.continue_forward:
                print('Forward round {}: {}, keep training'.format(
                    n_forward, self.value_change[-1]
                ))
            else:
                print('Iteration finished.')

    def iterate_forward_stepwise(self):
        """Forward -> Stepwise (-> Forward -> Stepwise)."""
        self.continue_training = True
        while self.continue_training:
            # Forward iteration
            self.continue_forward = True
            n_forward = 0
            while self.continue_forward:
                n_forward += 1
                self.step_forward()
                self.check_forward()
                self.check_limit()
                if self.continue_forward:
                    print('Forward round {}: {}, keep training'.format(
                        n_forward, self.value_change[-1]
                    ))
                else:  # abandon useless record
                    print('Forward round {}: {}, switch to Stepwise'.format(
                        n_forward, self.value_change[-1]
                    ))

            # Stepwise iteration
            self.continue_stepwise = True
            n_stepwise = 0
            while self.continue_stepwise:
                n_stepwise += 1
                self.back_and_forth()
                self.check_stepwise()
                self.check_limit()
                if self.continue_stepwise:
                    print('Stepwise round {}: {}, keep training'.format(
                        n_stepwise, self.value_change[-1]
                    ))
                else:
                    print('Stepwise round {}: {}, switch to Forward'.format(
                        n_stepwise, self.value_change[-1]
                    ))

            # stop training
            if n_forward == n_stepwise == 1:
                print('Failed Forward & Stepwise successively, training finish.')
                self.continue_training = False

    operations = dict(zip(iterative_methods,
                          [iterate_forward, iterate_forward_stepwise]))

    def train(self):
        """Total training process."""
        self.operations[self.iterative_method](self)
        self.tca_model = [self.chan_info[x] for x in self.chan_indices]


class ETCA(TCA):
    """TCA for mutliple-channel, multiple-event optimization.
    Target functions (2-D):
        DSP target function value.
    """
    tar_functions = {'DSP-coef': _dsp_coef,
                     'DSP-acc': _dsp_acc,
                     'TRCA-acc': _trca_acc}


# %% Terminal function for TCA
def _generate_sparse_filter(
        n_chans: int,
        chan_indices: List[int],
        spatial_filter: ndarray) -> ndarray:
    """Generate sparse spatial filter according to different channel group.

    Args:
        n_chans (int): Total number of all channels.
        chan_indices (List[int]): Indices of channels used in spatial_filter.
        spatial_filter (ndarray): (Nk,Nc(partial))
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns:
        sparse_filter (ndarray): (Nk,Nc(total))
    """
    sparse_filter = np.zeros((spatial_filter.shape[0], n_chans))  # (Nk,Nc)
    for nmci, mci in enumerate(chan_indices):
        sparse_filter[:, mci] = spatial_filter[:, nmci]
    return sparse_filter


class TCA_TRCA(trca.TRCA):
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        chan_info: List[str],
        tca_model: List[List[str]]
    ):
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.chan_info = chan_info
        self.event_type = np.unique(self.y_train)
        self.n_events = len(self.event_type)
        self.n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        self.n_points = X_train.shape[-1]

        # train models according to spatial-augmented data
        self.w = np.zeros((self.n_events, self.n_components, len(self.chan_info)))
        self.wX = np.zeros((self.n_events, self.n_components, self.n_points))
        for ne, et in enumerate(self.event_type):
            # select channels according to TCA model
            chan_indices = [self.chan_info.index(tm) for tm in tca_model[ne]]
            train_info = {
                'event_type': np.array([1]),
                'n_events': 1,
                'n_train': np.array([self.n_train[ne]]),
                'n_chans': len(chan_indices),
                'n_points': self.X_train.shape[-1],
                'standard': True,
                'ensemble': False
            }

            # train TRCA filters & templates
            training_model = trca._trca_kernel(
                X_train=self.X_train[self.y_train == et][:, chan_indices, :],
                y_train=np.ones((self.n_train[ne])),
                train_info=train_info,
                n_components=self.n_components
            )
            self.w[ne] = _generate_sparse_filter(
                chan_info=self.chan_info,
                model_chans=tca_model[ne],
                partial_filter=training_model['w'][0],
                n_components=self.n_components
            )
            self.wX[ne] = training_model['wX'][0]  # (Nk,Np)
        return self

    def transform(self, X_test: ndarray) -> ndarray:
        n_test = X_test.shape[0]  # Ne*Nte
        rho = np.zeros((n_test, self.n_events))
        for nte in range(n_test):
            for nem in range(self.n_events):
                X_temp = self.w[nem] @ X_test[nte]  # (Nk,Np)
                rho[nte, nem] = utils.pearson_corr(X=X_temp, Y=self.wX[nem])
        return rho

    def predict(self, X_test: ndarray) -> ndarray:
        self.rho = self.transform(X_test)
        self.y_predict = self.event_type[np.argmax(self.rho, axis=-1)]
        return self.y_predict


class TCA_DSP(dsp.DSP):
    pass
