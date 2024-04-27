# -*- coding: utf-8 -*-
"""
@ Author: Brynhildr Wu
@ Email: brynhildrwu@gmail.com

Spatialtemporal domain adaptation based on common latent subspace.

Notations:
    n_events: Ne
    n_train: Nt
    n_test: Nte
    train_trials: Ne*Nt
    test_trials: Ne*Nte
    n_chans: Nc
    n_points: Np
    n_components: Nk
    n_harmonics: Nh
    n_bands: Nb
    n_subjects: Ns

"""

# %% Basic modules
from abc import abstractmethod

import utils

import cca
import trca
import dsp

from typing import Optional, List, Tuple, Dict, Union
from numpy import ndarray

import numpy as np
import scipy.linalg as sLA

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% Backward-propagation model (spatial filter)
def joint_trca_kernel(
        X_target: ndarray,
        y_target: ndarray,
        train_info: dict,
        n_components: int = 1,
        joint: bool = False,
        theta: float = 0.5,
        X_source: ndarray = None,
        y_source: ndarray = None) -> Dict[str, ndarray]:
    """Calculate TRCA filters to obtain source activity on latent subspace.

    Args:
        X_target (ndarray): (Ne*Nt(t),Nc,Np). Sklearn-style target dataset. Nt>=2.
        y_target (ndarray): (Ne*Nt(t),). Labels for X_target.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int): Number of eigenvectors picked as filters. Nk.
        joint (bool): Use source dataset or not.
        X_source (ndarray): (Ne*Nt(s),Nc,Np). Source dataset of one subject.
        y_source (ndarray): (Ne*Nt(s),). Labels for X_source.

    Returns: Dict[str, ndarray]
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        w (ndarray): (Ne,Nk,Nc). Spatial filters of joint TRCA.
        ew (ndarray): (Ne*Nk,Nc). Common spatial filter of joint eTRCA.
        wX (ndarray): (Ne,Nk,Np). Source activity of target dataset (TRCA).
        ewX (ndarray): (Ne,Ne*Nk,Np). Source activity of target dataset (eTRCA).
    """
    # basic information of target dataset
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool
    target_train = train_info['n_train']  # [Nt1,Nt2,...]
    assert np.min(target_train) > 1, 'Insufficient target samples!'

    # S & Q of target dataset
    target_mean, target_var = utils.mean_and_var(X=X_target, y=y_target)
    S_target = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    for ne in range(n_events):
        S_target[ne] = target_mean[ne] @ target_mean[ne].T  # (Nc,Nc)
    # Q_target = target_var

    if not joint:  # only use target dataset
        Q, S = target_var, S_target
    else:  # use target & source dataset
        # basic information of source dataset
        source_train = np.array([np.sum(y_source == et) for et in event_type])
        assert np.min(source_train) > 1, 'Insufficient source samples!'

        # S & Q of source dataset
        source_mean, source_var = utils.mean_and_var(X=X_source, y=y_source)
        S_source = np.zeros_like(S_target)  # (Ne,Nc,Nc)
        for ne in range(n_events):
            S_source[ne] = source_mean[ne] @ source_mean[ne].T  # (Nc,Nc)
        # Q_source = source_var

        Q = (1 - theta) * target_var[ne] + theta * source_var[ne]
        S = (1 - theta) * S_target[ne] + theta * S_source[ne]

    # GEPs | train spatial filters
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        w[ne] = utils.solve_gep(A=S, B=Q, n_components=n_components)
    ew = np.reshape(w, (n_events*n_components, n_chans), 'C')  # (Ne*Nk,Nc)

    # source activities
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    ewX = np.zeros((n_events, ew.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX[ne] = w[ne] @ target_mean[ne]  # (Nk,Np)
        wX = utils.fast_stan_3d(wX)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = ew @ target_mean[ne]  # (Ne*Nk,Np)
        ewX = utils.fast_stan_3d(ewX)

    # backward-propagation model
    backward_model = {
        'Q': Q, 'S': S,
        'w': w, 'ew': ew,
        'wX': wX, 'ewX': ewX
    }
    return backward_model


def joint_dsp_kernel(
        X_target: ndarray,
        y_target: ndarray,
        train_info: dict,
        n_components: int = 1,
        joint: bool = False,
        theta: float = 0.5,
        X_source: ndarray = None,
        y_source: ndarray = None) -> Dict[str, ndarray]:
    """Construct DSP filters (backward-propagation model) based on the target-domain data
        (and source-domain data, maybe) to obtain source activity on latent subspace.

    Args:
        X_target (ndarray): (Ne*Nt(t),Nc,Np). Sklearn-style target dataset. Nt>=2.
        y_target (ndarray): (Ne*Nt(t),). Labels for X_target.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters. Nk.
        joint (bool): Use source dataset or not.
        X_source (ndarray): (Ne*Nt(s),Nc,Np). Source dataset of one subject.
        y_source (ndarray): (Ne*Nt(s),). Labels for X_source.

    Returns: Dict[str, ndarray]
        Sb (ndarray): (Nc,Nc). Scatter matrix of between-class difference.
        Sw (ndarray): (Nc,Nc). Scatter matrix of within-class difference.
        w (ndarray): (Nk,Nc). Common spatial filter of (joint) DSP.
        wX (ndarray): (Ne,Nk,Np). Source activity of target dataset.
    """
    # basic information of target dataset
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    target_train = train_info['n_train']  # [Nt1,Nt2,...]

    # Sb & Sw of target dataset
    target_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    target_total_mean = X_target.mean(axis=0)  # (Nc,Np)
    Sb_target = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    Sw_target = np.zeros_like(Sb_target)  # (Nc,Nc)
    for ne, et in enumerate(event_type):
        target_temp = X_target[y_target == et]  # (Nt,Nc,Np)
        target_mean[ne] = target_temp.mean(axis=0)  # (Nc,Np)
        Hb_target = target_mean[ne] - target_total_mean  # (Nc,Np)
        Hw_target = target_temp - target_mean[ne]  # (Nt,Nc,Np)

        Sb_target += Hb_target @ Hb_target.T
        for tt in range(target_train[ne]):  # samples for each event
            Sw_target += Hw_target[tt] @ Hw_target[tt].T
    Sb_target /= n_events
    Sw_target /= X_target.shape[0]

    if not joint:  # only use target dataset
        Sb, Sw = Sb_target, Sw_target
    else:
        # basic information of source dataset
        source_train = np.array([np.sum(y_source == et)
                                 for et in event_type])

        # Sb & Sw of target dataset
        source_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
        source_total_mean = X_source.mean(axis=0)  # (Nc,Np)
        Sb_source = np.zeros((n_chans, n_chans))  # (Nc,Nc)
        Sw_source = np.zeros_like(Sb_source)  # (Nc,Nc)
        for ne, et in enumerate(event_type):
            source_temp = X_source[y_source == et]  # (Nt,Nc,Np)
            source_mean[ne] = source_temp.mean(axis=0)  # (Nc,Np)
            Hb_source = source_mean[ne] - source_total_mean  # (Nc,Np)
            Hw_source = source_temp - source_mean[ne]  # (Nt,Nc,Np)

            Sb_source += Hb_source @ Hb_source.T
            for st in range(source_train[ne]):  # samples for each event
                Sw_source += Hw_source[st] @ Hw_source[st].T
        Sb_source /= n_events
        Sw_source /= X_source.shape[0]

        Sb = (1 - theta) * Sb_target + theta * Sb_source
        Sw = (1 - theta) * Sw_target + theta * Sw_source

    # GEPs | train spatial filter
    w = utils.solve_gep(A=Sb, B=Sw, n_components=n_components)  # (Nk,Nc)

    # source activities
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    for ne in range(n_events):
        wX[ne] = w @ target_mean[ne]
    wX = utils.fast_stan_3d(wX)

    # backward-propagation model
    backward_model = {
        'Sb': Sb, 'Sw': Sw,
        'w': w, 'wX': wX
    }
    return backward_model


# %% Forward-propagation model (aliasing matrix)
