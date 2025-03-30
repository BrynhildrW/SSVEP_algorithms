# -*- coding: utf-8 -*-
"""
Utilization functions.

Author: Brynhildr Wu <brynhildrwu@gmail.com>

Source codes: https://github.com/BrynhildrW/SSVEP_algorithms/blob/main/programs/utils.py

#. Data preprocessing:
    * centralization()
    * normalization()
    * standardization()
    * channel_normalization()
    * trial_normalization()
    * euclidean_alignment()
    * fast_stan_2d()
    * fast_stan_3d()
    * fast_stan_4d()
    * fast_stan_5d()
    * generate_data_info()

#. Data generation
    * augmented_events()
    * neighbor_edge()
    * neighbor_events()
    * selected_events()
    * reshape_dataset()
    * generate_mean()
    * generate_var()
    * generate_source_response()
    * spatial_filtering()

#. Feature integration
    * sign_sta()
    * combine_feature()

#. Algorithm evaluation
    * label_align()
    * acc_compute()
    * itr_compute()

#. Spatial distances
    * pearson_corr()
    * fast_corr_2d()
    * fast_corr_3d()
    * fast_corr_4d()
    * fisher_score()
    * snr_time()
    * euclidean_dist()
    * cosine_sim()
    * minkowski_dist()
    * s_estimator()

#. Temporally smoothing functions
    * tukeys_kernel()
    * weight_matrix()
    * laplacian_matrix()

#. Reduced QR decomposition
    * qr_projection()

#. Eigenvalue problems
    * pick_subspace()
    * normalize_eigenvectors()
    * solve_gep()

#. Signal generation
    * Imn()
    * sin_wave()
    * sine_template()
    * get_resample_sequence()
    * extract_periodic_impulse()
    * create_conv_matrix()
    * correct_conv_matrix()
    * solve_mcpe()

#. Filter-bank technology
    * generate_filter_bank()
    * (class) FilterBank()

#. Linear transformation
    * forward_propagation()
    * solve_coral()

**Notations**:

* Number of events: `Ne`
* Number of (training) samples (for each event): `Nt`
* Number of (testing) samples: `Nte`
* Total number of training trials: `Ne*Nt`
* Total number of testing trials: `Ne*Nte`
* Number of channels: `Nc`
* Number of sampling points: `Np`
* Number of dimensions of sub-space: `Nk`
* Number of harmonics for sinusoidal templates: `Nh`
* Number of filter banks: `Nb`
"""

# %% basic moduls
import numpy as np

from scipy import linalg as sLA
from scipy.signal import sosfiltfilt, cheby1, cheb1ord
from sklearn.base import BaseEstimator, TransformerMixin, clone
# from sklearn.metrics import confusion_matrix

from math import pi, log

import warnings
from numba import njit

import cv2

from pyriemann.utils.base import sqrtm, invsqrtm
from pyriemann.estimation import Shrinkage

from copy import deepcopy


# %% 1. data preprocessing
def centralization(X):
    r"""
    Zero-mean:
    :math:`\mathbb{E} \left(\pmb{x} \right) = 0`.

    Parameters
    ----------
    X: *ndarray of shape (...,Np).*
        Input dataset.

    Returns
    ----------
    out: *ndarray of shape (...,Np).*
        Data after centralization.
    """
    return X - X.mean(axis=-1, keepdims=True)


def normalization(X):
    r"""
    Normalization:
    :math:`\pmb{x} = \dfrac{\pmb{x} - \min(\pmb{x})}{\max(\pmb{x}) - \min(\pmb{x})}`.

    Parameters
    ----------
    X : *ndarray of shape (...,Np).*
        Input dataset.

    Returns
    ----------
    out : *ndarray of shape (...,Np).*
        Data after normalization.
    """
    X_min = np.min(X, axis=-1, keepdims=True)  # (...,1)
    X_max = np.max(X, axis=-1, keepdims=True)  # (...,1)
    return (X - X_min) / (X_max - X_min)


def standardization(X):
    r"""
    :math:`{\rm Var} \left( \pmb{x} \right) = 1`

    Parameters
    ----------
    X : *ndarray of shape (...,Np).*
        Input dataset

    Returns
    ----------
    out : *ndarray of shape (...,Np).*
        Data after standardization.
    """
    return centralization(X) / np.std(X, axis=-1, keepdims=True)


def channel_normalization(X):
    """
    Z-score normalization on each channel.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Input dataset

    Returns
    ----------
    out : *ndarray of shape (Ne*Nt,Nc,Np).*
        Data after normalization
    """
    return fast_stan_3d(X)


def trial_normalization(X):
    """
    After flattening the single-trial multi-channel data into a single channel,
    zero-mean and unit variance normalization are performed.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Input dataset.

    Returns
    ----------
    out : *ndarray of shape (Ne*Nt,Nc,Np).*
        Data after normalization.
    """
    trial_mean = np.mean(X, axis=(1, 2), keepdims=True)
    trial_var = np.std(X, axis=(1, 2), keepdims=True)
    return (X - trial_mean) / trial_var


def euclidean_alignment(X):
    r"""
    Euclidean alignment by projection :math:`\pmb{P}^T \pmb{X}:`

    :math:`\pmb{P} = \min \left\| \pmb{P}^T \pmb{X} @ \pmb{X}^T @ \pmb{P} - \pmb{I} \right\|_F^2.`

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Input dataset.

    Returns
    ----------
    P : *ndarray of shape (Nc,Nc).*
        Transformation matrix.
    X_ea : *ndarray of shape (Ne*Nt,Nc,Np).*
        Aligned ``X``.
    """
    # basic information
    X_ea = np.zeros_like(X)

    # alignment process
    P = invsqrtm(generate_var(X))  # faster than using solve_coral()
    for nts in range(X.shape[0]):
        X_ea[nts] = P @ X[nts]  # P = P.T
    return P, X_ea


@njit(fastmath=True)
def fast_stan_2d(X):
    """
    Use the JIT compiler to make ``standardization()`` run faster for 2-D input.

    Parameters
    ----------
    X : *ndarray of shape (d1,d2).*
        Input dataset.

    Returns
    ----------
    X_new : *ndarray of shape (d1,d2).*
        Data after standardization.
    """
    dim_1 = X.shape[0]
    X_new = np.zeros_like(X)
    for d1 in range(dim_1):
        X_new[d1, :] = X[d1, :] - np.mean(X[d1, :])  # centralization
        X_new[d1, :] = X_new[d1, :] / np.std(X_new[d1, :])
    return X_new


@njit(fastmath=True)
def fast_stan_3d(X):
    """
    Use the JIT compiler to make ``standardization()`` run faster for 3-D input.

    Parameters
    ----------
    X : *ndarray of shape (d1,d2,d3).*
        Input dataset.

    Returns
    ----------
    X_new : *ndarray of shape (d1,d2,d3).*
        Data after standardization.
    """
    X_new = np.zeros_like(X)
    for d1 in range(X.shape[0]):
        for d2 in range(X.shape[1]):
            X_new[d1, d2, :] = X[d1, d2, :] - np.mean(X[d1, d2, :])
            X_new[d1, d2, :] = X_new[d1, d2, :] / np.std(X_new[d1, d2, :])
    return X_new


@njit(fastmath=True)
def fast_stan_4d(X):
    """
    Use the JIT compiler to make ``standardization()`` run faster for 4-D input.

    Parameters
    ----------
    X : *ndarray of shape (d1,d2,d3,d4).*
        Input dataset.

    Returns
    ----------
    X_new : *ndarray of shape (d1,d2,d3,d4).*
        Data after standardization.
    """
    X_new = np.zeros_like(X)
    for d1 in range(X.shape[0]):
        for d2 in range(X.shape[1]):
            for d3 in range(X.shape[2]):
                X_new[d1, d2, d3, :] = X[d1, d2, d3, :] - np.mean(X[d1, d2, d3, :])
                X_new[d1, d2, d3, :] = X_new[d1, d2, d3, :] / np.std(X_new[d1, d2, d3, :])
    return X_new


@njit(fastmath=True)
def fast_stan_5d(X):
    """
    Use the JIT compiler to make ``standardization()`` run faster for 5-D input.

    Parameters
    ----------
    X : *ndarray of shape (d1,d2,d3,d4,d5).*
        Input dataset.

    Returns
    ----------
    X_new : *ndarray of shape (d1,d2,d3,d4,d5).*
        Data after standardization.
    """
    X_new = np.zeros_like(X)
    for d1 in range(X.shape[0]):
        for d2 in range(X.shape[1]):
            for d3 in range(X.shape[2]):
                for d4 in range(X.shape[3]):
                    X_new[d1, d2, d3, d4, :] = X[d1, d2, d3, d4, :] \
                        - np.mean(X[d1, d2, d3, d4, :])
                    X_new[d1, d2, d3, d4, :] = X_new[d1, d2, d3, d4, :] \
                        / np.std(X_new[d1, d2, d3, d4, :])
    return X_new


def generate_data_info(X, y):
    """
    Generate basic data information.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Input data.
    y : *ndarray of shape (Ne*Nt,).*
        Labels for ``X``.

    Returns (Dict)
    --------------
    'event_type' : *ndarray of shape (Ne,).*
        Unique labels.
    'n_events' : *int.*
        Number of events.
    'n_train' : *ndarray of shape (Ne,).*
        Trials of each event.
    'n_chans' : *int.*
        Number of channels.
    'n_points' : *int.*
        Number of sampling points.
    """
    event_type = np.unique(y)
    return {
        'event_type': event_type,
        'n_events': event_type.shape[0],
        'n_train': np.array([np.sum(y == et) for et in event_type]),
        'n_chans': X.shape[1],
        'n_points': X.shape[-1]
    }


# %% 2. data gerenation
def augmented_events(event_type, d):
    """
    Generate indices for merged events for each target event.
    Special function for ms- algorithms.

    Parameters
    ----------
    event_type : *ndarray of shape (Ne,).*
        Unique labels.
    d : *int.*
        The range of events to be merged.

    Returns
    ----------
    events_group : *dict.*
        {'events': [idx,]}
    """
    events_group = {}
    n_events = len(event_type)
    for ne, et in enumerate(event_type):
        if ne <= d / 2:
            events_group[str(et)] = np.arange(0, d, 1).tolist()
        elif ne >= int(n_events - d / 2):
            events_group[str(et)] = np.arange(n_events - d, n_events, 1).tolist()
        else:
            m = int(d / 2)  # forward augmentation
            events_group[str(et)] = np.arange(ne - m, ne - m + d, 1).tolist()
    return events_group


def neighbor_edge(total_length, neighbor_range, current_index):
    """
    Decide the edge index (based on labels) of neighboring stimulus area.

    Parameters
    ----------
    total_length : *int.*
        The length (sampling points) of data.
    neighbor_range : *int.*
        Must be an odd number.
    current_index : *int.*
        From ``0`` to ``total_lenth - 1``.

    Returns
    ----------
    (edge_idx_1, edge_idx_2) : *Tuple[int,int].*
        ``edge_idx_2`` is 1 more than the real index of the last element.
    """
    # check input
    assert neighbor_range % 2 == 0, "neighbor_range should be an odd number!"

    # main process
    half_length = int((neighbor_range - 1) / 2)
    if current_index <= half_length:  # left/upper edge
        return 0, current_index + half_length + 1
    # right/bottom edge
    elif current_index >= total_length - (half_length + 1):
        return current_index - half_length, total_length
    else:
        return current_index - half_length, current_index + half_length + 1


def neighbor_events(distribution, width):
    """
    Generate indices for merged events for each target event.

    Refers to: 10.1109/TIM.2022.3219497 (DOI).

    Parameters
    ----------
    distribution : *ndarray of shape (m,n).*
        Real spatial distribution (labels) of each stimuli.
    width : *int.*
        Parameter ``neighbor_range`` used in ``neighbor_edge()``. Must be an odd number.

    Returns
    ----------
    events_group : *Dict[str, List[int]].*
        {'event_id': [idx_1,idx_2,...]}.
    """
    n_rows, n_cols = distribution.shape
    events_group = {}
    for row in range(n_rows):
        upper, bottom = neighbor_edge(
            total_length=n_rows,
            neighbor_range=width,
            current_index=row
        )
        for col in range(n_cols):
            left, right = neighbor_edge(
                total_length=n_cols,
                neighbor_range=width,
                current_index=col
            )
            event_id = str(distribution[row, col])
            events_group[event_id] = distribution[upper:bottom, left:right].flatten().tolist()
    return events_group


def selected_events(n_events, select_num, select_method='2'):
    """
    Generate indices for selected events of total dataset. Special function for stCCA.

    Parameters
    ----------
    n_events : *int.*
        Number of total events.
    select_num : *int.*
        Number of selected events.
    method : *str, optional.*
        ``'1'``, ``'2'``, and ``'3'``. Defaults to ``'2'``.
        Details in https://ieeexplore.ieee.org/document/9177172/

    Returns
    ----------
    select_events : *List[int].*
        Indices of selected events.
    """
    if select_method == '1':
        return [1 + int((n_events - 1) * sen / (select_num - 1))
                for sen in range(select_num)]
    elif select_method == '2':
        return [int(n_events * (2 * sen + 1) / (2 * select_num))
                for sen in range(select_num)]
    elif select_method == '3':
        return [int(n_events * 2 * (sen + 1) / (2 * select_num))
                for sen in range(select_num)]


def reshape_dataset(data, labels=None, target_style='sklearn', filter_bank=False):
    """
    Reshape data array between public versionand sklearn version.

    Parameters
    ----------
    data : *ndarray. shape (Ne,Nt,Nc,Np) (public) or (Ne*Nt,Nc,Np) (sklearn).*
        If filter_bank is True, *shape (Nb,Ne,Nt,Nc,Np)* or *(Nb,Ne*Nt,Nc,Np)*.
    labels : *ndarray of shape (Ne*Nt,).*
        Labels for data (sklearn version). Defaults to ``None``.
    target_style : *str.*
        ``'public'`` or ``'sklearn'``. Target style of transformed dataset.
    filter_bank : *bool.*
        Multi-band data or single-band data.

    Returns
    ----------
    X_total : *ndarray of shape (Ne,Nt,Nc,Np) (public) or (Ne*Nt,Nc,Np) (sklearn).*
        If filter_bank is True, *shape (Nb,Ne,Nt,Nc,Np)* or *(Nb,Ne*Nt,Nc,Np)*.
    y_total : *ndarray of shape (Ne,).*
        Labels for ``X_total``. Only exist while target_style is ``'sklearn'``.
    """
    # basic information
    n_points = data.shape[-1]  # Np
    n_chans = data.shape[-2]  # Nc

    # main process
    if target_style == 'public':
        # extract labels:
        event_type = np.unique(labels)
        n_events = len(event_type)
        n_train = np.array([np.sum(labels == et) for et in event_type])
        n_trials = np.min(n_train)
        if n_trials != np.max(n_train):
            warnings.warn('Unbalanced dataset! Some trials will be discarded!')

        # reshape data
        if filter_bank:  # (Nb,Ne*Nt,Nc,Np)
            n_bands = data.shape[0]  # Nb
            X_total = np.zeros((n_bands, n_events, n_trials, n_chans, n_points))
            for nb in range(n_bands):
                for ne, et in enumerate(event_type):
                    X_total[nb, ne, ...] = data[nb][labels == et][:n_trials, ...]  # (Nt,Nc,Np)
        else:  # (Ne,Nt,Nc,Np)
            X_total = np.zeros((n_events, n_trials, n_chans, n_points))
            for ne, et in enumerate(event_type):
                X_total[ne] = data[labels == et][:n_trials, ...]  # (Nt,Nc,Np)
        return X_total
    elif target_style == 'sklearn':
        n_trials = data.shape[-3]  # Nt
        n_events = data.shape[-4]  # Ne

        # create labels
        y_total = []
        for ne in range(n_events):
            y_total += [ne for ntr in range(n_trials)]

        # reshape data
        if filter_bank:  # (Nb,Ne,Nt,Nc,Np)
            n_bands = data.shape[0]  # Nb
            X_total = np.zeros((n_bands, n_events * n_trials, n_chans, n_points))
            for nb in range(n_bands):
                for ne in range(n_events):
                    sp = ne * n_trials
                    X_total[nb, sp:sp + n_trials, ...] = data[nb, ne, ...]  # (Nt,Nc,Np)
        else:  # (Ne,Nt,Nc,Np)
            X_total = np.zeros((n_events * n_trials, n_chans, n_points))
            for ne in range(n_events):
                sp = ne * n_trials
                X_total[sp:sp + n_trials, ...] = data[ne, ...]  # (Nt,Nc,Np)
        return X_total, np.array(y_total).squeeze()


def generate_mean(X, y):
    """
    Calculate trial-averaged templates.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Input data.
    y : *ndarray of shape (Ne*Nt,).*
        Labels for ``X``.

    Returns
    ----------
    X_mean : *ndarray of shape (Ne,Nc,Np).*
        Trial-averaged ``X``.
    """
    # basic information
    event_type = np.unique(y)
    n_events = len(event_type)
    n_chans = X.shape[-2]
    n_points = X.shape[-1]

    # main process
    X_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        X_mean[ne] = np.mean(X[y == et], axis=0)
    return X_mean


def generate_var(X, y=None, unbias=False, shrinkage=0):
    """
    Calculate variance matrices.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Input data.
    y : *ndarray of shape (Ne*Nt,).*
        Labels for X. If None, ignore category information.
    unbias : *bool.*
        Unbias estimation. Defaults to False.
        When ``'True'``, the result may fluctuate by 0.05%.
    shrinkage : *float.*
        Strength for shrinkage covariance estimation. Defaults to 0 (no shrinkage).
        0.01 recommended while *Nt* is small (e.g. 2 for each class).

    Returns
    ----------
    X_var : *ndarray of shape (Ne,Nc,Nc) or (Nc,Nc).*
        Variance matrices of ``X``.
    """
    # basic information
    n_chans = X.shape[-2]  # Nc
    n_points = X.shape[-1]  # Np
    estimator = Shrinkage(shrinkage=shrinkage)

    # main process
    if y is not None:  # calculate the covariance matrices separately for each class
        event_type = np.unique(y)
        n_events = len(event_type)  # Ne
        n_train = np.array([np.sum(y == et) for et in event_type])  # Nt

        X_var = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
        for ne, et in enumerate(event_type):
            n_trials = n_train[ne]  # Nt
            X_temp = X[y == et]  # (Nt,Nc,Np)
            Var_temp = np.tile(A=np.eye(n_chans), reps=(n_trials, 1, 1))  # (Nt,Nc,Nc)
            for ntr in range(n_trials):  # faster than einsum
                Var_temp[ntr] = X_temp[ntr] @ X_temp[ntr].T
            Var_temp = estimator.transform(Var_temp)  # shrink the SPD matrices
            if not unbias:
                X_var[ne] = Var_temp.mean(axis=0)
            else:  # unbias estimation
                X_var[ne] = Var_temp.sum(axis=0) / (n_trials * n_points - 1)
    else:  # calculate the covariance matrix for all samples
        total_trials = X.shape[0]  # Ne*Nt
        X_var = np.zeros((total_trials, n_chans, n_chans))  # (Nc,Nc)
        for ntt in range(total_trials):  # faster than einsum
            X_var[ntt] += X[ntt] @ X[ntt].T
        X_var = estimator.transform(X_var)  # (Ne*Nt,Nc,Nc)
        if not unbias:
            X_var = X_var.mean(axis=0)
        else:
            X_var = X_var.sum(axis=0) / (total_trials * n_points - 1)
    return X_var


def generate_source_response(X, y, w):
    """
    Calculate source responses on latent subspace.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Sklearn-style dataset. *Nt>=2*.
    y : *ndarray of shape (Ne*Nt,).*
        Labels for ``X``.
    w : *ndarray of shape (Ne,Nk,Nc).*
        Spatial filters.

    Returns
    ----------
    S : *ndarray of shape (Ne*Nt,Nk,Np).*
        Source responses.
    """
    # basic information
    n_trials = X.shape[0]  # Ne*Nt
    n_components = w.shape[-2]  # Nk
    n_points = X.shape[-1]  # Np
    event_type = list(np.unique(y))  # (Ne,)

    # backward-propagation
    S = np.zeros((n_trials, n_components, n_points))
    for nt in range(X.shape[0]):
        event_idx = event_type.index(y[nt])
        S[nt] = w[event_idx] @ X[nt]
    return S


def spatial_filtering(w, X, y=None):
    """
    Generate spatial filtered data.

    Parameters
    ----------
    w : *ndarray of shape (Ne,Nk,Nc) or (Nk,Nc).*
        Spatial filters
    X : *ndarray of shape (Ne,Nc,Np) or (Ne*Nt,Nc,Np).*
        Input dataset.
    y : *ndarray of shape (Ne*Nt,).*
        If ``None``, ``X`` is trial-averaged; Else ``X`` is multi-trial.

    Returns
    ----------
    wX : *ndarray of shape (Ne,Nk,Np) or (Ne*Nt,Nk,Np).*
        Spatial-filtered ``X``.
    """
    # basic information
    n_components = w.shape[-2]  # Nk
    n_points = X.shape[-1]  # Np
    if y is not None:  # multi-trial data
        event_type = list(np.unique(y))
        n_events = len(event_type)  # Ne
    else:
        n_events = X.shape[0]  # Ne

    # check the dimension of filter w
    if w.ndim == 2:  # (Nk,Nc)
        w = np.tile(A=w, reps=(n_events, 1, 1))

    # spatial filtering process
    wX = np.zeros((X.shape[0], n_components, n_points))  # (Ne,Nk,Np) or (Ne*Nt,Nk,Np)
    if y is not None:
        for ntr in range(X.shape[0]):
            idx = event_type.index(y[ntr])
            wX[ntr] = w[idx] @ X[ntr]
    else:
        for ne in range(n_events):
            wX[ne] = w[ne] @ X[ne]
    return wX


# %% 3. feature integration
def sign_sta(x):
    r"""
    Standardization of decision coefficient based on :math:`{\rm sign} \left(x \right)`.

    Parameters
    ----------
    x : *int or float or ndarray.*
        Input data (or value).

    Returns
    ----------
    out : *int or float or ndarray.*
        :math:`{\rm sign} \left(x \right) \times x^2`
    """
    return np.sign(x) * (x**2)


def combine_feature(features, func=sign_sta):
    """
    Coefficient-level integration.

    Parameters
    ----------
    features : *List[Union[int, float, ndarray]].*
        Different features.
    func : *function.*
        Quantization function.

    Returns
    ----------
    coef : *List[Union[int, float, ndarray]].*
        Integrated coefficients.
    """
    coef = np.zeros_like(features[0])
    for feature in features:
        coef += func(feature)
    return coef


# %% 4. algorithm evaluation
def label_align(y_pred, event_type):
    """
    Alignment between predicted **indices** and **real labels**.

    Parameters
    ----------
    y_pred : *ndarray of shape (Nte,).*
        Predict labels.
    event_type : *ndarray of shape (Ne,).*
        Event ID arranged in ascending order.

    Returns
    ----------
    correct_predict : *ndarray of shape (Nte,).*
        Corrected labels.
    """
    correct_predict = np.zeros_like(y_pred)
    for npr, ypr in enumerate(y_pred):
        correct_predict[npr] = event_type[int(ypr)]
    return correct_predict


def acc_compute(y_true, y_pred):
    """
    Compute accuracy.

    Parameters
    ----------
    y_pred : *ndarray of shape (n_test,).*
        Predict labels.
    y_true : *ndarray of shape (n_test,).*
        Real labels for test dataset.

    Returns
    ----------
    out : *float.*
        Accuracy. 0-1.
    """
    return np.sum(y_pred == y_true) / len(y_true)


def itr_compute(num, time, acc):
    """
    Compute information transfer rate.

    Parameters
    ----------
    num : *int.*
        Number of targets.
    time : *int or float.*
        Unit: seconds.
    acc : *float.*
        Accuracy. 0-1.

    Returns
    ----------
    out : *float.*
        ITR value.
    """
    part_a = log(num, 2)
    if int(acc) == 1:  # avoid special situation
        part_b, part_c = 0, 0
    elif float(acc) == 0.0:
        return 0
    else:
        part_b = acc * log(acc, 2)
        part_c = (1 - acc) * log((1 - acc) / (num - 1), 2)
    return 60 / time * (part_a + part_b + part_c)


# %% 5. spatial distances
def pearson_corr(X, Y, parallel=False):
    """
    Pearson correlation coefficient for EEG classification.

    Parameters
    ----------
    X : *ndarray of shape (m,n).*
        Spatial filtered (or multi-dimension) single-trial data *(Nk,Np)*.
    Y : *ndarray of shape (l,m,n) or (m,n).*
        Templates while ``parallel=True`` *(Ne,Nk,Np)* or ``False`` *(Nk,Np)*.
    parallel : *bool.*
        An accelerator. Defaults to False.
            If False, ``X`` could only be compared with ``Y`` of shape *(m,n)*;

            If True, ``X`` could be compared with ``Y`` of shape *(l,m,n)*.

    Returns
    ----------
    out : *float or ndarray.*
            If *float*: ``X`` *(m,n)* & ``Y`` *(m,n)*.

            If *ndarray* of shape *(l,)*: ``X`` *(m,n)* & ``Y`` *(l,m,n)*.
    """
    # X_stan, Y_stan = standardization(X), standardization(Y)
    # reshape data into vector-style: reshape() is 5 times faster than flatten()
    X = np.reshape(X, -1, order='C')  # (m*n,)
    if parallel:  # Y: (l,m,n)
        Y = np.reshape(Y, (Y.shape[0], -1), order='C')  # (l,m*n)
    else:  # Y: (m,n)
        Y = np.reshape(Y, -1, order='C')  # (m*n,)
    X_stan, Y_stan = standardization(X), standardization(Y)
    return Y_stan @ X_stan / X_stan.shape[-1]


@njit(fastmath=True)
def fast_corr_2d(X, Y):
    """
    Use the JIT compiler to make ``pearson_corr()`` run faster for 2-D input.

    NOTE: X.shape[-1] is the reshaped length, not real length:
    e.g. (Ne,Ne * Nk,Np) -reshape-> (Ne,Ne * Nk * Np) -> (Ne,) (return)
    -> 1 / Np * (Ne,) (real corr)

    Parameters
    ----------
    X : *ndarray of shape (d1,d2).*
        Input data.
    Y : *ndarray of shape (d1,d2).*
        Input data.

    Returns
    ----------
    corr : *ndarray of shape (d1,).*
        Equivalent correlation coefficients.
    """
    dim_1 = X.shape[0]
    corr = np.zeros((dim_1))
    for d1 in range(dim_1):
        corr[d1] = X[d1, :] @ Y[d1, :].T
    return corr


@njit(fastmath=True)
def fast_corr_3d(X, Y):
    """
    Use the JIT compiler to make ``pearson_corr()`` run faster for 3-D input.

    Parameters
    ----------
    X : *ndarray of shape (d1,d2,d3).*
        Input data.
    Y : *ndarray of shape (d1,d2,d3).*
        Input data.

    Returns
    ----------
    corr : *ndarray of shape (d1,d2).*
        Equivalent correlation coefficients.
    """
    dim_1, dim_2 = X.shape[0], X.shape[1]
    corr = np.zeros((dim_1, dim_2))
    for d1 in range(dim_1):
        for d2 in range(dim_2):
            corr[d1, d2] = X[d1, d2, :] @ Y[d1, d2, :].T
    return corr


@njit(fastmath=True)
def fast_corr_4d(X, Y):
    """
    Use the JIT compiler to make ``pearson_corr()`` run faster for 4-D input.

    Parameters
    ----------
    X : *ndarray of shape (d1,d2,d3,d4).*
        Input data.
    Y : *ndarray of shape (d1,d2,d3,d4).*
        Input data.

    Returns
    ----------
    corr : *ndarray of shape (d1,d2,d3).*
        Equivalent correlation coefficients.
    """
    dim_1, dim_2, dim_3 = X.shape[0], X.shape[1], X.shape[2]
    corr = np.zeros((dim_1, dim_2, dim_3))
    for d1 in range(dim_1):
        for d2 in range(dim_2):
            for d3 in range(dim_3):
                corr[d1, d2, d3] = X[d1, d2, d3, :] @ Y[d1, d2, d3, :].T
    return corr


def fisher_score(X, y):
    """
    Fisher Score (sequence).

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Np).*
        Single-channel input data.
    y : *ndarray of shape (Ne*Nt,).*
        Labels for ``X``.

    Returns
    ----------
    fs : *ndarray of shape (Np,).*
        Fisher-Score sequence.
    """
    # data information
    event_type = np.unique(y)
    n_events = len(event_type)  # Ne
    n_train = np.array([np.sum(y == et) for et in event_type])
    n_features = X.shape[-1]

    # class center & total center
    class_center = np.zeros((n_events, n_features))
    for ne, et in enumerate(event_type):
        class_center[ne] = np.mean(X[y == et], axis=0)
    total_center = class_center.mean(axis=0, keepdims=True)

    # inter-class divergence
    decenter = class_center - total_center
    ite_d = n_train @ (decenter**2)

    # intra-class divergence
    itr_d = np.zeros((n_features))
    for ne, et in enumerate(event_type):
        temp = X[y == et] - class_center[ne]
        for ntr in range(n_train[ne]):
            itr_d += temp[ntr] @ temp[ntr].T

    # fisher-score
    return ite_d / itr_d


def snr_time(X, y=None, output='array'):
    """
    Signal-to-Noise ratio (sequence) in time domain.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Np).*
        Single-channel input data.
    y : *ndarray of shape (Ne*Nt,).*
        Labels for ``X``. If ``None``, ``X`` is single-category data.
    output : *str.*
        ``'array'`` or ``'db'``. Defaults to ``'array'``.
            If ``'array'``, return array of shape *(Ne,Np)*;

            If ``'db'``, return array of shape *(Ne,)*.

    Returns
    ----------
    snr : *ndarray, (Ne,Np) or (Ne,)*
        SNR sequences (or values).
    """
    # basic information
    if y is None:  # single-category X
        y = np.ones((X.shape[0]))
    event_type = np.unique(y)
    n_events = len(event_type)  # Ne
    n_points = X.shape[-1]  # Np

    # compute SNR in time domain
    signal_power = np.zeros((n_events, n_points))  # (Ne,Np)
    noise_power = np.zeros_like(signal_power)
    for ne, et in enumerate(event_type):
        signal = np.mean(X[y == et], axis=0, keepdims=True)  # (1,Np)
        signal_power[ne] = signal ** 2  # (1,Np)
        noise = X[y == et] - signal  # (Nt,Np)
        noise_power[ne] = np.mean(noise**2, axis=0, keepdims=True)  # (1,Np)
    if output == 'array':
        return signal_power / noise_power  # (Ne,Np)
    elif output == 'db':
        return 10 * np.log10(np.mean(signal_power / noise_power, axis=-1))


def euclidean_dist(X, Y):
    r"""
    Euclidean distance.

    :math:`D \left(\pmb{X}, \pmb{Y} \right) = \sum_{i} \sum_{j} {\left(\pmb{X}(i,j)
    - \pmb{Y}(i,j) \right)}^2`

    Parameters
    ----------
    X : *ndarray of shape (m, n).*
        Input data.
    Y : *ndarray of shape (m, n).*
        Input data.

    Returns
    ----------
    out : float.
        Results
    """
    return np.sqrt(np.sum((X - Y)**2))


def cosine_sim(x, y):
    r"""
    Cosine similarity.

    :math:`D \left(\pmb{x}, \pmb{y} \right)  = \dfrac{\pmb{x} \pmb{y}^T}
    {\left\|\pmb{x} \right\| \times \left\|\pmb{y} \right\|}`

    Parameters
    ----------
    x : *ndarray of shape (Np,).*
        Input data.
    y : *ndarray of shape (Np,).*
        Input data.

    Returns
    ----------
    out : *float.*
        Results.
    """
    return (x @ y) / np.sqrt((x @ x) * (y @ y))


def minkowski_dist(x, y, p):
    r"""
    Minkowski distance.

    :math:`D \left(\pmb{x}, \pmb{y} \right) = {\left( \sum_{i=1}^{n}
    {\left|x_i - y_i \right|}^p \right)}^{\frac{1}{p}}`

    Parameters
    ----------
    x : *ndarray of shape (Np,).*
        Input data.
    y : *ndarray of shape (Np,).*
        Input data.
    p : *int or float.*
        Hyper-parameter.

    Returns
    ----------
    out : *float.*
        Results.
    """
    return np.sum(abs(x - y)**p)**(1 / p)


def s_estimator(X):
    """
    Construct s-estimator.

    Parameters
    ----------
    X : *ndarray of shape (m,m).*
        Symmetric input matrix.

    Returns
    ----------
    s_estimator : *float.*
        Results
    """
    e_val, _ = sLA.eig(X)
    e_val = e_val / np.sum(e_val)  # normalized eigenvalues
    entropy = np.sum([i * log(i, 2) for i in e_val])
    return 1 + entropy / log(X.shape[0], 2)


# %% 6. temporally smoothing functions
def tukeys_kernel(x, r=3):
    r"""
    Tukeys tri-cube kernel function.

    :math:`f(x) = {\left( 1 - |x|^{r} \right)}^{r}`

    Parameters
    ----------
    x : *float.*
        Input value.
    r : *int or float.*
        Hyper-parameter. Defaults to 3.

    Returns
    ----------
    out : *float.*
        Results.
    """
    if abs(x) > 1:
        return 0
    else:
        return (1 - abs(x)**r)**r


def weight_matrix(n_points, tau, r=3):
    """
    Weighting matrix based on Tukeys kernel function.

    Parameters
    ----------
    n_points : *int.*
        Parameters that determine the size of the matrix.
    tau : *int or float.*
        Hyper-parameter for weighting matrix.
    r : *int or float.*
        Hyper-parameter for kernel funtion. Defaults to 3.

    Returns
    ----------
    W : *ndarray of shape (Np,Np).*
        Weighting matrix.
    """
    W = np.eye(n_points)
    for i in range(n_points):
        for j in range(n_points):
            W[i, j] = tukeys_kernel(x=(j - i) / tau, r=r)
    return W


def laplacian_matrix(W):
    """
    Laplace matrix for time smoothing.

    Parameters
    ----------
    W : *ndarray of shape (Np,Np).*
        Weighting matrix.

    Returns
    ----------
    out : *ndarray of shape (Np,Np).*
        Laplace matrix.
    """
    return np.diag(np.sum(W, axis=-1)) - W


# %% 7. reduced QR decomposition
def qr_projection(X):
    """
    Orthogonal projection based on QR decomposition.

    Parameters
    ----------
    X : *ndarray of shape (Np,m).*
        Input data.

    Returns
    ----------
    P : *ndarray of shape (Np,Np).*
        Projection matrix.
    """
    Q, _ = sLA.qr(X, mode='economic')
    return Q @ Q.T


# %% 8. Eigenvalue problems
def pick_subspace(e_vals, ratio, min_n=None, max_n=None):
    """
    Optimize the number of subspaces.

    Parameters
    ----------
    e_vals : *array-like of shape (Nk,).*
        Sequence of eigenvalues sorted in descending order.
    ratio : *float.*
        0-1. The ratio of the sum of picked eigenvalues to the total.
    min_n: *int.*
        Minimum number of the dimension of subspace.
    max_n: *int.*
        Maximum number of the dimension of subspace.

    Returns
    ----------
    n_components : *int.*
        The optimized number of subspaces.
    """
    # basic information
    e_vals_sum = np.sum(e_vals)
    threshould = ratio * e_vals_sum

    # check non-compliant input parameters
    if (min_n is None) or (min_n < 1):
        min_n = 1
    if (max_n is None) or (max_n > len(e_vals)):
        max_n = len(e_vals)

    # main process
    temp_sum = 0
    for nev, e_val in enumerate(e_vals):
        temp_sum += e_val
        n_components = nev + 1
        if temp_sum >= threshould:
            if n_components < min_n:
                return min_n
            elif n_components >= max_n:
                return max_n
            else:
                return n_components


def normalize_eigenvectors(e_vec):
    """
    Adjust the sign of the eigenvector by selecting the element with the largest
    absolute value and ensuring it is positive.

    Parameters
    ----------
    e_vec : *ndarray of shape (m,m).*
        Column vectors. i.e. ``e_vec[:, i]`` is an eigenvector.

    Returns
    ----------
    e_vec_norm : *ndarray of shape (m,m).*
        Normalized ``e_vec``.
    """
    e_vec_norm = np.zeros_like(e_vec)
    for i in range(e_vec.shape[1]):
        if np.argmax(np.abs(e_vec[:, i])) < 0:
            e_vec_norm[:, i] = e_vec[:, i] * -1
    return e_vec_norm


def solve_gep(
        A,
        B=None,
        n_components=1,
        mode='Max',
        ratio=None,
        min_n=None,
        max_n=None):
    r"""
    Solve generalized eigenvalue problems (GEPs) based on Rayleigh quotient:

    :math:`f(\pmb{w}) = \dfrac{\pmb{w} \pmb{A} {\pmb{w}}^T} {\pmb{w} \pmb{B} {\pmb{w}}^T}
    \Longrightarrow  \pmb{A} \pmb{w} = \pmb{\lambda} \pmb{Bw}.`

    If ``B`` is ``None``, solve eigenvalue problems (EPs):

    :math:`f(\pmb{w}) = \dfrac{\pmb{w} \pmb{A} {\pmb{w}}^T}{\pmb{w} {\pmb{w}}^T}
    \Longrightarrow \pmb{A} \pmb{w} = \pmb{\lambda} \pmb{w}.`

    Parameters
    ----------
    A : *ndarray of shape (m,m).*
        Input matrix.
    B : *ndarray of shape (m,m), optional.*
        Input matrix.
    n_components : *int.*
        Number of eigenvectors picked as filters.
    mode : *str.*
        ``'Max'`` or ``'Min'``. Depends on target function.
    ratio : *float.*
        0-1. The ratio of the sum of useful (``mode='Max'``)
        or deprecated (``mode='Min'``) eigenvalues to the total.
        Only useful when ``n_components=None``.
    min_n: *int.*
        Minimum number of the dimension of subspace.
    max_n: *int.*
        Maximum number of the dimension of subspace.

    Returns
    ----------
    w : *ndarray of shape (Nk,m).*
        Picked eigenvectors.
    """
    # solve EPs
    if B is not None:  # f(w) = (w @ A @ w^T) / (w @ B @ w^T)
        # faster than sLA.eig(a=A, b=B)
        e_val, e_vec = sLA.eig(sLA.solve(a=B, b=A, assume_a='sym'))  # ax=b -> x=a^{-1}b
    else:  # f(w) = (w @ A @ w^T) / (w @ w^T)
        e_val, e_vec = sLA.eig(A)

    # pick the optimal subspaces
    w_index = np.flip(np.argsort(e_val))
    if n_components is None:
        n_components = pick_subspace(
            e_vals=e_val[w_index],
            ratio=ratio,
            min_n=min_n,
            max_n=max_n
        )
    if mode == 'Min':
        return np.real(e_vec[:, w_index][:, n_components:].T)
    elif mode == 'Max':
        return np.real(e_vec[:, w_index][:, :n_components].T)


# %% 9. Signal generation
def Imn(m, n):
    """
    Concatenate identical matrices into a big matrix.

    Parameters
    ----------
    m : *int.*
        Total number of identity matrix.
    n : *int.*
        Dimensions of the identity matrix.

    Returns
    ----------
    out : *ndarray of shape (m*n, n).*
        Concatenated matrix.
    """
    Z = np.zeros((m * n, n))
    for i in range(m):
        Z[i * n:(i + 1) * n, :] = np.eye(n)
    return Z


def square_wave(freq, phase, n_points, duty=20, srate=1000):
    """
    Construct square waveforms.

    Adopted from: https://github.com/edwin465/SSVEP-Impulse-Response.

    Parameters
    ----------
    freq : *int or float.*
        Frequency (Hz).
    phase : *float.*
        Coefficients. 0-2 (pi).
    n_points : *int.*
        Number of sampling points.
    duty : *int or float.*
        Defaults to ``50`` (%).
    srate : *int or float.*
        Sampling rate (Hz). Defaults to ``1000``.

    Returns
    ----------
    out : *ndarray of shape (Np,).*
        Square sequence (amplitude ranging from 0 to 2).
    """
    time_points = np.arange(n_points) / srate

    # the black box from @Edwin465
    t = 2 * pi * freq * time_points + phase * pi
    tem = np.mod(t, 2 * pi)
    w0 = 2 * pi * duty / 100
    nodd = np.array((tem < w0))
    return 2 * nodd - 1


def sin_wave(freq, phase, n_points, srate=1000):
    """
    Construct sinusoidal waveforms.

    Parameters
    ----------
    freq : *int or float.*
        Frequency (Hz).
    phase : *float.*
        Coefficients. 0-2 (pi).
    n_points : *int.*
        Number of sampling points.
    srate : *int or float.*
        Sampling rate (Hz). Defaults to ``1000``.

    Returns
    ----------
    out : *ndarray of shape (Np,).*
        Sinusoidal sequence (amplitude ranging from -1 to 1).
    """
    time_points = np.arange(n_points) / srate
    return np.sin(2 * pi * freq * time_points + pi * phase)


def sine_template(freq, phase, n_points, n_harmonics, srate=1000):
    """
    Create sine-cosine template for SSVEP signals.

    Parameters
    ----------
    freq : *int or float.*
        Frequency (Hz).
    phase : *float.*
        Coefficients. 0-2 (pi).
    n_points : *int.*
        Number of sampling points.
    n_harmonics : *int.*
        Number of harmonics.
    srate : *int or float.*
        Sampling rate (Hz). Defaults to ``1000``.

    Returns
    ----------
    Y : *ndarray of shape (2*Nh,Np).*
        Sine-cosine templates.
    """
    Y = np.zeros((2 * n_harmonics, n_points))  # (2Nh,Np)
    for nh in range(n_harmonics):
        Y[2 * nh, :] = sin_wave(
            freq=(nh + 1) * freq,
            phase=0 + (nh + 1) * phase,
            n_points=n_points,
            srate=srate
        )
        Y[2 * nh + 1, :] = sin_wave(
            freq=(nh + 1) * freq,
            phase=0.5 + (nh + 1) * phase,
            n_points=n_points,
            srate=srate
        )
    return Y


def get_resample_sequence(seq, srate=1000, rrate=60, output_waveform=True):
    """
    Obtain the resampled sequence from original sequence.

    Parameters
    ----------
    seq : *ndarray of shape (1, signal_length).*
        Stimulus sequence of original sampling rate.
    srate : *int or float.*
        Sampling rate. Defaults to ``1000`` (Hz).
    rrate : *int.*
        Refresh rate of stimulus devices. Defaults to ``60`` (Hz).
    output_waveform : *bool.*
        Return resampled waveform or not. Defaults to ``False``.

    Returns
    ----------
    rsp_seq: *List[Tuple[int, float]].*
        *(index, value)*. Resampled values and indices of stimulus sequence.
    rsp_wave : *ndarray of shape (1, resampled_length).*
        Only exist when ``output_waveform=True``.
    """
    # construct resampled sequence
    n_points = seq.shape[-1]  # Np
    rsp_points = int(np.round(rrate * n_points / srate))  # resampled n_points
    rsp_idx = np.round(srate / rrate * np.arange(rsp_points) + 0.001)
    rsp_val = [seq[int(ri)] for ri in rsp_idx]
    rsp_seq = [(int(ri), rv) for ri, rv in zip(rsp_idx, rsp_val)]

    # construct waveform (for plotting)
    if output_waveform:
        rsp_wave = np.zeros((n_points))
        n_count = 0
        for nrw in range(rsp_wave.shape[0]):
            if nrw == 0:  # start point
                rsp_wave[nrw] = rsp_seq[n_count][1]
                n_count += 1
            elif rsp_seq[n_count - 1][0] < nrw < rsp_seq[n_count][0]:
                # keep the same value in the current period
                rsp_wave[nrw] = rsp_seq[n_count - 1][1]
            elif nrw >= rsp_seq[n_count][0]:  # switch to the next period
                if n_count == len(rsp_seq) - 1:  # max
                    rsp_wave[nrw] = rsp_seq[n_count][1]
                else:
                    n_count += 1
                    rsp_wave[nrw] = rsp_seq[n_count - 1][1]
        return rsp_seq, rsp_wave
    else:
        return rsp_seq


def extract_periodic_impulse(
        freq,
        phase,
        n_points,
        srate=1000,
        rrate=60,
        method='Square'):
    """
    Extract periodic impulse sequence from stimulus sequence.

    Parameters
    ---------
    freq : *int or float.*
        Stimulus frequency (Hz).
    phase : *int or float.*
        Stimulus phase (coefficients). 0-2 (pi).
    n_points : *int.*
        Total length of reconstructed signal.
    srate : *int or float.*
        Sampling rate. Defaults to ``1000`` (Hz).
    rrate : *int.*
        Refresh rate of stimulus devices. Defaults to ``60`` (Hz).
    method : *str.*
        ``'Square'`` or ``'Cosine'``. Defaults to ``'Square'``.
            If ``'Square'``, pick up the moment of the rising edge of the square wave.
            Adopted from: https://github.com/edwin465/SSVEP-Impulse-Response.

            If ``'Cosine'``, pick up the peak moment of the resampled cosine wave.

    Returns
    ---------
    out : *ndarray. shape (1,Np).*
        Periodic impulse sequence.
    """
    # obtain the actual stimulus strength
    extra_length = int(np.ceil(1 / freq * srate) + n_points)  # add a single period
    cos_seq = sin_wave(
        freq=freq,
        n_points=extra_length,
        phase=0.5 + phase,
        srate=srate
    )
    rsp_seq, rsp_wave = get_resample_sequence(
        seq=cos_seq,
        srate=srate,
        rrate=rrate,
        output_waveform=True
    )

    if method == 'Square':  # black box from @Edwin456
        periodic_impulse = square_wave(
            freq=freq,
            phase=phase,
            n_points=extra_length,
            duty=10,
            srate=srate
        ).astype(float) + 1  # square wave, (Np,)
        count_thred = np.floor(0.9 * srate / freq)
        count = count_thred + 1
        for el in range(extra_length):
            if periodic_impulse[el] == 0:
                count = count_thred + 1
            else:
                if count >= count_thred:
                    periodic_impulse[el] = rsp_wave[el] + 1
                    count = 1
                else:
                    count += 1
                    periodic_impulse[el] = 0
    elif method == 'Cosine':  # pick up the peak values of resampled sequence
        periodic_impulse = np.zeros_like(cos_seq)  # WARNING: not the real length
        if np.mod(phase, 2) == 0:
            periodic_impulse[0] = rsp_seq[0][1]
        for rs in range(1, len(rsp_seq) - 1):
            # >> the last point
            # bigger_than_left = rsp_seq[rs][1] - rsp_seq[rs - 1][1] > 10e-5
            # left_edge = bigger_than_left
            left_edge = rsp_seq[rs][1] - rsp_seq[rs - 1][1] > 10e-5

            # >= the next point
            bigger_than_right = rsp_seq[rs][1] > rsp_seq[rs + 1][1]
            close_to_right = abs(rsp_seq[rs][1] - rsp_seq[rs + 1][1]) < 10e-5
            right_edge = bigger_than_right or close_to_right
            if left_edge and right_edge:
                periodic_impulse[rsp_seq[rs][0]] = rsp_seq[rs][1]
    return periodic_impulse[:n_points]


def create_conv_matrix(periodic_impulse, response_length):
    """
    Create the convolution matrix of the periodic impulse.

    Parameters
    ----------
    periodic_impulse : *ndarray of shape (1,Np).*
        Impulse sequence of stimulus.
    response_length : *int.*
        Length of impulse response.

    Returns
    ----------
    out : *ndarray of shape (Nrl,Np).*
        Convolution matrix.
    """
    signal_length = periodic_impulse.shape[-1]
    H = np.zeros((response_length, response_length + signal_length - 1))
    for rl in range(response_length):
        H[rl, rl:rl + signal_length] = periodic_impulse
    return H[:, :signal_length]


def correct_conv_matrix(H, freq, srate, amp_scale=0.8, concat_method='dynamic'):
    """
    Replace the blank values at the front of the reconstructed data
    with its subsequent fragment.

    Parameters
    ----------
    H : *ndarray of shape (Nrl,Np).*
        Convolution matrix.
    freq : *int or float.*
        Stimulus frequency (Hz).
    srate : *int or float.*
        Sampling rate. Defaults to ``1000`` (Hz).
    amp_scale : *float.*
        The multiplying power when calculating the amplitudes of data.
        Defaults to ``0.8``.
    concat_method : *str.*
        ``'dynamic'`` or ``'static'``. Defaults to ``'dynamic'``.
            'static': Concatenated data is starting from 1 s.

            'dynamic': Concatenated data is starting from 1 period.

    Returns
    ----------
    out : *ndarray of shape (Nrl,Np).*
        Corrected convolution matrix.
    """
    shift_length = np.where(H[0] != 0)[0][0] + 1  # Tuple -> ndarray -> int
    shift_matrix = np.eye(H.shape[-1])
    if concat_method == 'static':
        sp = srate
    elif concat_method == 'dynamic':
        sp = int(np.ceil(srate / freq))
    try:
        ep = sp + shift_length
        shift_matrix[sp:ep, :shift_length] = amp_scale * np.eye(shift_length)
    except ValueError:
        raise Exception('Signal length is too short!')
    return H @ shift_matrix


def resize_conv_matrix(H, new_size, method='Lanczos'):
    """
    Resize the convolution matrix (``H``).

    Parameters
    ----------
    H : *ndarray of shape (Nrl,Np).*
        Convolution matrix. Nrl is the length of response.
    new_size : *Tuple[int, int].*
        New size of ``H``. *(length, width)*.

        **NOTE**: for a matrix with shape *(m,n)*, ``length=n``, ``width=m``.
    method : *str.*
        ``'nearest'``, ``'linear'``, ``'cubic'``, ``'area'``, ``'Lanczos'``,
        ``'linear-exact'``, ``'inverse-map'`` and ``'fill-outliers'``.
        Interpolation methods. Defaults to ``'Lanczos'``.

    Returns
    ----------
    H_new : *ndarray of shape (Nrl,Np).*
        Resized convolution matrix.
    """
    mapping = {
        'nearest': cv2.INTER_NEAREST,
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'Lanczos': cv2.INTER_LANCZOS4,
        'linear-exact': cv2.INTER_LINEAR_EXACT,
        'inverse-map': cv2.WARP_INVERSE_MAP,
        'fill-outliers': cv2.WARP_FILL_OUTLIERS
    }
    return cv2.resize(
        src=H,
        dsize=new_size,
        interpolation=mapping[method]
    )


def solve_mcpe(X, freq, srate=1000, steps=1000, average=False):
    """
    Calculate the phase for SSVEP signal by Maximum Correlation Phase Estimation (MCPE).

    Parameters
    ----------
    X : *ndarray of shape (Nt,Np).*
        Single-channel input data of one class.
    freq : *float.*
        Target (stimulus) frequency.
    srate : *int or float.*
        Sampling rate (Hz). Defaults to ``1000``.
    average : *bool.*
        Whether return the phase of ``X_mean``. Defaults to ``False``.
        i.e. return multiple-trial phases as an array.

    Returns
    ----------
    phases : *ndarray of shape (Nt,) or (1,).*
        Estimated phases (radians).
    """
    # initialization
    optional_phases = np.linspace(0, 2, steps + 1)[:steps]
    if average:
        X_ori = X.mean(axis=0, keepdims=True)  # (1,Np)
    else:
        X_ori = deepcopy(X)

    # traversal
    phases = []
    for nt in range(X_ori.shape[0]):
        rho = np.zeros_like(optional_phases)
        for nop, op in enumerate(optional_phases):
            sin_seq = sin_wave(
                freq=freq,
                phase=op,
                n_points=X.shape[-1],
                srate=srate
            )
            rho[nop] = np.corrcoef(X_ori[nt], sin_seq)[0, 1]
        phases.append(np.pi * optional_phases[np.argmax(rho)])
    return np.array(phases)


# %% 10. Filter-bank technology
def generate_filter_bank(
        passbands,
        stopbands,
        srate,
        order,
        gpass=3,
        gstop=40,
        rp=0.5):
    """
    Generate Chebyshev type I filters according to input paramters.

    Parameters
    ----------
    passbands : *List[Tuple[float,float]].*
        Passband edge frequencies.
    stopbands : *List[Tuple[float,float]].*
        Stopband edge frequencies.
    srate : *float.*
        Sampling rate.
    order : *int, optional.*
        Order of filters.
    gpass : *float.*
        The maximum loss in the passband (dB).
    gstop : *float.*
        The minimum attenuation in the stopband (dB).
    rp : *float.*
        Ripple of filters. Default to 0.5 dB.

    Returns
    ----------
    filter_bank : *List[ndarray].*
        Second-order sections representation of the IIR filters.
    """
    filter_bank = []
    for wp, ws in zip(passbands, stopbands):
        if order is None:  # apply default order selection
            N, wn = cheb1ord(wp, ws, gpass, gstop, fs=srate)  # order, 3dB freq
            sos = cheby1(N, rp, wn, btype='bandpass', output='sos', fs=srate)
        else:
            sos = cheby1(order, rp, wp, btype='bandpass', output='sos', fs=srate)
        filter_bank.append(sos)
    return filter_bank


class FilterBank(BaseEstimator, TransformerMixin):
    """
    Basic filter-bank object.

    Parameters
    ----------
    base_estimator : ``BaseEstimator``.
        Sub-models.
    filter_bank : *List[ndarray], optional.*
        See details in ``generate_filter_bank()``. Defaults to ``None``.
    with_filter_bank : *bool.*
        Whether the input data has been FB-preprocessed. Defaults to ``True``.
    version : *str, optional.*
        If ``'SSVEP'``, bank_weights will be set according to refrence:
        (DOI) 10.1088/1741-2560/12/4/046008.

    Attributes
    ----------
    Nb : *int.*
        Number of filter banks.
    bank_weights : *ndarray, shape (Nb,).*
        Weights for different filter banks. Defaults to ``None``.
    sub_estimator : *List[BaseEstimator].*
        ``BaseEstimator`` objects for each filter bank.
    sub_features : *List[ndarray] or List[Tuple[ndarray, ndarray]].*
        Features of each filter bank.

    Raises
    ----------
    Assertion
        If ``filter_bank`` is not ``None``, ``with_filter_bank`` must not be ``True``.
    """
    def __init__(
            self,
            base_estimator,
            filter_bank=None,
            with_filter_bank=True,
            version=None):
        """Basic configuration."""
        self.base_estimator = base_estimator
        assert bool(filter_bank) != with_filter_bank, 'Check filter bank configuration!'
        self.filter_bank = filter_bank
        self.with_filter_bank = with_filter_bank
        self.version = version

    def fit(
            self,
            X_train,
            y_train,
            bank_weights=None,
            **kwargs):
        """
        Load in training dataset and pass it to sub-esimators.

        Parameters
        ----------
        X_train : *ndarray of shape (Ne*Nt,...,Np) or (Nb,Ne*Nt,...,Np).*
            Sklearn-style training dataset.
        y_train : *ndarray of shape (Ne*Nt,).*
            Labels for ``X_train``.
        bank_weights : *ndarray of shape (Nb,), optional.*
            Weights for different filter banks. Defaults to ``None``.
        """
        if self.with_filter_bank:  # X_train has been filterd: (Nb,Ne*Nt,...,Np)
            self.Nb = X_train.shape[0]
        else:  # tranform X_train into shape: (Nb,Ne*Nt,...,Np)
            self.Nb = len(self.filter_bank)
            X_train = self.fb_transform(X_train)

        self.bank_weights = bank_weights
        if self.version == 'SSVEP':
            self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25 for nb in range(self.Nb)])

        self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]
        for nse, se in enumerate(self.sub_estimator):
            se.fit(
                X_train=X_train[nse],
                y_train=y_train,
                **kwargs
            )

    def transform(
            self,
            X_test,
            **kwargs):
        """
        Transform test dataset to multi-band discriminant features.

        Parameters
        ----------
        X_test : *ndarray of shape (Ne*Nte,...,Np) or (Nb,Ne*Nt,...,Np).*
            Sklearn-style test dataset.

        Returns
        ----------
        fb_rho : *ndarray of shape (Nb,Ne*Nte,Ne).*
            Multi-band decision coefficients.
        rho : *ndarray of shape (Ne*Nte,Ne).*
            Intergrated decision coefficients.
        """
        if not self.with_filter_bank:
            X_test = self.fb_transform(X_test)
        self.sub_features = [se.transform(X_test[nse], **kwargs)
                             for nse, se in enumerate(self.sub_estimator)]
        rho_fb = np.stack([sf['rho'] for sf in self.sub_features], axis=0)
        if self.bank_weights is None:
            rho = np.mean(sign_sta(rho_fb))
        else:
            rho = np.einsum('b,bte->te', self.bank_weights, sign_sta(rho_fb))
        features = {
            'rho_fb': rho_fb, 'rho': rho
        }
        return features

    def fb_transform(self, X_train):
        """
        Transform single-band ``X_train`` into multi-band ``X_train``.

        Parameters
        ----------
        X_train : *ndarray of shape (Ne*Nt,...,Np).*
            Sklearn-style training dataset.

        Returns
        ----------
        X_fb : *ndarray of shape (Nb,Ne*Nt,...,Np).*
            Multi-band ``X_train``.
        """
        return np.stack([sosfiltfilt(sos, X_train, axis=-1)
                         for sos in self.filter_bank])


# %% 11. Linear transformation
def solve_forward_propagation(X, y, w):
    r"""
    Calculate propagation matrice based on source responses.

    :math:`\hat{\pmb{A}} = \arg\min \left\| \pmb{A} \pmb{w} \pmb{X} - \pmb{X} \right\|`.

    Parameters
    ----------
    X : *ndarray of shape (Ne*Nt,Nc,Np).*
        Multi-trial dataset. *Nt>=2*.
    y : *ndarray of shape (Ne*Nt,).*
        Labels for ``X``.
    w : *ndarray of shape (Ne,Nk,Nc).*
        Spatial filters.

    Returns
    ----------
    A : *ndarray of shape (Ne,Nc,Nk).*
        Propagation matrices.
    s : *ndarray of shape (Ne*Nt,Nk,Np).*
        Spatial filtered data.
    """
    # basic information
    event_type = list(np.unique(y))
    n_events, n_components, n_chans = w.shape  # Ne,Nk,Nc
    n_trials, _, n_points = X.shape  # Ne*Nt,Nc,Np

    # construct task-related signal wX (s)
    s = np.zeros((n_trials, n_components, n_points))  # (Ne*Nt,Nk,Np)
    for ntr in range(n_trials):
        event_idx = event_type.index(y[ntr])
        s[ntr] = w[event_idx] @ X[ntr]

    # analytical solution
    X_var = generate_var(X=X, y=y)  # (Ne,Nc,Nc)
    s_var = generate_var(X=s, y=y)  # (Ne,Nk.Nk)
    A = np.zeros((n_events, n_chans, n_components))
    for ne in range(n_events):
        A[ne] = X_var[ne] @ w[ne].T @ sLA.inv(s_var[ne])
    return A, s


def solve_coral(Cs, Ct):
    r"""
    Solve CORAL problem

    :math:`\hat{\pmb{Q}} = \min \left\| {\pmb{Q}}^T \pmb{C}_s \pmb{Q}
    - \pmb{C}_t \right\|_F^2`.

    :math:`\hat{\pmb{Q}} = {\pmb{C}_s}^(-\frac{1}{2}) @ {\pmb{C}_t}^{\frac{1}{2}}`.

    Parameters
    ----------
    Cs : *ndarray of shape (n,n).*
        Second-order statistics of source dataset.
    Ct : *ndarray of shape (n,n).*
        Second-order statistics of target dataset.

    Returns
    ----------
    Q : *ndarray of shape (n,n).*
        Linear transformation matrix.
    """
    return np.real(invsqrtm(Cs) @ sqrtm(Ct))
