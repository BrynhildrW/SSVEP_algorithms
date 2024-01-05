# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

1. Data preprocessing:
    (1-1) centralization()
    (1-2) normalization()
    (1-3) standardization()
<<<<<<< HEAD
    (1-4) fast_stan_2d()
    (1-5) fast_stan_3d()
    (1-6) fast_stan_4d()
    (1-7) fast_stan_5d()
=======
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

2. Data preparation
    (2-1) sin_wave()
    (2-2) sine_template()
    (2-3) Imn()
    (2-4) augmented_events()
    (2-5) selected_events()
    (2-6) reshape_dataset()

<<<<<<< HEAD
3. Feature integration
    (3-1) sign_sta()
    (3-2) combine_feature()

4. Algorithm evaluation
=======
3. feature integration
    (3-1) sign_sta()
    (3-2) combine_feature()

4. algorithm evaluation
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    (4-1) acc_compute()
    (4-2) confusion_matrix()
    (4-3) itr_compute()

<<<<<<< HEAD
5. Spatial distances
    (5-1) pearson_corr()
    (5-2) fast_corr_2d()
    (5-3) fast_corr_3d()
    (5-4) fast_corr_4d()
    (5-5) fisher_score()
    (5-6) euclidean_dist()
    (5-7) cosine_sim()
    (5-8) minkowski_dist()
    (5-9) mahalanobis_dist()
    (5-10) nega_root()
    (5-11) s_estimator()

6. Temporally smoothing functions
=======
5. spatial distances
    (5-1) pearson_corr()
    (5-2) fisher_score()
    (5-3) euclidean_dist()
    (5-4) cosine_sim()
    (5-5) minkowski_dist()
    (5-6) mahalanobis_dist()
    (5-7) nega_root()
    (5-8) s_estimator()

6. temporally smoothing functions
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    (6-1) tukeys_kernel()
    (6-2) weight_matrix()
    (6-3) laplacian_matrix

<<<<<<< HEAD
7. Reduced QR decomposition
    (7-1) qr_projection()

8. Eigenvalue problems
=======
7. reduced QR decomposition
    (7-1) qr_projection()

8. eigenvalue problems
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    (8-1) pick_subspace()
    (8-2) solve_ep()
    (8-3) solve_gep()

<<<<<<< HEAD
9. Signal generation
=======
9. signal generation
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    (9-1) get_resample_sequence()
    (9-2) extract_periodic_impulse()
    (9-3) create_conv_matrix()
    (9-4) correct_conv_matrix()

<<<<<<< HEAD
10. Filter bank
    (10-1) generate_filter_bank
    (10-2) (class) FilterBank

=======
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
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
<<<<<<< HEAD
=======

update: 2023/07/06
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

"""

# %% basic moduls
from typing import Optional, List, Tuple, Dict, Union, Callable
from numpy import ndarray
import numpy as np
from numpy import sin, sqrt, einsum

from scipy import linalg as sLA
from scipy.signal import sosfiltfilt, cheby1, cheb1ord
from sklearn.base import BaseEstimator, TransformerMixin, clone
<<<<<<< HEAD
from sklearn.metrics import confusion_matrix
=======
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

from math import pi, log

import warnings
<<<<<<< HEAD
from numba import njit


# %% 1. data preprocessing
def centralization(X: ndarray) -> ndarray:
    """Transform vector x into y, s.t. mean(y) = 0.
=======


# %% 1. data preprocessing
def centralization(
        X: ndarray) -> ndarray:
    """Transform vector x into y
        s.t. mean(y) = 0.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

    Args:
        X (ndarray): (...,Np).

    Returns:
        Y (ndarray): Data after centralization.
    """
    return X - X.mean(axis=-1, keepdims=True)
<<<<<<< HEAD


def normalization(X: ndarray) -> ndarray:
    """Transform vector x into y, s.t. y = (x - min(x)) / (max(x) - min(x)).
        The range of y is [0,1].

    Args:
        X (ndarray): (...,Np).

    Returns:
        Y (ndarray): Data after normalization.
    """
    X_min = np.min(X, axis=-1, keepdims=True)  # (...,1)
    X_max = np.max(X, axis=-1, keepdims=True)  # (...,1)
    return (X - X_min) / (X_max - X_min)


def standardization(X: ndarray) -> ndarray:
    """Transform vector x into y, s.t. var(y) = 1.

    Args:
        X (ndarray): (...,Np).

    Returns:
        Y (ndarray): Data after standardization.
    """
    X = centralization(X)
    return X / np.std(X, axis=-1, keepdims=True)


@njit(fastmath=True)
def fast_stan_2d(X: ndarray) -> ndarray:
    """Special version of standardization() for 2-dimensional X.
    Use the JIT compiler to make python codes run faster.

    Args:
        X (ndarray): (d1,d2).

    Returns:
        X (ndarray): (d1,d2).
    """
    dim_1 = X.shape[0]
    for d1 in range(dim_1):
        X[d1, :] = X[d1, :] - np.mean(X[d1, :])  # centralization
        X[d1, :] = X[d1, :] / np.std(X[d1, :])
    return X


@njit(fastmath=True)
def fast_stan_3d(X: ndarray) -> ndarray:
    """Special version of standardization() for 3-D X.

    Args:
        X (ndarray): (d1,d2,d3).

    Returns:
        X (ndarray): (d1,d2,d3).
    """
    dim_1, dim_2 = X.shape[0], X.shape[1]
    for d1 in range(dim_1):
        for d2 in range(dim_2):
            X[d1, d2, :] = X[d1, d2, :] - np.mean(X[d1, d2, :])
            X[d1, d2, :] = X[d1, d2, :] / np.std(X[d1, d2, :])
    return X


@njit(fastmath=True)
def fast_stan_4d(X: ndarray) -> ndarray:
    """Special version of standardization() for 4-D X.

    Args:
        X (ndarray): (d1,d2,d3,d4).

    Returns:
        X (ndarray): (d1,d2,d3,d4).
    """
    dim_1, dim_2, dim_3 = X.shape[0], X.shape[1], X.shape[2]
    for d1 in range(dim_1):
        for d2 in range(dim_2):
            for d3 in range(dim_3):
                X[d1, d2, d3, :] = X[d1, d2, d3, :] - np.mean(X[d1, d2, d3, :])
                X[d1, d2, d3, :] = X[d1, d2, d3, :] / np.std(X[d1, d2, d3, :])
    return X


@njit(fastmath=True)
def fast_stan_5d(X: ndarray) -> ndarray:
    """Special version of standardization() for 5-D X.

    Args:
        X (ndarray): (d1,d2,d3,d4,d5).

    Returns:
        X (ndarray): (d1,d2,d3,d4,d5).
    """
    dim_1, dim_2, dim_3, dim_4 = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    for d1 in range(dim_1):
        for d2 in range(dim_2):
            for d3 in range(dim_3):
                for d4 in range(dim_4):
                    X[d1, d2, d3, d4, :] = X[d1, d2, d3, d4, :] - np.mean(X[d1, d2, d3, d4, :])
                    X[d1, d2, d3, d4, :] = X[d1, d2, d3, d4, :] / np.std(X[d1, d2, d3, d4, :])
    return X
=======


def normalization(
        X: ndarray) -> ndarray:
    """Transform vector x into y
        s.t. y = (x - min(x)) / (max(x) - min(x)).
        The range of y is [0,1].

    Args:
        X (ndarray): (...,Np).

    Returns:
        Y (ndarray): Data after normalization.
    """
    X_min = np.min(X, axis=-1, keepdims=True)  # (...,1)
    X_max = np.max(X, axis=-1, keepdims=True)  # (...,1)
    return (X - X_min) / (X_max - X_min)


def standardization(
        X: ndarray) -> ndarray:
    """Transform vector x into y
        s.t. var(y) = 1.

    Args:
        X (ndarray): (...,Np).

    Returns:
        Y (ndarray): Data after standardization.
    """
    X = centralization(X)
    return X / np.std(X, axis=-1, keepdims=True)
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138


# %% 2. data preparation
def sin_wave(
        freq: Union[int, float],
        n_points: int,
        phase: float,
        srate: Union[int, float] = 1000) -> ndarray:
    """Construct sinusoidal waveforms.

    Args:
        freq (Union[int, float]): Frequency / Hz.
        n_points (int): Number of sampling points.
        phase (float): 0-2.
        srate (Union[int, float]): Sampling rate. Defaults to 1000.

    Returns:
        wave (ndarray): (n_points,). Sinusoidal sequence.
    """
    time_points = np.arange(n_points) / srate
    wave = sin(2 * pi * freq * time_points + pi * phase)
    return wave


def sine_template(
        freq: Union[int, float],
        phase: Union[int, float],
        n_points: int,
        n_harmonics: int,
        srate: Union[int, float] = 1000) -> ndarray:
    """Create sine-cosine template for SSVEP signals.

    Args:
        freq (Union[int, float]): Basic frequency.
        phase (Union[int, float]): Initial phase.
        n_points (int): Sampling points.
        n_harmonics (int): Number of harmonics.
        srate (Union[int, float]): Sampling rate. Defaults to 1000.

    Returns:
        Y (ndarray): (2*Nh,Np).
    """
    Y = np.zeros((2 * n_harmonics, n_points))  # (2Nh,Np)
    for nh in range(n_harmonics):
        Y[2 * nh, :] = sin_wave((nh + 1) * freq, n_points, 0 + (nh + 1) * phase, srate)
        Y[2 * nh + 1, :] = sin_wave((nh + 1) * freq, n_points, 0.5 + (nh + 1) * phase, srate)
    return Y


<<<<<<< HEAD
def Imn(m: int, n: int) -> ndarray:
=======
def Imn(
        m: int,
        n: int) -> ndarray:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Concatenate identical matrices into a big matrix.

    Args:
        m (int): Total number of identity matrix.
        n (int): Dimensions of the identity matrix.

    Returns:
        target (ndarray): (m*n, n).
    """
    Z = np.zeros((m * n, n))
    for i in range(m):
        Z[i * n:(i + 1) * n, :] = np.eye(n)
    return Z


<<<<<<< HEAD
def augmented_events(event_type: ndarray, d: int) -> Dict[str, List[int]]:
=======
def augmented_events(
        event_type: ndarray,
        d: int) -> Dict[str, List[int]]:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Generate indices for merged events for each target event.
        Special function for ms- algorithms.

    Args:
        event_type (ndarray): Unique labels.
        d (int): The range of events to be merged.

    Returns:
        events_group (dict): {'events':[idx,]}
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


def neighbor_edge(
        total_length: int,
        neighbor_range: int,
        current_index: int) -> Tuple[int, int]:
    """Decide the edge index (based on labels) of neighboring stimulus area.

    Args:
        total_length (int).
        neighbor_range (int): Must be an odd number.
        current_index (int): From 0 to total_lenth-1.

    Returns: Tuple[int, int]
        (edge_idx_1, edge_idx_2): edge_idx_2 is 1 more than the real index of the last element.
    """
    assert int(neighbor_range/2) != neighbor_range / 2, "Please use an odd number as neighbor_range!"

    half_length = int((neighbor_range - 1) / 2)
    if current_index <= half_length:  # left/upper edge
        return 0, current_index + half_length + 1
    # right/bottom edge
    elif current_index >= total_length - (half_length + 1):
        return current_index - half_length, total_length
    else:
        return current_index - half_length, current_index + half_length + 1


<<<<<<< HEAD
def neighbor_events(distribution: ndarray, width: int) -> Dict[str, List[int]]:
=======
def neighbor_events(
        distribution: ndarray,
        width: int) -> Dict[str, List[int]]:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Generate indices for merged events for each target event.
        Refers to: 10.1109/TIM.2022.3219497 (DOI).

    Args:
        distribution (ndarray): Real spatial distribution (labels) of each stimuli.
        width (int): Parameter 'neighbor_range' used in neighbor_edge(). Must be an odd number.

    Returns:
        events_group (dict): {'event_id': [idx_1,idx_2,...]}.
    """
    n_rows, n_cols = distribution.shape[0], distribution.shape[1]
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


def selected_events(
        n_events: int,
        select_num: int,
        select_method: str = '2') -> List[int]:
    """Generate indices for selected events of total dataset.
        Special function for stCCA.

    Args:
        n_events (int): Number of total events.
        select_num (int): Number of selected events.
        method (str, optional): '1', '2', and '3'. Defaults to '2'.
            Details in https://ieeexplore.ieee.org/document/9177172/

    Returns:
        select_events (List[int]): Indices of selected events.
    """
    if select_method == '1':
        return [1 + int((n_events - 1) * sen / (select_num - 1)) for sen in range(select_num)]
    elif select_method == '2':
        return [int(n_events * (2 * sen + 1) / (2 * select_num)) for sen in range(select_num)]
    elif select_method == '3':
        return [int(n_events * 2 * (sen + 1) / (2 * select_num)) for sen in range(select_num)]


def reshape_dataset(
        data: ndarray,
        labels: Optional[ndarray] = None,
        target_style: str = 'sklearn',
        filter_bank: bool = False) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    """Reshape data array between public versionand sklearn version.

    Args:
        data (ndarray):
            public version: (Ne,Nt,Nc,Np) or (Nb,Ne,Nt,Nc,Np) (filter_bank==True).
            sklearn version: (Ne*Nt,Nc,Np) or (Nb,Ne*Nt,Nc,Np)  (filter_bank==True).
        labels (ndarray, optional): (Ne*Nt,). Labels for data (sklearn version). Defaults to None.
        target_style (str): 'public' or 'sklearn'. Target style of transformed dataset.
        filter_bank (bool): Multi-band data or single-band data.

    Returns:
        if style=='public':
            X_total (ndarray): (Ne,Nt,Nc,Np) or (Nb,Ne,Nt,Nc,Np).
        elif style=='sklearn':
            X_total (ndarray): (Ne*Nt,Nc,Np) or (Nb,Ne*Nt,Nc,Np).
            y_total (ndarray): (Ne,). Labels for X_total.
    """
    # basic information
    n_points = data.shape[-1]  # Np
    n_chans = data.shape[-2]  # Nc

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
            X_total = np.zeros(
                (n_bands, n_events, n_trials, n_chans, n_points))
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
            X_total = np.zeros((n_bands, n_events*n_trials, n_chans, n_points))
            for nb in range(n_bands):
                for ne in range(n_events):
                    sp = ne*n_trials
                    X_total[nb, sp:sp+n_trials, ...] = data[nb, ne, ...]  # (Nt,Nc,Np)
        else:  # (Ne,Nt,Nc,Np)
            X_total = np.zeros((n_events*n_trials, n_chans, n_points))
            for ne in range(n_events):
                sp = ne*n_trials
                X_total[sp:sp+n_trials, ...] = data[ne, ...]  # (Nt,Nc,Np)
        return X_total, np.array(y_total).squeeze()


# %% 3. feature integration
<<<<<<< HEAD
def sign_sta(x: Union[int, float, ndarray]) -> Union[int, float, ndarray]:
=======
def sign_sta(
        x: float) -> float:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Standardization of decision coefficient based on sign(x).

    Args:
        x (Union[int, float, ndarray])

    Returns:
        y (Union[int, float, ndarray]): y=sign(x)*x^2
    """
    x = np.real(x)
    return (abs(x) / x) * (x**2)


def combine_feature(
        features: List[Union[int, float, ndarray]],
        func: Callable[[Union[int, float, ndarray]], Union[int, float, ndarray]] = sign_sta) -> ndarray:
    """Coefficient-level integration.

    Args:
        features (List[Union[int, float, ndarray]]): Different features.
        func (function): Quantization function.

    Returns:
        coef (the same type with elements of features): Integrated coefficients.
    """
    coef = np.zeros_like(features[0])
    for feature in features:
        coef += func(feature)
    return coef


# %% 4. algorithm evaluation
<<<<<<< HEAD
def label_align(y_pred: ndarray, event_type: ndarray) -> ndarray:
    """Label alignment.
        For example, y_train = [1,2,5], y_pred=[0,1,2]
        (Correct but with hidden danger in codes).
        This function will transform y_pred to [1,2,5].

    Args:
        y_pred (ndarray): (Nte,). Predict labels.
=======
def label_align(
        y_predict: ndarray,
        event_type: ndarray) -> ndarray:
    """Label alignment.
        For example, y_train = [1,2,5], y_predict=[0,1,2]
        (Correct but with hidden danger in codes).
        This function will transform y_predict to [1,2,5].

    Args:
        y_predict (ndarray): (Nte,). Predict labels.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        event_type (ndarray): (Ne,). Event ID arranged in ascending order.

    Returns:
        correct_predict (ndarray): (Nte,). Corrected labels.
    """
<<<<<<< HEAD
    correct_predict = np.zeros_like(y_pred)
    for npr, ypr in enumerate(y_pred):
=======
    correct_predict = np.zeros_like(y_predict)
    for npr, ypr in enumerate(y_predict):
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        correct_predict[npr] = event_type[int(ypr)]
    return correct_predict


<<<<<<< HEAD
def acc_compute(y_true: ndarray, y_pred: ndarray) -> float:
=======
def acc_compute(
        y_predict: ndarray,
        y_test: ndarray) -> float:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Compute accuracy.

    Args:
        y_pred (ndarray): (n_test,). Predict labels.
        y_true (ndarray): (n_test,). Real labels for test dataset.

    Returns:
        acc (float)
    """
<<<<<<< HEAD
    return np.sum(y_pred == y_true) / len(y_true)


=======
    return np.sum(y_predict == y_test) / len(y_test)


def confusion_matrix(
        rho: ndarray) -> ndarray:
    """Compute confusion matrix.

    Args:
        rho (ndarray): (Ne(real),Nte,Ne(model)). Decision coefficients.

    Returns:
        cm (ndarray): (Ne,Ne).
    """
    n_events = rho.shape[0]
    n_test = rho.shape[1]
    cm = np.zeros((n_events, n_events))  # (Ne,Ne)
    for ner in range(n_events):
        for nte in range(n_test):
            cm[ner, np.argmax(rho[ner, nte, :])] += 1
    return cm / n_test


>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
def itr_compute(
        number: int,
        time: Union[int, float],
        acc: float) -> float:
    """Compute information transfer rate.

    Args:
        number (int): Number of targets.
        time (Union[int, float]): (unit) second.
        acc (float): 0-1

    Returns:
        result (float)
    """
    part_a = log(number, 2)
    if int(acc) == 1 or acc == 100:  # avoid special situation
        part_b, part_c = 0, 0
    elif float(acc) == 0.0:
        return 0
    else:
        part_b = acc * log(acc, 2)
        part_c = (1 - acc) * log((1 - acc) / (number - 1), 2)
    result = 60 / time * (part_a + part_b + part_c)
    return result


# %% 5. spatial distances
def pearson_corr(
        X: ndarray,
        Y: ndarray,
<<<<<<< HEAD
        parallel: bool = False) -> Union[float, ndarray]:
    """Pearson correlation coefficient.

    Args:
        X (ndarray): (m,n).
            e.g. Spatial filtered single-trial data (Nk,Np).
        Y (ndarray): (l,m,n) or (m,n).
            e.g. Templates while parallel=True (Ne,Nk,Np) or False (Nk,Np).
        parallel (bool): An accelerator. Defaults to False.
            If False, X could only be compared with Y of shape (m,n);
            If True, X could be compared with Y of shape (l,m,n).

    Returns:
        corr_coef (Union[float, ndarray]):
            float: X (m,n) & Y (m,n).
            ndarray (l,): X (m,n) & Y (l,m,n).
    """
    X, Y = standardization(X), standardization(Y)
    n = X.shape[-1]

    # reshape data into vector-style: reshape() is 5 times faster than flatten()
    X = np.reshape(X, -1, order='C')  # (m*n,)
    if parallel:  # Y: (l,m,n)
        Y = np.reshape(Y, (Y.shape[0], -1), order='C')  # (l,m*n)
    else:  # Y: (m,n)
        Y = np.reshape(Y, -1, order='C')  # (m*n,)
    return Y @ X / n


@njit(fastmath=True)
def fast_corr_2d(X: ndarray, Y: ndarray) -> ndarray:
    """Special version of pearson_corr() for 2-dimensional X.
    Use the JIT compiler to make python codes run faster.

    Args:
        X (ndarray): (d1,d2). d2 is reshaped length, not real length.
            e.g. (Ne,Ne*Nk,Np) -> (Ne,Ne*Nk*Np) (reshaped, d1=Ne, d2=Ne*Nk*Np)
                               -> (Ne,) (return) -> 1/Np*(Ne,) (real corr)
        Y (ndarray): (d1,d2).

    Returns:
        corr (ndarray): (d1,).
    """
    dim_1, dim_2 = X.shape[0], X.shape[1]
    corr = np.zeros((dim_2))
    for d1 in range(dim_1):
        corr[d1] = X[d1, :] @ Y[d1, :].T
    return corr


@njit(fastmath=True)
def fast_corr_3d(X: ndarray, Y: ndarray) -> ndarray:
    """Special version of pearson_corr() for 3-D X.

    Args:
        X (ndarray): (d1,d2,d3).
        Y (ndarray): (d1,d2,d3).

    Returns:
        corr (ndarray): (d1,d2).
    """
    dim_1, dim_2, dim_3 = X.shape[0], X.shape[1], X.shape[2]
    corr = np.zeros((dim_2, dim_3))
    for d1 in range(dim_1):
        for d2 in range(dim_2):
            corr[d1, d2] = X[d1, d2, :] @ Y[d1, d2, :].T
    return corr


@njit(fastmath=True)
def fast_corr_4d(X: ndarray, Y: ndarray) -> ndarray:
    """Special version of pearson_corr() for 4-D X.

    Args:
        X (ndarray): (d1,d2,d3,d4).
        Y (ndarray): (d1,d2,d3,d4).

    Returns:
        corr (ndarray): (d1,d2,d3).
    """
    dim_1, dim_2, dim_3, dim_4 = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    corr = np.zeros((dim_2, dim_3, dim_4))
    for d1 in range(dim_1):
        for d2 in range(dim_2):
            for d3 in range(dim_3):
                corr[d1, d2, d3] = X[d1, d2, d3, :] @ Y[d1, d2, d3, :].T
    return corr


def fisher_score(X: ndarray, y: ndarray) -> ndarray:
    """Fisher Score (sequence).

    Args:
        X (ndarray): (Ne*Nt,Np). Dataset.
        y (ndarray): (Ne*Nt,). Labels of X.

    Returns:
=======
        common_filter: bool = False) -> Union[float, ndarray]:
    """Pearson correlation coefficient.

    Args:
        X (ndarray): (Nk,Np). Spatial filtered single-trial data.
        Y (ndarray): (Ne,Nk,Np) or (Nk,Np). Templates while common_filter=True or False.
        common_filter (bool): An accelerator. Defaults to False.
            If False, X could only be compared with Y of one event (Nk,Np);
            If True, X could be compared with Y of any event, i.e. (Ne,Nk,Np).

    Returns:
        corr_coef (Union[float, ndarray]).
    """
    X, Y = standardization(X), standardization(Y)
    n_points = X.shape[-1]

    # reshape data into vector-style: reshape() is 5 times faster than flatten()
    X = np.reshape(X, -1, order='C')  # (Ne*Nk*Np,) or (Nk*Np,)
    if common_filter:
        # (Ne,Ne*Nk*Np) or (Ne,Nk*Np)
        Y = np.reshape(Y, (Y.shape[0], -1), order='C')
    else:
        Y = np.reshape(Y, -1, order='C')  # (Nk*Np,)
    return Y @ X / n_points


def fisher_score(
        X: ndarray,
        y: ndarray) -> ndarray:
    """Fisher Score (sequence).

    Args:
        X (ndarray): (Ne*Nt,Np). Dataset.
        y (ndarray): (Ne*Nt,). Labels of X.

    Returns:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        fs (ndarray): (Np,). Fisher-Score sequence.
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


<<<<<<< HEAD
def euclidean_dist(X: ndarray, Y: ndarray) -> float:
=======
def euclidean_dist(
        X: ndarray,
        Y: ndarray) -> float:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Euclidean distance.

    Args:
        X (ndarray): (m, n).
        Y (ndarray): (m, n).

    Returns:
        dist (float)
    """
    dist = sqrt(np.sum((X - Y)**2))
    return dist


<<<<<<< HEAD
def cosine_sim(x: ndarray, y: ndarray) -> float:
=======
def cosine_sim(
        x: ndarray,
        y: ndarray) -> float:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Cosine similarity.

    Args:
        x, y (ndarray): (Np,)

    Returns:
        sim (float)
    """
    sim = einsum('i,i->', x, y) / \
        sqrt(einsum('i,i->', x, x)*einsum('i,i->', y, y))
    return sim


def minkowski_dist(
        x: ndarray,
        y: ndarray,
        p: Union[int, float]) -> float:
    """Minkowski distance.

    Args:
        x (ndarray): (n_points,).
        y (ndarray): (n_points,).
        p (Union[int, float]): Hyper-parameter.

    Returns:
        dist (float)
    """
    dist = einsum('i->', abs(x - y)**p)**(1 / p)
    return dist


<<<<<<< HEAD
def mahalanobis_dist(X: ndarray, y: ndarray) -> float:
=======
def mahalanobis_dist(
        X: ndarray,
        y: ndarray) -> float:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Mahalanobis distance.

    Args:
        X (ndarray): (Nt,Np). Training dataset.
        y (ndarray): (Np,). Test data.

    Returns:
        dist (float)
    """
    cov_XX = X.T @ X  # (Np,Np)
    mean_X = X.mean(axis=0, keepdims=True)  # (1,Np)
    dist = sqrt((mean_X - y) @ sLA.solve(cov_XX, (mean_X - y).T))
    return dist


<<<<<<< HEAD
def nega_root(X: ndarray) -> ndarray:
=======
def nega_root(
        X: ndarray) -> ndarray:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Compute the negative root of a square matrix.

    Args:
        X (ndarray): (m,m). Square matrix.

    Returns:
        nr_X (ndarray): (m,m). X^(-1/2).
    """
    e_val, e_vec = sLA.eig(X)
    nr_lambda = np.diag(1 / sqrt(e_val))
    nr_X = e_vec @ nr_lambda @ sLA.inv(e_vec)
    return nr_X


<<<<<<< HEAD
def s_estimator(X: ndarray) -> float:
=======
def s_estimator(
        X: ndarray) -> float:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Construct s-estimator.

    Args:
        X (ndarray): (m,m). Symmetric matrix.

    Returns:
        s_estimator (float)
    """
    e_val, _ = sLA.eig(X)
<<<<<<< HEAD
    e_val = e_val / np.sum(e_val)  # normalized eigenvalues
    entropy = np.sum([i * log(i, 2) for i in e_val])
    return 1 + entropy / log(X.shape[0], 2)


# %% 6. temporally smoothing functions
def tukeys_kernel(x: float, r: Union[int, float] = 3) -> float:
=======
    norm_e_val = e_val/einsum('ii->', X)
    numerator = np.sum([x * log(x) for x in norm_e_val])
    s_estimator = 1 + numerator / X.shape[0]
    return s_estimator


# %% 6. temporally smoothing functions
def tukeys_kernel(
        x: float,
        r: Union[int, float] = 3) -> float:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Tukeys tri-cube kernel function.
    Args:
        x (float)
        r (Union[int, float]): Defaults to 3.

    Returns:
        value (float): Values after kernel function mapping.
    """
    if abs(x) > 1:
        return 0
    else:
        return (1 - abs(x)**r)**r


def weight_matrix(
        n_points: int,
        tau: Union[int, float],
        r: Union[int, float] = 3) -> ndarray:
    """Weighting matrix based on kernel function.

    Args:
        n_points (int): Parameters that determine the size of the matrix.
        tau (Union[int, float]): Hyper-parameter for weighting matrix.
        r (Union[int, float]): Hyper-parameter for kernel funtion.

    Returns:
        W (ndarray): (Np,Np). Weighting matrix.
    """
    W = np.eye(n_points)
    for i in range(n_points):
        for j in range(n_points):
            W[i, j] = tukeys_kernel(x=(j - i) / tau, r=r)
    return W


<<<<<<< HEAD
def laplacian_matrix(W: ndarray) -> ndarray:
=======
def laplacian_matrix(
        W: ndarray) -> ndarray:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Laplace matrix for time smoothing.

    Args:
        W (ndarray): (n_points, n_points). Weighting matrix.

    Returns:
        L (ndarray): (n_points, n_points). Laplace matrix.
    """
    D = np.diag(np.sum(W, axis=-1))
    return D - W


# %% 7. reduced QR decomposition
<<<<<<< HEAD
def qr_projection(X: ndarray) -> ndarray:
=======
def qr_projection(
        X: ndarray) -> ndarray:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Orthogonal projection based on QR decomposition of X.

    Args:
        X (ndarray): (Np,m).

    Return:
        P (ndarray): (Np,Np).
    """
    Q, _ = sLA.qr(X, mode='economic')
    P = Q @ Q.T  # (Np,Np)
    return P


# %% 8. Eigenvalue problems
def pick_subspace(
        descend_order: List[Tuple[int, float]],
        e_val_sum: float,
        ratio: float) -> int:
    """Config the number of subspaces. (Deprecated)

    Args:
        descend_order (List[Tuple[int,float]]): See it in solve_gep() or solve_ep().
        e_val_sum (float): Trace of covariance matrix.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.

    Returns:
        n_components (int): The number of subspaces.
    """
    temp_val_sum = 0
    for n_components, do in enumerate(descend_order):  # n_sp: n_subspace
        temp_val_sum += do[-1]
        if temp_val_sum > ratio * e_val_sum:
            return n_components + 1


def solve_ep(
        A: ndarray,
        n_components: int = 1,
        mode: str = 'Max') -> ndarray:
    """Solve eigenvalue problems
        Rayleigh quotient: f(w)=wAw^T/(ww^T) -> Aw = lambda w

    Args:
        A (ndarray): (m,m)
        n_components (int): Number of eigenvectors picked as filters.
        mode (str): 'Max' or 'Min'. Depends on target function.

    Returns:
        w (ndarray): (Nk,m). Picked eigenvectors.
    """
    e_val, e_vec = sLA.eig(A)
    descend_order = sorted(enumerate(e_val), key=lambda x: x[1], reverse=True)
    w_index = [do[0] for do in descend_order]
    if mode == 'Min':
        return np.real(e_vec[:, w_index][:, n_components:].T)
    elif mode == 'Max':
        return np.real(e_vec[:, w_index][:, :n_components].T)


def solve_gep(
        A: ndarray,
        B: ndarray,
        n_components: int = 1,
        mode: str = 'Max') -> ndarray:
    """Solve generalized problems | generalized Rayleigh quotient:
        f(w)=wAw^T/(wBw^T) -> Aw = lambda Bw -> B^{-1}Aw = lambda w

    Args:
        A (ndarray): (m,m).
        B (ndarray): (m,m).
        n_components (int): Number of eigenvectors picked as filters.
        mode (str): 'Max' or 'Min'. Depends on target function.

    Returns:
        w (ndarray): (Nk,m). Picked eigenvectors.
    """
    e_val, e_vec = sLA.eig(sLA.solve(a=B, b=A, assume_a='sym'))  # ax=b -> x=a^{-1}b
    descend_order = sorted(enumerate(e_val), key=lambda x: x[1], reverse=True)
    w_index = [do[0] for do in descend_order]
    if mode == 'Min':
        return np.real(e_vec[:, w_index][:, n_components:].T)
    elif mode == 'Max':
        return np.real(e_vec[:, w_index][:, :n_components].T)


# %% 9. Signal generation
def get_resample_sequence(
        sequence: ndarray,
        rrate: int,
        srate: Union[int, float]) -> List[Tuple[int, float]]:
    """Obtain the resampled sequence from original sequence.

    Args:
        sequence (ndarray): (1, signal_length). Stimulus sequence of original sampling rate.
        rrate (int): Refresh rate of stimulation presentation device.
        srate (Union[int, float]): Sampling rate.

    Return:
        resampled_sequence (List[Tuple[int,float]]): (index, value).
            Resampled values and indices of stimulus sequence.
    """
    signal_length = sequence.shape[-1]
    resample_points = int(np.ceil(rrate * signal_length / srate))
    resample_index = np.round(srate / rrate * np.arange(resample_points) + 0.001)
    resample_value = [sequence[int(i)] for i in resample_index]
    resampled_sequence = [(int(ri), rv) for ri, rv in zip(resample_index, resample_value)]
    return resampled_sequence


def extract_periodic_impulse(
        freq: Union[int, float],
        phase: Union[int, float],
        signal_length: int,
        srate: Union[int, float],
        rrate: int) -> ndarray:
    """Extract periodic impulse sequence from stimulus sequence.

    Args:
        freq (Union[int, float]): Stimulus frequency.
        phase (Union[int, float]): Stimulus phase.
        signal_length (int): Total length of reconstructed signal.
        srate (Union[int, float]): Sampling rate.
        rrate (int): Refresh rate of stimulation presentation device.

    Return:
        periodic_impulse (ndarray): (1, signal_length)
    """
    # obtain the actual stimulus strength
    sine_sequence = sin_wave(
        freq=freq,
        n_points=signal_length,
        phase=0.5 + phase,
        srate=srate
    )
    resampled_sequence = get_resample_sequence(
        sequence=sine_sequence,
        refresh_rate=rrate,
        srate=srate
    )

    # pick up the peak values of resampled sequence
    periodic_impulse = np.zeros_like(sine_sequence)
    if phase == 0.5:
        periodic_impulse[0] = resampled_sequence[0][1]
    for rs in range(1, len(resampled_sequence)-1):
        left_edge = resampled_sequence[rs][1] >= resampled_sequence[rs-1][1]
        right_edge = resampled_sequence[rs][1] >= resampled_sequence[rs+1][1]
        if left_edge and right_edge:
            periodic_impulse[resampled_sequence[rs][0]] = resampled_sequence[rs][1]
    return periodic_impulse


<<<<<<< HEAD
def create_conv_matrix(periodic_impulse: ndarray, response_length: int) -> ndarray:
=======
def create_conv_matrix(
        periodic_impulse: ndarray,
        response_length: int) -> ndarray:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Create the convolution matrix of the periodic impulse.

    Args:
        periodic_impulse (ndarray): (1, signal_length). Impulse sequence of stimulus.
        response_length (int): Length of impulse response.

    Return:
        H (ndarray): (response_length, signal_length). Convolution matrix.
    """
    signal_length = periodic_impulse.shape[-1]
    H = np.zeros((response_length, response_length + signal_length - 1))
    for rl in range(response_length):
        H[rl, rl:rl + signal_length] = periodic_impulse
    return H[:, :signal_length]


def correct_conv_matrix(
        H: ndarray,
        freq: Union[int, float],
        srate: Union[int, float],
        scale: Union[int, float] = 0.8,
        mode: str = 'dynamic') -> ndarray:
    """Replace the blank values at the front of the reconstructed data with its subsequent fragment.

    Args:
        H (ndarray): (impulse_length, signal_length). Convolution matrix.
        freq (Union[int, float]): Stimulus frequency.
        srate (Union[int, float]): Sampling rate.
        scale (Union[int, float]): Compression coefficient of subsequent fragment (0-1).
            Defaults to 0.8.
        mode (str): 'dynamic' or 'static'.
            'static': Data fragment is intercepted starting from 1 s.
            'dynamic': Data fragment is intercepted starting from 1 period after the end of all-blank area.

    Return:
        correct_H (ndarray): (impulse_length, signal_length). Corrected convolution matrix.
    """
    shift_length = np.where(H[0] != 0)[0][0]  # Tuple -> ndarray -> int
    shift_matrix = np.eye(H.shape[-1])
    if mode == 'static':
        sp = srate  # start point
    elif mode == 'dynamic':
        sp = int(np.ceil(srate/freq))
    try:
        shift_matrix[sp:sp + shift_length, shift_length] = scale * np.eye(shift_length)
    except ValueError:
        raise Exception('Signal length is too short!')
    return H @ shift_matrix


# %% 10. Filter-bank technology
def generate_filter_bank(
        passbands: List[Tuple[float, float]],
        stopbands: List[Tuple[float, float]],
        srate: float,
        order: Optional[int] = None,
        gpass: float = 3,
        gstop: float = 40,
<<<<<<< HEAD
        rp: float = 0.5) -> List[ndarray]:
=======
        rp: float = 0.5) -> List:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """Generate Chebyshev type I filters according to input paramters.

    Args:
        passbands (List[Tuple[float,float]]): Passband edge frequencies.
        stopbands (List[Tuple[float,float]]): Stopband edge frequencies.
        srate (float): Sampling rate.
        order (int, optional): Order of filters.
        gpass (float): The maximum loss in the passband (dB).
        gstop (float): The minimum attenuation in the stopband (dB).
        rp (float): Ripple of filters. Default to 0.5 dB.

    Return:
        filter_bank (List[ndarray]): Second-order sections representation of the IIR filters.
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
    """Basic filter-bank object"""
<<<<<<< HEAD
    def __init__(
        self,
        base_estimator: BaseEstimator,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        version: Optional[str] = None
    ):
=======
    def __init__(self,
                 base_estimator: BaseEstimator,
                 filter_bank: Optional[List] = None,
                 with_filter_bank: bool = True,
                 version: Optional[str] = None):
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """Basic configuration.

        Args:
            base_estimator (BaseEstimator): Sub-model.
            filter_bank (List[ndarray], optional): See details in generate_filter_bank().
                Defaults to False.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            version (str, optional): If 'SSVEP', bank_weights wiil be set according to refrence:
                (DOI) 10.1088/1741-2560/12/4/046008.
        """
        self.base_estimator = base_estimator
        assert bool(filter_bank) != with_filter_bank, 'Check filter bank configuration!'
        self.filter_bank = filter_bank
        self.with_filter_bank = with_filter_bank
        self.version = version

<<<<<<< HEAD
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        bank_weights: Optional[ndarray] = None,
        **kwargs
    ):
=======
    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            bank_weights: Optional[ndarray] = None,
            **kwargs):
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """Load in training dataset and pass it to sub-esimators.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np) or (Nb,Ne*Nt,...,Np) (with_filter_bank=True).
                Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            bank_weights (ndarray, optional): Weights for different filter banks.
                Defaults to None (equal).
        """
        if self.with_filter_bank:  # X_train has been filterd: (Nb,Ne*Nt,...,Np)
            self.Nb = X_train.shape[0]
        else:  # tranform X_train into shape: (Nb,Ne*Nt,...,Np)
            self.Nb = len(self.filter_bank)
            X_train = self.fb_transform(X_train)

        self.bank_weights = bank_weights
        if self.version == 'SSVEP':
<<<<<<< HEAD
            self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25
                                          for nb in range(self.Nb)])

        self.sub_estimator = [clone(self.base_estimator)
                              for nb in range(self.Nb)]
        for nse, se in enumerate(self.sub_estimator):
            se.fit(
                X_train=X_train[nse],
                y_train=y_train,
                **kwargs
            )
        return self

    def transform(self, X_test: ndarray, **kwargs) -> Dict[str, ndarray]:
=======
            self.bank_weights = np.array(
                [(nb + 1)**(-1.25) + 0.25 for nb in range(self.Nb)])

        self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]
        for nse, se in enumerate(self.sub_estimator):
            se.fit(X_train=X_train[nse], y_train=y_train, **kwargs)
        return self

    def transform(self,
                  X_test: ndarray,
                  **kwargs) -> Tuple[ndarray, ndarray]:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """Transform test dataset to multi-band discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np) or (Nb,Ne*Nt,...,Np).
                Sklearn-style test dataset.

<<<<<<< HEAD
        Returns: Dict[str, ndarray]
=======
        Returns:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
            fb_rho (ndarray): (Nb,Ne*Nte,Ne). Multi-band decision coefficients.
            rho (ndarray): (Ne*Nte,Ne). Intergrated decision coefficients.
        """
        if not self.with_filter_bank:
            X_test = self.fb_transform(X_test)
<<<<<<< HEAD
        sub_features = [se.transform(X_test[nse], **kwargs)
                        for nse, se in enumerate(self.sub_estimator)]
        fb_rho = np.stack(sub_features, axis=0)
=======
        features = [se.transform(X_test[nse], **kwargs) for nse, se in enumerate(self.sub_estimator)]
        fb_rho = np.stack(features, axis=0)
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        if self.bank_weights is None:
            rho = fb_rho.mean(axis=0)
        else:
            rho = np.einsum('b,bte->te', self.bank_weights, fb_rho)
<<<<<<< HEAD
        features = {
            'fb_rho': fb_rho, 'rho': rho
        }
        return features

    def fb_transform(self, X_train: ndarray) -> ndarray:
=======
        return fb_rho, rho

    def fb_transform(self,
                     X_train: ndarray) -> ndarray:
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """Transform single-band X_train into multi-band X_train.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np). Sklearn-style training dataset.

        Returns:
            X_fb (ndarray): (Nb,Ne*Nt,...,Np). Multi-band X_train.
        """
<<<<<<< HEAD
        X_fb = np.stack([sosfiltfilt(sos, X_train, axis=-1)
                         for sos in self.filter_bank])
=======
        X_fb = np.stack([sosfiltfilt(sos, X_train, axis=-1) for sos in self.filter_bank])
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        return X_fb
