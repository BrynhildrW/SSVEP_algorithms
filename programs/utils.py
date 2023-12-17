# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

1. Data preprocessing:
    (1-1) centralization()
    (1-2) normalization()
    (1-3) standardization()

2. Data preparation
    (2-1) sin_wave()
    (2-2) sine_template()
    (2-3) Imn()
    (2-4) augmented_events()
    (2-5) selected_events()
    (2-6) reshape_dataset()

3. feature integration
    (3-1) sign_sta()
    (3-2) combine_feature()
    (3-3) combine_fb_feature()

4. algorithm evaluation
    (4-1) acc_compute()
    (4-2) confusion_matrix()
    (4-3) itr_compute()

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
    (6-1) tukeys_kernel()
    (6-2) weight_matrix()
    (6-3) laplacian_matrix

7. reduced QR decomposition
    (7-1) qr_projection()

8. eigenvalue problems
    (8-1) pick_subspace()
    (8-2) solve_ep()
    (8-3) solve_gep()

9. signal generation
    (9-1) get_resample_sequence()
    (9-2) extract_periodic_impulse()
    (9-3) create_conv_matrix()
    (9-4) correct_conv_matrix()

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

update: 2023/07/06

"""

# %% basic moduls
from typing import Optional, List, Tuple, Any
from numpy import ndarray
import numpy as np
from numpy import sin, sqrt, einsum

from scipy import linalg as sLA

from math import (pi, log, pow)

import warnings


# %% 1. data preprocessing
def centralization(
    X: ndarray) -> ndarray:
    """Transform vector x into y, s.t. mean(y) = 0.
    
    Args:
        X (ndarray): (...,Np).
    
    Returns:
        Y (ndarray): Data after centralization.
    """
    return X - X.mean(axis=-1, keepdims=True)


def normalization(
    X: ndarray) -> ndarray:
    """Transform vector x into y, s.t. y = (x - min(x)) / (max(x) - min(x)).
        The range of y is [0,1].
    
    Args:
        X (ndarray): (...,Np).
    
    Returns:
        Y (ndarray): Data after normalization.
    """
    X_min = np.min(X, axis=-1, keepdims=True)  # (...,1)
    X_max = np.max(X, axis=-1, keepdims=True)  # (...,1)
    return (X-X_min)/(X_max-X_min)


def standardization(
    X: ndarray) -> ndarray:
    """Transform vector x into y, s.t. var(y) = 1.
    
    Args:
        X (ndarray): (...,Np).
    
    Returns:
        Y (ndarray): Data after standardization.
    """
    X = centralization(X)
    return X / np.std(X, axis=-1, keepdims=True)


# %% 2. data preparation
def sin_wave(
    freq: float,
    n_points: int,
    phase: float,
    sfreq: Optional[float] = 1000) -> ndarray:
    """Construct sinusoidal waveforms.

    Args:
        freq (float): Frequency / Hz.
        n_points (int): Number of sampling points.
        phase (float): 0-2.
        sfreq (int, optional): Sampling frequency. Defaults to 1000.

    Returns:
        wave (ndarray): (n_points,). Sinusoidal sequence.
    """
    time_points = np.arange(n_points) / sfreq
    wave = sin(2*pi*freq*time_points + pi*phase)
    return wave


def sine_template(
    freq: float,
    phase: float,
    n_points: int,
    n_harmonics: int,
    sfreq: Optional[float] = 1000) -> ndarray:
    """Create sine-cosine template for SSVEP signals.

    Args:
        freq (float or int): Basic frequency.
        phase (float or int): Initial phase.
        n_points (int): Sampling points.
        n_harmonics (int): Number of harmonics.
        sfreq (float or int): Sampling frequency. Defaults to 1000.

    Returns:
        Y (ndarray): (2*Nh,Np).
    """
    Y = np.zeros((2*n_harmonics, n_points))  # (2Nh,Np)
    for nh in range(n_harmonics):
        Y[2*nh,:] = sin_wave((nh+1)*freq, n_points, 0+(nh+1)*phase, sfreq)
        Y[2*nh+1,:] = sin_wave((nh+1)*freq, n_points, 0.5+(nh+1)*phase, sfreq)
    return Y


def Imn(
    m: int,
    n: int) -> ndarray:
    """Concatenate identical matrices into a big matrix.

    Args:
        m (int): Total number of identity matrix.
        n (int): Dimensions of the identity matrix.

    Returns:
        target (ndarray): (m*n, n).
    """
    Z = np.zeros((m*n,n))
    for i in range(m):
        Z[i*n:(i+1)*n, :] = np.eye(n)
    return Z


def augmented_events(
    event_type: ndarray,
    d: int):
    """Generate indices for merged events for each target event.
    Special function for ms- algorithms.

    Args:
        event_type (ndarray): Unique labels.
        d (int): The range of events to be merged.

    Returns:
        events_group (dict): {'events':[start index,end index]}
    """
    events_group = {}
    n_events = len(event_type)
    for ne,et in enumerate(event_type):
        if ne <= d/2:
            events_group[str(et)] = [0,d]
        elif ne >= int(n_events-d/2):
            events_group[str(et)] = [n_events-d,n_events]
        else:
            m = int(d/2)  # forward augmentation
            events_group[str(et)] = [ne-m,ne-m+d]
    return events_group


def neighbor_edge(total_length, neighbor_range, current_index):
    """Decide the edge index (based on labels) of neighboring stimulus area.

    Args:
        total_length (int).
        neighbor_range (int): Must be an odd number.
        current_index (int): From 0 to total_lenth-1.

    Returns: Tuple[int]
        edge_idx_1, edge_idx_2: edge_idx_2 is 1 more than the real index of the last element.
    """
    assert int(neighbor_range/2) != neighbor_range/2, "Please use an odd number as neighbor_range!"
    
    half_length = int((neighbor_range-1)/2)
    if current_index <= half_length:  # left/upper edge
        return 0, current_index + half_length + 1
    elif current_index >= total_length - (half_length + 1):  # right/bottom edge
        return current_index - half_length, total_length
    else:
        return current_index - half_length, current_index + half_length + 1


def neighbor_events(distribution, width):
    """Generate indices for merged events for each target event.
    Refers to: 10.1109/TIM.2022.3219497 (DOI).
    
    Args:
        distribution (ndarray of int): Real spatial distribution (labels) of each stimuli.
        width (int): Parameter 'neighbor_range' used in neighbor_edge(). Must be an odd number.
    
    Returns: dict
        events_group (dict[str, List[int]]): {'event_id':[idx_1,idx_2,...]}.
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
            event_id = str(distribution[row,col])
            events_group[event_id] = distribution[upper:bottom, left:right].flatten().tolist()
    return events_group


def selected_events(
    n_events: int,
    select_num: int,
    select_method: Optional[str] = 'A2') -> List[int]:
    """Generate indices for selected events of total dataset.
    Special function for stCCA.

    Args:
        n_events (int): Number of total events.
        select_num (int): Number of selected events.
        method (str, optional): 'A1', 'A2', and 'A3'.
            Defaults to '2'. Details in https://ieeexplore.ieee.org/document/9177172/

    Returns:
        select_events (List[int]): Indices of selected events.
    """
    if select_method == '1':
        return [1 + int((n_events-1)*sen/(select_num-1)) for sen in range(select_num)]
    elif select_method == '2':
        return [int(n_events*(2*sen+1)/(2*select_num)) for sen in range(select_num)]
    elif select_method == '3':
        return [int(n_events*2*(sen+1)/(2*select_num)) for sen in range(select_num)]


def reshape_dataset(
    data: ndarray,
    labels: Optional[ndarray] = None,
    target_style: Optional[str] = 'sklearn',
    filter_bank: Optional[bool] = False) -> Tuple[ndarray]:
    """Reshape data array between public versionand sklearn version.

    Args:
        data (ndarray):
            public version: (Ne,Nt,Nc,Np) or (Nb,Ne,Nt,Nc,Np) (filter_bank==True).
            sklearn version: (Ne*Nt,Nc,Np) or (Nb,Ne*Nt,Nc,Np)  (filter_bank==True).
        labels (ndarray): (Ne*Nt,). Labels for data (sklearn version). Defaults to None.
        target_style (str): 'public' or 'sklearn'. Target style of transformed dataset.

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
        n_train = np.array([np.sum(labels==et) for et in event_type])
        n_trials = np.min(n_train)
        if n_trials != np.max(n_train):
            warnings.warn('Unbalanced dataset! Some trials will be discarded!')

        # reshape data
        if filter_bank:  # (Nb,Ne*Nt,Nc,Np)
            n_bands = data.shape[0]  # Nb
            X_total = np.zeros((n_bands, n_events, n_trials, n_chans, n_points))
            for nb in range(n_bands):
                for ne,et in enumerate(event_type):
                    X_total[nb,ne,...] = data[nb][labels==et][:n_trials,...]  # (Nt,Nc,Np)
        else:  # (Ne,Nt,Nc,Np)
            X_total = np.zeros((n_events, n_trials, n_chans, n_points))
            for ne,et in enumerate(event_type):
                X_total[ne] = data[labels==et][:n_trials,...]  # (Nt,Nc,Np)
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
                    X_total[nb,sp:sp+n_trials,...] = data[nb,ne,...]  # (Nt,Nc,Np)
        else:  # (Ne,Nt,Nc,Np)
            X_total = np.zeros((n_events*n_trials, n_chans, n_points))
            for ne in range(n_events):
                sp = ne*n_trials
                X_total[sp:sp+n_trials,...] = data[ne,...]  # (Nt,Nc,Np)
        return X_total, np.array(y_total).squeeze()


# %% 3. feature integration
def sign_sta(
    x: float) -> float:
    """Standardization of decision coefficient based on sign(x).

    Args:
        x (float)

    Returns:
        y (float): y=sign(x)*x^2
    """
    x = np.real(x)
    return (abs(x)/x)*(x**2)


def combine_feature(
    features: List[ndarray],
    func: Optional[Any] = sign_sta) -> ndarray:
    """Coefficient-level integration.

    Args:
        features (List[float or int or ndarray]): Different features.
        func (function): Quantization function.

    Returns:
        coef (the same type with elements of features): Integrated coefficients.
    """
    coef = np.zeros_like(features[0])
    for feature in features:
        coef += func(feature)
    return coef


def combine_fb_feature(
    features: List[ndarray]) -> float:
    """Coefficient-level integration specially for filter-bank design.

    Args:
        features (List[ndarray]): Coefficient matrices of different sub-bands.

    Returns:
        coef (float): Integrated coefficients.

    """
    coef = np.zeros_like(features[0])
    for nf,feature in enumerate(features):
        coef += (pow(nf+1, -1.25) + 0.25) * (feature**2)
    return coef


# %% 4. algorithm evaluation
def label_align(
    y_predict: ndarray,
    event_type: ndarray) -> ndarray:
    """Label alignment.
        For example, y_train = [1,2,5], y_predict=[0,1,2]
        (Correct but with hidden danger in codes).
        This function will transform y_predict to [1,2,5].

    Args:
        y_predict (ndarray): (Nte,). Predict labels.
        event_type (ndarray): (Ne,). Event ID arranged in ascending order.

    Returns:
        correct_predict (ndarray): (Nte,). Corrected labels.
    """
    correct_predict = np.zeros_like(y_predict)
    for npr,ypr in enumerate(y_predict):
        correct_predict[npr] = event_type[int(ypr)]
    return correct_predict


def acc_compute(
    y_predict: ndarray,
    y_test: ndarray) -> float:
    """Compute accuracy.

    Args:
        y_predict (ndarray): (n_test,). Predict labels.
        y_test (ndarray): (n_test,). Real labels for test dataset.

    Returns:
        acc (float)
    """
    return np.sum(y_predict==y_test)/len(y_test)


def confusion_matrix(
    rou: ndarray) -> ndarray:
    """Compute confusion matrix.

    Args:
        rou (ndarray): (Ne(real),Nte,Ne(model)). Decision coefficients.

    Returns:
        cm (ndarray): (Ne,Ne).
    """
    n_events = rou.shape[0]
    n_test = rou.shape[1]
    cm = np.zeros((n_events, n_events))  # (Ne,Ne)
    for ner in range(n_events):
        for nte in range(n_test):
            cm[ner,np.argmax(rou[ner,nte,:])] += 1
    return cm/n_test


def itr_compute(
    number: int,
    time: float,
    acc: float) -> float:
    """Compute information transfer rate.

    Args:
        number (int): Number of targets.
        time (float): (unit) second.
        acc (float): 0-1

    Returns:
        result (float)
    """
    part_a = log(number,2)
    if int(acc)==1 or acc==100:  # avoid special situation
        part_b, part_c = 0, 0
    elif float(acc)==0.0:
        return 0
    else:
        part_b = acc*log(acc,2)
        part_c = (1-acc)*log((1-acc)/(number-1),2)
    result = 60 / time * (part_a+part_b+part_c)
    return result


# %% 5. spatial distances
def pearson_corr(
    X: ndarray,
    Y: ndarray,
    common_filter: bool = False) -> float:
    """Pearson correlation coefficient.

    Args:
        X (ndarray): (Nk,Np). Spatial filtered single-trial data.
        Y (ndarray): (Ne,Nk,Np) or (Nk,Np). Templates while common_filter=True or False.

    Returns:
        corr_coef (ndarray or float): (Ne,) or float.
    """
    X, Y = standardization(X), standardization(Y)
    n_points = X.shape[-1]

    # reshape data into vector-style: reshape() is 5 times faster than flatten()
    X = np.reshape(X, -1, order='C')  # (Ne*Nk*Np,) or (Nk*Np,)
    if common_filter:
        Y = np.reshape(Y, (Y.shape[0], -1), order='C')  # (Ne,Ne*Nk*Np) or (Ne,Nk*Np)
    else:
        Y = np.reshape(Y, -1, order='C')  # (Nk*Np,)
    return Y @ X / n_points


def fisher_score(
    dataset: Optional[Tuple[ndarray]] = None,
    *args: Tuple,
    **kwargs: dict) -> ndarray:
    """Fisher Score (sequence).

    Args:
        dataset (Tuple[ndarray] or List[ndarray]): (event1, event2, ...).
            The shape of each data matrix must be (Nt, n_features).
            n_features must be the same (n_trials could be various).

    Returns:
        fs (ndarray): (n_features,). Fisher-Score sequence.
    """
    # data information
    n_events = len(dataset)
    trials = np.array([data.shape[0] for data in dataset])  # (Ne,)
    n_features = dataset[0].shape[-1]

    # class center & total center
    class_center = np.zeros((n_events, n_features))
    for ne in range(n_events):
        class_center[ne,:] = dataset[ne].mean(axis=0)
    total_center = class_center.mean(axis=0)
    # total_center = (trials @ class_center)/trials.sum()

    # inter-class divergence
    decenter = class_center - total_center
    ite_d = trials @ decenter**2

    # intra-class divergence
    itr_d = np.zeros((n_features))
    for ne in range(n_events):
        itr_d += np.sum((dataset[ne] - class_center[ne,:])**2, axis=0)

    # fisher-score
    return ite_d/itr_d


def euclidean_dist(
    X: ndarray,
    Y: ndarray) -> float:
    """Euclidean distance.
    
    Args:
        X (ndarray): (m, n).
        Y (ndarray): (m, n).
        
    Returns:
        dist (float)
    """
    dist = sqrt(np.sum((X-Y)**2))
    return dist


def cosine_sim(
    x: ndarray,
    y: ndarray) -> float:
    """Cosine similarity.
    Equal to pearson_corr() if x & y are zero-normalized.

    Args:
        x, y (ndarray): (Np,)

    Returns:
        sim (float)
    """
    sim = einsum('i,i->',x,y)/sqrt(einsum('i,i->',x,x)*einsum('i,i->',y,y))
    return sim


def minkowski_dist(
    x: ndarray,
    y: ndarray,
    p: int) -> float:
    """Minkowski distance.

    Args:
        x (ndarray): (n_points,).
        y (ndarray): (n_points,).
        p (int): Hyper-parameter.

    Returns:
        dist (float)
    """
    dist = einsum('i->',abs(x-y)**p)**(1/p)
    return dist


def mahalanobis_dist(
    X: ndarray,
    y: ndarray) -> float:
    """Mahalanobis distance.

    Args:
        X (ndarray): (Nt,Np). Training dataset.
        y (ndarray): (Np,). Test data.

    Returns:
        dist (float)
    """
    cov_XX = X.T @ X  # (Np,Np)
    mean_X = X.mean(axis=0, keepdims=True)  # (1,Np)
    dist = sqrt((mean_X-y) @ sLA.solve(cov_XX, (mean_X-y).T))
    return dist


def nega_root(
    X: ndarray) -> ndarray:
    """Compute the negative root of a square matrix.

    Args:
        X (ndarray): (m,m). Square matrix.

    Returns:
        nr_X (ndarray): (m,m). X^(-1/2).
    """
    e_val, e_vec = sLA.eig(X)
    nr_lambda = np.diag(1/sqrt(e_val))
    nr_X = e_vec @ nr_lambda @ sLA.inv(e_vec)
    return nr_X


def s_estimator(
    X: ndarray) -> float:
    """Construct s-estimator.

    Args:
        X (ndarray): (m,m). Square matrix.

    Returns:
        s_estimator (float)
    """
    e_val, _ = sLA.eig(X)
    norm_e_val = e_val/einsum('ii->', X)
    numerator = np.sum([x*log(x) for x in norm_e_val])
    s_estimator = 1 + numerator/X.shape[0]
    return s_estimator


# %% 6. temporally smoothing functions
def tukeys_kernel(
    x: float,
    r: Optional[float] = 3) -> float:
    """Tukeys tri-cube kernel function.
    Args:
        x (float)
        r (int, optional): Defaults to 3.

    Returns:
        value (float): Values after kernel function mapping.
    """
    if abs(x)>1:
        return 0
    else:
        return (1-abs(x)**r)**r


def weight_matrix(
    n_points: int,
    tau: int,
    r: Optional[int] = 3) -> ndarray:
    """Weighting matrix based on kernel function.

    Args:
        n_points (int): Parameters that determine the size of the matrix.
        tau (int): Hyper-parameter for weighting matrix.
        r (int): Hyper-parameter for kernel funtion.

    Returns:
        W (ndarray): (Np,Np). Weighting matrix.
    """
    W = np.eye(n_points)
    for i in range(n_points):
        for j in range(n_points):
            W[i,j] = tukeys_kernel(x=(j-i)/tau, r=r)
    return W


def laplacian_matrix(
    W: ndarray) -> ndarray:
    """Laplace matrix for time smoothing.

    Args:
        W (ndarray): (n_points, n_points). Weighting matrix.

    Returns:
        L (ndarray): (n_points, n_points). Laplace matrix.
    """
    D = np.diag(np.sum(W, axis=-1))
    return D-W


# %% 7. reduced QR decomposition
def qr_projection(
    X: ndarray) -> ndarray:
    """Orthogonal projection based on QR decomposition of X.

    Args:
        X (ndarray): (Np,m).

    Return:
        P (ndarray): (Np,Np).
    """
    Q,_ = sLA.qr(X, mode='economic')
    P = Q @ Q.T  # (Np,Np)
    return P


# %% 8. Eigenvalue problems
def pick_subspace(
    descend_order: List[Tuple[int,float]],
    e_val_sum: float,
    ratio: float) -> int:
    """Config the number of subspaces.

    Args:
        descend_order (List[Tuple[int,float]]): See it in solve_gep() or solve_ep().
        e_val_sum (float): Trace of covariance matrix.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.

    Returns:
        n_components (int): The number of subspaces.
    """
    temp_val_sum = 0
    for n_components,do in enumerate(descend_order):  # n_sp: n_subspace
        temp_val_sum += do[-1]
        if temp_val_sum > ratio*e_val_sum:
            return n_components+1


def solve_ep(
    A: ndarray,
    n_components: Optional[int] = None,
    ratio: Optional[float] = None,
    mode: Optional[str] = 'Max') -> ndarray:
    """Solve eigenvalue problems | Rayleigh quotient: 
        f(w)=wAw^T/(ww^T) -> Aw = lambda w

    Args:
        A (ndarray): (m,m)
        B (ndarray): (m,m)
        n_components (int): Number of eigenvectors picked as filters.
            Eigenvectors are referring to eigenvalues sorted in descend order.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
        mode (str): 'Max' or 'Min'. Depends on target function.

    Returns:
        w (ndarray): (Nk,m). Picked eigenvectors.
    """
    e_val, e_vec = sLA.eig(A)
    e_val_sum = np.sum(e_val)
    descend_order = sorted(enumerate(e_val), key=lambda x:x[1], reverse=True)
    w_index = [do[0] for do in descend_order]
    if not n_components:
        n_components = pick_subspace(descend_order, e_val_sum, ratio)
    if mode == 'Min':
        return np.real(e_vec[:,w_index][:,n_components:].T)
    elif mode == 'Max':
        return np.real(e_vec[:,w_index][:,:n_components].T)


def solve_gep(
    A: ndarray,
    B: ndarray,
    n_components: Optional[int] = None,
    ratio: Optional[float] = None,
    mode: Optional[str] = 'Max') -> ndarray:
    """Solve generalized problems | generalized Rayleigh quotient:
        f(w)=wAw^T/(wBw^T) -> Aw = lambda Bw -> B^{-1}Aw = lambda w

    Args:
        A (ndarray): (m,m).
        B (ndarray): (m,m).
        n_components (int): Number of eigenvectors picked as filters.
            Eigenvectors are referring to eigenvalues sorted in descend order.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
        mode (str): 'Max' or 'Min'. Depends on target function.

    Returns:
        w (ndarray): (Nk,m). Picked eigenvectors.
    """
    e_val, e_vec = sLA.eig(sLA.solve(a=B, b=A, assume_a='sym'))  # ax=b -> x=a^{-1}b
    e_val_sum = np.sum(e_val)
    descend_order = sorted(enumerate(e_val), key=lambda x:x[1], reverse=True)
    w_index = [do[0] for do in descend_order]
    if not n_components:
        n_components = pick_subspace(descend_order, e_val_sum, ratio)
    if mode == 'Min':
        return np.real(e_vec[:,w_index][:,n_components:].T)
    elif mode == 'Max':
        return np.real(e_vec[:,w_index][:,:n_components].T)


def correct_direction():
    pass



# %% 9. Signal generation
def get_resample_sequence(
    sequence: ndarray,
    refresh_rate: int,
    sfreq: float) -> List[Tuple[int,float]]:
    """Obtain the resampled sequence from original sequence.

    Args:
        sequence (ndarray): (1, signal_length). Stimulus sequence of original sampling rate.
        refresh_rate (int): Refresh rate of stimulation presentation device.
        sfreq (float): Sampling frequency.

    Return:
        resampled_sequence (List[Tuple[int,float]]): (index, value).
            Resampled values and indices of stimulus sequence.
    """
    signal_length = sequence.shape[-1]
    resample_points = int(np.ceil(refresh_rate * signal_length / sfreq))
    resample_index = np.round(sfreq / refresh_rate * np.arange(resample_points) + 0.001)    
    resample_value = [sequence[int(i)] for i in resample_index]
    resampled_sequence = [(int(ri), rv) for ri, rv in zip(resample_index, resample_value)]
    return resampled_sequence


def extract_periodic_impulse(
    freq: float,
    phase: float,
    signal_length: int,
    sfreq: float,
    refresh_rate: int) -> ndarray:
    """Extract periodic impulse sequence from stimulus sequence.

    Args:
        freq (float): Stimulus frequency.
        phase (float): Stimulus phase.
        signal_length (int): Total length of reconstructed signal.
        sfreq (float): Sampling frequency.
        refresh_rate (int): Refresh rate of stimulation presentation device. 

    Return:
        periodic_impulse (ndarray): (1, signal_length)
    """
    # obtain the actual stimulus strength
    sine_sequence = sin_wave(
        freq=freq,
        n_points=signal_length,
        phase=0.5+phase,
        sfreq=sfreq
    )
    resampled_sequence = get_resample_sequence(
        sequence=sine_sequence,
        refresh_rate=refresh_rate,
        sfreq=sfreq
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


def create_conv_matrix(
    periodic_impulse: ndarray,
    response_length: int) -> ndarray:
    """Create the convolution matrix of the periodic impulse.

    Args:
        periodic_impulse (ndarray): (1, signal_length). Impulse sequence of stimulus.
        response_length (int): Length of impulse response.

    Return:
        H (ndarray): (response_length, signal_length). Convolution matrix.
    """
    signal_length = periodic_impulse.shape[-1]
    H = np.zeros((response_length, response_length+signal_length-1))
    for rl in range(response_length):
        H[rl, rl:rl+signal_length] = periodic_impulse
    return H[:,:signal_length]


def correct_conv_matrix(
        H: ndarray,
        freq: float,
        sfreq: float,
        scale: Optional[float] = 0.8,
        mode: Optional[str] = 'dynamic') -> ndarray:
    """Replace the blank values at the front of the reconstructed data with its subsequent fragment.

    Args:
        H (ndarray): (impulse_length, signal_length). Convolution matrix.
        freq (float): Stimulus frequency.
        sfreq (float): Sampling frequency.
        scale (float, Optional): Compression coefficient of subsequent fragment (0-1).
            Defaults to 0.8.
        mode (str, Optional): 'dynamic' or 'static'.
            'static': Data fragment is intercepted starting from 1 s.
            'dynamic': Data fragment is intercepted starting from 1 period after the end of all-blank area.

    Return:
        correct_H (ndarray): (impulse_length, signal_length). Corrected convolution matrix.
    """
    shift_length = np.where(H[0]!=0)[0][0]  # Tuple -> ndarray -> int
    shift_matrix = np.eye(H.shape[-1])
    if mode == 'static':
        start_point = sfreq
    elif mode == 'dynamic':
        start_point = int(np.ceil(sfreq/freq))
    try:
        shift_matrix[start_point:start_point+shift_length, :shift_length] = scale*np.eye(shift_length)
    except ValueError:
        raise Exception('Signal length is too short!')
    return H @ shift_matrix