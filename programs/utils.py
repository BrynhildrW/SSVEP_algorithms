# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

1. Data preprocessing:
    (1-1) zero_mean()

2. Data preparation
    (2-1) sin_wave()
    (2-2) sine_template()
    (2-3) time_shift()
    (2-4) Imn()
    (2-5) augmented_events()
    (2-6) selected_events()
    (2-7) reshape_dataset()

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

update: 2023/05/23

"""

# %% basic moduls
import numpy as np
from numpy import sin, sqrt, einsum

from scipy import linalg as sLA

from math import (pi, log, pow)


# %% 1. data preprocessing
def zero_mean(X):
    """Zero-mean normalization.

    Args:
        X (ndarray): (..., n_points). Data array.

    Returns:
        X (ndarray): (..., n_points). Data after normalization.
    """
    X -= X.mean(axis=-1, keepdims=True)
    return X


# %% 2. data preparation
def sin_wave(freq, n_points, phase, sfreq=1000):
    """Construct sinusoidal waveforms.

    Args:
        freq (float): Frequency / Hz.
        n_points (int): Number of sampling points.
        phase (float): 0-2.
        sfreq (int, optional): Sampling frequency. Defaults to 1000.

    Returns:
        wave (ndarray): (n_points,). Sinusoidal sequence.
    """
    time_points = np.linspace(0, (n_points-1)/sfreq, n_points)
    wave = sin(2*pi*freq*time_points + pi*phase)
    return wave


def sine_template(freq, phase, n_points, n_harmonics, sfreq=1000):
    """Create sine-cosine template for SSVEP signals.

    Args:
        freq (float or int): Basic frequency.
        phase (float or int): Initial phase.
        n_points (int): Sampling points.
        n_harmonics (int): Number of harmonics.
        sfreq (float or int): Sampling frequency. Defaults to 1000.

    Returns:
        Y (ndarray): (2*n_harmonics, n_points).
    """
    Y = np.zeros((2*n_harmonics, n_points))  # (2Nh, Np)
    for nh in range(n_harmonics):
        Y[2*nh,:] = sin_wave((nh+1)*freq, n_points, 0+(nh+1)*phase, sfreq)
        Y[2*nh+1,:] = sin_wave((nh+1)*freq, n_points, 0.5+(nh+1)*phase, sfreq)
    return Y


def time_shift(data, step, axis=None):
    """Shift data cyclically.

    Args:
        data (ndarray): (n_chans, n_points). Input data array.
        step (int or float): The length of the scroll.
        axis (int or tuple, optional): Dimension of scrolling, 0 - vertical, 1 - horizontal.
            Defaults to None. By default(None), the array will be flattened before being shifted,
            and then restored to its original shape.

    Returns:
        tf_data (ndarray): (n_chans, n_points).
    """
    n_chans = data.shape[0]
    tf_data = np.zeros_like(data)
    tf_data[0,:] = data[0,:]
    for nc in range(n_chans-1):
        tf_data[nc+1,:] = np.roll(data[nc+1,:], shift=round(step*(nc+1)), axis=axis)
    return tf_data


def Imn(m,n):
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
    event_type: np.ndarray,
    d: int
):
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


def selected_events(n_events, select_num, select_method='A2'):
    """Generate indices for selected events of total dataset.
    Special function for stCCA.

    Args:
        n_events (int): Number of total events.
        select_num (int): Number of selected events.
        method (str, optional): 'A1', 'A2', and 'A3'.
            Defaults to '2'. Details in https://ieeexplore.ieee.org/document/9177172/

    Returns:
        select_events (list of int): Indices of selected events.
    """
    if select_method == '1':
        return [1 + int((n_events-1)*sen/(select_num-1)) for sen in range(select_num)]
    elif select_method == '2':
        return [int(n_events*(2*sen+1)/(2*select_num)) for sen in range(select_num)]
    elif select_method == '3':
        return [int(n_events*2*(sen+1)/(2*select_num)) for sen in range(select_num)]


def reshape_dataset(data):
    """Reshape dataset from SSVEP version (Ne,Nt,Nc,Np) into common version (Ne*Nt,Nc,Np).

    Args:
        data (ndarray): (Ne,Nt,Nc,Np) or (Nb,Ne,Nt,Nc,Np).

    Returns:
        X_total (ndarray): (Ne*Nt,Nc,Np) or (Nb,Ne*Nt,Nc,Np).
        y_total (ndarray): (Ne,). Labels for X_total.
    """
    # basic information
    n_points = data.shape[-1]  # Np
    n_chans = data.shape[-2]  # Nc
    n_trials = data.shape[-3]  # Nt
    n_events = data.shape[-4]  # Ne

    # create labels
    y_total = []
    for ne in range(n_events):
        y_total += [ne for ntr in range(n_trials)]

    # reshape data
    if data.ndim == 4:  # (Ne,Nt,Nc,Np)
        X_total = np.zeros((n_events*n_trials, n_chans, n_points))
        for ne in range(n_events):
            sp = ne*n_trials
            X_total[sp:sp+n_trials,...] = data[ne,...]  # (Nt,Nc,Np)
    elif data.ndim == 5:  # (Nb,Ne,Nt,Nc,Np)
        n_bands = data.shape[0]  # Nb
        X_total = np.zeros((n_bands, n_events*n_trials, n_chans, n_points))
        for nb in range(n_bands):
            for ne in range(n_events):
                sp = ne*n_trials
                X_total[nb,sp:sp+n_trials,...] = data[nb,ne,...]  # (Nt,Nc,Np)
    return X_total, np.array(y_total)


# %% 3. feature integration
def sign_sta(x):
    """Standardization of decision coefficient based on sign(x).

    Args:
        x (float)

    Returns:
        y (float): y=sign(x)*x^2
    """
    x = np.real(x)
    return (abs(x)/x)*(x**2)


def combine_feature(features, func=sign_sta):
    """Coefficient-level integration.

    Args:
        features (list of float/int/ndarray): Different features.
        func (functions): Quantization function.

    Returns:
        coef (the same type with elements of features): Integrated coefficients.

    """
    coef = np.zeros_like(features[0])
    for feature in features:
        coef += func(feature)
    return coef


def combine_fb_feature(features):
    """Coefficient-level integration specially for filter-bank design.

    Args:
        features (list of ndarray): Coefficient matrices of different sub-bands.

    Returns:
        coef (float): Integrated coefficients.

    """
    coef = np.zeros_like(features[0])
    for nf,feature in enumerate(features):
        coef += (pow(nf+1, -1.25) + 0.25) * (feature**2)
    return coef


# %% 4. algorithm evaluation
def label_align(y_predict, event_type):
    """Label alignment.
        For example, y_train = [1,2,5], y_predict=[0,1,2]
        (Correct but with hidden danger in codes).
        This function will transform y_predict to [1,2,5].

    Args:
        y_predict (ndarray): (n_test,). Predict labels.
        event_type (ndarray): (n_events,). Event ID arranged in ascending order.

    Returns:
        correct_predict (ndarray): (n_test,). Corrected labels.
    """
    correct_predict = np.zeros_like(y_predict)
    for npr,ypr in enumerate(y_predict):
        correct_predict[npr] = event_type[int(ypr)]
    return correct_predict


def acc_compute(y_predict, y_test):
    """Compute accuracy.

    Args:
        y_predict (ndarray): (n_test,). Predict labels.
        y_test (ndarray): (n_test,). Real labels for test dataset.

    Returns:
        acc (float)
    """
    return np.sum(y_predict==y_test)/len(y_test)


def confusion_matrix(rou):
    """Compute confusion matrix.

    Args:
        rou (ndarray): (n_events(real), n_test, n_events(models)). Decision coefficients.

    Returns:
        cm (ndarray): (n_events, n_events).
    """
    n_events = rou.shape[0]
    n_test = rou.shape[1]
    cm = np.zeros((n_events, n_events))  # (Ne,Ne)
    for ner in range(n_events):
        for nte in range(n_test):
            cm[ner,np.argmax(rou[ner,nte,:])] += 1
    return cm/n_test


def itr_compute(number, time, acc):
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
def pearson_corr(X, Y):
    """Pearson correlation coefficient (1-D or 2-D).
    
    Args:
        X (ndarray): (..., n_points)
        Y (ndarray): (..., n_points). The dimension must be same with X.
        
    Returns:
        corrcoef (float)
    """
    # check if not zero_mean():
    # X,Y = zero_mean(X), zero_mean(Y)
    cov_xy = np.sum(X*Y)
    var_x = np.sum(X**2)
    var_y = np.sum(Y**2)
    corrcoef = cov_xy / sqrt(var_x*var_y)
    return corrcoef


def fisher_score(dataset=None, *args, **kwargs):
    """Fisher Score (sequence) in time domain.

    Args:
        dataset (tuple or list of ndarray): (event1, event2, ...).
            The shape of each data matrix must be (n_trials, n_features).
            n_features must be the same (n_trials could be various).

    Returns:
        fs (ndarray): (n_features). Fisher-Score sequence.
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


def euclidean_dist(X, Y):
    """Euclidean distance.
    
    Args:
        X (ndarray): (m, n).
        Y (ndarray): (m, n).
        
    Returns:
        dist (float)
    """
    dist = sqrt(einsum('ij->',(X-Y)**2))
    return dist


def cosine_sim(x, y):
    """Cosine similarity.
    Equal to pearson_corr() if x & y are zero-normalized.

    Args:
        x (ndarray or list): (n_points,)
        y (ndarray or list): (n_points,)

    Returns:
        sim (float)
    """
    sim = einsum('i,i->',x,y)/sqrt(einsum('i,i->',x,x)*einsum('i,i->',y,y))
    return sim


def minkowski_dist(x, y, p):
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


def mahalanobis_dist(X, y):
    """Mahalanobis distance.

    Args:
        X (ndarray): (n_trials, n_points). Training dataset.
        y (ndarray): (n_points,). Test data.

    Returns:
        dist (float)
    """
    cov_XX = einsum('ij,ik->jk', X,X)  # (Np, Np)
    mean_X = X.mean(axis=0, keepdims=True)  # (1, Np)
    dist = sqrt((mean_X-y) @ sLA.solve(cov_XX, (mean_X-y).T))
    return dist


def nega_root(X):
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


def s_estimator(X):
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
def tukeys_kernel(x, r=3):
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


def weight_matrix(n_points, tao, r=3):
    """Weighting matrix based on kernel function.

    Args:
        n_points (int): Parameters that determine the size of the matrix.
        tao (int): Hyper-parameter for weighting matrix.
        r (int): Hyper-parameter for kernel funtion.

    Returns:
        W (ndarray): (n_points, n_points). Weighting matrix.
    """
    W = np.eye(n_points)
    for i in range(n_points):
        for j in range(n_points):
            W[i,j] = tukeys_kernel(x=(j-i)/tao, r=r)
    return W


def laplacian_matrix(W):
    """Laplace matrix for time smoothing.

    Args:
        W (ndarray): (n_points, n_points). Weighting matrix.

    Returns:
        L (ndarray): (n_points, n_points). Laplace matrix.
    """
    D = np.diag(einsum('ij->i',W))
    return D-W


# %% 7. reduced QR decomposition
def qr_projection(X):
    """Orthogonal projection based on QR decomposition of X.

    Args:
        X (ndarray): (n_points, m).

    Return:
        P (ndarray): (n_points, n_points).
    """
    Q,_ = sLA.qr(X, mode='economic')
    P = Q @ Q.T  # (Np,Np)
    return P


# %% 8. Eigenvalue problems
def pick_subspace(descend_order, e_val_sum, ratio):
    """Config the number of subspaces.

    Args:
        descend_order (List of tuple, (idx,e_val)): See it in solve_gep() or solve_ep().
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


def solve_ep(A, n_components=None, ratio=None, mode='Max'):
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
        w (ndarray): (n_components,m). Picked eigenvectors.
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


def solve_gep(A, B, n_components=None, ratio=None, mode='Max'):
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
        w (ndarray): (n_components,m). Picked eigenvectors.
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
