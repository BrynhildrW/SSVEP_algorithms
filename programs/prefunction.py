# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Prefunctions for specific algorithms.

update: 2022/7/12

"""

# %% basic moduls
import numpy as np
from numpy import (sin, sqrt, diagonal, einsum)

from scipy import linalg as LA

from math import (pi, log)

from itertools import combinations

# %% prefunctions
def zero_mean(X):
    """Zero-mean normalization.

    Args:
        X (ndarray): (..., n_points). Data array.
    
    Returns:
        X (ndarray): (..., n_points). Data after normalization.
    """
    X -= X.mean(axis=-1, keepdims=True)
    return X


def sin_wave(freq, n_points, phase, sfreq=1000):
    """Construct sinusoidal wave manually.

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


def time_shift(data, step, axis=None):
    """Cyclic shift data.

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
    for i in range(n_chans-1):
        tf_data[i+1,:] = np.roll(data[i+1,:], shift=round(step*(i+1)), axis=axis)
    return tf_data


def Imn(m,n):
    """Make concatenated eye matrices.

    Args:
        m (int): Total number of identity matrix.
        n (int): Dimensions of the identity matrix.

    Returns:
        target (ndarray): (m*n, n). Vertical concatenation of m unity matrices (n,n)
    """
    Z = np.zeros((m*n,n))
    for i in range(m):
        Z[i*n:(i+1)*n, :] = np.eye(n)
    return Z


def combine_feature(X):
    """Two-level feature extraction.
    
    Args:
        X (list of float): List of one-level features.
    
    Returns:
        tl_feature (float): Two-level feature.
    """
    tl_feature = 0
    for feature in X:
        sign = abs(feature)/feature  # sign(*) function
        tl_feature += sign*(feature**2)
    return tl_feature


def sign_sta(x):
    """Standardization of decision coefficient based on sign() function.
    
    Args:
        x (float)
        
    Returns:
        y (float): y=sign(x)*x^2
    """
    return (abs(x)/x)*(x**2)


def acc_compute(rou):
    """Compute accuracy.

    Args:
        rou (ndarray): (n_events(real), n_test, n_events(models)). Decision coefficients.

    Returns:
        correct (list): (n_events,). Correct trials for each event.
        acc (float)
    """
    n_events = rou.shape[0]
    n_test = rou.shape[1]
    correct = []
    for netr in range(n_events):
        temp = 0
        for nte in range(n_test):
            if np.argmax(rou[netr,nte,:])==netr:
                temp += 1
        correct.append(temp)
    return correct, np.sum(correct)/(n_test*n_events)


def itr(number, time, acc):
    """Compute information transfer rate.

    Args:
        number (int): Number of targets.
        time (float): (unit) second.
        acc (float): 0-1

    Returns:
        correct (list): (n_events,). Correct trials for each event.
        acc (float)
    """
    part_a = log(number,2)
    if acc==1.0 or acc==100:  # avoid spectial situation
        part_b, part_c = 0, 0
    else:
        part_b = acc*log(acc,2)
        part_c = (1-acc)*log((1-acc)/(number-1),2)
    result = 60 / time * (part_a+part_b+part_c)
    return result



# %% spatial distances
def corr_coef(X, y):
    """Pearson's Correlation Coefficient.
    Args:
        X (ndarray): (m, n_points). Multi-pieces of data.
            m could be 1 but the shape of X must be (1, n_points) then.
        y (ndarray): (1, n_points) or (n_points,)
    Returns:
        corrcoef (ndarray): (1,m) or (m,)
    """
    cov_yX = y @ X.T
    var_XX, var_yy = sqrt(diagonal(X @ X.T)), sqrt(y @ y.T)
    corrcoef = cov_yX / (var_XX*var_yy)
    return corrcoef


def corr2_coef(X, Y):
    """2-D Pearson correlation coefficient.
    Args:
        X (ndarray): (m, n_points)
        Y (ndarray): (m, n_points)
    Returns:
        corrcoef (float)
    """
    mean_X, mean_Y = X.mean(), Y.mean()
    numerator = einsum('ij->', (X-mean_X)*(Y-mean_Y))
    denominator_X = einsum('ij->', (X-mean_X)**2)
    denominator_Y = einsum('ij->', (Y-mean_Y)**2)
    # using np.einsum() here is nearly 10% faster than np.sum(), i.e.
    # numerator = np.sum((X-mean_X)*(Y-mean_Y))
    # denominator_X = np.sum((X-mean_X)**2)
    # denominator_Y = np.sum((Y-mean_Y)**2)
    corrcoef = numerator / sqrt(denominator_X*denominator_Y)
    return corrcoef


def fisher_score(dataset=(), *args):
    """Fisher Score (sequence) in time domain.
    Args:
        dataset (tuple of ndarray): (event1, event2, ...).
            The shape of each data matrix must be (n_trials, n_features).
            n_features must be the same (n_trials could be various).
    Returns:
        fs (ndarray): (1, n_features). Fisher-Score sequence.
    """
    # data information
    n_events = len(dataset)
    trials = np.array([x.shape[0] for x in dataset])  # (Ne,)
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
    fs = ite_d / itr_d
    return fs


def euclidean_dist(X, Y):
    """Euclidean distance.
    Args:
        X (ndarray): (n_chans, n_points).
        Y (ndarray): (n_chans, n_points).
    Returns:
        dist (float)
    """
    dist = sqrt(einsum('ij->',(X-Y)**2))
    return dist


def cosine_sim(x, y):
    """Cosine similarity.
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
        y (ndarray): (n_points,) or (1, n_points). Test data.
    Returns:
        dist (float)
    """
    cov_XX = einsum('ij,ik->jk', X,X)  # (Np, Np)
    mean_X = X.mean(axis=0, keepdims=True)  # (1, Np)
    dist = sqrt((mean_X-y) @ LA.solve(cov_XX, (mean_X-y).T))
    return dist


def nega_root(X):
    """Compute the negative root of a square matrix.

    Args:
        X (ndarray): (m,m). Square matrix.

    Returns:
        nr_X (ndarray): (m,m). X^(-1/2).
    """
    e_val, e_vec = LA.eig(X)
    nr_lambda = np.diag(1/sqrt(e_val))
    nr_X = e_vec @ nr_lambda @ LA.inv(e_vec)
    return nr_X


def s_estimator(X):
    """Construct s-estimator.

    Args:
        X (ndarray): (m,m). Square matrix.

    Returns:
        s_estimator (float)
    """
    e_val, _ = LA.eig(X)
    norm_e_val = e_val/einsum('ii->', X)
    numerator = np.sum([x*log(x) for x in norm_e_val])
    s_estimator = 1 + numerator/X.shape[0]
    return s_estimator


# %% temporally smoothing functions
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

# %% (future)
# %%
