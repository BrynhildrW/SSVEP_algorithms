# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Prefunctions for specific algorithms.

update: 2022/7/12

"""

# %% basic moduls
import numpy as np
from numpy import (ndarray, sin, sqrt, diagonal, einsum)

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


# %% spatial distances
def corr_coef(X, y):
    """Pearson's Correlation Coefficient.
    Args:
        X (ndarray): (m, n_points)
        y (ndarray): (1, n_points)
    Returns:
        corrcoef (ndarray): (1,m) 
    """
    cov_yX = y @ X.T
    var_XX, var_yy = sqrt(diagonal(X @ X.T)), sqrt(y @ y.T)
    corrcoef = cov_yX / (var_XX*var_yy)
    return corrcoef

def corr2_coef(X, Y):
    """2-D Pearson correlation coefficient.
    Args:
        X (ndarray): (n_chans, n_points)
        Y (ndarray): (n_chans, n_points)
    Returns:
        corrcoef (float)
    """
    mean_X, mean_Y = X.mean(), Y.mean()
    numerator = einsum('ij->', (X-mean_X)*(Y-mean_Y))
    denominator_X = einsum('ij->', (X-mean_X)**2)
    denominator_Y = einsum('ij->', (Y-mean_Y)**2)
    corrcoef = numerator / sqrt(denominator_X*denominator_Y)
    return corrcoef

def fisher_score(X, Y, *args):
    """Fisher Score (sequence) in time domain.
    Args:
        X (ndarray): (n_trials_X, n_points).
        Y (ndarray): (n_trials_Y, n_points).
    Returns:
        fs (ndarray): (1, n_points). Fisher-Score sequence.
    """
    # data initialization
    mean_X = X.mean(axis=0, keepdims=True)  # (1, Np)
    mean_Y = Y.mean(axis=0, keepdims=True)  # (1, Np)
    mean_total = 0.5* (mean_X + mean_Y)  # (1, Np)
    # inter-class divergence
    ite_d = X.shape[0]*((mean_X-mean_total)**2) + Y.shape[0]*((mean_Y-mean_total)**2)
    # intra-class divergence
    itr_d = X.shape[0]*np.sum((X-mean_X)**2, axis=0) + Y.shape[0]*np.sum((Y-mean_Y)**2, axis=0)
    # fisher-score
    fs = ite_d / itr_d  # (1, Np)
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