# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Other design.

update: 2022/10/22

"""

# %% basic modules
from utils import *


# %% PT projection
def pt_proj(X, theta):
    """
    Compute the PT projection matrix

    Args:
        X (ndarray): (n_train, n_chans, n_times)
        theta (float): Hyper-parameter. 0-1.

    Returns:
        projection (ndarray): (n_chans, n_chans)
    """
    # basic information
    n_train = X.shape[0]

    # projection formula
    A = einsum('tcp,thp->ch', X,X)  # (Nc,Nc)
    A /= n_train
    Xmean = X.mean(axis=0)  # (Nc,Np)
    B = Xmean @ Xmean.T  # (Nc,Nc)
    projection = sLA.solve(theta*A + (1-2*theta)*B, (1-theta)*B)
    return projection.T


# %% Discriminant Spatial Patterns
def dsp(train_data, class_center, Nk=1, ratio=None):
    """Discriminant Spatial Patterns for multi-class problems.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        class_center (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        w (ndarray): (Nk, n_chans). Common spatial filter.
    """
    # between-class difference Hb -> scatter matrix Sb
    total_center = class_center.mean(axis=0)  # (Nc,Np)
    
    pass


