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
    projection = LA.solve(theta*A + (1-2*theta)*B, (1-theta)*B)
    return projection.T


# %% Discriminant Canonical Pattern Matching
def dcpm(X, Y):
    pass


# %% Filter-bank framework