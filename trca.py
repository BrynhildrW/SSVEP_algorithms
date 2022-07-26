# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

TRCA series.

update: 2022/7/20

"""

# %% basic modules
from prefunction import *

# %% Task-related component analysis
# (ensemble) TRCA | (e)TRCA
def trca_compute(X):
    """Task-related component analysis.

    Args:
        X (ndarray): (n_train, n_chans, n_points). Training dataset.

    Returns:
        w (ndarray): (1, n_chans). Eigenvector refering to the largest eigenvalue.
    """
    # basic information
    X = zero_mean(X)

    # Q: inter-channel covariance
    Q = einsum('tcp,thp->ch', X,X)

    # S: inter-channels' inter-trial covariance
    Xsum = np.sum(X, axis=0)  # (Nc,Np)
    S = Xsum @ Xsum.T

    # GEPs
    e_val, e_vec = LA.eig(LA.solve(Q,S))
    w_index = np.argmax(e_val)
    w = e_vec[:,[w_index]].T
    return w


def etrca(train_data, test_data):
    """Using TRCA & eTRCA to compute decision coefficients.
    Comment/uncomment lines with (*) to use TRCA only.
    
    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
    
    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
        erou (ndarray): (n_events, n_test, n_events).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]
    n_chans = test_data.shape[2]
    n_points = test_data.shape[-1]

    # training models & filters
    w = np.zeros((n_events, n_chans))  # (Ne,Nc)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model = np.zeros((n_events, n_points))  # (Ne,Np)
    emodel = np.zeros((n_events, n_events, n_points))  # (*) (Ne real,Ne model,Np)
    for ne in range(n_events):
        w[ne,:] = trca_compute(train_data[ne,...])
    for ne in range(n_events):
        model[ne,:] = w[ne,:] @ train_mean[ne,...]  # (1,Np)
        emodel[ne,...] = w @ train_mean[ne,...]  # (*) (Ne,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)  # (*)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = corr_coef(w[[nem],:]@temp, model[[nem],:])[0,0]
                erou[ner,nte,nem] = corr2_coef(w@temp, emodel[nem,...])  # (*)
    # return rou  # (*)
    return rou, erou  # (*)

