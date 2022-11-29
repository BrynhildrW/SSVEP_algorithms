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
def dsp_compute(train_data, class_center, Nk=1, ratio=None):
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
    # basic information
    n_events = train_data.shape[0]
    n_train = train_data.shape[1]
    n_chans = train_data.shape[2]

    # between-class difference Hb -> scatter matrix Sb
    total_center = class_center.mean(axis=0)  # (Nc,Np)
    Hb = class_center - total_center  # (Ne,Nc,Np)
    Sb = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for ne in range(n_events):
        Sb += Hb[ne] @ Hb[ne].T
    Sb /= n_events
    # Sb = einsum('ecp,ehp->ch', Hb,Hb)/n_events | clearer but slower

    # within-class difference Hw -> scatter matrix Sw
    Hw = train_data - class_center[:,None,...]  # (Ne,Nt,Nc,Np)
    Sw = np.zeros_like(Sb)  # (Nc,Nc)
    for ne in range(n_events):
        for ntr in range(n_train):
            Sw += Hw[ne,ntr,...] @ Hw[ne,ntr,...].T
    Sw /= (n_events*n_train)
    # Sw = einsum('etcp,ethp->ch', Hw,Hw)/(n_events*n_train) | clearer but slower

    # GEPs
    return solve_gep(Sb, Sw, Nk, ratio)  # (Nk,Nc)


def dsp_m1(train_data, class_center, test_data, Nk=1, ratio=None):
    """Using DSP onto original EEG data.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        class_center (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # training models & filters
    class_center = train_data.mean(axis=1)  # (Ne,Nc,Np)
    w = dsp_compute(train_data=train_data, class_center=class_center,
        Nk=Nk, ratio=ratio)  # (Nk,Nc)
    model = einsum('kc,ecp->ekp', w,class_center)  # (Ne,Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne(real),Nt,Ne(model))
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = pearson_corr(w@temp, model[nem])
    return rou


# %% Filter-bank TRCA series | FB-
def fb_dsp_m1(train_data, class_center, test_data, Nk=1, ratio=None):
    """DSP-M1 algorithms with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        class_center (ndarray): (n_bands, n_events, n_chans, n_points). Trial-averaged data.
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_bands = test_data.shape[0]
    
    # multiple TDCA classification
    rou = []
    for nb in range(n_bands):
        temp_rou = dsp_m1(train_data=train_data[nb], class_center=class_center[nb],
            test_data=test_data[nb], Nk=Nk, ratio=ratio)
        rou.append(temp_rou)
    return combine_fb_feature(rou)