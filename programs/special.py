# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Other design.
    (1) DSP: https://ieeexplore.ieee.org/document/8930304/
        DOI: 10.1109/TBME.2019.2958641
    (2) DCPM: https://ieeexplore.ieee.org/document/8930304/
        DOI: 10.1109/TBME.2019.2958641
    (3) PT projection: None
        DOI: None


update: 2022/10/22

"""

# %% basic modules
from utils import *

import numpy as np

from scipy import linalg as sLA

# %% (1) Discriminant Spatial Patterns
def dsp_compute(train_data, n_components=1, ratio=None):
    """Discriminant Spatial Patterns (DSP).

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        n_components (int, optional): Number of eigenvectors picked as filters.
            Defaults to 1. Set to 'None' if ratio is not 'None'.
        ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not 'None'.

    Returns:
        Sb (ndarray): (n_chans, n_chans). Scatter matrix of between-class difference.
        Sw (ndarray): (n_chans, n_chans). Scatter matrix of within-class difference.
        w (ndarray): (n_components, n_chans). Common spatial filter.
        model (ndarray): (n_events, n_components, n_points). EEG templates.
    """
    # basic information
    n_events = train_data.shape[0]  # Ne
    n_train = train_data.shape[1]  # Nt
    n_chans = train_data.shape[2]  # Nc
    
    # between-class difference Hb -> scatter matrix Sb
    class_center = train_data.mean(axis=1)  # (Ne,Nc,Np)
    total_center = class_center.mean(axis=0, keepdims=True)  # (Nc,Np)
    Hb = class_center - total_center  # (Ne,Nc,Np)
    Sb = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for ne in range(n_events):
        Sb += Hb[ne] @ Hb[ne].T
    Sb /= n_events
    # Sb = np.einsum('ecp,ehp->ch', Hb,Hb)/n_events | clearer but slower

    # within-class difference Hw -> scatter matrix Sw
    Hw = train_data - class_center[:,None,...]  # (Ne,Nt,Nc,Np)
    Sw = np.zeros_like(Sb)  # (Nc,Nc)
    for ne in range(n_events):
        for ntr in range(n_train):
            Sw += Hw[ne,ntr,...] @ Hw[ne,ntr,...].T
    Sw /= (n_events*n_train)
    # Sw = einsum('etcp,ethp->ch', Hw,Hw)/(n_events*n_train) | clearer but slower

    # GEPs | training spatial filter
    w = solve_gep(A=Sb, B=Sw, n_components=n_components, ratio=ratio)  # (Nk,Nc)

    # signal templates
    template = np.einsum('kc,ecp->ekp', w,class_center)  # (Ne,Nk,Np)
    return Sb, Sw, w, template


class DSP(object):
    """Discriminant Spatial Patterns (DSP) for multi-target classification problems."""
    def __init__(self, n_components=1, ratio=None):
        """Config model dimension.

        Args:
            n_components (int, optional): Number of eigenvectors picked as filters.
                Defaults to 1. Set to 'None' if ratio is not 'None'.
            ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
                Defaults to None when n_component is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio


    def fit(self, train_data):
        """Train DSP model.

        Args:
            train_data (ndarray): (n_events, n_train, n_chans, n_points).
        """
        # basic information
        self.train_data = train_data
        self.n_events = self.train_data.shape[0]  # Ne
        self.n_train = self.train_data.shape[1]  # Nt
        self.n_chans = self.train_data.shape[2]  # Nc
        
        # train DSP models & templates
        self.Sb, self.Sw, self.w, self.template = dsp_compute(
            train_data=self.train_data,
            n_components=self.n_components,
            ratio=self.ratio
            )
        return self


    def predict(self, test_data):
        """Using DSP algorithm onto original EEG data.

        Args:
            test_data (ndarray): (n_events, n_test, n_chans, n_points).

        Return:
            rou (ndarray): (n_events(real), n_test, n_events(model)).
            label (ndarray): (n_events, n_test, predict_label). Updated in future version.
        """
        # basic information
        self.n_test = test_data.shape[1]

        # pattern matching
        self.rou = np.zeros((self.n_events, self.n_test, self.n_events))  # (Ne(real),Nt,Ne(model))
        for ner in range(self.n_events):
            for nte in range(self.n_test):
                temp = test_data[ner,nte,...]  # (Nc,Np)
                for nem in range(self.n_events):
                    self.rou[ner,nte,nem] = pearson_corr(self.w@temp, self.template[nem])
        return self.rou


# %% (2) Discriminant Canonical Pattern Matching | DCPM



# %% (3) PT projection
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
    A = np.einsum('tcp,thp->ch', X,X)  # (Nc,Nc)
    A /= n_train
    Xmean = X.mean(axis=0)  # (Nc,Np)
    B = Xmean @ Xmean.T  # (Nc,Nc)
    projection = sLA.solve(theta*A + (1-2*theta)*B, (1-theta)*B)
    return projection.T


# %% Filter-bank TRCA series | FB-
def fb_dsp_m1(train_data, test_data, n_components=1, ratio=None):
    """DSP-M1 algorithms with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        models (list of DSP objects). DSP objects created under various filter-bank.
        rou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_bands = test_data.shape[0]

    # multiple DSP classification
    models, rou = [], []
    for nb in range(n_bands):
        sub_band = DSP(n_components=n_components, ratio=ratio)
        sub_band.fit(train_data=train_data[nb])
        models.append(sub_band)
        rou.append(sub_band.predict(test_data=test_data[nb]))
    return models, combine_fb_feature(rou)