# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

TRCA series.

update: 2022/7/20

"""

# %% basic modules
from utils import *

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


def trca_compute_V2(X):
    """Another version of trca_compute().

    Args:
        X (ndarray): (2, n_train, n_chans, n_points).

    Returns:
        w (ndarray): (2, n_chans).
    """
    # preprocess
    X = zero_mean(X)
    
    # basic information
    n_events = X.shape[0]
    n_chans = X.shape[2]
    
    # Q: covariance of averaged data | (Ne,Nc,Nc)
    Q = einsum('etcp,ethp->ech', X,X)

    # S: variance of original data | (Ne,Nc,Nc)
    Xsum = np.sum(X, axis=1)  # (Ne,Nc,Np)
    S = einsum('ecp,ehp->ech', Xsum,Xsum)

    # GEPs with symbol verification
    w = np.zeros((n_events, n_chans))  # (nrep,Ne,Nc)
    for ne in range(n_events):
        e_val, e_vec = LA.eig(LA.solve(Q[ne,...],S[ne,...]))
        w[ne,:] = e_vec[:,[np.argmax(e_val)]].T
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


def etrca_V2(train_data, test_data):
    """Another version for etrca().
       Because eigenvectors are not number-preserving, we cannot judge whether
    the spatial filter trained on target data will cause the phase inversion
    only based on the values of the filter vector when the frequencies of
    the stimuli are the same with their phases opposite.
       So I provide a temporary solution. After both filters are trained, the 
    Pearson correlation coefficient of two projected templates are checked. If 
    that value is positive, it means that one of those filters has a phase inversion.
    Then a filter is randomly selected and multiplied by -1.
    
    Args:
        train_data (ndarray): (2, n_train, n_chans, n_points).
        test_data (ndarray): (2, n_test, n_chans, n_points).
    
    Returns:
        rou (ndarray): (2, n_test, 2).
        erou (ndarray): (2, n_test, 2).
    """
    # basic information
    n_test = test_data.shape[1]
    n_chans = test_data.shape[2]
    n_points = test_data.shape[-1]
    
    # training models & filters
    w = trca_compute_V2(train_data)  # (Ne,Nc)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model = np.zeros((2, n_points))  # (Ne,Np)
    emodel = np.zeros((2, 2, n_points))  # (Ne real, Ne model, Np)
    for ne in range(2):
        model[ne,:] = w[ne,:] @ train_mean[ne,...]  # (1,Np)
        emodel[ne,...] = w @ train_mean[ne,...]  # (Ne,Np)
    
    # sign check for filters & models
    coef = corr_coef(model[[0],:], model[[1],:])[0,0]
    if coef > 0:
        w[-1,:] *= -1
        model[-1,:] *= -1
        emodel[-1,:] *= -1
    
    # pattern matching
    rou = np.zeros((2, n_test, 2))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)
    for ner in range(2):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(2):
                rou[ner,nte,nem] = corr_coef(w[[nem],:]@temp, model[[nem],:])[0,0]
                erou[ner,nte,nem] = corr2_coef(w@temp, emodel[nem,...])
    return rou, erou


# multi-stimulus (e)TRCA | ms-(e)TRCA
def augmented_events(n_events, d):
    """Generate indices for merged events for each target event.
    
    Args:
        n_events (int)
        d (int): The range of events to be merged.
    
    Returns:
        events_sheet (dict): {'events':[start index,end index]}
    """
    events_sheet = {}
    for ne in range(n_events):
        if ne <= d/2:
            events_sheet[str(ne)] = [0,d]
        elif ne >= int(n_events-d/2):
            events_sheet[str(ne)] = [n_events-d,n_events]
        else:
            m = int(d/2)  # forward augmentation
            events_sheet[str(ne)] = [ne-m,ne-m+d]
    return events_sheet


def mstrca_compute(X):
    """Multi-stimulus TRCA.

    Args:
        X (ndarray): (n_events, n_train, n_chans, n_points). Merged training dataset.

    Returns:
        w (ndarray): (1, n_chans). Eigenvector refering to the largest eigenvalue.
    """
    pass


# (e)TRCA-R




# similarity constrained (e)TRCA | sc-(e)TRCA



# group TRCA | gTRCA