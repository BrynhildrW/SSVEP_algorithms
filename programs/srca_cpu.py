# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Spatial Regression Component Analysis (SRCA) series.

Supported objects
1. SRCA: single channel & single-event
    Target functions (1-D): SNR, pCORR (1-D)
    Optimization methods: Traversal, Recursion, Mix

2. ESRCA: single channel & multi-event (Ensemble-SRCA)
    Target functions (1-D): SNR, FS, pCORR
    Optimization methods: Traversal, Recursion, Mix

3. TDSRCA: single channel & single-event (Two-dimensional SRCA)
    Target functions (2-D): TRCA coef, TRCA eval
    Optimization methods: Traversal, Recursion, Mix

4. MCSRCA: multi-channel & single-event (Future Version)
    Target functions (2-D): DSP coef, 
    Optimization methods: Traversal, Recursion, Mix
    Combination optimization methods: SA(Simulated annealing), IBI(Item-by-item)

5. MCESRCA: multi-channel & multi-event (Future Version)
    Target functions:
    Optimization methods: Traversal, Recursion, Mix
    Combination optimization methods: SA, IBI

update: 2022/11/30

"""

# %% basic modules
import utils
import special

import numpy as np


# %% 1-D target functions | single channel
def snr_sequence(train_data, *args, **kwargs):
    """Signal-to-Noise ratio (sequence) in time domain.

    Args:
        train_data (ndarray): (..., n_trials, n_points). Input data.

    Returns:
        snr (ndarray): (..., 1, n_points). SNR sequence in time domain. 
    """
    pure_signal = train_data.mean(axis=-2, keepdims=True)  # (..., 1, n_points)
    signal_power = pure_signal**2  # (..., 1, n_points)
    noise_power = ((train_data-pure_signal)**2).mean(axis=-2, keepdims=True)  # (..., 1, n_points)
    snr = signal_power / noise_power  # (..., 1, n_points)
    return snr


def fs_sequence(train_data, *args, **kwargs):
    """Fisher Score (sequence) in time domain.
 
    Args:
        train_data (ndarray): (n_events, n_trials, n_points). Data array.

    Returns:
        fs (ndarray): (1, n_points). Fisher-Score sequence.
    """
    n_events = train_data.shape[0]  # Ne
    dataset = [train_data[ne] for ne in range(n_events)]  # (event1, event2, ...)
    return utils.fisher_score(dataset)


# %% 2-D target functions | multiple channels



def dsp_func(train_data, Nk=1, ratio=None):
    """f(w)=(w @ S_b @ w.T)/(w @ S_w @ w.T).

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.
        Nk=Nk, ratio=ratio)  # (Nk,Nc)

    Returns:
        w (ndarray): (Nk, n_chans). Common spatial filter.
        coef (float): f(w)
    """
    # basic information
    n_events = train_data.shape[0]
    n_train = train_data.shape[1]
    n_chans = train_data.shape[2]

    # between-class difference Hb -> scatter matrix Sb
    class_center = train_data.mean(axis=1)  # (Ne,Nc,Np)
    total_center = class_center.mean(axis=0)  # (Nc,Np)
    Hb = class_center - total_center  # (Ne,Nc,Np)
    Sb = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for ne in range(n_events):
        Sb += Hb[ne] @ Hb[ne].T
    Sb /= n_events

    # within-class difference Hw -> scatter matrix Sw
    Hw = train_data - class_center[:,None,...]  # (Ne,Nt,Nc,Np)
    Sw = np.zeros_like(Sb)  # (Nc,Nc)
    for ne in range(n_events):
        for ntr in range(n_train):
            Sw += Hw[ne,ntr,...] @ Hw[ne,ntr,...].T
    Sw /= (n_events*n_train)
    
    # GEPs
    w = utils.solve_gep(A=Sb, B=Sw, Nk=Nk, ratio=ratio)  # (Nk,Nc)
    return w, np.mean((w@Sb@w.T)/(w@Sw@w.T))

