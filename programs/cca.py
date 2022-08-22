# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

CCA series.

update: 2022/7/20

"""

# %% basic modules
from utils import *

# %% Canonical correlation analysis
# standard CCA | CCA
def cca_model(base_freq, n_bands, n_points, phase=0, sfreq=1000):
    """Sinusoidal models used in CCA.

    Args:
        base_freq (int or float): Frequency / Hz.
        n_bands (int): Number of harmonic bands.
        n_points (int): Number of sampling points.
        phase (float, optional): 0-2. Defaults to 0.
        sfreq (float or int, optional): Sampling frequency. Defaults to 1000.

    Returns:
        Y (ndarray): (2*n_bands, n_points). Artificial model.
    """
    Y = np.zeros((2*n_bands, n_points))
    for nb in range(n_bands):
        Y[nb*2, :] = sin_wave((nb+1)*base_freq, n_points, 0+phase, sfreq)
        Y[nb*2+1, :] = sin_wave((nb+1)*base_freq, n_points, 0.5+phase, sfreq)
    return Y


def cca_compute(X, Y):
    """Canonical correlation analysis.
    
    Args:
        X (ndarray): (n_chans, n_points). Real EEG data of a single trial.
        Y (ndarray): (2*n_harmonics, n_points). Artificial sinusoidal template.
    
    Returns:
        U (ndarray): (1,n_chans). Spatial filter for EEG.
        V (ndarray): (1,n_chans). Spatial filter for template.
    """
    # GEPs' conditions
    Cxx = X @ X.T  # (Nc,Nc)
    Cyy = Y @ Y.T  # (2Nh,2Nh)
    Cxy = X @ Y.T  # (Nc,2Nh)
    Cyx = Y @ X.T  # (2Nh,Nc)
    A = LA.solve(Cxx,Cxy) @ LA.solve(Cyy,Cyx)  # AU = lambda*U
    B = LA.solve(Cyy,Cyx) @ LA.solve(Cxx,Cxy)  # Bv = lambda*V
    
    # EEG part
    e_val, e_vec = LA.eig(A)
    U = e_vec[:,[np.argmax(e_val)]].T  # (1,Nc)
    
    # template part
    e_val, e_vec = LA.eig(B)
    V = e_vec[:,[np.argmax(e_val)]].T  # (1,2Nh)
    return U, V


def cca(template, test_data):
    """Using CCA to compute decision coefficients.
    
    Args:
        template (ndarray): (n_events, 2*n_harmonics, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        
    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_events = test_data.shape[0]
    n_test = test_data.shape[1]
    
    # CCA classification
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real, Nt, Ne model)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                U, V = cca_compute(temp, template[nem,...])
                rou[ner,nte,nem] = corr_coef(U@temp, V@template[nem,...])[0,0]
    return rou


# Extended CCA | eCCA
def ecca_compute(Xmean, Y, data):
    """CCA with individual calibration data.
    
    Args:
        Xmean (ndarray): (n_chans, n_points). Trial-averaged SSVEP template.
        Y (ndarray): (2*n_harmonics, n_points). Sinusoidal SSVEP template.
        data (ndarray): (n_chans, n_points). Test-trial EEG.
        
    Returns:
        rou (float): feature coefficient.
    """
    # correlation coefficient from CCA process
    U1, V1 = cca(X=data, Y=Y)
    r1 = corr_coef(U1@data, V1@Y)[0,0]
    
    # correlation coefficients between single-trial EEG and SSVEP templates
    U2, V2 = cca(X=data, Y=Xmean)  # (1,Nc)
    r2 = corr_coef(U2@data, U2@Xmean)[0,0]
    
    # U3, _ = cca(X=data, Y=Y)  # (1,Nc)
    r3 = corr_coef(U1@data, U1@Xmean)[0,0]
    
    U4, _ = cca(X=Xmean, Y=Y)  # (1,Nc)
    r4 = corr_coef(U4@data, U4@Xmean)[0,0]
    
    # similarity between filters corresponding to single-trial EEG and SSVEP templates
    # U5, V5 = cca(X=data, Y=Xmean)  # (1,Nc), (1,Nc)
    r5 = corr_coef(U2@Xmean, V2@Xmean)[0,0]
    
    # combined features
    rou = combine_feature([r1, r2, r3, r4, r5])
    return rou


def ecca(train_data, test_data, Y):
    """Use eCCA to compute decision coefficient.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Y (ndarray): (n_events, 2*n_harmonics, n_points).
    
    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]
    
    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                Xmean = train_data[nem,...].mean(axis=0)  # (Nc,Np)
                rou[ner,nte,nem] = ecca_compute(Xmean, Y[nem,...], temp)
    return rou


# Multi-stimulus CCA | msCCA
def mscca_compute(Xmean, Qy):
    """Multi-stimulus CCA.
    
    Args:
        Xmean (ndarray): (n_events, n_chans, n_points). Trial-averaged template.
        Qy (ndarray): (n_events, n_points, n_points). QR decompostion of artificial template Y.
    
    Returns:
        w (ndarray): (1, n_chans). Common filter.
        template (ndarray): (n_events, n_points).
    """
    # basic information
    n_events = Xmean.shape[0]
    n_chans = Xmean.shape[1]
    
    # A: power of SSVEP-related component
    A = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for nek in range(n_events):
        for nej in range(n_events):
            for nei in range(n_events):
                A += Xmean[nei,...]@Qy[nek,...]@Qy[nei,...].T@Qy[nej,...]@Qy[nek,...].T@Xmean[nej,...].T
    
    # B: inter-channel covariance of all events
    B = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for ne in range(n_events):
        B += Xmean[ne,...] @ Xmean[ne,...].T
        
    # GEPs
    e_val, e_vec = LA.eig(LA.solve(B,A))
    w_index = np.argmax(e_val)
    w = e_vec[:,[w_index]].T
    return w


def mscca(train_data, test_data, Qy):
    """Use msCCA to compute decision coefficients.
    
    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Qy (ndarray): (n_events, n_points, n_points). QR decompostion of artificial template Y.
        
    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]
    n_chans = test_data.shape[2]
    n_points = test_data.shape[-1]
    
    # training models & filters
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model = np.zeros((n_events, n_points))  # (Ne,Np)
    w = mscca_compute(train_mean, Qy)  # (1,Nc)
    for ne in range(n_events):
        model[ne,:] = w @ train_mean[ne,...]  # (1,Np)
    
    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = corr_coef(w@temp, model[[nem],:])[0,0]
    return rou


# Cross-subject spatial filter transfer method | CSSFT
def cssft_compute():
    pass


def cssft():
    pass