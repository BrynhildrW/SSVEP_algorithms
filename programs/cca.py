# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Canonical correlation analysis (CCA) series.
    (1) CCA: http://ieeexplore.ieee.org/document/4203016/
            DOI: 10.1109/TBME.2006.889197
    (2) eCCA: http://www.pnas.org/lookup/doi/10.1073/pnas.1508080112
            DOI: 10.1073/pnas.1508080112
    (3) msCCA: https://ieeexplore.ieee.org/document/9006809/
            DOI: 10.1109/TBME.2020.2975552
    (4) ms-eCCA: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
            DOI: 10.1088/1741-2552/ab2373

update: 2022/11/11

"""

# %% basic modules
from utils import *


# %% standard CCA | CCA
def cca_compute(X, Y, Nk=1, ratio=None):
    """Canonical correlation analysis.

    Args:
        X (ndarray): (n_chans, n_points). Real EEG data of a single trial.
        Y (ndarray): (2*n_harmonics, n_points). Artificial sinusoidal template.
        Nk (int): Number of eigenvectors picked as filters.
            Eigenvectors are referring to eigenvalues sorted in descend order.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        U (ndarray): (Nk, n_chans). Spatial filter for EEG.
        V (ndarray): (Nk, 2*n_harmonics). Spatial filter for template.
    """
    # GEPs' conditions
    Cxx = X @ X.T  # (Nc,Nc)
    Cyy = Y @ Y.T  # (2Nh,2Nh)
    Cxy = X @ Y.T  # (Nc,2Nh)
    Cyx = Cyx.T  # (2Nh,Nc)
    A = LA.solve(Cxx,Cxy) @ LA.solve(Cyy,Cyx)  # AU = lambda*U
    B = LA.solve(Cyy,Cyx) @ LA.solve(Cxx,Cxy)  # BV = lambda*V

    # EEG part
    U = solve_ep(A, Nk, ratio)  # (Nk,Nc)

    # template part
    V = solve_ep(B, Nk, ratio)  # (Nk,2Nh)
    return U, V


def cca(test_data, template, Nk=1, ratio=None):
    """Using CCA to compute decision coefficients.

    Args:
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        template (ndarray): (n_events, 2*n_harmonics, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

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
                U, V = cca_compute(temp, template[nem,...], Nk, ratio)
                rou[ner,nte,nem] = pearson_corr(U@temp, V@template[nem,...])
    return rou


# %% Extended CCA | eCCA
def ecca_compute(Xmean, Y, data, Nk=1, ratio=None):
    """CCA with individual calibration data.

    Args:
        Xmean (ndarray): (n_chans, n_points). Trial-averaged SSVEP template.
        Y (ndarray): (2*n_harmonics, n_points). Sinusoidal SSVEP template.
        data (ndarray): (n_chans, n_points). Test-trial EEG.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (float): feature coefficient.
    """
    # correlation coefficient from CCA process
    U1, V1 = cca(data, Y, Nk, ratio)
    r1 = pearson_corr(U1@data, V1@Y)

    # correlation coefficients between single-trial EEG and SSVEP templates
    U2, V2 = cca(data, Xmean, Nk, ratio)
    r2 = pearson_corr(U2@data, U2@Xmean)

    r3 = pearson_corr(U1@data, U1@Xmean)

    U3, _ = cca(Xmean, Y, Nk, ratio)
    r4 = pearson_corr(U3@data, U3@Xmean)

    # similarity between filters corresponding to single-trial EEG and SSVEP templates
    r5 = pearson_corr(U2@Xmean, V2@Xmean)

    # combined features
    rou = combine_feature([r1, r2, r3, r4, r5])
    return rou


def ecca(train_data, test_data, Y, Nk=1, ratio=None):
    """Use eCCA to compute decision coefficient.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Y (ndarray): (n_events, 2*n_harmonics, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

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
                rou[ner,nte,nem] = ecca_compute(Xmean, Y[nem,...], temp, Nk, ratio)
    return rou


def ecca_sp_compute(Xmean, Y, data):
    """CCA with individual calibration data. | More coefficient.

    Args:
        Xmean (ndarray): (n_chans, n_points). Trial-averaged SSVEP template.
        Y (ndarray): (2*n_harmonics, n_points). Sinusoidal SSVEP template.
        data (ndarray): (n_chans, n_points). Test-trial EEG.

    Returns:
        rou (float): feature coefficient.
    """
    U1, V1 = cca(test_data=data, template=Y)
    U2, V2 = cca(test_data=data, template=Xmean)
    U3, V3 = cca(test_data=Xmean, template=Y)


# %% Multi-stimulus eCCA | ms-eCCA
# msCCA is only part of ms-eCCA. Personally, i dont like this design
def mscca_compute(X, Y, Nk=1, ratio=None):
    """Multi-stimulus CCA.

    Args:
        X (ndarray): (n_events, n_chans, n_points). Averaged template.
        Y (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        w (ndarray): (Nk, n_chans). Common spatial filter.
    """
    # GEPs' conditions
    Czz = einsum('ecp,ehp->ch', X,X)  # (Nc,Nc)
    Cyy = einsum('ecp,ehp->ch', Y,Y)  # (2Nh,2Nh)
    Czy = einsum('ecp,ehp->ch', X,Y)  # (Nc,2Nh)
    Cyz = Czy.T  # (2Nh,Nc)
    A = LA.solve(Czz,Czy) @ LA.solve(Cyy,Cyz)  # AU = lambda*U

    # ms-CCA part
    w = solve_ep(A, Nk, ratio)  # (Nk,Nc)
    return w


def mscca(train_data, Y, test_data, Nk=1, ratio=None):
    """Use msCCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        Y (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

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
    w = mscca_compute(train_mean, Y, Nk, ratio)  # (1,Nc)
    for ne in range(n_events):
        model[ne,:] = w @ train_mean[ne,...]  # (1,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = pearson_corr(w@temp, model[[nem],:])
    return rou


def msecca_compute(X, Y, events_group=None, Nk=1, ratio=None):
    """Multi-stimulus eCCA.

    Args:
        X (ndarray): (n_events, n_chans, n_points). Averaged template.
        Y (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        events_group (dict): {'events':[start index,end index]}
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        U (ndarray): (n_events, Nk, n_chans). Spatial filters for EEG.
        V (ndarray): (n_events, Nk, 2*n_harmonics). Spatial filters for templates.
    """
    # basic information
    n_events = X.shape[0]
    n_chans = X.shape[2]
    n_harmonics_2 = Y.shape[1]  # 2Nh

    # GEPs' conditions
    Czz_total = einsum('ecp,ehp->ech', X,X)  # (Ne,Nc,Nc)
    Cyy_total = einsum('ecp,ehp->ech', Y,Y)  # (Ne,2Nh,2Nh)
    Czy_total = einsum('ecp,ehp->ech', X,Y)  # (Ne,Nc,2Nh)

    # GEPs with merged data
    U = np.zeros((n_events, Nk, n_chans))  # (Ne,Nk,Nc)
    V = np.zeros((n_events, Nk, n_harmonics_2))  # (Ne,Nk,2*Nh)
    for ne in range(n_events):
        # GEPs' conditions
        idx = str(ne)
        st, ed = events_group[idx][0], events_group[idx][1]
        Czz = np.sum(Czz_total[st:ed], axis=0)  # (Nc,Nc)
        Cyy = np.sum(Cyy_total[st:ed], axis=0)  # (2Nh,2Nh)
        Czy = np.sum(Czy_total[st:ed], axis=0)  # (Nc,2Nh)
        Cyz = Czy.T  # (2Nh,Nc)
        A = LA.solve(Czz,Czy) @ LA.solve(Cyy,Cyz)  # AU = lambda*U
        B = LA.solve(Cyy,Cyz) @ LA.solve(Czz,Czy)  # AU = lambda*U

        # solve GEPs
        U[ne,...] = solve_ep(A, Nk, ratio)  # EEG part: (Nk,Nc)
        V[ne,...] = solve_ep(B, Nk, ratio)  # template part: (Nk,2Nh)
    return U, V


def msecca(train_data, Y, test_data, d, Nk=1, ratio=None, **kwargs):
    """Using ms-eCCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        Y (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        d (int): The range of events to be merged.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]
    try:
        events_group = kwargs['events_group']
    except KeyError:
        events_group = augmented_events(n_events, d)

    # training models & filters
    U,V = msecca_compute(train_data, Y, events_group, Nk, ratio)  # U: (Ne,Nk,Nc), V (Ne,Nk,2Nh)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model_eeg = einsum('ekc,ecp->ekp', U,train_mean)  # (Ne,Nk,Np)
    model_template = einsum('ekh,ehp->ekp', V,Y)  # (Ne,Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                r1 = pearson_corr(U[nem,...]@temp, model_template[nem,...])
                r2 = pearson_corr(U[nem,...]@temp, model_eeg[nem,...])
                rou[ner,nte,nem] = combine_feature([r1, r2])
    return rou


# Cross-subject spatial filter transfer method | CSSFT
def cssft_compute():
    pass


def cssft():
    pass