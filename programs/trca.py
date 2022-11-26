# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Task-related component analysis (TRCA) series.
    (1) (e)TRCA: https://ieeexplore.ieee.org/document/7904641/
            DOI: 10.1109/TBME.2017.2694818
    (2) ms-(e)TRCA: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
            DOI: 10.1088/1741-2552/ab2373
    (3) (e)TRCA-R: https://ieeexplore.ieee.org/document/9006809/
            DOI: 10.1109/TBME.2020.2975552
    (4) sc-(e)TRCA: https://iopscience.iop.org/article/10.1088/1741-2552/abfdfa
            DOI: 10.1088/1741-2552/abfdfa
    (5) CORRCA: 
            DOI:
    (6) gTRCA: 
            DOI:
    (7) xTRCA: 
            DOI:
    (8) LA-TRCA: 
            DOI:
    (9) TDCA: 
            DOI:

update: 2022/11/15

"""

# %% basic modules
from utils import *


# %% (1) (ensemble) TRCA | (e)TRCA
def trca_compute(train_data, avg_template, Nk=1, ratio=None):
    """Task-related component analysis.
    n_events is a non-ignorable parameter, could be 1 if necessary.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        avg_template (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        w (list of ndarray): n_events * (Nk, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*Nk, n_chans). Concatenated filter.
        ndim (list of int): 1st dimension of each spatial filter.
    """
    # basic information
    n_events = train_data.shape[0]
    n_chans = train_data.shape[2]

    # Q: covariance of original data | (Ne,Nc,Nc)
    Q = einsum('etcp,ethp->ech', train_data,train_data)

    # S: covariance of template | (Ne,Nc,Nc)
    # avg_template = np.sum(X, axis=1)  # (Ne,Nc,Np)
    S = einsum('ecp,ehp->ech', avg_template,avg_template)

    # GEPs
    w, ndim = [], []
    for ne in range(n_events): 
        spatial_filter = solve_gep(S[ne,...], Q[ne,...], Nk, ratio)  # (Nk,Nc)
        w.append(spatial_filter)
        ndim.append(spatial_filter.shape[0])  # Nk

    # avoid np.concatenate() to speed up
    w_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    start_idx = 0
    for ne,dims in enumerate(ndim):
        w_concat[start_idx:start_idx+dims] = w[ne]
        start_idx += dims
    return w, w_concat, ndim


def etrca(train_data, avg_template, test_data, Nk=1, ratio=None):
    """Using TRCA & eTRCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        avg_template (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events(real), n_test, n_events(model)).
        erou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # training models & filters
    w, w_concat, _ = trca_compute(train_data=train_data, avg_template=avg_template,
        Nk=Nk, ratio=ratio)  # list, (Ne*Nk,Nc), list
    model, emodel = [], []
    for ne in range(n_events):
        model.append(w[ne] @ avg_template[ne])  # (Nk,Np)
        emodel.append(w_concat @ avg_template[ne])  # (Ne(model)*Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne(real),Nt,Ne(model))
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = pearson_corr(w[nem]@temp, model[nem])
                erou[ner,nte,nem] = pearson_corr(w_concat@temp, emodel[nem])
    return rou, erou


# %% (2) multi-stimulus (e)TRCA | ms-(e)TRCA
def mstrca_compute(train_data, avg_template, events_group=None, Nk=1, ratio=None):
    """Multi-stimulus TRCA.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        avg_template (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        events_group (dict): {'events':[start index,end index]}.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        w (list of ndarray): n_events * (Nk, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*Nk, n_chans). Concatenated filter.
        ndim (list of int): 1st dimension of each spatial filter.
    """
    # basic information
    n_events = train_data.shape[0]
    n_chans = train_data.shape[2]

    # Q: covariance of original data for each event | (Ne,Nc,Nc)
    total_Q = einsum('etcp,ethp->ech', train_data,train_data)

    # S: covariance of averaged data | (Ne,Nc,Nc)
    # avg_template = np.sum(X, axis=1)
    total_S = einsum('ecp,ehp->ech', avg_template,avg_template)

    # GEPs with merged data
    w, ndim = [], []
    for ne in range(n_events):
        # GEPs' conditions
        idx = str(ne)
        st, ed = events_group[idx][0], events_group[idx][1]
        temp_Q = np.sum(total_Q[st:ed], axis=0)  # (Nc,Nc)
        temp_S = np.sum(total_S[st:ed], axis=0)  # (Nc,Nc)
        spatial_filter = solve_gep(temp_S, temp_Q, Nk, ratio)  # (Nk,Nc)
        w.append(spatial_filter)
        ndim.append(spatial_filter.shape[0])  # Nk
    w_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    start_idx = 0
    for ne, dims in enumerate(ndim):
        w_concat[start_idx:start_idx+dims] = w[ne]
        start_idx += dims
    return w, w_concat, ndim


def msetrca(train_data, avg_template, test_data, d, Nk=1, ratio=None, **kwargs):
    """Using ms-(e)TRCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        avg_template (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        d (int): The range of events to be merged.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events(real), n_test, n_events(model)).
        erou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]
    try:
        events_group = kwargs['events_group']
    except KeyError:
        events_group = augmented_events(n_events, d)

    # training models & filters
    w, w_concat, _ = mstrca_compute(train_data=train_data, avg_template=avg_template,
        events_group=events_group, Nk=Nk, ratio=ratio)  # list, (Ne*Nk,Nc), list
    model, emodel = [], []
    for ne in range(n_events):
        model.append(w[ne] @ avg_template[ne])  # (Nk,Np)
        emodel.append(w_concat @ avg_template[ne])  # (Ne(model)*Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = pearson_corr(w[nem]@temp, model[nem])
                erou[ner,nte,nem] = pearson_corr(w_concat@temp, emodel[nem])
    return rou, erou


# %% (e)TRCA-R
def trcar_compute(train_data, avg_template, sine_template, Nk=1, ratio=None):
    """(e)TRCA-R.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        avg_template (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        w (list of ndarray): n_events * (Nk, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*Nk, n_chans). Concatenated filter.
        ndim (list of int): 1st dimension of each spatial filter.
    """
    # basic information
    n_events = train_data.shape[0]
    n_chans = train_data.shape[2]

    # Q: variance of original data | (Ne,Nc,Nc)
    Q = einsum('etcp,ethp->ech', train_data,train_data)

    # S: covariance of averaged data | (Ne,Nc,Nc)
    # avg_template = np.sum(X, axis=1)  # (Ne,Nc,Np)
    pX = einsum('ecp,ehp->ech', avg_template,sine_template)  # (Ne,Nc,2Nh)
    S = einsum('ech,eah->eca', pX,pX)  # (Ne,Nc,Nc)

    # GEPs
    w, ndim = [], []
    for ne in range(n_events):
        spatial_filter = solve_gep(S[ne,...], Q[ne,...], Nk, ratio)  # (Nk,Nc)
        w.append(spatial_filter)
        ndim.append(spatial_filter.shape[0])    # Nk
    w_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    start_idx = 0
    for ne,dims in enumerate(ndim):
        w_concat[start_idx:start_idx+dims] = w[ne]
        start_idx += dims
    return w, w_concat, ndim


def etrcar(train_data, avg_template, sine_template, test_data, Nk=1, ratio=None):
    """Use (e)TRCA-R to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        avg_template (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events(real), n_test, n_events(model)).
        erou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # training models & filters
    w, w_concat, _ = trcar_compute(train_data=train_data, avg_template=avg_template,
        sine_template=sine_template, Nk=Nk, ratio=ratio)  # list, (Ne*Nk,Nc), list
    model, emodel = [], []
    for ne in range(n_events):
        model.append(w[ne] @ avg_template[ne])  # (Nk,Np)
        emodel.append(w_concat @ avg_template[ne])  # (Ne(model)*Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne(real),Nt,Ne(model))
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = pearson_corr(w[nem]@temp, model[nem])
                erou[ner,nte,nem] = pearson_corr(w_concat@temp, emodel[nem])
    return rou, erou


# %% similarity constrained (e)TRCA | sc-(e)TRCA
def sctrca_compute(train_data, concat_template, Nk=1, ratio=None):
    """Similarity-constrained TRCA.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        concat_template (ndarray): (n_events, n_chans+2*n_harmonics, n_points). Concatenated template.
            Trial-averaged data & Sinusoidal template.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return:
        w (list of ndarray): n_events * (Nk, n_chans+2*n_harmonics). Spatial filters.
        w_concat (ndarray): (n_events*Nk, n_chans+2*n_harmonics). Concatenated filter.
        ndim (list of int): 1st dimension of each spatial filter.
    """
    # basic information
    n_events = train_data.shape[0]
    n_train = train_data.shape[1]
    n_chans = train_data.shape[2]  # Nc
    n_2harmonics = concat_template.shape[1] - n_chans  # 2*Nh

    # block covariance matrix S: [[XX.T,XY.T],[YX.T,YY.T]]
    S = einsum('ecp,ehp->ech', concat_template,concat_template)  # (Ne,Nc+2Nh,Nc+2Nh)

    # block variance matrix Q: blkdiag(Q1,Q2)
    Q = np.zeros_like(S)

    # u @ Q1 @ u^T: variace of filtered EEG
    Q[:,:n_chans,:n_chans] = einsum('etcp,ethp->ech', train_data,train_data)/n_train  # (Ne,Nc,Nc)

    # v @ Q2 @ v^T: variance of filtered sine-cosine template
    Q[:,n_chans:,n_chans:] = S[:,-n_2harmonics:,-n_2harmonics:]  # (Ne,2Nh,2Nh)

    # GEPs
    w, ndim = [], []  # w=[u,v]
    for ne in range(n_events):
        spatial_filter = solve_gep(S[ne,...], Q[ne,...], Nk, ratio)
        w.append(spatial_filter)  # (Nk,Nc+2Nh)
        ndim.append(spatial_filter.shape[0])  # Nk
    w_concat = np.zeros((np.sum(ndim), n_chans+n_2harmonics))  # (Ne*Nk,Nc+2Nh)
    start_idx = 0
    for ne,dims in enumerate(ndim):
        w_concat[start_idx:start_idx+dims] = w[ne]
        start_idx += dims
    return w, w_concat, ndim


def scetrca(train_data, concat_template, test_data, Nk=1, ratio=None):
    """Use sc-(e)TRCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        concat_template (ndarray): (n_events, n_chans+2*n_harmonics, n_points). Concatenated template.
            Trial-averaged data & Sinusoidal template.
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events(real), n_test, n_events(model)).
        erou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]
    n_chans = train_data.shape[2]

    # training models & filters
    w, w_concat, _ = sctrca_compute(train_data=train_data, concat_template=concat_template,
        Nk=Nk, ratio=ratio)
    model_eeg, model_sin, emodel_eeg, emodel_sin   = [], [], [], []
    # model_mix, emodel_mix = [], []
    for ne in range(n_events):
        model_eeg.append(w[ne][:,:n_chans] @ concat_template[ne,:n_chans,:])  # u @ Xmean | (Nk,Np)
        emodel_eeg.append(w_concat[:,:n_chans] @ concat_template[ne,:n_chans,:])  # (Ne(model)*Nk,Np)
        model_sin.append(w[ne][:,n_chans:] @ concat_template[ne,n_chans:,:])  # v @ Y | (Nk,Np)
        emodel_sin.append(w_concat[:,n_chans:] @ concat_template[ne,n_chans:,:])  # (Ne(model)*Nk,Np)
        # model_mix.append(w[ne] @ concat_template[ne])  # [u,v] @ [Xmean.T,Y.T].T | (Nk,Np)
        # emodel_mix.append(w_concat @ concat_template[ne])  # (Ne(model)*Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                # sc-TRCA
                f_temp = w[nem][:,:n_chans] @ temp  # (Nk,Np)
                rou1 = pearson_corr(f_temp, model_eeg[nem])
                rou2 = pearson_corr(f_temp, model_sin[nem])
                # rou3 = pearson_corr(f_temp, model_mix[nem])
                # rou[ner,nte,nem] = combine_feature([rou1, rou2, rou3])
                rou[ner,nte,nem] = combine_feature([rou1, rou2])

                # sc-eTRCA
                f_temp = w_concat[:,:n_chans] @ temp  # (Ne*Nk,Np)
                erou1 = pearson_corr(f_temp, emodel_eeg[nem])
                erou2 = pearson_corr(f_temp, emodel_sin[nem])
                # erou3 = pearson_corr(f_temp, emodel_mix[nem])
                # erou[ner,nte,nem] = combine_feature([erou1, erou2, erou3])
                erou[ner,nte,nem] = combine_feature([erou1, erou2])
    return rou, erou


# group TRCA | gTRCA



# cross-correlation TRCA | xTRCA



# latency-aligned TRCA | LA-TRCA



# task-discriminant component analysis | TDCA



# optimized TRCA | op-TRCA
