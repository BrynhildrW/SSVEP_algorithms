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
from special import dsp_compute

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
def aug_2(data, projection, m, mode='train'):
    """Construct secondary augmented data.

    Args:
        data (ndarray): (n_chans, n_points+m or n_points).
            m must be larger than n_points while mode is 'train'.
        projection (ndarray): (n_points, n_points). Y.T@Y
        m (int): Extra data length.
        mode (str, optional): 'train' or 'test'.

    Returns:
        data_aug2 (ndarray): ((m+1)*n_chans, 2*n_points).
    """
    # basic information
    n_chans = data.shape[0]  # Nc
    n_points = projection.shape[0]  # Np
    # n_2harmonics = projection.shape[1]  # 2*Nh

    # secondary augmented data
    data_aug2 = np.zeros(((m+1)*n_chans, 2*n_points))  # ((m+1)*Nc,Np+2Nh)
    if mode == 'train':
        for nm in range(m+1):
            sp, ep = nm*n_chans, (nm+1)*n_chans
            data_aug2[sp:ep,:n_points] = data[:,nm:n_points+nm]  # augmented data
            data_aug2[sp:ep,n_points:] = data_aug2[sp:ep,:n_points] @ projection
    elif mode == 'test':
        for nm in range(m+1):
            sp, ep = nm*n_chans, (nm+1)*n_chans
            data_aug2[sp:ep,:n_points-nm] = data[:,nm:n_points]
            data_aug2[sp:ep,n_points:] = data_aug2[sp:ep,:n_points] @ projection
    return data_aug2


def tdca_compute(train_data, projection, m, Nk=1, ratio=None):
    """Task-discriminant component analysis.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points+m).
        projection (ndarray): (n_events, n_points, n_points).
        m (int): Extra data length.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        w (ndarray): (Nk,(m+1)*n_chans). Spatial filter.
        aug_template (ndarray): (n_events,Nk,2*n_points). Augmented template.
    """
    # basic information
    n_events = train_data.shape[0]
    n_train = train_data.shape[1]
    n_chans = train_data.shape[2]
    n_points = projection.shape[1]

    # create secondary augmented data
    train_data_aug2 = np.zeros((n_events, n_train, (m+1)*n_chans, 2*n_points))
    for ne in range(n_events):
        for ntr in range(n_train):
            train_data_aug2[ne,ntr,...] = aug_2(data=train_data[ne,ntr,...],
                                                projection=projection[ne], m=m)

    # DSP filters based on augmented data
    class_center = train_data_aug2.mean(axis=1)  # (Ne,(m+1)*Nc,2*Np)
    w = dsp_compute(train_data=train_data_aug2, class_center=class_center,
                    Nk=Nk, ratio=ratio)  # (Nk,(m+1)*Nc)
    aug_template = einsum('kc,ecp->ekp', w,class_center)
    return w, aug_template


def tdca(train_data, test_data, projection, m, Nk=1, ratio=None):
    """Using TDCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points+m).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        projection (ndarray): (n_events, n_points, 2*n_harmonics).
        m (int): Extra data length.
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
    w, aug_template = tdca_compute(train_data=train_data, projection=projection,
                                   m=m, Nk=Nk, ratio=ratio)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                temp_aug2 = aug_2(data=temp, projection=projection[nem], m=m, mode='test')
                rou[ner,nte,nem] = pearson_corr(w@temp_aug2, aug_template[nem])
    return rou


# optimized TRCA | op-TRCA



# %% Filter-bank TRCA series | FB-
def fb_etrca(train_data, avg_template, test_data, Nk=1, ratio=None):
    """(e)TRCA algorithms with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        avg_template (ndarray): (n_bands, n_events, n_chans, n_points). Trial-averaged data.
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events(real), n_test, n_events(model)).
        erou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_bands = test_data.shape[0]

    # multiple (e)TRCA classification
    rou, erou = [], []
    for nb in range(n_bands):
        temp_rou, temp_erou = etrca(train_data=train_data[nb], avg_template=avg_template[nb],
            test_data=test_data[nb], Nk=Nk, ratio=ratio)
        rou.append(temp_rou)
        erou.append(temp_erou)
    return combine_fb_feature(rou), combine_fb_feature(erou)


def fb_msetrca(train_data, avg_template, test_data, d, Nk=1, ratio=None, **kwargs):
    """ms-(e)TRCA algorithms with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        avg_template (ndarray): (n_bands, n_events, n_chans, n_points). Trial-averaged data.
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
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
    n_bands = test_data.shape[0]
    n_events = train_data.shape[1]
    try:
        events_group = kwargs['events_group']
    except KeyError:
        events_group = augmented_events(n_events, d)

    # multiple ms-(e)TRCA classification
    rou, erou = [], []
    for nb in range(n_bands):
        temp_rou, temp_erou = msetrca(train_data=train_data[nb], avg_template=avg_template[nb],
            test_data=test_data[nb], d=d, Nk=Nk, ratio=ratio, events_group=events_group)
        rou.append(temp_rou)
        erou.append(temp_erou)
    return combine_fb_feature(rou), combine_fb_feature(erou)


def fb_etrcar(train_data, avg_template, sine_template, test_data, Nk=1, ratio=None):
    """(e)TRCA-R algorithms with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        avg_template (ndarray): (n_bands, n_events, n_chans, n_points). Trial-averaged data.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events(real), n_test, n_events(model)).
        erou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_bands = test_data.shape[0]

    # multiple (e)TRCA-R classification
    rou, erou = [], []
    for nb in range(n_bands):
        temp_rou, temp_erou = etrcar(train_data=train_data[nb], avg_template=avg_template[nb],
            sine_template=sine_template, test_data=test_data[nb], Nk=Nk, ratio=ratio)
        rou.append(temp_rou)
        erou.append(temp_erou)
    return combine_fb_feature(temp_rou), combine_fb_feature(temp_erou)


def fb_scetrca(train_data, concat_template, test_data, Nk=1, ratio=None):
    """sc-(e)TRCA algorithms with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        concat_template (ndarray): (n_bands, n_events, n_chans+2*n_harmonics, n_points). Concatenated template.
            Trial-averaged data & Sinusoidal template.
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events(real), n_test, n_events(model)).
        erou (ndarray): (n_events(real), n_test, n_events(model)).
    """
    # basic information
    n_bands = test_data.shape[0]

    # multiple sc-(e)TRCA classification
    rou, erou = [], []
    for nb in range(n_bands):
        temp_rou, temp_erou = scetrca(train_data=train_data[nb], concat_template=concat_template[nb],
            test_data=test_data[nb], Nk=Nk, ratio=ratio)
        rou.append(temp_rou)
        erou.append(temp_erou)
    return combine_fb_feature(temp_rou), combine_fb_feature(temp_erou)


def fb_tdca(train_data, test_data, projection, m, Nk=1, ratio=None):
    """TDCA algorithms with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points+m).
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        projection (ndarray): (n_events, n_points, 2*n_harmonics).
        m (int): Extra data length.
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
        temp_rou = tdca(train_data=train_data[nb], test_data=test_data[nb],
            projection=projection, m=m, Nk=Nk, ratio=ratio)
        rou.append(temp_rou)
    return combine_fb_feature(rou)
