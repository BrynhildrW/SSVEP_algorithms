# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Task-related component analysis (TRCA) series.
    (1) (e)TRCA: 
            DOI: 
    (2) ms-(e)TRCA: 
            DOI: 
    (3) (e)TRCA-R:
            DOI:
    (4) sc-(e)TRCA:
            DOI:
    (5) gTRCA:
            DOI:
    (6) xTRCA:
            DOI:
    (7) LA-TRCA:
            DOI:
    (8) TDCA:
            DOI:

update: 2022/11/11

"""

# %% basic modules
from utils import *


# %% 
# (ensemble) TRCA | (e)TRCA
def trca_compute(X, Nk=1, ratio=None):
    """Task-related component analysis.
    n_events is a non-ignorable parameter, could be 1 if necessary.

    Args:
        X (ndarray): (n_events, n_train, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        w (ndarray): (n_events, Nk, n_chans). Spatial filters.
    """
    # basic information
    n_events = X.shape[0]
    n_chans = X.shape[2]

    # Q: covariance of original data | (Ne,Nc,Nc)
    Q = einsum('etcp,ethp->ech', X,X)

    # S: covariance of averaged data | (Ne,Nc,Nc)
    Xsum = np.sum(X, axis=1)  # (Ne,Nc,Np)
    S = einsum('ecp,ehp->ech', Xsum,Xsum)

    # GEPs
    w = np.zeros((n_events, Nk, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        w[ne,...] = solve_gep(S[ne,...], Q[ne,...], Nk, ratio)
    return w


def etrca(train_data, test_data, Nk=1, ratio=None):
    """Using TRCA & eTRCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
        erou (ndarray): (n_events, n_test, n_events).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # training models & filters
    w = trca_compute(train_data, Nk, ratio)  # (Ne,Nc)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model = einsum('ekc,ecp->ekp', w,train_mean)  # (Ne,Np)
    emodel = einsum('ekc,vcp->vekp', w,train_mean)  # (Ne real,Ne model,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = pearson_corr(w[[nem],...]@temp, model[[nem],...])
                erou[ner,nte,nem] = pearson_corr(w@temp, emodel[nem,...])
    return rou, erou


def etrca_sp(train_data, test_data, Nk=1):
    """Special version for etrca() | with signature check
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
        Nk (int): Number of eigenvectors picked as filters. Defaults to be 1.

    Returns:
        rou (ndarray): (2, n_test, 2).
        erou (ndarray): (2, n_test, 2).
    """
    # basic information
    n_test = test_data.shape[1]

    # training models & filters
    w = trca_compute(train_data, Nk)  # (Ne,Nc)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model = einsum('ekc,ecp->ekp', w,train_mean)  # (Ne,Np)
    emodel = einsum('ekc,vcp->vekp', w,train_mean)  # (Ne real,Ne model,Np)

    # sign check for filters & models
    coef = pearson_corr(model[[0],:], model[[1],:])
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
                rou[ner,nte,nem] = pearson_corr(w[[nem],:]@temp, model[[nem],:])
                erou[ner,nte,nem] = pearson_corr(w@temp, emodel[nem,...])
    return rou, erou


# multi-stimulus (e)TRCA | ms-(e)TRCA
def mstrca_compute(X, Nk=1, events_group=None):
    """Multi-stimulus TRCA.

    Args:
        X (ndarray): (n_events, n_train, n_chans, n_points). Training dataset.
        Nk (int): Number of eigenvectors picked as filters.
        events_group (dict): {'events':[start index,end index]}

    Returns:
        w (ndarray): (n_events, Nk, n_chans). Spatial filters.
    """
    # basic information
    n_events = X.shape[0]
    n_chans = X.shape[2]

    # Q: covariance of original data for each event | (Ne,Nc,Nc)
    total_Q = einsum('etcp,ethp->ech', X,X)

    # S: covariance of averaged data | (Ne,Nc,Nc)
    Xsum = np.sum(X, axis=1)
    total_S = einsum('ecp,ehp->ech', Xsum, Xsum)

    # GEPs with merged data
    w = np.zeros((n_events, Nk, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        idx = str(ne)
        st, ed = events_group[idx][0], events_group[idx][1]
        temp_Q = np.sum(total_Q[st:ed], axis=0)  # (Nc,Nc)
        temp_S = np.sum(total_S[st:ed], axis=0)  # (Nc,Nc)
        w[ne,...] = solve_gep(temp_S, temp_Q, Nk)
    return w


def msetrca(train_data, test_data, d, Nk, **kwargs):
    """Using ms-(e)TRCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        d (int): The range of events to be merged.
        Nk (int): Number of eigenvectors picked as filters.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
        erou (ndarray): (n_events, n_test, n_events).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]
    try:
        events_group = kwargs['events_group']
    except KeyError:
        events_group = augmented_events(n_events, d)

    # training models & filters
    w = mstrca_compute(train_data, Nk, events_group)  # (Ne,Nc)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model = einsum('ekc,ecp->ekp', w,train_mean)  # (Ne,Nk,Np)
    emodel = einsum('ekc,vcp->vekp', w,train_mean)  # (Ne real,Ne model,Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = pearson_corr(w[[nem],...]@temp, model[[nem],...])
                erou[ner,nte,nem] = pearson_corr(w@temp, emodel[nem,...])
    return rou, erou


# (e)TRCA-R
def trcar_compute(X, Y, Nk=1):
    """(e)TRCA-R.

    Args:
        X (ndarray): (n_events, n_train, n_chans, n_points). Training dataset
        Y (ndarray): (n_events, 2*n_harmonics, n_points). Sine-cosine template.
        Nk (int): Number of eigenvectors picked as filters. Defaults to be 1.

    Returns:
        w (ndarray): (n_events, Nk, n_chans). Spatial filters.
    """
    # preprocess
    X = zero_mean(X)

    # basic information
    n_events = X.shape[0]
    n_chans = X.shape[2]

    # Q: variance of original data | (Ne,Nc,Nc)
    Q = einsum('etcp,ethp->ech', X,X)

    # S: covariance of averaged data | (Ne,Nc,Nc)
    Xsum = np.sum(X, axis=1)  # (Ne,Nc,Np)
    pX = einsum('ecp,ehp->ech', Xsum,Y)  # (Ne,Nc,2Nh)
    S = einsum('ech,eah->eca', pX,pX)  # (Ne,Nc,Nc)

    # GEPs
    w = np.zeros((n_events, Nk, n_chans))  # (Ne,Nc)
    for ne in range(n_events):
        w[ne,...] = solve_gep(S[ne,...], Q[ne,...], Nk)
    return w


def etrcar(train_data, sine_template, test_data):
    """Use (e)TRCA-R to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
        erou (ndarray): (n_events, n_test, n_events).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # training models & filters
    w = trcar_compute(train_data, sine_template)  # (Ne,Nc)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model = einsum('ekc,ecp->ekp', w,train_mean)  # (Ne,Nk,Np)
    emodel = einsum('ekc,vcp->vekp', w,train_mean)  # (Ne real,Ne model,Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = pearson_corr(w[[nem],:]@temp, model[[nem],:])
                erou[ner,nte,nem] = pearson_corr(w@temp, emodel[nem,...])
    return rou, erou


# similarity constrained (e)TRCA | sc-(e)TRCA
def sctrca_compute(X, Y, Nk=1):
    """Similarity-constrained TRCA.

    Args:
        X (ndarray): (n_train, n_chans, n_points). Training dataset.
        Y (ndarray): (2*n_harmonics, n_points). Sine-cosine template.
        Nk (int): Number of eigenvectors picked as filters. Defaults to be 1.

    Return:
        U (ndarray): (1, n_chans). Filters for EEG data.
        V (ndarray): (1, 2*n_harmonics). Filters for artificial template.
    """
    # basic information
    n_train = X.shape[0]
    n_chans = X.shape[1]  # Nc
    n_points = X.shape[-1]  # Np
    n_harmonics = int(Y.shape[0]/2)  # 2*Nh

    # block covariance matrix S: [[S11,S12],[S21,S22]]
    Xmean = X.mean(axis=0)  # (Nc,Np)

    # S11: inter-trial covariance
    S11 = Xmean @ Xmean.T  # (Nc,Nc)
    coef = n_train/((n_train-1)*n_points)
    S11 *= coef # could be ignored?

    # S12 & S21: covariance between EEG and sine-cosine template
    S12 = Xmean @ Y.T  # (Nc,2*Nh)
    S12 /= n_points
    S21 = S12.T

    # S22: covariance within sine-cosine template
    S22 = Y @ Y.T  # (2*Nh,2*Nh)
    S22 /= n_points

    S = np.eye((n_chans+2*n_harmonics))  # (Nc+2Nh,Nc+2Nh)
    S[:n_chans, :n_chans] = S11
    S[:n_chans, n_chans:] = S12
    S[n_chans:, :n_chans] = S21
    S[n_chans:, n_chans:] = S22

    # block variance matrix Q: blkdiag(Q1,Q2)
    Q = np.zeros_like(S)

    # Q1: variace of EEG
    Q1 = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for ntr in range(n_train):
        Q1 += X[ntr,...] @ X[ntr,...].T
    Q1 /= (n_train*n_points)
    Q[:n_chans,:n_chans] = Q1

    # Q2: variance of sine-cosine template
    Q2 = S22
    Q[n_chans:,n_chans:] = Q2

    # GEPs
    w = solve_gep(S, Q, Nk)  # (1,Nc+2Nh)
    return w[:,:n_chans], w[:,n_chans:]  # u,v


def sctrca_compute_V2(X, Y, Nk=1):
    """Another version of sctrca_compute(). (slower for np.concatenate())

    Args:
        X (ndarray): (n_events, n_train, n_chans, n_points). Training dataset.
        Y (ndarray): (n_events, 2*n_harmonics, n_points). Sine-cosine template.
        Nk (int): Number of eigenvectors picked as filters. Defaults to be 1.

    Return:
        U (ndarray): (n_events, n_chans). Filters for EEG data.
        V (ndarray): (n_events, 2*n_harmonics). Filters for artificial template.
    """
    # basic information
    n_events = X.shape[0]
    n_train = X.shape[1]
    n_chans = X.shape[2]  # Nc
    n_harmonics = int(Y.shape[1]/2)  # Nh

    # block covariance matrix S: covariance of [X.T,Y.T].T
    Xmean = X.mean(axis=1)  # (Ne,Nc,Np)
    Xhat = np.concatenate((Xmean,Y), axis=1)  # (Ne,Nc+2Nh,Np)
    S = einsum('ecp,ehp->ech', Xhat,Xhat)  # (Ne,Nc+2Nh,Nc+2Nh)

    # block variance matrix Q: blkdiag(Q1,Q2)
    Q = np.zeros_like(S)

    # u @ Q1 @ u^T: variace of filtered EEG
    Q[:,:n_chans,:n_chans] = einsum('etcp,ethp->ech', X,X)/n_train

    # v @ Q2 @ v^T: variance of filtered sine-cosine template
    Q[:,n_chans:,n_chans:] = S[:,-2*n_harmonics:,-2*n_harmonics:]

    # GEPs
    w = np.zeros((n_events, n_chans+2*n_harmonics))  # (Ne,Nc+2Nh)
    for ne in range(n_events):
        w[ne,...] = solve_gep(S[ne,...], Q[ne,...], Nk)
    return w[...,:n_chans].squeeze(), w[...,n_chans:].squeeze()  # u,v


def sctrca_compute_sp(X, Y, Nk=1):
    """New version of sctrca_compute() | update Q

    Args:
        X (ndarray): (n_events, n_train, n_chans, n_points). Training dataset.
        Y (ndarray): (n_events, 2*n_harmonics, n_points). Sine-cosine template.
        Nk (int): Number of eigenvectors picked as filters. Defaults to be 1.

    Return:
        U (ndarray): (n_events, n_chans). Filters for EEG data.
        V (ndarray): (n_events, 2*n_harmonics). Filters for artificial template.
    """
    # basic information
    n_events = X.shape[0]
    n_train = X.shape[1]
    n_chans = X.shape[2]  # Nc
    n_harmonics = int(Y.shape[1]/2)  # Nh

    # block covariance matrix S: covariance of [Xmean.T,Y.T].T
    Xmean = X.mean(axis=1)  # (Ne,Nc,Np)
    Xhat = np.concatenate((Xmean,Y), axis=1)  # (Ne,Nc+2Nh,Np)
    S = einsum('ecp,ehp->ech', Xhat,Xhat)  # (Ne,Nc+2Nh,Nc+2Nh)

    # block covariance matrix S: covariance of [X.T,Y.T].T
    Q = np.zeros_like(S)

    # u @ Q11 @ u^T: variace of filtered EEG
    Q[:,:n_chans,:n_chans] = einsum('etcp,ethp->ech', X,X)/n_train

    # v @ Q2 @ v^T: variance of filtered sine-cosine template
    Q[:,n_chans:,n_chans:] = S[:,-2*n_harmonics:,-2*n_harmonics:]

    # u @ Q12 @ v^T: covariance of filtered EEG & filtered template
    # Q12 = Q21^T
    for ne in range(n_events):
        temp = einsum('tcp,hp->ch', X[ne],Y[ne])/n_train
        Q[ne,:n_chans,n_chans:] = temp
        Q[ne,n_chans:,:n_chans] = temp.T
    
    # GEPs
    w = np.zeros((n_events, n_chans+2*n_harmonics))  # (Ne,Nc+2Nh)
    for ne in range(n_events):
        w[ne,...] = solve_gep(S[ne,...], Q[ne,...], Nk)
    return w[...,:n_chans].squeeze(), w[...,n_chans:].squeeze()  # u,v


def scetrca(train_data, sine_template, test_data):
    """Use sc-(e)TRCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
        erou (ndarray): (n_events, n_test, n_events).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # training models & filters
    u,v = sctrca_compute_V2(train_data, sine_template)  # (Ne,Nc) & (Ne,2Nh)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model_eeg = einsum('ec,ecp->ep', u,train_mean)  # (Ne,Np)
    model_sin = einsum('eh,ehp->ep', v,sine_template)  # (Ne,Np)
    emodel_eeg = einsum('ec,vcp->vep', u,train_mean)  # (Ne real,Ne model,Np)
    emodel_sin = einsum('eh,ahp->aep', v,sine_template)  # (Ne real,Ne model,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                # sc-TRCA
                f_temp = u[[nem],:]@temp  # (1,Np)
                rou1 = corr_coef(f_temp, model_eeg[[nem],:])[0,0]
                rou2 = corr_coef(f_temp, model_sin[[nem],:])[0,0]
                rou[ner,nte,nem] = sign_sta(rou1) + sign_sta(rou2)
                
                # sc-eTRCA
                f_temp = u@temp  # (Ne,Np)
                rou3 = corr2_coef(f_temp, emodel_eeg[nem,...])
                rou4 = corr2_coef(f_temp, emodel_sin[nem,...])
                erou[ner,nte,nem] = sign_sta(rou3) + sign_sta(rou4)
    return rou, erou


def scetrca_md(train_data, sine_template, test_data, Nk):
    """Special version for scetrca() | multiple dimension, Nk>1

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
        erou (ndarray): (n_events, n_test, n_events).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # training models & filters
    u,v = sctrca_compute_V2(train_data, sine_template, Nk)  # (Ne,Nk,Nc) & (Ne,Nk,2Nh)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model_eeg = einsum('ekc,ecp->ekp', u,train_mean)  # (Ne,Nk,Np)
    model_sin = einsum('ekh,ehp->ekp', v,sine_template)  # (Ne,Nk,Np)
    emodel_eeg = einsum('ekc,vcp->vekp', u,train_mean)  # (Ne real,Ne model,Nk,Np)
    emodel_sin = einsum('ekh,ahp->aekp', v,sine_template)  # (Ne real,Ne model,Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                # sc-TRCA
                f_temp = u[nem,...]@temp  # (Nk,Np)
                rou1 = corr2_coef(f_temp, model_eeg[nem,...])
                rou2 = corr2_coef(f_temp, model_sin[nem,...])
                rou[ner,nte,nem] = sign_sta(rou1) + sign_sta(rou2)

                # sc-eTRCA
                f_temp = einsum('ekc,cp->ekp', u,temp)  # (Ne,Nk,Np)
                rou3 = corr2_coef(f_temp, emodel_eeg[nem,...])
                rou4 = corr2_coef(f_temp, emodel_sin[nem,...])
                erou[ner,nte,nem] = sign_sta(rou3) + sign_sta(rou4)
    return rou, erou


def scetrca_sp(train_data, sine_template, test_data):
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # training models & filters
    u,v = sctrca_compute_sp(train_data, sine_template)  # (Ne,Nc) & (Ne,2Nh)
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    model_eeg = einsum('ec,ecp->ep', u,train_mean)  # (Ne,Np)
    model_sin = einsum('eh,ehp->ep', v,sine_template)  # (Ne,Np)
    emodel_eeg = einsum('ec,vcp->vep', u,train_mean)  # (Ne real,Ne model,Np)
    emodel_sin = einsum('eh,ahp->aep', v,sine_template)  # (Ne real,Ne model,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    erou = np.zeros_like(rou)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                # sc-TRCA
                f_temp = u[[nem],:]@temp  # (1,Np)
                rou1 = corr_coef(f_temp, model_eeg[[nem],:])[0,0]
                rou2 = corr_coef(f_temp, model_sin[[nem],:])[0,0]
                rou[ner,nte,nem] = sign_sta(rou1) + sign_sta(rou2)
                
                # sc-eTRCA
                f_temp = u@temp  # (Ne,Np)
                rou3 = corr2_coef(f_temp, emodel_eeg[nem,...])
                rou4 = corr2_coef(f_temp, emodel_sin[nem,...])
                erou[ner,nte,nem] = sign_sta(rou3) + sign_sta(rou4)
    return rou, erou


# group TRCA | gTRCA



# cross-correlation TRCA | xTRCA



# latency-aligned TRCA | LA-TRCA



# task-discriminant component analysis | TDCA



# optimized TRCA | op-TRCA
