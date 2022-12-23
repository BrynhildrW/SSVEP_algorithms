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
    (9) TDCA: https://ieeexplore.ieee.org/document/9541393/
            DOI: 10.1109/TNSRE.2021.3114340

update: 2022/11/15

"""

# %% basic modules
import utils
from special import dsp_compute

import numpy as np


# %% (1) (ensemble) TRCA | (e)TRCA
def trca_compute(X_train, y_train, train_info, n_components=1, ratio=None):
    """Task-related component analysis (TRCA).

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        train_info (dict): {'n_events':int,
                            'event_type':ndarray (n_events,),
                            'n_train':ndarray (n_events,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int, optional): Number of eigenvectors picked as filters.
            Defaults to 1. Set to 'None' if ratio is not 'None'.
        ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not 'None'.

    Returns: | all contained in a tuple
        Q (ndarray): (n_events, n_chans, n_chans). Covariance of original data.
        S (ndarray): (n_events, n_chans, n_chans). Covariance of template data.
        w (list of ndarray): n_events*(n_components, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*n_components(total_components), n_chans). Concatenated filter.
        template (ndarray): n_events*(n_components, n_points). TRCA templates.
        ensemble_template (ndarray): (n_events, total_components, n_points). eTRCA templates.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool

    # S: covariance of template
    S = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    class_center = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne in range(n_events):
        class_center[ne] = X_train[y_train==ne].mean(axis=0)  # (Nc,Np)
        S[ne] = class_center[ne] @ class_center[ne].T
    # S = np.einsum('ecp,ehp->ech', class_center,class_center) | clearer but slower

    # Q: covariance of original data
    Q = np.zeros_like(S)  # (Ne,Nc,Nc)
    for ne in range(n_events):
        temp = X_train[y_train==ne]  # (Nt,Nc,Np)
        for ntr in range(n_train[ne]):
            Q[ne] += temp[ntr] @ temp[ntr].T
    # Q = np.einsum('etcp,ethp->ech', train_data,train_data) | clearer but slower

    # GEPs | train spatial filters
    w, ndim = [], []
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(
            A=S[ne],
            B=Q[ne],
            n_components=n_components,
            ratio=ratio
        )
        ndim.append(spatial_filter.shape[0])  # n_components, Nk
        w.append(spatial_filter)  # (Nk,Nc)
    w_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    start_idx = 0
    for ne,dims in enumerate(ndim):
        w_concat[start_idx:start_idx+dims] = w[ne]
        start_idx += dims

    # signal templates
    template = []  # Ne*(Nk,Np)
    ensemble_template = np.zeros((n_events, w_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            template.append(w[ne] @ class_center[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ensemble_template[ne] = w_concat @ class_center[ne]  # (Ne*Nk,Np)
    model = (
        Q, S,
        w, w_concat,
        template, ensemble_template
    )
    return model


class TRCA(object):
    def __init__(self, standard=True, ensemble=True, n_components=1, ratio=None):
        """Config model dimension.

        Args:
            standard (bool, optional): Standard TRCA model. Defaults to True.
            ensemble (bool, optional): Ensemble TRCA model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None' when n_components is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio
        self.standard = standard
        self.ensemble = ensemble


    def fit(self, X_train, y_train):
        """Train (e)TRCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        n_events = len(event_type)
        self.train_info = {'event_type':event_type,
                           'n_events':n_events,
                           'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
                           'n_chans':self.X_train.shape[-2],
                           'n_points':self.X_train.shape[-1],
                           'standard':self.standard,
                           'ensemble':self.ensemble}

        # train TRCA models & templates
        results = trca_compute(
            X_train=self.X_train,
            y_train = self.y_train,
            train_info = self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Q, self.S = results[0], results[1]
        self.w, self.w_concat = results[2], results[3]
        self.template, self.ensemble_template = results[4], results[5]
        return self


    def predict(self, X_test, y_test):
        """Using (e)TRCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients of TRCA.
                Not empty when self.standard is True.
            y_standard (ndarray): (test_trials,). Predict labels of TRCA.
            erou (ndarray): (test_trials, n_events). Decision coefficients of eTRCA.
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (test_trials,). Predict labels of eTRCA.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.erou = np.zeros_like(self.rou)
        self.y_standard = np.empty((n_test))
        self.y_ensemble = np.empty_like(self.y_standard)
        if self.standard:
            for nte in range(n_test):
                for nem in range(n_events):
                    self.rou[nte,nem] = utils.pearson_corr(
                        X=self.w[nem] @ X_test[nte],
                        Y=self.template[nem]
                    )
                self.y_standard[nte] = np.argmax(self.rou[nte,:])
        if self.ensemble:
            for nte in range(n_test):
                for nem in range(n_events):
                    self.erou[nte,nem] = utils.pearson_corr(
                        X=self.w_concat @ X_test[nte],
                        Y=self.ensemble_template[nem]
                    )
                self.y_ensemble[nte] = np.argmax(self.erou[nte,:])
        return self.rou, self.y_standard, self.erou, self.y_ensemble


# %% (2) multi-stimulus (e)TRCA | ms-(e)TRCA
def mstrca_compute(X_train, y_train, train_info, n_components=1, ratio=None):
    """Multi-stimulus TRCA (ms-TRCA).

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        train_info (dict): {'event_type':ndarray (n_events,),
                            'n_events':int,
                            'n_train':ndarray (n_events,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True,
                            'events_group':{'event_id':[start index,end index]}}
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns: | all contained in a tuple
        Q (ndarray): (n_events, n_chans, n_chans). Covariance of original data.
        S (ndarray): (n_events, n_chans, n_chans). Covariance of template data.
        w (list of ndarray): n_events*(n_components, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*n_components, n_chans). Concatenated filter.
        template (ndarray): n_events*(n_components, n_points). ms-TRCA templates.
        ensemble_template (ndarray): (n_events, total_components, n_points). ms-eTRCA templates.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool
    events_group = train_info['events_group']  # dict

    # S: covariance of template | same with TRCA
    total_S = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    class_center = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne in range(n_events):
        class_center[ne] = X_train[y_train==ne].mean(axis=0)  # (Nc,Np)
        total_S[ne] = class_center[ne] @ class_center[ne].T

    # Q: covariance of original data | same with TRCA
    total_Q = np.zeros_like(total_S)  # (Ne,Nc,Nc)
    for ne in range(n_events):
        temp = X_train[y_train==ne]  # (Nt,Nc,Np)
        for ntr in range(n_train[ne]):
            total_Q[ne] += temp[ntr] @ temp[ntr].T

    # GEPs with merged data
    w, ndim = [], []
    for ne in range(n_events):
        # GEPs' conditions
        idx = str(ne)
        st, ed = events_group[idx][0], events_group[idx][1]
        temp_Q = np.sum(total_Q[st:ed], axis=0)  # (Nc,Nc)
        temp_S = np.sum(total_S[st:ed], axis=0)  # (Nc,Nc)
        spatial_filter = utils.solve_gep(
            A=temp_S,
            B=temp_Q,
            n_components=n_components,
            ratio=ratio
        )
        ndim.append(spatial_filter.shape[0])  # Nk
        w.append(spatial_filter)  # (Nk,Nc)
    w_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    start_idx = 0
    for ne, dims in enumerate(ndim):
        w_concat[start_idx:start_idx+dims] = w[ne]
        start_idx += dims

    # signal templates
    template = []  # Ne*(Nk,Np)
    ensemble_template = np.zeros((n_events, w_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            template.append(w[ne] @ class_center[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ensemble_template[ne] = w_concat @ class_center[ne]  # (Ne*Nk,Np)
    model = (
        total_Q, total_S,
        w, w_concat,
        template, ensemble_template
    )
    return model


class MS_TRCA(TRCA):
    def fit(self, X_train, y_train, events_group=None, d=None):
        """Train ms-(e)TRCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            events_group (dict): {'event_id':[start index,end index]}
            d (int): The range of events to be merged.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        if events_group:  # given range
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(len(event_type), d)
        self.train_info = {'event_type':event_type,
                           'n_events':len(event_type),
                           'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
                           'n_chans':self.X_train.shape[-2],
                           'n_points':self.X_train.shape[-1],
                           'standard':self.standard,
                           'ensemble':self.ensemble,
                           'events_group':self.events_group}

        # train ms-TRCA models & templates
        results = mstrca_compute(
            X_train=self.X_train,
            y_train = self.y_train,
            train_info = self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Q, self.S = results[0], results[1]
        self.w, self.w_concat = results[2], results[3]
        self.template, self.ensemble_template = results[4], results[5]
        return self


# %% (3) (e)TRCA-R
def trcar_compute(X_train, y_train, projection, train_info, n_components=1, ratio=None):
    """TRCA-R.

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        projection (ndarray): (n_events, n_points, n_points).
            Orthogonal projection matrices.
        train_info (dict): {'event_type':ndarray (n_events,),
                            'n_events':int,
                            'n_train':ndarray (n_events,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns: | all contained in a tuple
        Q (ndarray): (n_events, n_chans, n_chans). Covariance of original data.
        S (ndarray): (n_events, n_chans, n_chans). Covariance of template data.
        w (list of ndarray): n_events*(n_components, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*n_components, n_chans). Concatenated filter.
        template (ndarray): n_events*(n_components, n_points). TRCA-R templates.
        ensemble_template (ndarray): (n_events, total_components, n_points). eTRCA-R templates.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool

    # S: covariance of projected template
    S = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    class_center = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne in range(n_events):
        class_center[ne] = X_train[y_train==ne].mean(axis=0)  # (Nc,Np)
        XP = class_center[ne] @ projection[ne]  # (Nc,Np)
        S[ne] = XP @ XP.T

    # Q: covariance of original data
    Q = np.zeros_like(S)  # (Ne,Nc,Nc)
    for ne in range(n_events):
        temp = X_train[y_train==ne]  # (Nt,Nc,Np)
        for ntr in range(n_train[ne]):
            Q[ne] += temp[ntr] @ temp[ntr].T

    # GEPs | train spatial filters
    w, ndim = [], []
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(
            A=S[ne],
            B=Q[ne],
            n_components=n_components,
            ratio=ratio
        )
        ndim.append(spatial_filter.shape[0])  # Nk
        w.append(spatial_filter)  # (Nk,Nc)
    w_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    start_idx = 0
    for ne,dims in enumerate(ndim):
        w_concat[start_idx:start_idx+dims] = w[ne]
        start_idx += dims

    # signal templates
    template = []  # Ne*(Nk,Np)
    ensemble_template = np.zeros((n_events, w_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            template.append(w[ne] @ class_center[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ensemble_template[ne] = w_concat @ class_center[ne]  # (Ne*Nk,Np)
    model = (
        Q, S,
        w, w_concat,
        template, ensemble_template
    )
    return model


class TRCA_R(TRCA):
    def fit(self, X_train, y_train, projection):
        """Train (e)TRCA-R model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            projection (ndarray): (n_events, n_points, n_points). Orthogonal projection matrices.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {'event_type':event_type,
                           'n_events':len(event_type),
                           'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
                           'n_chans':self.X_train.shape[-2],
                           'n_points':self.X_train.shape[-1],
                           'standard':self.standard,
                           'ensemble':self.ensemble}
        self.projection = projection

        # train TRCA-R models & templates
        results = trcar_compute(
            X_train=self.X_train,
            y_train = self.y_train,
            projection = self.projection,
            train_info = self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Q, self.S = results[0], results[1]
        self.w, self.w_concat = results[2], results[3]
        self.template, self.ensemble_template = results[4], results[5]
        return self


# %% (4) similarity constrained (e)TRCA | sc-(e)TRCA
def sctrca_compute(X_train, y_train, sine_template, train_info, n_components=1, ratio=None):
    """Similarity-constrained TRCA (sc-TRCA).

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
            Sinusoidal template.
        train_info (dict): {'event_type':ndarray (n_events,),
                            'n_events':int,
                            'n_train':ndarray (n_events,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: all contained in a tuple
        Q (ndarray): (n_events, n_chans, n_chans). Covariance of original data.
        S (ndarray): (n_events, n_chans, n_chans). Covariance of template data.
        u (list of ndarray): n_events*(n_components, n_chans). Spatial filters for EEG signal.
        v (list of ndarray): n_events*(n_components, 2*n_harmonics). Spatial filters for sinusoidal signal.
        u_concat (ndarray): (n_events*n_components, n_chans). Concatenated filter for EEG signal.
        v_concat (ndarray): (n_events*n_components, 2*n_harmonics). Concatenated filter for sinusoidal signal.
        template_eeg (ndarray): n_events*(n_components, n_points). sc-TRCA templates for EEG signal.
        template_sin (ndarray): n_events*(n_components, n_points). sc-TRCA templates for sinusoidal signal.
        ensemble_template_eeg (ndarray): (n_events, total_components, n_points). sc-eTRCA templates for EEG signal.
        ensemble_template_sin (ndarray): (n_events, total_components, n_points). sc-eTRCA templates for sinusoidal signal.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool
    n_2harmonics = sine_template.shape[1]  # 2*Nh

    # block covariance matrix S: [[S11,S12],[S21,S22]]
    S = np.zeros((n_events, n_chans+n_2harmonics, n_chans+n_2harmonics))  # (Ne,Nc+2Nh,Nc+2Nh)
    class_center = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne in range(n_events):
        class_center[ne] = X_train[y_train==ne].mean(axis=0)  # (Nc,Np)
        S[ne,:n_chans,:n_chans] = class_center[ne] @ class_center[ne].T  # S11: XX.T
        S[ne,:n_chans,n_chans:] = class_center[ne] @ sine_template[ne].T  # S12: XY.T
        S[ne,n_chans:,:n_chans] = S[ne,:n_chans,n_chans:].T  # S21: YX.T
        S[ne,n_chans:,n_chans:] = sine_template[ne] @ sine_template[ne].T  # S22: YY.T

    # block covariance matrix Q: blkdiag(Q1,Q2)
    Q = np.zeros_like(S)  # (Ne,Nc+2Nh,Nc+2Nh)
    for ne in range(n_events):
        temp = X_train[y_train==ne]  # (Nt,Nc,Np)
        Q[ne,n_chans:,n_chans:] = n_train[ne] * S[ne,n_chans:,n_chans:]  # Q2
        for ntr in range(n_train[ne]):
            Q[ne,:n_chans,:n_chans] += temp[ntr] @ temp[ntr].T  # Q1

    # GEP | train spatial filters
    u, v, ndim = [], [], []
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(
            A=S[ne],
            B=Q[ne],
            n_components=n_components,
            ratio=ratio
        )
        ndim.append(spatial_filter.shape[0])  # Nk
        u.append(spatial_filter[:,:n_chans])  # (Nk,Nc)
        v.append(spatial_filter[:,n_chans:])  # (Nk,2Nh)
    u_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    v_concat = np.zeros((np.sum(ndim), n_2harmonics))  # (Ne*Nk,2Nh)
    start_idx = 0
    for ne,dims in enumerate(ndim):
        u_concat[start_idx:start_idx+dims] = u[ne]
        v_concat[start_idx:start_idx+dims] = v[ne]
        start_idx += dims

    # signal templates
    template_eeg = []  # Ne*(Nk,Np)
    ensemble_template_eeg = np.zeros((n_events, u_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    template_sin = []
    ensemble_template_sin = np.zeros_like(ensemble_template_eeg)
    if standard:
        for ne in range(n_events):
            template_eeg.append(u[ne] @ class_center[ne])  # (Nk,Np)
            template_sin.append(v[ne] @ sine_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ensemble_template_eeg[ne] = u_concat @ class_center[ne]  # (Nk*Ne,Np)
            ensemble_template_sin[ne] = v_concat @ sine_template[ne]  # (Nk*Ne,Np)
    model = (
        Q, S,
        u, v,
        u_concat, v_concat,
        template_eeg, template_sin,
        ensemble_template_eeg, ensemble_template_sin
    )
    return model


class SC_TRCA(TRCA):
    def fit(self, X_train, y_train, sine_template):
        """Train sc-(e)TRCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
                Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {'event_type':event_type,
                           'n_events':len(event_type),
                           'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
                           'n_chans':self.X_train.shape[-2],
                           'n_points':self.X_train.shape[-1],
                           'standard':self.standard,
                           'ensemble':self.ensemble}

        # train sc-TRCA models & templates
        results = sctrca_compute(
            X_train=self.X_train,
            y_train = self.y_train,
            sine_template=sine_template,
            train_info = self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
            )
        self.Q, self.S = results[0], results[1]
        self.u, self.v = results[2], results[3]
        self.u_concat, self.v_concat = results[4], results[5]
        self.template_eeg, self.template_sin = results[6], results[7]
        self.ensemble_template_eeg = results[8]
        self.ensemble_template_sin = results[9]
        return self


    def predict(self, X_test, y_test):
        """Using sc-(e)TRCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients of TRCA.
                Not empty when self.standard is True.
            y_standard (ndarray): (test_trials,). Predict labels of TRCA.
            erou (ndarray): (test_trials, n_events). Decision coefficients of eTRCA.
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (test_trials,). Predict labels of eTRCA.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']

        # pattern matching (2-step)
        self.rou = np.zeros((n_test, n_events))
        self.rou_eeg = np.zeros_like(self.rou)
        self.rou_sin = np.zeros_like(self.rou)
        self.erou = np.zeros_like(self.rou)
        self.erou_eeg = np.zeros_like(self.rou)
        self.erou_sin = np.zeros_like(self.rou)
        self.y_standard = np.empty((n_test))
        self.y_ensemble = np.empty_like(self.y_standard)
        if self.standard:
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_standard = self.u[nem] @ X_test[nte]
                    self.rou_eeg[nte,nem] = utils.pearson_corr(
                        X=temp_standard,
                        Y=self.template_eeg[nem]
                    )
                    self.rou_sin[nte,nem] = utils.pearson_corr(
                        X=temp_standard,
                        Y=self.template_sin[nem]
                    )
                    self.rou[nte,nem] = utils.combine_feature([
                        self.rou_eeg[nte,nem],
                        self.rou_sin[nte,nem]
                    ])
                self.y_standard[nte] = np.argmax(self.rou[nte,:])
        if self.ensemble:
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_ensemble = self.u_concat @ X_test[nte]
                    self.erou_eeg[nte,nem] = utils.pearson_corr(
                        X=temp_ensemble,
                        Y=self.ensemble_template_eeg[nem]
                    )
                    self.erou_sin[nte,nem] = utils.pearson_corr(
                        X=temp_ensemble,
                        Y=self.ensemble_template_sin[nem]
                    )
                    self.erou[nte,nem] = utils.combine_feature([
                        self.erou_eeg[nte,nem],
                        self.erou_sin[nte,nem]
                    ])
                self.y_ensemble[nte] = np.argmax(self.erou[nte,:])
        return self.rou, self.y_standard, self.erou, self.y_ensemble


# %% (5) group TRCA | gTRCA



# %% (6) cross-correlation TRCA | xTRCA



# %% (7) latency-aligned TRCA | LA-TRCA



# %% (8) task-discriminant component analysis | TDCA
def aug_2(data, projection, extra_length, mode='train'):
    """Construct secondary augmented data.

    Args:
        data (ndarray): (n_chans, n_points+m or n_points).
            m must be larger than n_points while mode is 'train'.
        projection (ndarray): (n_points, n_points). Y.T@Y
        extra_length (int): Extra data length.
        mode (str, optional): 'train' or 'test'.

    Returns:
        data_aug2 (ndarray): ((m+1)*n_chans, 2*n_points).
    """
    # basic information
    n_chans = data.shape[0]  # Nc
    n_points = projection.shape[0]  # Np

    # secondary augmented data
    data_aug2 = np.zeros(((extra_length+1)*n_chans, 2*n_points))  # ((m+1)*Nc,Np+2Nh)
    if mode == 'train':
        for el in range(extra_length+1):
            sp, ep = el*n_chans, (el+1)*n_chans
            data_aug2[sp:ep,:n_points] = data[:,el:n_points+el]  # augmented data
            data_aug2[sp:ep,n_points:] = data_aug2[sp:ep,:n_points] @ projection
    elif mode == 'test':
        for el in range(extra_length+1):
            sp, ep = el*n_chans, (el+1)*n_chans
            data_aug2[sp:ep,:n_points-el] = data[:,el:n_points]
            data_aug2[sp:ep,n_points:] = data_aug2[sp:ep,:n_points] @ projection
    return data_aug2


class TDCA(object):
    """Task-discriminant component analysis."""
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


    def fit(self, X_train, y_train, projection, extra_length):
        """Train TDCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points+extra_length).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            projection (ndarray): (n_events, n_points, n_points).
                Orthogonal projection matrices.
            extra_length (int).
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.projection = projection
        self.extra_length = extra_length

        # create secondary augmented data | (Ne*Nt,(el+1)*Nc,2*Np)
        self.X_train_aug2 = np.zeros((len(self.y_train),
                                      (self.extra_length+1)*self.X_train.shape[-2],
                                      2*(self.X_train.shape[-1] - self.extra_length)))
        for ntr in range(len(self.y_train)):
            self.X_train_aug2[ntr] = aug_2(
                data=X_train[ntr],
                projection=projection[y_train[ntr]],
                extra_length=extra_length
            )
        self.train_info = {'event_type':event_type,
                           'n_events':len(event_type),
                           'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
                           'n_chans':self.X_train_aug2.shape[-2],
                           'n_points':self.X_train_aug2.shape[-1]}

        # train DSP models & templates
        results = dsp_compute(
            X_train=self.X_train_aug2,
            y_train=self.y_train,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Sb, self.Sw = results[0], results[1]
        self.w, self.template = results[2], results[3]
        return self


    def predict(self, X_test, y_test):
        """Using TDCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = y_test.shape[0]
        n_events = self.train_info['n_events']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                temp_test_aug2 = aug_2(
                    data=X_test[nte],
                    projection=self.projection[ne],
                    extra_length=self.extra_length,
                    mode='test'
                )
                self.rou[nte,ne] = utils.pearson_corr(
                    X=self.w @ temp_test_aug2,
                    Y=self.template[ne]
                )
                self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


# %% (9) optimized TRCA | op-TRCA



# %% Filter-bank TRCA series | FB-
class FB_TRCA(TRCA):
    def fit(self, X_train, y_train):
        """Train filter-bank (e)TRCA model.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.n_bands = X_train.shape[0]

        # train TRCA models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train
            )
        return self


    def predict(self, X_test, y_test):
        """Using filter-bank (e)TRCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_bands, n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients of filter-bank TRCA.
                Not empty when self.standard is True.
            y_standard (ndarray): (test_trials,). Predict labels of filter-bank TRCA.
            erou (ndarray): (test_trials, n_events). Decision coefficients of filter-bank eTRCA.
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (test_trials,). Predict labels of filter-bank eTRCA.
        """
        # basic information
        n_test = X_test.shape[1]

        # apply TRCA().predict() in each sub-band
        self.fb_rou = [[] for nb in range(self.n_bands)]
        self.fb_y_standard = [[] for nb in range(self.n_bands)]
        self.fb_erou = [[] for nb in range(self.n_bands)]
        self.fb_y_ensemble = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            fb_results = self.sub_models[nb].predict(
                X_test=X_test[nb],
                y_test=y_test
            )
            self.fb_rou[nb], self.fb_y_standard[nb] = fb_results[0], fb_results[1]
            self.fb_erou[nb], self.fb_y_ensemble[nb] = fb_results[2], fb_results[3]

        # integration of multi-bands' results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.erou = utils.combine_fb_feature(self.fb_erou)
        self.y_standard = np.empty((n_test))
        self.y_ensemble = np.empty_like(self.y_standard)
        for nte in range(n_test):
            self.y_standard[nte] = np.argmax(self.rou[nte,:])
            self.y_ensemble[nte] = np.argmax(self.erou[nte,:])
        return self.rou, self.y_standard, self.erou, self.y_ensemble


class FB_MS_TRCA(FB_TRCA):
    def fit(self, X_train, y_train, events_group=None, d=None):
        """Train filter-bank ms-(e)TRCA model.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            events_group (dict): {'event_id':[start index,end index]}
            d (int): The range of events to be merged.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.events_group = events_group
        self.d = d
        self.n_bands = X_train.shape[0]

        # train ms-TRCA models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = MS_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                events_group=self.events_group,
                d=self.d
            )
        return self


class FB_TRCA_R(FB_TRCA):
    def fit(self, X_train, y_train, projection):
        """Train filter-bank (e)TRCA-R model.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            projection (ndarray): (n_events, n_points, n_points). Orthogonal projection matrices.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.projection = projection
        self.n_bands = X_train.shape[0]

        # train TRCA-R models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = TRCA_R(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                projection=self.projection
            )
        return self


class FB_SC_TRCA(FB_TRCA):
    def fit(self, X_train, y_train, sine_template):
        """Train filter-bank sc-(e)TRCA model.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.n_bands = X_train.shape[0]

        # train sc-TRCA models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = SC_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                sine_template=self.sine_template
            )
        return self


class FB_TDCA(TDCA):
    def fit(self, X_train, y_train, projection, extra_length):
        """Train TDCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points+extra_length).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            projection (ndarray): (n_events, n_points, n_points).
                Orthogonal projection matrices.
            extra_length (int).
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.projection = projection
        self.extra_length = extra_length
        self.n_bands = X_train.shape[0]

        # train TDCA models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = TDCA(
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                projection=self.projection,
                extra_length=self.extra_length
            )
        return self


    def predict(self, X_test, y_test):
        """Using filter-bank TDCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_bands, n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[1]

        # apply TDCA().predict() in each sub-band
        self.fb_rou = [[] for nb in range(self.n_bands)]
        self.fb_y_predict = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.fb_rou[nb], self.fb_y_predict[nb] = self.sub_models[nb].predict(
                X_test=X_test[nb],
                y_test=y_test
            )

        # integration of multi-bands' results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


# %%