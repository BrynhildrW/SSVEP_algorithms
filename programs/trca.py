# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Task-related component analysis (TRCA) series.
    1. (e)TRCA: https://ieeexplore.ieee.org/document/7904641/
            DOI: 10.1109/TBME.2017.2694818
    2. ms-(e)TRCA: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
            DOI: 10.1088/1741-2552/ab2373
    3. (e)TRCA-R: https://ieeexplore.ieee.org/document/9006809/
            DOI: 10.1109/TBME.2020.2975552
    4. sc-(e)TRCA: https://iopscience.iop.org/article/10.1088/1741-2552/abfdfa
            DOI: 10.1088/1741-2552/abfdfa
    5. TS-CORRCA: https://ieeexplore.ieee.org/document/8387802/
            DOI: 10.1109/TNSRE.2018.2848222
    6. gTRCA: 
            DOI:
    7. xTRCA: 
            DOI:
    8. LA-TRCA: 
            DOI:
    9. TL-(e)TRCA (unofficial name): https://ieeexplore.ieee.org/document/10057002/
            DOI: 10.1109/TNSRE.2023.3250953

update: 2023/06/03

"""

# basic modules
import utils

import numpy as np

import scipy.linalg as sLA

from abc import abstractmethod, ABCMeta


# %% Basic TRCA object
class BasicTRCA(metaclass=ABCMeta):
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


    @abstractmethod
    def fit(self, X_train, y_train):
        pass


    @abstractmethod
    def predict(self, X_test):
        pass


class BasicFBTRCA(metaclass=ABCMeta):
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


    @abstractmethod
    def fit(self, X_train, y_train):
        """Load in training dataset.

        Args:
            X_train (ndarray): (n_bands, train_trials, ..., n_points). Training dataset.
            y_train (ndarray): (train_trials,). Labels for X_train.
        """
        # # basic information
        # self.X_train = X_train
        # self.y_train = y_train
        # self.n_bands = self.X_train.shape[0]
        pass


    def predict(self, X_test):
        """Using filter-bank TRCA algorithms to predict test data.

        Args:
            X_test (ndarray): (n_bands, n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.

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

        # apply predict() method in each sub-band
        self.fb_rou = [[] for nb in range(self.n_bands)]
        self.fb_y_standard = [[] for nb in range(self.n_bands)]
        self.fb_erou = [[] for nb in range(self.n_bands)]
        self.fb_y_ensemble = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
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


# %% 1. (ensemble) TRCA | (e)TRCA
def trca_compute(X_train, y_train, train_info, n_components=1, ratio=None):
    """Task-related component analysis (TRCA).

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if necessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        train_info (dict): {'event_type':ndarray (n_events,),
                            'n_events':int,
                            'n_train':ndarray (n_events,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int, optional): Number of eigenvectors picked as filters.
            Defaults to 1. Set to 'None' if ratio is not 'None'.
        ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not 'None'.

    Returns: | all contained in a dict
        Q (ndarray): (n_events, n_chans, n_chans). Covariance of original data.
        S (ndarray): (n_events, n_chans, n_chans). Covariance of template data.
        w (list of ndarray): n_events*(n_components, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*n_components(total_components), n_chans). Concatenated filter.
        wX (ndarray): n_events*(n_components, n_points). TRCA templates.
        ewX (ndarray): (n_events, total_components, n_points). eTRCA templates.
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
    for ne,et in enumerate(event_type):
        class_center[ne] = X_train[y_train==et].mean(axis=0)  # (Nc,Np)
        S[ne] = class_center[ne] @ class_center[ne].T
    # S = np.einsum('ecp,ehp->ech', class_center,class_center) | clearer but slower

    # Q: covariance of original data
    Q = np.zeros_like(S)  # (Ne,Nc,Nc)
    for ne,et in enumerate(event_type):
        temp = X_train[y_train==et]  # (Nt,Nc,Np)
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
    wX = []  # Ne*(Nk,Np)
    ewX = np.zeros((n_events, w_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX.append(w[ne] @ class_center[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = w_concat @ class_center[ne]  # (Ne*Nk,Np)
    return {'Q':Q, 'S':S, 'w':w, 'w_concat':w_concat, 'wX':wX, 'ewX':ewX}


class TRCA(BasicTRCA):
    def fit(self, X_train, y_train):
        """Train (e)TRCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if necessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {
            'event_type':event_type,
            'n_events':len(event_type),
            'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
            'n_chans':self.X_train.shape[-2],
            'n_points':self.X_train.shape[-1],
            'standard':self.standard,
            'ensemble':self.ensemble
        }

        # train TRCA filters & templates
        results = trca_compute(
            X_train=self.X_train,
            y_train = self.y_train,
            train_info = self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Q, self.S = results['Q'], results['S']
        self.w, self.w_concat = results['w'], results['w_concat']
        self.wX, self.ewX = results['wX'], results['ewX']
        return self


    def predict(self, X_test):
        """Using (e)TRCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if necessary.

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
                        Y=self.wX[nem]
                    )
                self.y_standard[nte] = np.argmax(self.rou[nte,:])
        if self.ensemble:
            for nte in range(n_test):
                for nem in range(n_events):
                    self.erou[nte,nem] = utils.pearson_corr(
                        X=self.w_concat @ X_test[nte],
                        Y=self.ewX[nem]
                    )
                self.y_ensemble[nte] = np.argmax(self.erou[nte,:])
        return self.rou, self.y_standard, self.erou, self.y_ensemble


class FB_TRCA(BasicFBTRCA):
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


# %% 2. multi-stimulus (e)TRCA | ms-(e)TRCA
def mstrca_compute(X_train, y_train, train_info, n_components=1, ratio=None):
    """Multi-stimulus TRCA (ms-TRCA).

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        train_info (dict): {'n_events':int,
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

    Returns: | all contained in a dict
        Q (ndarray): (n_events, n_chans, n_chans). Covariance of original data.
        S (ndarray): (n_events, n_chans, n_chans). Covariance of template data.
        w (list of ndarray): n_events*(n_components, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*n_components, n_chans). Concatenated filter.
        wX (list of ndarray): n_events*(n_components, n_points). ms-TRCA templates.
        ewX (ndarray): (n_events, total_components, n_points). ms-eTRCA templates.
    """
    # basic information
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

    # signal templates: normal and ensemble
    wX = []  # Ne*(Nk,Np)
    ewX = np.zeros((n_events, w_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX.append(w[ne] @ class_center[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = w_concat @ class_center[ne]  # (Ne*Nk,Np)
    return {'Q':total_Q, 'S':total_S, 'w':w, 'w_concat':w_concat, 'wX':wX, 'ewX':ewX}


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
        self.train_info = {
            'event_type':event_type,
            'n_events':len(event_type),
            'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
            'n_chans':self.X_train.shape[-2],
            'n_points':self.X_train.shape[-1],
            'standard':self.standard,
            'ensemble':self.ensemble,
            'events_group':self.events_group
        }

        # train ms-TRCA models & templates
        results = mstrca_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Q, self.S = results['Q'], results['S']
        self.w, self.w_concat = results['w'], results['w_concat']
        self.wX, self.ewX = results['wX'], results['ewX']
        return self


class FB_MS_TRCA(BasicFBTRCA):
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


# %% 3. (e)TRCA-R
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

    Returns: | all contained in a dict
        Q (ndarray): (n_events, n_chans, n_chans). Covariance of original data.
        S (ndarray): (n_events, n_chans, n_chans). Covariance of template data.
        w (list of ndarray): n_events*(n_components, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*n_components, n_chans). Concatenated filter.
        wX (ndarray): n_events*(n_components, n_points). TRCA-R templates.
        ewX (ndarray): (n_events, total_components, n_points). eTRCA-R templates.
    """
    # basic information
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
    wX = []  # Ne*(Nk,Np)
    ewX = np.zeros((n_events, w_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX.append(w[ne] @ class_center[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = w_concat @ class_center[ne]  # (Ne*Nk,Np)
    return {'Q':Q, 'S':S, 'w':w, 'w_concat':w_concat, 'wX':wX, 'ewX':ewX}


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
        self.train_info = {
            'event_type':event_type,
            'n_events':len(event_type),
            'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
            'n_chans':self.X_train.shape[-2],
            'n_points':self.X_train.shape[-1],
            'standard':self.standard,
            'ensemble':self.ensemble
        }
        self.projection = projection

        # train TRCA-R models & templates
        results = trcar_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            projection=self.projection,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Q, self.S = results['Q'], results['S']
        self.w, self.w_concat = results['w'], results['w_concat']
        self.wX, self.ewX = results['wX'], results['ewX']
        return self


class FB_TRCA_R(BasicFBTRCA):
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


# %% 4. similarity constrained (e)TRCA | sc-(e)TRCA
def sctrca_compute(X_train, y_train, sine_template, train_info, n_components=1, ratio=None):
    """Similarity-constrained TRCA (sc-TRCA).

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
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

    Return: all contained in a dict (model).
        Q (ndarray): (n_events, n_chans, n_chans).
            Covariance of original data & template data.
        S (ndarray): (n_events, n_chans, n_chans).
            Covariance of template data.
        u (list of ndarray): n_events*(n_components, n_chans).
            Spatial filters for EEG signal.
        v (list of ndarray): n_events*(n_components, 2*n_harmonics).
            Spatial filters for sinusoidal signal.
        u_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for EEG signal.
        v_concat (ndarray): (n_events*n_components, 2*n_harmonics).
            Concatenated filter for sinusoidal signal.
        uX (ndarray): n_events*(n_components, n_points).
            sc-TRCA templates for EEG signal.
        vY (ndarray): n_events*(n_components, n_points).
            sc-TRCA templates for sinusoidal signal.
        euX (ndarray): (n_events, total_components, n_points).
            sc-eTRCA templates for EEG signal.
        evY (ndarray): (n_events, total_components, n_points).
            sc-eTRCA templates for sinusoidal signal.
    """
    # basic information
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
    uX = []  # Ne*(Nk,Np)
    euX = np.zeros((n_events, u_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    vY = []
    evY = np.zeros_like(euX)
    if standard:
        for ne in range(n_events):
            uX.append(u[ne] @ class_center[ne])  # (Nk,Np)
            vY.append(v[ne] @ sine_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            euX[ne] = u_concat @ class_center[ne]  # (Nk*Ne,Np)
            evY[ne] = v_concat @ sine_template[ne]  # (Nk*Ne,Np)
    model = {
        'Q':Q, 'S':S,
        'u':u, 'v':v, 'u_concat':u_concat, 'v_concat':v_concat,
        'uX':uX, 'vY':vY, 'euX':euX, 'evY':evY
    }
    return model


class SC_TRCA(BasicTRCA):
    def fit(self, X_train, y_train, sine_template):
        """Train sc-(e)TRCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
                Sinusoidal template.
        """
        # basic information1
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {
            'event_type':event_type,
            'n_events':len(event_type),
            'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
            'n_chans':self.X_train.shape[-2],
            'n_points':self.X_train.shape[-1],
            'standard':self.standard,
            'ensemble':self.ensemble
        }

        # train sc-TRCA models & templates
        results = sctrca_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Q, self.S = results['Q'], results['S']
        self.u, self.v = results['u'], results['v']
        self.u_concat, self.v_concat = results['u_concat'], results['v_concat']
        self.uX = results['uX']
        self.vY = results['vY']
        self.euX = results['euX']
        self.evY = results['evY']
        return self


    def predict(self, X_test):
        """Using sc-(e)TRCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients of sc-TRCA.
                Not empty when self.standard is True.
            y_standard (ndarray): (test_trials,). Predict labels of sc-TRCA.
            erou (ndarray): (test_trials, n_events). Decision coefficients of sc-eTRCA.
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (test_trials,). Predict labels of sc-eTRCA.
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
                        Y=self.uX[nem]
                    )
                    self.rou_sin[nte,nem] = utils.pearson_corr(
                        X=temp_standard,
                        Y=self.vY[nem]
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
                        Y=self.euX[nem]
                    )
                    self.erou_sin[nte,nem] = utils.pearson_corr(
                        X=temp_ensemble,
                        Y=self.evY[nem]
                    )
                    self.erou[nte,nem] = utils.combine_feature([
                        self.erou_eeg[nte,nem],
                        self.erou_sin[nte,nem]
                    ])
                self.y_ensemble[nte] = np.argmax(self.erou[nte,:])
        return self.rou, self.y_standard, self.erou, self.y_ensemble


class FB_SC_TRCA(BasicFBTRCA):
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



# %% 5. two-stage CORRCA | TS-CORRCA



# %% 6. group TRCA | gTRCA



# %% 7. cross-correlation TRCA | xTRCA



# %% 8. latency-aligned TRCA | LA-TRCA



# %% 9. cross-subject transfer learning TRCA | TL-TRCA
def tltrca_source_compute(X_train, y_train, sine_template, n_components=1, ratio=None):
    """Transfer learning for source subject in TL-(e)TRCA.

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Dataset of source subjects. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: source_model (dict).
        Q (ndarray): (n_events, 2*n_chans+2*n_harmonics, 2*n_chans+2*n_harmonics).
            Covariance of original data & template data.
        S (ndarray): (n_events, 2*n_chans+2*n_harmonics, 2*n_chans+2*n_harmonics).
            Covariance of template data.
        w (List of ndarray): n_events*(n_components, n_chans).
            Spatial filters for original signal.
        u (List of ndarray): n_events*(n_components, n_chans).
            Spatial filters for average template.
        v (List of ndarray): n_events*(n_components, 2*n_harmonics).
            Spatial filters for sinusoidal template.
        w_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for original signal.
        u_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for average template.
        v_concat (ndarray): (n_events*n_components, 2*n_harmonics).
            Concatenated filter for sinusoidal signal.
        I (ndarray): (n_events, n_events*n_components, n_points).
            Transferred individual templates of source subjects.
        R (ndarray): (n_events, n_events*n_components, n_points).
            Transferred reference templates of source subjects.
    """
    # basic information
    event_type = np.unique(y_train)
    n_events = len(event_type)  # Ne
    n_train = np.array([np.sum(y_train==et) for et in event_type])  # [Nt1,Nt2,...]
    n_chans = X_train.shape[-2]  # Nc
    n_points = X_train.shape[-1]  # Np
    n_2harmonics = sine_template.shape[1]  # 2*Nh

    # initialization
    S = np.zeros((n_events, 2*n_chans+n_2harmonics, 2*n_chans+n_2harmonics))
    Q = np.zeros_like(S)
    w, u, v, w_concat, u_concat, v_concat = [], [], [], [], [], []
    I, R = [], []

    # block covariance matrices: S & Q, (Ne,2Nc+2Nh,2Nc+2Nh)
    class_center = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne in range(n_events):
        class_center[ne] = X_train[y_train==ne].mean(axis=0)  # (Nc,Np)
        train_trials = n_train[ne]  # Nt
        X_temp = X_train[y_train==ne]  # (Nt,Nc,Np)

        # S11: inter-trial covariance, (Nc,Nc)
        S11 = np.zeros((n_chans, n_chans))
        for ttr in range(train_trials):
            for tti in range(train_trials):
                if ttr != tti:
                    S11 += X_temp[ttr] @ X_temp[tti].T

        # S12 & S21.T: covariance between the SSVEP trials & the individual template, (Nc,Nc)
        S12 = np.zeros_like(S11)
        for tt in range(train_trials):
            S12 += X_temp[tt] @ class_center[ne].T

        # S13 & S31.T: similarity between the SSVEP trials & sinusoidal template, (Nc,2Nh)
        S13 = np.zeros((n_chans, n_2harmonics))
        for tt in range(train_trials):
            S13 += X_temp[tt] @ sine_template[ne].T

        # S23 & S32.T: covariance between the individual template & sinusoidal template, (Nc,2Nh)
        S23 = class_center[ne] @ sine_template[ne].T

        # S22 & S33: variance of average template & sinusoidal template, (Nc,Nc) & (2Nh,2Nh)
        S22 = class_center[ne] @ class_center[ne].T
        S33 = sine_template[ne] @ sine_template[ne].T

        # S: [[S11,S12,S13],[S21,S22,S23],[S31,S32,S33]]
        S[ne, :n_chans, :n_chans] = S11
        S[ne, :n_chans, n_chans:2*n_chans] = S12
        S[ne, n_chans:2*n_chans, :n_chans] = S12.T
        S[ne, :n_chans, 2*n_chans:] = S13
        S[ne, 2*n_chans:, :n_chans] = S13.T
        S[ne, n_chans:2*n_chans, n_chans:2*n_chans] = S22
        S[ne, n_chans:2*n_chans, 2*n_chans:] = S23
        S[ne, 2*n_chans:, n_chans:2*n_chans] = S23.T
        S[ne, 2*n_chans:, 2*n_chans:] = S33

        # Q1 = np.einsum('tcp,thp->ch', X_sub[ne], X_sub[ne])  # (Nc,Nc) | clear but slow
        Q2 = class_center[ne] @ class_center[ne].T  # (Nc,Nc)
        Q3 = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh)
        Q1 = np.zeros_like(Q2)  # (Nc,Nc)
        for tt in range(train_trials):
            Q1 += X_temp[tt] @ X_temp[tt].T  # faster way

        # Q: blkdiag(Q1,Q2,Q3)
        Q[ne, :n_chans, :n_chans] = Q1
        Q[ne, n_chans:2*n_chans, n_chans:2*n_chans] = Q2
        Q[ne, 2*n_chans:, 2*n_chans:] = Q3

    # GEP | train spatial filters for transfer subjects
    w, u, v, ndim = [], [], [], []
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(
            A=S[ne],
            B=Q[ne],
            n_components=n_components,
            ratio=ratio
        )
        ndim.append(spatial_filter.shape[0])  # Nk
        w.append(spatial_filter[:,:n_chans])  # (Nk,Nc) | for raw signal
        u.append(spatial_filter[:,n_chans:2*n_chans])  # (Nk,Nc) | for average template
        v.append(spatial_filter[:,2*n_chans:])  # (Nk,2Nh) | for sine-cosine template
    w_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    u_concat = np.zeros_like(w_concat)  # (Ne*Nk,Nc)
    v_concat = np.zeros((np.sum(ndim), n_2harmonics))  # (Ne*Nk,2Nh)
    start_idx = 0
    for ne,dims in enumerate(ndim):
        w_concat[start_idx:start_idx+dims] = w[ne]
        u_concat[start_idx:start_idx+dims] = u[ne]
        v_concat[start_idx:start_idx+dims] = v[ne]
        start_idx += dims

    # transferred individual template & reference template
    I = np.einsum('kc,ecp->ekp', u_concat, class_center)  # (Ne,Ne*Nk,Np)
    R = np.einsum('kh,ehp->ekp', v_concat, sine_template)  # (Ne,Ne*Nk,Np)
    source_model = {
        'Q':Q, 'S':S,
        'w':w, 'u':u, 'v':v,
        'w_concat':w_concat, 'u_concat':u_concat, 'v_concat':v_concat,
        'I':I, 'R':R
    }
    return source_model


def source_model_train(X_source, y_source, sine_template, n_components=1, ratio=None):
    """Train transferred models from all source subjects.

    Args:
        X_source (List-like object): n_subjects*(n_events*n_train(train_trials), n_chans, n_points).
            Dataset of source subjects. train_trials could be 1 if neccessary.
        y_source (List-like object): n_subjects*(train_trials,). Labels for X.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return:
        source_model (list of dict): List of source_model.
    """
    # basic information
    n_subjects = len(X_source)

    # model integration
    source_model = []
    for nsub in range(n_subjects):
        model = tltrca_source_compute(
            X_train=X_source[nsub],
            y_train=y_source[nsub],
            sine_template=sine_template,
            n_components=n_components,
            ratio=ratio
        )
        source_model.append(model)
        # print('Finish source subject ID: %d' %(nsub+1))
    return source_model


def tltrca_target_compute(X_train, y_train, source_model, sine_template, train_info,
                          n_components=1, ratio=None):
    """Transfer learning for target subjects in TL-(e)TRCA.

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        source_model (List of dict): Details in tltrca_source_compute().
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
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

    Return: all contained in a dict (target_model).
        TI (List of ndarray): n_subjects*(n_events, n_events*n_components, n_chans).
            Transferred spatial filters for individual template.
        TR (List of ndarray): n_subjects*(n_events, n_events*n_components, n_chans).
            Transferred spatial filters for reference template.
        dist_I (ndarray): (n_subjects, n_events).
            Distances between transferred target data and source individual template.
        dist_R (ndarray): (n_subjects, n_events).
            Distances between transferred target data and source reference template.
        cs_I (ndarray): (n_subjects, n_events).
            Contribution scores of dI within source subjects.
        cs_R (ndarray): (n_subjects, n_events).
            Contribution scores of dR within source subjects.
        Q (ndarray): (n_events, 2*n_chans+2*n_harmonics, 2*n_chans+2*n_harmonics).
            Covariance of original data & template data.
        S (ndarray): (n_events, 2*n_chans+2*n_harmonics, 2*n_chans+2*n_harmonics).
            Covariance of template data.
        w (List of ndarray): n_events*(n_components, n_chans).
            Spatial filters for original signal.
        u (List of ndarray): n_events*(n_components, n_chans).
            Spatial filters for average template.
        v (List of ndarray): n_events*(n_components, 2*n_harmonics).
            Spatial filters for sinusoidal template.
        w_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for original signal.
        u_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for average template.
        v_concat (ndarray): (n_events*n_components, 2*n_harmonics).
            Concatenated filter for sinusoidal signal.
        uX (List of ndarray): n_events*(n_components, n_points).
            TL-TRCA average templates of target subject.
        vY (List of ndarray): n_events*(n_components, n_points).
            TL-TRCA sinusoidal templates of target subject.
        euX (ndarray): (n_events, total_components, n_points).
            TL-eTRCA average templates of target subject.
        evY (ndarray): (n_events, total_components, n_points).
            TL-eTRCA sinusoidal templates of target subject.
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool
    n_subjects = len(source_model)

    # transferred learning for target subject
    TI, TR = [], []
    dist_I = np.zeros((n_subjects, n_events))  # (Ns,Ne)
    dist_R = np.zeros_like(dist_I)
    cs_I, cs_R = np.zeros_like(dist_I), np.zeros_like(dist_I)
    class_center = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for nsub in range(n_subjects):
        I, R = source_model[nsub]['I'], source_model[nsub]['R']  # (Ne,Ne*Nk,Np)
        TI_sub = np.zeros((n_events, I.shape[1], n_chans))  # (Ne,Ne*Nk,Nc)
        TR_sub = np.zeros((n_events, R.shape[1], n_chans))  # (Ne,Ne*Nk,Nc)
        for ne in range(n_events):
            temp = X_train[y_train==ne]  # (Nt,Nc,Np)
            class_center[ne] = temp.mean(axis=0)  # (Nc,Np)
            train_trials = n_train[ne]  # Nt

            # transferred spatial filters for individual template & reference template
            for tt in range(train_trials):
                TI_sub[ne] += (sLA.inv(temp[tt] @ temp[tt].T) @ temp[tt] @ I[ne].T).T  # (Nc,Ne*Nk)
                TR_sub[ne] += (sLA.inv(temp[tt] @ temp[tt].T) @ temp[tt] @ R[ne].T).T  # (Nc,Ne*Nk)
            TI_sub[ne] /= train_trials
            TR_sub[ne] /= train_trials

            # distances between transferred target data & source template
            for tt in range(train_trials):
                dist_I[nsub,ne] += utils.pearson_corr(TI_sub[ne] @ temp[tt], I[ne])
                dist_R[nsub,ne] += utils.pearson_corr(TR_sub[ne] @ temp[tt], R[ne])
        TI.append(TI_sub)
        TR.append(TR_sub)

    # contribution scores
    cs_I = dist_I / np.sum(dist_I, axis=0, keepdims=True)
    cs_R = dist_R / np.sum(dist_R, axis=0, keepdims=True)

    # self-training for target subject
    results = tltrca_source_compute(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            n_components=n_components,
            ratio=ratio
        )
    u, v = results['u'], results['v']
    u_concat, v_concat = results['u_concat'], results['v_concat']
    
    # self-trained templates: EEG template (u@X) & sinusoidal template (v@Y)
    uX, vY = [], []
    euX = np.zeros((n_events, u_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros_like(euX)
    if standard:
        for ne in range(n_events):
            uX.append(u[ne] @ class_center[ne])  # (Nk,Np)
            vY.append(v[ne] @ sine_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            euX[ne] = u_concat @ class_center[ne]  # (Nk*Ne,Np)
            evY[ne] = v_concat @ sine_template[ne]  # (Nk*Ne,Np)
    target_model = {
        'TI':TI, 'TR':TR,
        'dist_I':dist_I, 'dist_R':dist_R,
        'cs_I':cs_I, 'cs_R':cs_R,
        'Q':results['Q'], 'S':results['S'],
        'w':results['w'], 'u':u, 'v':v,
        'w_concat':results['w_concat'], 'u_concat':u_concat, 'v_concat':v_concat,
        'uX':uX, 'vY':vY, 'euX':euX, 'evY':evY
    }
    return target_model


class TL_TRCA(BasicTRCA):
    def fit(self, X_train, y_train, source_model, sine_template):
        """Train TL-(e)TRCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            source_model (List of dict): Details in tltrca_compute_source().
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
                Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.source_model = source_model
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {
            'event_type':event_type,
            'n_events':len(event_type),
            'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
            'n_chans':self.X_train.shape[-2],
            'n_points':self.X_train.shape[-1],
            'standard':self.standard,
            'ensemble':self.ensemble
        }

        # train transferred model for target subject
        target_model = tltrca_target_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            source_model=self.source_model,
            sine_template=sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.TI, self.TR = target_model['TI'], target_model['TR']
        self.dist_I, self.dist_R = target_model['dist_I'], target_model['dist_R']
        self.cs_I, self.cs_R = target_model['cs_I'], target_model['cs_R']
        self.w, self.u, self.v = target_model['w'], target_model['u'], target_model['v']
        self.w_concat = target_model['w_concat']
        self.u_concat = target_model['u_concat']
        self.v_concat = target_model['v_concat']
        self.uX, self.vY = target_model['uX'], target_model['vY']
        self.euX, self.evY = target_model['euX'], target_model['evY']
        return self


    def predict(self, X_test):
        """Using TL-(e)TRCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients of TL-TRCA.
                Not empty when self.standard is True.
            y_standard (ndarray): (test_trials,). Predict labels of TL-TRCA.
            erou (ndarray): (test_trials, n_events). Decision coefficients of TL-eTRCA.
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (test_trials,). Predict labels of TL-eTRCA.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']

        self.rou = np.zeros((n_test, n_events, 4))
        self.final_rou = np.zeros((n_test, n_events))
        self.erou = np.zeros_like(self.rou)
        self.final_erou = np.zeros_like(self.final_rou)
        self.y_standard = np.empty((n_test))
        self.y_ensemble = np.empty_like(self.y_standard)
        n_subjects = len(self.source_model)

        # rou 1 & 2: transferred pattern matching
        for nte in range(n_test):
            for nem in range(n_events):
                for nsub in range(n_subjects):
                    self.rou[nte,nem,0] += self.cs_I[nsub,nem]*utils.pearson_corr(
                        X=self.TI[nsub][nem] @ X_test[nte],
                        Y=self.source_model[nsub]['I'][nem]
                    )
                    self.rou[nte,nem,1] += self.cs_R[nsub,nem]*utils.pearson_corr(
                        X=self.TR[nsub][nem] @ X_test[nte],
                        Y=self.source_model[nsub]['R'][nem]
                    )

        # rou 3 & 4: self-trained pattern matching (similar to sc-(e)TRCA)
        if self.standard:
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_standard = self.w[nem] @ X_test[nte]
                    self.rou[nte,nem,2] = utils.pearson_corr(
                        X=temp_standard,
                        Y=self.uX[nem]
                    )
                    self.rou[nte,nem,3] = utils.pearson_corr(
                        X=temp_standard,
                        Y=self.vY[nem]
                    )
                    self.final_rou[nte,nem] = utils.combine_feature([
                        self.rou[nte,nem,0],
                        self.rou[nte,nem,1],
                        self.rou[nte,nem,2],
                        self.rou[nte,nem,3],
                    ])
                self.y_standard[nte] = np.argmax(self.final_rou[nte,:])
        if self.ensemble:
            self.erou[...,0], self.erou[...,1] = self.rou[...,0], self.rou[...,1]
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_ensemble = self.w_concat @ X_test[nte]
                    self.erou[nte,nem,2] = utils.pearson_corr(
                        X=temp_ensemble,
                        Y=self.euX
                    )
                    self.erou[nte,nem,3] = utils.pearson_corr(
                        X=temp_ensemble,
                        Y=self.evY
                    )
                    self.final_erou[nte,nem] = utils.combine_feature([
                        self.erou[nte,nem,0],
                        self.erou[nte,nem,1],
                        self.erou[nte,nem,2],
                        self.erou[nte,nem,3],
                    ])
                self.y_ensemble[nte] = np.argmax(self.final_erou[nte,:])
        return self.rou, self.y_standard, self.erou, self.y_ensemble


def fb_source_model_train(X_source, y_source, sine_template, n_components=1, ratio=None):
    """Train transferred models from all source subjects (Filter-bank version).

    Args:
        X_source (List-like object): n_subjects*(n_bands, train_trials, n_chans, n_points).
            Dataset of source subjects. train_trials could be 1 if neccessary.
        y_source (List-like object): n_subjects*(train_trials,). Labels for X.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return:
        source_model (List of dict): List of source_model.
            Add a dimension of n_bands for each variable in source_model.
    """
    # basic information
    n_subjects = len(X_source)
    n_bands = X_source[0].shape[0]

    # model integration
    fb_source_model = [[] for nb in range(n_bands)]
    for nb in range(n_bands):
        for nsub in range(n_subjects):
            model = tltrca_source_compute(
                X_train=X_source[nsub][nb],
                y_train=y_source[nsub],
                sine_template=sine_template,
                n_components=n_components,
                ratio=ratio
            )
            fb_source_model[nb].append(model)
    return fb_source_model


class FB_TL_TRCA(BasicFBTRCA):
    def fit(self, X_train, y_train, fb_source_model, sine_template):
        """Train filter-bank TL-(e)TRCA model.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            fb_source_model (List of dict): Details in tltrca_compute_source().
                Add a dimension of n_bands for each variable in source_model.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.fb_source_model = fb_source_model
        self.n_bands = X_train.shape[0]

        # train TL-TRCA models
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = TL_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                source_model=self.fb_source_model[nb],
                sine_template=sine_template
            )
        return self