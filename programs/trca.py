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

update: 2023/07/04

"""

# %% basic modules
import utils

from abc import abstractmethod, ABCMeta
from typing import Optional, List, Tuple

import numpy as np
from numpy import ndarray
import scipy.linalg as sLA


# %% Basic TRCA object
class BasicTRCA(metaclass=ABCMeta):
    def __init__(self,
        standard: Optional[bool] = True,
        ensemble: Optional[bool] = True,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None):
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
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        pass


    @abstractmethod
    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        pass


class BasicFBTRCA(metaclass=ABCMeta):
    def __init__(self,
        standard: Optional[bool] = True,
        ensemble: Optional[bool] = True,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None):
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
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        pass


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
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

        # apply model.predict() method in each sub-band
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
def trca_compute(
    X_train: ndarray,
    y_train: ndarray,
    train_info: dict,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict:
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

    Return: TRCA model (dict)
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
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne,et in enumerate(event_type):
        avg_template[ne] = X_train[y_train==et].mean(axis=0)  # (Nc,Np)
        S[ne] = avg_template[ne] @ avg_template[ne].T
    # S = np.einsum('ecp,ehp->ech', avg_template,avg_template) | clearer but slower

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
            wX.append(w[ne] @ avg_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = w_concat @ avg_template[ne]  # (Ne*Nk,Np)
    return {'Q':Q, 'S':S, 'w':w, 'w_concat':w_concat, 'wX':wX, 'ewX':ewX}


class TRCA(BasicTRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
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


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
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
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
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

        # train TRCA models in each band
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
def mstrca_compute(
    X_train: ndarray,
    y_train: ndarray,
    train_info: dict,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict:
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

    Return: | all contained in a dict
        Q (ndarray): (n_events, n_chans, n_chans). Covariance of original data.
        S (ndarray): (n_events, n_chans, n_chans). Covariance of template data.
        w (list of ndarray): n_events*(n_components, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*n_components, n_chans). Concatenated filter.
        wX (list of ndarray): n_events*(n_components, n_points). ms-TRCA templates.
        ewX (ndarray): (n_events, total_components, n_points). ms-eTRCA templates.
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
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne,et in enumerate(event_type):
        avg_template[ne] = X_train[y_train==et].mean(axis=0)  # (Nc,Np)
        total_S[ne] = avg_template[ne] @ avg_template[ne].T

    # Q: covariance of original data | same with TRCA
    total_Q = np.zeros_like(total_S)  # (Ne,Nc,Nc)
    for ne,et in enumerate(event_type):
        temp = X_train[y_train==et]  # (Nt,Nc,Np)
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
            wX.append(w[ne] @ avg_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = w_concat @ avg_template[ne]  # (Ne*Nk,Np)
    return {'Q':total_Q, 'S':total_S, 'w':w, 'w_concat':w_concat, 'wX':wX, 'ewX':ewX}


class MS_TRCA(TRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        events_group: Optional[dict] = None,
        d: Optional[int] = 5):
        """Train ms-(e)TRCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            events_group (dict): {'event_id':[start index,end index]}
            d (int): The range of events to be merged. Defaults to 5.
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
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        events_group: Optional[dict] = None,
        d: Optional[int] = 5):
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
def trcar_compute(
    X_train: ndarray,
    y_train: ndarray,
    projection: ndarray,
    train_info: dict,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict:
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

    Return: | all contained in a dict
        Q (ndarray): (n_events, n_chans, n_chans). Covariance of original data.
        S (ndarray): (n_events, n_chans, n_chans). Covariance of template data.
        w (list of ndarray): n_events*(n_components, n_chans). Spatial filters.
        w_concat (ndarray): (n_events*n_components, n_chans). Concatenated filter.
        wX (ndarray): n_events*(n_components, n_points). TRCA-R templates.
        ewX (ndarray): (n_events, total_components, n_points). eTRCA-R templates.
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
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne,et in enumerate(event_type):
        avg_template[ne] = X_train[y_train==et].mean(axis=0)  # (Nc,Np)
        XP = avg_template[ne] @ projection[ne]  # (Nc,Np)
        S[ne] = XP @ XP.T

    # Q: covariance of original data
    Q = np.zeros_like(S)  # (Ne,Nc,Nc)
    for ne,et in enumerate(event_type):
        temp = X_train[y_train==et]  # (Nt,Nc,Np)
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
            wX.append(w[ne] @ avg_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = w_concat @ avg_template[ne]  # (Ne*Nk,Np)
    return {'Q':Q, 'S':S, 'w':w, 'w_concat':w_concat, 'wX':wX, 'ewX':ewX}


class TRCA_R(TRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        projection: ndarray):
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
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        projection: ndarray):
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
def sctrca_compute(
    X_train: ndarray,
    y_train: ndarray,
    sine_template: ndarray,
    train_info: dict,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict[str, ndarray]:
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
        u (List[ndarray]): n_events*(n_components, n_chans).
            Spatial filters for EEG signal.
        v (List[ndarray]): n_events*(n_components, 2*n_harmonics).
            Spatial filters for sinusoidal signal.
        u_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for EEG signal.
        v_concat (ndarray): (n_events*n_components, 2*n_harmonics).
            Concatenated filter for sinusoidal signal.
        uX (List[ndarray]): n_events*(n_components, n_points).
            sc-TRCA templates for EEG signal.
        vY (List[ndarray]): n_events*(n_components, n_points).
            sc-TRCA templates for sinusoidal signal.
        euX (List[ndarray]): (n_events, total_components, n_points).
            sc-eTRCA templates for EEG signal.
        evY (List[ndarray]): (n_events, total_components, n_points).
            sc-eTRCA templates for sinusoidal signal.
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

    S = np.zeros((n_events, n_chans+n_2harmonics, n_chans+n_2harmonics))  # (Ne,Nc+2Nh,Nc+2Nh)
    Q = np.zeros_like(S)  # (Ne,Nc+2Nh,Nc+2Nh)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        train_trials = n_train[ne]  # Nt
        X_temp = X_train[y_train==et]  # (Nt,Nc,Np)
        avg_template[ne] = X_temp.mean(axis=0)  # (Nc,Np)

        YY = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh)
        XX = np.zeros((n_chans, n_chans))  # (Nc,Nc)
        for tt in range(train_trials):
            XX += X_temp[tt] @ X_temp[tt].T
        XmXm = avg_template[ne] @ avg_template[ne].T  # (Nc,Nc)
        XmY = avg_template[ne] @ sine_template[ne].T  # (Nc,2Nh)

        # block covariance matrix S: [[S11,S12],[S21,S22]]
        S[ne,:n_chans,:n_chans] = XmXm  # S11
        S[ne,:n_chans,n_chans:] = (1-1/train_trials) * XmY  # S12
        S[ne,n_chans:,:n_chans] = S[ne,:n_chans,n_chans:].T  # S21
        S[ne,n_chans:,n_chans:] = YY  # S22

        # block covariance matrix Q: blkdiag(Q1,Q2)
        for ntr in range(n_train[ne]):
            Q[ne,:n_chans,:n_chans] += X_temp[ntr] @ X_temp[ntr].T  # Q1
        Q[ne,n_chans:,n_chans:] = train_trials * YY  # Q2

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
    uX, vY = [], []  # Ne*(Nk,Np)
    euX = np.zeros((n_events, u_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros_like(euX)
    if standard:
        for ne in range(n_events):
            uX.append(u[ne] @ avg_template[ne])  # (Nk,Np)
            vY.append(v[ne] @ sine_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            euX[ne] = u_concat @ avg_template[ne]  # (Nk*Ne,Np)
            evY[ne] = v_concat @ sine_template[ne]  # (Nk*Ne,Np)

    # sc-(e)TRCA model
    model = {
        'Q':Q, 'S':S,
        'u':u, 'v':v, 'u_concat':u_concat, 'v_concat':v_concat,
        'uX':uX, 'vY':vY, 'euX':euX, 'evY':evY
    }
    return model


class SC_TRCA(BasicTRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray):
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


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
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
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray):
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

