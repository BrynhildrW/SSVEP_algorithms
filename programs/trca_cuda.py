# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Use cupy module to accelerate some operation.

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

Notations:
    n_events: Ne
    n_train: Nt
    n_test: Nte
    train_trials: Ne*Nt
    test_trials: Ne*Nte
    n_chans: Nc
    n_points: Np
    n_components: Nk
    n_harmonics: Nh
    n_bands: Nb

update: 2023/10/17

"""

# %% Basic modules
import utils
import utils_cuda
import trca

from abc import abstractmethod, ABCMeta
from typing import Optional, List, Tuple, Any

import numpy as np
from numpy import ndarray
import cupy as cp
import scipy.linalg as sLA

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% 1. (ensemble) TRCA | (e)TRCA
def _trca_kernel(
    X_train: ndarray,
    y_train: ndarray,
    train_info: dict,
    n_components: int = 1) -> dict:
    """The modeling process of (e)TRCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int, optional): Number of eigenvectors picked as filters. Nk.

    Return: (e)TRCA model (dict)
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        w (ndarray): (Ne,Nk,Np). Spatial filters of TRCA.
        ew (ndarray): (Ne*Nk,Nc). Common spatial filter of eTRCA.
        wX (ndarray): (Ne,Nk,Np). TRCA templates.
        ewX (ndarray): (Ne,Ne*Nk,Np). eTRCA templates.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool

    # S & Q: covariance of template & original data
    S = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    Q = np.zeros_like(S)  # (Ne,Nc,Nc)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne,et in enumerate(event_type):
        temp = X_train[y_train==et]
        avg_template[ne] = np.mean(temp, axis=0)  # (Nc,Np)
        S[ne] = avg_template[ne] @ avg_template[ne].T
        for ntr in range(n_train[ne]):
            Q[ne] += temp[ntr] @ temp[ntr].T
    # S = np.einsum('ecp,ehp->ech', avg_template,avg_template)
    # Q = np.einsum('etcp,ethp->ech', train_data,train_data)

    # GEPs | train spatial filters
    # w, ndim = [], []
    w = np.zeros((n_events, n_components, n_chans))
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(
            A=S[ne],
            B=Q[ne],
            n_components=n_components,
            ratio=None
        )
        w[ne] = spatial_filter  # (Nk,Nc)
    ew = np.reshape(w, (n_events*n_components, n_chans), order='C')

    # signal templates
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    ewX = np.zeros((n_events, ew.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX[ne] = w[ne] @ avg_template[ne]  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = ew @ avg_template[ne]  # (Ne*Nk,Np)

    # (e)TRCA model
    training_model = {
        'Q':Q, 'S':S,
        'w':w, 'ew':ew,
        'wX':wX, 'ewX':ewX
    }
    return training_model


def _trca_feature(
    X_test: ndarray,
    training_model: dict,
    standard: bool,
    ensemble: bool) -> ndarray:
    """The pattern matching process of (e)TRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        training_model (dict): See details in _trca_kernel().
        standard (bool): Standard TRCA model. Defaults to True.
        ensemble (bool): Ensemble TRCA model. Defaults to True.

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of TRCA.
        erho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of eTRCA.
    """
    w, wX = cp.array(training_model['w']), cp.array(training_model['wX'])
    ew, ewX = cp.array(training_model['ew']), cp.array(training_model['ewX'])
    n_events, n_test = w.shape[0], X_test.shape[0]  # Ne, Ne*Nte
    rho = cp.zeros((n_test, n_events))
    erho = cp.zeros_like(rho)
    if standard:
        for nte in range(n_test):
            for nem in range(n_events):
                temp = w[nem] @ cp.array(X_test[nte])  # (Nk,Np)
                rho[nte,nem] = utils_cuda.pearson_corr(X=temp, Y=wX[nem], common_filter=False)
    if ensemble:
        for nte in range(n_test):
            temp = ew @ cp.array(X_test[nte])
            erho[nte,:] = utils_cuda.pearson_corr(X=temp, Y=ewX, common_filter=True)
    return rho.get(), erho.get()


class TRCA(trca.BasicTRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        """Train (e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
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
        self.training_model = _trca_kernel(
            X_train=self.X_train,
            y_train = self.y_train,
            train_info = self.train_info,
            n_components=self.n_components
        )
        return self


    def transform(self,
        X_test: ndarray) -> Tuple:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return: Tuple
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of TRCA.
                Not empty when self.standard is True.
            erho (ndarray): (Ne*Nte,Ne). Ensemble decision coefficients of eTRCA.
                Not empty when self.ensemble is True.
        """
        rho, erho = _trca_feature(
            X_test=X_test,
            training_model=self.training_model,
            standard=self.standard,
            ensemble=self.ensemble
        )
        return rho, erho


    def predict(self,
        X_test: ndarray) -> Tuple:
        """Using (e)TRCA algorithm to predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return: Tuple
            y_standard (ndarray): (Ne*Nte,). Predict labels of TRCA.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels of eTRCA.
        """
        # config cuda low-level modules
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        # output predicted labels
        self.rho, self.erho = self.transform(X_test)
        self.y_standard = self.train_info['event_type'][np.argmax(self.rho, axis=-1)]
        self.y_ensemble = self.train_info['event_type'][np.argmax(self.erho, axis=-1)]

        # free used GPU RAM
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return self.y_standard, self.y_ensemble


# %% 2. multi-stimulus (e)TRCA | ms-(e)TRCA
def _mstrca_kernel(
    X_train: ndarray,
    y_train: ndarray,
    train_info: dict,
    n_components: int = 1) -> dict:
    """The modeling process of ms-(e)TRCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True,
                            'events_group':{'event_id':[idx,]}}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Return: ms-(e)TRCA model (dict)
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        w (ndarray): (Ne,Nk,Nc). Spatial filters of ms-TRCA.
        ew (ndarray): (Ne*Nk,Nc). Common spatial filter of ms-eTRCA.
        wX (ndarray): (Ne,Nk,Np). ms-TRCA templates.
        ewX (ndarray): (Ne,Ne*Nk,Np). ms-eTRCA templates.
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

    # S & Q: same with TRCA
    total_S = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    total_Q = np.zeros_like(total_S)  # (Ne,Nc,Nc)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne,et in enumerate(event_type):
        temp = X_train[y_train==et]  # (Nt,Nc,Np)
        avg_template[ne] = np.mean(temp, axis=0)  # (Nc,Np)
        total_S[ne] = avg_template[ne] @ avg_template[ne].T
        for ntr in range(n_train[ne]):
            total_Q[ne] += temp[ntr] @ temp[ntr].T

    # GEPs with merged data
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        merged_indices = events_group[str(event_type[ne])]
        temp_Q = np.sum(total_Q[merged_indices,...], axis=0)  # (Nc,Nc)
        temp_S = np.sum(total_S[merged_indices,...], axis=0)  # (Nc,Nc)
        spatial_filter = utils.solve_gep(
            A=temp_S,
            B=temp_Q,
            n_components=n_components,
            ratio=None
        )
        w[ne] = spatial_filter
    ew = np.reshape(w, (n_events*n_components, n_chans), order='C')

    # signal templates
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    ewX = np.zeros((n_events, ew.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX[ne] = w[ne] @ avg_template[ne]  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = ew @ avg_template[ne]  # (Ne*Nk,Np)

    # ms-(e)TRCA model
    training_model = {
        'Q':total_Q, 'S':total_S,
        'w':w, 'ew':ew,
        'wX':wX, 'ewX':ewX
    }
    return training_model


class MS_TRCA(TRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        events_group: dict):
        """Train ms-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            events_group (dict): {'event_id':[idx,]}
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
            'ensemble':self.ensemble,
            'events_group':events_group
        }

        # train ms-TRCA models & templates
        self.training_model = _mstrca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            train_info=self.train_info,
            n_components=self.n_components
        )
        return self


# %% 3. (e)TRCA-R
def _trcar_kernel(
    X_train: ndarray,
    y_train: ndarray,
    projection: ndarray,
    train_info: dict,
    n_components: int = 1) -> dict:
    """The modeling process of (e)TRCA-R.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: (e)TRCA-R model (dict)
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        w (ndarray): (Ne,Nk,Nc). Spatial filters of ms-TRCA.
        ew (ndarray): (Ne*Nk,Nc). Common spatial filter of ms-eTRCA.
        wX (ndarray): (Ne,Nk,Np). ms-TRCA templates.
        ewX (ndarray): (Ne,Ne*Nk,Np). ms-eTRCA templates.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool

    # S & Q: covariance of projected template & original data
    S = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    Q = np.zeros_like(S)  # (Ne,Nc,Nc)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne,et in enumerate(event_type):
        temp = X_train[y_train==et]  # (Nt,Nc,Np)
        avg_template[ne] = np.mean(temp, axis=0)  # (Nc,Np)
        projected_template = avg_template[ne] @ projection[ne]  # (Nc,Np)
        S[ne] = projected_template @ projected_template.T
        for ntr in range(n_train[ne]):
            Q[ne] += temp[ntr] @ temp[ntr].T

    # GEPs with projected data
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(
            A=S[ne],
            B=Q[ne],
            n_components=n_components,
            ratio=None
        )
        w[ne] = spatial_filter
    ew = np.reshape(w, (n_events*n_components, n_chans), order='C')

    # signal templates
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    ewX = np.zeros((n_events, ew.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX[ne] = w[ne] @ avg_template[ne]  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = ew @ avg_template[ne]  # (Ne*Nk,Np)

    # (e)TRCA-R model
    training_model = {
        'Q':Q, 'S':S,
        'w':w, 'ew':ew,
        'wX':wX, 'ewX':ewX
    }
    return training_model


class TRCA_R(TRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        projection: ndarray):
        """Train (e)TRCA-R model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
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
        self.training_model = _trcar_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            projection=self.projection,
            train_info=self.train_info,
            n_components=self.n_components
        )
        return self


# %% 4. similarity constrained (e)TRCA | sc-(e)TRCA
def _sctrca_kernel(
    X_train: ndarray,
    y_train: ndarray,
    sine_template: ndarray,
    train_info: dict,
    n_components: int = 1) -> dict:
    """Training process of sc-(e)TRCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Return: sc-(e)TRCA model (dict).
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data & average template.
        S (ndarray): (Ne,Nc,Nc). Covariance of template.
        u (ndarray): (Ne,Nk,Nc). Spatial filters for EEG signal.
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters for sinusoidal signal.
        eu (ndarray): (Ne*Nk,Nc). Concatenated filter for EEG signal.
        ev (ndarray): (Ne*Nk,2*Nh). Concatenated filter for sinusoidal signal.
        uX (ndarray): (Ne,Nk,Np). sc-TRCA templates for EEG signal.
        vY (ndarray): (Ne,Nk,Np). sc-TRCA templates for sinusoidal signal.
        euX (ndarray): (Ne,Ne*Nk,Np). sc-eTRCA templates for EEG signal.
        evY (ndarray): (Ne,Ne*Nk,Np). sc-eTRCA templates for sinusoidal signal.
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

    # S & Q
    S = np.zeros((n_events, n_chans+n_2harmonics, n_chans+n_2harmonics))  # (Ne,Nc+2Nh,Nc+2Nh)
    Q = np.zeros_like(S)  # (Ne,Nc+2Nh,Nc+2Nh)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne,et in enumerate(event_type):
        train_trials = n_train[ne]  # Nt
        X_temp = X_train[y_train==et]  # (Nt,Nc,Np)
        avg_template[ne] = np.mean(X_temp, axis=0)  # (Nc,Np)

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

    # GEPs for EEG and sinusoidal templates
    u = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    v = np.zeros((n_events, n_components, n_2harmonics))  # (Ne,Nk,2Nh)
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(
            A=S[ne],
            B=Q[ne],
            n_components=n_components,
            ratio=None
        )
        u[ne] = spatial_filter[:,:n_chans]  # (Nk,Nc)
        v[ne] = spatial_filter[:,n_chans:]  # (Nk,2Nh)
    eu = np.reshape(u, (n_events*n_components, n_chans), order='C')  # (Ne*Nk,Nc)
    ev = np.reshape(v, (n_events*n_components, n_2harmonics), order='C')  # (Ne*Nk,2Nh)

    # signal templates
    uX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    vY = np.zeros_like(uX)  # (Ne,Nk,Np)
    euX = np.zeros((n_events, eu.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros_like(euX)  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            uX[ne] = u[ne] @ avg_template[ne]  # (Nk,Np)
            vY[ne] = v[ne] @ sine_template[ne]  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            euX[ne] = eu @ avg_template[ne]  # (Nk*Ne,Np)
            evY[ne] = ev @ sine_template[ne]  # (Nk*Ne,Np)

    # sc-(e)TRCA model
    training_model = {
        'Q':Q, 'S':S,
        'u':u, 'v':v, 'eu':eu, 'ev':ev,
        'uX':uX, 'vY':vY, 'euX':euX, 'evY':evY
    }
    return training_model


def _sctrca_feature(
    X_test: ndarray,
    training_model: dict,
    standard: bool,
    ensemble: bool) -> ndarray:
    """The pattern matching process of sc-(e)TRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        training_model (dict): See details in _sctrca_feature().
        standard (bool): sc-TRCA model. Defaults to True.
        ensemble (bool): sc-eTRCA model. Defaults to True.

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of sc-TRCA.
        erho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of eTRCA.
    """
    u, uX = cp.array(training_model['u']), cp.array(training_model['uX'])
    vY = cp.array(training_model['vY'])
    eu, euX = cp.array(training_model['eu']), cp.array(training_model['euX'])
    evY = cp.array(training_model['evY'])

    n_events, n_test = u.shape[0], X_test.shape[0]
    rho_eeg = cp.zeros((n_test, n_events))  # (Ne*Nte,Ne)
    erho_eeg = cp.zeros_like(rho_eeg)  # (Ne*Nte,Ne)
    rho_sin, erho_sin = cp.zeros_like(rho_eeg), cp.zeros_like(rho_eeg)
    rho, erho = cp.zeros_like(rho_eeg), cp.zeros_like(rho_eeg)

    # 2-step pattern matching
    if standard:
        for nte in range(n_test):
            for nem in range(n_events):
                temp = u[nem] @ cp.array(X_test[nte])  # (Nk,Np)
                rho_eeg[nte,nem] = utils_cuda.pearson_corr(X=temp, Y=uX[nem], common_filter=False)
                rho_sin[nte,nem] = utils_cuda.pearson_corr(X=temp, Y=vY[nem], common_filter=False)
                rho[nte,nem] = utils_cuda.combine_feature([rho_eeg[nte,nem], rho_sin[nte,nem]])
    if ensemble:
        for nte in range(n_test):
            temp = eu @ cp.array(X_test[nte])  # (Ne*Nk,Np)
            erho_eeg[nte,:] = utils_cuda.pearson_corr(X=temp, Y=euX, common_filter=True)
            erho_sin[nte,:] = utils_cuda.pearson_corr(X=temp, Y=evY, common_filter=True)
            erho[nte,:] = utils_cuda.combine_feature([rho_eeg[nte,:], rho_sin[nte,:]])
    return rho.get(), erho.get()


class SC_TRCA(TRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray):
        """Train sc-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
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

        # train sc-TRCA models & templates
        self.training_model = _sctrca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=sine_template,
            train_info=self.train_info,
            n_components=self.n_components
        )
        return self


    def transform(self,
        X_test: ndarray) -> Tuple:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return: Tuple
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of TRCA.
                Not empty when self.standard is True.
            erho (ndarray): (Ne*Nte,Ne). Ensemble decision coefficients of eTRCA.
                Not empty when self.ensemble is True.
        """
        rho, erho = _sctrca_feature(
            X_test=X_test,
            training_model=self.training_model,
            standard=self.standard,
            ensemble=self.ensemble
        )
        return rho, erho


# %%