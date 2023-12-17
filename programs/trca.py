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

update: 2023/07/04

"""

# %% Basic modules
import utils

from abc import abstractmethod, ABCMeta
from typing import Optional, List, Tuple, Any

import numpy as np
from numpy import ndarray
import scipy.linalg as sLA

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% Basic TRCA object
class BasicTRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1):
        """Basic configuration.

        Args:
            standard (bool): Standard TRCA model. Defaults to True.
            ensemble (bool): Ensemble TRCA model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        # config model
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble


    @abstractmethod
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: Optional[ndarray] = None):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, Optional): (Ne,2*Nh,Np). Sinusoidal templates.
        """
        pass


    @abstractmethod
    def transform(self,
        X_test: ndarray) -> Tuple:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            erho (ndarray): (Ne*Nte,Ne). Ensemble decision coefficients.
                Not empty when self.ensemble is True.
        """
        pass


    @abstractmethod
    def predict(self,
        X_test: ndarray) -> Tuple:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            y_standard (ndarray): (Ne*Nte,). Predict labels.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels (ensemble).
        """
        self.rho, self.erho = self.transform(X_test)
        self.y_standard, self.y_ensemble = np.empty(()), np.empty(())
        if self.standard:
            self.y_standard = self.event_type[np.argmax(self.rho, axis=-1)]
        if self.ensemble:
            self.y_ensemble = self.event_type[np.argmax(self.erho, axis=-1)]
        return self.y_standard, self.y_ensemble


class BasicFBTRCA(metaclass=ABCMeta):
    def __init__(self,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1):
        """Basic configuration.

        Args:
            standard (bool, optional): Standard TRCA model. Defaults to True.
            ensemble (bool, optional): Ensemble TRCA model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        # config model
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble


    @abstractmethod
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: Optional[ndarray] = None):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,...,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, Optional): (Nb,Ne,2*Nh,Np). Sinusoidal templates.
        """
        pass


    @abstractmethod
    def transform(self,
        X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
            rho (ndarray): (Nb,Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            erho (ndarray): (Nb,Ne*Nte,Ne). Ensemble decision coefficients.
                Not empty when self.ensemble is True.
        """
        pass


    def predict(self,
        X_test: ndarray) -> Tuple:
        """Using filter-bank algorithms to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
            y_standard (ndarray): (Ne*Nte,). Predict labels.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels (ensemble).
        """
        # apply model.predict() method in each sub-band
        self.fb_rho = [[] for nb in range(self.n_bands)]
        self.fb_erho = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.fb_rho[nb], self.fb_erho[nb] = self.sub_models[nb].transform(X_test=X_test[nb])

        # integration of multi-bands' results
        self.rho = utils.combine_fb_feature(self.fb_rho)
        self.erho = utils.combine_fb_feature(self.fb_erho)
        if self.standard:
            self.y_standard = self.event_type[np.argmax(self.rho, axis=-1)]
        if self.ensemble:
            self.y_ensemble = self.event_type[np.argmax(self.erho, axis=-1)]
        return self.y_standard, self.y_ensemble


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
    model = {
        'Q':Q, 'S':S,
        'w':w, 'ew':ew,
        'wX':wX, 'ewX':ewX
    }
    return model


def _trca_feature(
    X_test: ndarray,
    trca_model: dict,
    standard: bool,
    ensemble: bool) -> ndarray:
    """The pattern matching process of (e)TRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        trca_model (dict): See details in _trca_kernel().
        standard (bool): Standard TRCA model. Defaults to True.
        ensemble (bool): Ensemble TRCA model. Defaults to True.

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of TRCA.
        erho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of eTRCA.
    """
    w, wX = trca_model['w'], trca_model['wX']
    ew, ewX = trca_model['ew'], trca_model['ewX']
    n_events = w.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))
    erho = np.zeros_like(rho)
    if standard:
        for nte in range(n_test):
            for nem in range(n_events):
                temp_X = w[nem] @ X_test[nte]  # (Nk,Np)
                rho[nte,nem] = utils.pearson_corr(X=temp_X, Y=wX[nem])
    if ensemble:
        for nte in range(n_test):
            temp_X = ew @ X_test[nte]
            erho[nte,:] = utils.pearson_corr(X=temp_X, Y=ewX, common_filter=True)
    return rho, erho


class TRCA(BasicTRCA):
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
            trca_model=self.training_model,
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
        self.rho, self.erho = self.transform(X_test)
        self.y_standard = self.train_info['event_type'][np.argmax(self.rho, axis=-1)]
        self.y_ensemble = self.train_info['event_type'][np.argmax(self.erho, axis=-1)]
        return self.y_standard, self.y_ensemble


class FB_TRCA(BasicFBTRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        """Train filter-bank (e)TRCA model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
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


    def transform(self,
        X_test: ndarray) -> ndarray:
        pass


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
        w (ndarray): Ne*(Nk,Nc). Spatial filters of ms-TRCA.
        ew (ndarray): (Ne*Nk,Nc). Common spatial filter of ms-eTRCA.
        wX (ndarray): Ne*(Nk,Np). ms-TRCA templates.
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


class FB_MS_TRCA(BasicFBTRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        events_group: Optional[dict] = None,
        d: Optional[int] = 5):
        """Train filter-bank ms-(e)TRCA model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
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


class FB_TRCA_R(BasicFBTRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        projection: ndarray):
        """Train filter-bank (e)TRCA-R model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
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
    ratio: Optional[float] = None) -> dict:
    """(Ensemble) similarity-constrained TRCA (sc-(e)TRCA).

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
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: sc-(e)TRCA model (dict).
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data & average template.
        S (ndarray): (Ne,Nc,Nc). Covariance of template.
        u (List[ndarray]): Ne*(Nk,Nc). Spatial filters for EEG signal.
        v (List[ndarray]): Ne*(Nk,2*Nh). Spatial filters for sinusoidal signal.
        u_concat (ndarray): (Ne*Nk,Nc). Concatenated filter for EEG signal.
        v_concat (ndarray): (Ne*Nk,2*Nh). Concatenated filter for sinusoidal signal.
        uX (List[ndarray]): Ne*(Nk,Np). sc-TRCA templates for EEG signal.
        vY (List[ndarray]): Ne*(Nk,Np). sc-TRCA templates for sinusoidal signal.
        euX (List[ndarray]): (Ne,Ne*Nk,Np). sc-eTRCA templates for EEG signal.
        evY (List[ndarray]): (Ne,Ne*Nk,Np). sc-eTRCA templates for sinusoidal signal.
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

    # GEP | train spatial filters
    u, v, ndim, correct = [], [], [], [False for ne in range(n_events)]
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
        'uX':uX, 'vY':vY, 'euX':euX, 'evY':evY, 'correct':correct
    }
    return model


class SC_TRCA(BasicTRCA):
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
        model = sctrca_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=None
        )
        self.Q, self.S = model['Q'], model['S']
        self.u, self.v = model['u'], model['v']
        self.u_concat, self.v_concat = model['u_concat'], model['v_concat']
        self.uX, self.vY = model['uX'], model['vY']
        self.euX, self.evY = model['euX'], model['evY']
        self.correct = model['correct']
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using sc-(e)TRCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (Nt*Nte,Nc,Np). Test dataset.

        Return:
            rou (ndarray): (Nt*Nte,Ne). Decision coefficients of sc-TRCA.
                Not empty when self.standard is True.
            y_standard (ndarray): (Nt*Nte,). Predict labels of sc-TRCA.
            erou (ndarray): (Nt*Nte,Ne). Decision coefficients of sc-eTRCA.
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (Nt*Nte,). Predict labels of sc-eTRCA.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']
        event_type = self.train_info['event_type']

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
                self.y_standard[nte] = event_type[np.argmax(self.rou[nte,:])]
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
                self.y_ensemble[nte] = event_type[np.argmax(self.erou[nte,:])]
        return self.rou, self.y_standard, self.erou, self.y_ensemble


class FB_SC_TRCA(BasicFBTRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray):
        """Train filter-bank sc-(e)TRCA model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
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
