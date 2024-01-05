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
    6. xTRCA:
            DOI:
    7. LA-TRCA:
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

"""

# %% Basic modules
import utils

from abc import abstractmethod
from typing import Optional, List, Tuple, Dict

import numpy as np
from numpy import ndarray

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% Basic TRCA object
class BasicTRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
        self,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        # config model
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble

    @abstractmethod
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: Optional[ndarray] = None
    ):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np).
                Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, Optional): (Ne,2*Nh,Np).
                Sinusoidal templates.
        """
        pass

    @abstractmethod
    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients.
            erho (ndarray): (Ne*Nte,Ne). Ensemble decision coefficients.
        """
        pass

    def predict(self, X_test: ndarray) -> Tuple[ndarray, ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Tuple[ndarray, ndarray]
            y_standard (ndarray): (Ne*Nte,). Predict labels.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels (ensemble).
        """
        self.features = self.transform(X_test)
        event_type = self.train_info['event_type']
        self.y_standard = event_type[np.argmax(self.features['rho'], axis=-1)]
        self.y_ensemble = event_type[np.argmax(self.features['erho'], axis=-1)]
        return self.y_standard, self.y_ensemble


class BasicFBTRCA(utils.FilterBank, ClassifierMixin):
    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np) or (Ne*Nte,Nc,Np).
                Test dataset.

        Returns: Dict[str, ndarray]
            fb_rho (ndarray): (Nb,Ne*Nte,Ne). Decision coefficients of each band.
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of all bands.
            fb_erho (ndarray): (Nb,Ne*Nte,Ne). Decision ensemble coefficients.
            erho (ndarray): (Ne*Nte,Ne). Decision ensemble coefficients of all bands.
        """
        if not self.with_filter_bank:  # tranform X_test
            X_test = self.fb_transform(X_test)  # (Nb,Ne*Nte,Nc,Np)
        sub_features = [se.transform(X_test[nse])
                        for nse, se in enumerate(self.sub_estimator)]
        fb_rho = np.stack([sf['rho'] for sf in sub_features], axis=0)
        fb_erho = np.stack([sf['erho'] for sf in sub_features], axis=0)
        rho = np.einsum('b,bte->te', self.bank_weights, fb_rho)
        erho = np.einsum('b,bte->te', self.bank_weights, fb_erho)
        features = {
            'fb_rho': fb_rho, 'rho': rho,
            'fb_erho': fb_erho, 'erho': erho
        }
        return features

    def predict(self, X_test: ndarray) -> Tuple[ndarray, ndarray]:
        """Using filter-bank TRCA-like algorithm to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np) or (Ne*Nte,Nc,Np).
                Test dataset.

        Returns: Tuple[ndarray, ndarray]
            y_standard (ndarray): (Ne*Nte,). Predict labels of standard algorithm.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels of ensemble algorithm.
        """
        self.features = self.transform(X_test)
        event_type = self.sub_estimator[0].train_info['event_type']
        self.y_standard = event_type[np.argmax(self.features['rho'], axis=-1)]
        self.y_ensemble = event_type[np.argmax(self.features['erho'], axis=-1)]
        return self.y_standard, self.y_ensemble


# %% 1. (ensemble) TRCA | (e)TRCA
def _trca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
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
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, ndarray]
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
    X_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        temp = X_train[y_train == et]
        X_mean[ne] = np.mean(temp, axis=0)  # (Nc,Np)
        S[ne] = X_mean[ne] @ X_mean[ne].T
        for ntr in range(n_train[ne]):
            Q[ne] += temp[ntr] @ temp[ntr].T
    # S = np.einsum('ecp,ehp->ech', X_mean,X_mean)
    # Q = np.einsum('etcp,ethp->ech', train_data,train_data)

    # GEPs | train spatial filters
    w = np.zeros((n_events, n_components, n_chans))
    for ne in range(n_events):
        w[ne] = utils.solve_gep(A=S[ne], B=Q[ne], n_components=n_components)
    ew = np.reshape(w, (n_events*n_components, n_chans), order='C')

    # signal templates
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    ewX = np.zeros((n_events, ew.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX[ne] = w[ne] @ X_mean[ne]  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = ew @ X_mean[ne]  # (Ne*Nk,Np)

    # (e)TRCA model
    training_model = {
        'Q': Q, 'S': S,
        'w': w, 'ew': ew,
        'wX': wX, 'ewX': ewX
    }
    return training_model


def _trca_feature(
        X_test: ndarray,
        trca_model: Dict[str, ndarray],
        standard: bool = True,
        ensemble: bool = True) -> Dict[str, ndarray]:
    """The pattern matching process of (e)TRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        trca_model (Dict[str, ndarray]): See details in _trca_kernel().
        standard (bool): Standard model. Defaults to True.
        ensemble (bool): Ensemble model. Defaults to True.

    Returns: Dict[str, ndarray]
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
                X_temp = w[nem] @ X_test[nte]  # (Nk,Np)
                rho[nte, nem] = utils.pearson_corr(X=X_temp, Y=wX[nem])
    if ensemble:
        for nte in range(n_test):
            X_temp = ew @ X_test[nte]
            erho[nte, :] = utils.pearson_corr(X=X_temp, Y=ewX, parallel=True)
    features = {
        'rho': rho, 'erho': erho
    }
    return features


class TRCA(BasicTRCA):
    def fit(self, X_train: ndarray, y_train: ndarray):
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
            'event_type': event_type,
            'n_events': len(event_type),
            'n_train': np.array([np.sum(self.y_train == et)
                                 for et in event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
            'standard': self.standard,
            'ensemble': self.ensemble
        }

        # train TRCA filters & templates
        self.training_model = _trca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            train_info=self.train_info,
            n_components=self.n_components
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of TRCA.
            erho (ndarray): (Ne*Nte,Ne). Ensemble decision coefficients of eTRCA.
        """
        return _trca_feature(
            X_test=X_test,
            trca_model=self.training_model,
            standard=self.standard,
            ensemble=self.ensemble
        )


class FB_TRCA(BasicFBTRCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble
        super().__init__(
            base_estimator=TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components
            ),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 2. multi-stimulus (e)TRCA | ms-(e)TRCA
def _mstrca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
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
                            'events_group':{'event_id':[idx_1,idx_2,...]}}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, ndarray]
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
    S_total = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    Q_total = np.zeros_like(S_total)  # (Ne,Nc,Nc)
    X_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        temp = X_train[y_train == et]  # (Nt,Nc,Np)
        X_mean[ne] = np.mean(temp, axis=0)  # (Nc,Np)
        S_total[ne] = X_mean[ne] @ X_mean[ne].T
        for ntr in range(n_train[ne]):
            Q_total[ne] += temp[ntr] @ temp[ntr].T

    # GEPs with merged data
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        merged_indices = events_group[str(event_type[ne])]
        Q_temp = np.sum(Q_total[merged_indices], axis=0)  # (Nc,Nc)
        S_temp = np.sum(S_total[merged_indices], axis=0)  # (Nc,Nc)
        w[ne] = utils.solve_gep(A=S_temp, B=Q_temp, n_components=n_components)
    ew = np.reshape(w, (n_events*n_components, n_chans), order='C')

    # signal templates
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    ewX = np.zeros((n_events, ew.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX[ne] = w[ne] @ X_mean[ne]  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = ew @ X_mean[ne]  # (Ne*Nk,Np)

    # ms-(e)TRCA model
    training_model = {
        'Q': Q_total, 'S': S_total,
        'w': w, 'ew': ew,
        'wX': wX, 'ewX': ewX
    }
    return training_model


class MS_TRCA(TRCA):
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        events_group: Dict[str, List[int]] = None,
        d: int = 2
    ):
        """Train ms-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            events_group (Dict[str, List[int]], optional): {'event_id':[idx_1,idx_2,...]}.
                If None, events_group will be generated according to parameter 'd'.
            d (int): The range of events to be merged.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.d = d
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        if events_group is not None:
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(event_type=event_type, d=self.d)
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_train': np.array([np.sum(self.y_train == et)
                                 for et in event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
            'standard': self.standard,
            'ensemble': self.ensemble,
            'events_group': self.events_group
        }

        # train ms-TRCA models & templates
        self.training_model = _mstrca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            train_info=self.train_info,
            n_components=self.n_components
        )


class FB_MS_TRCA(BasicFBTRCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble
        super().__init__(
            base_estimator=MS_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components
            ),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 3. (e)TRCA-R
def _trcar_kernel(
        X_train: ndarray,
        y_train: ndarray,
        projection: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
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

    Returns: Dict[str, ndarray]
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
    X_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        X_temp = X_train[y_train == et]  # (Nt,Nc,Np)
        X_mean[ne] = np.mean(X_temp, axis=0)  # (Nc,Np)
        projected_template = X_mean[ne] @ projection[ne]  # (Nc,Np)
        S[ne] = projected_template @ projected_template.T
        for ntr in range(n_train[ne]):
            Q[ne] += X_temp[ntr] @ X_temp[ntr].T

    # GEPs with projected data
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        w[ne] = utils.solve_gep(A=S[ne], B=Q[ne], n_components=n_components)
    ew = np.reshape(w, (n_events*n_components, n_chans), order='C')

    # signal templates
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    ewX = np.zeros((n_events, ew.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        for ne in range(n_events):
            wX[ne] = w[ne] @ X_mean[ne]  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            ewX[ne] = ew @ X_mean[ne]  # (Ne*Nk,Np)

    # (e)TRCA-R model
    training_model = {
        'Q': Q, 'S': S,
        'w': w, 'ew': ew,
        'wX': wX, 'ewX': ewX
    }
    return training_model


class TRCA_R(TRCA):
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        projection: ndarray
    ):
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
            'event_type': event_type,
            'n_events': len(event_type),
            'n_train': np.array([np.sum(self.y_train == et)
                                 for et in event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
            'standard': self.standard,
            'ensemble': self.ensemble
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


class FB_TRCA_R(BasicFBTRCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble
        super().__init__(
            base_estimator=TRCA_R(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components
            ),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 4. similarity constrained (e)TRCA | sc-(e)TRCA
def _sctrca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The modeling process of sc-(e)TRCA.

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

    Returns: Dict[str, ndarray]
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data & average template.
        S (ndarray): (Ne,Nc,Nc). Covariance of template.
        u (ndarray): (Ne,Nk,Nc). Spatial filters (EEG signal).
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters (sinusoidal signal).
        eu (ndarray): (Ne*Nk,Nc). Concatenated filter (EEG signal)..
        ev (ndarray): (Ne*Nk,2*Nh). Concatenated filter for sinusoidal signal.
        uX (ndarray): (Ne,Nk,Np). sc-TRCA templates (EEG signal).
        vY (ndarray): (Ne,Nk,Np). sc-TRCA templates (sinusoidal signal).
        euX (ndarray): (Ne,Ne*Nk,Np). sc-eTRCA templates (EEG signal).
        evY (ndarray): (Ne,Ne*Nk,Np). sc-eTRCA templates (sinusoidal signal).
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool
    n_dims = sine_template.shape[1]  # 2*Nh

    # covariance matrices: (Ne,Nc+2Nh,Nc+2Nh)
    S = np.zeros((n_events, n_chans+n_dims, n_chans+n_dims))
    Q = np.zeros_like(S)  # (Ne,Nc+2Nh,Nc+2Nh)
    X_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        train_trials = n_train[ne]  # Nt
        X_temp = X_train[y_train == et]  # (Nt,Nc,Np)
        X_mean[ne] = np.mean(X_temp, axis=0)  # (Nc,Np)

        Cyy = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh)
        Cxx = np.zeros((n_chans, n_chans))  # (Nc,Nc)
        for tt in range(train_trials):
            Cxx += X_temp[tt] @ X_temp[tt].T
        Cxmxm = X_mean[ne] @ X_mean[ne].T  # (Nc,Nc)
        Cxmy = X_mean[ne] @ sine_template[ne].T  # (Nc,2Nh)

        # block covariance matrix S: [[S11,S12],[S21,S22]]
        S[ne, :n_chans, :n_chans] = Cxmxm  # S11
        S[ne, :n_chans, n_chans:] = (1 - 1 / train_trials) * Cxmy  # S12
        S[ne, n_chans:, :n_chans] = S[ne, :n_chans, n_chans:].T  # S21
        S[ne, n_chans:, n_chans:] = Cyy  # S22

        # block covariance matrix Q: blkdiag(Q1,Q2)
        Q[ne, :n_chans, :n_chans] = Cxx  # Q1
        Q[ne, n_chans:, n_chans:] = train_trials * Cyy  # Q2

    # GEPs | train spatial filters
    u = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    v = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,2Nh)
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(A=S[ne], B=Q[ne], n_components=n_components)
        u[ne] = spatial_filter[:, :n_chans]  # (Nk,Nc)
        v[ne] = spatial_filter[:, n_chans:]  # (Nk,2Nh)
    eu = np.reshape(u, (n_events*n_components, n_chans), order='C')
    ev = np.reshape(v, (n_events*n_components, n_dims), order='C')

    # signal templates
    uX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    vY = np.zeros_like(uX)
    euX = np.zeros((n_events, eu.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros_like(euX)
    if standard:
        for ne in range(n_events):
            uX[ne] = u[ne] @ X_mean[ne]  # (Nk,Np)
            vY[ne] = v[ne] @ sine_template[ne]  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            euX[ne] = eu @ X_mean[ne]  # (Nk*Ne,Np)
            evY[ne] = ev @ sine_template[ne]  # (Nk*Ne,Np)

    # sc-(e)TRCA model
    training_model = {
        'Q': Q, 'S': S,
        'u': u, 'v': v, 'eu': eu, 'ev': ev,
        'uX': uX, 'vY': vY, 'euX': euX, 'evY': evY
    }
    return training_model


def _sctrca_feature(
        X_test: ndarray,
        sctrca_model: Dict[str, ndarray],
        standard: bool,
        ensemble: bool) -> Dict[str, ndarray]:
    """The pattern matching process of sc-(e)TRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        sctrca_model (dict): See details in _sctrca_kernel().
        standard (bool): Standard model. Defaults to True.
        ensemble (bool): Ensemble model. Defaults to True.

    Returns: Dict[str, ndarray]
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of sc-TRCA.
        rho_eeg (ndarray): (Ne*Nte,Ne). EEG part rho.
        rho_sin (ndarray): (Ne*Nte,Ne). Sinusoidal signal part rho.
        erho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of sc-eTRCA.
        erho_eeg (ndarray): (Ne*Nte,Ne). EEG part erho.
        erho_sin (ndarray): (Ne*Nte,Ne). Sinusoidal signal part erho.
    """
    u, eu = sctrca_model['u'], sctrca_model['eu']
    uX, euX = sctrca_model['uX'], sctrca_model['euX']
    vY, evY = sctrca_model['vY'], sctrca_model['evY']
    n_events = u.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte

    rho = np.zeros((n_test, n_events))  # (Ne*Nte,Ne)
    rho_eeg, rho_sin = np.zeros_like(rho), np.zeros_like(rho)
    erho = np.zeros_like(rho)
    erho_eeg, erho_sin = np.zeros_like(rho), np.zeros_like(rho)
    if standard:
        for nte in range(n_test):
            for nem in range(n_events):
                X_temp = u[nem] @ X_test[nte]
                rho_eeg[nte, nem] = utils.pearson_corr(X=X_temp, Y=uX[nem])
                rho_sin[nte, nem] = utils.pearson_corr(X=X_temp, Y=vY[nem])
        rho = utils.combine_feature([rho_eeg, rho_sin])
    if ensemble:
        for nte in range(n_test):
            X_temp = eu @ X_test[nte]
            erho_eeg[nte] = utils.pearson_corr(X=X_temp, Y=euX, parallel=True)
            erho_sin[nte] = utils.pearson_corr(X=X_temp, Y=evY, parallel=True)
        erho = utils.combine_feature([erho_eeg, erho_sin])
    features = {
        'rho': rho, 'rho_eeg': rho_eeg, 'rho_sin': rho_sin,
        'erho': erho, 'erho_eeg': erho_eeg, 'erho_sin': erho_sin
    }
    return features


class SC_TRCA(BasicTRCA):
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray
    ):
        """Train sc-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_train': np.array([np.sum(self.y_train == et)
                                 for et in event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
            'standard': self.standard,
            'ensemble': self.ensemble
        }

        # train sc-TRCA models & templates
        self.training_model = _sctrca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=sine_template,
            train_info=self.train_info,
            n_components=self.n_components
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of sc-TRCA.
            rho_eeg (ndarray): (Ne*Nte,Ne). EEG part rho.
            rho_sin (ndarray): (Ne*Nte,Ne). Sinusoidal signal part rho.
            erho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of sc-eTRCA.
            erho_eeg (ndarray): (Ne*Nte,Ne). EEG part erho.
            erho_sin (ndarray): (Ne*Nte,Ne). Sinusoidal signal part erho.
        """
        return _sctrca_feature(
            X_test=X_test,
            sctrca_model=self.training_model,
            standard=self.standard,
            ensemble=self.ensemble
        )


class FB_SC_TRCA(BasicFBTRCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray,], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble
        super().__init__(
            base_estimator=SC_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components
            ),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 5. two-stage CORRCA | TS-CORRCA


# %% 6. group TRCA | gTRCA


# %% 7. cross-correlation TRCA | xTRCA


# %% 8. latency-aligned TRCA | LA-TRCA
