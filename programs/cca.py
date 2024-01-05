# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Canonical correlation analysis (CCA) series.
    1. CCA: http://ieeexplore.ieee.org/document/4203016/
            DOI: 10.1109/TBME.2006.889197 (unsupervised)
    2. MEC: http://ieeexplore.ieee.org/document/4132932/
            DOI: 10.1109/TBME.2006.889160 (unsupervised)
    3. MCC: http://ieeexplore.ieee.org/document/4132932/
            DOI: 10.1109/TBME.2006.889160 (unsupervised)
    4. MSI: https://linkinghub.elsevier.com/retrieve/pii/S0165027013002677
            DOI: 10.1016/j.jneumeth.2013.07.018 (unsupervised)
    5. tMSI:
            DOI:  (unsupervised)
    6. eMSI:
            DOI:  (unsupervised)
    7. itCCA: https://iopscience.iop.org/article/10.1088/1741-2560/8/2/025015
            DOI: 10.1088/1741-2560/8/2/025015
    8. eCCA: http://www.pnas.org/lookup/doi/10.1073/pnas.1508080112
            DOI: 10.1073/pnas.1508080112
    9. msCCA: https://ieeexplore.ieee.org/document/9006809/
            DOI: 10.1109/TBME.2020.2975552
    10. ms-eCCA: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
            DOI: 10.1088/1741-2552/ab2373
    11. MsetCCA1: https://www.worldscientific.com/doi/abs/10.1142/S0129065714500130
            DOI: 10.1142/S0129065714500130
    12. MsetCCA2: https://ieeexplore.ieee.org/document/8231203/
            DOI: 10.1109/TBME.2017.2785412
    13. MwayCCA:
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

# %% basic modules
import utils

from abc import abstractmethod
from typing import Optional, List, Dict, Union

import numpy as np
from numpy import ndarray
import scipy.linalg as sLA
from scipy.sparse import block_diag

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% Basic CCA object
class BasicCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1):
        """Basic configuration.

        Args:
            n_components (int): Number of eigenvectors picked as filters.
        """
        # config model
        self.n_components = n_components

    @abstractmethod
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: Optional[ndarray] = None
    ):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, Optional): (Ne,2*Nh,Np). Sinusoidal templates.
        """
        pass

    @abstractmethod
    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients.
        """
        pass

    def predict(self, X_test: ndarray) -> Union[int, ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            y_pred (Union[int, ndarray]): int or (Ne*Nte,). Predict labels.
        """
        self.rho = self.transform(X_test)
        event_type = self.train_info['event_type']
        self.y_pred = event_type[np.argmax(self.rho, axis=-1)]
        return self.y_pred


class BasicFBCCA(utils.FilterBank, ClassifierMixin):
    def predict(self, X_test: ndarray) -> Union[int, ndarray]:
        """Using filter-bank algorithms to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Returns:
            y_pred (Union[int, ndarray]): int or (Ne*Nte,). Predict labels.
        """
        self.features = self.transform(X_test)
        event_type = self.sub_estimator[0].train_info['event_type']
        self.y_pred = event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


# %% 1. standard CCA | CCA
def _cca_kernel(
        X: ndarray,
        Y: ndarray,
        n_components: int = 1) -> Dict[str, Union[int, ndarray]]:
    """The modeling process of CCA.

    Args:
        X (ndarray): (Nc,Np).
        Y (ndarray): (2Nh,Np).
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, Union[int, ndarray]]
        Cxx (ndarray): (Nc,Nc). Covariance of X.
        Cxy (ndarray): (Nc,2Nh). Covariance of X & Y.
        Cyy (ndarray): (2Nh,2Nh). Covariance of Y.
        u (ndarray): (Nk,Nc). Spatial filter for X.
        v (ndarray): (Nk,2Nh). Spatial filter for Y.
        uX (ndarray): (Nk,Np). Filtered X
        vY (ndarray): (Nk,Np). Filtered Y.
        coef (float): corr(uX, vY).
    """
    # covariance matrices
    Cxx = X @ X.T
    Cxy = X @ Y.T
    Cyy = Y @ Y.T

    # GEPs | train spatial filters
    u = utils.solve_gep(
        A=Cxy @ sLA.solve(Cyy, Cxy.T),
        B=Cxx,
        n_components=n_components
    )
    v = utils.solve_gep(
        A=Cxy.T @ sLA.solve(Cxx, Cxy),
        B=Cyy,
        n_components=n_components
    )

    # signal templates & positive-negative correction
    uX, vY = u @ X, v @ Y
    cca_coef = utils.pearson_corr(X=uX, Y=vY)
    if cca_coef < 0:
        v *= -1
        vY *= -1

    # CCA model
    training_model = {
        'Cxx': Cxx, 'Cxy': Cxy, 'Cyy': Cyy,
        'u': u, 'v': v,
        'uX': uX, 'vY': vY, 'coef': cca_coef
    }
    return training_model


def _cca_feature(
        X_test: ndarray,
        template: ndarray,
        train_info: dict,
        n_components: int = 1) -> ndarray:
    """The pattern matching process of CCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        template (ndarray): (Ne,2Nh,Np). Signal templates.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int'}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of CCA.
    """
    n_events = train_info['n_events']  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))  # (Ne*Nte,Ne)
    for nte in range(n_test):
        for nem in range(n_events):
            cca_model = _cca_kernel(
                X=X_test[nte],
                Y=template[nem],
                n_components=n_components
            )
            rho[nte, nem] = cca_model['coef']
    return rho


class CCA(BasicCCA):
    def fit(
        self,
        sine_template: ndarray,
        X_train: Optional[ndarray] = None,
        y_train: Optional[ndarray] = None
    ):
        """Train CCA model.

        Args:
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
            X_train (Nonetype): To meet the input requirements of FilterBank.fit().
            y_train (ndarray): (Ne*Nt,). Labels for dataset.
                If None, y_train will be set to np.arange(n_events).
        """
        self.sine_template = sine_template
        self.X_train = X_train
        self.y_train = y_train
        if self.y_train is not None:
            event_type = np.unique(self.y_train)
        else:
            event_type = np.arange(self.sine_template.shape[0])
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_points': self.sine_template.shape[-1]
        }
        return self

    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of CCA.
        """
        return _cca_feature(
            X_test=X_test,
            template=self.sine_template,
            train_info=self.train_info,
            n_components=self.n_components
        )


class FB_CCA(BasicFBCCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=CCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 2. Minimum Energy Combination | MEC
def mec_kernel(
        X_test: ndarray,
        sine_template: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The modeling process of MEC.

    Args:
        X_test (ndarray): (Nc,Np). Single-trial test data.
        sine_template (ndarray): (Ne,2Nh,Np). Sinusoidal templates.
        train_info (dict): {'n_events':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, ndarray]
        X_hat (ndarray): (Ne,Nc,Np). X_train after removing SSVEP components.
        projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
        Cxhxh (ndarray): (Ne,Nc,Nc). Covariance of X_hat.
        w (ndarray): (Ne,Nk,Nc). Spatial filters.
        wX (ndarray): (Ne,Nk,Np). Filtered EEG signal
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    n_points = train_info['n_points']  # Np
    n_chans = X_test.shape[0]  # Nc

    # generate orthogonal projection & remove SSVEP components
    X_hat = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    projection = np.zeros((n_events, n_points, n_points))  # (Ne,Np,Np)
    Cxhxh = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    for ne in range(n_events):
        projection[ne] = sine_template[ne].T @ sLA.solve(
            a=sine_template[ne] @ sine_template[ne].T,
            b=sine_template[ne],
            assume_a='sym'
        )
        X_hat[ne] = X_test[ne] - X_test[ne] @ projection[ne]  # (Nc,Np)
        Cxhxh[ne] = X_hat[ne] @ X_hat[ne].T

    # GEPs | train spatial filters
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        w[ne] = utils.solve_ep(A=Cxhxh, n_components=n_components, mode='Min')

    # signal templates
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    for ne in range(n_events):
        wX[ne] = w[ne] @ X_test

    # MEC model
    training_model = {
        'X_hat': X_hat, 'projection': projection, 'Cxhxh': Cxhxh,
        'w': w, 'wX': wX
    }
    return training_model


# %% 3. Maximum Contrast Combination | MCC
def mcc_kernel(
        X_test: ndarray,
        sine_template: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The modeling process of MCC.

    Args:
        X_test (ndarray): (Nc,Np). Single-trial test data.
        sine_template (ndarray): (Ne,2Nh,Np). Sinusoidal templates.
        train_info (dict): {'n_events':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, ndarray]
        X_hat (ndarray): (Ne,Nc,Np). X_train after removing SSVEP components.
        projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
        Cxhxh (ndarray): (Ne,Nc,Nc). Covariance of X_hat.
        Cxx (ndarray): (Nc,Nc). Covariance of X_train.
        w (ndarray): (Ne,Nk,Nc). Spatial filters.
        wX (ndarray): (Ne,Nk,Np). Filtered EEG signal
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    n_points = train_info['n_points']  # Np
    n_chans = X_test.shape[0]  # Nc

    # generate orthogonal projection & remove SSVEP components
    X_hat = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    projection = np.zeros((n_events, n_points, n_points))  # (Ne,Np,Np)
    Cxhxh = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    for ne in range(n_events):
        projection[ne] = sine_template[ne].T @ sLA.solve(
            a=sine_template[ne] @ sine_template[ne].T,
            b=sine_template[ne],
            assume_a='sym'
        )
        X_hat[ne] = X_test[ne] - X_test[ne] @ projection[ne]  # (Nc,Np)
        Cxhxh[ne] = X_hat[ne] @ X_hat[ne].T
    Cxx = X_test @ X_test.T  # (Nc,Nc)

    # GEPs | train spatial filters
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        w[ne] = utils.solve_gep(A=Cxx, B=Cxhxh, n_components=n_components)

    # signal templates
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    for ne in range(n_events):
        wX[ne] = w[ne] @ X_test

    # MEC model
    training_model = {
        'X_hat': X_hat, 'projection': projection,
        'Cxhxh': Cxhxh, 'Cxx': Cxx,
        'w': w, 'wX': wX
    }
    return training_model


# %% 4. Multivariant synchronization index | MSI
def _msi_kernel(
        train_info: dict,
        X_test: ndarray,
        sine_template: ndarray) -> Dict[str, ndarray]:
    """The modeling process of MSI.

    Args:
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_points':int}
        X_test (ndarray): (Nc,Np). Single-trial test data.
        sine_template (ndarray): (Ne,2Nh,Np). Sinusoidal templates.

    Returns: Dict[str, ndarray]
        Cxx (ndarray): (Nc,Nc). Covariance of X_train.
        Cyy (ndarray): (Ne,2Nh,2Nh). Covariance of sine_template.
        Cxy (ndarray): (Ne,Nc,2Nh). Covariance of X_train & sine_template.
        R (ndarray): (Ne,Nc+2Nh,Nc+2Nh). Linear-transformed correlation matrix.
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    n_chans = X_test.shape[0]  # Nc
    n_points = train_info['n_points']  # Np
    n_dims = sine_template.shape[1]  # 2Nh

    # covariance of X_train & sine_template
    Cxx = X_test @ X_test.T / n_points  # (Nc,Nc)
    Cyy = np.zeros((n_events, n_dims, n_dims))  # (Ne,2Nh,2Nh)
    Cxy = np.zeros((n_events, n_chans, n_dims))  # (Ne,Nc,2Nh)
    for ne in range(n_events):
        Cyy[ne] = sine_template[ne] @ sine_template[ne].T
        Cxy[ne] = X_test @ sine_template[ne].T
    Cyy, Cxy = Cyy / n_points, Cxy / n_points

    # R: linear-transformed correlation matrix
    R = np.tile(np.eye((n_chans + n_dims))[None, ...], (n_events, 1, 1))  # (Ne,Nc+2Nh,Nc+2Nh)
    for ne in range(n_events):
        R[ne, :n_chans, n_chans:] = utils.nega_root(Cxx) @ Cxy[ne] @ utils.nega_root(Cyy[ne])
        R[ne, n_chans:, :n_chans] = R[ne, :n_chans, n_chans:].T

    # MSI model
    training_model = {
        'Cxx': Cxx, 'Cyy': Cyy, 'Cxy': Cxy, 'R': R
    }
    return training_model


def _msi_feature(
        train_info: dict,
        X_test: ndarray,
        sine_template: ndarray) -> ndarray:
    """The pattern matching process of MSI.

    Args:
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int}
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        sine_template (ndarray): (Ne,2Nh,Np). Sinusoidal templates.

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of MSI.
    """
    n_events = sine_template.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))  # (Ne*Nte,Ne)
    for nte in range(n_test):
        msi_model = _msi_kernel(
            train_info=train_info,
            X_test=X_test[nte],
            sine_template=sine_template,
        )
        R = msi_model['R']  # (Ne,Nc+2Nh,Nc+2Nh)
        for nem in range(n_events):
            rho[nte, nem] = utils.s_estimator(R[nem])
    return rho


class MSI(BasicCCA):
    def fit(
        self,
        sine_template: ndarray,
        X_train: Optional[ndarray] = None,
        y_train: Optional[ndarray] = None
    ):
        """Train MSI model.

        Args:
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
            X_train (Nonetype): To meet the input requirements of FilterBank.fit().
            y_train (ndarray): (Ne*Nt,). Labels for dataset.
                If None, y_train will be set to np.arange(n_events).
        """
        self.sine_template = sine_template
        self.X_train = X_train
        self.y_train = y_train
        if self.y_train is not None:
            event_type = np.unique(self.y_train)
        else:
            event_type = np.arange(self.sine_template.shape[0])
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_points': self.sine_template.shape[-1]
        }
        return self

    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Nc,Np). Single-trial test data.

        Returns: Dict[str, ndarray]
            rho (ndarray): (Ne,). Decision coefficients of MSI.
        """
        return _msi_feature(
            train_info=self.train_info,
            X_test=X_test,
            sine_template=self.sine_template
        )


class FB_MSI(BasicFBCCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=MSI(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 5. tMSI


# %% 6. extend-MSI | eMSI


# %% 7. Individual template CCA | itCCA
class ITCCA(BasicCCA):
    def fit(self, X_train: ndarray, y_train: ndarray):
        """Train itCCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(self.y_train)
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_chans': self.X_train.shape[1],
            'n_points': self.X_train.shape[-1]
        }
        self.avg_template = np.zeros((
            self.train_info['n_events'],
            self.train_info['n_chans'],
            self.train_info['n_points']
        ))
        for ne, et in enumerate(event_type):
            self.avg_template[ne] = np.mean(X_train[y_train == et], axis=0)
        return self

    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of itCCA.
        """
        return _cca_feature(
            X_test=X_test,
            template=self.avg_template,
            n_components=self.n_components
        )


class FB_ITCCA(BasicFBCCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=ITCCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 8. Extended CCA | eCCA
def _ecca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        X_test: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
    """CCA with individual calibration data.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        X_test (ndarray): (Nc,Np). Single-trial test data.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, ndarray]
        u_xy (ndarray): (Ne,Nk,Nc). Spatial filters of CCA(X_test, sine_template).
        v_xy (ndarray): (Ne,Nk,2Nh). Spatial filters of CCA(X_test, sine_template).
        u_xa (ndarray): (Ne,Nk,Nc). Spatial filters of CCA(X_test, avg_template).
        v_xa (ndarray): (Ne,Nk,Nc). Spatial filters of CCA(X_test, avg_template).
        u_ay (ndarray): (Ne,Nk,Nc). Spatial filters of CCA(avg_template, sine_template).
        v_ay (ndarray): (Ne,Nk,2Nh). Spatial filters of CCA(avg_template, sine_template).
        avg_template (ndarray): (Ne,Nc,Np). Trial-averaged template of X_train.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    n_dims = sine_template.shape[1]  # 2Nh

    # CCA(X_test, sine_template)
    u_xy = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    v_xy = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,2Nh)
    for ne in range(n_events):
        cca_model = _cca_kernel(
            X=X_test,
            Y=sine_template[ne],
            n_components=n_components
        )
        u_xy[ne] = cca_model['u']
        v_xy[ne] = cca_model['v']

    # CCA(X_test, avg_template)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    u_xa = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    v_xa = np.zeros_like(u_xa)  # (Ne,Nk,Nc)
    for ne, et in enumerate(event_type):
        avg_template[ne] = np.mean(X_train[y_train == et], axis=0)
        cca_model = _cca_kernel(
            X=X_test,
            Y=avg_template[ne],
            n_components=n_components
        )
        u_xa[ne] = cca_model['u']
        v_xa[ne] = cca_model['v']

    # CCA(avg_template, sine_template)
    u_ay = np.zeros_like(u_xy)  # (Ne,Nk,Nc)
    v_ay = np.zeros_like(v_xy)  # (Ne,Nk,2Nh)
    for ne in range(n_events):
        cca_model = _cca_kernel(
            X=avg_template[ne],
            Y=sine_template[ne],
            n_components=n_components
        )
        u_ay[ne] = cca_model['u']
        v_ay[ne] = cca_model['v']

    # eCCA model
    training_model = {
        'u_xy': u_xy, 'v_xy': v_xy,
        'u_xa': u_xa, 'v_xa': v_xa,
        'u_ay': u_ay, 'v_ay': v_ay,
        'avg_template': avg_template
    }
    return training_model


def _ecca_feature(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        X_test: ndarray,
        train_info: dict,
        n_components: int = 1,
        method_list: List[str] = ['1', '2', '3', '4', '5']) -> ndarray:
    """The pattern matching process of eCCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters. Nk.
        method_list (List[str]): Different coefficient. Labeled as '1' to '5'.

    Returns:
        rho (ndarray): (Ne,). Discriminant coefficients of eCCA.
    """
    n_events = train_info['n_events']  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))  # (Ne*Nte, Ne)
    for nte in range(n_test):
        temp_rho = []
        ecca_model = _ecca_kernel(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            X_test=X_test[nte],
            train_info=train_info,
            n_components=n_components
        )
        if '1' in method_list:
            u_xy, v_xy = ecca_model['u_xy'], ecca_model['v_xy']
            rho_1 = np.zeros((n_events))
            for nem in range(n_events):
                rho_1[nem] = utils.pearson_corr(
                    X=u_xy[nem] @ X_test,
                    Y=v_xy[nem] @ sine_template[nem]
                )
            temp_rho.append(abs(rho_1))

        if '2' in method_list:
            avg_template = ecca_model['avg_template']
            u_xa, v_xa = ecca_model['u_xa'], ecca_model['v_xa']
            rho_2 = np.zeros((n_events))
            for nem in range(n_events):
                rho_2[nem] = utils.pearson_corr(
                    X=u_xa[nem] @ X_test,
                    Y=v_xa[nem] @ avg_template[nem]
                )
            temp_rho.append(rho_2)

        if '3' in method_list:
            u_xy, avg_template = ecca_model['u_xy'], ecca_model['avg_template']
            rho_3 = np.zeros((n_events))
            for ne in range(n_events):
                rho_3[ne] = utils.pearson_corr(
                    X=u_xy[ne] @ X_test,
                    Y=u_xy[ne] @ avg_template[ne]
                )
            temp_rho.append(rho_3)

        if '4' in method_list:
            u_ay, avg_template = ecca_model['u_ay'], ecca_model['avg_template']
            rho_4 = np.zeros((n_events))
            for ne in range(n_events):
                rho_4[ne] = utils.pearson_corr(
                    X=u_ay[ne] @ X_test,
                    Y=u_ay[ne] @ avg_template[ne]
                )
            temp_rho.append(rho_4)

        if '5' in method_list:
            u_xa, v_xa = ecca_model['u_xa'], ecca_model['v_xa']
            avg_template = ecca_model['avg_template']
            rho_5 = np.zeros((n_events))
            for ne in range(n_events):
                rho_5[ne] = utils.pearson_corr(
                    X=u_xa[ne] @ avg_template[ne],
                    Y=v_xa[ne] @ avg_template[ne]
                )
            temp_rho.append(rho_5)
        rho[nte, :] = utils.combine_feature(temp_rho)
    return rho


class ECCA(BasicCCA):
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        method_list: List[str] = ['1', '2', '3', '4', '5']
    ):
        """Train eCCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=1.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
            method_list (List[str]): Different coefficient. Labeled as '1' to '5'.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.method_list = method_list
        event_type = np.unique(self.y_train)
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1]
        }
        return self

    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Nte*Ne,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho (ndarray): (Ne,). Decision coefficients of eCCA.
        """
        return _ecca_feature(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            X_test=X_test,
            train_info=self.train_info,
            n_components=self.n_components,
            method_list=self.method_list
        )


class FB_ECCA(BasicFBCCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=ECCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 9-10. Multi-stimulus eCCA | ms-eCCA
def _msecca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The modeling process of multi-stimulus eCCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int,
                            'events_group':{'event_id':[start index,end index]}}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, ndarray]
        Cxx (ndarray): (Ne,Nc,Nc). Covariance of averaged EEG template.
        Cxy (ndarray): (Ne,Nc,2*Nh). Covariance between EEG and sinusoidal template.
        Cyy (ndarray): (Ne,2*Nh,2*Nh). Covariance of sinusoidal template.
        u (ndarray): (Ne,Nk,Nc). Spatial filters (EEG signal).
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters (sinusoidal signal).
        uX (ndarray): (Ne,Nk,Np). ms-eCCA templates for EEG signal.
        vY (ndarray): (Ne,Nk,Np). ms-eCCA templates for sinusoidal signal.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    events_group = train_info['events_group']  # dict
    n_dims = sine_template.shape[1]  # 2Nh

    # covariance matrices of merged data
    Cxx_total = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    Cxy_total = np.zeros((n_events, n_chans, n_dims))  # (Ne,Nc,2Nh)
    Cyy_total = np.zeros((n_events, n_dims, n_dims))  # (Ne,2Nh,2Nh)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        avg_template[ne] = np.mean(X_train[y_train == et], axis=0)
        Cxx_total[ne] = avg_template[ne] @ avg_template[ne].T
        Cxy_total[ne] = avg_template[ne] @ sine_template[ne].T
        Cyy_total[ne] = sine_template[ne] @ sine_template[ne].T

    # GEPs | train spatial filters & templates
    u = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    v = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,2Nh)
    uX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    vY = np.zeros_like(uX)  # (Ne,Nk,Np)
    for ne, et in enumerate(event_type):
        merged_indices = events_group[str(event_type[ne])]
        Cxx_temp = np.sum(Cxx_total[merged_indices], axis=0)  # (Nc,Nc)
        Cxy_temp = np.sum(Cxy_total[merged_indices], axis=0)  # (Nc,2Nh)
        Cyy_temp = np.sum(Cyy_total[merged_indices], axis=0)  # (2Nh,2Nh)
        u[ne] = utils.solve_gep(
            A=Cxy_temp @ sLA.solve(Cyy_temp, Cxy_temp.T),
            B=Cxx_temp,
            n_components=n_components
        )
        v[ne] = utils.solve_gep(
            A=Cxy_temp.T @ sLA.solve(Cxx_temp, Cxy_temp),
            B=Cyy_temp,
            n_components=n_components
        )

        # positive-negative correction
        uX[ne] = u[ne] @ avg_template[ne]  # (Nk,Np)
        vY[ne] = v[ne] @ sine_template[ne]  # (Nk,Np)
        if utils.pearson_corr(X=uX[ne], Y=vY[ne]) < 0:
            v[ne] *= -1
            vY[ne] *= -1

    # ms-eCCA model
    training_model = {
        'Cxx': Cxx_total, 'Cxy': Cxy_total, 'Cyy': Cyy_total,
        'u': u, 'v': v, 'uX': uX, 'vY': vY
    }
    return training_model


def _msecca_feature(
        X_test: ndarray,
        msecca_model: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """The pattern matching process of ms-eCCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        msecca_model (dict): See details in _msecca_kernel().

    Returns: Dict[str, ndarray]
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of ms-eCCA.
        rho_eeg (ndarray): (Ne*Nte,Ne). EEG part rho.
        rho_sin (ndarray): (Ne*Nte,Ne). Sinusoidal signal part rho.
    """
    u, uX, vY = msecca_model['u'], msecca_model['uX'], msecca_model['vY']
    n_events = u.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte

    rho_eeg = np.zeros((n_test, n_events))  # (Ne*Nte,Ne)
    rho_sin = np.zeros_like(rho_eeg)
    for nte in range(n_test):
        for nem in range(n_events):
            temp_X = u[nem] @ X_test[nte]
            rho_eeg[nte, nem] = utils.pearson_corr(X=temp_X, Y=uX[nem])
            rho_sin[nte, nem] = utils.pearson_corr(X=temp_X, Y=vY[nem])
    rho = utils.combine_feature([rho_eeg, rho_sin])
    features = {
        'rho': rho, 'rho_eeg': rho_eeg, 'rho_sin': rho_sin
    }
    return features


class MS_ECCA(BasicCCA):
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        events_group: Optional[Dict[str, List[int]]] = None,
        d: int = 2
    ):
        """Train ms-eCCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=1.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
            events_group (Dict[str, List[int]], optional): {'event_id':[idx_1,idx_2,...]}.
                If None, events_group will be generated according to parameter 'd'.
            d (int): The range of events to be merged.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.d = d
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        if events_group is not None:
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(event_type, self.d)
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
            'events_group': self.events_group
        }

        # train_ms-eCCA filters and templates
        self.training_model = _msecca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            train_info=self.train_info,
            n_components=self.n_components
        )
        return self

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of ms-eCCA.
            rho_eeg (ndarray): (Ne*Nte,Ne). EEG part rho.
            rho_sin (ndarray): (Ne*Nte,Ne). Sinusoidal signal part rho.
        """
        return _msecca_feature(
            X_test=X_test,
            msecca_model=self.training_model
        )

    def predict(self, X_test: ndarray) -> ndarray:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            y_pred (ndarray): int or (Ne*Nte,). Predict labels.
        """
        self.features = self.transform(X_test)
        event_type = self.train_info['event_type']
        self.y_pred = event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


class FB_MS_ECCA(BasicFBCCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=MS_ECCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np) or (Ne*Nte,Nc,Np).
                Test dataset.

        Returns: Dict[str, ndarray]
            fb_rho (ndarray): (Nb,Ne*Nte,Ne). Decision coefficients of each band.
            rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of ms-eCCA.
            fb_rho_eeg (ndarray): (Nb,Ne*Nte,Ne). EEG part rho of each band.
            rho_eeg (ndarray): (Ne*Nte,Ne). EEG part rho.
            fb_rho_sin (ndarray): (Nb,Ne*Nte,Ne). Sinusoidal signal part rho of each band.
            rho_sin (ndarray): (Ne*Nte,Ne). Sinusoidal signal part rho.
        """
        if not self.with_filter_bank:  # tranform X_test
            X_test = self.fb_transform(X_test)  # (Nb,Ne*Nte,Nc,Np)
        sub_features = [se.transform(X_test[nse]) for nse, se in enumerate(self.sub_estimator)]

        fb_rho = np.stack([sf['rho'] for sf in sub_features], axis=0)  # (Nb,Ne*Nte,Ne)
        fb_rho_eeg = np.stack([sf['rho_eeg'] for sf in sub_features], axis=0)
        fb_rho_sin = np.stack([sf['rho_sin'] for sf in sub_features], axis=0)

        rho = np.einsum('b,bte->te', self.bank_weights, fb_rho)  # (Ne*Nte,Ne)
        rho_eeg = np.einsum('b,bte->te', self.bank_weights, fb_rho_eeg)
        rho_sin = np.einsum('b,bte->te', self.bank_weights, fb_rho_sin)

        features = {
            'fb_rho': fb_rho, 'rho': rho,
            'fb_rho_eeg': fb_rho_eeg, 'rho_eeg': rho_eeg,
            'fb_rho_sin': fb_rho_sin, 'rho_sin': rho_sin
        }
        return features


def _mscca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The modeling process of multi-stimulus CCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, ndarray]
        Cxx (ndarray): (Ne,Nc,Nc). Covariance of averaged EEG template.
        Cxy (ndarray): (Ne,Nc,2*Nh). Covariance between EEG and sinusoidal template.
        Cyy (ndarray): (Ne,2*Nh,2*Nh). Covariance of sinusoidal template.
        w (ndarray): (Nk,Nc). Common spatial filters.
        wX (ndarray): (Ne,Nk,Np). msCCA templates.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    n_dims = sine_template.shape[1]  # 2Nh

    # covariance matrices of merged data
    Cxx_total = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    Cxy_total = np.zeros((n_events, n_chans, n_dims))  # (Ne,Nc,2Nh)
    Cyy_total = np.zeros((n_events, n_dims, n_dims))  # (Ne,2Nh,2Nh)
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        avg_template[ne] = np.mean(X_train[y_train == et], axis=0)
        Cxx_total[ne] = avg_template[ne] @ avg_template[ne].T
        Cxy_total[ne] = avg_template[ne] @ sine_template[ne].T
        Cyy_total[ne] = sine_template[ne] @ sine_template[ne].T

    # GEPs | train spatial filters & templates
    Cxx_temp = Cxx_total.sum(axis=0)
    Cxy_temp = Cxy_total.sum(axis=0)
    Cyy_temp = Cyy_total.sum(axis=0)
    w = utils.solve_gep(
        A=Cxy_temp @ sLA.solve(Cyy_temp, Cxy_temp.T),
        B=Cxx_temp,
        n_components=n_components
    )  # (Nk,Nc)
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    for ne in range(n_events):
        wX[ne] = w @ avg_template[ne]
    # wX = np.einsum('kc,ecp->ekp', wX, avg_template)  # clearer but slower

    # msCCA model
    training_model = {
        'Cxx': Cxx_total, 'Cxy': Cxy_total, 'Cyy': Cyy_total,
        'w': w, 'wX': wX
    }
    return training_model


def _mscca_feature(
        X_test: ndarray,
        mscca_model: Dict[str, ndarray]) -> ndarray:
    """The pattern matching process of msCCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        mscca_model (dict): See details in _mscca_kernel().

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of msCCA.
    """
    w, wX = mscca_model['w'], mscca_model['wX']
    n_events = wX.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte

    rho = np.zeros((n_test, n_events))  # (Ne*Nte,Ne)
    for nte in range(n_test):
        temp_X = w @ X_test[nte]  # common filter, (Nk,Np)
        rho[nte] = utils.pearson_corr(X=temp_X, Y=wX, parallel=True)
    return rho


class MSCCA(BasicCCA):
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray
    ):
        """Train msCCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=1.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        event_type = np.unique(self.y_train)
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1]
        }

        # train_msCCA filters and templates
        self.training_model = _mscca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            train_info=self.train_info,
            n_components=self.n_components
        )
        return self

    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of msCCA.
        """
        return _mscca_feature(
            X_test=X_test,
            mscca_model=self.training_model
        )


class FB_MSCCA(BasicFBCCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=MSCCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 11-12. Multiset CCA | MsetCCA1
def _msetcca1_kernel(
        X_train: ndarray,
        y_train: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, List[ndarray]]:
    """The modeling process of multiset CCA (1).

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        train_info (dict): {'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, List[ndarray]]
        R (List[ndarray]): Ne*(Nt*Nc,Nt*Nc). Inter-trial covariance of EEG.
        S (List[ndarray]): Ne*(Nt*Nc,Nt*Nc). Intra-trial covariance of EEG.
        w (List[ndarray]): Ne*(Nk,Nt*Nc). Spatial filters for training dataset.
        wX (List[ndarray]): Ne*(Nt*Nk,Np). MsetCCA(1) templates.
    """
    # basic information
    event_type = train_info['event_type']
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np

    # covariance matrices of block-concatenated data | signal templates
    R, S, w, wX = [], [], [], []
    for ne, et in enumerate(event_type):
        train_trials = n_train[ne]  # Nt
        X_temp = X_train[y_train == et]  # (Nt,Nc,Np)
        X_concat = np.reshape(X_temp, (train_trials * n_chans, n_points), order='C')
        R_temp = X_concat @ X_concat.T  # (Nt*Nc,Nt*Nc)
        S_temp = block_diag([X_temp[tt] @ X_temp[tt].T
                             for tt in range(train_trials)]).toarray()
        w_temp = utils.solve_gep(
            A=R_temp - S_temp,
            B=S_temp,
            n_components=n_components
        )  # (Nk,Nt*Nc), Nt maybe different for each stimulus
        wX_temp = np.zeros((train_trials * n_components, n_points))  # (Nt*Nk,Np)
        for tt in range(train_trials):
            stn, edn = tt * n_components, (tt + 1) * n_components
            stc, edc = tt * n_chans, (tt + 1) * n_chans
            wX_temp[stn:edn] = w_temp[:, stc:edc] @ X_temp[tt]  # (Nk,Nc) @ (Nc,Np)
        R.append(R_temp)
        S.append(S_temp)
        w.append(w_temp)
        wX.append(wX_temp)

    # MsetCCA1 model
    training_model = {
        'R': R, 'S': S, 'w': w, 'wX': wX
    }
    return training_model


def _msetcca1_feature(
        X_test: ndarray,
        msetcca1_model: dict,
        n_components: int = 1) -> ndarray:
    """The pattern matching process of MsetCCA1.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        msetcca1_model (Dict[str, ndarray]): See details in _msetcca1_kernel().
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of MsetCCA1.
    """
    wX = msetcca1_model['wX']  # List: Ne*(Nt*Nk,Np)
    n_events = len(wX)  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))
    for nte in range(n_test):
        for nem in range(n_events):
            cca_model = _cca_kernel(
                X=X_test[nte],
                Y=wX[nem],
                n_components=n_components
            )
            rho[nte, nem] = cca_model['coef']
    return rho


class MSETCCA1(BasicCCA):
    def fit(self, X_train: ndarray, y_train: ndarray):
        """Train MsetCCA1 model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...Ne-1]
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_train': np.array([np.sum(self.y_train == et) for et in event_type]),
            'n_chans': X_train.shape[-2],
            'n_points': X_train.shape[-1]
        }

        # train MsetCCA(1) filters & templates
        self.training_model = _msetcca1_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            train_info=self.train_info,
            n_components=self.n_components
        )
        return self

    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: ndarray
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of MsetCCA1.
        """
        return _msetcca1_feature(
            X_test=X_test,
            msetcca1_model=self.training_model,
            n_components=self.n_components
        )


class FB_MSETCCA1(BasicFBCCA):
    def __init__(
        self,
        filter_bank: Optional[List] = None,
        with_filter_bank: bool = True,
        n_components: int = 1
    ):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=MSETCCA1(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %%
