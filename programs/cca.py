# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Canonical correlation analysis (CCA) series.
    1. CCA: http://ieeexplore.ieee.org/document/4203016/
<<<<<<< HEAD
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
=======
            DOI: 10.1109/TBME.2006.889197
    2. MEC: http://ieeexplore.ieee.org/document/4132932/
            DOI: 10.1109/TBME.2006.889160
    3. MCC: http://ieeexplore.ieee.org/document/4132932/
            DOI: 10.1109/TBME.2006.889160
    4. MSI:
    
    5. tMSI:
    
    6. eMSI:
    
    7. eCCA: http://www.pnas.org/lookup/doi/10.1073/pnas.1508080112
            DOI: 10.1073/pnas.1508080112
    8. msCCA: https://ieeexplore.ieee.org/document/9006809/
            DOI: 10.1109/TBME.2020.2975552
    9. ms-eCCA: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
            DOI: 10.1088/1741-2552/ab2373
    10. MsetCCA1: https://www.worldscientific.com/doi/abs/10.1142/S0129065714500130
            DOI: 10.1142/S0129065714500130
    11. MsetCCA2: https://ieeexplore.ieee.org/document/8231203/
            DOI: 10.1109/TBME.2017.2785412
    12. MwayCCA: 
            DOI: 
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138


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
<<<<<<< HEAD
=======

update: 2023/07/04
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

"""

# %% basic modules
import utils

<<<<<<< HEAD
from abc import abstractmethod
from typing import Optional, List, Dict, Union
=======
from abc import abstractmethod, ABCMeta
from typing import Optional, List, Tuple, Any
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

import numpy as np
from numpy import ndarray
import scipy.linalg as sLA
<<<<<<< HEAD
from scipy.sparse import block_diag

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% Basic CCA object
class BasicCCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1):
=======


# %% Basic CCA object
class BasicCCA(metaclass=ABCMeta):
    def __init__(self,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None):
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """Basic configuration.

        Args:
            n_components (int): Number of eigenvectors picked as filters.
        """
        # config model
        self.n_components = n_components

    @abstractmethod
<<<<<<< HEAD
    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: Optional[ndarray] = None
    ):
        """Load in training dataset and train model.
=======
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        pass
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

        Args:
            X_train (ndarray): (Ne*Nt,...,Np). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, Optional): (Ne,2*Nh,Np). Sinusoidal templates.
        """
        pass

    @abstractmethod
<<<<<<< HEAD
    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients.
=======
    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients
            y_predict (ndarray): (Ne*Nte,). Predict labels.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """
        pass

    def predict(self, X_test: ndarray) -> Union[int, ndarray]:
        """Predict test data.

<<<<<<< HEAD
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
=======
class BasicFBCCA(metaclass=ABCMeta):
    def __init__(self,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None):
        """Basic configuration.

        Args:
            n_components (int): Number of eigenvectors picked as filters.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None' when n_components is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio


    @abstractmethod
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,...,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        pass


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using filter-bank algorithms to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[1]
        event_type = self.train_info['event_type']

        # apply model.predict() in each sub-band
        self.fb_rou = [[] for nb in range(self.n_bands)]
        self.fb_y_predict = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
            self.fb_rou[nb], self.fb_y_predict[nb] = fb_results[0], fb_results[1]

        # integration of multi-bands' results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            self.y_predict[nte] = event_type[np.argmax(self.rou[nte,:])]
        return self.rou, self.y_predict


# %% 1. standard CCA | CCA
def cca_compute(
    data: ndarray,
    template: ndarray,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict[str, ndarray]:
    """Canonical correlation analysis (CCA).

    Args:
        data (ndarray): (Nc,Np). Real EEG data (single trial).
        template (ndarray): (2Nh or m,Np). Sinusoidal template or averaged template.
        n_components (int): Number of eigenvectors picked as filters.
            Eigenvectors are referring to eigenvalues sorted in descend order.
            Set to None if ratio is not None.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not None.

    Return: CCA model (dict).
        Cxx (ndarray): (Nc,Nc). Covariance of EEG.
        Cxy (ndarray): (Nc,2*Nh). Covariance of EEG & sinusoidal template.
        Cyy (ndarray): (2*Nh,2*Nh). Covariance of sinusoidal template.
        u (ndarray): (Nk,Nc). Spatial filter for EEG.
        v (ndarray): (Nk,2*Nh). Spatial filter for template.
        uX (ndarray): (Nk,Np). Filtered EEG signal
        vY (ndarray): (Nk,Np). Filtered sinusoidal template.
    """
    # GEPs' conditions
    Cxx = data @ data.T  # (Nc,Nc)
    Cyy = template @ template.T  # (2Nh,2Nh)
    Cxy = data @ template.T  # (Nc,2Nh)

    # Spatial filter for EEG: (Nk,2Nh)
    u = utils.solve_gep(
        A=Cxy @ sLA.solve(Cyy,Cxy.T),
        B=Cxx,
        n_components=n_components,
        ratio=ratio
    )

    # Spatial filter for template: (Nk,2Nh)
    v = utils.solve_gep(
        A=Cxy.T @ sLA.solve(Cxx,Cxy),
        B=Cyy,
        n_components=n_components,
        ratio=ratio
    )

    # filter data
    uX = u @ data  # (Nk,Np)
    vY = v @ template  # (Nk,Np)

    # CCA model
    model = {
        'Cxx':Cxx, 'Cxy':Cxy, 'Cyy':Cyy,
        'u':u, 'v':v,
        'uX':uX, 'vY':vY
    }
    return model


def cca_coef():
    pass


class CCA(BasicCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        """Load in CCA template. CCA is an unsupervised algorithm, 
            so there's no EEG training dataset.

        Args:
            X_train (ndarray): (Ne,2Nh,Np)
                Sinusoidal template.
            y_train (ndarray): (Ne,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(self.y_train)
        self.train_info = {
            'event_type':event_type,
            'n_events':len(event_type),
            'n_points':self.X_train.shape[-1]
        }
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using CCA to predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np).
                Test dataset. Ne*Nte could be 1 if necessary.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']
        event_type = self.train_info['event_type']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                model = cca_compute(
                    data=X_test[nte],
                    template=self.X_train[ne],
                    n_components=self.n_components,
                    ratio=self.ratio
                )
                self.rou[nte,ne] = utils.pearson_corr(model['uX'], model['vY'])
            self.y_predict[nte] = event_type[np.argmax(self.rou[nte,:])]
        return self.rou, self.y_predict


class FB_CCA(BasicCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        """Train filter-bank CCA model.

        Args:
            X_train (ndarray): (Nb,Ne,2*Nh,Np). Sinusoidal template.
            y_train (ndarray): (Ne,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.n_bands = self.X_train.shape[0]

        # train CCA models in each band
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = CCA(
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train
            )
        return self


# %% 2. Minimum Energy Combination | MEC
def mec_compute(
    data: ndarray,
    template: ndarray,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict[str, ndarray]:
    """Minimum energy combination.

    Args:
        data (ndarray): (Nc,Np). Real EEG data (single trial).
        template (ndarray): (2Nh or m,Np). Sinusoidal template or averaged template.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Eigenvectors are referring to eigenvalues sorted in descend order.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: mec model (dict)
        w (ndarray): (Nk,Nc). Spatial filter.
        wX (ndarray): (Nk,Np). Filtered EEG data.
    """
    # projection = template.T @ sLA.inv(template @ template.T) @ template  # slow way
    projection = template.T @ template / np.sum(template[0]**2)  # fast way
    X_hat = data - data @ projection  # (Nc,Np)

    # GEP's conditions
    A = X_hat @ X_hat.T  # (Nc,Nc)

    # spatial filter & template: (Nk,Nc)
    w = utils.solve_ep(
        A=A,
        n_components=n_components,
        ratio=ratio,
        mode='Min'
    )
    wX = w @ data  # (Nk,Np)

    # MEC model
    model = {
        'w':w, 'wX':wX
    }
    return model


# %% 3. Maximum Contrast Combination | MCC


# %% 4. MSI | MSI


# %% 5. tMSI


# %% 6. extend-MSI | eMSI


# %% 7. Extended CCA | eCCA
def ecca_compute(
    avg_template: ndarray,
    sine_template: ndarray,
    X_test: ndarray,
    coef_idx: Optional[List] = [1,2,3,4,5],
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict[str, Any]:
    """CCA with individual calibration data.

    Args:
        avg_template (ndarray): (Nc,Np). Template averaged across trials.
        sine_template (ndarray): (2*Nh,Np). Sinusoidal template.
        X_test (ndarray): (Nc,Np). Single trial test data.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: eCCA model (dict).
        u_xy, v_xy (ndarray): Spatial filters created from CCA(X_test, sine_template).
        u_xa, v_xa (ndarray): Spatial filters created from CCA(X_test, avg_template).
        u_ay, v_ay (ndarray): Spatial filters created from CCA(avg_template, sine_template).
        coef (List[float]): 5 feature coefficients.
        rou (float): Integrated feature coefficient.
    """
    # standard CCA process: CCA(X_test, sine_template)
    coef = []
    cca_model_xy = cca_compute(
        data=X_test,
        template=sine_template,
        n_components=n_components,
        ratio=ratio
    )
    Cxx, Cyy = cca_model_xy['Cxx'], cca_model_xy['Cyy']
    u_xy = cca_model_xy['u']
    if 1 in coef_idx:
        coef.append(utils.pearson_corr(cca_model_xy['uX'], cca_model_xy['vY']))

    # correlation between X_test and average templates: CCA(X_test, avg_template)
    Caa = avg_template @ avg_template.T
    Cxa = X_test @ avg_template.T
    u_xa = utils.solve_gep(
        A=Cxa @ sLA.solve(Caa,Cxa.T),
        B=Cxx,
        n_components=n_components,
        ratio=ratio
    )
    v_xa = utils.solve_gep(
        A=Cxa.T @ sLA.solve(Cxx,Cxa),
        B=Caa,
        n_components=n_components,
        ratio=ratio
    )
    if 2 in coef_idx:
        coef.append(utils.pearson_corr(u_xa@X_test, v_xa@avg_template))
    if 3 in coef_idx:
        coef.append(utils.pearson_corr(u_xy@X_test, u_xy@avg_template))
    # slower but clearer way (maybe):
    # cca_model_xa = cca_compute(
    #     data=X_test,
    #     template=avg_template,
    #     n_components=n_components,
    #     ratio=ratio
    # )
    # coef[1] = utils.pearson_corr(cca_model_xa['uX'], cca_model_xa['vY'])
    # # the covariance matrix of X_test (Cxx) has been computed before.

    # CCA(avg_template, sine_template)
    Cay = avg_template @ sine_template.T
    u_ay = utils.solve_gep(
        A=Cay @ sLA.solve(Cyy,Cay.T),
        B=Caa,
        n_components=n_components,
        ratio=ratio
    )
    v_ay = utils.solve_gep(
        A=Cay.T @ sLA.solve(Caa,Cay),
        B=Cyy,
        n_components=n_components,
        ratio=ratio
    )
    if 4 in coef_idx:
        coef.append(utils.pearson_corr(u_ay@X_test, u_ay@avg_template))
    # slower but clearer way (maybe):
    # cca_model_ay = cca_compute(
    #     data=avg_template,
    #     template=sine_template,
    #     n_components=n_components,
    #     ratio=ratio
    # )
    # u_ay = cca_model_ay['u']
    # # the covariance matrix (Caa, Cyy) have been computed before.

    # similarity between filters corresponding to X_test and avg_template
    if 5 in coef_idx:
        coef.append(utils.pearson_corr(u_xa@avg_template, v_xa@avg_template))

    # combined features
    rou = utils.combine_feature(coef)

    # eCCA model
    model = {
        'u_xy':u_xy, 'v_xy':cca_model_xy['v'],
        'u_xa':u_xa, 'v_xa':v_xa,
        'u_ay':u_ay, 'v_ay':v_ay,
        'coef':coef, 'rou':np.real(rou)
    }
    return model


class ECCA(BasicCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        coef_idx: Optional[List] = [1,2,3,4,5]):
        """Load in eCCA templates.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=1.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
<<<<<<< HEAD
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
=======
        self.coef_idx = coef_idx
        event_type = np.unique(self.y_train)
        n_events = len(event_type)
        n_chans = self.X_train.shape[-2]
        n_points = self.X_train.shape[-1]
        self.train_info = {
            'event_type':event_type,
            'n_events':n_events,
            'n_chans':n_chans,
            'n_points':n_points
        }

        # config average template
        self.avg_template = np.zeros((n_events, n_chans, n_points))
        for ne,et in enumerate(event_type):
            temp = self.X_train[self.y_train==et]
            if temp.ndim == 2:  # (Nc,Np), Nt=1
                self.avg_template[ne] = temp
            elif temp.ndim > 2:  # (Nt,Nc,Np)
                self.avg_template[ne] = np.mean(temp, axis=0)
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using eCCA to predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']
        event_type = self.train_info['event_type']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                model = ecca_compute(
                    avg_template=self.avg_template[ne],
                    sine_template=self.sine_template[ne],
                    X_test=X_test[nte],
                    coef_idx=self.coef_idx,
                    n_components=self.n_components,
                    ratio=self.ratio
                )
                self.rou[nte,ne] = model['rou']
            self.y_predict[nte] = event_type[np.argmax(self.rou[nte,:])]
        return self.rou, self.y_predict


class FB_ECCA(BasicFBCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray):
        """Load in filter-bank eCCA templates.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Nb,Ne,2*Nh,Np). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.n_bands = self.X_train.shape[0]

        # train eCCA models in each band
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = ECCA(
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                sine_template=self.sine_template[nb]
            )
        return self


# %% 8-9. Multi-stimulus eCCA | ms-eCCA
def msecca_compute(
    avg_template: ndarray,
    sine_template: ndarray,
    train_info: dict,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict[str, Any]:
    """Multi-stimulus eCCA.

    Args:
        avg_template (ndarray): (Ne,Nc,Np). Template averaged across trials.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int,
                            'events_group':{'event_id':[start index,end index]}}
        n_components (int): Number of eigenvectors picked as filters. Nk.
<<<<<<< HEAD

    Returns: Dict[str, ndarray]
        Cxx (ndarray): (Ne,Nc,Nc). Covariance of averaged EEG template.
        Cxy (ndarray): (Ne,Nc,2*Nh). Covariance between EEG and sinusoidal template.
        Cyy (ndarray): (Ne,2*Nh,2*Nh). Covariance of sinusoidal template.
        u (ndarray): (Ne,Nk,Nc). Spatial filters (EEG signal).
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters (sinusoidal signal).
        uX (ndarray): (Ne,Nk,Np). ms-eCCA templates for EEG signal.
        vY (ndarray): (Ne,Nk,Np). ms-eCCA templates for sinusoidal signal.
=======
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: ms-eCCA model (dict)
        Cxx (ndarray): (Ne,Nc,Nc). Covariance of averaged EEG template.
        Cxy (ndarray): (Ne,Nc,2*Nh). Covariance between EEG and sinusoidal template.
        Cyy (ndarray): (Ne,2*Nh,2*Nh). Covariance of sinusoidal template.
        u (List[ndarray]): Ne*(Nk,Nc). Spatial filters for EEG.
        v (List[ndarray]): Ne*(Nk,2*Nh). Spatial filters for sinusoidal templates.
        uX (List[ndarray]): Ne*(Nk,Np). ms-CCA templates for EEG part.
        vY (List[ndarray]): Ne*(Nk,Np). ms-CCA templates for sinusoidal template part.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
<<<<<<< HEAD
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
=======
    events_group = train_info['events_group']  # dict
    n_2harmonics = sine_template.shape[1]

    # GEPs' conditions
    # Cxx = np.einsum('ecp,ehp->ech', avg_template,avg_template)
    # Cyy = np.einsum('ecp,ehp->ech', sine_template,sine_template)
    # Cxy = np.einsum('ecp,ehp->ech', avg_template,sine_template)
    Cxx = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    Cxy = np.zeros((n_events, n_chans, n_2harmonics))  # (Ne,Nc,2Nh)
    Cyy = np.zeros((n_events, n_2harmonics, n_2harmonics))  # (Ne,2Nh,2Nh)
    for ne in range(n_events):
        Cxx[ne] = avg_template[ne] @ avg_template[ne].T
        Cxy[ne] = avg_template[ne] @ sine_template[ne].T
        Cyy[ne] = sine_template[ne] @ sine_template[ne].T

    # GEPs with merged data
    u, uX, v, vY = [], [], [], []
    correct = [False for ne in range(n_events)]
    for ne,et in enumerate(event_type):
        # GEPs' conditions
        st, ed = events_group[str(et)][0], events_group[str(et)][1]
        temp_Cxx = np.sum(Cxx[st:ed], axis=0)  # (Nc,Nc)
        temp_Cxy = np.sum(Cxy[st:ed], axis=0)  # (Nc,2Nh)
        temp_Cyy = np.sum(Cyy[st:ed], axis=0)  # (2Nh,2Nh)

        # EEG part: (Nk,Nc)
        temp_u = utils.solve_gep(
            A=temp_Cxy @ sLA.solve(temp_Cyy,temp_Cxy.T),
            B=temp_Cxx,
            n_components=n_components,
            ratio=ratio
        )

        # sinusoidal template part: (Nk,2Nh)
        temp_v = utils.solve_gep(
            A=temp_Cxy.T @ sLA.solve(temp_Cxx,temp_Cxy),
            B=temp_Cyy,
            n_components=n_components,
            ratio=ratio
        )

        # correct direction
        temp_uX = temp_u @ avg_template[ne]  # (Nk,Np)
        temp_vY = temp_v @ sine_template[ne]  # (Nk,Np)
        if utils.pearson_corr(temp_uX, temp_vY) < 0:
            temp_u *= -1
            temp_uX *= -1
            correct[ne] = True
        u.append(temp_u)
        v.append(temp_v)

        # signal templates
        uX.append(temp_uX)
        vY.append(temp_vY)

    # ms-eCCA model
    model = {
        'Cxx':Cxx, 'Cxy':Cxy, 'Cyy':Cyy,
        'u':u, 'v':v,
        'uX':uX, 'vY':vY , 'correct':correct
    }
    return model


class MS_ECCA(BasicCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        events_group: Optional[dict] = None,
        d: Optional[int] = 2):
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """Train ms-eCCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=1.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
<<<<<<< HEAD
            events_group (Dict[str, List[int]], optional): {'event_id':[idx_1,idx_2,...]}.
                If None, events_group will be generated according to parameter 'd'.
=======
            events_group (dict): {'event_id':[start index,end index]}
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
            d (int): The range of events to be merged.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.d = d
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
<<<<<<< HEAD
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
=======
        if events_group:  # given range
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(event_type, self.d)
        n_events = len(event_type)
        n_chans = X_train.shape[-2]
        n_points = X_train.shape[-1]
        self.train_info = {
            'event_type':event_type,
            'n_events':n_events,
            'n_chans':n_chans,
            'n_points':n_points,
            'events_group':self.events_group
        }

        # config average template | (Ne,Nc,Np)
        self.avg_template = np.zeros((n_events, n_chans, n_points))
        for ne,et in enumerate(event_type):
            temp = self.X_train[self.y_train==et]
            if temp.ndim == 2:  # (Nc,Np), Nt=1
                self.avg_template[ne] = temp
            elif temp.ndim > 2:  # (Nt,Nc,Np)
                self.avg_template[ne] = np.mean(temp, axis=0)

        # train_ms-CCA filters and templates
        model = msecca_compute(
            avg_template=self.avg_template,
            sine_template=self.sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Cxx, self.Cxy, self.Cyy = model['Cxx'], model['Cxy'], model['Cyy']
        self.u, self.v = model['u'], model['v']
        self.uX, self.vY = model['uX'], model['vY']
        self.correct = model['correct']
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using ms-eCCA algorithm to predict test data.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

<<<<<<< HEAD
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
=======
        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']
        event_type = self.train_info['event_type']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.rou_eeg = np.zeros_like(self.rou)
        self.rou_sin = np.zeros_like(self.rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            f_test = self.u @ X_test[nte]
            for ne in range(n_events):
                self.rou_eeg[nte,ne] = utils.pearson_corr(f_test, self.uX[ne])
                self.rou_sin[nte,ne] = utils.pearson_corr(f_test, self.vY[ne])
                self.rou[nte,ne] = utils.combine_feature([
                    self.rou_eeg[nte,ne],
                    self.rou_sin[nte,ne]
                ])
            self.y_predict[nte] = event_type[np.argmax(self.rou[nte,:])]
        return self.rou, self.y_predict


class FB_MS_ECCA(BasicFBCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        events_group: Optional[dict] = None,
        d: Optional[int] = 2):
        """Train filter-bank ms-eCCA model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=1.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Nb,Ne,2*Nh,Np). Sinusoidal template.
            events_group (dict): {'event_id':[start index,end index]}
            d (int): The range of events to be merged.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.n_bands = X_train.shape[0]
        self.d = d
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        if events_group:  # given range
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(len(event_type), d)

        # train ms-eCCA models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = MS_ECCA(
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                sine_template=self.sine_template[nb],
                events_group=self.events_group,
                d=self.d
            )
        return self


# msCCA is only part of ms-eCCA. Personally, i dont like this design
def mscca_compute(
    avg_template: ndarray,
    sine_template: ndarray,
    train_info: dict,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict[str, Any]:
    """Multi-stimulus CCA.

    Args:
        avg_template (ndarray): (Ne,Nc,Np). Template averaged across trials.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters. Nk.
<<<<<<< HEAD

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
=======
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: msCCA model (dict).
        Cxx (ndarray): (Ne,Nc,Nc). Covariance of averaged EEG template.
        Cxy (ndarray): (Ne,Nc,2*Nh). Covariance between EEG and sinusoidal template.
        Cyy (ndarray): (Ne,2*Nh,2*Nh). Covariance of sinusoidal template.
        w (ndarray): (Nk,Nc). Common spatial filter.
        wX (ndarray): (Ne,Nk,Np). msCCA templates.
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    n_2harmonics = sine_template.shape[1]

    # GEPs' conditions
    Cxx = np.zeros((n_chans, n_chans))  # (Ne,Nc,Nc)
    Cxy = np.zeros((n_chans, n_2harmonics))  # (Ne,Nc,2Nh)
    Cyy = np.zeros((n_2harmonics, n_2harmonics))  # (Ne,2Nh,2Nh)
    for ne in range(n_events):
        Cxx[ne] += avg_template[ne] @ avg_template[ne].T
        Cxy[ne] += avg_template[ne] @ sine_template[ne].T
        Cyy[ne] += sine_template[ne] @ sine_template[ne].T
    # A = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    # for nea in range(n_events):
    #     for neb in range(n_events):
    #         A += avg_template[nea] @ Q[nea] @ Q[neb].T @ avg_template[neb].T

    # B = np.zeros_like(A)
    # for ne in range(n_events):
    #     B += avg_template[ne] @ avg_template[ne].T
    # B = np.einsum('ecp,ehp->ch', avg_template, avg_template)  | slower but clearer

    # GEPs with merged data
    u = utils.solve_gep(
        A=Cxy @ sLA.solve(Cyy,Cxy.T),
        B=Cxx,
        n_components=n_components,
        ratio=ratio
    )  # (Nk,Nc)

    # signal templates
    uX = np.zeros((n_events, u.shape[0], n_points))
    for ne in range(n_events):
        uX[ne] = u @ avg_template[ne]

    # msCCA model
    model = {
        'Cxx':Cxx, 'Cxy':Cxy, 'Cyy':Cyy,
        'u':u, 'uX':uX
    }
    return model


class MS_CCA(BasicCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray):
        """Train ms-CCA model.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

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
<<<<<<< HEAD
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
=======
        n_events = len(event_type)
        n_chans = self.X_train.shape[-2]
        n_points = self.X_train.shape[-1]
        self.train_info = {
            'event_type':event_type,
            'n_events':n_events,
            'n_chans':n_chans,
            'n_points':n_points
        }

        # config average template | (Ne,Nc,Np)
        self.avg_template = np.zeros((n_events, n_chans, n_points))
        for ne,et in enumerate(event_type):
            temp = self.X_train[self.y_train==et]
            if temp.ndim == 2:  # (Nc,Np), Nt=1
                self.avg_template[ne] = temp
            elif temp.ndim > 2:  # (Nt,Nc,Np)
                self.avg_template[ne] = np.mean(temp, axis=0)

        # train ms-CCA filters & templates
        model = mscca_compute(
            avg_template=self.avg_template,
            sine_template=self.sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Cxx, self.Cxy, self.Cyy = model['Cxx'], model['Cxy'], model['Cyy']
        self.u, self.uX = model['u'], model['uX']
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using ms-CCA algorithm to predict test data.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

<<<<<<< HEAD
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
=======
        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']
        event_type = self.train_info['event_type']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            f_test = self.u @ X_test[nte]
            for ne in range(n_events):
                self.rou[nte,ne] = utils.pearson_corr(f_test, self.uX[ne])
            self.y_predict[nte] = event_type[np.argmax(self.rou[nte,:])]
        return self.rou, self.y_predict


class FB_MS_CCA(BasicFBCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray):
        """Train filter-bank ms-CCA model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=1.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Nb,Ne,2*Nh,Np). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.n_bands = X_train.shape[0]

        # train ms-CCA models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = MS_CCA(
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                sine_template=self.sine_template[nb]
            )
        return self


# %% 10-11 Multiset CCA | MsetCCA1
def msetcca1_compute(
    X_train: ndarray,
    y_train: ndarray,
    train_info: dict,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict[str, Any]:
    """Multiset CCA (1).
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        train_info (dict): {'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int}
<<<<<<< HEAD
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, List[ndarray]]
        R (List[ndarray]): Ne*(Nt*Nc,Nt*Nc). Inter-trial covariance of EEG.
        S (List[ndarray]): Ne*(Nt*Nc,Nt*Nc). Intra-trial covariance of EEG.
        w (List[ndarray]): Ne*(Nk,Nt*Nc). Spatial filters for training dataset.
        wX (List[ndarray]): Ne*(Nt*Nk,Np). MsetCCA(1) templates.
=======
        n_components (int, optional): Number of eigenvectors picked as filters.
            Defaults to 1. Set to 'None' if ratio is not 'None'.
        ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not 'None'.

    Return: MsetCCA1 model (dict)
        R (List[ndarray]): Ne*(Nt*Nc,Nt*Nc). Covariance of original data (various trials).
        S (List[ndarray]): Ne*(Nt*Nc,Nt*Nc). Covariance of original data (same trials).
        w (List[ndarray]): Ne*(Nk, Nt*Nc). Spatial filters for training dataset.
        wX (List[ndarray]): Ne*(Nt*Nk, Np). MsetCCA(1) templates.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """
    # basic information
    event_type = train_info['event_type']
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np

<<<<<<< HEAD
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
=======
    # GEPs with block-concatenated data | Nt maybe different for each stimulus
    R, S, w = [], [], []
    for ne,et in enumerate(event_type):
        n_sample = n_train[ne]
        temp_R = np.zeros((n_sample*n_chans, n_sample*n_chans))  # (Nt*Nc,Nt*Nc)
        temp_S = np.zeros_like(temp_R)
        temp = X_train[y_train==et]
        for nsa in range(n_sample):  # loop in columns
            spc, epc = nsa*n_chans, (nsa+1)*n_chans  # start/end point of columns
            temp_S[spc:epc,spc:epc] = temp[nsa] @ temp[nsa].T
            for nsm in range(n_sample):  # loop in rows
                spr, epr = nsm*n_chans, (nsm+1)*n_chans  # start/end point of rows
                if nsm < nsa:  # upper triangular district
                    temp_R[spr:epr,spc:epc] = temp_R[spc:epc,spr:epr].T
                elif nsm == nsa:  # diagonal district
                    temp_R[spr:epr,spc:epc] = temp_S[spc:epc,spc:epc]
                else:
                    temp_R[spr:epr,spc:epc] = temp[nsm] @ temp[nsa].T
        temp_w = utils.solve_gep(
            A=temp_R,
            B=temp_S,
            n_components=n_components,
            ratio=ratio
        )
        R.append(temp_R)  # Ne*(Nt*Nc,Nt*Nc)
        S.append(temp_S)  # Ne*(Nt*Nc,Nt*Nc)
        w.append(temp_w)  # Ne*(Nk,Nt*Nc), Nk maybe different for each stimulus

    # signal templates
    wX = []  # Ne*(Nt,Np)
    for ne,et in enumerate(event_type):
        n_sample = n_train[ne]  # Nt>=2
        n_dim = w[ne].shape[0]  # Nk
        temp_wX = np.zeros((n_sample*n_dim, n_points))  # (Nt*Nk,Np)
        temp = X_train[y_train==et]  # (Nt,Nc,Np)
        for nsa in range(n_sample):
            temp_wX[nsa*n_dim:(nsa+1)*n_dim,:] = w[ne][:,nsa*n_chans:(nsa+1)*n_chans] @ temp[nsa]
        wX.append(temp_wX)

    # MsetCCA1 model
    model = {'R':R, 'S':S, 'w':w, 'wX':wX}
    return model


class MSETCCA1(BasicCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
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
<<<<<<< HEAD
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
=======
            'event_type':event_type,
            'n_events':len(event_type),
            'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
            'n_chans':X_train.shape[-2],
            'n_points':X_train.shape[-1]
        }

        # train MsetCCA(1) filters and templates
        model = msetcca1_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.R, self.S = model['R'], model['S']
        self.w, self.wX = model['w'], model['wX']
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using MsetCCA1 algorithm to predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
            y_test (ndarray): (Ne*Nte,). Labels for X_test.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']
        event_type = self.train_info['event_type']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                model = cca_compute(
                    data=X_test[nte],
                    template=self.template[ne],
                    n_components=self.n_components,
                    ratio=self.ratio
                )
                self.rou[nte,ne] = utils.pearson_corr(model['uX'], model['vY'])
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


def msetcca2_compute():
    pass


class MSETCCA2(BasicCCA):
    pass
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138


# %%
