"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Other design.
    (1) DSP: https://ieeexplore.ieee.org/document/8930304/
            DOI: 10.1109/TBME.2019.2958641
    (2) DCPM: https://ieeexplore.ieee.org/document/8930304/
            DOI: 10.1109/TBME.2019.2958641
    (3) TDCA: https://ieeexplore.ieee.org/document/9541393/
            DOI: 10.1109/TNSRE.2021.3114340

"""

# %% basic modules
import utils

from abc import abstractmethod
from typing import Optional, List, Dict, Union, Tuple

import numpy as np
from numpy import ndarray

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% Basic object
class BasicDSP(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, n_components: int = 1):
        """Basic configuration.

        Args:
            n_components (int): Number of eigenvectors picked as filters. Nk.
                Defaults to 1.
        """
        # config model
        self.n_components = n_components

    @abstractmethod
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: Optional[ndarray] = None):
        """Load in training dataset and train model.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, Optional): (Ne,2*Nh,Np).
                Sinusoidal templates.
        """
        pass

    @abstractmethod
    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients.
        """
        pass

    def predict(self, X_test: ndarray) -> Union[int, ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            y_pred (ndarray): (Ne*Nte,). Predict labels.
        """
        self.features = self.transform(X_test)
        self.y_pred = self.event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


class BasicFBDSP(utils.FilterBank, ClassifierMixin):
    def predict(self, X_test: ndarray) -> ndarray:
        """Using filter-bank DSP algorithms to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
            y_pred (ndarray): (Ne*Nte,). Predict labels.
        """
        self.features = self.transform(X_test)
        event_type = self.sub_estimator[0].event_type
        self.y_pred = event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


# %% 1. Discriminant Spatial Patterns | DSP
def generate_dsp_mat(
        X: ndarray,
        y: ndarray) -> Tuple[ndarray, ndarray]:
    """Generate covariance matrices Sb & Sw for DSP model.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Sklearn-style dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X.

    Returns:
        Sb (ndarray): (Nc,Nc). Scatter matrix of between-class difference.
        Sw (ndarray): (Nc,Nc). Scatter matrix of within-class difference.
        X_mean (ndarray): (Ne,Nc,Np). Trial-averaged X.
    """
    # basic information
    event_type = np.unique(y)
    n_events = len(event_type)
    n_chans = X.shape[-2]
    n_train = np.array([np.sum(y == et) for et in event_type])

    # inter-class divergence: Sb
    X_mean = utils.generate_mean(X=X, y=y)  # (Ne,Nc,Np)
    # DO NOT use np.mean(X_mean, axis=0) in case of imbalanced samples
    Hb = X_mean - X.mean(axis=0, keepdims=True)  # (Ne,Nc,Np)
    Sb = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for ne in range(n_events):
        Sb += Hb[ne] @ Hb[ne].T
    Sb /= n_events
    # slower: Sb = np.einsum('ecp,ehp->ech', Hb, Hb)

    # intra-class divergence: Sw
    Sw = np.zeros_like(Sb)  # (Nc,Nc)
    for ne, et in enumerate(event_type):
        Hw = X[y == et] - X_mean[[ne]]  # (Nt,Nc,Np)
        for ntr in range(n_train[ne]):
            Sw += Hw[ntr] @ Hw[ntr].T
    Sw /= X.shape[0]
    return Sb, Sw, X_mean


def dsp_kernel(
        X_train: ndarray,
        y_train: ndarray,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The modeling process of DSP.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.

    Returns:
        Sb (ndarray): (Nc,Nc). Scatter matrix of between-class difference.
        Sw (ndarray): (Nc,Nc). Scatter matrix of within-class difference.
        w (ndarray): (Nk,Nc). Common spatial filter.
        wX (ndarray): (Ne,Nk,Np). DSP templates.
    """
    # solve target functions
    Sb, Sw, X_mean = generate_dsp_mat(X=X_train, y=y_train)
    w = utils.solve_gep(A=Sb, B=Sw, n_components=n_components)  # (Nk,Nc)

    # generate spatial-filtered templates
    wX = utils.spatial_filtering(w=w, X_mean=X_mean)  # (Ne,Nk,Np)
    return {'Sb': Sb, 'Sw': Sw, 'w': w, 'wX': wX}


def dsp_feature(
        X_test: ndarray,
        dsp_model: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """The pattern matching process of DSP.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        dsp_model (Dict[str, ndarray]): See details in dsp_kernel().

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Features of DSP.
    """
    # load in model
    w, wX = dsp_model['w'], dsp_model['wX']  # (Nk,Nc), (Ne,Nk*Np)
    n_events = wX.shape[0]  # Ne
    wX = np.reshape(wX, (n_events, -1), 'C')  # (Ne,Nk*Np)

    # pattern matching
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))
    for nte in range(n_test):
        X_temp = utils.fast_stan_2d(w @ X_test[nte])  # (Nk,Np)
        X_temp = np.tile(np.reshape(X_temp, -1, 'C'), (n_events, 1))  # (Ne,Nk*Np)
        rho[nte] = utils.fast_corr_2d(X=X_temp, Y=wX)
    return {'rho': rho}


class DSP(BasicDSP):
    def fit(self, X_train: ndarray, y_train: ndarray):
        """Train DSP model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'

        # train DSP models & templates
        self.training_model = dsp_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            n_components=self.n_components,
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of DSP.
        """
        return dsp_feature(
            X_test=X_test,
            dsp_model=self.training_model
        )


class FB_DSP(BasicFBDSP):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=DSP(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 2. Discriminant Canonical Pattern Matching | DCPM


# %% 3. Task-discriminant component analysis | TDCA
def tdca_augmentation(
        X: ndarray,
        projection: ndarray,
        extra_length: int,
        extra_data: Optional[ndarray] = None) -> ndarray:
    """Construct secondary augmented data.

    Args:
        X (ndarray): (Nc,Np).
        projection (ndarray): (Np,Np). Orthogonal projection matrix.
        extra_length (int): m.
        extra_data (ndarray, optional): (Nc,m). Extra data for training dataset.
            If None, prepared augmented data for test dataset.

    Returns:
        X_aug2 (ndarray): ((m+1)*Nc, 2*Np).
    """
    # basic information
    n_chans = X.shape[0]  # Nc
    n_points = projection.shape[0]  # Np

    # secondary augmented data
    X_aug2 = np.tile(np.zeros_like(X), (extra_length + 1, 2))  # ((m+1)*Nc,2*Np)
    if extra_data is not None:  # for training dataset
        X_temp = np.concatenate((X, extra_data), axis=-1)  # with extra length
        for el in range(extra_length + 1):
            sp, ep = el * n_chans, (el + 1) * n_chans
            X_aug2[sp:ep, :n_points] = X_temp[:, el:n_points + el]
            X_aug2[sp:ep, n_points:] = X_aug2[sp:ep, :n_points] @ projection
    else:  # for test dataset
        for el in range(extra_length + 1):
            sp, ep = el * n_chans, (el + 1) * n_chans
            X_aug2[sp:ep, :n_points - el] = X[:, el:n_points]
            X_aug2[sp:ep, n_points:] = X_aug2[sp:ep, :n_points] @ projection
    return X_aug2


def tdca_feature(
        X_test: ndarray,
        tdca_model: Dict[str, ndarray],
        projection: ndarray,
        extra_length: int) -> Dict[str, ndarray]:
    """The pattern matching process of TDCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        tdca_model (Dict[str, ndarray]): See details in _dsp_kernel().
        projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
        extra_length (int): m.

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Features of TDCA.
    """
    w, wX = tdca_model['w'], tdca_model['wX']
    n_events = wX.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))
    for nte in range(n_test):
        for nem in range(n_events):
            X_test_aug2 = tdca_augmentation(
                X=X_test[nte],
                projection=projection[nem],
                extra_length=extra_length
            )  # ((m+1)*Nc,2*Np)
            rho[nte, nem] = utils.pearson_corr(X=w @ X_test_aug2, Y=wX[nem])
    return {'rho': rho}


class TDCA(BasicDSP):
    def fit(
            self,
            X_train: ndarray,
            X_extra: ndarray,
            y_train: ndarray,
            projection: ndarray):
        """Train TDCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            X_extra (ndarray): (Ne*Nt,Nc,m). Extra training data for X_train.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
        """
        # basic information
        self.X_train = X_train
        self.X_extra = X_extra
        self.extra_length = self.X_extra.shape[-1]
        self.y_train = y_train
        self.projection = projection
        self.event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'

        # create secondary augmented data | (Ne*Nt,(el+1)*Nc,2*Np)
        self.X_train_aug2 = np.tile(
            A=np.zeros_like(self.X_train),
            reps=(1, (self.extra_length + 1), 2)
        )
        for ntr, label in enumerate(self.y_train):
            event_idx = list(self.event_type).index(label)
            self.X_train_aug2[ntr] = tdca_augmentation(
                X=self.X_train[ntr],
                projection=self.projection[event_idx],
                extra_length=self.extra_length,
                extra_data=self.X_extra[ntr]
            )

        # train DSP models & wXs
        self.training_model = dsp_kernel(
            X_train=self.X_train_aug2,
            y_train=self.y_train,
            n_components=self.n_components
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of DSP.
        """
        return tdca_feature(
            X_test=X_test,
            tdca_model=self.training_model,
            projection=self.projection,
            extra_length=self.extra_length
        )


class FB_TDCA(BasicFBDSP):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional): See details in utils.generate_filter_bank().
                Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=TDCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )
