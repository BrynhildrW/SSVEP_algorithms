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
    trail-normalization: TN

"""

# %% Basic modules
import utils

from abc import abstractmethod
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
from numpy import ndarray

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% Basic TRCA object
class BasicTRCA(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(
            self,
            standard: bool = True,
            ensemble: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
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
            sine_template: Optional[ndarray] = None):
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

        Returns:
            rho (ndarray): (Ne*Nte,Ne). Features.
            erho (ndarray): (Ne*Nte,Ne). Ensemble features.
        """
        pass

    def predict(self, X_test: ndarray) -> Union[Tuple[ndarray, ndarray],
                                                Tuple[int, int]]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            y_standard (ndarray or int): (Ne*Nte,). Predict label(s).
            y_ensemble (ndarray or int): (Ne*Nte,). Predict label(s) (ensemble).
        """
        self.features = self.transform(X_test)
        self.y_standard = self.event_type[np.argmax(self.features['rho'], axis=-1)]
        self.y_ensemble = self.event_type[np.argmax(self.features['erho'], axis=-1)]
        return self.y_standard, self.y_ensemble


class BasicFBTRCA(utils.FilterBank, ClassifierMixin):
    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np) or (Ne*Nte,Nc,Np).
                Test dataset.

        Returns:
            rho_fb (ndarray): (Nb,Ne*Nte,Ne). Features of each band.
            rho (ndarray): (Ne*Nte,Ne). Features of all bands.
            erho_fb (ndarray): (Nb,Ne*Nte,Ne). Ensemble features of each band.
            erho (ndarray): (Ne*Nte,Ne). Ensemble features of all bands.
        """
        if not self.with_filter_bank:  # tranform X_test
            X_test = self.fb_transform(X_test)  # (Nb,Ne*Nte,Nc,Np)
        sub_features = [se.transform(X_test[nse])
                        for nse, se in enumerate(self.sub_estimator)]
        rho_fb = np.stack([sf['rho'] for sf in sub_features], axis=0)
        erho_fb = np.stack([sf['erho'] for sf in sub_features], axis=0)
        rho = np.einsum('b,bte->te', self.bank_weights, utils.sign_sta(rho_fb))
        erho = np.einsum('b,bte->te', self.bank_weights, utils.sign_sta(erho_fb))
        features = {
            'rho_fb': rho_fb, 'rho': rho,
            'erho_fb': erho_fb, 'erho': erho
        }
        return features

    def predict(self, X_test: ndarray) -> Union[Tuple[ndarray, ndarray],
                                                Tuple[int, int]]:
        """Using filter-bank TRCA-like algorithm to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np) or (Ne*Nte,Nc,Np).
                Test dataset.

        Returns:
            y_standard (ndarray or int): (Ne*Nte,). Predict label(s).
            y_ensemble (ndarray or int): (Ne*Nte,). Predict label(s) (ensemble).
        """
        self.features = self.transform(X_test)
        event_type = self.sub_estimator[0].event_type
        self.y_standard = event_type[np.argmax(self.features['rho'], axis=-1)]
        self.y_ensemble = event_type[np.argmax(self.features['erho'], axis=-1)]
        return self.y_standard, self.y_ensemble


# %% 1. (ensemble) TRCA | (e)TRCA
def generate_trca_mat(
        X: ndarray,
        y: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """Generate covariance matrices Q & S for TRCA model.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Sklearn-style dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X.

    Returns:
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        X_mean (ndarray): (Ne,Nc,Np). Trial-averaged X.
    """
    # basic information
    X_mean = utils.generate_mean(X=X, y=y)  # (Ne,Nc,Np)
    n_events = X_mean.shape[0]  # Ne

    # covariance matrices: (Ne,Nc,Nc)
    Q = utils.generate_var(X=X, y=y)  # (Ne,Nc,Nc)
    S = np.zeros_like(Q)  # (Ne,Nc,Nc)
    for ne in range(n_events):   # Ne
        S[ne] = X_mean[ne] @ X_mean[ne].T  # (Nc,Nc)
    # 8-10 times slower: S = np.einsum('ecp,ehp->ech', X_mean, X_mean)
    return Q, S, X_mean


def solve_trca_func(
        Q: ndarray,
        S: ndarray,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> Tuple[ndarray, ndarray]:
    """Solve TRCA target function.

    Args:
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.

    Returns:
        w (ndarray): (Ne,Nk,Nc) or Ne*(Nk,Nc). Spatial filters of TRCA.
        ew (ndarray): (Ne*Nk,Nc). Ensemble spatial filter of eTRCA.
    """
    # basic information
    n_events = Q.shape[0]  # Ne
    n_chans = Q.shape[1]  # Nc

    # solve GEPs
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        w[ne] = utils.solve_gep(A=S[ne], B=Q[ne], n_components=n_components)
    ew = np.reshape(w, (n_events * n_components, n_chans), 'C')  # (Ne*Nk,Nc)
    return w, ew


def generate_trca_template(
        w: ndarray,
        ew: ndarray,
        X_mean: ndarray,
        standard: bool = True,
        ensemble: bool = True) -> Tuple[ndarray, ndarray]:
    """Generate (e)TRCA templates.

    Args:
        w (ndarray): (Ne,Nk,Nc). Spatial filters of TRCA.
        ew (ndarray): (Ne*Nk,Nc). Ensemble spatial filter of eTRCA.
        X_mean (ndarray): (Ne,Nc,Np). Trial-averaged data.
        standard (bool): Use TRCA model. Defaults to True.
        ensemble (bool): Use eTRCA model. Defaults to True.

    Returns:
        wX (ndarray): (Ne,Nk,Np). TRCA templates.
        ewX (ndarray): (Ne,Ne*Nk,Np). eTRCA templates.
    """
    # basic information
    n_events = X_mean.shape[0]  # Ne
    n_components = w.shape[1]  # Nk
    n_points = X_mean.shape[-1]  # Np

    # spatial filtering process
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    ewX = np.zeros((n_events, n_events * n_components, n_points))  # (Ne,Ne*Nk,Np)
    if standard:
        wX = utils.spatial_filtering(w=w, X=X_mean)  # (Ne,Nk,Np)
    if ensemble:
        ewX = utils.spatial_filtering(w=ew, X=X_mean)  # (Ne,Ne*Nk,Np)
    return wX, ewX


def trca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1) -> Dict[str, ndarray]:
    """
    The modeling process of (e)TRCA.

    Parameters
    -------
    X_train : ndarray, shape (Ne*Nt,Nc,Np).
        Sklearn-style training dataset. Nt>=2.
    y_train : ndarray, shape (Ne*Nt,).
        Labels for X_train.
    standard : bool.
        Standard model. Defaults to True.
    ensemble : bool.
        Ensemble model. Defaults to True.
    n_components : int.
        Number of eigenvectors picked as filters. Defaults to 1.

    Returns
    -------
    Q : ndarray, shape (Ne,Nc,Nc).
        Covariance of original data.
    S : ndarray, shape (Ne,Nc,Nc).
        Covariance of template data.
    w : ndarray, shape (Ne,Nk,Nc).
        Spatial filters of TRCA.
    ew : ndarray, shape (Ne*Nk,Nc).
        Common spatial filter of eTRCA.
    wX : ndarray, shape (Ne,Nk,Np).
        TRCA templates.
    ewX : ndarray, shape (Ne,Ne*Nk,Np).
        eTRCA templates.
    """
    # solve target functions
    Q, S, X_mean = generate_trca_mat(X=X_train, y=y_train)
    w, ew = solve_trca_func(Q=Q, S=S, n_components=n_components)

    # generate spatial-filtered templates
    wX, ewX = generate_trca_template(
        w=w,
        ew=ew,
        X_mean=X_mean,
        standard=standard,
        ensemble=ensemble
    )
    return {'Q': Q, 'S': S, 'w': w, 'ew': ew, 'wX': wX, 'ewX': ewX}


def trca_feature(
        X_test: ndarray,
        trca_model: Dict[str, ndarray],
        standard: bool = True,
        ensemble: bool = True) -> Dict[str, ndarray]:
    """
    The pattern matching process of (e)TRCA.

    Parameters
    -------
    X_test : ndarray, shape (Ne*Nte,Nc,Np).
        Test dataset.
    trca_model : Dict[str, ndarray].
        See details in trca_kernel().
    standard : bool.
        Standard model. Defaults to True.
    ensemble : bool.
        Ensemble model. Defaults to True.

    Returns
    -------
    rho : ndarray, shape (Ne*Nte,Ne).
        Features of TRCA.
    erho : ndarray, shape (Ne*Nte,Ne).
        Features of eTRCA.
    """
    # basic information
    n_test = X_test.shape[0]  # Ne*Nte

    # reshape & standardization for faster computing
    if standard:
        w, wX = trca_model['w'], trca_model['wX']  # (Ne,Nk,Nc), (Ne,Nk,Np)
        n_events = wX.shape[0]  # Ne
        wX = utils.fast_stan_2d(np.reshape(wX, (n_events, -1), 'C'))  # (Ne,Nk*Np)
    if ensemble:
        ew, ewX = trca_model['ew'], trca_model['ewX']  # (Ne*Nk,Nc), (Ne,Ne*Nk,Np)
        n_events = ewX.shape[0]  # Ne
        ewX = utils.fast_stan_2d(np.reshape(ewX, (n_events, -1), 'C'))  # (Ne,Ne*Nk*Np)

    # pattern matching
    rho = np.zeros((n_test, n_events))
    erho = np.zeros_like(rho)
    if standard:
        for nte in range(n_test):
            X_temp = np.reshape(w @ X_test[nte], (n_events, -1), 'C')  # (Ne,Nk*Np)
            X_temp = utils.fast_stan_2d(X_temp)  # (Ne,Nk*Np)

            rho[nte] = utils.fast_corr_2d(X=X_temp, Y=wX) / X_temp.shape[-1]
    if ensemble:
        for nte in range(n_test):
            X_temp = np.tile(
                A=np.reshape(ew @ X_test[nte], -1, 'C'),
                reps=(n_events, 1)
            )  # (Ne*Nk,Np) -reshape-> (Ne*Nk*Np,) -repeat-> (Ne,Ne*Nk*Np)
            X_temp = utils.fast_stan_2d(X_temp)  # (Ne,Ne*Nk*Np)

            erho[nte] = utils.fast_corr_2d(X=X_temp, Y=ewX) / X_temp.shape[-1]
    return {'rho': rho, 'erho': erho}


class TRCA(BasicTRCA):
    def fit(self, X_train: ndarray, y_train: ndarray):
        """
        Train (e)TRCA model.

        Parameters
        -------
        X_train : ndarray, shape (Ne*Nt,Nc,Np).
            Training dataset. Nt>=2.
        y_train : ndarray, shape (Ne*Nt,).
            Labels for X_train.
        """
        # basic information
        self.X_train = utils.fast_stan_3d(X_train)
        self.y_train = y_train
        self.event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'

        # train TRCA filters & templates
        self.training_model = trca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            standard=self.standard,
            ensemble=self.ensemble,
            n_components=self.n_components
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """
        Transform test dataset to discriminant features.

        Parameters
        -------
        X_test : ndarray, shape (Ne*Nte,Nc,Np).
            Test dataset.

        Returns
        -------
        rho : ndarray, shape (Ne*Nte,Ne).
            Decision coefficients of TRCA.
        erho : ndarray, shape (Ne*Nte,Ne).
            Ensemble decision coefficients of eTRCA.
        """
        return trca_feature(
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
            n_components: int = 1):
        """
        Basic configuration.

        Parameters
        -------
        filter_bank : List[ndarray], optional.
            See details in utils.generate_filter_bank(). Defaults to None.
        with_filter_bank : bool.
            Whether the input data has been FB-preprocessed. Defaults to True.
        standard : bool.
            Standard model. Defaults to True.
        ensemble : bool.
            Ensemble model. Defaults to True.
        n_components : int.
            Number of eigenvectors picked as filters. Defaults to 1.
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
def solve_mstrca_func(
        Q: ndarray,
        S: ndarray,
        event_type: ndarray,
        events_group: Dict[str, List[int]],
        n_components: int = 1) -> Tuple[ndarray, ndarray]:
    """Solve ms-TRCA target function.

    Args:
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        event_type (ndarray): (Ne,). All kinds of labels.
        events_group (Dict[str, List[int]]): {'event_id':[idx_1,idx_2,...]}.
            Event indices being emerged for each event.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.

    Returns:
        w (ndarray): (Ne,Nk,Nc). Spatial filters of ms-TRCA.
        ew (ndarray): (Ne*Nk,Nc). Ensemble spatial filter of ms-eTRCA.
    """
    # basic information
    n_events, n_chans = Q.shape[0], Q.shape[1]  # Ne,Nc

    # solve GEPs with merged data
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        merged_indices = events_group[str(event_type[ne])]
        Q_temp = np.sum(Q[merged_indices], axis=0)  # (Nc,Nc)
        S_temp = np.sum(S[merged_indices], axis=0)  # (Nc,Nc)
        w[ne] = utils.solve_gep(A=S_temp, B=Q_temp, n_components=n_components)
    ew = np.reshape(w, (n_events * n_components, n_chans), 'C')  # (Ne*Nk,Nc)
    return w, ew


def mstrca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        events_group: Dict[str, List[int]],
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The modeling process of ms-(e)TRCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        events_group (Dict[str, List[int]]): {'event_id':[idx_1,idx_2,...]}.
            Event indices being emerged for each event.
        standard (bool): Use TRCA model. Defaults to True.
        ensemble (bool): Use eTRCA model. Defaults to True.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.

    Returns:
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        w (ndarray): (Ne,Nk,Nc). Spatial filters of ms-TRCA.
        ew (ndarray): (Ne*Nk,Nc). Common spatial filter of ms-eTRCA.
        wX (ndarray): (Ne,Nk,Np). ms-TRCA templates.
        ewX (ndarray): (Ne,Ne*Nk,Np). ms-eTRCA templates.
    """
    # solve target functions
    Q_total, S_total, X_mean = generate_trca_mat(X=X_train, y=y_train)
    w, ew = solve_mstrca_func(
        Q=Q_total,
        S=S_total,
        event_type=np.unique(y_train),
        events_group=events_group,
        n_components=n_components
    )

    # generate spatial-filtered templates
    wX, ewX = generate_trca_template(
        X_mean=X_mean,
        w=w,
        ew=ew,
        standard=standard,
        ensemble=ensemble
    )
    return {'Q': Q_total, 'S': S_total, 'w': w, 'ew': ew, 'wX': wX, 'ewX': ewX}


class MS_TRCA(TRCA):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            events_group: Optional[Dict[str, List[int]]] = None,
            d: int = 2):
        """Train ms-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            events_group (Dict[str, List[int]], optional):
                {'event_id':[idx_1,idx_2,...]}.
                If None, events_group will be generated according to parameter 'd'.
            d (int): The range of events to be merged. Defaults to 2.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.d = d
        self.event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        if events_group is not None:
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(
                event_type=self.event_type,
                d=self.d
            )
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'

        # train ms-TRCA models & templates
        self.training_model = mstrca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            events_group=self.events_group,
            standard=self.standard,
            ensemble=self.ensemble,
            n_components=self.n_components
        )


class FB_MS_TRCA(BasicFBTRCA):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            standard: bool = True,
            ensemble: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
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
def generate_trcar_mat(
        X: ndarray,
        y: ndarray,
        projection: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """Generate covariance matrices Q & S for TRCA-R model.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Sklearn-style dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X.
        projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.

    Returns:
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        X_mean (ndarray): (Ne,Nc,Np). Trial-averaged X.
    """
    # basic information
    X_mean = utils.generate_mean(X=X, y=y)  # (Ne,Nc,Np)
    n_events = X_mean.shape[0]  # Ne

    # covariance matrices: (Ne,Nc,Nc)
    Q = utils.generate_var(X=X, y=y)  # (Ne,Nc,Nc)
    S = np.zeros_like(Q)  # (Ne,Nc,Nc)
    for ne in range(n_events):   # Ne
        X_pro = X_mean[ne] @ projection[ne]  # projected X_mean: (Nc,Np)
        S[ne] = X_pro @ X_pro.T  # (Nc,Nc)
    # slower: X_pro = np.einsum('ecp,epo->eco', X_mean, projection)
    # slower: S = np.einsum('ecp,ehp->ech', X_pro, X_pro)
    return Q, S, X_mean


def trcar_kernel(
        X_train: ndarray,
        y_train: ndarray,
        projection: ndarray,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The modeling process of (e)TRCA-R.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
        standard (bool): Use TRCA model. Defaults to True.
        ensemble (bool): Use eTRCA model. Defaults to True.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.

    Returns:
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        w (ndarray): (Ne,Nk,Nc). Spatial filters of ms-TRCA.
        ew (ndarray): (Ne*Nk,Nc). Common spatial filter of ms-eTRCA.
        wX (ndarray): (Ne,Nk,Np). TRCA-R templates.
        ewX (ndarray): (Ne,Ne*Nk,Np). eTRCA-R templates.
    """
    # solve target functions
    Q, S, X_mean = generate_trcar_mat(
        X=X_train,
        y=y_train,
        projection=projection
    )
    w, ew = solve_trca_func(Q=Q, S=S, n_components=n_components)

    # generate spatial-filtered templates
    wX, ewX = generate_trca_template(
        X_mean=X_mean,
        w=w,
        ew=ew,
        standard=standard,
        ensemble=ensemble
    )
    return {'Q': Q, 'S': S, 'w': w, 'ew': ew, 'wX': wX, 'ewX': ewX}


class TRCA_R(TRCA):
    def fit(
            self,
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
        self.event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'
        self.projection = projection

        # train TRCA-R models & templates
        self.training_model = trcar_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            projection=self.projection,
            standard=self.standard,
            ensemble=self.ensemble,
            n_components=self.n_components
        )


class FB_TRCA_R(BasicFBTRCA):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            standard: bool = True,
            ensemble: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
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
def generate_sctrca_mat(
        X: ndarray,
        y: ndarray,
        sine_template: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """Generate covariance matrices Q & S for sc-TRCA model.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Sklearn-style dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.

    Returns:
        Q (ndarray): (Ne,Nc+2Nh,Nc+2Nh). Covariance of X & sine_template.
        S (ndarray): (Ne,Nc+2Nh,Nc+2Nh). Covariance of [[X_mean],[sine_template]].
        X_mean (ndarray): (Ne,Nc,Np). Trial-averaged X.
    """
    # basic information
    X_mean = utils.generate_mean(X=X, y=y)  # (Ne,Nc,Np)
    X_var = utils.generate_var(X=X, y=y)  # (Ne,Nc,Nc)
    n_events = X_mean.shape[0]  # Ne
    n_chans = X_mean.shape[1]  # Nc
    n_dims = sine_template.shape[1]  # 2*Nh
    n_train = np.array([np.sum(y == et) for et in np.unique(y)])

    # covariance matrices: (Ne,Nc+2Nh,Nc+2Nh)
    S = np.zeros((n_events, n_chans + n_dims, n_chans + n_dims))
    Q = np.zeros_like(S)
    for ne in range(n_events):
        train_trials = n_train[ne]  # Nt

        # Cxx = X_var[ne]
        Cyy = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh)
        Cxmxm = X_mean[ne] @ X_mean[ne].T  # (Nc,Nc)
        Cxmy = X_mean[ne] @ sine_template[ne].T  # (Nc,2Nh)

        # block covariance matrix S: [[S11,S12],[S21,S22]]
        S[ne, :n_chans, :n_chans] = Cxmxm  # S11
        S[ne, :n_chans, n_chans:] = (1 - 1 / train_trials) * Cxmy  # S12
        S[ne, n_chans:, :n_chans] = S[ne, :n_chans, n_chans:].T  # S21
        S[ne, n_chans:, n_chans:] = Cyy  # S22

        # block covariance matrix Q: blkdiag(Q1,Q2)
        Q[ne, :n_chans, :n_chans] = X_var[ne]  # Q1
        Q[ne, n_chans:, n_chans:] = train_trials * Cyy  # Q2
    return Q, S, X_mean


def solve_sctrca_func(
        Q: ndarray,
        S: ndarray,
        n_chans: int,
        n_components: int = 1) -> Tuple[ndarray, ndarray,
                                        ndarray, ndarray]:
    """Solve sc-TRCA target function.

    Args:
        Q (ndarray): (Ne,Nc+2Nh,Nc+2Nh). Covariance of X & sine_template.
        S (ndarray): (Ne,Nc+2Nh,Nc+2Nh). Covariance of [[X_mean],[sine_template]].
        n_chans (int): Number of EEG channels.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Defaults to 1.

    Returns:
        u (ndarray): (Ne,Nk,Nc). Spatial filters (EEG signal).
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters (sinusoidal signal).
        eu (ndarray): (Ne*Nk,Nc). Concatenated filter (EEG signal).
        ev (ndarray): (Ne*Nk,2*Nh). Concatenated filter (sinusoidal signal).
    """
    # basic information
    n_events = Q.shape[0]  # Ne
    n_dims = Q.shape[1] - n_chans  # 2*Nh

    # solve GEPs
    u = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    v = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,2Nh)
    for ne in range(n_events):
        w = utils.solve_gep(A=S[ne], B=Q[ne], n_components=n_components)
        u[ne] = w[:, :n_chans]  # (Nk,Nc)
        v[ne] = w[:, n_chans:]  # (Nk,2Nh)
    eu = np.reshape(u, (n_events * n_components, n_chans), 'C')  # (Ne*Nk,Nc)
    ev = np.reshape(v, (n_events * n_components, n_dims), 'C')  # (Ne*Nk,2*Nh)
    return u, v, eu, ev


def generate_sctrca_template(
        u: ndarray,
        v: ndarray,
        eu: ndarray,
        ev: ndarray,
        X_mean: ndarray,
        sine_template: ndarray,
        standard: bool = True,
        ensemble: bool = True) -> Tuple[ndarray, ndarray,
                                        ndarray, ndarray]:
    """Generate sc-(e)TRCA templates.

    Args:
        u (ndarray): (Ne,Nk,Nc). Spatial filters (EEG signal).
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters (sinusoidal signal).
        eu (ndarray): (Ne*Nk,Nc). Concatenated filter (EEG signal).
        ev (ndarray): (Ne*Nk,2*Nh). Concatenated filter (sinusoidal signal).
        X_mean (ndarray): (Ne,Nc,Np). Trial-averaged data.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        standard (bool): Use TRCA model. Defaults to True.
        ensemble (bool): Use eTRCA model. Defaults to True.

    Returns:
        uX, vY (ndarray): (Ne,Nk,Np). sc-TRCA templates.
        euX, evY (ndarray): (Ne,Ne*Nk,Np). sc-eTRCA templates.
    """
    # basic information
    n_events = X_mean.shape[0]  # Ne
    n_points = X_mean.shape[-1]  # Np
    n_components = u.shape[1]  # Nk

    # spatial filtering process
    uX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    vY = np.zeros_like(uX)
    euX = np.zeros((n_events, eu.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros_like(euX)
    if standard:
        uX = utils.spatial_filtering(w=u, X=X_mean)
        vY = utils.spatial_filtering(w=v, X=sine_template)
    if ensemble:
        euX = utils.spatial_filtering(w=eu, X=X_mean)
        evY = utils.spatial_filtering(w=ev, X=sine_template)
    return uX, vY, euX, evY


def sctrca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        standard: bool = True,
        ensemble: bool = True,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The modeling process of sc-(e)TRCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        standard (bool): Use TRCA model. Defaults to True.
        ensemble (bool): Use eTRCA model. Defaults to True.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.

    Returns:
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data & average templates.
        S (ndarray): (Ne,Nc,Nc). Covariance of templates.
        u (ndarray): (Ne,Nk,Nc). Spatial filters (EEG signal).
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters (sinusoidal signal).
        eu (ndarray): (Ne*Nk,Nc). Concatenated filter (EEG signal).
        ev (ndarray): (Ne*Nk,2*Nh). Concatenated filter (sinusoidal signal).
        uX, vY (ndarray): (Ne,Nk,Np). sc-TRCA templates.
        euX, evY (ndarray): (Ne,Ne*Nk,Np). sc-eTRCA templates.
    """
    # basic information
    n_chans = X_train.shape[1]  # Nc

    # solve target functions
    Q, S, X_mean = generate_sctrca_mat(
        X=X_train,
        y=y_train,
        sine_template=sine_template
    )
    u, v, eu, ev = solve_sctrca_func(
        Q=Q,
        S=S,
        n_chans=n_chans,
        n_components=n_components
    )

    # generate spatial-filtered templates
    uX, vY, euX, evY = generate_sctrca_template(
        u=u,
        v=v,
        eu=eu,
        ev=ev,
        X_mean=X_mean,
        sine_template=sine_template,
        standard=standard,
        ensemble=ensemble
    )
    return {
        'Q': Q, 'S': S,
        'u': u, 'v': v, 'eu': eu, 'ev': ev,
        'uX': uX, 'vY': vY, 'euX': euX, 'evY': evY
    }


def sctrca_feature(
        X_test: ndarray,
        sctrca_model: Dict[str, ndarray],
        standard: bool,
        ensemble: bool) -> Dict[str, ndarray]:
    """The pattern matching process of sc-(e)TRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        sctrca_model (dict): See details in sctrca_kernel().
        standard (bool): Standard model. Defaults to True.
        ensemble (bool): Ensemble model. Defaults to True.

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Features of sc-TRCA.
        rho_eeg (ndarray): (Ne*Nte,Ne). (EEG part).
        rho_sin (ndarray): (Ne*Nte,Ne). (Sinusoidal signal part).
        erho (ndarray): (Ne*Nte,Ne). Features of sc-eTRCA.
        erho_eeg (ndarray): (Ne*Nte,Ne). (EEG part).
        erho_sin (ndarray): (Ne*Nte,Ne). (Sinusoidal signal part).
    """
    # basic information & load in model
    u, eu = sctrca_model['u'], sctrca_model['eu']  # (Ne,Nk,Nc), (Ne*Nk,Nc)
    uX, euX = sctrca_model['uX'], sctrca_model['euX']  # (Ne,Nk*Np), (Ne,Ne*Nk*Np)
    vY, evY = sctrca_model['vY'], sctrca_model['evY']  # (Ne,Nk*Np), (Ne,Ne*Nk*Np)
    n_events = u.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte

    # reshape & standardization for faster computing
    uX = utils.fast_stan_2d(np.reshape(uX, (n_events, -1), 'C'))  # (Ne,Nk*Np)
    vY = utils.fast_stan_2d(np.reshape(vY, (n_events, -1), 'C'))  # (Ne,Nk*Np)
    euX = utils.fast_stan_2d(np.reshape(euX, (n_events, -1), 'C'))  # (Ne,Ne*Nk*Np)
    evY = utils.fast_stan_2d(np.reshape(evY, (n_events, -1), 'C'))  # (Ne,Ne*Nk*Np)

    # pattern matching
    rho = np.zeros((n_test, n_events))  # (Ne*Nte,Ne)
    rho_eeg, rho_sin = np.zeros_like(rho), np.zeros_like(rho)
    erho = np.zeros_like(rho)
    erho_eeg, erho_sin = np.zeros_like(rho), np.zeros_like(rho)
    if standard:
        for nte in range(n_test):
            X_temp = np.reshape(u @ X_test[nte], (n_events, -1), 'C')  # (Ne,Nk*Np)
            X_temp = utils.fast_stan_2d(X_temp)  # (Ne,Nk*Np)

            rho_eeg[nte] = utils.fast_corr_2d(X=X_temp, Y=uX) / X_temp.shape[-1]
            rho_sin[nte] = utils.fast_corr_2d(X=X_temp, Y=vY) / X_temp.shape[-1]
        rho = utils.combine_feature([rho_eeg, rho_sin])
    if ensemble:
        for nte in range(n_test):
            X_temp = np.tile(
                A=np.reshape(eu @ X_test[nte], -1, 'C'),
                reps=(n_events, 1)
            )  # (Ne*Nk,Np) -reshape-> (Ne*Nk*Np,) -repeat-> (Ne,Ne*Nk*Np)
            X_temp = utils.fast_stan_2d(X_temp)  # (Ne,Ne*Nk*Np)

            erho_eeg[nte] = utils.fast_corr_2d(X=X_temp, Y=euX) / X_temp.shape[-1]
            erho_sin[nte] = utils.fast_corr_2d(X=X_temp, Y=evY) / X_temp.shape[-1]
        erho = utils.combine_feature([erho_eeg, erho_sin])
    return {
        'rho': rho, 'rho_eeg': rho_eeg, 'rho_sin': rho_sin,
        'erho': erho, 'erho_eeg': erho_eeg, 'erho_sin': erho_sin
    }


class SC_TRCA(BasicTRCA):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """Train sc-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.event_type = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'

        # train sc-TRCA models & templates
        self.training_model = sctrca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=sine_template,
            standard=self.standard,
            ensemble=self.ensemble,
            n_components=self.n_components
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho (ndarray): (Ne*Nte,Ne). Features of sc-TRCA.
            rho_eeg (ndarray): (Ne*Nte,Ne). (EEG part).
            rho_sin (ndarray): (Ne*Nte,Ne). (Sinusoidal signal part).
            erho (ndarray): (Ne*Nte,Ne). Features of sc-eTRCA.
            erho_eeg (ndarray): (Ne*Nte,Ne). (EEG part).
            erho_sin (ndarray): (Ne*Nte,Ne). (Sinusoidal signal part).
        """
        return sctrca_feature(
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
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
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


# %% 6. cross-correlation TRCA | xTRCA


# %% 7. latency-aligned TRCA | LA-TRCA
