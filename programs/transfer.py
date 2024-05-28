# -*- coding: utf-8 -*-
"""
@ Author: Brynhildr Wu
@ Email: brynhildrwu@gmail.com

Transfer learning based on matrix decomposition.
    (1) SAME: https://ieeexplore.ieee.org/document/9971465/
            DOI: 10.1109/TBME.2022.3227036
    (2) TNSRE_20233250953: https://ieeexplore.ieee.org/document/10057002/
            DOI: 10.1109/TNSRE.2023.3250953
    (3) stCCA: https://ieeexplore.ieee.org/document/9177172/
            DOI: 10.1109/TNSRE.2020.3019276
    (4) tlCCA: https://ieeexplore.ieee.org/document/9354064/
            DOI: 10.1109/TASE.2021.3054741
    (5) sd-LST: https://ieeexplore.ieee.org/document/9967845/
            DOI: 10.1109/TNSRE.2022.3225878
    (6) TNSRE_20233305202: https://ieeexplore.ieee.org/document/10216996/
            DOI: 10.1109/TNSRE.2023.3305202
    (7) gTRCA: http://www.nature.com/articles/s41598-019-56962-2
            DOI: 10.1038/s41598-019-56962-2
    (8) IISMC: https://ieeexplore.ieee.org/document/9350285/
            DOI: 10.1109/TNSRE.2021.3057938
    (9) ASS-IISCCA: https://ieeexplore.ieee.org/document/10159132/
            DOI: 10.1109/TNSRE.2023.3288397
    (10) SDA-CLS
    (11)

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
    n_subjects: Ns

"""

# %% Basic modules
from abc import abstractmethod

import utils

import cca
import trca
import dsp

from typing import Optional, List, Tuple, Dict, Union
from numpy import ndarray

import numpy as np
import scipy.linalg as sLA

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% Basic Transfer object
class BasicTransfer(BaseEstimator, TransformerMixin, ClassifierMixin):
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
        """
        # config model
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble

    @abstractmethod
    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        pass

    @abstractmethod
    def inter_source_training(self):
        """Inter-domain model training for multiple source datasets."""
        pass

    @abstractmethod
    def transfer_learning(self):
        """Transfer learning between source & target datasets."""
        pass

    @abstractmethod
    def target_augmentation(self):
        """Data augmentation for target dataset."""
        pass

    @abstractmethod
    def dist_calc(self):
        """Calculate spatial distance of source & target datasets."""
        pass

    @abstractmethod
    def weight_calc(self):
        """Optimize the transfer weight for each source domain."""
        pass

    @abstractmethod
    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        pass

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            X_source: List[ndarray],
            y_source: List[ndarray],
            sine_template: Optional[ndarray] = None):
        """Load data and train classification models.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset (target domain). Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            sine_template (ndarray, optional): (Ne,2*Nh,Np). Sinusoidal template.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template

        # main process
        self.intra_source_training()
        self.inter_source_training()
        self.transfer_learning()
        self.target_augmentation()
        self.dist_calc()
        self.weight_calc()
        self.intra_target_training()
        return self

    @abstractmethod
    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients, etc.
        """
        pass

    def predict(
            self,
            X_test: ndarray) -> Union[
                Tuple[ndarray, ndarray],
                Tuple[int, int], ndarray, int]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Union[Tuple[ndarray, ndarray], Tuple[int, int], ndarray, int]
            y_pred (ndarray): (Ne*Nte,). Predict label(s).
        """
        self.features = self.transform(X_test)
        event_type = self.target_train_info['event_type']
        self.y_pred = event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


class BasicFBTransfer(utils.FilterBank, ClassifierMixin):
    def predict(
            self,
            X_test: ndarray) -> Union[
                Tuple[ndarray, ndarray],
                Tuple[int, int], ndarray, int]:
        """Using filter-bank transfer algorithms to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return: Union[Tuple[ndarray, ndarray], Tuple[int, int], ndarray, int]
            y_pred (ndarray): (Ne*Nte,). Predict label(s).
        """
        self.features = self.transform(X_test)
        event_type = self.sub_estimator[0].target_train_info['event_type']
        self.y_pred = event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


# %% 1. source aliasing matrix estimation, SAME


# %% 2. 10.1109/TNSRE.2023.3250953
def tnsre_20233250953_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
    """Intra-domain modeling process of algorithm TNSRE_20233250953.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters.

    Returns: Dict[str, ndarray]
        Q (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Covariance matrices.
        S (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Variance matrices. Q^{-1}Sw = lambda w.
        w (ndarray): (Ne,Nk,Nc). Spatial filters for original signal.
        u (ndarray): (Ne,Nk,Nc). Spatial filters for averaged template.
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters for sinusoidal template.
        ew (ndarray): (Ne*Nk,Nc). Concatenated w.
        eu (ndarray): (Ne*Nk,Nc). Concatenated u.
        ev (ndarray): (Ne*Nk,2*Nh). Concatenated v.
        euX (ndarray): (Ne,Ne*Nk*Np). Filtered averaged templates (reshaped).
        evY (ndarray): (Ne,Ne*Nk*Np). Filtered sinusoidal templates (reshaped).
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    n_dims = sine_template.shape[1]  # 2*Nh

    # block covariance matrices: S & Q
    S = np.tile(
        A=np.eye(2 * n_chans + n_dims)[None, ...],
        reps=(n_events, 1, 1)
    )  # (Ne,2*Nc+2*Nh,2*Nc+2*Nh)
    Q = np.zeros_like(S)
    X_sum = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    X_mean = np.zeros_like(X_sum)  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        train_trials = n_train[ne]  # Nt
        assert train_trials > 1, 'The number of training samples is too small!'

        X_temp = X_train[y_train == et]  # (Nt,Nc,Np)
        X_sum[ne] = np.sum(X_temp, axis=0)
        X_mean[ne] = X_sum[ne] / train_trials

        Cxsxs = X_sum[ne] @ X_sum[ne].T  # (Nc,Nc)
        Cxsxm = X_sum[ne] @ X_mean[ne].T  # (Nc,Nc)
        Cxmxm = X_mean[ne] @ X_mean[ne].T  # (Nc,Nc)
        Cxsy = X_sum[ne] @ sine_template[ne].T  # (Nc,2*Nh)
        Cxmy = X_mean[ne] @ sine_template[ne].T  # (Nc,2*Nh)
        Cyy = sine_template[ne] @ sine_template[ne].T  # (2*Nh,2*Nh)
        Cxx = np.zeros((n_chans, n_chans))  # (Nc,Nc)
        for tt in range(train_trials):
            Cxx += X_temp[tt] @ X_temp[tt].T

        # block covariance matrices S: [[S11,S12,S13],[S21,S22,S23],[S31,S32,S33]]
        # S11: inter-trial covariance
        S[ne, :n_chans, :n_chans] = Cxsxs

        # S12 & S21.T covariance between the SSVEP trials & the individual template
        S[ne, :n_chans, n_chans:2 * n_chans] = Cxsxm
        S[ne, n_chans:2 * n_chans, :n_chans] = Cxsxm.T

        # S13 & S31.T: similarity between the SSVEP trials & sinusoidal template
        S[ne, :n_chans, 2 * n_chans:] = Cxsy
        S[ne, 2 * n_chans:, :n_chans] = Cxsy.T

        # S23 & S32.T: covariance between the individual template & sinusoidal template
        S[ne, n_chans:2 * n_chans, 2 * n_chans:] = Cxmy
        S[ne, 2 * n_chans:, n_chans:2 * n_chans] = Cxmy.T

        # S22 & S33: variance of individual template & sinusoidal template
        S[ne, n_chans:2 * n_chans, n_chans:2 * n_chans] = 2 * Cxmxm
        S[ne, 2 * n_chans:, 2 * n_chans:] = 2 * Cyy

        # block covariance matrices Q: blkdiag(Q1,Q2,Q3)
        # Q1: variance of the single-trial SSVEP
        Q[ne, :n_chans, :n_chans] = Cxx

        # Q2 & Q3: variance of individual template & sinusoidal template
        Q[ne, n_chans:2 * n_chans, n_chans:2 * n_chans] = Cxmxm
        Q[ne, 2 * n_chans:, 2 * n_chans:] = Cyy

    # GEPs | train spatial filters
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    u = np.zeros_like(w)  # (Ne,Nk,Nc)
    v = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,2*Nh)
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(A=S[ne], B=Q[ne], n_components=n_components)
        w[ne] = spatial_filter[:, :n_chans]  # for raw signal
        u[ne] = spatial_filter[:, n_chans:2 * n_chans]  # for averaged template
        v[ne] = spatial_filter[:, 2 * n_chans:]  # for sinusoidal template
    ew = np.reshape(w, (n_events * n_components, n_chans), 'C')  # (Ne*Nk,Nc)
    eu = np.reshape(u, (n_events * n_components, n_chans), 'C')  # (Ne*Nk,Nc)
    ev = np.reshape(v, (n_events * n_components, n_dims), 'C')  # (Ne*Nk,Nc)

    # signal templates
    euX = np.zeros((n_events, eu.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros((n_events, ev.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    for ne in range(n_events):
        euX[ne] = eu @ X_mean[ne]
        evY[ne] = ev @ sine_template[ne]
    euX = utils.fast_stan_3d(euX)
    evY = utils.fast_stan_3d(evY)
    euX = np.reshape(euX, (n_events, -1), 'C')  # (Ne,Ne*Nk*Np)
    evY = np.reshape(evY, (n_events, -1), 'C')  # (Ne,Ne*Nk*Np)

    # training model
    training_model = {
        'Q': Q, 'S': S,
        'w': w, 'u': u, 'v': v, 'ew': ew, 'eu': eu, 'ev': ev,
        'euX': euX, 'evY': evY
    }
    return training_model


def tnsre_20233250953_feature(
        X_test: ndarray,
        source_model: Dict[str, ndarray],
        trans_model: Dict[str, ndarray],
        target_model: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """The pattern matching process of algorithm TNSRE_20233250953.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        source_model (dict): {'euX_source': ndarray (Ns,Ne,Ne*Nk,Np),
                              'evY_source': ndarray (Ns,Ne,Ne*Nk,Np)}
            See details in TNSRE_20233250953.intra_source_training()
        trans_model (dict): {'euX_trans': ndarray (Ns,Ne,Ne*Nk,Nc),
                             'evY_trans': ndarray (Ns,Ne,Ne*Nk,Nc),
                             'weight_euX': ndarray (Ns,Ne),
                             'weight_evY': ndarray (Ns,Ne)}
            See details in:
            TNSRE_20233250953.transfer_learning();
            TNSRE_20233250953.distance_calculation();
            TNSRE_20233250953.weight_optimization().
        target_model (dict): {'ew': ndarray (Ne*Nk,Nc),
                              'euX': ndarray (Ne,Ne*Nk*Np),
                              'evY': ndarray (Ne,Ne*Nk*Np)}
            See details in TNSRE_20233250953.intra_target_training()

    Returns: Dict[str, ndarray]
        rho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models & basic information
    euX_source = source_model['euX_source']  # (Ns,Ne,Ne*Nk*Np)
    evY_source = source_model['evY_source']  # (Ns,Ne,Ne*Nk*Np)
    euX_trans = trans_model['euX_trans']  # (Ns,Ne,Ne*Nk,Nc)
    evY_trans = trans_model['evY_trans']  # (Ns,Ne,Ne*Nk,Nc)
    weight_euX = trans_model['weight_euX']  # (Ns,Ne)
    weight_evY = trans_model['weight_evY']  # (Ns,Ne)
    ew_target = target_model['ew']  # (Ne*Nk,Nc)
    euX_target = target_model['euX']  # (Ne,Ne*Nk*Np)
    evY_target = target_model['evY']  # (Ne,Ne*Nk*Np)
    n_subjects = euX_source.shape[0]  # Ns
    n_events = euX_source.shape[1]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    # n_points = X_test.shape[-1]  # Np, unnecessary

    # 4-D features
    rho_temp = np.zeros((n_test, n_events, 4))  # (Ne*Nte,Ne,4)
    for nte in range(n_test):
        X_trans_x = np.reshape(
            a=utils.fast_stan_4d(euX_trans @ X_test[nte]),
            newshape=(n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Ne*Nk*Np)
        X_trans_y = np.reshape(
            a=utils.fast_stan_4d(evY_trans @ X_test[nte]),
            newshape=(n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Ne*Nk*Np)
        X_temp = np.tile(
            A=np.reshape(
                a=utils.fast_stan_2d(ew_target @ X_test[nte]),
                newshape=-1,
                order='C'
            ),
            reps=(n_events, 1)
        )  # (Ne,Ne*Nk*Np)

        # rho 1 & 2: transferred pattern matching
        rho_temp[nte, :, 0] = np.sum(
            a=weight_euX * utils.fast_corr_3d(X=X_trans_x, Y=euX_source),
            axis=0
        )
        rho_temp[nte, :, 1] = np.sum(
            a=weight_evY * utils.fast_corr_3d(X=X_trans_y, Y=evY_source),
            axis=0
        )

        # rho 3 & 4: target-domain pattern matching
        rho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp, Y=euX_target)
        rho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp, Y=evY_target)
    # rho_temp /= n_points  # real Pearson correlation coefficients in scale
    features = {
        'rho_temp': rho_temp,
        'rho': utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1],
            rho_temp[..., 2],
            rho_temp[..., 3]
        ])
    }
    return features


class TNSRE_20233250953(BasicTransfer):
    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        # basic information & initialization
        self.source_intra_model, self.source_model = [], {}
        euX_source, evY_source = [], []  # List[ndarray]: Ns*(Ne,Ne*Nk,Np)

        # obtain source model
        for nsub in range(self.n_subjects):
            intra_model = tnsre_20233250953_kernel(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                train_info=self.source_train_info[nsub],
                n_components=self.n_components
            )
            self.source_intra_model.append(intra_model)
            euX_source.append(intra_model['euX'])  # (Ne,Ne*Nk*Np)
            evY_source.append(intra_model['evY'])  # (Ne,Ne*Nk*Np)
        self.source_model['euX_source'] = np.stack(euX_source)  # (Ns,Ne,Ne*Nk*Np)
        self.source_model['evY_source'] = np.stack(evY_source)  # (Ns,Ne,Ne*Nk*Np)

    def transfer_learning(self):
        """Transfer learning process."""
        # basic information
        n_events = self.target_train_info['n_events']  # Ne
        n_chans = self.target_train_info['n_chans']  # Nc
        n_train = self.target_train_info['n_train']  # [Nt1,Nt2,...]

        # obtain transfer model (partial)
        self.trans_model = {}
        eu_trans, ev_trans = [], []  # List[ndarray]: Ns*(Ne,Ne*Nk,Nc)
        for nsub in range(self.n_subjects):
            euX, evY = self.euX_source[nsub], self.evY_source[nsub]  # (Ne,Ne*Nk,Np)

            # LST alignment
            eu_trans.append(np.zeros((n_events, euX.shape[1], n_chans)))  # (Ne,Ne*Nk,Nc)
            ev_trans.append(np.zeros((n_events, evY.shape[1], n_chans)))  # (Ne,Ne*Nk,Nc)
            for ne, et in enumerate(self.event_type):
                X_temp = self.X_train[self.y_train == et]  # (Nt,Nc,Np)
                train_trials = n_train[ne]
                for tt in range(train_trials):  # w = min ||b - A w||
                    trans_uX_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=euX[ne].T)
                    trans_vY_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=evY[ne].T)
                    eu_trans[nsub][ne] += trans_uX_temp.T
                    ev_trans[nsub][ne] += trans_vY_temp.T
                eu_trans[nsub][ne] /= train_trials
                ev_trans[nsub][ne] /= train_trials
        self.trans_model['eu_trans'] = np.stack(eu_trans)  # (Ns,Ne,Ne*Nk,Nc)
        self.trans_model['ev_trans'] = np.stack(ev_trans)  # (Ns,Ne,Ne*Nk,Nc)

    def dist_calc(self):
        """Calculate the spatial distances between source and target domain."""
        # load in models & basic information
        n_events = self.target_train_info['n_events']  # Ne
        n_train = self.target_train_info['n_train']  # [Nt1,Nt2,...]
        eu_trans = self.trans_model['eu_trans']  # (Ns,Ne,Ne*Nk,Nc)
        ev_trans = self.trans_model['ev_trans']  # (Ns,Ne,Ne*Nk,Nc)
        euX_source = self.source_model['euX_source']  # (Ns,Ne,Ne*Nk*Np)
        evY_source = self.source_model['evY_source']  # (Ns,Ne,Ne*Nk*Np)

        # calculate distances
        dist_euX = np.zeros((self.n_subjects, n_events))  # (Ns,Ne)
        dist_evY = np.zeros_like(self.dist_euX)
        for ne, et in enumerate(self.event_type):
            X_temp = self.X_train[self.y_train == et]  # (Nt,Nc,Np)
            train_trials = n_train[ne]
            for tt in range(train_trials):
                X_trans_x = eu_trans[:, ne, ...] @ X_temp[tt]  # (Ns,Ne*Nk,Np)
                X_trans_y = ev_trans[:, ne, ...] @ X_temp[tt]  # (Ns,Ne*Nk,Np)
                X_trans_x = np.reshape(X_trans_x, (self.n_subjects, -1), 'C')
                X_trans_y = np.reshape(X_trans_y, (self.n_subjects, -1), 'C')
                dist_euX[:, ne] += utils.fast_corr_2d(X=X_trans_x, Y=euX_source[:, ne, :])
                dist_evY[:, ne] += utils.fast_corr_2d(X=X_trans_y, Y=evY_source[:, ne, :])
        self.trans_model['dist_euX'] = dist_euX
        self.trans_model['dist_evY'] = dist_evY

    def weight_calc(self):
        """Optimize the transfer weights."""
        dist_euX = self.trans_model['dist_euX']  # (Ns,Ne)
        dist_evY = self.trans_model['dist_evY']  # (Ns,Ne)
        self.trans_model['weight_euX'] = dist_euX / np.sum(dist_euX, axis=0, keepdims=True)
        self.trans_model['weight_evY'] = dist_evY / np.sum(dist_evY, axis=0, keepdims=True)

    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        self.target_training_model = tnsre_20233250953_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            train_info=self.target_train_info,
            n_components=self.n_components
        )

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            X_source: List[ndarray],
            y_source: List[ndarray],
            sine_template: ndarray):
        """Train model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template

        # basic information of source domain
        self.n_subjects = len(self.X_source)
        self.source_train_info = []
        for nsub in range(self.n_subjects):
            source_event_type = np.unique(self.y_source[nsub])
            self.source_train_info.append({
                'event_type': source_event_type,
                'n_events': source_event_type.shape[0],
                'n_train': np.array([np.sum(self.y_source[nsub] == et)
                                     for et in source_event_type]),
                'n_chans': self.X_source[nsub].shape[-2],
                'n_points': self.X_source[nsub].shape[-1],
            })

        # basic information of target domain
        target_event_type = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        self.target_train_info = {
            'event_type': target_event_type,
            'n_events': target_event_type.shape[0],
            'n_train': np.array([np.sum(self.y_train == et)
                                 for et in target_event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
        }

        # main process
        self.intra_source_training()
        self.transfer_learning()
        self.dist_calc()
        self.weight_calc()
        self.intra_target_training()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return tnsre_20233250953_feature(
            X_test=X_test,
            source_model=self.source_model,
            trans_model=self.trans_model,
            target_model=self.target_training_model
        )


class FB_TNSRE_20233250953(BasicFBTransfer):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=TNSRE_20233250953(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 3. subject transfer based CCA, stCCA
def stcca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        train_info: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
    """Intra-domain modeling process of stCCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters.

    Returns: Dict[str, ndarray]
        Cxx (ndarray): (Ne,Nc,Nc). Covariance of averaged EEG template.
        Cxy (ndarray): (Ne,Nc,2*Nh). Covariance between EEG and sinusoidal template.
        Cyy (ndarray): (Ne,2*Nh,2*Nh). Covariance of sinusoidal template.
        u (ndarray): (Nk,Nc). Spatial filters (EEG signal).
        v (ndarray): (Nk,2*Nh). Spatial filters (sinusoidal signal).
        uX (ndarray): (Ne,Nk*Np). Reshaped stCCA templates (EEG signal).
        vY (ndarray): (Ne,Nk*Np). Reshaped stCCA templates (sinusoidal signal).
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    n_dims = sine_template.shape[1]  # 2*Nh

    # covariance matrices: Cxx, Cyy, Cxy
    Cxx = np.tile(np.eye(n_chans), (n_events, 1, 1))  # (Ne,Nc,Nc)
    Cyy = np.tile(np.eye(n_dims), (n_events, 1, 1))  # (Ne,2*Nh,2*Nh)
    Cxy = np.zeros((n_events, n_chans, n_dims))  # (Ne,Nc,2*Nh)
    X_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        X_mean[ne] = np.mean(X_train[y_train == et], axis=0)  # (Nc,Np)
        Cxx[ne] = X_mean[ne] @ X_mean[ne].T
        Cyy[ne] = sine_template[ne] @ sine_template[ne].T
        Cxy[ne] = X_mean[ne] @ sine_template[ne].T
    Cxx = np.sum(Cxx, axis=0)  # (Nc,Nc)
    Cyy = np.sum(Cyy, axis=0)  # (2*Nh,2*Nh)
    Cxy = np.sum(Cxy, axis=0)  # (Nc,2*Nh)

    # GEPs | train spatial filters & templates
    u = utils.solve_gep(
        A=Cxy @ sLA.solve(Cyy, Cxy.T),
        B=Cxx,
        n_components=n_components
    )  # (Nk,Nc)
    v = utils.solve_gep(
        A=Cxy.T @ sLA.solve(Cxx, Cxy),
        B=Cyy,
        n_components=n_components
    )  # (Nk,2*Nh)
    uX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    vY = np.zeros_like(uX)  # (Ne,Nk,Np)
    for ne in range(n_events):
        uX[ne] = u @ X_mean[ne]
        vY[ne] = v @ sine_template[ne]
    uX = utils.fast_stan_3d(uX)
    uX = np.reshape(uX, (n_events, -1), 'C')  # (Ne,Nk*Np)
    vY = utils.fast_stan_3d(vY)
    vY = np.reshape(vY, (n_events, -1), 'C')  # (Ne,Nk*Np)

    # stCCA source model
    training_model = {
        'Cxx': Cxx, 'Cyy': Cyy, 'Cxy': Cxy,
        'u': u, 'v': v, 'uX': uX, 'vY': vY
    }
    return training_model


def stcca_feature(
        X_test: ndarray,
        trans_model: Dict[str, ndarray],
        target_model: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """The pattern matching process of stCCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        trans_model (dict): {'uX_trans': ndarray (Ne,Nk*Np)}
            See details in STCCA.tranfer_learning()
        target_model (dict): {'u': ndarray (Nk,Nc),
                              'vY': ndarray (Ne,Nk*Np)}
            See details in STCCA.intra_target_training()

    Returns: Dict[str, ndarray]
        rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models & basic information
    uX_trans = trans_model['uX_trans']  # (Ne,Nk*Np)
    u_target = target_model['u']  # (Nk,Nc)
    vY_target = target_model['vY']  # (Ne,Nk*Np)
    n_events = vY_target.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    # n_points = X_test.shape[-1]  # Np, unnecessary

    # 2-part discriminant coefficients
    rho_temp = np.zeros((n_test, n_events, 2))  # (Ne*Nte,Ne,2)
    for nte in range(n_test):
        X_temp = utils.fast_stan_2d(u_target @ X_test[nte])  # (Nk,Np)
        X_temp = np.tile(np.reshape(X_temp, -1, 'C'), (n_events, 1))  # (Ne,Nk*Np)

        # rho 1: target-domain pattern matching
        rho_temp[nte, :, 0] = utils.fast_corr_2d(X=X_temp, Y=vY_target)

        # rho 2: transferred pattern matching
        rho_temp[nte, :, 1] = utils.fast_corr_2d(X=X_temp, Y=uX_trans)
    # rho_temp /= n_points  # real Pearson correlation coefficient in scale
    features = {
        'rho_temp': rho_temp,
        'rho': utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1]
        ])
    }
    return features


class STCCA(BasicTransfer):
    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        self.source_intra_model, self.source_model = [], {}
        source_uX = []  # List[ndarray]: Ns*(Ne,Nk*Np)
        for nsub in range(self.n_subjects):
            intra_model = stcca_kernel(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                train_info=self.source_train_info[nsub],
                n_components=self.n_components
            )
            self.source_intra_model.append(intra_model)
            source_uX.append(intra_model['uX'])  # (Ne,Nk*Np)
        self.source_model['source_uX'] = np.stack(source_uX)  # (Ns,Ne,Nk*Np)

    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        self.target_training_model = stcca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            train_info=self.target_train_info,
            n_components=self.n_components
        )

    def weight_calc(self):
        """Optimize the transfer weights."""
        # basic information
        target_event_type = self.target_train_info['event_type'].tolist()
        source_event_type = self.source_train_info[0]['event_type'].tolist()  # Ne (common)
        event_indices = [source_event_type.index(tet) for tet in target_event_type]

        # solve LST problem: w = min||b - A w||
        self.buX = np.reshape(
            a=self.target_training_model['uX'],
            newshape=-1,
            order='C'
        )  # (Ne',Nk*Np) -> (Ne'*Nk*Np) | Ne' <= Ne
        self.AuX = np.transpose(np.reshape(
            a=self.source_model['source_uX'][:, event_indices, :],
            newshape=(self.n_subjects, -1),
            order='C'
        ))  # (Ns,Ne',Nk*Np) -> (Ns,Ne'*Nk*Np) -> (Ne'*Nk*Np,Ns)
        self.weight_uX, _, _, _ = sLA.lstsq(a=self.AuX, b=self.buX)  # (Ns,)

    def transfer_learning(self):
        """Transfer learning process."""
        self.trans_model = {}
        uX_trans = np.einsum('s,sep->ep', self.weight_uX, self.source_model['uX_source'])
        self.trans_model['uX_trans'] = utils.fast_stan_2d(uX_trans)  # (Ne,Nk*Np)

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            X_source: List[ndarray],
            y_source: List[ndarray],
            sine_template: ndarray):
        """Train stCCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template

        # basic information of source domain
        self.n_subjects = len(self.X_source)
        self.source_train_info = []
        for nsub in range(self.n_subjects):
            source_event_type = np.unique(self.y_source[nsub])
            self.source_train_info.append({
                'event_type': source_event_type,
                'n_events': source_event_type.shape[0],
                'n_train': np.array([np.sum(self.y_source[nsub] == et)
                                     for et in source_event_type]),
                'n_chans': self.X_source[nsub].shape[-2],
                'n_points': self.X_source[nsub].shape[-1],
            })

        # basic information of target domain
        target_event_type = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        self.target_train_info = {
            'event_type': target_event_type,
            'n_events': target_event_type.shape[0],
            'n_train': np.array([np.sum(self.y_train == et)
                                 for et in target_event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
        }

        # main process
        self.intra_source_training()
        self.intra_target_training()
        self.weight_calc()
        self.transfer_learning()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return stcca_feature(
            X_test=X_test,
            trans_model=self.trans_model,
            target_model=self.target_training_model
        )


class FB_STCCA(BasicFBTransfer):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=STCCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 4. transfer learning CCA, tlCCA
def tlcca_init_model(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: Optional[ndarray],
        n_components: int = 1,
        method: Optional[str] = None,
        w: Optional[ndarray] = None) -> Dict[str, ndarray]:
    """Create initial filter(s) w and template(s) wX for tlCCA.

    Args:
        X_train (ndarray): (Ne(s)*Nt,Nc,Np). Source training dataset.
            Nt>=2, Ne (source) < Ne (full).
        y_train (ndarray): (Ne(s)*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne(s),2*Nh,Np). Sinusoidal templates.
        n_components (int): Number of eigenvectors picked as filters.
        method (str, optional): Support 'msCCA', 'TRCA', or 'DSP' for now.
            If None, parameter 'w' should be given.
        w (ndarray, optional): (Nk,Nc) or (Ne(s),Nk,Nc).
            If None, parameter 'method' should be given.

    Returns: Dict[str, ndarray]
        w_init (ndarray): (Ne(s),Nk,Nc). Spatial filter w.
        X_mean (ndarray): (Ne(s),Nc,Np). Averaged templates of X_train.
    """
    # basic information
    event_type = np.unique(y_train)
    n_events = event_type.shape[0]
    train_info = {
        'event_type': event_type,
        'n_events': n_events,
        'n_train': np.array([np.sum(y_train == et) for et in event_type]),
        'n_chans': X_train.shape[-2],
        'n_points': X_train.shape[-1],
    }
    X_mean = np.zeros((
        train_info['n_events'],
        train_info['n_chans'],
        train_info['n_points']
    ))
    for ne, et in enumerate(event_type):
        X_mean[ne] = np.mean(X_train[y_train == et], axis=0)

    # train initial model
    if method == 'msCCA':
        model = cca.mscca_kernel(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            train_info=train_info,
            n_components=n_components
        )
        w_init = np.tile(A=model['w'], reps=(n_events, 1, 1))  # (Ne,Nk,Nc)
    elif method == 'TRCA':
        train_info['standard'] = True
        train_info['ensemble'] = False
        model = trca.trca_kernel(
            X_train=X_train,
            y_train=y_train,
            train_info=train_info,
            n_components=n_components
        )
        w_init = model['w']  # (Ne,Nk,Nc)
    elif method == 'DSP':
        model = dsp.dsp_kernel(
            X_train=X_train,
            y_train=y_train,
            train_info=train_info,
            n_components=n_components
        )
        w_init = np.tile(A=model['w'], reps=(n_events, 1, 1))  # (Ne,Nk,Nc)
    elif w is not None:
        if w.ndim == 2:  # common filter, (Nk,Nc)
            w_init = np.tile(A=w, reps=(n_events, 1, 1))  # (Ne,Nk,Nc)
        elif w.ndim == 3:  # (Ne,Nk,Nc)
            w_init = w
    else:
        raise Exception("Unknown initial model! Check the input 'initial_model'!")
    return {'w_init': w_init, 'X_mean': X_mean}


def tlcca_conv_matrix(
        freq: Union[int, float],
        phase: Union[int, float],
        n_points: int,
        srate: Union[int, float] = 1000,
        rrate: int = 60,
        len_scale: float = 0.99,
        amp_scale: float = 0.8,
        concat_method: str = 'dynamic') -> Tuple[ndarray]:
    """Create convolution matrix H (H_correct) for tlCCA (single-event).

    Args:
        freq (int or float): Stimulus frequency.
        phase (int or float): Stimulus phase (coefficients). 0-2 (pi).
        n_points (int): Data length.
        srate (int or float): Sampling rate. Defaults to 1000 Hz.
        rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
        len_scale (float): The multiplying power when calculating the length of data.
            Defaults to 0.99.
        amp_scale (float): The multiplying power when calculating the amplitudes of data.
            Defaults to 0.8.
        concat_method (str): 'dynamic' or 'static'.
            'static': Concatenated data is starting from 1 s.
            'dynamic': Concatenated data is starting from 1 period.

    Returns:
        H (ndarray): (response_length,Np). Convolution matrix.
        H_correct (ndarray): (response_length,Np). Corrected convolution matrix.
    """
    periodic_impulse = utils.extract_periodic_impulse(
        freq=freq,
        phase=phase,
        n_points=n_points,
        srate=srate,
        rrate=rrate
    )
    response_length = int(np.ceil(srate * len_scale / freq))
    H = utils.create_conv_matrix(
        periodic_impulse=periodic_impulse,
        response_length=response_length
    )  # (response_length, Np)
    H_correct = utils.correct_conv_matrix(
        H=H,
        freq=freq,
        srate=srate,
        amp_scale=amp_scale,
        concat_method=concat_method
    )  # (response_length, Np)
    return H, H_correct


def tlcca_als_optimize(
        w_init: ndarray,
        X_mean: ndarray,
        H: ndarray,
        freq: Union[int, float],
        n_points: int,
        srate: Union[int, float] = 1000,
        iter_limit: int = 200,
        err_th: float = 0.00001):
    """Alternative least-square (ALS) optimization for impulse response r
        and spatial filter w (single-event or common).

    Args:
        w_init (ndarray): (Nk,Nc). Initial spatial filter w.
        X_mean (ndarray): (Nc,Np). Averaged template.
        H (ndarray): (response_length,Np). Convolution matrix.
        freq (int or float): Stimulus frequency.
        n_points (int): Data length.
        srate (int or float): Sampling rate. Defaults to 1000 Hz.
        iter_limit (int): Number of maximum iteration times. Defaults to 200.
        err_th (float): The threshold (th) of ALS error. Stop iteration while
            ALS error is smaller than err_th. Defaults to 10^-5.

    Returns:
        w (ndarray): (Nk,Nc). Optimized spatial filter.
        r (ndarray): (Nk, response length). Optimized impulse response.
        wX (ndarray): (Nk,Np). w @ X_mean.
        rH (ndarray): (Nk,Np). r @ H.
        n_iter (int): Number of iteration.
    """
    # initial impulse response (r) & template (rH)
    r_init = np.tile(
        A=utils.sin_wave(
            freq=freq,
            n_points=n_points,
            phase=0,
            srate=srate
        ),
        reps=(w_init.shape[0], 1)
    )  # (Nk, response_length)
    r_init = np.diag(1 / np.sqrt(np.sum(r_init**2, axis=1))) @ r_init  # ||r(i,:)|| = 1
    rH_init = r_init @ H  # (Nk,Np)

    # initial spatial filter (w) & template (wX)
    wX_init = w_init @ X_mean  # (Nk,Np)

    # iteration initialization
    err_init = np.sum((wX_init - rH_init)**2)
    err_old, err_change = err_init, 0
    log_w, log_r, log_err = [w_init], [r_init], [err_init]

    # start iteration
    n_iter = 1
    w_old = w_init
    continue_training = True
    while continue_training:
        # calculate new impulse response (r_new)
        wX_temp = w_old @ X_mean  # (Nk,Np)
        r_new, _, _, _ = sLA.lstsq(a=H.T, b=wX_temp.T)  # (response length, Nk)
        r_new = r_new.T  # (Nk, response length)
        r_new = np.diag(1 / np.sqrt(np.sum(r_new**2, axis=0))) @ r_new.T  # ||r(i,:)|| = 1

        # calculate new spatial filter (w_new)
        rH_temp = r_new @ H  # (Nk,Np)
        w_new, _, _, _ = sLA.lstsq(a=X_mean.T, b=rH_temp.T)  # (Nc,Nk)
        w_new = w_new.T  # (Nk,Nc)
        w_new = np.diag(1 / np.sqrt(np.sum(w_new**2, axis=0))) @ w_new.T  # ||w(i,:)|| = 1

        # update ALS error
        err_new = np.sum((wX_temp - rH_temp)**2)
        err_change = err_old - err_new
        log_err.append(err_new)

        # termination criteria
        n_iter += 1
        continue_training = (n_iter < iter_limit) * (abs(err_change) > err_th)

        # update w & r
        log_w.append(w_new)
        log_r.append(r_new)
        w_old = w_new
        err_old = err_new
    return {
        'w': w_new, 'r': r_new,
        'wX': w_new @ X_mean, 'rH': r_new @ H, 'n_iter': n_iter
    }


def tlcca_feature(
        X_test: ndarray,
        sine_template: ndarray,
        trans_model: Dict[str, ndarray],
        n_components: int = 1) -> Dict[str, ndarray]:
    """The pattern matching process of tlCCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        trans_model (dict): {'u_trans': ndarray (Nk,Nc),
                             'v_trans': ndarray (Nk,2Nh),
                             'r_trans': List[ndarray] Ne*(Nk, response length),
                             'w_trans': ndarray (Ne,Nk,Nc),
                             'vY_trans': ndarray (Ne,Nk*Np),
                             'rH_trans': ndarray (Ne,Nk*Np)}
            See details in TLCCA.tranfer_learning()
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Defaults to 1 in tlCCA algorithm.

    Returns: Dict[str, ndarray]
        rho_temp (ndarray): (Ne*Nte,Ne,3). 3-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models & basic information
    u_trans = trans_model['u_trans']  # (Nk,Nc)
    w_trans = trans_model['w_trans']  # (Ne,Nk,Nc)
    vY_trans = trans_model['vY_trans']  # (Ne,Nk*Np)
    rH_trans = trans_model['rH_trans']  # (Ne,Nk*Np)
    n_events = w_trans.shape[0]  # Ne
    n_points = X_test.shape[-1]  # Np
    n_test = X_test.shape[0]  # Ne*Nte

    # 3-part discriminant coefficients
    rho_temp = np.zeros((n_test, n_events, 3))
    for nte in range(n_test):
        uX_temp = utils.fast_stan_2d(u_trans @ X_test[nte])  # (Nk,Np)
        uX_temp = np.tile(np.reshape(uX_temp, -1, 'C'), (n_events, 1))  # (Ne,Nk*Np)
        wX_temp = utils.fast_stan_3d(w_trans @ X_test[nte])  # (Ne,Nk,Np)
        # wX_temp = np.reshape(wX_temp, (n_events, -1), 'C')  # (Ne,Nk*Np)

        # rho 1: corr(uX, vY)
        rho_temp[nte, :, 0] = utils.fast_corr_2d(X=uX_temp, Y=vY_trans) / n_points

        # rho 2: corr(wX, rH)
        rho_temp[nte, :, 1] = utils.fast_corr_2d(
            X=np.reshape(wX_temp, (n_events, -1), 'C'),
            Y=rH_trans
        ) / n_points

        # rho 3: CCA(wX, Y) | SLOW!
        for nem in range(n_events):
            cca_model = cca.cca_kernel(
                X=wX_temp[nem],
                Y=sine_template[nem],
                n_components=n_components
            )
            rho_temp[nte, nem, 2] = cca_model['coef']
    features = {
        'rho_temp': rho_temp,
        'rho': utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1],
            rho_temp[..., 2]
        ])
    }
    return features


class TLCCA(BasicTransfer):
    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        # initialization
        self.source_model = {}

        # obtain initial filter(s) w & template(s) wX
        init_model = tlcca_init_model(
            X_train=self.X_source,
            y_train=self.y_source,
            sine_template=self.source_sine_template,
            n_components=self.n_components,
            method=self.source_train_info['init_model'],
            w=self.source_train_info['w_init']
        )
        self.source_model['w_init'] = init_model['w_init']  # (Nk,Nc) or (Ne(s),Nk,Nc)

        # alternating least square (ALS) optimization: {r, w} = argmin||wX - rH||
        w, r, wX, rH = [], [], [], []
        for ne, et in enumerate(self.source_train_info['event_type']):
            als_model = tlcca_als_optimize(
                w_init=self.source_model['w_init'][ne],
                X_mean=init_model['X_mean'][ne],
                H=self.source_H_correct[ne],
                freq=self.stim_info[str(et)][0],
                srate=self.srate,
                iter_limit=self.iter_limit,
                err_th=self.err_th
            )
            w.append(als_model['w'])  # (Nk,Nc)
            r.append(als_model['r'])  # (Nk, response length)
            wX.append(als_model['wX'])  # (Nk,Np)
            rH.append(als_model['rH'])  # (Nk,Np)
        self.source_model['w'] = np.stack(w, axis=0)  # (Ne(s),Nk,Nc)
        self.source_model['r'] = r  # Ne(s)*(Nk, response length)
        self.source_model['wX'] = np.stack(wX, axis=0)  # (Ne(s),Nk,Np)
        self.source_model['rH'] = np.stack(rH, axis=0)  # (Ne(s),Nk,Np)

    def transfer_learning(self):
        """Transfer learning between exist & missing events."""
        # spatial filters u, v from ms-eCCA process (common)
        source_event_type = self.transfer_info['source_event_type']
        source_n_events = source_event_type.shape[0]
        self.source_train_info['events_group'] = [
            np.arange(source_n_events) for et in source_event_type
        ]  # same filters u, v across all events
        common_model = cca.msecca_kernel(
            X_train=self.X_source,
            y_train=self.y_source,
            sine_template=self.source_sine_template,
            train_info=self.source_train_info,
            n_components=self.n_components
        )
        u_trans, v_trans = common_model['u'][0], common_model['v'][0]  # (Nk,Nc), (Nk,2Nh)

        # shifted r, w of unknown events
        target_event_type = self.transfer_info['target_event_type']
        r_trans, w_trans = [], []
        for tet in target_event_type:  # tet is a label (int)
            for tp in self.transfer_pair:
                if tet == tp[0]:  # find the transfer pair
                    source_idx = list(source_event_type).index(tp[1])
                    r_trans.append(self.source_model['r'][source_idx])
                    w_trans.append(self.source_model['w'][source_idx])
                    break
        w_trans = np.stack(w_trans, axis=0)  # (Ne,Nk,Nc)

        # transferred templates rH & vY
        n_events = target_event_type.shape[0]
        n_points = self.X_source.shape[-1]
        rH_trans = np.zeros((n_events, self.n_components, n_points))  # (Ne,Nk,Np)
        vY_trans = np.zeros_like(rH_trans)
        for ne in range(n_events):
            rH_trans[ne] = r_trans[ne] @ self.H[ne]
            vY_trans[ne] = v_trans @ self.sine_template[ne]
        rH_trans = utils.fast_stan_3d(rH_trans)
        rH_trans = np.reshape(rH_trans, (n_events, -1), order='C')  # (Ne,Nk*Np)
        vY_trans = utils.fast_stan_3d(vY_trans)
        vY_trans = np.reshape(vY_trans, (n_events, -1), order='C')  # (Ne,Nk*Np)

        # transfer model
        self.trans_model = {
            'u_trans': u_trans, 'v_trans': v_trans,
            'r_trans': r_trans, 'w_trans': w_trans,
            'vY_trans': vY_trans, 'rH_trans': rH_trans
        }

    def fit(
            self,
            X_source: ndarray,
            y_source: ndarray,
            stim_info: Dict[str, Tuple[Union[float, int], Union[float, int]]],
            n_harmonics: int = 1,
            method: Optional[str] = 'msCCA',
            w_init: Optional[ndarray] = None,
            transfer_pair: Optional[List[Tuple[int, int]]] = None,
            srate: Union[float, int] = 1000,
            rrate: int = 60,
            len_scale: float = 0.99,
            amp_scale: float = 0.8,
            concat_method: str = 'dynamic',
            iter_limit: int = 200,
            err_th: float = 0.00001):
        """Train tlCCA model.

        Args:
            X_source (ndarray): (Ne(s)*Nt,Nc,Np). Source training dataset.
                Nt>=2, Ne (source) < Ne (full).
            y_source (ndarray): (Ne(s)*Nt,). Labels for X_source.
            stim_info (dict): {'label': (frequency, phase)}.
            n_harmonics (int): Number of harmonic components for sinusoidal templates.
                Defaults to 1.
            method (str): 'msCCA', 'TRCA' or 'DSP'. Defaults to 'msCCA'.
            w_init (ndarray): (Ne,Nk,Nc). Initial spatial filter(s). Defaults to None.
            transfer_pair (List[Tuple[int, int]], Optional): (target label, source label).
            srate (int or float): Sampling rate. Defaults to 1000 Hz.
            rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
            len_scale (float): The multiplying power when calculating the length of data.
                Defaults to 0.99.
            amp_scale (float): The multiplying power when calculating the amplitudes of data.
                Defaults to 0.8.
            concat_method (str): 'dynamic' or 'static'.
                'static': Concatenated data is starting from 1 s.
                'dynamic': Concatenated data is starting from 1 period.
            iter_limit (int): Number of maximum iteration times. Defaults to 200.
            err_th (float): The threshold (th) of ALS error. Stop iteration while
                ALS error is smaller than err_th. Defaults to 10^-5.
        """
        # load in data
        self.X_source = X_source
        self.y_source = y_source
        self.stim_info = stim_info
        self.n_harmonics = n_harmonics
        self.transfer_pair = transfer_pair
        self.srate = srate
        self.rrate = rrate
        self.len_scale = len_scale
        self.amp_scale = amp_scale
        self.concat_method = concat_method
        self.iter_limit = iter_limit
        self.err_th = err_th

        # special check: Nk must be 1
        assert self.n_components == 1, 'Only support Nk=1 for now!'

        # create sinusoidal templates for full events
        event_type = [tp[0] for tp in self.transfer_pair]
        event_type = np.array(list(set(event_type)))
        n_events = event_type.shape[0]  # Ne (total)
        n_points = self.X_source.shape[-1]  # Np
        self.sine_template = np.zeros((n_events, self.n_harmonics, n_points))
        for ne, et in enumerate(event_type):
            self.sine_template[ne] = utils.sine_template(
                freq=self.stim_info[str(et)][0],
                phase=self.stim_info[str(et)][1],
                n_points=n_points,
                n_harmonics=self.n_harmonics,
                srate=self.srate
            )

        # create convolution matrices (H) and their corrected version (H_correct)
        self.H = [[] for ne in range(n_events)]
        self.H_correct = [[] for ne in range(n_events)]
        for ne, et in enumerate(event_type):
            self.H[ne], self.H_correct[ne] = tlcca_conv_matrix(
                freq=self.stim_info[str(et)][0],
                phase=self.stim_info[str(et)][1],
                n_points=n_points,
                srate=self.srate,
                rrate=self.rrate,
                len_scale=self.len_scale,
                amp_scale=self.amp_scale,
                concat_method=self.concat_method
            )  # (response length, Np)

        # basic information of source domain
        source_event_type = np.unique(self.y_source)
        self.source_train_info = {
            'event_type': source_event_type,
            'n_events': source_event_type.shape[0],
            'n_train': np.array([np.sum(self.y_source == et) for et in source_event_type]),
            'n_chans': self.X_source.shape[-2],
            'n_points': self.X_source.shape[-1],
            'init_model': method,
            'w_init': w_init
        }
        self.source_sine_template, self.source_H, self.source_H_correct = [], [], []
        for ne, et in enumerate(event_type):
            if et in list(source_event_type):
                self.source_sine_template.append(self.sine_template[ne])
                self.source_H.append(self.H[ne])
                self.source_H_correct.append(self.H_correct[ne])
        self.source_sine_template = np.stack(self.source_sine_template, axis=0)

        # main process
        self.intra_source_training()
        self.transfer_learning()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho_temp (ndarray): (Ne*Nte,Ne,3). 3-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return tlcca_feature(
            X_test=X_test,
            sine_template=self.sine_template,
            trans_model=self.trans_model,
            n_components=self.n_components
        )


class FB_TLCCA(BasicFBTransfer):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=TLCCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 5. small data least-squares transformation, sd-LST
def sdlst_align_subject(
        X_source: ndarray,
        y_source: ndarray,
        avg_template_target: ndarray,
        target_event_type: ndarray) -> Tuple[ndarray]:
    """One-step LST aligning in sd-LST (subject to subject).

    Args:
        X_source (ndarray): (Ne*Nt,Nc(s),Np). Training dataset of source subject.
        y_source (ndarray): (Ne*Nt,). Labels for X_source.
        avg_template_target (ndarray): (Ne(t),Nc(t),Np).
            Averaged template of target training dataset. Nc(t) may not be equal to Nc(s).
        target_event_type (ndarray): (Ne(t),). Event types of avg_template_target.

    Returns: Tuple[ndarray]
        projection_matrix (ndarray): (Nc(t),Nc(s)). Transfer matrix 1.
        aligned_source (ndarray): (Ne*Nt,Nc(t),Np). Transferred X_source.
    """
    # basic information
    n_events_limited = len(target_event_type)  # Ne(t)
    n_chans_target = avg_template_target.shape[-2]  # Nc(t)
    n_chans_source = X_source.shape[-2]  # Nc(s)
    n_points = avg_template_target.shape[-1]  # Np

    # extract limited event information from source dataset
    X_source_limited_avg = np.zeros((n_events_limited, n_chans_source, n_points))
    for ntet, tet in enumerate(target_event_type):
        temp = X_source[y_source == tet]
        if temp.ndim == 2:  # (Nc,Np), Nt=1
            X_source_limited_avg[ntet] = temp
        elif temp.ndim > 2:  # (Nt,Nc,Np)
            X_source_limited_avg[ntet] = np.mean(temp, axis=0)

    # 1st-step LST aligning
    part_1 = np.zeros((n_chans_target, n_chans_source))  # (Nc(t),Nc(s))
    part_2 = np.zeros((n_chans_source, n_chans_source))  # (Nc(s),Nc(s))
    for nel in range(n_events_limited):
        part_1 += avg_template_target[nel] @ X_source_limited_avg[nel].T
        part_2 += X_source_limited_avg[nel] @ X_source_limited_avg[nel].T
    projection_matrix = part_1 @ sLA.inv(part_2)

    # apply projection onto each trial of source dataset (full events)
    # aligned_source = np.einsum('tcp,hc->thp', X_source,projection_matrix)
    aligned_source = np.zeros((X_source.shape[0], n_chans_target, X_source.shape[-1]))
    for tt in range(X_source.shape[0]):
        aligned_source[tt] = projection_matrix @ X_source[tt]
    return projection_matrix, aligned_source


def sdlst_source_compute(
        X_source: List[ndarray],
        y_source: List[ndarray],
        X_target: ndarray,
        y_target: ndarray) -> Tuple:
    """Two-step LST algining in sd-LST. Obtain augmented training dataset.

    Args:
        X_source (List[ndarray]): Ns*(Ne*Nt,Nc(s),Np). Source-domain dataset. Nt>=1.
        y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
        X_target (ndarray): (Ne(t)*Nt,Nc(t),Np). Target-domain dataset. Nt>=1
        y_target (ndarray): (Ne(t)*Nt,). Labels for X_target.

    Returns: source_model (dict)
        LST-1 (List[ndarray]): Ns*(Nc(t),Nc(s)). One-step LST projection matrices.
        LST-2 (ndarray): (Nc(t),Nc(t)). Two-step LST projection matrix.
        X-LST-1 (ndarray): (Ns*Ne*Nt,Nc(t),Np). Source dataset with one-step LST aligned.
        X-LST-2 (ndarray): (Ns*Ne*Nt,Nc(t),Np). Source dataset with two-step LST aligned.
        y (ndarray): (Ns*Ne*Nt,). Labels for X-LST-1 (and X-LST-2)
    """
    # basic information
    n_subjects = len(X_source)  # Ns
    target_event_type = np.unique(y_target)
    n_events_limited = len(target_event_type)  # Ne(t)
    n_chans = X_target.shape[-2]  # Nc
    n_points = X_target.shape[-1]  # Np
    
    # obtain averaged template of target training dataset
    avg_template_target = np.zeros((n_events_limited, n_chans, n_points))  # (Ne(t),Nc,Np)
    for ntet, tet in enumerate(target_event_type):
        temp = X_target[y_target == tet]
        if temp.ndim == 2:  # (Nc,Np), Nt=1
            avg_template_target[ntet] = temp
        elif temp.ndim > 2:  # (Nt,Nc,Np)
            avg_template_target[ntet] = np.mean(temp, axis=0)

    # obtain once-aligned signal
    projection_matrices = []
    source_trials = []
    X_lst_1 = np.empty((1, n_chans, n_points))
    y_lst_1 = np.empty((1))
    for nsub in range(n_subjects):
        pm, ast = sdlst_align_subject(
            X_source=X_source[nsub],
            y_source=y_source[nsub],
            avg_template_target=avg_template_target,
            target_event_type=target_event_type
        )
        source_trials.append(ast.shape[0])
        projection_matrices.append(pm)
        X_lst_1 = np.concatenate((X_lst_1, ast), axis=0)
        y_lst_1 = np.concatenate((y_lst_1, y_source[nsub]))
    X_lst_1 = np.delete(X_lst_1, 0, axis=0)
    y_lst_1 = np.delete(y_lst_1, 0, axis=0)

    # apply LST projection again
    X_lst_1_limited_avg = np.zeros((n_events_limited, n_chans, n_points))
    part_1 = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    part_2 = np.zeros_like(part_1)  # (Nc,Nc)
    for ntet, tet in enumerate(target_event_type):
        temp = X_lst_1[y_lst_1 == tet]
        if temp.ndim == 2:  # (Nc,Np), Nt=1
            X_lst_1_limited_avg[ntet] = temp
        elif temp.ndim > 2:  # (Nt,Nc,Np)
            X_lst_1_limited_avg[ntet] = np.mean(temp, axis=0)
        part_1 += avg_template_target[ntet] @ X_lst_1_limited_avg[ntet].T
        part_2 += X_lst_1_limited_avg[ntet] @ X_lst_1_limited_avg[ntet].T
    projection_matrix = part_1 @ sLA.inv(part_2)  # (Nc,Nc)

    # obtrain twice-aligned signal
    # X_final = np.einsum('tcp,hc->thp', X_lst_1,projection_matrix)
    X_final = np.zeros_like(X_lst_1)
    for tt in range(X_final.shape[0]):
        X_final[tt] = projection_matrix @ X_lst_1[tt]

    # source model
    source_model = {
        'LST-1': projection_matrices,
        'LST-2': projection_matrix,
        'X-LST-1': X_lst_1,
        'X-LST-2': X_final,
        'y': y_lst_1,
        'avg_template_target': avg_template_target,
        'source_trials': source_trials
    }
    return source_model


class SDLST(BasicTransfer):
    def intra_source_training(self):
        pass

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            X_source: List[ndarray],
            y_source: List[ndarray],
            sine_template: ndarray):
        """Train sd-kLST model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        """

    def nfit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            X_source: List[ndarray],
            y_source: List[ndarray],
            sine_template: Optional[ndarray] = None,
            coef_idx: Optional[List] = [1, 2, 4]):
        """Train sd-LST model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source-domain dataset. Nt>=1.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
            coef_idx (List[int]): Details in cca.ecca_compute()
        """
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template
        self.event_type = np.unique(y_source[0])
        self.n_events = len(self.event_type)
        self.coef_idx = coef_idx
        source_model = sdlst_source_compute(
            X_source=self.X_source,
            y_source=self.y_source,
            X_target=X_train,
            y_target=y_train
        )
        self.X_train, self.y_train = source_model['X-LST-2'], source_model['y']
        self.lst_1, self.lst_2 = source_model['LST-1'], source_model['LST-2']
        self.avg_template = source_model['avg_template_target']

        self.trca_model = trca.TRCA().fit(
            X_train=self.X_train,
            y_train=self.y_train
        )
        self.ecca_model = cca.ECCA().fit(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            coef_idx=self.coef_idx
        )
        return self

    def predict(self, X_test: ndarray) -> Tuple[ndarray]:
        """Using sd-LST to predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[0]

        # pattern matching
        self.rou = np.zeros((n_test, self.n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for nem in range(self.n_events):
                rou_etrca = utils.pearson_corr(
                    X=self.trca_model.w_concat @ X_test[nte],
                    Y=self.trca_model.ewX[nem]
                )
                rou_ecca = cca.ecca_compute(
                    avg_template=self.avg_template[nem],
                    sine_template=self.sine_template[nem],
                    X_test=X_test[nte],
                    coef_idx=self.coef_idx,
                )['rou']
                self.rou[nte, nem] = utils.combine_feature([
                    rou_etrca,
                    rou_ecca
                ])
                self.rou[nte, nem] = rou_etrca
            self.y_predict[nte] = self.event_type[np.argmax(self.rou[nte, :])]
        return self.rou, self.y_predict


# %% 6. cross-subject transfer method based on domain generalization
class TNSRE_20233305202(BasicTransfer):
    pass


class FB_TNSRE_20233305202(BasicFBTransfer):
    pass


# %% 7. group TRCA | gTRCA


# %% 8. inter- and intra-subject maximal correlation | IISMC
def iismc_feature(
        X_test: ndarray,
        trans_model: Dict[str, ndarray],
        target_model: Dict[str, ndarray],
        standard: bool = True,
        ensemble: bool = True) -> Dict[str, ndarray]:
    """The pattern matching process of IISMC.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        trans_model (dict): {'trans_u': ndarray (Ns,Ne,Nk,Nc),
                             'trans_uX': ndarray (Ns,Ne,Nk*Np),
                             'trans_vY': ndarray (Ns,Ne,Nk*Np),
                             'trans_uY': ndarray (Ns,Ne,Nk*Np),
                             'trans_eu': ndarray (Ns,Ne,Ne*Nk,Nc),
                             'trans_euX': ndarray (Ns,Ne,Ne*Nk*Np),
                             'trans_evY': ndarray (Ns,Ne,Ne*Nk*Np),
                             'trans_euY': ndarray (Ns,Ne,Ne*Nk*Np)}
            See details in IISMC.tranfer_learning().
        target_model (dict): {'w': ndarray (Ne,Nk,Nc) | (filter 'v'),
                              'ew': ndarray (Ne*Nk,Nc) | (filter 'ev')
                              'wX': ndarray (Ne,Nk*Np) | (template 'vX'),
                              'ewX': ndarray (Ne,Ne*Nk*Np) | (template 'evX')}
            See details in IISMC.intra_target_training().
        standard (bool): Standard model. Defaults to True.
        ensemble (bool): Ensemble model. Defaults to True.

    Returns: Dict[str, ndarray]
        rho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        erho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features (ensemble).
        erho (ndarray): (Ne*Nte,Ne). Intergrated features (ensemble).
    """
    # load in models & basic information
    trans_u, v = trans_model['trans_u'], target_model['w']
    trans_uX, vX = trans_model['trans_uX'], target_model['wX']
    trans_vY, trans_uY = trans_model['trans_vY'], trans_model['trans_uY']

    trans_eu, ev = trans_model['trans_eu'], target_model['ew']
    trans_euX, evX = trans_model['trans_euX'], target_model['ewX']
    trans_evY, trans_euY = trans_model['trans_evY'], trans_model['trans_euY']

    n_subjects = trans_u.shape[0]  # Ns
    n_events = trans_u.shape[1]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    # n_points = X_test.shape[-1]  # Np

    # 4-part discriminant coefficients: standard & ensemble
    rho_temp = np.zeros((n_test, n_events, 4))  # (Ne*Nte,Ne,4)
    if standard:
        for nte in range(n_test):
            temp_vX = np.reshape(
                a=utils.fast_stan_2d(v @ X_test[nte]),
                newshape=(n_events, -1),
                order='C'
            )  # (Ne,Nk,Nc) @ (Nc,Np) -reshape-> (Ne,Nk*Np)
            temp_uX = np.reshape(
                a=utils.fast_stan_3d(trans_u @ X_test[nte]),
                newshape=(n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Nk,Nc) @ (Nc,Np) -reshape-> (Ns,Ne,Nk*Np)

            rho_temp[nte, :, 0] = utils.fast_corr_2d(
                X=temp_vX,
                Y=vX
            )  # vX & vX: (Ne,Nk*Np) -corr-> (Ne,)
            rho_temp[nte, :, 1] = np.mean(
                a=utils.fast_corr_3d(X=temp_uX, Y=trans_uX),
                axis=0
            )  # uX & uX: (Ns,Ne,Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
            rho_temp[nte, :, 2] = np.mean(
                a=utils.fast_corr_3d(
                    X=np.tile(A=temp_vX, reps=(n_subjects, 1, 1)),
                    Y=trans_vY
                ),
                axis=0
            )  # vX & vY: (Ne,Nk*Np) -tile-> (Ns,Ne,Nk*Np) -corr-> (Ns,Ne,) -mean-> (Ne,)
            rho_temp[nte, :, 3] = np.mean(
                a=utils.fast_corr_3d(X=temp_uX, Y=trans_uY),
                axis=0
            )  # uX & uY: (Ns,Ne,Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
        # rho_temp /= n_points  # real Pearson correlation coefficients in scale
        rho = utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1],
            rho_temp[..., 2],
            rho_temp[..., 3]
        ])

    erho_temp = np.zeros_like(rho_temp)  # (Ne*Nte,Ne,4)
    if ensemble:
        for nte in range(n_test):
            temp_evX = np.reshape(
                a=utils.fast_stan_2d(ev @ X_test[nte]),
                newshape=-1,
                order='C'
            )  # (Ne*Nk,Nc) @ (Nc,Np) -reshape-> (Ne*Nk*Np,)
            temp_euX = np.reshape(
                a=utils.fast_stan_3d(trans_eu @ X_test[nte]),
                newshape=(n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Ne*Nk,Nc) @ (Nc,Np) -reshape-> (Ns,Ne,Ne*Nk*Np)

            erho_temp[nte, :, 0] = utils.fast_corr_2d(
                X=np.tile(A=temp_evX, reps=(n_events, 1)),
                Y=evX
            )  # evX & evX: (Ne*Nk*Np,) -tile-> (Ne,Ne*Nk*Np) -corr-> (Ne,)
            erho_temp[nte, :, 1] = np.mean(
                a=utils.fast_corr_3d(X=temp_euX, Y=trans_euX),
                axis=0
            )  # euX & euX: (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
            erho_temp[nte, :, 2] = np.mean(
                a=utils.fast_corr_3d(
                    X=np.tile(A=temp_evX, reps=(n_subjects, n_events, 1)),
                    Y=trans_evY
                ),
                axis=0
            )  # evX & evY: (Ne*Nk*Np,) -tile-> (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
            erho_temp[nte, :, 3] = np.mean(
                a=utils.fast_corr_3d(X=temp_euX, Y=trans_euY),
                axis=0
            )  # euX & euY: (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
        # erho_temp /= n_points  # real Pearson correlation coefficients in scale
        erho = utils.combine_feature([
            erho_temp[..., 0],
            erho_temp[..., 1],
            erho_temp[..., 2],
            erho_temp[..., 3]
        ])
    features = {
        'rho_temp': rho_temp,
        'rho': rho,
        'erho_temp': erho_temp,
        'erho': erho
    }
    return features


class IISMC(BasicTransfer):
    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        self.target_training_model = trca.trca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            train_info=self.target_train_info,
            n_components=self.n_components
        )

    def transfer_learning(self):
        """Transfer learning process."""
        # basic information
        event_type = self.target_train_info['event_type']
        n_events = len(event_type)  # Ne
        n_chans = self.target_train_info['n_chans']  # Nc
        n_points = self.target_train_info['n_points']  # Np

        target_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
        for ne, et in enumerate(event_type):
            target_mean[ne] = np.mean(self.X_train[self.y_train == et], axis=0)  # (Nc,Np)

        # initialization
        self.trans_model = {}
        Cxx = self.target_training_model['Q']  # (Ne,Nc,Nc)
        Cxy = np.tile(np.zeros_like(Cxx), (self.n_subjects, 1, 1, 1))  # (Ns,Ne,Nc,Nc)
        Cyy = np.zeros_like(Cxy)  # (Ns,Ne,Nc,Nc)

        # standard version
        v = self.target_training_model['w']  # (Ne,Nk,Nc)
        trans_u = np.zeros((self.n_subjects, n_events, self.n_components, n_chans))
        trans_uX = np.zeros((self.n_subjects, n_events, self.n_components, n_points))
        trans_vY, trans_uY = np.zeros_like(trans_uX), np.zeros_like(trans_uX)

        # ensemble version
        ev = self.target_training_model['ew']  # (Ne*Nk,Nc)
        trans_eu = np.reshape(
            a=trans_u,
            newshape=(self.n_subjects, n_events * self.n_components, n_chans),
            order='C'
        )  # (Ns,Ne*Nk,Nc)
        trans_euX = np.zeros((self.n_subjects, n_events, trans_eu.shape[1], n_points))
        trans_evY, trans_euY = np.zeros_like(trans_euX), np.zeros_like(trans_euX)

        # obtain transfer model
        for nsub in range(self.n_subjects):
            X_source, y_source = self.X_source[nsub], self.y_source[nsub]
            n_train = self.source_train_info[nsub]['n_train']
            source_mean = np.zeros_like(target_mean)  # (Ne,Nc,Np)
            for ne, et in enumerate(event_type):
                source_trials = n_train[ne]
                assert source_trials > 1, 'The number of training samples is too small!'

                source_temp = X_source[y_source == et]  # (Nt,Nc,Np)
                source_mean[ne] = np.mean(source_temp, axis=0)  # (Nc,Np)
                Cxy[nsub, ne, ...] = target_mean[ne] @ source_mean[ne].T  # (Nc,Nc)
                for st in range(source_trials):
                    Cyy[nsub, ne, ] += source_temp[st] @ source_temp[st].T

                # obtain transferred spatial filters
                trans_u[nsub, ne, ...] = utils.solve_gep(
                    A=Cxy[nsub, ne, ...] + Cxy[nsub, ne, ...].T,
                    B=Cxx[ne] + Cyy[nsub, ne, ...],
                    n_components=self.n_components
                )  # standard | (Nk,Nc)

                # obtain standard transferred templates: (Ns,Ne,Nk,Np)
                trans_uX[nsub, ne, ...] = trans_u[nsub, ne, ...] @ target_mean[ne]
                trans_vY[nsub, ne, ...] = v[ne] @ source_mean[ne]
                trans_uY[nsub, ne, ...] = trans_u[nsub, ne, ...] @ source_mean[ne]

            # obtain ensembled transferred model: spatial filters & templates
            trans_eu[nsub] = np.reshape(
                a=trans_u[nsub],
                newshape=(n_events * self.n_components, n_chans),
                order='C'
            )  # (Ne*Nk,Nc)
            for ne in range(n_events):
                trans_euX[nsub, ne, ...] = trans_eu[nsub] @ target_mean[ne]
                trans_evY[nsub, ne, ...] = ev @ source_mean[ne]
                trans_euY[nsub, ne, ...] = trans_eu[nsub] @ source_mean[ne]

        # standardize & reshape
        self.trans_model['trans_u'] = trans_u  # filter: (Ns,Ne,Nk,Nc)
        self.trans_model['trans_eu'] = trans_eu  # filter: (Ns,Ne,Ne*Nk,Nc)
        self.trans_model['trans_uX'] = np.reshape(
            a=utils.fast_stan_4d(trans_uX),
            newshape=(self.n_subjects, n_events, self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Nk*Np)
        self.trans_model['trans_vY'] = np.reshape(
            a=utils.fast_stan_4d(trans_vY),
            newshape=(self.n_subjects, n_events, self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Nk*Np)
        self.trans_model['trans_uY'] = np.reshape(
            a=utils.fast_stan_4d(trans_uY),
            newshape=(self.n_subjects, n_events, self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Nk*Np)
        self.trans_model['trans_euX'] = np.reshape(
            a=utils.fast_stan_4d(trans_euX),
            newshape=(self.n_subjects, n_events, n_events * self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Ne*Nk*Np)
        self.trans_model['trans_evY'] = np.reshape(
            a=utils.fast_stan_4d(trans_evY),
            newshape=(self.n_subjects, n_events, n_events * self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Ne*Nk*Np)
        self.trans_model['trans_euY'] = np.reshape(
            a=utils.fast_stan_4d(trans_euY),
            newshape=(self.n_subjects, n_events, n_events * self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Ne*Nk*Np)

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            X_source: List[ndarray],
            y_source: List[ndarray]):
        """Load data and train classification models.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source

        # basic information of source domain
        self.n_subjects = len(self.X_source)
        self.source_train_info = []
        for nsub in range(self.n_subjects):
            source_event_type = np.unique(self.y_source[nsub])
            self.source_train_info.append({
                'event_type': source_event_type,
                'n_events': source_event_type.shape[0],
                'n_train': np.array([np.sum(self.y_source[nsub] == et)
                                     for et in source_event_type]),
                'n_chans': self.X_source[nsub].shape[-2],
                'n_points': self.X_source[nsub].shape[-1],
                'standard': True,
                'ensemble': True
            })

        # basic information of target domain
        target_event_type = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        self.target_train_info = {
            'event_type': target_event_type,
            'n_events': target_event_type.shape[0],
            'n_train': np.array([np.sum(self.y_train == et)
                                 for et in target_event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
            'standard': True,
            'ensemble': False
        }

        # main process
        self.intra_target_training()
        self.transfer_learning()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns: Dict[str, ndarray]
            rho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
            erho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features (ensemble).
            erho (ndarray): (Ne*Nte,Ne). Intergrated features (ensemble).
        """
        return iismc_feature(
            X_test=X_test,
            trans_model=self.trans_model,
            target_model=self.target_training_model,
            standard=self.standard,
            ensemble=self.ensemble
        )


# %% 9. CCA with intra- & inter-subject EEG (ACC-based subject selection) | ASS-IISCCA
class ASS_IISCCA(BasicTransfer):
    pass


# %% x. 
