# -*- coding: utf-8 -*-
"""
@ Author: Brynhildr Wu
@ Email: brynhildrwu@gmail.com

Transfer learning based on matrix decomposition.
    (1) SAME: https://ieeexplore.ieee.org/document/9971465/
            DOI: 10.1109/TBME.2022.3227036
<<<<<<< HEAD
    (2) TNSRE_20233250953: https://ieeexplore.ieee.org/document/10057002/
=======
    (2) TL-TRCA: (unofficial name): https://ieeexplore.ieee.org/document/10057002/
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
            DOI: 10.1109/TNSRE.2023.3250953
    (3) stCCA: https://ieeexplore.ieee.org/document/9177172/
            DOI: 10.1109/TNSRE.2020.3019276
    (4) tlCCA: https://ieeexplore.ieee.org/document/9354064/
            DOI: 10.1109/TASE.2021.3054741
    (5) sd-LST: https://ieeexplore.ieee.org/document/9967845/
            DOI: 10.1109/TNSRE.2022.3225878
<<<<<<< HEAD
    (6) TNSRE_20233305202: https://ieeexplore.ieee.org/document/10216996/
            DOI: 10.1109/TNSRE.2023.3305202
    (7) gTRCA: http://www.nature.com/articles/s41598-019-56962-2
            DOI: 10.1038/s41598-019-56962-2
=======
    (6) algo_TNSRE_2023_3305202: https://ieeexplore.ieee.org/document/10216996/
            DOI: 10.1109/TNSRE.2023.3305202
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
    n_subjects: Ns

<<<<<<< HEAD
"""

# %% Basic modules
from abc import abstractmethod

import utils

import cca
from cca import BasicCCA, BasicFBCCA

import trca

from typing import Optional, List, Tuple, Dict, Union
from numpy import ndarray

import numpy as np
from numba import njit
import scipy.linalg as sLA

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


# %% Basic Transfer object
class BasicTransfer(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self,
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
    def intra_source_training(self,):
        """Intra-domain model training for source dataset."""
        pass

    @abstractmethod
    def inter_source_training(self,):
        """Inter-domain model training for multiple source datasets."""
        pass

    @abstractmethod
    def transfer_learning(self,):
        """Transfer learning between source & target datasets."""
        pass

    @abstractmethod
    def target_augmentation(self,):
        """Data augmentation for target dataset."""
        pass

    @abstractmethod
    def dist_calc(self,):
        """Calculate spatial distance of source & target datasets."""
        pass

    @abstractmethod
    def weight_calc(self,):
        """Optimize the transfer weight for each source domain."""
        pass

    @abstractmethod
    def intra_target_training(self,):
        """Intra-domain model training for target dataset."""
        pass

    def fit(self,
            X_train: ndarray,
            y_train: ndarray,
            X_source: List[ndarray],
            y_source: List[ndarray],
            sine_template: Optional[ndarray] = None):
=======
update: 2023/07/04

"""

# %% Basic modules
from abc import abstractmethod, ABCMeta

import utils
import cca
from cca import BasicCCA, BasicFBCCA, cca_compute, msecca_compute
import trca
# from trca import BasicTRCA, BasicFBTRCA

from typing import Optional, List, Tuple, Any
from numpy import ndarray

import numpy as np
import scipy.linalg as sLA


# %% Basic Transfer object
class BasicTransfer(metaclass=ABCMeta):
    def __init__(self,
        standard: Optional[bool] = True,
        ensemble: Optional[bool] = True,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None):
        """Basic configuration.

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
    def source_intra_training(self,):
        """Intra-subject model training for source dataset."""
        pass


    @abstractmethod
    def transfer_learning(self,):
        """Transfer learning for source datasets."""
        pass


    @abstractmethod
    def data_augmentation(self,):
        """Data augmentation for target dataset."""
        pass


    @abstractmethod
    def dist_calculation(self,):
        """Calculate spatial distance of target & source datasets."""
        pass


    @abstractmethod
    def weight_optimization(self,):
        """Optimize the transfer weight for each source subject."""
        pass


    @abstractmethod
    def target_intra_training(self,):
        """Intra-subject model training for target dataset."""
        pass


    @abstractmethod
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        stim_info: Optional[dict] = None,
        sine_template: Optional[ndarray] = None,
        ):
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """Load data and train classification models.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset (target domain). Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
<<<<<<< HEAD
            sine_template (ndarray, optional): (Ne,2*Nh,Np). Sinusoidal template.
=======
            stim_info (Optional[dict]): Information of stimulus.
                {'event_type':ndarray, (Ne,),
                 'freqs':List or ndarray, (Ne,),
                 'phases':List or ndarray, (Ne,), etc}
            sine_template (Optional[ndarray]): (Ne,2*Nh,Np). Sinusoidal template.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
<<<<<<< HEAD
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
    def transform(self,
                  X_test: ndarray) -> Union[tuple, ndarray, Dict[str, ndarray]]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients, etc.
        """
        pass

    def predict(self,
                X_test: ndarray) -> Union[tuple, ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            y_pred (ndarray): (Ne*Nte,). Predict labels.
        """
        self.rho = self.transform(X_test)
        event_type = self.trian_info['event_type']
        self.y_pred = event_type[np.argmax(self.rho, axis=-1)]
        return self.y_pred


class BasicFBTransfer(utils.FilterBank, ClassifierMixin):
    def predict(self,
                X_test: ndarray) -> Tuple[ndarray]:
        """Using filter-bank transfer algorithms to predict test data.
=======
        self.stim_info = stim_info
        self.sine_template = sine_template

        # main process
        self.transfer_learning()
        self.data_augmentation()
        self.target_intra_training()
        self.dist_calculation()
        self.weight_optimization()
        return self


    @abstractmethod
    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            y_standard (ndarray): (Ne*Nte,). Predict labels.
            erou (ndarray): (Ne*Nte,Ne). Decision coefficients (ensemble).
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels (ensemble).
        """
        pass


class BasicFBTransfer(metaclass=ABCMeta):
    def __init__(self,
        standard: Optional[bool] = True,
        ensemble: Optional[bool] = True,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None):
        """Basic configuration.

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
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        stim_info: Optional[dict] = None,
        sine_template: Optional[ndarray] = None,
        ):
        """Load data and train classification models.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training target dataset. Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Nb,Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            stim_info (Optional[dict]): Information of stimulus.
                {'event_type':ndarray, (Ne,),
                 'freqs':List or ndarray, (Ne,),
                 'phases':List or ndarray, (Ne,), etc}
            sine_template (Optional[ndarray]): (Ne,2*Nh,Np). Sinusoidal template.
        """
        pass


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using filter-bank algorithms to predict test data.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
<<<<<<< HEAD
            y_pred (ndarray): (Ne*Nte,). Predict labels.
        """
        self.features = self.transform(X_test)
        event_type = self.sub_estimator[0].train_info['event_type']
        self.y_pred = event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred
=======
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            y_standard (ndarray): (Ne*Nte,). Predict labels.
            erou (ndarray): (Ne*Nte,Ne). Decision coefficients (ensemble).
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels (ensemble).
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
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138


# %% 1. source aliasing matrix estimation, SAME


<<<<<<< HEAD
# %% 2. 10.1109/TNSRE.2023.3250953
def _tnsre_20233250953_kernel(
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

    # block covariance matrices: S & Q, (Ne,2Nc+2Nh,2Nc+2Nh)
    S = np.tile(np.eye(2 * n_chans + n_dims)[None, ...], (n_events, 1, 1))
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
        Cxsy = X_sum[ne] @ sine_template[ne].T  # (Nc,2Nh)
        Cxmy = X_mean[ne] @ sine_template[ne].T  # (Nc,2Nh)
        Cyy = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh)
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
    v = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,2Nh)
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(A=S[ne], B=Q[ne], n_components=n_components)
        w[ne] = spatial_filter[:, :n_chans]  # for raw signal
        u[ne] = spatial_filter[:, n_chans:2 * n_chans]  # for averaged template
        v[ne] = spatial_filter[:, 2 * n_chans:]  # for sinusoidal template
    ew = np.reshape(w, (n_events*n_components, n_chans), order='C')  # (Ne*Nk,Nc)
    eu = np.reshape(u, (n_events*n_components, n_chans), order='C')  # (Ne*Nk,Nc)
    ev = np.reshape(v, (n_events*n_components, n_dims), order='C')  # (Ne*Nk,Nc)

    # signal templates
    euX = np.zeros((n_events, eu.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros((n_events, ev.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    for ne in range(n_events):
        euX[ne] = eu @ X_mean[ne]
        evY[ne] = ev @ sine_template[ne]
    euX = utils.fast_stan_3d(euX)
    evY = utils.fast_stan_3d(evY)
    euX = np.reshape(euX, (n_events, -1), order='C')  # (Ne,Ne*Nk*Np)
    evY = np.reshape(evY, (n_events, -1), order='C')  # (Ne,Ne*Nk*Np)

    # training model
    training_model = {
        'Q': Q, 'S': S,
        'w': w, 'u': u, 'v': v, 'ew': ew, 'eu': eu, 'ev': ev,
        'euX': euX, 'evY': evY
    }
    return training_model


def _tnsre_20233250953_feature(
        X_test: ndarray,
        source_model: Dict[str, ndarray],
        trans_model: Dict[str, ndarray],
        target_model: Dict[str, ndarray]) -> ndarray:
    """The pattern matching process of algorithm TNSRE_20233250953.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        source_model (dict): {'source_euX': ndarray (Ns,Ne,Ne*Nk,Np),
                              'source_evY': ndarray (Ns,Ne,Ne*Nk,Np)}
            See details in TNSRE_20233250953.intra_source_training()
        trans_model (dict): {'trans_euX': ndarray (Ns,Ne,Ne*Nk,Nc),
                             'trans_evY': ndarray (Ns,Ne,Ne*Nk,Nc),
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
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of TNSRE_20233250953.
    """
    # load in models & basic information
    source_euX = source_model['source_euX']  # (Ns,Ne,Ne*Nk*Np)
    source_evY = source_model['source_evY']  # (Ns,Ne,Ne*Nk*Np)
    trans_euX = trans_model['trans_euX']  # (Ns,Ne,Ne*Nk,Nc)
    trans_evY = trans_model['trans_evY']  # (Ns,Ne,Ne*Nk,Nc)
    weight_euX = trans_model['weight_euX']  # (Ns,Ne)
    weight_evY = trans_model['weight_evY']  # (Ns,Ne)
    target_ew = target_model['ew']  # (Ne*Nk,Nc)
    target_euX = target_model['euX']  # (Ne,Ne*Nk*Np)
    target_evY = target_model['evY']  # (Ne,Ne*Nk*Np)
    n_subjects = source_euX.shape[0]  # Ns
    n_events = source_euX.shape[1]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    # n_points = X_test.shape[-1]  # Np, unnecessary

    # 4-part discriminant coefficients
    rho_temp = np.zeros((n_test, n_events, 4))  # (Ne*Nte,Ne,4)
    for nte in range(n_test):
        X_trans_x = utils.fast_stan_4d(trans_euX @ X_test[nte])  # (Ns,Ne,Ne*Nk,Np)
        X_trans_x = np.reshape(
            a=X_trans_x,
            newshape=(n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Ne*Nk*Np)
        X_trans_y = utils.fast_stan_4d(trans_evY @ X_test[nte])  # (Ns,Ne,Ne*Nk,Np)
        X_trans_y = np.reshape(
            a=X_trans_y,
            newshape=(n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Ne*Nk*Np)
        X_temp = utils.fast_stan_3d(target_ew @ X_test[nte])  # (Ne*Nk,Np)
        X_temp = np.tile(np.reshape(X_temp, -1, order='C'), (n_events, 1, 1))  # (Ne,Ne*Nk*Np)

        # rho 1 & 2: transferred pattern matching
        rho_temp[nte, :, 0] = np.sum(
            a=weight_euX * utils.fast_corr_3d(X=X_trans_x, Y=source_euX),
            axis=0
        )
        rho_temp[nte, :, 1] = np.sum(
            a=weight_evY * utils.fast_corr_3d(X=X_trans_y, Y=source_evY),
            axis=0
        )

        # rho 3 & 4: target-domain pattern matching
        rho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp, Y=target_euX)
        rho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp, Y=target_evY)
    # rho_temp /= n_points  # real Pearson correlation coefficient in scale
    return utils.combine_feature([
        rho_temp[..., 0],
        rho_temp[..., 1],
        rho_temp[..., 2],
        rho_temp[..., 3]])


class TNSRE_20233250953(BasicTransfer):
    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        self.source_intra_model, self.source_model = [], {}
        source_euX, source_evY = [], []  # List[ndarray]: Ns*(Ne,Ne*Nk,Np)
        for nsub in range(self.n_subjects):
            intra_model = _tnsre_20233250953_kernel(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                n_components=self.n_components
            )
            self.source_intra_model.append(intra_model)
            source_euX.append(intra_model['euX'])  # (Ne,Ne*Nk*Np)
            source_evY.append(intra_model['evY'])  # (Ne,Ne*Nk*Np)
        self.source_model['source_euX'] = np.stack(source_euX)  # reshaped, standardized
        self.source_model['source_evY'] = np.stack(source_evY)  # (Ns,Ne,Ne*Nk*Np)

    def transfer_learning(self):
        """Transfer learning process."""
        # basic information
        n_events = self.train_info['n_events']  # Ne
        n_chans = self.train_info['n_chans']  # Nc
        n_train = self.train_info['n_train']  # [Nt1,Nt2,...]

        # obtain transfer model (partial)
        self.trans_model = {}
        trans_euX, trans_evY = [], []  # List[ndarray]: Ns*(Ne,Ne*Nk,Nc)
        for nsub in range(self.n_subjects):
            euX, evY = self.source_euX[nsub], self.source_evY[nsub]  # (Ne,Ne*Nk,Np)

            # LST alignment
            trans_euX.append(np.zeros((n_events, euX.shape[1], n_chans)))  # (Ne,Ne*Nk,Nc)
            trans_evY.append(np.zeros((n_events, evY.shape[1], n_chans)))  # (Ne,Ne*Nk,Nc)
            for ne, et in enumerate(self.event_type):
                X_temp = self.X_train[self.y_train == et]  # (Nt,Nc,Np)
                train_trials = n_train[ne]
                for tt in range(train_trials):
                    # b * a^T * (a * a^T)^{-1}
                    trans_uX_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=euX[ne].T)
                    trans_vY_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=evY[ne].T)
                    trans_euX[nsub][ne] += trans_uX_temp.T
                    trans_evY[nsub][ne] += trans_vY_temp.T
                trans_euX[nsub][ne] /= train_trials
                trans_evY[nsub][ne] /= train_trials
        self.trans_model['trans_euX'] = np.stack(trans_euX)  # (Ns,Ne,Ne*Nk,Nc)
        self.trans_model['trans_evY'] = np.stack(trans_evY)  # (Ns,Ne,Ne*Nk,Nc)

    def dist_calc(self):
        """Calculate the spatial distances between source and target domain."""
        # load in models & basic information
        n_subjects = self.train_info['n_subjects']  # Ns
        n_events = self.train_info['n_events']  # Ne
        n_train = self.train_info['n_train']  # [Nt1,Nt2,...]
        trans_euX = self.trans_model['trans_euX']  # (Ns,Ne,Ne*Nk,Nc)
        trans_evY = self.trans_model['trans_evY']  # (Ns,Ne,Ne*Nk,Nc)
        source_euX = self.source_model['source_euX']  # (Ns,Ne,Ne*Nk*Np)
        source_evY = self.source_model['source_evY']  # (Ns,Ne,Ne*Nk*Np)

        # calculate distances
        dist_euX = np.zeros((n_subjects, n_events))  # (Ns,Ne)
        dist_evY = np.zeros_like(self.dist_euX)
        for ne, et in enumerate(self.event_type):
            X_temp = self.X_train[self.y_train == et]  # (Nt,Nc,Np)
            train_trials = n_train[ne]
            for tt in range(train_trials):
                X_trans_x = trans_euX[:, ne, ...] @ X_temp[tt]  # (Ns,Ne*Nk,Np)
                X_trans_y = trans_evY[:, ne, ...] @ X_temp[tt]  # (Ns,Ne*Nk,Np)
                X_trans_x = np.reshape(X_trans_x, (n_subjects, -1), order='C')
                X_trans_y = np.reshape(X_trans_y, (n_subjects, -1), order='C')
                dist_euX[:, ne] += utils.fast_corr_2d(X=X_trans_x, Y=source_euX[:, ne, :])
                dist_evY[:, ne] += utils.fast_corr_2d(X=X_trans_y, Y=source_evY[:, ne, :])
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
        self.target_training_model = _tnsre_20233250953_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            n_components=self.n_components
        )

    def fit(
        self,
=======
# %% 2. cross-subject transfer learning TRCA, TL-TRCA
class TL_TRCA(BasicTransfer):
    def source_intra_training(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> dict:
        """Intra-subject training for source dataset.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
            n_components (int): Number of eigenvectors picked as filters. Nk.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None'.

        Returns: dict
            Q (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Covariance matrices.
            S (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Variance matrices. Q^{-1}Sw = lambda w.
            w (List[ndarray]): Ne*(Nk,Nc). Spatial filters for original signal.
            u (List[ndarray]): Ne*(Nk,Nc). Spatial filters for averaged template.
            v (List[ndarray]): Ne*(Nk,2*Nh). Spatial filters for sinusoidal template.
            w_concat (ndarray): (Ne*Nk,Nc). Concatenated filter for w.
            u_concat (ndarray): (Ne*Nk,Nc). Concatenated filter for u.
            v_concat (ndarray): (Ne*Nk,2*Nh). Concatenated filter for v.
            uX (ndarray): (Ne,Ne*Nk,Np). Filtered averaged templates.
            vY (ndarray): (Ne,Ne*Nk,Np). Filtered sinusoidal templates.
        """
        # basic information
        event_type = np.unique(y_train)
        n_events = len(event_type)  # Ne of source dataset
        n_train = np.array([np.sum(y_train==et) for et in event_type])  # [Nt1,Nt2,...]
        n_chans = X_train.shape[-2]  # Nc
        n_points = X_train.shape[-1]  # Np
        n_2harmonics = sine_template.shape[1]  # 2*Nh

        # initialization
        S = np.zeros((n_events, 2*n_chans+n_2harmonics, 2*n_chans+n_2harmonics))
        Q = np.zeros_like(S)
        w, u, v, w_concat, u_concat, v_concat = [], [], [], [], [], []
        uX, vY = [], []

        # block covariance matrices: S & Q, (Ne,2Nc+2Nh,2Nc+2Nh)
        class_sum = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
        class_center = np.zeros_like(class_sum)
        for ne,et in enumerate(event_type):
            train_trials = n_train[ne]  # Nt
            assert train_trials>1, 'The number of training samples is too small!'
            X_temp = X_train[y_train==et]
            class_sum[ne] = np.sum(X_temp, axis=0)
            class_center[ne] = np.mean(X_temp, axis=0)
            XsXs = class_sum[ne] @ class_sum[ne].T  # (Nc,Nc)
            XsXm = class_sum[ne] @ class_center[ne].T  # (Nc,Nc), (Nc,Nc)
            XmXm = class_center[ne] @ class_center[ne].T  # (Nc,Nc), (Nc,Nc)
            XsY = class_sum[ne] @ sine_template[ne].T  # (Nc,2Nh), (2Nh,Nc)
            XmY = class_center[ne] @ sine_template[ne].T  # (Nc,2Nh), (2Nh,Nc)
            YY = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh), (2Nh,2Nh)
            XX = np.zeros((n_chans, n_chans))  # (Nc,Nc)
            for tt in range(train_trials):
                XX += X_temp[tt] @ X_temp[tt].T
            # XX = np.einsum('tcp,thp->ch', X_sub[ne], X_sub[ne]) # clear but slow

            # block covariance matrices S: [[S11,S12,S13],[S21,S22,S23],[S31,S32,S33]]
            # S11: inter-trial covariance
            S[ne, :n_chans, :n_chans] = XsXs

            # S12 & S21.T covariance between the SSVEP trials & the individual template
            S[ne, :n_chans, n_chans:2*n_chans] = XsXm
            S[ne, n_chans:2*n_chans, :n_chans] = XsXm.T

            # S13 & S31.T: similarity between the SSVEP trials & sinusoidal template
            S[ne, :n_chans, 2*n_chans:] = XsY
            S[ne, 2*n_chans:, :n_chans] = XsY.T

            # S23 & S32.T: covariance between the individual template & sinusoidal template
            S[ne, n_chans:2*n_chans, 2*n_chans:] = XmY
            S[ne, 2*n_chans:, n_chans:2*n_chans] = XmY.T

            # S22 & S33: variance of individual template & sinusoidal template
            S[ne, n_chans:2*n_chans, n_chans:2*n_chans] = XmXm
            S[ne, 2*n_chans:, 2*n_chans:] = YY

            # block covariance matrices Q: blkdiag(Q1,Q2,Q3)
            # Q1: variance of the single-trial SSVEP
            Q[ne, :n_chans, :n_chans] = XX

            # Q2 & Q3: variance of individual template & sinusoidal template
            Q[ne, n_chans:2*n_chans, n_chans:2*n_chans] = XmXm
            Q[ne, 2*n_chans:, 2*n_chans:] = YY

        # solve GEPs
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
            u.append(spatial_filter[:,n_chans:2*n_chans])  # (Nk,Nc) | for averaged template
            v.append(spatial_filter[:,2*n_chans:])  # (Nk,2Nh) | for sinusoidal template
        w_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
        u_concat = np.zeros_like(w_concat)  # (Ne*Nk,Nc)
        v_concat = np.zeros((np.sum(ndim), n_2harmonics))  # (Ne*Nk,2Nh)
        start_idx = 0
        for ne,dims in enumerate(ndim):
            w_concat[start_idx:start_idx+dims] = w[ne]
            u_concat[start_idx:start_idx+dims] = u[ne]
            v_concat[start_idx:start_idx+dims] = v[ne]
            start_idx += dims

        # intra-subject templates
        uX = np.zeros((n_events, u_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
        vY = np.zeros((n_events, v_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
        for ne in range(n_events):  # ensemble version
            uX[ne] = u_concat @ class_center[ne]
            vY[ne] = v_concat @ sine_template[ne]

        # intra-subject model
        intra_source_model = {
            'Q':Q, 'S':S,
            'w':w, 'u':u, 'v':v,
            'w_concat':w_concat, 'u_concat':u_concat, 'v_concat':v_concat,
            'uX':uX, 'vY':vY
        }
        return intra_source_model


    def transfer_learning(self):
        """Transfer learning process.

        Updates: object attributes
            n_subjects (int): The number of source subjects.
            source_intra_model (List[object]): See details in source_intra_training().
            event_type (ndarray): (Ne,).
            n_events: (int). Total number of stimuli.
            n_chans: (int). Total number of channels.
            partial_transfer_model (dict[str, List]):{
                'source_uX': List[ndarray]: Ns*(Ne,Ne*Nk,Np). uX of each source subject.
                'source_vY': List[ndarray]: Ns*(Ne,Ne*Nk,Np). vY of each source subject.
                'trans_uX': List[ndarray]: Ns*(Ne,Ne*Nk,Nc). Transfer matrices for uX.
                'trans_vY': List[ndarray]: Ns*(Ne,Ne*Nk,Nc). Transfer matrices for vY.
            }
        """
        # basic information
        self.n_subjects = len(self.X_source)  # Ns
        self.source_intra_model = []
        self.event_type = np.unique(self.y_train)
        self.n_events = len(self.event_type)
        self.n_chans = self.X_train.shape[-2]  # Nc for target dataset

        # obtain partial transfer model
        trans_uX, trans_vY = [], []
        source_uX, source_vY = [], []
        for nsub in range(self.n_subjects):
            intra_model = self.source_intra_training(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.source_intra_model.append(intra_model)
            uX, vY = intra_model['uX'], intra_model['vY']
            source_uX.append(uX)  # (Ne,Ne*Nk,Np)
            source_vY.append(vY)  # (Ne,Ne*Nk,Np)

            # LST alignment
            trans_uX.append(np.zeros((self.n_events, uX.shape[1], self.n_chans)))  # (Ne,Ne*Nk,Nc)
            trans_vY.append(np.zeros((self.n_events, vY.shape[1], self.n_chans)))  # (Ne,Ne*Nk,Nc)
            for ne,et in enumerate(self.event_type):
                X_temp = self.X_train[self.y_train==et]  # (Nt,Nc,Np)
                train_trials = X_temp.shape[0]
                for tt in range(train_trials):
                    trans_uX_temp, _, _, _ = sLA.lstsq(
                        a=X_temp[tt].T,
                        b=uX[ne].T
                    )  # b * a^T * (a * a^T)^{-1}
                    trans_vY_temp, _, _, _ = sLA.lstsq(
                        a=X_temp[tt].T,
                        b=vY[ne].T
                    )
                    trans_uX[nsub][ne] += trans_uX_temp.T
                    trans_vY[nsub][ne] += trans_vY_temp.T
                trans_uX[nsub][ne] /= train_trials
                trans_vY[nsub][ne] /= train_trials
        self.part_trans_model = {
            'source_uX':source_uX, 'source_vY':source_vY,
            'trans_uX':trans_uX, 'trans_vY':trans_vY
        }


    def data_augmentation(self,):
        """Do nothing."""
        pass


    def dist_calculation(self):
        """Calculate the spatial distances between source and target domain.

        Updates:
            dist_uX, dist_vY (ndarray): (Ns,Ne).
        """
        self.dist_uX = np.zeros((self.n_subjects, self.n_events))  # (Ns,Ne)
        self.dist_vY = np.zeros_like(self.dist_uX)
        for nsub in range(self.n_subjects):
            for ne,et in enumerate(self.event_type):
                X_temp = self.X_train[self.y_train==et]
                train_trials = X_temp.shape[0]
                for tt in range(train_trials):
                    self.dist_uX[nsub,ne] += utils.pearson_corr(
                        X=self.part_trans_model['trans_uX'][nsub][ne] @ X_temp[tt],
                        Y=self.part_trans_model['source_uX'][nsub][ne]
                    )
                    self.dist_vY[nsub,ne] += utils.pearson_corr(
                        X=self.part_trans_model['trans_vY'][nsub][ne] @ X_temp[tt],
                        Y=self.part_trans_model['source_vY'][nsub][ne]
                    )


    def weight_optimization(self):
        """Optimize the transfer weights.

        Updates:
            weight_uX, weight_vY (ndarray): (Ns,Ne)
        """
        self.weight_uX = self.dist_uX / np.sum(self.dist_uX, axis=0, keepdims=True)
        self.weight_vY = self.dist_vY / np.sum(self.dist_vY, axis=0, keepdims=True)


    def target_intra_training(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> dict:
        """Intra-subject training for target dataset.

        Args:
            See details in source_intra_training().

        Returns: dict
            See details in source_intra_training().
        """
        self.target_model = self.source_intra_training(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            n_components=n_components,
            ratio=ratio
        )


    def fit(self,
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
<<<<<<< HEAD
        sine_template: ndarray
    ):
        """Train model.
=======
        stim_info: Optional[dict] = None,
        sine_template: Optional[ndarray] = None):
        """Load data and train TL-TRCA models.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
<<<<<<< HEAD
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
=======
            stim_info (Optional[dict]): Information of stimulus.
                {'event_type':ndarray, (Ne,),
                 'freqs':List or ndarray, (Ne,),
                 'phases':List or ndarray, (Ne,), etc}. No need here.
            sine_template (Optional[ndarray]): (Ne,2*Nh,Np). Sinusoidal template.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
<<<<<<< HEAD
        self.sine_template = sine_template

        # basic information
        event_type = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {
            'event_type': event_type,
            'n_events': len(event_type),
            'n_train': np.array([np.sum(self.y_train == et)
                                 for et in event_type]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
        }

        # main process
        self.intra_source_training()
        self.transfer_learning()
        self.dist_calculation()
        self.weight_optimization()
        self.intra_target_training()

    def transform(self, X_test: ndarray) -> ndarray:
        """Transform test dataset to discriminant features.
=======
        self.stim_info = stim_info
        self.sine_template = sine_template

        # main process
        self.transfer_learning()
        self.dist_calculation()
        self.weight_optimization()
        self.target_intra_training(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            n_components=self.n_components,
            ratio=self.ratio
        )
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using TL-(e)TRCA algorithm to compute decision coefficients.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
<<<<<<< HEAD
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients.
        """
        return _tnsre_20233250953_feature(
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
            base_estimator=TNSRE_20233250953(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )
=======
            rou (ndarray): (Ne*Nte,Ne,4). Decision coefficients of TL-TRCA.
            y_predict (ndarray): (Ne*Nte,). Predict labels of TL-TRCA.
        """
        # basic information
        n_test = X_test.shape[0]

        self.rou = np.zeros((n_test, self.n_events, 4))
        self.final_rou = np.zeros((n_test, self.n_events))
        self.y_predict = np.empty((n_test))

        # rou 1 & 2: transferred pattern matching
        for nte in range(n_test):
            for nsub in range(self.n_subjects):
                trans_uX = self.part_trans_model['trans_uX'][nsub]  # (Ne,Ne*Nk,Nc)
                trans_vY = self.part_trans_model['trans_vY'][nsub]  # (Ne,Ne*Nk,Nc)
                source_uX = self.part_trans_model['source_uX'][nsub]  # (Ne,Ne*Nk,Np)
                source_vY = self.part_trans_model['source_vY'][nsub]  # (Ne,Ne*Nk,Np)
                for nem in range(self.n_events):
                    self.rou[nte,nem,0] += self.weight_uX[nsub,nem]*utils.pearson_corr(
                        X=trans_uX[nem] @ X_test[nte],
                        Y=source_uX[nem]
                    )
                    self.rou[nte,nem,1] += self.weight_vY[nsub,nem]*utils.pearson_corr(
                        X=trans_vY[nem] @ X_test[nte],
                        Y=source_vY[nem]
                    )

        # rou 3 & 4: self-trained pattern matching (similar to sc-(e)TRCA)
        for nte in range(n_test):
            for nem in range(self.n_events):
                temp_standard = self.target_model['w'][nem] @ X_test[nte]  # (Nk,Np)
                self.rou[nte,nem,2] = utils.pearson_corr(
                    X=temp_standard,
                    Y=self.target_model['uX'][nem]
                )
                self.rou[nte,nem,3] = utils.pearson_corr(
                    X=temp_standard,
                    Y=self.target_model['vY'][nem]
                )
                self.final_rou[nte,nem] = utils.combine_feature([
                    self.rou[nte,nem,0],
                    self.rou[nte,nem,1],
                    self.rou[nte,nem,2],
                    self.rou[nte,nem,3],
                ])
            self.y_predict[nte] = self.event_type[np.argmax(self.final_rou[nte,:])]
        return self.rou, self.y_predict


class FB_TL_TRCA(BasicFBTransfer):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        stim_info: Optional[dict] = None,
        sine_template: Optional[ndarray] = None):
        """Load data and train FB-TL-TRCA models.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training target dataset. Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Nb,Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            stim_info (Optional[dict]): Information of stimulus.
                {'event_type':ndarray, (Ne,),
                 'freqs':List or ndarray, (Ne,),
                 'phases':List or ndarray, (Ne,), etc}. No need here.
            sine_template (Optional[ndarray]): (Nb,Ne,2*Nh,Np). Sinusoidal template.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.stim_info = stim_info
        self.sine_template = sine_template
        self.n_bands = self.X_train.shape[0]
        
        # train TL-TRCA models in each band
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = TL_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            fb_X_source = [data[nb] for data in self.X_source]
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train[nb],
                X_source=fb_X_source,
                y_source=self.y_source,
                stim_info=self.stim_info,
                sine_template=self.sine_template[nb]
            )
        return self
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138


# %% 3. subject transfer based CCA, stCCA
class STCCA(BasicTransfer):
    def source_intra_training(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> dict:
        """Intra-subject training for source dataset.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
            n_components (int): Number of eigenvectors picked as filters. Nk.
                Nk of each subject must be same (given by input number or 1).
                i.e. ratio must be None.
            ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
                Defaults to None when n_component is not 'None'.

        Returns: dict
            Cxx (ndarray): (Nc,Nc). Variance matrices of EEG.
            Cxy (ndarray): (Nc,2*Nh). Covariance matrices.
            Cyy (ndarray): (2*Nh,2*Nh). Variance matrices of sinusoidal template.
            u (ndarray): (Nk,Nc). Common spatial filter for averaged template.
            v (ndarray): (Nk,2*Nh). Common spatial filter for sinusoidal template.
            uX (ndarray): (Ne,Nk,Np). Filtered averaged templates.
            vY (ndarray): (Ne,Nk,Np). Filtered sinusoidal templates.
            event_type (ndarray): (Ne,). Event id for current dataset.
        """
        # basic information
        event_type = np.unique(y_train)
        n_events = len(event_type)
        n_train = np.array([np.sum(y_train==et) for et in event_type])
        n_chans = X_train.shape[-2]  # Nc
        n_points = X_train.shape[-1]  # Np
        n_2harmonics = sine_template.shape[1]  # 2*Nh

        # obtain averaged template
        avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
        for ne,et in enumerate(event_type):
            train_trials = n_train[ne]
            assert train_trials > 1, 'The number of training samples is too small!'
            avg_template[ne] = np.mean(X_train[y_train==et], axis=0)  # (Nc,Np)

        # initialization
        Cxx = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
        Cyy = np.zeros((n_events, n_2harmonics, n_2harmonics))  # (Ne,2Nh,2Nh)
        Cxy = np.zeros((n_events, n_chans, n_2harmonics))  # (Ne,Nc,2Nh)

        # covariance matrices
        for ne in range(n_events):
            Cxx[ne] = avg_template[ne] @ avg_template[ne].T
            Cxy[ne] = avg_template[ne] @ sine_template[ne].T
            Cyy[ne] = sine_template[ne] @ sine_template[ne].T
        Cxx = np.sum(Cxx, axis=0)
        Cxy = np.sum(Cxy, axis=0)
        Cyy = np.sum(Cyy, axis=0)

        # solve GEPs
        u = utils.solve_gep(
            A=Cxy @ sLA.solve(Cyy, Cxy.T),
            B=Cxx,
            n_components=n_components,
            ratio=ratio
        )  # (Nk,Nc)
        v = utils.solve_gep(
            A=Cxy.T @ sLA.solve(Cxx, Cxy),
            B=Cyy,
            n_components=n_components,
            ratio=ratio
        )  # (Nk,2Nh)

        # intra-subject templates
        uX = np.zeros((n_events, u.shape[0], n_points))  # (Ne,Nk,Np)
        vY = np.zeros((n_events, v.shape[0], n_points))  # (Ne,Nk,Np)
        for ne in range(n_events):
            uX[ne] = u @ avg_template[ne]
            vY[ne] = v @ sine_template[ne]

        # intra-subject model
        intra_source_model = {
            'Cxx':Cxx, 'Cxy':Cxy, 'Cyy':Cyy,
            'u':u, 'v':v,
            'uX':uX, 'vY':vY, 'event_type':event_type
        }
        return intra_source_model


    def transfer_learning(self):
        """Transfer learning process. Actually there is no so-called transfer process.
            This function is only used for intra-subject training for source dataset.

        Updates: object attributes
            n_subjects (int): The number of source subjects.
            source_intra_model (List[object]): See details in source_intra_training().
            event_type (ndarray): (Ne,).
            n_events: (int). Total number of stimuli.
            n_chans: (int). Total number of channels.
        """
        # basic information
        self.n_subjects = len(self.X_source)  # Ns
        self.source_intra_model = []
        self.event_type = np.unique(self.y_train)
        self.n_events = len(self.event_type)
        self.n_chans = self.X_train.shape[-2]  # Nc for target dataset

        # intra-subject training for all source subjects
        for nsub in range(self.n_subjects):
            intra_model = self.source_intra_training(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.source_intra_model.append(intra_model)


    def data_augmentation(self):
        """Do nothing."""
        pass


    def dist_calculation(self):
        """Do nothing."""
        pass


    def weight_optimization(self):
        """Optimize the transfer weights.

        Updates:
            n_points (int).
            u (ndarray): (Nk,Nc). Spatial filters for averaged template (target).
            v (ndarray): (Nk,2Nh). Spatial filters for sinusoidal template (target).
            uX (ndarray): (Ne(t), Nk, Np). Filtered averaged templates (target).
            vY (ndarray): (Ne(t), Nk, Np). Filtered sinusoidal templates (target).
            buX (ndarray): (Nk, Ne(t)*Np). Concatenated uX (target).
            AuX (ndarray): (Ns*Nk, Ne(s)*Np). Concatenated uX (source).
            bvY (ndarray): (Nk, Ne(t)*Np). Concatenated vY (target).
            AvY (ndarray): (Ns*Nk, Ne(s)*Np). Concatenated vY (source).
            weight_uX (ndarray): (Ns,Nk). Transfer weights of averaged templates (source).
            weight_vY (ndarray): (Ns,Nk). Transfer weights of sinusoidal templates (source).
            wuX (ndarray): (Ne(s), Nk, Np). Transferred averaged templates (full-event).
            wvY (ndarray): (Ne(s), Nk, Np). Transferred sinusoidal templates (full-event).
        """
        # basic information
        self.n_points = self.X_train.shape[-1]  # Np

        # intra-subject training for target domain
        self.target_intra_model = self.target_intra_training(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            n_components=self.n_components,
            ratio=self.ratio
        )
        u, v = self.target_model['u'], self.target_model['v']
        uX, vY = self.target_model['uX'], self.target_model['vY']

        # transfer weights training based on LST: min||b-Aw||_2^2
        self.buX = np.zeros((u.shape[0], self.n_events*self.n_points))  # (Nk,Ne*Np)
        # self.bvY = np.zeros((v.shape[0], self.n_events*self.n_points))  # (Nk,Ne*Np)
        for ne in range(self.n_events):  # Ne for target dataset
            self.buX[:, ne*self.n_points:(ne+1)*self.n_points] = uX[ne]
            # self.bvY[:, ne*self.n_points:(ne+1)*self.n_points] = vY[ne]
        uX_total_Nk = 0
        # vY_total_Nk = 0
        for nsub in range(self.n_subjects):
            uX_total_Nk += self.source_intra_model[nsub]['u'].shape[0]
            # vY_total_Nk += self.source_intra_model[ns]['v'].shape[0]
        self.AuX = np.zeros((uX_total_Nk, self.n_events*self.n_points))  # (Ns*Nk,Ne*Np)
        # self.AvY = np.zeros((vY_total_Nk, self.n_events*self.n_points))  # (Ns*Nk,Ne*Np)
        row_uX_idx = 0
        # row_vY_idx = 0
        for nsub in range(self.n_subjects):
            uX_Nk = self.source_intra_model[nsub]['u'].shape[0]
            # vY_Nk = self.source_intra_model[nsub]['v'].shape[0]
            source_uX = self.source_intra_model[nsub]['uX']
            # source_vY = self.source_intra_model[nsub]['vY']
            source_event_type = self.source_intra_model[nsub]['event_type'].tolist()
            for ne,et in enumerate(self.event_type):
                event_id = source_event_type.index(et)
                self.AuX[row_uX_idx:row_uX_idx+uX_Nk, ne*self.n_points:(ne+1)*self.n_points] = source_uX[event_id]
                # AvY[row_vY_idx:row_vY_idx+vY_Nk, ne*self.n_points:(ne+1)*self.n_points] = source_vY[event_id]
            row_uX_idx += uX_Nk
            # row_vY_idx += vY_Nk
        self.weight_uX, _, _, _ = sLA.lstsq(a=self.AuX.T, b=self.buX.T)  # (Ns,Nk)
        # self.weight_vY, _, _, _ = sLA.lstsq(a=self.AvY.T, b=self.bvY.T)  # (Ns,Nk)

        # cross subject averaged templates
        self.wuX = np.zeros_like(self.source_intra_model[0]['uX'])  # (Ne(s),Nk,Np)
        # self.wvY = np.zeros_like(self.source_intra_model[0]['vY'])
        for nsub in range(self.n_subjects):
            self.wuX += np.einsum('k,ekp->ekp', self.weight_uX[nsub], self.source_intra_model[nsub]['uX'])
            # self.wvY += np.einsum('k,ekp->ekp', self.weight_vY[nsub], self.source_intra_model[nsub]['vY'])
        self.wuX /= self.n_subjects
        # self.wvY /= self.n_subjects


    def target_intra_training(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> dict:
        """Intra-subject training for target dataset.

        Args:
            See details in source_intra_training().

        Returns: dict
            See details in source_intra_training().
        """
        self.target_model = self.source_intra_training(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            n_components=n_components,
            ratio=ratio
        )


    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        sine_template: Optional[ndarray] = None):
        """Load data and train stCCA model.

        Args:
            X_train (ndarray): (Ne(t)*Nt,Nc,Np). Target training dataset. Typically Nt>=2.
            y_train (ndarray): (Ne(t)*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne(s)*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne(s)*Nt,). Labels for X_source.
            sine_template (ndarray): (Ne(t), 2*Nh, Np). Sinusoidal template.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template
        
        # main process
        self.transfer_learning()
        self.weight_optimization()
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using stCCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rou (ndarray): (Ne*Nte,Ne,4). Decision coefficients of TL-TRCA.
            y_predict (ndarray): (Ne*Nte,). Predict labels of TL-TRCA.
        """
        # basic information
        n_test = X_test.shape[0]
        self.rou = np.zeros((n_test, self.n_events, 2))
        self.final_rou = np.zeros((n_test, self.n_events))
        self.y_predict = np.empty((n_test))

        # rou 1 & 2
        for nte in range(n_test):
            temp = self.target_model['u'] @ X_test[nte]  # (Nk,Np)
            for nem in range(self.n_events):
                self.rou[nte,nem,0] = utils.pearson_corr(
                    X=temp,
                    Y=self.target_model['v'] @ self.sine_template[nem]
                )
                self.rou[nte,nem,1] = utils.pearson_corr(
                    X=temp,
                    Y=self.wuX[nem]
                )
                # self.rou[nte,nem,2] = utils.pearson_corr(
                #     X=temp,
                #     Y=self.wvY[nem]
                # )
                self.final_rou[nte,nem] = utils.combine_feature([
                    self.rou[nte,nem,0],
                    self.rou[nte,nem,1],
                    # self.rou[nte,nem,2]
                ])
            self.y_predict[nte] = self.event_type[np.argmax(self.final_rou[nte,:])]
        return self.rou, self.y_predict


class FB_STCCA(BasicFBTransfer):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        sine_template: Optional[ndarray] = None):
        """Load data and train FB-stCCA models.

        Args:
            X_train (ndarray): (Nb,Ne(t)*Nt,Nc,Np). Target training dataset. Typically Nt>=2.
            y_train (ndarray): (Ne(t)*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Nb,Ne(s)*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne(s)*Nt,). Labels for X_source.
            sine_template (ndarray): (Nb,Ne(t), 2*Nh, Np). Sinusoidal template.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template
        self.n_bands = self.X_train.shape[0]

        # train stCCA models in each band
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = STCCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            fb_X_source = [data[nb] for data in self.X_source]
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train[nb],
                X_source=fb_X_source,
                y_source=self.y_source,
                sine_template=self.sine_template[nb]
            )
        return self


# %% 4. transfer learning CCA, tlCCA
def tlcca_source_compute(
    X_train: ndarray,
    y_train: ndarray,
    sine_template: ndarray,
    source_filter: ndarray,
    train_info: dict,
    length_coef: Optional[float] = 0.99,
    scale: Optional[float] = 0.8,
    correct_mode: Optional[str] = 'dynamic',
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict:
    """Train tlCCA model from source-domain dataset (with training data).

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=1.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        source_filter (ndarray): (Ne,Nk,Nc). Pre-trained spatial filter w.
        train_info (dict): {'event_type':ndarray (Ne(t),), Ne(t)<=Ne(s)
                            'n_events':int,
                            'n_train':ndarray (Ne(t),),
                            'n_chans':int,
                            'n_points':int,
                            'freqs':List[float],
                            'phases':List[float],
                            'sfreq':float,
                            'refresh_rate':int}
        length_coef (float): Fuzzy coefficient while computing the length of impulse response.
        scale (float, Optional): Compression coefficient of subsequent fragment (0-1).
            Defaults to 0.8.
        correct_mode (str, Optional): 'dynamic' or 'static'.
            'static': Data fragment is intercepted starting from 1 s.
            'dynamic': Data fragment is intercepted starting from 1 period after th
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Nk of each subject must be same (given by input number or 1).
            i.e. ratio must be None.
        ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not 'None'.

    Returns: tlCCA model (with training data) (dict)
        periodic_impulse (ndarray): (Ne, response_length).
        conv_matrix_H (ndarray): (Ne, response_length, Np). Convolution matrix.
        w (ndarray): (Ne,Nk,Nc). Spatial filter w.
        r (List[ndarray]): Ne*(Nk, response_length). Impulse response.
        u (ndarray): (Nk,Nc). Common spatial filters for averaged template.
        v (ndarray): (Nk,2*Nh). Common spatial filters for sinusoidal template.
        wX (ndarray): (Ne,Nk,Np). Filtered averaged template. (w[Ne] @ X)
        rH (ndarray): (Ne,Nk,Np). Reconstructed signal.
        uX (ndarray): (Ne,Nk,Np). Filtered averaged templates. (u @ X)
        vY (ndarray): (Ne,Nk,Np). Filtered sinusoidal templates.
        log_w, log_r (List[ndarray]): Log files of filter w & r during iteration optimization.
        log_error (List[float]): Log files of norm error during iteration optimization.
        ms-eCCA model (dict): Model created from stcca_intra_compute().
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    freqs = train_info['freqs']
    phases = train_info['phases']
    sfreq = train_info['sfreq']  # float
    refresh_rate = train_info['refresh_rate']  # int

    # obtain convolution matrix H
    periodic_impulse = np.zeros((n_events, n_points))
    conv_matrix_H = [[[],[]] for ne in range(n_events)]
    for ne in range(n_events):
        periodic_impulse[ne] = utils.extract_periodic_impulse(
            freq=freqs[ne],
            phase=phases[ne],
            signal_length=n_points,
            sfreq=sfreq,
            refresh_rate=refresh_rate
        )
        response_length = int(np.ceil(sfreq*length_coef/freqs[ne]))
        H = utils.create_conv_matrix(
            periodic_impulse=periodic_impulse[ne],
            response_length=response_length
        )
        H_new = utils.correct_conv_matrix(
            H=H,
            freq=freqs[ne],
            sfreq=sfreq,
            scale=scale,
            mode=correct_mode
        )
        conv_matrix_H[ne][0] = H
        conv_matrix_H[ne][1] = H_new

    # obtain class center
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        avg_template[ne] = np.mean(X_train[y_train==et], axis=0)

    # alternating least square (ALS) optimization
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    r = [[] for ne in range(n_events)]
    rH = np.zeros_like(wX)
    log_w = [[] for ne in range(n_events)]
    log_r = [[] for ne in range(n_events)]
    log_error = [[] for ne in range(n_events)]
    for ne in range(n_events):
        # iteration initialization
        H = conv_matrix_H[ne][1]
        init_w = source_filter[ne]  # (Nk,Nc)
        init_r = utils.sin_wave(
            freq=freqs[ne],
            n_points=H.shape[0],
            phase=phases[ne],
            sfreq=sfreq
        )  # (response_length,)
        init_r /= np.sqrt(init_r @ init_r.T)
        init_wX = init_w @ avg_template[ne]  # (Nk,Np)
        init_rH = init_r @ H  # (Nk,Np)
        init_error = np.sum((init_wX - init_rH)**2)  # float
        old_error = init_error
        error_change = 0

        # update log files
        log_w[ne].append(init_w)
        log_r[ne].append(init_r)
        log_error[ne].append(init_error)

        # start iteration
        iter_count = 1
        old_w = init_w
        continue_training = True
        while continue_training:
            # compute new impulse response (r)
            temp_wX = old_w @ avg_template[ne]  # (Nk,Np)
            new_r, _, _, _ = sLA.lstsq(a=H.T, b=temp_wX.T)  # (response_length, Nk)
            new_r = new_r.T/np.sqrt(new_r.T @ new_r)  # (Nk, response_length)

            # compute new spatial filter (w)
            temp_rH = new_r @ H  # (Nk,Np)
            new_w, _, _, _, = sLA.lstsq(a=avg_template[ne].T, b=temp_rH.T)  # (Nc,Nk)
            new_w = new_w.T/np.sqrt(new_w.T @ new_w)  # (Nk,Nc)

            # update ALS error
            new_error = np.sum((temp_wX - temp_rH)**2)
            error_change = old_error - new_error
            log_error[ne].append(new_error)
            
            # deside whether stop training
            iter_count += 1
            continue_training = (iter_count < 200) * (abs(error_change) > 0.00001)

            # update w, r
            log_r[ne].append(new_r)
            log_w[ne].append(new_w)
            old_w = new_w
            old_error = new_error

        w[ne] = new_w  # (Nk,Nc)
        r[ne] = new_r  # (Nk, response_length)
        wX[ne] = w[ne] @ avg_template[ne]  # (Nk,Np)
        rH[ne] = r[ne] @ H  # (Nk,Np)

    # obtain ms-eCCA model (filter u & v)
    common_model = stcca_intra_compute(
        X_train=X_train,
        y_train=y_train,
        sine_template=sine_template,
        n_components=n_components,
        ratio=ratio
    )

    # tlCCA model (with training data)
    model = {
        'periodic_impulse':periodic_impulse, 'conv_matrix_H':conv_matrix_H,
        'w':w, 'r':r, 'u':common_model['u'], 'v':common_model['v'],
        'wX':wX, 'rH':rH, 'uX':common_model['uX'], 'vY':common_model['vY'],
        'log_w':log_w, 'log_r':log_r, 'log_error':log_error,
        'ms-eCCA model':common_model
    }
    return model


def tlcca_target_compute(
    sine_template: ndarray,
    source_model: dict,
    train_info: dict,
    length_coef: Optional[float] = 0.99,
    scale: Optional[float] = 0.8,
    correct_mode: Optional[str] = 'dynamic',
    n_components: Optional[int] = 1) -> dict:
    """Train tlCCA model from target-domain dataset (without training data).

    Args:
        sine_template (Ne(t),2*Nh,Np). Sinusoidal template.
        source_model (dict). Details in tlcca_source_compute().
        train_info (dict): {'event_type':ndarray (Ne(t),), Ne(t)<=Ne(s)
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int,
                            'transfer_group':Tuple[List[int], List[float]],
                            'phases':List[float],
                            'sfreq':float,
                            'refresh_rate':int}.
        length_coef (float): Fuzzy coefficient while computing the length of impulse response.
        scale (float, Optional): Compression coefficient of subsequent fragment (0-1).
            Defaults to 0.8.
        correct_mode (str, Optional): 'dynamic' or 'static'.
            'static': Data fragment is intercepted starting from 1 s.
            'dynamic': Data fragment is intercepted starting from 1 period after th
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Nk of each subject must be same (given by input number or 1).
            i.e. ratio must be None.

    Returns: tlCCA model (without training data) (dict)
        periodic_impulse (ndarray): (Ne, response_length).
        conv_matrix_H (ndarray): (Ne, response_length, Np). Convolution matrix.
        w (ndarray): (Ne,Nk,Nc). Spatial filter w.
        r (List[ndarray]): Ne*(Nk, response_length). Impulse response.
        u (ndarray): (Nk,Nc). Common spatial filters for averaged template.
        v (ndarray): (Nk,2*Nh). Common spatial filters for sinusoidal template.
        rH (ndarray): (Ne,Nk,Np). Reconstructed signal.
        vY (ndarray): (Ne,Nk,Np). Filtered sinusoidal templates.
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    n_points = train_info['n_points']  # Np
    (source_idx, freqs) = train_info['transfer_group']
    phases = train_info['phases']
    sfreq = train_info['sfreq']  # float
    refresh_rate = train_info['refresh_rate']  # int

    # obtain parameters from pre-trained models
    v, r = source_model['v'], source_model['r']
    assert n_components==v.shape[0], 'Nk must be equal between source & target domain!'

    # obtain convolution matrix H
    periodic_impulse = np.zeros((n_events, n_points))
    conv_matrix_H = [[[],[]] for ne in range(n_events)]
    for ne in range(n_events):
        periodic_impulse[ne] = utils.extract_periodic_impulse(
            freq=freqs[ne],
            phase=phases[ne],
            signal_length=n_points,
            sfreq=sfreq,
            refresh_rate=refresh_rate
        )
        response_length = int(np.ceil(sfreq*length_coef/freqs[ne]))
        H = utils.create_conv_matrix(
            periodic_impulse=periodic_impulse[ne],
            response_length=response_length
        )
        H_new = utils.correct_conv_matrix(
            H=H,
            freq=freqs[ne],
            sfreq=sfreq,
            scale=scale,
            mode=correct_mode
        )
        conv_matrix_H[ne][0] = H
        conv_matrix_H[ne][1] = H_new

    # obtain transfer templates
    rH = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    vY = np.zeros_like(rH)
    for ne in range(n_events):
        rH[ne] = r[source_idx[ne]] @ conv_matrix_H[ne][1]  # (Nk,Np)
        vY[ne] = v @ sine_template[ne]  # (Nk,Np)

    # tlCCA model (without training data)
    model = {
        'periodic_impulse':periodic_impulse, 'conv_matrix_H':conv_matrix_H,
        'w':source_model['w'], 'r':r, 'u':source_model['u'], 'v':v,
        'rH':rH, 'vY':vY
    }
    return model


class TLCCA(BasicCCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        stim_info: dict,
        target_events: List,
        source_sine_template: ndarray,
        target_sine_template: ndarray,
        source_filter: ndarray,
        transfer_group: Optional[Tuple[List[int], List[float]]] = None,
        sfreq: Optional[float] = 1000,
        refresh_rate: Optional[float] = 60,
        length_coef: Optional[float] = 0.99,
        scale: Optional[float] = 0.8,
        correct_mode: Optional[str] = 'dynamic'):
        """Train tlCCA model.

        Args:
            X_train (ndarray): (Ne(s)*Nt, Nc, Np). Training dataset (source). Nt>=1.
            y_train (ndarray): (Ne(s)*Nt,). Labels for X_train.
            stim_info (dict): {label:(freq, phase)}.
            target_events (List): Event labels of missing dataset (target domain).
            source_sine_template (ndarray): (Ne(s),2*Nh,Np). Sinusoidal template (source).
            target_sine_template (ndarray): (Ne(t),2*Nh,Np). Sinusoidal template (target).
            source_filter (ndarray): (Ne(s),Nk,Nc). Pre-trained spatial filters.
            transfer_group (List[Tuple[int,float]]): (source index, target frequency).
            length_coef (float): Fuzzy coefficient while computing the length of impulse response.
            scale (float, Optional): Compression coefficient of subsequent fragment (0-1).
                Defaults to 0.8.
            correct_mode (str, Optional): 'dynamic' or 'static'.
                'static': Data fragment is intercepted starting from 1 s.
                'dynamic': Data fragment is intercepted starting from 1 period after th
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.stim_info = stim_info
        source_events = np.unique(self.y_train)  # labels, int
        self.source_freqs = [self.stim_info[str(se)][0] for se in source_events]
        self.source_phases = [self.stim_info[str(se)][1] for se in source_events]
        self.source_sine_template = source_sine_template
        self.target_sine_template = target_sine_template

        self.source_filter = source_filter
        self.sfreq = sfreq
        self.refresh_rate = refresh_rate
        self.length_coef = length_coef
        self.scale = scale
        self.correct_mode = correct_mode

        self.source_train_info = {
            'event_type':source_events,
            'n_events':len(source_events),
            'n_train':np.array([np.sum(self.y_train==se) for se in source_events]),
            'n_chans':self.X_train.shape[-2],
            'n_points':self.X_train.shape[-1],
            'freqs':self.source_freqs,
            'phases':self.source_phases,
            'sfreq':self.sfreq,
            'refresh_rate':self.refresh_rate
        }

        # train tlCCA model for source-domain dataset
        self.source_model = tlcca_source_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.source_sine_template,
            source_filter=self.source_filter,
            train_info=self.source_train_info,
            length_coef=self.length_coef,
            scale=self.scale,
            correct_mode=self.correct_mode,
            n_components=self.n_components,
            ratio=self.ratio
        )

        # prepare for training target-domain dataset
        self.target_freqs = [self.stim_info[str(te)][0] for te in target_events]
        self.target_phases = [self.stim_info[str(te)][1] for te in target_events]
        if transfer_group:
            self.transfer_group = transfer_group
        else:
            self.transfer_group = (
                [i for i in range(self.source_train_info['n_events'])],
                self.target_freqs
            )
        self.target_train_info = {
            'event_type':np.array(target_events),
            'n_events':len(target_events),
            'n_chans':self.source_train_info['n_chans'],
            'n_points':self.source_train_info['n_points'],
            'transfer_group':self.transfer_group,
            'phases':self.target_phases,
            'sfreq':self.sfreq,
            'refresh_rate':self.refresh_rate
        }

        # train tlCCA model for target-domain dataset
        self.target_model = tlcca_target_compute(
            sine_template=self.target_sine_template,
            source_model=self.source_model,
            train_info=self.target_train_info,
            length_coef=self.length_coef,
            scale=self.scale,
            correct_mode=self.correct_mode,
            n_components=self.n_components
        )

        # combine source model & target model
        self.event_type = source_events.tolist() + target_events
        self.sine_template = np.concatenate((
            self.source_sine_template,
            self.target_sine_template), axis=0)
        self.w = np.concatenate((self.source_model['w'], self.target_model['w']), axis=0)
        self.r = np.concatenate((self.source_model['r'], self.target_model['r']), axis=0)
        self.u, self.v = self.source_model['u'], self.source_model['v']
        self.rH = np.concatenate((self.source_model['rH'], self.target_model['rH']), axis=0)
        self.vY = np.concatenate((self.source_model['vY'], self.target_model['vY']), axis=0)
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using tlCCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rou (ndarray): (Ne*Nte,Ne,3). Decision coefficients of tlCCA.
            y_predict (ndarray): (Ne*Nte,). Predict labels of tlCCA.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = len(self.event_type)  # Ne, full

        self.rou = np.zeros((n_test, n_events, 3))
        self.final_rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))

        # rou 1-3
        for nte in range(n_test):
            temp_uX = self.u @ X_test[nte]  # (Nk,Np)
            for nem in range(n_events):
                self.rou[nte,nem,0] = utils.pearson_corr(
                    X=temp_uX,
                    Y=self.vY[nem]
                )
                temp_wX = self.w[nem] @ X_test[nte]  # (Nk,Np)
                self.rou[nte,nem,1] = utils.pearson_corr(
                    X=temp_wX,
                    Y=self.rH[nem]
                )
                cca_model = cca_compute(
                    data=temp_wX,
                    template=self.sine_template[nem],
                    n_components=self.n_components,
                    ratio=self.ratio
                )
                self.rou[nte,nem,2] = utils.pearson_corr(
                    X=cca_model['uX'],
                    Y=cca_model['vY']
                )
                self.final_rou[nte,nem] = utils.combine_feature([
                    self.rou[nte,nem,0],
                    self.rou[nte,nem,1],
                    self.rou[nte,nem,2]
                ])
            self.y_predict[nte] = self.event_type[np.argmax(self.final_rou[nte,:])]
        return self.rou, self.y_predict


def fb_tlcca_source_compute():
    pass


class FB_TLCCA(BasicFBCCA):
    pass


# %% 5. small data least-squares transformation
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
    for ntet,tet in enumerate(target_event_type):
        temp = X_source[y_source==tet]
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
    for ntet,tet in enumerate(target_event_type):
        temp = X_target[y_target==tet]
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
    for ntet,tet in enumerate(target_event_type):
        temp = X_lst_1[y_lst_1==tet]
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
        'LST-1':projection_matrices, 'LST-2':projection_matrix,
        'X-LST-1':X_lst_1, 'X-LST-2':X_final, 'y':y_lst_1,
        'avg_template_target':avg_template_target, 'source_trials':source_trials
    }
    return source_model


class SD_LST(trca.BasicTRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        sine_template: Optional[ndarray] = None,
        coef_idx: Optional[List] = [1,2,4]):
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


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
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
                self.rou[nte,nem] = utils.combine_feature([
                    rou_etrca,
                    rou_ecca
                ])
                self.rou[nte,nem] = rou_etrca
            self.y_predict[nte] = self.event_type[np.argmax(self.rou[nte,:])]
        return self.rou, self.y_predict


# %% 6. cross-subject transfer method based on domain generalization
def internally_invariant(
    stim_target: ndarray,
    stim_neighbor: ndarray,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict:
    """Learning the internally-invariant spatial filter and template to extract
        common frequency information across neighboring stimuli.

    Args:
        stim_target (ndarray): (Nt,Nc,Np). Dataset of target stimulus. Nt>=2.
        stim_neighbor (ndarray): (Ne,Nt,Nc,Np). Dataset of neighboring stimulus. Nt>=2.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: internally-invariant model (dict)
        Q (ndarray): (Nc,Nc). Covariance of original data.
        S (ndarray): (Nc,Nc). Covariance of template data.
        w (ndarray): (Nk,Nc). Internally-invariant spatial filter.
        wX (ndarray): (Nk,Np). Internally-invariant template.
    """
    # basic information
    n_train = stim_target.shape[0]  # Nt
    n_chans = stim_target.shape[-2]  # Nc

    # S & Q: same with TRCA
    S = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    Q = np.zeros_like(S)
    avg_target = stim_target.mean(axis=0)  # (Nc,Np)
    avg_neighbor = stim_neighbor.mean(axis=1)  # (Ne,Nc,Np)

    for nn in range(stim_neighbor.shape[0]):
        S += avg_neighbor[nn] @ avg_neighbor[nn].T
    S += avg_target @ avg_target.T

    for ntr in range(n_train):
        Q += stim_target[ntr] @ stim_target[ntr].T
        for nn in range(stim_neighbor.shape[0]):
            Q += stim_neighbor[nn,ntr,...] @ stim_neighbor[nn,ntr,...].T

    # GEPs with merged data
    w = utils.solve_gep(
        A=S,
        B=Q,
        n_components=n_components,
        ratio=ratio
    )  # (Nk,Nc)

    # create templates
    wX = w @ avg_target  # (Nk,Np)

    model = {
        'Q':Q, 'S':S, 'w':w, 'wX':wX
    }
    return model


def mutually_invariant():
    pass

def single_trial():
    pass

class ALGO_TNSRE_2023_3305202(BasicTransfer):
    pass



# %% x. impulse response component analysis | IRCA
def irca_compute(
    X_train: ndarray,
    y_train: ndarray,
    pre_trained_filter: ndarray,
    train_info: dict,
    length_coef: Optional[float] = 0.99,
    scale: Optional[float] = 0.8,
    correct_mode: Optional[str] = 'dynamic',
    n_components: Optional[int] = 1,
    ratio: Optional[int] = None) -> dict:
    """Train IRCA model from training dataset.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=1.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        pre_trained_filter (ndarray): Pre-trained spatial filter w.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int,
                            'freqs':List[float],
                            'phases':List[float],
                            'sfreq':float,
                            'refresh_rate':float}
        freqs (List[float]): Stimulus frequencies.
        phases (List[float]): Stimulus phases.
        sfreq (Optional[float]): Sampling frequency. Defaults to 1000.
        refresh_rate (Optional[float]): Refresh rate of stimulation presentation device.
            Defaults to 240.
        length_coef (float): Fuzzy coefficient while computing the length of impulse response.
            Defaults to 0.99.
        scale (Optional[float]): Compression coefficient of subsequent fragment (0-1).
            Defaults to 0.8.
        correct_mode (Optional[str]): 'dynamic' or 'static'.
            'static': Data fragment is intercepted starting from 1 s.
            'dynamic': Data fragment is intercepted starting from 1 period after th
        n_components (Optional[int]): Number of eigenvectors picked as filters. Nk.
            Nk of each subject must be same (given by input number or 1).
            i.e. ratio must be None.
        ratio (Optional[int]): The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not 'None'.

    Returns: IRCA model (with training data) (dict)
        periodic_impulse (ndarray): (Ne, response_length).
        conv_matrix_H (ndarray): (Ne, response_length, Np). Convolution matrix.
        w (ndarray): (Ne,Nk,Nc). Spatial filter w.
        r (List[ndarray]): Ne*(Nk, response_length). Impulse response.
        wX (ndarray): (Ne,Nk,Np). Filtered averaged template. (w[Ne] @ X)
        rH (ndarray): (Ne,Nk,Np). Reconstructed signal.
        log_w, log_r (List[ndarray]): Log files of filter w & r during iteration optimization.
        log_error (List[float]): Log files of norm error during iteration optimization.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']
    n_chans = train_info['n_chans']
    n_points = train_info['n_points']
    freqs = train_info['freqs']
    phases = train_info['phases']
    sfreq = train_info['sfreq']
    refresh_rate = train_info['refresh_rate']

    # obtain convolution matrix H
    periodic_impulse = np.zeros((n_events, n_points))
    conv_matrix_H = [[[],[]] for ne in range(n_events)]
    for ne in range(n_events):
        periodic_impulse[ne] = utils.extract_periodic_impulse(
            freq=freqs[ne],
            phase=phases[ne],
            signal_length=n_points,
            sfreq=sfreq,
            refresh_rate=refresh_rate
        )
        response_length = int(np.ceil(sfreq*length_coef/freqs[ne]))
        H = utils.create_conv_matrix(
            periodic_impulse=periodic_impulse[ne],
            response_length=response_length
        )
        H_new = utils.correct_conv_matrix(
            H=H,
            freq=freqs[ne],
            sfreq=sfreq,
            scale=scale,
            mode=correct_mode
        )
        conv_matrix_H[ne][0] = H
        conv_matrix_H[ne][1] = H_new

    # obtain class center
    avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        avg_template[ne] = np.mean(X_train[y_train==et], axis=0)

    # alternating least square (ALS) optimization
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    r = [[] for ne in range(n_events)]
    rH = np.zeros_like(wX)
    log_w = [[] for ne in range(n_events)]
    log_r = [[] for ne in range(n_events)]
    log_error = [[] for ne in range(n_events)]
    for ne in range(n_events):
        # iteration initialization
        H = conv_matrix_H[ne][1]
        init_w = pre_trained_filter[ne]  # (Nk,Nc)
        init_r = utils.sin_wave(
            freq=freqs[ne],
            n_points=H.shape[0],
            phase=phases[ne],
            sfreq=sfreq
        )  # (response_length,)
        init_r /= np.sqrt(init_r @ init_r.T)
        init_wX = init_w @ avg_template[ne]  # (Nk,Np)
        init_rH = init_r @ H  # (Nk,Np)
        init_error = np.sum((init_wX - init_rH)**2)  # float
        old_error = init_error
        error_change = 0

        # update log files
        log_w[ne].append(init_w)
        log_r[ne].append(init_r)
        log_error[ne].append(init_error)

        # start iteration
        iter_count = 1
        old_w = init_w
        continue_training = True
        while continue_training:
            # compute new impulse response (r)
            temp_wX = old_w @ avg_template[ne]  # (Nk,Np)
            new_r, _, _, _ = sLA.lstsq(a=H.T, b=temp_wX.T)  # (response_length, Nk)
            new_r = new_r.T/np.sqrt(new_r.T @ new_r)  # (Nk, response_length)

            # compute new spatial filter (w)
            temp_rH = new_r @ H  # (Nk,Np)
            new_w, _, _, _, = sLA.lstsq(a=avg_template[ne].T, b=temp_rH.T)  # (Nc,Nk)
            new_w = new_w.T/np.sqrt(new_w.T @ new_w)  # (Nk,Nc)

            # update ALS error
            new_error = np.sum((temp_wX - temp_rH)**2)
            error_change = old_error - new_error
            log_error[ne].append(new_error)

            # deside whether stop training
            iter_count += 1
            continue_training = (iter_count < 200) * (abs(error_change) > 0.00001)

            # update w, r
            log_r[ne].append(new_r)
            log_w[ne].append(new_w)
            old_w = new_w
            old_error = new_error

        w[ne] = new_w  # (Nk,Nc)
        r[ne] = new_r  # (Nk, response_length)
        wX[ne] = w[ne] @ avg_template[ne]  # (Nk,Np)
        rH[ne] = r[ne] @ H  # (Nk,Np)

    # IRCA model
    model = {
        'periodic_impulse':periodic_impulse, 'conv_matrix_H':conv_matrix_H,
        'w':w, 'r':r, 'wX':wX, 'rH':rH,
        'log_w':log_w, 'log_r':log_r, 'log_error':log_error
    }
    return model


class IRCA(trca.BasicTRCA):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        pre_trained_filter: ndarray,
        freqs: List[float],
        phases: List[float],
        sfreq: Optional[float] = 1000,
        refresh_rate: Optional[float] = 240,
        length_coef: Optional[float] = 0.99,
        scale: Optional[float] = 0.8,
        correct_mode: Optional[str] = 'dynamic'):
        """Train IRCA model.

        Args:
            X_train (ndarray): (Ne*Nt, Nc, Np). Training dataset. Nt>=1.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            pre_trained_filter (ndarray): (Ne,Nk,Nc). Pre-trained spatial filter w.
            freqs (List[float]): Stimulus frequencies.
            phases (List[float]): Stimulus phases.
            sfreq (Optional[float]): Sampling frequency.
            refresh_rate (Optional[int]): Refresh rate of stimulation presentation device.
                Defaults to 240.
            length_coef (Optional[float]): Fuzzy coefficient while computing the length of
                impulse response. Defaults to 0.99.
            scale (Optional[float]): Compression coefficient of subsequent fragment (0-1).
                Defaults to 0.8.
            correct_mode (str, Optional): 'dynamic' or 'static'.
                'static': Data fragment is intercepted starting from 1 s.
                'dynamic': Data fragment is intercepted starting from 1 period after th
        """
        # basic information
        
        pass