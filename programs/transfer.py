# -*- coding: utf-8 -*-
"""
@ Author: Brynhildr Wu
@ Email: brynhildrwu@gmail.com

Transfer learning based on matrix decomposition.
    (1) SAME: https://ieeexplore.ieee.org/document/9971465/
            DOI: 10.1109/TBME.2022.3227036
    (2) msSAME: https://iopscience.iop.org/article/10.1088/1741-2552/ad0b8f
            DOI: 10.1088/1741-2552/ad0b8f
    (3) TNSRE-20233250953: https://ieeexplore.ieee.org/document/10057002/
            DOI: 10.1109/TNSRE.2023.3250953
    (4) stCCA: https://ieeexplore.ieee.org/document/9177172/
            DOI: 10.1109/TNSRE.2020.3019276
    (5) tlCCA: https://ieeexplore.ieee.org/document/9354064/
            DOI: 10.1109/TASE.2021.3054741
    (6) sd-LST: https://ieeexplore.ieee.org/document/9967845/
            DOI: 10.1109/TNSRE.2022.3225878
    (7) TNSRE-20233305202: https://ieeexplore.ieee.org/document/10216996/
            DOI: 10.1109/TNSRE.2023.3305202
    (8) gTRCA: http://www.nature.com/articles/s41598-019-56962-2
            DOI: 10.1038/s41598-019-56962-2
    (9) IISMC: https://ieeexplore.ieee.org/document/9350285/
            DOI: 10.1109/TNSRE.2021.3057938
    (10) ASS-IISCCA: https://ieeexplore.ieee.org/document/10159132/
            DOI: 10.1109/TNSRE.2023.3288397
    (11) LST: https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e
            DOI: 10.1088/1741-2552/abcb6e
    (12) ALPHA: https://ieeexplore.ieee.org/document/9516951/
            DOI: 10.1109/TBME.2021.3105331
    (13) TIM-20243374314: https://ieeexplore.ieee.org/document/10462176/
            DOI: 10.1109/TIM.2024.3374314
    (14) TBME-20243406603: https://ieeexplore.ieee.org/document/10632864/
            DOI: 10.1109/TBME.2024.3406603
    (15) CSSFT: http://iopscience.iop.org/article/10.1088/1741-2552/ac6b57
            DOI: 10.1088/1741-2552/ac6b57
    (16) eCSSFT: https://iopscience.iop.org/article/10.1088/1741-2552/ac81ee
            DOI: 10.1088/1741-2552/ac81ee

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
    response length: Nrl

"""

# %% Basic modules
from abc import abstractmethod

import utils

import cca
import trca
import dsp

from typing import Optional, List, Tuple, Dict, Union, Any
from numpy import ndarray

import numpy as np
import scipy.linalg as sLA
from copy import deepcopy

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, StratifiedShuffleSplit


# %% Basic Transfer object & functions
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

        Returns:
            y_pred (ndarray): (Ne*Nte,). Predict label(s).
        """
        self.features = self.transform(X_test)
        self.y_pred = self.event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


class BasicFBTransfer(utils.FilterBank, ClassifierMixin):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            X_source: Optional[Union[List[ndarray], ndarray]] = None,
            y_source: Optional[Union[List[ndarray], ndarray]] = None,
            bank_weights: Optional[ndarray] = None,
            **kwargs):
        """Load in training dataset and pass it to sub-esimators.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray] or ndarray): List[(Nb,Ne*Nt,Nc,Np)]. Source dataset.
            y_source (List[ndarray] or ndarray): List[(Ne*Nt,)]. Labels for X_source.
            bank_weights (ndarray, optional): Weights for different filter banks.
                Defaults to None (equal).
        """
        # basic information
        self.Nb = X_train.shape[0]

        # initialization
        self.bank_weights = bank_weights
        if self.version == 'SSVEP':
            self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25 for nb in range(self.Nb)])
        self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]

        # apply in each sub-band
        if isinstance(X_source, list):  # multi-person's data
            n_subjects = len(X_source)  # Ns
            for nb, se in enumerate(self.sub_estimator):
                se.fit(
                    X_train=X_train[nb],
                    y_train=y_train,
                    X_source=[X_source[nsub][nb] for nsub in range(n_subjects)],
                    y_source=y_source,
                    **kwargs
                )
        elif isinstance(X_source, ndarray):  # single-domain
            for nb, se in enumerate(self.sub_estimator):
                se.fit(
                    X_train=X_train[nb],
                    y_train=y_train,
                    X_source=X_source[nb],
                    y_source=y_source,
                    **kwargs
                )
        else:  # NoneType
            for nb, se in enumerate(self.sub_estimator):
                se.fit(X_train=X_train[nb], y_train=y_train, **kwargs)

    def predict(
            self,
            X_test: ndarray) -> Union[
                Tuple[ndarray, ndarray],
                Tuple[int, int], ndarray, int]:
        """Using filter-bank transfer algorithms to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
            y_pred (ndarray): (Ne*Nte,). Predict label(s).
        """
        self.features = self.transform(X_test)
        event_type = self.sub_estimator[0].event_type
        self.y_pred = event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


class BasicASS(object):
    """Accuracy-based subject selection (ASS)."""
    def __init__(
            self,
            X_source: List[ndarray],
            y_source: List[ndarray],
            X_target: ndarray,
            y_target: ndarray,
            standard: bool = True,
            ensemble: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            X_source (List[ndarray]): List[(Ne*Nt,Nc,Np)]. Source dataset.
            y_source (List[ndarray]): List[(Ne*Nt,)]. Labels for X_source.
            X_target (ndarray): (Ne*Nt,Nc,Np). Target dataset.
            y_target (ndarray): (Ne*Nt,). Labels for X_target.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        # config model
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_target
        self.y_target = y_target
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble

    @abstractmethod
    def evaluation(self):
        """Calculate some sort of classification accuracy for each source subject."""
        pass

    def sort_subject_list(self):
        """Arrange source subjects in descending order of classification accuracy."""
        self.sorted_pair = sorted(
            list(enumerate(self.acc_list)),
            key=lambda pair: pair[1],
            reverse=True
        )
        self.sorted_idx = [sp[0] for sp in self.sorted_pair]

    @abstractmethod
    def select_subjects(self) -> List[int]:
        """Main process.

        Returns:
            subject_indices (List[int]).
        """
        self.evaluation()
        self.sort_subject_list()
        # return self.sorted_idx[:threshold]


# %% 1. source aliasing matrix estimation | SAME
def same_augmentation(
        X_mean: ndarray,
        sine_template: ndarray,
        generate_trials: int = 3,
        noise_intensity: float = 0.05) -> ndarray:
    """Data augmentation of SAME.

    Args:
        X_mean (ndarray): (Nc,Np). Real source data.
        sine_template (ndarray): (2*Nh,Np). Sinusoidal template corresponding with X_mean.
        generate_trials (int). The number of generated trials. Ntg.
            Defaults to 3 (while Nt is 1).
        noise_intensity (float). The intensity of Guass noise.
            Defaults to 0.05.

    Returns:
        X_aug (ndarray): (Ntg,Nc,Np). Generated data.
    """
    # basic information
    n_chans, n_points = X_mean.shape  # Nc,Np

    # generate estimation of source signal
    P_trans = X_mean @ sine_template.T @ sLA.inv(sine_template @ sine_template.T)
    X_src = P_trans @ sine_template  # (Nc,2Nh) @ (2Nh,Np) = (Nc,Np)

    # add noise to generate multiple simulated trials
    X_var = np.eye((n_chans))  # Nc,Nc
    for nc in range(n_chans):
        X_var[nc, nc] = np.var(X_src[nc])
    X_aug = np.zeros((generate_trials, n_chans, n_points))  # (Ntg,Nc,Np)
    for gt in range(generate_trials):
        noise = np.random.multivariate_normal(
            mean=np.zeros((n_chans)),
            cov=X_var,
            size=n_points
        )
        X_aug[gt] = X_src + noise_intensity * noise.T
    return X_aug


class SAME(BasicTransfer):
    def target_augmentation(self):
        """SAME augmentation for fast-calibration dataset."""
        # basic information
        X_mean = utils.generate_mean(X=self.X_train, y=self.y_train)  # (Ne,Nc,Np)
        n_events = X_mean.shape[0]  # Ne

        # data augmentation
        self.X_aug = np.tile(
            A=np.zeros_like(X_mean),
            reps=(self.generate_trials, 1, 1)
        )  # (Ne*Ntg,Nc,Np)
        self.y_aug = []
        for ne in range(n_events):
            st, ed = ne * self.generate_trials, (ne + 1) * self.generate_trials
            self.X_aug[st:ed, ...] = same_augmentation(
                X_mean=X_mean[ne],
                sine_template=self.sine_template[ne],
                generate_trials=self.generate_trials,
                noise_intensity=self.noise_intensity
            )
            self.y_aug += [self.event_type[ne]] * self.generate_trials
        self.y_aug = np.array(self.y_aug)  # (Ne*Ntg,)

        # update target dataset
        self.X_train = np.concatenate((self.X_train, self.X_aug), axis=0)
        self.y_train = np.concatenate((self.y_train, self.y_aug))

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray,
            generate_trials: int = 3,
            noise_intensity: float = 0.05):
        """Train model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, optional): (Ne,2*Nh,Np). Sinusoidal template.
            generate_trials (int). The number of generated trials. Ntg.
                Defaults to 3 (e.g while Nt is 1).
            noise_intensity (float). The intensity of Guass noise.
                Defaults to 0.05.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.generate_trials = generate_trials
        self.noise_intensity = noise_intensity
        self.event_type = np.unique(self.y_train)

        # SAME augmentation
        self.target_augmentation()


class FB_SAME(BasicFBTransfer):
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
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=SAME(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray,
            generate_trials: int = 3,
            noise_intensity: float = 0.05,
            **kwargs):
        """Load in training dataset and pass it to sub-esimators.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, optional): (Ne,2*Nh,Np). Sinusoidal template.
            generate_trials (int). The number of generated trials. Ntg.
                Defaults to 3 (e.g while Nt is 1).
            noise_intensity (float). The intensity of Guass noise.
                Defaults to 0.05.
        """
        # basic information
        self.Nb = X_train.shape[0]

        # initialization
        self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]

        # apply in each sub-band
        X_train_fb = []
        for nb, se in enumerate(self.sub_estimator):
            se.fit(
                X_train=X_train[nb],
                y_train=y_train,
                sine_template=sine_template,
                generate_trials=generate_trials,
                noise_intensity=noise_intensity,
                **kwargs
            )
            X_train_fb.append(se.X_train)  # (Ne,Nt+Ntg,Nc,Np)
            self.y_train = se.y_train
        self.X_train = np.stack(X_train_fb, axis=0)  # (Nb,Ne,Nt+Ntg,Nc,Np)


# %% 2. Multi-stimulus SAME | msSAME
def mssame_augmentation(
        X_mean: ndarray,
        sine_template: ndarray,
        event_type: ndarray,
        events_group: Dict[str, List[int]],
        generate_trials: int = 3,
        noise_intensity: float = 0.05) -> ndarray:
    """Data augmentation of msSAME.

    Args:
        X_mean (ndarray): (Ne(s),Nc,Np). Source data.
            Ne(s) may be smaller than Ne (semi-supervised scenario).
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        event_type (ndarray): (Ne,). All kinds of labels.
        events_group (Dict[str, List[int]]): {'event_id':[idx_1,idx_2,...]}.
            Event indices being emerged for each event.
        target_n_trials (int). The number of generated trials. Ntg.
            Defaults to 3 (while Nt is 1).
        noise_intensity (float). The intensity of Guass noise.
            Defaults to 0.05.

    Returns:
        X_aug (ndarray): (Ne*Ntg,Nc,Np). Generated data.
        y_aug (ndarray): (Ne*Ntg,). Labels for X_aug
    """
    # basic information
    n_events = len(event_type)  # Ne
    n_chans, n_points = X_mean.shape[1], X_mean.shape[2]  # Nc,Np
    n_2harmonics = sine_template.shape[1]  # 2Nh

    # generate source aliasing matrices
    P_trans = np.zeros((n_events, n_chans, n_2harmonics))  # (Ne,Nc,2Nh)
    for ne in range(n_events):
        P_temp_1 = np.zeros((n_chans, n_2harmonics))  # (Nc,2Nh)
        P_temp_2 = np.zeros((n_2harmonics, n_2harmonics))  # (2Nh,2Nh)
        merged_indices = events_group[str(event_type[ne])]  # List[idx]
        for mi in merged_indices:
            P_temp_1 += X_mean[mi] @ sine_template[mi].T
            P_temp_2 += sine_template[mi] @ sine_template[mi].T
        P_trans[ne] = P_temp_1 @ sLA.inv(P_temp_2)  # (Nc,2Nh)

    # generate estimation of source signal
    X_src = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne in range(n_events):
        X_src[ne] = P_trans[ne] @ sine_template[ne]  # (Nc,Np)

    # add noise to generate multiple simulated trials
    X_aug = np.zeros((n_events, generate_trials, n_chans, n_points))  # (Ne,Ntg,Nc,Np)
    y_aug = []
    for ne in range(n_events):
        X_var = np.eye((n_chans))  # (Nc,Nc)
        label = event_type[ne]
        for nc in range(n_chans):
            X_var[nc, nc] = np.var(X_src[ne, nc, :])
        for gt in range(generate_trials):
            noise = np.random.multivariate_normal(
                mean=np.zeros((n_chans)),
                cov=X_var,
                size=n_points
            )
            X_aug[ne, gt, ...] = X_src[ne] + noise_intensity * noise.T
            y_aug.append(label)
    X_aug, _ = utils.reshape_dataset(data=X_aug)
    return X_aug, np.array(y_aug)


class MS_SAME(BasicTransfer):
    def target_augmentation(self):
        """msSAME augmentation for fast-calibration dataset."""
        # basic information
        X_mean = utils.generate_mean(X=self.X_train, y=self.y_train)  # (Ne,Nc,Np)

        # data augmentation
        self.X_aug, self.y_aug = mssame_augmentation(
            X_mean=X_mean,
            sine_template=self.sine_template,
            event_type=self.event_type,
            events_group=self.events_group,
            generate_trials=self.generate_trials,
            noise_intensity=self.noise_intensity
        )  # (Ne,Ntg,Nc,Np)

        # update target dataset
        self.X_train = np.concatenate((self.X_train, self.X_aug), axis=0)
        self.y_train = np.concatenate((self.y_train, self.y_aug))

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray,
            event_type: ndarray,
            events_group: Optional[Dict[str, List[int]]] = None,
            d: int = 2,
            generate_trials: int = 3,
            noise_intensity: float = 0.05):
        """Train model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, optional): (Ne,2*Nh,Np).
            event_type (ndarray): (Ne,). All kinds of labels.
            events_group (Dict[str, List[int]]): {'event_id':[idx_1,idx_2,...]}.
                Event indices being emerged for each event.
            d (int): The range of events to be merged. Defaults to 2.
            generate_trials (int). The number of generated trials. Ntg.
                Defaults to 3 (e.g while Nt is 1).
            noise_intensity (float). The intensity of Guass noise.
                Defaults to 0.05.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.generate_trials = generate_trials
        self.noise_intensity = noise_intensity
        self.event_type = event_type
        self.d = d
        if events_group is not None:
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(
                event_type=self.event_type,
                d=self.d
            )

        # SAME augmentation
        self.target_augmentation()


class FB_MS_SAME(BasicFBTransfer):
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
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=MS_SAME(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray,
            generate_trials: int = 3,
            noise_intensity: float = 0.05,
            **kwargs):
        """Load in training dataset and pass it to sub-esimators.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray, optional): (Ne,2*Nh,Np). Sinusoidal template.
            generate_trials (int). The number of generated trials. Ntg.
                Defaults to 3 (e.g while Nt is 1).
            noise_intensity (float). The intensity of Guass noise.
                Defaults to 0.05.
        """
        # basic information
        self.Nb = X_train.shape[0]

        # initialization
        self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]

        # apply in each sub-band
        X_train_fb = []
        for nb, se in enumerate(self.sub_estimator):
            se.fit(
                X_train=X_train[nb],
                y_train=y_train,
                sine_template=sine_template,
                generate_trials=generate_trials,
                noise_intensity=noise_intensity,
                **kwargs
            )
            X_train_fb.append(se.X_train)  # (Ne,Nt+Ntg,Nc,Np)
            self.y_train = se.y_train
        self.X_train = np.stack(X_train_fb, axis=0)  # (Nb,Ne,Nt+Ntg,Nc,Np)


# %% 3. 10.1109/TNSRE.2023.3250953
def generate_tnsre_20233250953_mat(
        X: ndarray,
        y: ndarray,
        sine_template: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """Generate covariance matrices Q & S for TNSRE_20233250953 model.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Sklearn-style dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.

    Returns:
        Q (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Covariance matrices.
        S (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Variance matrices.
        X_mean (ndarray): (Ne,Nc,Np). Trial-averaged X.
    """
    # basic information
    event_type = np.unique(y)
    n_events = event_type.shape[0]  # Ne
    n_chans = X.shape[1]  # Nc
    n_points = X.shape[-1]  # Np
    n_dims = sine_template.shape[1]  # 2Nh

    # block covariance matrices: S & Q
    S = np.tile(A=np.eye(2 * n_chans + n_dims), reps=(n_events, 1, 1))
    Q = np.zeros_like(S)  # (Ne,2*Nc+2*Nh,2*Nc+2*Nh)
    X_mean = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne, et in enumerate(event_type):
        X_temp = X[y == et]  # (Nt,Nc,Np)
        n_train = X_temp.shape[0]  # Nt
        assert n_train > 1, 'The number of training samples is too small!'

        X_sum = np.sum(X_temp, axis=0)  # (Nc,Np)
        X_mean[ne] = X_sum / n_train  # (Nc,Np)

        # blocks preparation
        Css = X_sum @ X_sum.T  # (Nc,Nc)
        Csm = X_sum @ X_mean[ne].T  # (Nc,Nc)
        Cmm = X_mean[ne] @ X_mean[ne].T  # (Nc,Nc)
        Csy = X_sum @ sine_template[ne].T  # (Nc,2Nh)
        Cmy = X_mean[ne] @ sine_template[ne].T  # (Nc,2Nh)
        Cyy = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh)
        Cxx = np.zeros_like(Css)  # (Nc,Nc)
        for ntr in range(n_train):
            Cxx += X_temp[ntr] @ X_temp[ntr].T

        # block covariance matrices S: [[S11,S12,S13],[S21,S22,S23],[S31,S32,S33]]
        # S11: inter-trial covariance
        S[ne, :n_chans, :n_chans] = Css

        # S12 & S21.T covariance between the SSVEP trials & the individual template
        S[ne, :n_chans, n_chans:2 * n_chans] = Csm
        S[ne, n_chans:2 * n_chans, :n_chans] = Csm.T

        # S13 & S31.T: similarity between the SSVEP trials & sinusoidal template
        S[ne, :n_chans, 2 * n_chans:] = Csy
        S[ne, 2 * n_chans:, :n_chans] = Csy.T

        # S23 & S32.T: covariance between the individual template & sinusoidal template
        S[ne, n_chans:2 * n_chans, 2 * n_chans:] = Cmy
        S[ne, 2 * n_chans:, n_chans:2 * n_chans] = Cmy.T

        # S22 & S33: variance of individual template & sinusoidal template
        S[ne, n_chans:2 * n_chans, n_chans:2 * n_chans] = 2 * Cmm
        S[ne, 2 * n_chans:, 2 * n_chans:] = 2 * Cyy

        # block covariance matrices Q: blkdiag(Q1,Q2,Q3)
        # Q1: variance of the single-trial SSVEP
        Q[ne, :n_chans, :n_chans] = Cxx

        # Q2 & Q3: variance of individual template & sinusoidal template
        Q[ne, n_chans:2 * n_chans, n_chans:2 * n_chans] = Cmm
        Q[ne, 2 * n_chans:, 2 * n_chans:] = Cyy
    return Q, S, X_mean


def solve_tnsre_20233250953_func(
        Q: ndarray,
        S: ndarray,
        n_chans: int,
        n_components: int = 1) -> Tuple[ndarray, ndarray, ndarray,
                                        ndarray, ndarray, ndarray]:
    """Solve TNSRE_20233250953 target function.

    Args:
        Q (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Covariance matrices.
        S (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Variance matrices.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.

    Returns:
        w (ndarray): (Ne,Nk,Nc). Spatial filters for original signal.
        u (ndarray): (Ne,Nk,Nc). Spatial filters for averaged template.
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters for sinusoidal template.
        ew (ndarray): (Ne*Nk,Nc). Concatenated w.
        eu (ndarray): (Ne*Nk,Nc). Concatenated u.
        ev (ndarray): (Ne*Nk,2*Nh). Concatenated v.
    """
    # basic information
    n_events = Q.shape[0]  # Ne
    n_dims = int(Q.shape[1] - 2 * n_chans)  # 2Nh

    # solve GEPs
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    u = np.zeros_like(w)  # (Ne,Nk,Nc)
    v = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,2Nh)
    for ne in range(n_events):
        spatial_filter = utils.solve_gep(A=S[ne], B=Q[ne], n_components=n_components)
        w[ne] = spatial_filter[:, :n_chans]  # for raw signal
        u[ne] = spatial_filter[:, n_chans:2 * n_chans]  # for averaged template
        v[ne] = spatial_filter[:, 2 * n_chans:]  # for sinusoidal template
    ew = np.reshape(w, (n_events * n_components, n_chans), 'C')  # (Ne*Nk,Nc)
    eu = np.reshape(u, (n_events * n_components, n_chans), 'C')  # (Ne*Nk,Nc)
    ev = np.reshape(v, (n_events * n_components, n_dims), 'C')  # (Ne*Nk,Nc)
    return w, u, v, ew, eu, ev


def generate_tnsre_20233250953_template(
        eu: ndarray,
        ev: ndarray,
        X_mean: ndarray,
        sine_template: ndarray) -> Tuple[ndarray, ndarray]:
    """Generate TNSRE_20233250953 templates.

    Args:
        eu (ndarray): (Ne*Nk,Nc). Concatenated u.
            See details in solve_tnsre_20233250953_func().
        ev (ndarray): (Ne*Nk,2*Nh). Concatenated v.
            See details in solve_tnsre_20233250953_func().
        X_mean (ndarray): (Ne,Nc,Np). Trial-averaged data.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.

    Returns:
        euX (ndarray): (Ne,Ne*Nk,Np). Filtered averaged templates.
        evY (ndarray): (Ne,Ne*Nk,Np). Filtered sinusoidal templates.
    """
    # spatial filtering process
    euX = utils.spatial_filtering(w=eu, X=X_mean)  # (Ne,Ne*Nk,Np)
    evY = utils.spatial_filtering(w=ev, X=sine_template)  # (Ne,Ne*Nk,Np)
    return euX, evY


def tnsre_20233250953_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: int = 1) -> Dict[str, ndarray]:
    """Intra-domain modeling process of algorithm TNSRE_20233250953.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        n_components (int): Number of eigenvectors picked as filters.

    Returns:
        Q (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Covariance matrices.
        S (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Variance matrices.
        w (ndarray): (Ne,Nk,Nc). Spatial filters for original signal.
        u (ndarray): (Ne,Nk,Nc). Spatial filters for averaged template.
        v (ndarray): (Ne,Nk,2*Nh). Spatial filters for sinusoidal template.
        ew (ndarray): (Ne*Nk,Nc). Concatenated w.
        eu (ndarray): (Ne*Nk,Nc). Concatenated u.
        ev (ndarray): (Ne*Nk,2*Nh). Concatenated v.
        euX (ndarray): (Ne,Ne*Nk*Np). Filtered averaged templates.
        evY (ndarray): (Ne,Ne*Nk*Np). Filtered sinusoidal templates.
    """
    # solve target functions
    Q, S, X_mean = generate_tnsre_20233250953_mat(
        X=X_train,
        y=y_train,
        sine_template=sine_template
    )
    w, u, v, ew, eu, ev = solve_tnsre_20233250953_func(
        Q=Q,
        S=S,
        n_chans=X_mean.shape[1],
        n_components=n_components
    )

    # generate spatial-filtered templates
    euX, evY = generate_tnsre_20233250953_template(
        eu=eu,
        ev=ev,
        X_mean=X_mean,
        sine_template=sine_template
    )

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
                              'euX': ndarray (Ne,Ne*Nk,Np),
                              'evY': ndarray (Ne,Ne*Nk,Np)}
            See details in TNSRE_20233250953.intra_target_training()

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models
    euX_source = source_model['euX_source']  # (Ns,Ne,Ne*Nk,Np)
    evY_source = source_model['evY_source']  # (Ns,Ne,Ne*Nk,Np)
    euX_trans = trans_model['euX_trans']  # (Ns,Ne,Ne*Nk,Nc)
    evY_trans = trans_model['evY_trans']  # (Ns,Ne,Ne*Nk,Nc)
    weight_euX = trans_model['weight_euX']  # (Ns,Ne)
    weight_evY = trans_model['weight_evY']  # (Ns,Ne)
    ew_target = target_model['ew']  # (Ne*Nk,Nc)
    euX_target = target_model['euX']  # (Ne,Ne*Nk,Np)
    evY_target = target_model['evY']  # (Ne,Ne*Nk,Np)

    # basic information
    n_subjects = euX_source.shape[0]  # Ns
    n_events = euX_source.shape[1]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    n_points = X_test.shape[-1]  # Np

    # reshape matrix for faster computing
    euX_source = np.reshape(euX_source, (n_subjects, n_events, -1), 'C')  # (Ns,Ne,Ne*Nk*Np)
    evY_source = np.reshape(evY_source, (n_subjects, n_events, -1), 'C')  # (Ns,Ne,Ne*Nk*Np)
    euX_target = np.reshape(euX_target, (n_events, -1), 'C')  # (Ne,Ne*Nk*Np)
    evY_target = np.reshape(evY_target, (n_events, -1), 'C')  # (Ne,Ne*Nk*Np)

    # 4-D features
    rho_temp = np.zeros((n_test, n_events, 4))  # (Ne*Nte,Ne,4)
    for nte in range(n_test):
        X_trans_x = np.reshape(
            a=utils.fast_stan_4d(euX_trans @ X_test[nte]),
            newshape=(n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Ne*Nk,Nc) @ (Nc,Np) -flatten-> (Ns,Ne,Ne*Nk*Np)
        X_trans_y = np.reshape(
            a=utils.fast_stan_4d(evY_trans @ X_test[nte]),
            newshape=(n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Ne*Nk,Nc) @ (Nc,Np) -flatten-> (Ns,Ne,Ne*Nk*Np)
        X_temp = np.tile(
            A=np.reshape(
                a=utils.fast_stan_2d(ew_target @ X_test[nte]),
                newshape=-1,
                order='C'
            ),
            reps=(n_events, 1)
        )  # (Ne*Nk,Nc) @ (Nc,Np) -flatten-> -repeat-> (Ne,Ne*Nk*Np)

        # rho 1 & 2: transferred pattern matching
        rho_temp[nte, :, 0] = np.sum(
            a=weight_euX * utils.fast_corr_3d(X=X_trans_x, Y=euX_source),
            axis=0
        )  # (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -sum-> (Ne,)
        rho_temp[nte, :, 1] = np.sum(
            a=weight_evY * utils.fast_corr_3d(X=X_trans_y, Y=evY_source),
            axis=0
        )  # (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -sum-> (Ne,)

        # rho 3 & 4: target-domain pattern matching
        # (Ne,Ne*Nk*Np) -corr-> (Ne,)
        rho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp, Y=euX_target)
        rho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp, Y=evY_target)
    rho_temp /= n_points  # real Pearson correlation coefficients in scale
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
        # self.intra_model_source = []
        euX_source, evY_source = [], []  # List[ndarray]: Ns*(Ne,Ne*Nk,Np)

        # obtain source model
        for nsub in range(self.n_subjects):
            intra_model = tnsre_20233250953_kernel(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                n_components=self.n_components
            )
            # self.intra_model_source.append(intra_model)
            euX_source.append(intra_model['euX'])  # (Ne,Ne*Nk,Np)
            evY_source.append(intra_model['evY'])  # (Ne,Ne*Nk,Np)
        self.source_model = {
            'euX_source': np.stack(euX_source),
            'evY_source': np.stack(evY_source)
        }  # (Ns,Ne,Ne*Nk,Np)

    def transfer_learning(self):
        """Transfer learning process."""
        # basic information
        n_events = self.target_info['n_events']  # Ne
        n_chans = self.target_info['n_chans']  # Nc
        n_train = self.target_info['n_train']  # [Nt1,Nt2,...]

        # obtain transfer model (partial)
        eu_trans, ev_trans = [], []  # List[ndarray]: Ns*(Ne,Ne*Nk,Nc)
        for nsub in range(self.n_subjects):
            euX = self.source_model['euX_source'][nsub]  # (Ne,Ne*Nk,Np)
            evY = self.source_model['evY_source'][nsub]  # (Ne,Ne*Nk,Np)

            # LST alignment
            eu_trans.append(np.zeros((n_events, euX.shape[1], n_chans)))  # (Ne,Ne*Nk,Nc)
            ev_trans.append(np.zeros((n_events, evY.shape[1], n_chans)))  # (Ne,Ne*Nk,Nc)
            for ne, et in enumerate(self.event_type):
                X_temp = self.X_train[self.y_train == et]  # (Nt,Nc,Np)
                train_trials = n_train[ne]
                for tt in range(train_trials):  # w = min ||b - A w||
                    uX_trans_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=euX[ne].T)
                    vY_trans_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=evY[ne].T)
                    eu_trans[nsub][ne] += uX_trans_temp.T
                    ev_trans[nsub][ne] += vY_trans_temp.T
                eu_trans[nsub][ne] /= train_trials
                ev_trans[nsub][ne] /= train_trials
        self.trans_model = {
            'eu_trans': np.stack(eu_trans),
            'ev_trans': np.stack(ev_trans)
        }  # (Ns,Ne,Ne*Nk,Nc)

    def dist_calc(self):
        """Calculate the spatial distances between source and target domain."""
        # load in models & basic information
        n_events = self.target_info['n_events']  # Ne
        n_train = self.target_info['n_train']  # [Nt1,Nt2,...]
        eu_trans = self.trans_model['eu_trans']  # (Ns,Ne,Ne*Nk,Nc)
        ev_trans = self.trans_model['ev_trans']  # (Ns,Ne,Ne*Nk,Nc)
        euX_source = self.source_model['euX_source']  # (Ns,Ne,Ne*Nk,Np)
        evY_source = self.source_model['evY_source']  # (Ns,Ne,Ne*Nk,Np)

        # reshape for fast computing: (Ns,Ne,Ne*Nk,Np) -reshape-> (Ns,Ne,Ne*Nk*Np)
        euX_source = np.reshape(euX_source, (self.n_subjects, n_events, -1), 'C')
        evY_source = np.reshape(evY_source, (self.n_subjects, n_events, -1), 'C')

        # calculate distances
        dist_euX = np.zeros((self.n_subjects, n_events))  # (Ns,Ne)
        dist_evY = np.zeros_like(self.dist_euX)
        for ne, et in enumerate(self.event_type):
            X_temp = self.X_train[self.y_train == et]  # (Nt,Nc,Np)
            train_trials = n_train[ne]  # Nt
            for tt in range(train_trials):
                X_trans_x = np.reshape(
                    a=eu_trans[:, ne, ...] @ X_temp[tt],
                    newshape=(self.n_subjects, -1),
                    order='C'
                )  # (Ns,Ne*Nk,Nc) @ (Nc,Np) -reshape-> (Ns,Ne*Nk*Np)
                X_trans_y = np.reshape(
                    a=ev_trans[:, ne, ...] @ X_temp[tt],
                    newshape=(self.n_subjects, -1),
                    order='C'
                )  # (Ns,Ne*Nk,Np) @ (Nc,Np) -reshape-> (Ns,Ne*Nk*Np)
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
        self.target_model = tnsre_20233250953_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            n_components=self.n_components
        )

    def fit(
            self,
            X_source: List[ndarray],
            y_source: List[ndarray],
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """Train model.

        Args:
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template

        # basic information of source & target domain
        self.n_subjects = len(self.X_source)
        self.source_info = []
        for nsub in range(self.n_subjects):
            self.source_info.append(
                utils.generate_data_info(X=self.X_source[nsub], y=self.y_source[nsub])
            )
        self.target_info = utils.generate_data_info(X=self.X_train, y=self.y_train)
        self.event_type = self.source_info[0]['event_type']

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

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return tnsre_20233250953_feature(
            X_test=X_test,
            source_model=self.source_model,
            trans_model=self.trans_model,
            target_model=self.target_model
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
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=TNSRE_20233250953(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 4. subject transfer based CCA, stCCA
def stcca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: int = 1) -> Dict[str, ndarray]:
    """Intra-domain modeling process of stCCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        n_components (int): Number of eigenvectors picked as filters.

    Returns:
        Cxx (ndarray): (Ne,Nc,Nc). Covariance of averaged EEG template.
        Cxy (ndarray): (Ne,Nc,2*Nh). Covariance between EEG and sinusoidal template.
        Cyy (ndarray): (Ne,2*Nh,2*Nh). Covariance of sinusoidal template.
        u (ndarray): (Nk,Nc). Spatial filters (EEG signal).
        v (ndarray): (Nk,2*Nh). Spatial filters (sinusoidal signal).
        uX (ndarray): (Ne,Nk*Np). Reshaped stCCA templates (EEG signal).
        vY (ndarray): (Ne,Nk*Np). Reshaped stCCA templates (sinusoidal signal).
    """
    # solve target functions
    Cxx, Cxy, Cyy, X_mean = cca.generate_msecca_mat(
        X=X_train,
        y=y_train,
        sine_template=sine_template
    )  # (Ne,Nc,Nc), (Ne,Nc,2*Nh), (Ne,2*Nh,2*Nh), (Ne,Nc,Np)
    Cxx = np.sum(Cxx, axis=0)  # (Nc,Nc)
    Cyy = np.sum(Cyy, axis=0)  # (2*Nh,2*Nh)
    Cxy = np.sum(Cxy, axis=0)  # (Nc,2*Nh)
    u, v = cca.solve_cca_func(Cxx=Cxx, Cxy=Cxy, Cyy=Cyy, n_components=n_components)

    # generate spatial-filtered templates
    uX, vY = cca.generate_msecca_template(
        u=u,
        v=v,
        X_mean=X_mean,
        sine_template=sine_template,
        check_direc=True
    )  # (Ne,Nk,Np), (Ne,Nk,Np)
    return {
        'Cxx': Cxx, 'Cyy': Cyy, 'Cxy': Cxy,
        'u': u, 'v': v, 'uX': uX, 'vY': vY
    }


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

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models
    uX_trans = trans_model['uX_trans']  # (Ne,Nk,Np)
    u_target = target_model['u']  # (Nk,Nc)
    vY_target = target_model['vY']  # (Ne,Nk,Np)

    # basic information
    n_events = vY_target.shape[0]  # Ne
    uX_trans = np.reshape(uX_trans, (n_events, -1), 'C')  # (Ne,Nk*Np)
    vY_target = np.reshape(vY_target, (n_events, -1), 'C')  # (Ne,Nk*Np)
    n_test = X_test.shape[0]  # Ne*Nte
    n_points = X_test.shape[-1]  # Np

    # 2-part discriminant coefficients
    rho_temp = np.zeros((n_test, n_events, 2))  # (Ne*Nte,Ne,2)
    for nte in range(n_test):
        X_temp = utils.fast_stan_2d(u_target @ X_test[nte])  # (Nk,Np)
        X_temp = np.tile(np.reshape(X_temp, -1, 'C'), (n_events, 1))  # (Ne,Nk*Np)

        # rho 1: target-domain pattern matching
        rho_temp[nte, :, 0] = utils.fast_corr_2d(X=X_temp, Y=vY_target)

        # rho 2: transferred pattern matching
        rho_temp[nte, :, 1] = utils.fast_corr_2d(X=X_temp, Y=uX_trans)
    rho_temp /= n_points  # real Pearson correlation coefficient in scale
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
        # self.intra_model_source = []
        uX_source = []  # List[ndarray]: Ns*(Ne,Nk*Np)
        for nsub in range(self.n_subjects):
            intra_model = stcca_kernel(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                n_components=self.n_components
            )
            # self.intra_model_source.append(intra_model)
            uX_source.append(intra_model['uX'])  # (Ne,Nk,Np)
        self.source_model = {'uX_source': np.stack(uX_source)}  # (Ns,Ne,Nk,Np)

    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        self.target_model = stcca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            n_components=self.n_components
        )

    def weight_calc(self):
        """Optimize the transfer weights."""
        # basic information
        event_type_target = self.target_info['event_type'].tolist()
        event_type_source = self.source_info[0]['event_type'].tolist()  # Ne (full)
        event_indices = [event_type_source.index(ett) for ett in event_type_target]

        # solve LST problem: w = min||b - A w||
        self.buX = np.reshape(
            a=self.target_model['uX'],
            newshape=(len(event_type_target), -1),
            order='C'
        )  # (Ne',Nk,Np) -reshape-> (Ne',Nk*Np)
        self.buX = np.reshape(a=self.buX, newshape=-1, order='C')  # -reshape-> (Ne'*Nk*Np)
        self.AuX = np.transpose(np.reshape(
            a=self.source_model['uX_source'][:, event_indices, :],
            newshape=(self.n_subjects, -1),
            order='C'
        ))  # (Ns,Ne',Nk*Np) -reshape-> (Ns,Ne'*Nk*Np) -transpose-> (Ne'*Nk*Np,Ns)
        self.weight_uX, _, _, _ = sLA.lstsq(a=self.AuX, b=self.buX)  # (Ns,)

    def transfer_learning(self):
        """Transfer learning process."""
        self.trans_model = {}
        uX_trans = np.einsum('s,sekp->ekp', self.weight_uX, self.source_model['uX_source'])
        self.trans_model['uX_trans'] = utils.fast_stan_3d(uX_trans)  # (Ne,Nk,Np)

    def fit(
            self,
            X_source: List[ndarray],
            y_source: List[ndarray],
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """Train stCCA model.

        Args:
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template

        # basic information of source & target domain
        self.n_subjects = len(self.X_source)
        self.source_info = []
        for nsub in range(self.n_subjects):
            self.source_info.append(
                utils.generate_data_info(X=self.X_source[nsub], y=self.y_source[nsub])
            )
        self.target_info = utils.generate_data_info(X=self.X_train, y=self.y_train)
        self.event_type = self.source_info[0]['event_type']  # full events

        # main process
        self.intra_source_training()
        self.intra_target_training()
        self.weight_calc()
        self.transfer_learning()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return stcca_feature(
            X_test=X_test,
            trans_model=self.trans_model,
            target_model=self.training_model_target
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
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=STCCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 5. transfer learning CCA, tlCCA
def fast_init_model(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: Optional[ndarray] = None,
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
        method (str, optional): 'msCCA', 'TRCA', 'eTRCA' or 'DSP'.
            If None, parameter 'w' should be given.
        w (ndarray, optional): (Nk,Nc) or (Ne(s),Nk,Nc).
            If None, parameter 'method' should be given.

    Returns:
        w_init (ndarray): (Ne(s),Nk,Nc). Spatial filter w.
        X_mean (ndarray): (Ne(s),Nc,Np). Averaged templates of X_train.
    """
    # basic information
    event_type = np.unique(y_train)
    n_events = event_type.shape[0]
    X_mean = utils.generate_mean(X=X_train, y=y_train)  # (Ne,Nc,Np)

    # train initial model according to selected method
    if method == 'msCCA':
        assert sine_template is not None, 'sine_template cannot be Nonetype!'
        model = cca.mscca_kernel(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            n_components=n_components
        )
        return {
            'w_init': np.tile(A=model['w'], reps=(n_events, 1, 1)),
            'X_mean': X_mean
        }  # w_init: (Ne,Nk,Nc)
    elif method == 'TRCA':
        model = trca.trca_kernel(
            X_train=X_train,
            y_train=y_train,
            ensemble=False,
            n_components=n_components
        )
        return {
            'w_init': model['w'],
            'X_mean': X_mean
        }  # w_init: (Ne,Nk,Nc)
    elif method == 'eTRCA':
        model = trca.trca_kernel(
            X_train=X_train,
            y_train=y_train,
            ensemble=True,
            n_components=n_components
        )
        return {
            'w_init': np.tile(A=model['ew'], reps=(n_events, 1, 1)),
            'X_mean': X_mean
        }  # w_init: (Ne,Ne*Nk,Nc)
    elif method == 'DSP':
        model = dsp.dsp_kernel(
            X_train=X_train,
            y_train=y_train,
            n_components=n_components
        )
        return {
            'w_init': np.tile(A=model['w'], reps=(n_events, 1, 1)),
            'X_mean': X_mean
        }  # w_init: (Ne,Nk,Nc)

    # reshape initial spatial filters (if given)
    if w is not None:
        if w.ndim == 2:  # common filter, (Nk,Nc)
            w_init = np.tile(A=w, reps=(n_events, 1, 1))  # (Ne,Nk,Nc)
        elif w.ndim == 3:  # (Ne,Nk,Nc)
            w_init = w
    else:
        raise Exception("Unknown initial model! Please check the input 'initial_model'!")
    return {'w_init': w_init, 'X_mean': X_mean}


def tlcca_conv_matrix(
        freq: Union[int, float],
        phase: Union[int, float],
        n_points: int,
        srate: Union[int, float] = 1000,
        rrate: int = 60,
        len_scale: float = 1.05,
        extract_method: str = 'Square',
        amp_scale: float = 0.8,
        concat_method: str = 'dynamic',
        response_length: Optional[int] = None) -> Tuple[ndarray, ndarray]:
    """Create convolution matrix H (H_correct) for tlCCA (single-event).

    Args:
        freq (int or float): Stimulus frequency.
        phase (int or float): Stimulus phase (coefficients). 0-2 (pi).
        n_points (int): Data length.
        srate (int or float): Sampling rate. Defaults to 1000 Hz.
        rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
        len_scale (float): The multiplying power when calculating the length of data.
            Defaults to 1.05.
        extract_method (str): 'Square' or 'Cosine'. Defaults to 'Square'.
            See details in utils.extract_periodic_impulse().
        amp_scale (float): The multiplying power when calculating the amplitudes of data.
            Defaults to 0.8.
        concat_method (str): 'dynamic' or 'static'.
            'static': Concatenated data is starting from 1 s.
            'dynamic': Concatenated data is starting from 1 period.
        response_length (int): The number of sampling points of a single period.
            Defaults to None.

    Returns:
        H (ndarray): (response_length,Np). Convolution matrix.
        H_correct (ndarray): (response_length,Np). Corrected convolution matrix.
    """
    # generate periodic impulse sequence (spikes)
    periodic_impulse = utils.extract_periodic_impulse(
        freq=freq,
        phase=phase,
        n_points=n_points,
        srate=srate,
        rrate=rrate,
        method=extract_method
    )  # (Np,)

    # config response length (sampling points)
    if response_length is None:
        response_length = int(np.round(srate * len_scale / freq))

    # generate convolution matrices
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


def solve_tlcca_func(
        w_init: Union[ndarray, None],
        X_mean: ndarray,
        H: ndarray,
        freq: Union[int, float],
        phase: Union[int, float] = 0,
        srate: Union[int, float] = 1000,
        optimize_method: str = 'CCA',
        iter_limit: int = 200,
        err_th: float = 0.00001,
        n_components: int = 1,
        check_symbol: bool = True,
        target_chan_idx: Optional[int] = None) -> Dict[str, Union[ndarray, int]]:
    """Solve the optimization problem:
    r, w = argmin||rH - wX||_2^2. In which r represents for impulse response,
    and w for spatial filters.

    Args:
        w_init (Union[ndarray, None]): (Nk,Nc). Initial spatial filter w.
            Only useful when optimize_method is 'ALS'.
            If optimize_method is 'CCA', w_init could be NoneType.
        X_mean (ndarray): (Nc,Np). Averaged template.
        H (ndarray): (response_length,Np). Convolution matrix.
        freq (Union[int, float]): Frequency of impulse response.
        phase (Union[int, float]): Initial phase of impulse response. Defaults to 0.
        srate (Union[int, float]): Sampling rate. Defaults to 1000 Hz.
        optimize_method (str): 'CCA' or 'ALS'. Defaults to 'CCA'(faster and better).
            If 'CCA', r,w = CCA(H, X);
            If 'ALS', r,w = argmix||rH - wX||.
        cal_method (str): 'lstsq' or 'pinv'. Methods to solve least squares problem:
            x = min ||b - Ax||_2. If 'lstsq', use sLA.lstsq(); If 'pinv', use sLA.pinv().
            Only useful when optimize_method is 'ALS'.
        iter_limit (int): Number of maximum iteration times. Defaults to 200.
            Only useful when optimize_method is 'ALS'.
        err_th (float): The threshold (th) of ALS error. Stop iteration while
            ALS error is smaller than err_th. Defaults to 10^-5.
            Only useful when optimize_method is 'ALS'.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Defaults to 1. Only useful when optimize_method is 'CCA'.
        check_symbol (bool): Whether to correct the symbol of w & r. Defaults to True.
        target_chan_idx (int): The index of target channel to correct the symbols of w & r.
            Recommend to set to the channel 'Oz'.

    Returns:
        w (ndarray): (Nk,Nc). Optimized spatial filter.
        r (ndarray): (Nk,Nrl). Optimized impulse response.
        wX (ndarray): (Nk,Np). w @ X_mean.
        rH (ndarray): (Nk,Np). r @ H.
    """
    if optimize_method == 'CCA':
        cca_model = cca.cca_kernel(
            X=X_mean,
            Y=utils.centralization(H),
            n_components=n_components
        )
        w, r = cca_model['u'], cca_model['v']
        wX, rH = cca_model['uX'], cca_model['vY']
    elif optimize_method == 'ALS':
        # initial impulse response (r) & template (rH)
        r_init = np.tile(
            A=utils.sin_wave(freq=freq, n_points=H.shape[0], phase=phase, srate=srate),
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
            wX_temp = w_old @ X_mean  # (Nk,Np)

            # calculate new impulse response (r_new)
            # lstsq method: (more stable)
            # r_new, _, _, _ = sLA.lstsq(a=H.T, b=wX_temp.T)  # (Nrl, Nk)
            # r_new = r_new.T
            # Penrose inverse method (faster)
            r_new = wX_temp @ H.T @ sLA.pinv(H @ H.T)  # (Nk,Nrl)
            r_new = np.diag(1 / np.sqrt(np.diag(r_new @ r_new.T))) @ r_new  # ||r(i,:)|| = 1

            # calculate new spatial filter (w_new)
            rH_temp = r_new @ H  # (Nk,Np)
            # w_new, _, _, _ = sLA.lstsq(a=X_mean.T, b=rH_temp.T)  # (Nc,Nk)
            # w_new = w_new.T  # (Nk,Nc)
            w_new = rH_temp @ X_mean.T @ sLA.pinv(X_mean @ X_mean.T)  # (Nk,Nc)
            w_new = np.diag(1 / np.sqrt(np.diag(w_new @ w_new.T))) @ w_new  # ||w(i,:)|| = 1

            # update ALS error
            # err_new = np.norm(wX_temp - rH_temp)**2  # slower
            err_new = np.sum((wX_temp - rH_temp)**2)  # faster
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
        # update optimized results
        w, r, wX, rH = log_w[-1], log_r[-1], w @ X_mean, r @ H
        _, r, rH = cca.symbol_correction(uX=wX, vY=rH, v=r)  # symbol correction

    # symbol correction (final)
    if check_symbol:
        X_tar = X_mean[target_chan_idx][None, :]  # (1,Np)
        r_tar = X_tar @ H.T @ sLA.pinv(H @ H.T)  # (1,response_length)
        if utils.pearson_corr(X=r_tar, Y=r) < 0:
            return {'w': -1 * w, 'r': -1 * r, 'wX': -1 * wX, 'rH': -1 * rH}
    return {'w': w, 'r': r, 'wX': wX, 'rH': rH}


def tlcca_kernel(
        X_train: ndarray,
        y_train: ndarray,
        stim_info: Dict[str, Tuple[Union[float, int], Union[float, int]]],
        H: List[ndarray],
        w_init: Optional[ndarray] = None,
        srate: Union[float, int] = 1000,
        optimize_method: str = 'CCA',
        n_components: int = 1,
        iter_limit: int = 200,
        err_th: float = 0.00001,
        target_chan_idx: int = 7) -> Dict[str, Union[List[ndarray], ndarray]]:
    """Intra-domain modeling process of tlCCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        stim_info (dict): {'label': (frequency, phase)}.
        H (List[ndarray]): List[(Nrl,Np)]. Convolution matrix.
        w_init (Union[ndarray, None]): (Nk,Nc). Initial spatial filter w.
            Only useful when optimize_method is 'ALS'.
            If optimize_method is 'CCA', w_init could be NoneType.
        srate (Union[int, float]): Sampling rate. Defaults to 1000 Hz.
        optimize_method (str): 'CCA' or 'ALS'. Defaults to 'CCA'(faster and better).
            If 'CCA', r,w = CCA(H, X);
            If 'ALS', r,w = argmix||rH - wX||.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1. Only useful when optimize_method is 'CCA'.
        iter_limit (int): Number of maximum iteration times. Defaults to 200.
            Only useful when optimize_method is 'ALS'.
        err_th (float): The threshold (th) of ALS iteration error. Defaults to 10^-5.
            Only useful when optimize_method is 'ALS'.
        target_chan_idx (int): The index of target channel to correct the symbols of w & r.
            Recommend to set to the channel 'Oz'.

    Returns:
        w (ndarray): (Ne,Nk,Nc). Optimized spatial filters.
        r (ndarray): List[(Nk,Nrl)]. Optimized impulse responses.
        wX (ndarray): (Ne,Nk,Np). w @ X_mean.
        rH (ndarray): (Ne,Nk,Np). r @ H.
    """
    # initialization
    data_info = utils.generate_data_info(X=X_train, y=y_train)
    X_mean = utils.generate_mean(X=X_train, y=y_train)  # (Ne,Nc,Np)
    w, r, wX, rH, tlcca_model = [], [], [], [], {}

    # iteration or analytical solution
    if optimize_method == 'ALS':  # ALS optimization
        for ne, et in enumerate(data_info['event_type']):
            als_model = solve_tlcca_func(
                w_init=w_init[ne],
                X_mean=X_mean[ne],
                H=H[ne],
                freq=stim_info[str(et)][0],
                phase=stim_info[str(et)][1],
                srate=srate,
                optimize_method=optimize_method,
                iter_limit=iter_limit,
                err_th=err_th,
                target_chan_idx=target_chan_idx
            )
            w.append(als_model['w'])  # (Nk,Nc)
            r.append(als_model['r'])  # (Nk,Nrl)
            wX.append(als_model['wX'])  # (Nk,Np)
            rH.append(als_model['rH'])  # (Nk,Np)
    elif optimize_method == 'CCA':  # CCA solution: {r, w} = CCA(H, X)
        for ne, et in enumerate(data_info['event_type']):
            cca_model = solve_tlcca_func(
                w_init=None,
                X_mean=X_mean[ne],
                H=H[ne],
                freq=stim_info[str(et)][0],
                phase=stim_info[str(et)][1],
                srate=srate,
                optimize_method=optimize_method,
                n_components=n_components,
                target_chan_idx=target_chan_idx
            )
            w.append(cca_model['w'])  # (Nk,Nc)
            r.append(cca_model['r'])  # (Nk,Nrl)
            wX.append(cca_model['wX'])  # (Nk,Np)
            rH.append(cca_model['rH'])  # (Nk,Np)
    tlcca_model['w'] = np.stack(w, axis=0)  # List[(Nk,Nc)] -> (Ne(s),Nk,Nc)
    tlcca_model['r'] = deepcopy(r)  # List[(Nk,Nrl)]
    tlcca_model['wX'] = np.stack(wX, axis=0)  # List[(Nk,Nc)] -> (Ne(s),Nk,Np)
    tlcca_model['rH'] = np.stack(rH, axis=0)  # List[(Nk,Nc)] -> (Ne(s),Nk,Np)
    return tlcca_model


def tlcca_feature(
        X_test: ndarray,
        sine_template: ndarray,
        trans_model: dict,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The pattern matching process of tlCCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        trans_model (dict): {'u_trans': ndarray (Nk,Nc),
                             'w_trans': ndarray (Ne,Nk,Nc),
                             'vY_trans': ndarray (Ne,Nk,Np),
                             'rH_trans': ndarray (Ne,Nk,Np)}
            See details in TLCCA.tranfer_learning()
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Defaults to 1.

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,3). 3-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models
    u_trans = trans_model['u_trans']  # (Ne,Nk,Nc)
    w_trans = trans_model['w_trans']  # (Ne,Nk,Nc)
    vY_trans = trans_model['vY_trans']  # (Ne,Nk,Np)
    rH_trans = trans_model['rH_trans']  # (Ne,Nk,Np)

    # basic information
    n_events = w_trans.shape[0]  # Ne
    n_points = X_test.shape[-1]  # Np
    n_test = X_test.shape[0]  # Ne*Nte
    vY_trans = np.reshape(vY_trans, (n_events, -1), 'C')  # (Ne,Nk*Np)
    rH_trans = np.reshape(rH_trans, (n_events, -1), 'C')  # (Ne,Nk*Np)

    # 3-part discriminant coefficients
    rho_temp = np.zeros((n_test, n_events, 3))
    for nte in range(n_test):
        uX_temp = np.reshape(
            a=utils.fast_stan_3d(u_trans @ X_test[nte]),
            newshape=(n_events, -1),
            order='C'
        )  # (Ne,Nk,Np) -reshape-> (Ne,Nk*Np)
        wX_temp = utils.fast_stan_3d(w_trans @ X_test[nte])  # (Ne,Nk,Np)

        # rho 1: corr(uX, vY)
        rho_temp[nte, :, 0] = utils.fast_corr_2d(X=uX_temp, Y=vY_trans) / n_points

        # rho 2: corr(wX, rH)
        rho_temp[nte, :, 1] = utils.fast_corr_2d(
            X=np.reshape(a=wX_temp, newshape=(n_events, -1), order='C'),
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
    return {
        'rho_temp': rho_temp,
        'rho': utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1],
            rho_temp[..., 2]
        ])
    }


class TLCCA(BasicTransfer):
    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        # initialization
        if self.optimize_method == 'ALS':
            # alternating least square (ALS) optimization: {r, w} = argmin||wX - rH||
            init_model = fast_init_model(
                X_train=self.X_source,
                y_train=self.y_source,
                sine_template=self.source_sine_template,
                n_components=self.n_components,
                method=self.source_info['init_model'],
                w=self.source_info['w_init']
            )
            w_init = init_model['w_init']  # (Ne(s),Nk,Nc)
        elif self.optimize_method == 'CCA':
            w_init = None

        # train tlCCA model for source domain
        self.source_model = tlcca_kernel(
            X_train=self.X_source,
            y_train=self.y_source,
            stim_info=self.stim_info,
            H=self.H_source,
            w_init=w_init,
            srate=self.srate,
            optimize_method=self.optimize_method,
            n_components=self.n_components,
            iter_limit=self.iter_limit,
            err_th=self.err_th,
            target_chan_idx=self.target_chan_idx
        )

        # spatial filters u, v from ms-eCCA process
        msecca_model = cca.msecca_kernel(
            X_train=self.X_source,
            y_train=self.y_source,
            sine_template=self.sine_template_source,
            events_group=self.source_info['events_group'],
            n_components=self.n_components
        )
        self.source_model['u'] = msecca_model['u']  # (Ne(s),Nk,Np)
        self.source_model['v'] = msecca_model['v']  # (Ne(s),Nk,Np)

    def transfer_learning(self):
        """Transfer learning between exist & missing events."""
        # load in models
        r_source = self.source_model['r']  # List, Ne(s)*(Nk,Nrl)
        w_source = self.source_model['w']  # (Ne(s),Nk,Nc)
        u_source = self.source_model['u']  # (Ne(s),Nk,Nc)
        v_source = self.source_model['v']  # (Ne(s),Nk,2*Nh)

        # basic information
        n_events = self.event_type.shape[0]  # Ne (full)
        n_chans = w_source.shape[-1]  # Nc
        n_points = self.source_info['n_points']  # Np

        # create tranferred model for unknown events
        r_new = []
        w_new = np.empty((1, self.n_components, n_chans))
        u_new = np.empty_like((w_new))
        v_new = np.empty((1, self.n_components, 2 * self.n_harmonics))
        for tp in self.transfer_pair:
            # the indices corresponding to self.source_info['event_type']
            idx_src = list(self.source_info['event_type']).index(tp[0])

            # make a copy
            r_new.append(r_source[idx_src])
            w_new = np.concatenate((w_new, w_source[idx_src][None, ...]), axis=0)
            u_new = np.concatenate((u_new, u_source[idx_src][None, ...]), axis=0)
            v_new = np.concatenate((v_new, v_source[idx_src][None, ...]), axis=0)
        # remove the redundant parts
        w_new = np.delete(w_new, 0, axis=0)
        u_new = np.delete(u_new, 0, axis=0)
        v_new = np.delete(v_new, 0, axis=0)

        # concatenate source model & transferred model as target model
        r_trans_temp = r_source + r_new
        r_trans = [r_trans_temp[so] for so in self.sorted_order]
        H_trans_temp = self.H_correct_source + self.H_correct_new
        H_trans = [H_trans_temp[so] for so in self.sorted_order]
        w_trans = np.concatenate((w_source, w_new), axis=0)[self.sorted_order]
        u_trans = np.concatenate((u_source, u_new), axis=0)[self.sorted_order]
        v_trans = np.concatenate((v_source, v_new), axis=0)[self.sorted_order]
        del r_trans_temp, H_trans_temp

        # create full-event templates rH & vY
        rH_trans = np.zeros((n_events, self.n_components, n_points))  # (Ne,Nk,Np)
        vY_trans = np.zeros_like(rH_trans)  # (Ne,Nk,Np)
        for ne in range(n_events):
            rH_trans[ne] = r_trans[ne] @ H_trans[ne]
            vY_trans[ne] = v_trans[ne] @ self.sine_template[ne]
        rH_trans = utils.fast_stan_3d(rH_trans)
        vY_trans = utils.fast_stan_3d(vY_trans)

        # transfer model
        self.trans_model = {
            'r_trans': r_trans, 'H_trans': H_trans,
            'u_trans': u_trans, 'w_trans': w_trans, 'v_trans': v_trans,
            'vY_trans': vY_trans, 'rH_trans': rH_trans
        }

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            stim_info: Dict[str, Tuple[Union[float, int], Union[float, int]]],
            n_harmonics: int = 1,
            init_method: Optional[str] = 'msCCA',
            w_init: Optional[ndarray] = None,
            events_group: Optional[Dict[str, List[int]]] = None,
            d: int = 2,
            transfer_pair: Optional[List[Tuple[int, int]]] = None,
            srate: Union[float, int] = 1000,
            rrate: int = 60,
            len_scale: float = 1.05,
            optimize_method: str = 'CCA',
            amp_scale: float = 0.8,
            concat_method: str = 'dynamic',
            iter_limit: int = 200,
            err_th: float = 0.00001,
            target_chan_idx: int = 7):
        """Train tlCCA model.

        Args:
            X_train (ndarray): (Ne(s)*Nt,Nc,Np). Source training dataset.
                Nt>=2, Ne (source) < Ne (full).
            y_train (ndarray): (Ne(s)*Nt,). Labels for X_source.
            stim_info (dict): {'label': (frequency, phase)}.
            n_harmonics (int): Number of harmonic components for sinusoidal templates.
                Defaults to 1.
            init_method (str): 'msCCA', 'TRCA' or 'DSP'. Defaults to 'msCCA'.
                Only useful when optimize_method is 'ALS'.
            w_init (ndarray): (Ne,Nk,Nc). Initial spatial filter(s). Defaults to None.
                Only useful when optimize_method is 'ALS'.
            events_group (Dict[str, List[int]], optional): {'event_id':[idx_1,idx_2,...]}.
                If None, events_group will be generated according to parameter 'd'.
                Only useful when init_method is 'msCCA' and optimize_method is 'ALS'.
            d (int): The range of events to be merged.
                Only useful when init_method is 'msCCA' and optimize_method is 'ALS'.
            transfer_pair (List[Tuple[int, int]]): (source label, target label).
            srate (int or float): Sampling rate. Defaults to 1000 Hz.
            rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
            len_scale (float): The multiplying coefficient for the length of data.
                Defaults to 1.05.
            amp_scale (float): The multiplying coefficient for the amplitudes of data.
                Defaults to 0.8.
            concat_method (str): 'dynamic' or 'static'.
                'static': Concatenated data is starting from 1 s.
                'dynamic': Concatenated data is starting from 1 period.
            optimize_method (str): 'CCA' or 'ALS'. Defaults to 'CCA'(faster and better).
                If 'CCA', r,w = CCA(H, X);
                If 'ALS', r,w = argmix||rH - wX||.
            iter_limit (int): Number of maximum iteration times. Defaults to 200.
                Only useful when optimize_method is 'ALS'.
            err_th (float): The threshold (th) of ALS iteration error. Defaults to 10^-5.
                Only useful when optimize_method is 'ALS'.
            target_chan_idx (int): The index of target channel to correct of w & r.
                Recommend to set to the channel 'Oz'. Defaults to 7.
                See details in solve_tlcca_func().
        """
        # load in data
        self.X_source = X_train
        self.y_source = y_train
        self.stim_info = stim_info
        self.n_harmonics = n_harmonics
        self.transfer_pair = transfer_pair
        self.srate = srate
        self.rrate = rrate
        self.len_scale = len_scale
        self.amp_scale = amp_scale
        self.concat_method = concat_method
        self.optimize_method = optimize_method
        self.iter_limit = iter_limit
        self.err_th = err_th
        self.target_chan_idx = target_chan_idx

        # special check: Nk must be 1
        assert self.n_components == 1, 'Only support Nk=1 for now!'
        assert self.transfer_pair is not None, "Please provide the input 'transfer_pair'"

        # create sinusoidal templates for full events
        self.event_type = np.array(list(set(
            [tp[0] for tp in self.transfer_pair] + [tp[1] for tp in self.transfer_pair]
        )))
        n_events = self.event_type.shape[0]  # Ne (total)
        n_points = self.X_source.shape[-1]  # Np
        self.sine_template = np.zeros((n_events, 2 * self.n_harmonics, n_points))
        for ne, etf in enumerate(self.event_type):
            self.sine_template[ne] = utils.sine_template(
                freq=self.stim_info[str(etf)][0],
                phase=self.stim_info[str(etf)][1],
                n_points=n_points,
                n_harmonics=self.n_harmonics,
                srate=self.srate
            )
        del ne, etf

        # basic information of source domain
        self.source_info = utils.generate_data_info(X=self.X_source, y=self.y_source)
        self.source_info['init_model'] = init_method
        self.source_info['w_init'] = w_init
        event_type_source = self.source_info['event_type']

        # config events_group information for ms-eCCA model
        if events_group is not None:
            self.source_info['events_group'] = events_group
        else:
            self.source_info['events_group'] = utils.augmented_events(
                event_type=event_type_source,
                d=d
            )

        # config sinusoidal templates & convolution matrices of source domain
        self.sine_template_source, self.H_source, self.H_correct_source = [], [], []
        for ne, et in enumerate(self.event_type):
            if et in list(event_type_source):
                self.sine_template_source.append(self.sine_template[ne])
                H_temp, H_correct_temp = tlcca_conv_matrix(
                    freq=self.stim_info[str(et)][0],
                    phase=self.stim_info[str(et)][1],
                    n_points=n_points,
                    srate=self.srate,
                    rrate=self.rrate,
                    len_scale=self.len_scale,
                    amp_scale=self.amp_scale,
                    extract_method='Square',
                    concat_method='dynamic',
                    response_length=None
                )
                self.H_source.append(H_temp)
                self.H_correct_source.append(H_correct_temp)
        self.sine_template_source = np.stack(self.sine_template_source, axis=0)
        del H_temp, H_correct_temp, ne, et

        # config information of transferred events
        event_type_new = np.array([tp[1] for tp in self.transfer_pair])
        event_type_concat = np.array(list(event_type_source) + list(event_type_new))
        self.sorted_order = [item[0] for item in sorted(
            enumerate(event_type_concat),
            key=lambda x: x[1]
        )]
        del event_type_new, event_type_concat, event_type_source

        # config convolution matrices of transferred events
        self.H_new, self.H_correct_new = [], []
        for (label_src, label_tar) in self.transfer_pair:
            freq_src = self.stim_info[str(label_src)][0]  # frequency of source stimulus
            response_length = int(np.round(self.srate * self.len_scale / freq_src))
            H_temp, H_correct_temp = tlcca_conv_matrix(
                freq=self.stim_info[str(label_tar)][0],
                phase=self.stim_info[str(label_tar)][1],
                n_points=n_points,
                srate=srate,
                rrate=rrate,
                len_scale=len_scale,
                amp_scale=amp_scale,
                extract_method='Square',
                concat_method='dynamic',
                response_length=response_length
            )
            self.H_new.append(H_temp)
            self.H_correct_new.append(H_correct_temp)
        del label_src, label_tar, freq_src, response_length, H_temp, H_correct_temp

        # main process
        self.intra_source_training()
        self.transfer_learning()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
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
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=TLCCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 6. small data least-squares transformation, sd-LST
def sdlst_align_subject(
        X_source: ndarray,
        y_source: ndarray,
        train_info_source: dict,
        train_info_target: dict,
        X_target_mean: ndarray) -> Tuple[ndarray, ndarray]:
    """One-step LST aligning in sd-LST (source subject to target subject).

    Args:
        X_source (ndarray): (Ne(s)*Nt(s),Nc(s),Np). Source dataset.
        y_source (ndarray): (Ne(s)*Nt(s),). Labels for X_source.
        train_info_source (dict): {'n_chans':int}.
        train_info_target (dict): {'event_type':ndarray (Ne,),
                                   'n_events':int,
                                   'n_chans':int,
                                   'n_points':int}.
        X_target_mean (ndarray): (Ne(t),Nc(t),Np).
            Averaged template of X_target.
            Nc(t) may not be equal to Nc(s). Typically Nc(t)<=Nc(s).
            Ne(t) may not be equal to Ne(s). Typically Ne(t)<=Ne(s).

    Returns:
        P_trans (ndarray): (Nc(t),Nc(s)). Subject-level transfer matrix.
        X_source_trans (ndarray): (Ne(s)*Nt(s),Nc(t),Np). Transferred X_source.
    """
    # basic information
    event_type_target = train_info_target['event_type']
    n_events_target = train_info_target['n_events']  # Ne(t)
    n_chans_target = train_info_target['n_chans']  # Nc(t)
    n_chans_source = train_info_source['n_chans']  # Nc(s)
    n_points = train_info_target['n_points']  # Np

    # extract limited event information from source dataset
    X_source_lim_mean = np.zeros((n_events_target, n_chans_source, n_points))
    for nett, ett in enumerate(event_type_target):
        X_temp = X_source[y_source == ett]
        if X_temp.ndim == 2:  # (Nc,Np), Nt=1
            X_source_lim_mean[nett] = X_temp
        elif X_temp.ndim > 2:  # (Nt,Nc,Np)
            X_source_lim_mean[nett] = np.mean(X_temp, axis=0)

    # LST aligning
    part_1 = np.zeros((n_chans_target, n_chans_source))  # (Nc(t),Nc(s))
    part_2 = np.zeros((n_chans_source, n_chans_source))  # (Nc(s),Nc(s))
    for net in range(n_events_target):
        part_1 += X_target_mean[net] @ X_source_lim_mean[net].T
        part_2 += X_source_lim_mean[net] @ X_source_lim_mean[net].T
    P_trans = part_1 @ sLA.inv(part_2)

    # apply projection onto each trial of source dataset (full events)
    # aligned_source = np.einsum('tcp,hc->thp', X_source, P_trans)
    X_source_trans = np.zeros((X_source.shape[0], n_chans_target, X_source.shape[-1]))
    for tt in range(X_source.shape[0]):
        X_source_trans[tt] = P_trans @ X_source[tt]
    return P_trans, X_source_trans


def sdlst_align_dataset(
        X_source: List[ndarray],
        y_source: List[ndarray],
        train_info_source: List[dict],
        train_info_target: dict,
        X_target: ndarray,
        y_target: ndarray) -> dict:
    """Two-step LST algining in sd-LST (source dataset to target dataset).
        Obtain augmented training dataset.

    Args:
        X_source (List[ndarray]): Ns*(Ne(s)*Nt(s),Nc(s),Np). Source-domain dataset.
        y_source (List[ndarray]): Ns*(Ne(s)*Nt(s),). Labels for each X_source.
        train_info_source (List[dict]): {'n_chans':int}.
        train_info_target (dict): {'event_type':ndarray (Ne,),
                                   'n_events':int,
                                   'n_chans':int,
                                   'n_points':int}.
        X_target (ndarray): (Ne(t)*Nt(t),Nc(t),Np). Target-domain dataset. Nt(t)>=1
        y_target (ndarray): (Ne(t)*Nt(t),). Labels for X_target.

    Returns:
        LST-1 (List[ndarray]): Ns*(Nc(t),Nc(s)). One-step LST projection matrices.
        LST-2 (ndarray): (Nc(t),Nc(t)). Two-step LST projection matrix.
        X-LST-1 (ndarray): (Ns*Ne*Nt,Nc(t),Np). Source dataset with one-step LST aligned.
        X-LST-2 (ndarray): (Ns*Ne*Nt,Nc(t),Np). Source dataset with two-step LST aligned.
        y (ndarray): (Ns*Ne*Nt,). Labels for X-LST-1 (and X-LST-2).
        X_target_mean (ndarray): (Ne(t),Nc(t),Np). Trial-averaged X_target.
    """
    # basic information
    event_type_target = train_info_target['event_type']
    n_events_target = train_info_target['n_events']  # Ne(t)
    n_chans_target = train_info_target['n_chans']  # Nc(t)
    n_points = train_info_target['n_points']  # Np

    # obtain averaged template of target training dataset
    X_target_mean = utils.generate_mean(X=X_target, y=y_target)  # (Ne(t),Nc(t),Np)

    # obtain once-aligned signal
    P_trans_1_lst = []
    source_trials = []
    X_align_1 = np.empty((1, n_chans_target, n_points))
    y_align_1 = np.empty((1))
    for nsub in range(len(X_source)):
        P_trans, X_source_trans = sdlst_align_subject(
            X_source=X_source[nsub],
            y_source=y_source[nsub],
            train_info_source=train_info_source[nsub],
            train_info_target=train_info_target,
            X_target_mean=X_target_mean
        )
        source_trials.append(X_source_trans.shape[0])
        P_trans_1_lst.append(P_trans)
        X_align_1 = np.concatenate((X_align_1, X_source_trans), axis=0)
        y_align_1 = np.concatenate((y_align_1, y_source[nsub]))
    X_align_1 = np.delete(X_align_1, 0, axis=0)
    y_align_1 = np.delete(y_align_1, 0, axis=0)

    # apply LST projection again | limited events
    X_align_1_lim_mean = np.zeros((n_events_target, n_chans_target, n_points))
    part_1 = np.zeros((n_chans_target, n_chans_target))  # (Nc(t),Nc(t))
    part_2 = np.zeros_like(part_1)  # (Nc(t),Nc(t))
    for nett, ett in enumerate(event_type_target):
        X_temp = X_align_1[y_align_1 == ett]
        if X_temp.ndim == 2:  # (Nc,Np), Nt=1
            X_align_1_lim_mean[nett] = X_temp
        elif X_temp.ndim > 2:  # (Nt,Nc,Np)
            X_align_1_lim_mean[nett] = np.mean(X_temp, axis=0)
        part_1 += X_target_mean[nett] @ X_align_1_lim_mean[nett].T
        part_2 += X_align_1_lim_mean[nett] @ X_align_1_lim_mean[nett].T
    P_trans_2 = part_1 @ sLA.inv(part_2)  # (Nc,Nc)

    # obtrain twice-aligned signal
    # X_final = np.einsum('tcp,hc->thp', X_lst_1, projection_matrix)
    X_final = np.zeros_like(X_align_1)
    for nt in range(X_final.shape[0]):
        X_final[nt] = P_trans_2 @ X_align_1[nt]

    # source model
    source_model = {
        'LST-1': P_trans_1_lst,
        'LST-2': P_trans_2,
        'X-LST-1': X_align_1,
        'X-LST-2': X_final,
        'y': y_align_1,
        'X_target_mean': X_target_mean,
        'source_trials': source_trials
    }
    return source_model


class SDLST(BasicTransfer):
    def transfer_learning(self):
        """Transfer learning process."""
        source_model = sdlst_align_dataset(
            X_source=self.X_source,
            y_source=self.y_source,
            train_info_source=self.train_info_source,
            train_info_target=self.train_info_target,
            X_target=self.X_train,
            y_target=self.y_train
        )
        self.X_trans, self.y_trans = source_model['X-LST-2'], source_model['y']

    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        # LST-TRCA
        self.trca_model = trca.TRCA(standard=True, ensemble=True)
        self.trca_model.fit(
            X_train=self.X_trans,
            y_train=self.y_trans
        )

        # LST-CCA
        self.ecca_model = cca.ECCA()
        self.ecca_model.fit(
            X_train=self.X_trans,
            y_train=self.y_trans,
            sine_template=self.sine_template,
            method_list=self.method_list
        )

    def fit(
            self,
            X_source: List[ndarray],
            y_source: List[ndarray],
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray,
            method_list: List[str] = ['1', '2', '3', '4', '5']):
        """Train sd-LST model.

        Args:
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
            method_list (List[str]): See details in cca.ecca_feature()
        """
        # load in data
        self.X_source = X_source
        self.y_source = y_source
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.method_list = method_list

        # basic information of source domain
        self.n_subjects = len(self.X_source)
        self.train_info_source = []
        for nsub in range(self.n_subjects):
            event_type_source = np.unique(self.y_source[nsub])
            self.train_info_source.append({
                'event_type': event_type_source,
                'n_events': event_type_source.shape[0],
                'n_train': np.array([np.sum(self.y_source[nsub] == ets)
                                     for ets in event_type_source]),
                'n_chans': self.X_source[nsub].shape[-2],
                'n_points': self.X_source[nsub].shape[-1],
            })

        # basic information of target domain
        event_type_target = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        self.train_info_target = {
            'event_type': event_type_target,
            'n_events': event_type_target.shape[0],
            'n_train': np.array([np.sum(self.y_train == ett)
                                 for ett in event_type_target]),
            'n_chans': self.X_train.shape[-2],
            'n_points': self.X_train.shape[-1],
        }

        # main process
        self.transfer_learning()
        self.intra_target_training()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho_ecca (ndarray): (Ne*Nte,Ne). Features from LST-CCA.
            rho_etrca (ndarray): (Ne*Nte,Ne). Features from LST-TRCA.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        rho_etrca = self.trca_model.transform(X_test=X_test)['erho']
        rho_ecca = self.ecca_model.transform(X_test=X_test)['rho']
        rho = utils.combine_feature(features=[rho_ecca / 3, rho_etrca])
        return {'rho_ecca': rho_ecca, 'rho_etrca': rho_etrca, 'rho': rho}


class FB_SDLST(BasicFBTransfer):
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
            base_estimator=SDLST(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 7. cross-subject transfer method based on domain generalization
class TNSRE_20233305202(BasicTransfer):
    pass


class FB_TNSRE_20233305202(BasicFBTransfer):
    pass


# %% 8. group TRCA | gTRCA


# %% 9. inter- and intra-subject maximal correlation | IISMC
def iismc_feature(
        X_test: ndarray,
        trans_model: Dict[str, ndarray],
        target_model: Dict[str, ndarray],
        standard: bool = True,
        ensemble: bool = True) -> Dict[str, ndarray]:
    """The pattern matching process of IISMC.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        trans_model (dict): {'u_trans': ndarray (Ns,Ne,Nk,Nc),
                             'uX_trans': ndarray (Ns,Ne,Nk*Np),
                             'vY_trans': ndarray (Ns,Ne,Nk*Np),
                             'uY_trans': ndarray (Ns,Ne,Nk*Np),
                             'eu_trans': ndarray (Ns,Ne,Ne*Nk,Nc),
                             'euX_trans': ndarray (Ns,Ne,Ne*Nk*Np),
                             'evY_trans': ndarray (Ns,Ne,Ne*Nk*Np),
                             'euY_trans': ndarray (Ns,Ne,Ne*Nk*Np)}
            See details in IISMC.tranfer_learning().
        target_model (dict): {'w': ndarray (Ne,Nk,Nc) | (filter 'v'),
                              'ew': ndarray (Ne*Nk,Nc) | (filter 'ev')
                              'wX': ndarray (Ne,Nk*Np) | (template 'vX'),
                              'ewX': ndarray (Ne,Ne*Nk*Np) | (template 'evX')}
            See details in IISMC.intra_target_training().
        standard (bool): Standard model. Defaults to True.
        ensemble (bool): Ensemble model. Defaults to True.

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        erho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features (ensemble).
        erho (ndarray): (Ne*Nte,Ne). Intergrated features (ensemble).
    """
    # load in models & basic information
    u_trans, v = trans_model['u_trans'], target_model['w']
    uX_trans, vX = trans_model['uX_trans'], target_model['wX']
    vY_trans, uY_trans = trans_model['vY_trans'], trans_model['uY_trans']

    eu_trans, ev = trans_model['eu_trans'], target_model['ew']
    euX_trans, evX = trans_model['euX_trans'], target_model['ewX']
    evY_trans, euY_trans = trans_model['evY_trans'], trans_model['euY_trans']

    n_subjects = u_trans.shape[0]  # Ns
    n_events = u_trans.shape[1]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    # n_points = X_test.shape[-1]  # Np

    # 4-part discriminant coefficients: standard & ensemble
    rho_temp = np.zeros((n_test, n_events, 4))  # (Ne*Nte,Ne,4)
    if standard:
        for nte in range(n_test):
            vX_temp = np.reshape(
                a=utils.fast_stan_2d(v @ X_test[nte]),
                newshape=(n_events, -1),
                order='C'
            )  # (Ne,Nk,Nc) @ (Nc,Np) -reshape-> (Ne,Nk*Np)
            uX_temp = np.reshape(
                a=utils.fast_stan_3d(u_trans @ X_test[nte]),
                newshape=(n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Nk,Nc) @ (Nc,Np) -reshape-> (Ns,Ne,Nk*Np)

            rho_temp[nte, :, 0] = utils.fast_corr_2d(
                X=vX_temp,
                Y=vX
            )  # vX & vX: (Ne,Nk*Np) -corr-> (Ne,)
            rho_temp[nte, :, 1] = np.mean(
                a=utils.fast_corr_3d(X=uX_temp, Y=uX_trans),
                axis=0
            )  # uX & uX: (Ns,Ne,Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
            rho_temp[nte, :, 2] = np.mean(
                a=utils.fast_corr_3d(
                    X=np.tile(A=vX_temp, reps=(n_subjects, 1, 1)),
                    Y=vY_trans
                ),
                axis=0
            )  # vX & vY: (Ne,Nk*Np) -tile-> (Ns,Ne,Nk*Np) -corr-> (Ns,Ne,) -mean-> (Ne,)
            rho_temp[nte, :, 3] = np.mean(
                a=utils.fast_corr_3d(X=uX_temp, Y=uY_trans),
                axis=0
            )  # uX & uY: (Ns,Ne,Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
        # rho_temp /= n_points  # real Pearson correlation coefficients in scale

    erho_temp = np.zeros_like(rho_temp)  # (Ne*Nte,Ne,4)
    if ensemble:
        for nte in range(n_test):
            evX_temp = np.reshape(
                a=utils.fast_stan_2d(ev @ X_test[nte]),
                newshape=-1,
                order='C'
            )  # (Ne*Nk,Nc) @ (Nc,Np) -reshape-> (Ne*Nk*Np,)
            euX_temp = np.reshape(
                a=utils.fast_stan_3d(eu_trans @ X_test[nte]),
                newshape=(n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Ne*Nk,Nc) @ (Nc,Np) -reshape-> (Ns,Ne,Ne*Nk*Np)

            erho_temp[nte, :, 0] = utils.fast_corr_2d(
                X=np.tile(A=evX_temp, reps=(n_events, 1)),
                Y=evX
            )  # evX & evX: (Ne*Nk*Np,) -tile-> (Ne,Ne*Nk*Np) -corr-> (Ne,)
            erho_temp[nte, :, 1] = np.mean(
                a=utils.fast_corr_3d(X=euX_temp, Y=euX_trans),
                axis=0
            )  # euX & euX: (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
            erho_temp[nte, :, 2] = np.mean(
                a=utils.fast_corr_3d(
                    X=np.tile(A=evX_temp, reps=(n_subjects, n_events, 1)),
                    Y=evY_trans
                ),
                axis=0
            )  # evX & evY: (Ne*Nk*Np,) -tile-> (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
            erho_temp[nte, :, 3] = np.mean(
                a=utils.fast_corr_3d(X=euX_temp, Y=euY_trans),
                axis=0
            )  # euX & euY: (Ns,Ne,Ne*Nk*Np) -corr-> (Ns,Ne) -mean-> (Ne,)
        # erho_temp /= n_points  # real Pearson correlation coefficients in scale
    features = {
        'rho_temp': rho_temp,
        'rho': utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1],
            rho_temp[..., 2],
            rho_temp[..., 3]
        ]),
        'erho_temp': erho_temp,
        'erho': utils.combine_feature([
            erho_temp[..., 0],
            erho_temp[..., 1],
            erho_temp[..., 2],
            erho_temp[..., 3]
        ])
    }
    return features


class IISMC(BasicTransfer):
    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        self.training_model_target = trca.trca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            standard=self.standard,
            ensemble=self.ensemble,
            n_components=self.n_components
        )

    def transfer_learning(self):
        """Transfer learning process."""
        # basic information
        event_type = self.train_info_target['event_type']
        n_events = len(event_type)  # Ne
        n_chans = self.train_info_target['n_chans']  # Nc
        n_points = self.train_info_target['n_points']  # Np
        X_target_mean = utils.generate_mean(X=self.X_train, y=self.y_train)  # (Ne,Nc,Np)

        # initialization
        self.trans_model = {}
        Cxx = self.training_model_target['Q']  # (Ne,Nc,Nc)
        Cxy = np.tile(np.zeros_like(Cxx), (self.n_subjects, 1, 1, 1))  # (Ns,Ne,Nc,Nc)
        Cyy = np.zeros_like(Cxy)  # (Ns,Ne,Nc,Nc)

        # standard version
        v = self.training_model_target['w']  # (Ne,Nk,Nc)
        u_trans = np.zeros((self.n_subjects, n_events, self.n_components, n_chans))
        uX_trans = np.zeros((self.n_subjects, n_events, self.n_components, n_points))
        vY_trans, uY_trans = np.zeros_like(uX_trans), np.zeros_like(uX_trans)

        # ensemble version
        ev = self.training_model_target['ew']  # (Ne*Nk,Nc)
        eu_trans = np.reshape(
            a=u_trans,
            newshape=(self.n_subjects, n_events * self.n_components, n_chans),
            order='C'
        )  # (Ns,Ne*Nk,Nc)
        euX_trans = np.zeros((self.n_subjects, n_events, eu_trans.shape[1], n_points))
        evY_trans, euY_trans = np.zeros_like(euX_trans), np.zeros_like(euX_trans)

        # obtain transfer model
        for nsub in range(self.n_subjects):
            X_source, y_source = self.X_source[nsub], self.y_source[nsub]
            n_train = self.train_info_source[nsub]['n_train']
            X_source_mean = np.zeros_like(X_target_mean)  # (Ne,Nc,Np)
            for ne, et in enumerate(event_type):
                source_trials = n_train[ne]
                assert source_trials > 1, 'The number of training samples is too small!'

                X_source_temp = X_source[y_source == et]  # (Nt,Nc,Np)
                X_source_mean[ne] = np.mean(X_source_temp, axis=0)  # (Nc,Np)
                Cxy[nsub, ne, ...] = X_target_mean[ne] @ X_source_mean[ne].T  # (Nc,Nc)
                for st in range(source_trials):
                    Cyy[nsub, ne, ] += X_source_temp[st] @ X_source_temp[st].T

                # obtain transferred spatial filters
                u_trans[nsub, ne, ...] = utils.solve_gep(
                    A=Cxy[nsub, ne, ...] + Cxy[nsub, ne, ...].T,
                    B=Cxx[ne] + Cyy[nsub, ne, ...],
                    n_components=self.n_components
                )  # standard | (Nk,Nc)

                # obtain standard transferred templates: (Ns,Ne,Nk,Np)
                uX_trans[nsub, ne, ...] = u_trans[nsub, ne, ...] @ X_target_mean[ne]
                vY_trans[nsub, ne, ...] = v[ne] @ X_source_mean[ne]
                uY_trans[nsub, ne, ...] = u_trans[nsub, ne, ...] @ X_source_mean[ne]

            # obtain ensembled transferred model: spatial filters & templates
            eu_trans[nsub] = np.reshape(
                a=u_trans[nsub],
                newshape=(n_events * self.n_components, n_chans),
                order='C'
            )  # (Ne*Nk,Nc)
            for ne in range(n_events):
                euX_trans[nsub, ne, ...] = eu_trans[nsub] @ X_target_mean[ne]
                evY_trans[nsub, ne, ...] = ev @ X_source_mean[ne]
                euY_trans[nsub, ne, ...] = eu_trans[nsub] @ X_source_mean[ne]

        # standardize & reshape
        self.trans_model['u_trans'] = u_trans  # filter: (Ns,Ne,Nk,Nc)
        self.trans_model['eu_trans'] = eu_trans  # filter: (Ns,Ne,Ne*Nk,Nc)
        self.trans_model['uX_trans'] = np.reshape(
            a=utils.fast_stan_4d(uX_trans),
            newshape=(self.n_subjects, n_events, self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Nk*Np)
        self.trans_model['vY_trans'] = np.reshape(
            a=utils.fast_stan_4d(vY_trans),
            newshape=(self.n_subjects, n_events, self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Nk*Np)
        self.trans_model['uY_trans'] = np.reshape(
            a=utils.fast_stan_4d(uY_trans),
            newshape=(self.n_subjects, n_events, self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Nk*Np)
        self.trans_model['euX_trans'] = np.reshape(
            a=utils.fast_stan_4d(euX_trans),
            newshape=(self.n_subjects, n_events, n_events * self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Ne*Nk*Np)
        self.trans_model['evY_trans'] = np.reshape(
            a=utils.fast_stan_4d(evY_trans),
            newshape=(self.n_subjects, n_events, n_events * self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Ne*Nk*Np)
        self.trans_model['euY_trans'] = np.reshape(
            a=utils.fast_stan_4d(euY_trans),
            newshape=(self.n_subjects, n_events, n_events * self.n_components * n_points),
            order='C'
        )  # template: (Ns,Ne,Ne*Nk*Np)

    def fit(
            self,
            X_source: List[ndarray],
            y_source: List[ndarray],
            X_train: ndarray,
            y_train: ndarray):
        """Load data and train classification models.

        Args:
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source

        # basic information of source domain
        self.n_subjects = len(self.X_source)
        self.train_info_source = []
        for nsub in range(self.n_subjects):
            event_type_source = np.unique(self.y_source[nsub])
            self.train_info_source.append({
                'event_type': event_type_source,
                'n_events': event_type_source.shape[0],
                'n_train': np.array([np.sum(self.y_source[nsub] == et)
                                     for et in event_type_source]),
                'n_chans': self.X_source[nsub].shape[-2],
                'n_points': self.X_source[nsub].shape[-1],
                'standard': True,
                'ensemble': True
            })

        # basic information of target domain
        event_type_target = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        self.train_info_target = {
            'event_type': event_type_target,
            'n_events': event_type_target.shape[0],
            'n_train': np.array([np.sum(self.y_train == et)
                                 for et in event_type_target]),
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

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
            erho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features (ensemble).
            erho (ndarray): (Ne*Nte,Ne). Intergrated features (ensemble).
        """
        return iismc_feature(
            X_test=X_test,
            trans_model=self.trans_model,
            target_model=self.training_model_target,
            standard=self.standard,
            ensemble=self.ensemble
        )


class FB_IISMC(BasicFBTransfer):
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
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=IISMC(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 10. CCA with intra- & inter-subject EEG (ACC-based subject selection) | ASS-IISCCA
def standardize_data_size(
        X_source: List[ndarray],
        y_source: List[ndarray],
        X_train: ndarray,
        y_train: ndarray) -> Dict[str, ndarray]:
    """Standardize the size of X_train & X_source.

    Args:
        X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
        y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.

    Returns:
        X_source (ndarray): (Ne*Nt,Nc,Np). Subject-averaged & unifed X_source.
        y_source (ndarray): (Ne*Nt,). Labels for X_source.
        X_train (ndarray): (Ne*Nt,Nc,Np). Unified X_train.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
    """
    # reshape sklearn-style dataset to public-style dataset
    X_train_reshaped = utils.reshape_dataset(
        data=X_train,
        labels=y_train,
        target_style='public'
    )  # (Ne,Nt,Nc,Np)
    X_source_reshaped, n_trials_source = [], []
    for nsub, Xs in enumerate(X_source):
        X_temp = utils.reshape_dataset(
            data=Xs,
            labels=y_source[nsub],
            target_style='public'
        )  # (Ne,Nt,Nc,Np)
        X_source_reshaped.append(X_temp)
        n_trials_source.append(X_temp.shape[1])

    # align the size of all dataset
    n_trials_min = min(np.min(n_trials_source), X_train_reshaped.shape[1])
    X_train_reshaped = X_train_reshaped[:, :n_trials_min, ...]
    for Xsr in X_source_reshaped:
        Xsr = Xsr[:, :n_trials_min, ...]
    X_source_reshaped = np.stack(X_source_reshaped)  # (Ns,Ne,Nt,Nc,Np)
    X_sub, y_sub = utils.reshape_dataset(data=np.mean(X_source_reshaped, axis=0))
    X_train, y_train = utils.reshape_dataset(data=X_train_reshaped)
    unified_dataset = {
        'X_source': X_sub, 'y_source': y_sub,
        'X_train': X_train, 'y_train': y_train
    }
    return unified_dataset


def inter_subject_cca_kernel(
        X_source: ndarray,
        y_source: ndarray,
        X_train: ndarray,
        y_train: ndarray,
        n_components: int = 1) -> Dict[str, ndarray]:
    """The inter-subject CCA process. (Just like cca.tdcca_kernel())

    Args:
        X_source (ndarray): (Ne*Nt,Nc,Np). Subject-averaged & unifed X_source.
            See details in standardize_data_size().
        y_source (ndarray): (Ne*Nt,). Labels for X_source.
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.

    Returns:
        w (ndarray): (Ne,Nk,Nc). Spatial filter.
    """
    # basic information
    event_type = np.unique(y_train)
    n_events = event_type.shape[0]  # Ne
    n_chans = X_train.shape[1]  # Nc

    # solve target functions
    Cxy = np.zeros((n_events, n_chans, n_chans))
    Cxx, Cyy = np.zeros_like(Cxy), np.zeros_like(Cxy)
    for ne, et in enumerate(event_type):
        X_temp = X_train[y_train == et]  # (Nt,Nc,Np)
        Y_temp = X_source[y_source == et]  # (Nt,Nc,Np)
        n_trials = X_temp.shape[0]
        for nt in range(n_trials):
            Cxx_temp, Cxy_temp, Cyy_temp = cca.generate_cca_mat(
                X=X_temp[nt],
                Y=Y_temp[nt]
            )
            Cxx[ne] += Cxx_temp
            Cxy[ne] += Cxy_temp
            Cyy[ne] += Cyy_temp
    w = np.zero((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    for ne in range(n_events):
        w[ne] = cca.solve_cca_func(
            Cxx=Cxx,
            Cxy=Cxy,
            Cyy=Cyy,
            mode=['X'],
            n_components=n_components
        )
    return {'w': w}


def iiscca_feature(
        X_test: ndarray,
        iiscca_model: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """The pattern matching process of IISCCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        tdcca_model (dict): See details in iiscca_kernel().

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,6). 6-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # basic information
    w1, w2, w3 = iiscca_model['w1'], iiscca_model['w2'], iiscca_model['w3']
    n_events = w1.shape[0]  # Ne
    w1_Xt = np.reshape(iiscca_model['w1_Xt'], (n_events, -1), 'C')  # (Ne,Nk*Np)
    w1_Xs = np.reshape(iiscca_model['w1_Xs'], (n_events, -1), 'C')  # (Ne,Nk*Np)
    w2_Xt = np.reshape(iiscca_model['w2_Xt'], (n_events, -1), 'C')  # (Ne,Nk*Np)
    w2_Xs = np.reshape(iiscca_model['w2_Xs'], (n_events, -1), 'C')  # (Ne,Nk*Np)
    w3_Xt = np.reshape(iiscca_model['w3_Xt'], (n_events, -1), 'C')  # (Ne,Nk*Np)
    w3_Xs = np.reshape(iiscca_model['w3_Xs'], (n_events, -1), 'C')  # (Ne,Nk*Np)
    n_test = X_test.shape[0]  # Ne*Nte
    # n_points = X_test.shape[-1]  # Np, unnecessary

    # 6-D features
    rho_temp = np.zeros((n_test, n_events, 6))  # (Ne*Nte,Ne,6)
    for nte in range(n_test):
        # spatial filtering & reshape
        X_temp_1 = np.reshape(
            a=utils.fast_stan_3d(w1 @ X_test[nte]),
            newshape=(n_events, -1),
            order='C'
        )  # (Ne,Nk,Np) -> (Ne,Nk*Np)
        X_temp_2 = np.reshape(
            a=utils.fast_stan_3d(w2 @ X_test[nte]),
            newshape=(n_events, -1),
            order='C'
        )  # (Ne,Nk,Np) -> (Ne,Nk*Np)
        X_temp_3 = np.reshape(
            a=utils.fast_stan_3d(w3 @ X_test[nte]),
            newshape=(n_events, -1),
            order='C'
        )  # (Ne,Nk,Np) -> (Ne,Nk*Np)

        # pattern matching
        rho_temp[nte, :, 0] = utils.fast_corr_2d(X=X_temp_1, Y=w1_Xt)
        rho_temp[nte, :, 1] = utils.fast_corr_2d(X=X_temp_1, Y=w1_Xs)
        rho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp_2, Y=w2_Xt)
        rho_temp[nte, :, 3] = utils.fast_corr_2d(X=X_temp_2, Y=w2_Xs)
        rho_temp[nte, :, 4] = utils.fast_corr_2d(X=X_temp_3, Y=w3_Xt)
        rho_temp[nte, :, 5] = utils.fast_corr_2d(X=X_temp_3, Y=w3_Xs)
    # rho_temp /= n_points  # real Pearson correlation coefficients in scale
    features = {
        'rho_temp': rho_temp,
        'rho': utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1],
            rho_temp[..., 2],
            rho_temp[..., 3],
            rho_temp[..., 4],
            rho_temp[..., 5]
        ])
    }
    return features


class TDCCA_ASS(BasicASS):
    def evaluation_1st(self):
        """Calculate the cross-subject classification accuracy for each source subject."""
        # basic information
        self.n_subjects = len(self.X_source)  # Ns

        # apply TDCCA classification
        self.acc_list = np.zeros((self.n_subjects))
        for nsub in range(self.n_subjects):
            model = cca.TDCCA()
            model.fit(X_train=self.X_source[nsub], y_train=self.y_source[nsub])
            y_pred = model.predict(X_test=self.X_target)
            self.acc_list[nsub] = utils.acc_compute(y_true=self.y_target, y_pred=y_pred)

    def evaluation_2nd(self):
        """Find the best source subjects."""
        # basic information
        n_chans = self.X_target.shape[-2]  # Nc
        n_points = self.X_target.shape[-1]  # Np

        # forward iteration
        self.iter_list = np.zeros((self.n_subjects))
        for nsub in range(self.n_subjects):
            X_list = [self.X_source[:nsub + 1]]
            y_list = [self.y_source[:nsub + 1]]
            X_source_temp = np.zeros((1, n_chans, n_points))
            y_source_temp = np.zeros((1))
            for nx in len(X_list):
                X_source_temp = np.concatenate((X_source_temp, X_list[nx]), axis=0)
                y_source_temp = np.concatenate((y_source_temp, y_list[nx]), axis=0)
            X_source_temp = np.delete(X_source_temp, 0, axis=0)  # (Ns*Ne*Nt,Nc,Np)
            y_source_temp = np.delete(y_source_temp, 0, axis=0)  # (Ns*Ne*Nt,)
            model = cca.TDCCA()
            model.fit(X_train=X_source_temp, y_train=y_source_temp)
            y_pred = model.predict(X_test=self.X_target)
            self.iter_list[nsub] = utils.acc_compute(y_true=self.y_target, y_pred=y_pred)

        # select best subjects
        self.max_idx = np.argmax(self.iter_list)

    def select_subjects(self) -> List[int]:
        """Main process.

        Returns:
            subject_indices (List[int]).
        """
        self.evaluation_1st()
        self.sort_subject_list()
        self.evaluation_2nd()
        return self.sorted_idx[:self.max_idx + 1]


class ASS_IISCCA(BasicTransfer):
    def intra_source_training(self):
        """Intra-subject CCA for source dataset."""
        self.source_intra_model = cca.tdcca_kernel(
            X_train=self.X_source,
            y_train=self.y_source,
            n_components=self.n_components
        )

    def intra_target_training(self):
        """Intra-subject CCA for target dataset."""
        self.target_intra_model = cca.tdcca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            n_components=self.n_components
        )

    def transfer_learning(self):
        """Inter-subject CCA & signal templates."""
        # spatial filters: (Ne,Nk,Nc)
        w1, w2 = self.target_intra_model['w'], self.source_intra_model['w']
        w3 = inter_subject_cca_kernel(
            X_source=self.X_source,
            y_source=self.y_source,
            X_train=self.X_train,
            y_train=self.y_train,
            n_components=self.n_components
        )

        # filtered templates: (Ne,Nk,Np)
        w1_Xt = self.target_intra_model['wX']
        w2_Xs = self.source_intra_model['wX']
        w1_Xs = utils.generate_mean(
            X=utils.generate_source_response(X=self.X_source, y=self.y_source, w=w1),
            y=self.y_source
        )
        w2_Xt = utils.generate_mean(
            X=utils.generate_source_response(X=self.X_train, y=self.y_train, w=w2),
            y=self.y_train
        )
        w3_Xt = utils.generate_mean(
            X=utils.generate_source_response(X=self.X_train, y=self.y_train, w=w3),
            y=self.y_train
        )
        w3_Xs = utils.generate_mean(
            X=utils.generate_source_response(X=self.X_source, y=self.y_source, w=w3),
            y=self.y_source
        )
        self.trans_model = {
            'w1': w1, 'w2': w2, 'w3': w3,
            'w1_Xt': utils.fast_stan_3d(w1_Xt), 'w1_Xs': utils.fast_stan_3d(w1_Xs),
            'w2_Xt': utils.fast_stan_3d(w2_Xt), 'w2_Xs': utils.fast_stan_3d(w2_Xs),
            'w3_Xt': utils.fast_stan_3d(w3_Xt), 'w3_Xs': utils.fast_stan_3d(w3_Xs)
        }

    def fit(
            self,
            X_source: List[ndarray],
            y_source: List[ndarray],
            X_train: ndarray,
            y_train: ndarray,
            selection: Optional[List[int]] = None):
        """Load data and train ASS-IISCCA models.

        Args:
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            X_train (ndarray): (Ne*Nt,Nc,Np). Target training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            selection (optional, List[int]): If None, use TDCCA_ASS() to select subjects.
        """
        # select best source subjects
        if selection is not None:
            self.selection = selection
        else:
            ass = TDCCA_ASS(
                X_source=X_source,
                y_source=y_source,
                X_target=X_train,
                y_target=y_train,
                n_components=self.n_components
            )
            self.selection = ass.select_subjects()

        # preprocess & load in data
        unified_dataset = standardize_data_size(
            X_source=X_source[self.selection],
            y_source=y_source[self.selection],
            X_train=X_train,
            y_train=y_train
        )
        self.X_source = unified_dataset['X_source']
        self.y_source = unified_dataset['y_source']
        self.X_train = unified_dataset['X_train']
        self.y_train = unified_dataset['y_train']

        # main process
        self.intra_source_training()
        self.intra_target_training()
        self.transfer_learning()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,6). 6-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return iiscca_feature(
            X_test=X_test,
            iiscca_model=self.trans_model
        )


class FB_ASS_IISCCA(BasicFBTransfer):
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
            base_estimator=ASS_IISCCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 11. LST-based cross-domain transfer learning | LST-TRCA
class LST_TRCA(BasicTransfer):
    def transfer_learning(self):
        """Transfer learning process."""
        # basic information
        n_trials = self.X_source.shape[0]  # Ne*Nt(s)
        n_chans_source = self.X_source.shape[-2]  # Nc(s)
        n_chans_target = self.X_train.shape[-2]  # Nc(t)
        n_points = self.X_source.shape[-1]  # Np

        # analytical solution
        self.P_trans = np.zeros((n_trials, n_chans_source, n_chans_target))
        self.X_trans = np.zeros((n_trials, n_chans_target, n_points))
        X_target_mean = utils.generate_mean(X=self.X_train, y=self.y_train)
        for ntr in range(n_trials):
            event_idx = list(self.event_type).index(self.y_source[ntr])
            X_tar = X_target_mean[event_idx]  # (Nc(t),Np)
            X_sou = self.X_source[ntr]  # (Nc(s),Np)

            # transfer process: P_trans @ X_sou = X_tar
            # solution: P_trans = X_tar @ X_sou.T @ (X_sou @ X_sou.T)^{-1}
            self.P_trans[ntr] = X_tar @ X_sou.T @ sLA.inv(X_sou @ X_sou.T)  # (Nc(t),Nc(s))
            self.X_trans[ntr] = self.P_trans[ntr] @ X_sou

    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        # solve TRCA target functions with X_train & X_trans
        Q, S, X_mean = trca.generate_trca_mat(
            X=np.concatenate((self.X_train, self.X_trans), axis=0),
            y=np.concatenate((self.y_train, self.y_source), axis=0)
        )
        w, ew = trca.solve_trca_func(Q=Q, S=S, n_components=self.n_components)

        # generate spatial-filtered templates
        wX, ewX = trca.generate_trca_template(
            X_mean=X_mean,
            w=w,
            ew=ew,
            standard=self.standard,
            ensemble=self.ensemble
        )

        # (e)TRCA model
        self.training_model = {
            'Q': Q, 'S': S,
            'w': w, 'ew': ew,
            'wX': wX, 'ewX': ewX
        }

    def fit(
            self,
            X_source: ndarray,
            y_source: ndarray,
            X_train: ndarray,
            y_train: ndarray):
        """Train LST-TRCA model.

        Args:
            X_source (ndarray): (Ne*Nt(s),Nc(s),Np). Source dataset.
            y_source (ndarray): (Ne*Nt(s),). Labels for X_source.
            X_train (ndarray): (Ne*Nt(t),Nc(t),Np). Target training dataset. Nt(t)>=2.
            y_train (ndarray): (Ne*Nt(t),). Labels for X_train.
        """
        # load in data
        self.X_source = X_source
        self.y_source = y_source
        self.X_train = X_train
        self.y_train = y_train
        self.event_type = np.unique(self.y_source)

        # basic information
        self.train_info_target = {'event_type': self.event_type}

        # main process
        self.transfer_learning()
        self.intra_target_training()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc(t),Np). Test dataset.

        Returns:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of TRCA.
            erho (ndarray): (Ne*Nte,Ne). Ensemble decision coefficients of eTRCA.
        """
        return trca.trca_feature(
            X_test=X_test,
            trca_model=self.training_model,
            standard=self.standard,
            ensemble=self.ensemble
        )

    def predict(
            self,
            X_test: ndarray) -> Union[
                Tuple[ndarray, ndarray],
                Tuple[int, int]]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            y_standard (ndarray or int): (Ne*Nte,). Predict label(s).
            y_ensemble (ndarray or int): (Ne*Nte,). Predict label(s) (ensemble).
        """
        self.features = self.transform(X_test)
        event_type = self.train_info_target['event_type']
        self.y_standard = event_type[np.argmax(self.features['rho'], axis=-1)]
        self.y_ensemble = event_type[np.argmax(self.features['erho'], axis=-1)]
        return self.y_standard, self.y_ensemble


class FB_LST_TRCA(BasicFBTransfer):
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
            base_estimator=LST_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 12. Align and Pool for EEG Headset Domain Adaptation | ALPHA
def alpha_source_decomposition(
        X_source: ndarray,
        y_source: ndarray,
        sine_template: ndarray,
        n_components: int = 1) -> Dict[str, ndarray]:
    """Matrix decompostion process of ALPHA based on source dataset.

    Args:
        X_source (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_source (ndarray): (Ne*Nt,). Labels for X_source.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Defaults to 1.

    Returns:
        u_2 (ndarray): (Ne,Nk,Nc). Basic spatial filters for source dataset.
            See details in refer[11] blog, eq (4).
        u_3 (ndarray): (Ne,Nk,Nc). Basic spatial filters for source dataset.
            See details in refer[11] blog, eq (5).
        w (ndarray): (Nk,Nc). DSP spatial filter for source dataset.
            See details in refer[11] blog, eq (7).
        X_source_mean (ndarray): (Ne,Nc,Np). Trial-averaged X_source.
        uX_2 (ndarray): (Ne,Nk,Np). The results of X_source_mean filtered by u_2.
        uX_3 (ndarray): (Ne,Nk,Np). The results of X_source_mean filtered by u_3.
        wX (ndarray): (Ne,Nk,Np). The results of X_source_mean filtered by w.
    """
    # basic information
    X_source_mean = utils.generate_mean(X=X_source, y=y_source)  # (Ne,Nc,Np)
    n_events = X_source_mean.shape[0]  # Ne
    n_chans = X_source_mean.shape[1]  # Nc
    n_points = X_source_mean.shape[-1]  # Np

    # 2nd model (CCA-like)
    u_2 = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    uX_2 = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    for ne in range(n_events):
        cca_model = cca.cca_kernel(
            X=X_source_mean[ne],
            Y=sine_template[ne],
            n_components=n_components,
            check_direc=False
        )['u']
        u_2[ne] = cca_model['u']  # (Nk,Nc)
        uX_2[ne] = cca_model['uX']  # (Nk,Np)
        del cca_model

    # 3rd model (TDCCA)
    tdcca_model = cca.tdcca_kernel(
        X_train=X_source,
        y_train=y_source,
        n_components=n_components
    )['w']  # (Ne,Nk,Nc)
    u_3 = tdcca_model['w']
    uX_3 = tdcca_model['wX']

    # 5th model (DSP)
    dsp_model = dsp.dsp_kernel(
        X_train=X_source,
        y_train=y_source,
        n_components=n_components
    )
    w = dsp_model['w']  # (Nk,Nc)
    wX = dsp_model['wX']  # (Ne,Nk,Np)

    return {
        'u_2': u_2, 'u_3': u_3, 'w': w,
        'X_source_mean': X_source_mean,
        'uX_2': uX_2, 'uX_3': uX_3, 'wX': wX
    }


def alpha_target_decomposition(
        X_source_mean: ndarray,
        X_test: ndarray,
        sine_template: ndarray,
        n_components: int = 1) -> Dict[str, ndarray]:
    """Matrix decompostion process of ALPHA based on target dataset.

    Args:
        X_source_mean (ndarray): (Ne,Nc,Np). Trial-averaged X_source.
        X_test (ndarray): (Nc,Np). Test dataset.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Defaults to 1.

    Returns:
        u_1 (ndarray): (Ne,Nk,Nc). Basic spatial filters.
            See details in refer[11] blog, eq (3).
        v_1 (ndarray): (Ne,Nk,Nc). Basic spatial filters.
            See details in refer[11] blog, eq (3).
        u_4 (ndarray): (Ne,Nk,Nc). Basic spatial filters.
            See details in refer[11] blog, eq (6).
        uX_1 (ndarray): (Ne,Nk,Np). The results of X_test filtered by u_1.
        vY_1 (ndarray): (Ne,Nk,Np). The results of sine_template filtered by v_1.
    """
    # basic information
    n_events = sine_template.shape[0]  # Ne
    n_dims = sine_template.shape[1]  # 2Nh
    n_chans = X_test.shape[0]  # Nc
    n_points = X_test.shape[-1]  # Np

    # 1st model (CCA)
    u_1 = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    v_1 = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,2Nh)
    uX_1 = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    vY_1 = np.zeros_like(uX_1)
    for ne in range(n_events):
        cca_model = cca.cca_kernel(
            X=X_test,
            Y=sine_template[ne],
            n_components=n_components,
            check_direc=True
        )
        u_1[ne] = cca_model['u']
        v_1[ne] = cca_model['v']
        uX_1[ne] = cca_model['uX']
        vY_1[ne] = cca_model['vY']
        del cca_model

    # 4th model (ttCCA)
    u_4 = np.zeros_like(u_1)  # (Ne,Nk,Nc)
    for ne in range(n_events):
        u_4[ne] = cca.cca_kernel(
            X=X_test,
            Y=X_source_mean[ne],
            n_components=n_components,
            check_direc=False
        )['u']

    return {
        'u_1': u_1, 'v_1': v_1, 'u_4': u_4,
        'uX_1': uX_1, 'vY_1': vY_1
    }


def align_spatial_pattern(
        X_source: ndarray,
        y_source: ndarray,
        X_target: ndarray,
        w_source: ndarray,
        w_target: ndarray) -> Tuple[ndarray, ndarray]:
    """Align spatial pattern (ASP) process.

    Args:
        X_source (ndarray): (Ne*Nt,Nc,Np). Source dataset.
        y_source (ndarray): (Ne*Nt,). Labels for X_source.
        X_target (ndarray): (Nc,Np). Single-trial test data.
        w_source (ndarray): (Ne,Nk,Nc). Spatial filters of source domain.
        w_target (ndarray): (Ne,Nk,Nc). Spatial filters of target domain.

    Returns:
        P (ndarray): (Ne,Nk,Nk). Transformation matrices.
        w_asp (ndarray): (Ne,Nk,Nc). ASP-processed spatial filters.
    """
    # basic information
    event_type = np.unique(y_source)
    n_events = event_type.shape[0]  # Ne
    n_components = w_source.shape[-2]  # Nk

    # forward-propagation of source dataset: (Ne*Nt,Nc,Np) & (Ne,Nc,Nk)
    S_source = utils.generate_source_response(X=X_source, y=y_source, w=w_source)
    A_source = utils.forward_propagation(
        X=X_source,
        y=y_source,
        S=S_source,
        w=w_source
    )  # (Ne,Nc,Nk)

    # single-trial data augmentation
    X_target = np.tile(X_target, (n_events, 1, 1))  # (Ne,Nc,Np)
    y_target = np.arange(n_events)  # (Ne,)

    # forward-propagation of target data: (Ne,Nk,Np) & (Ne,Nc,Nk)
    S_target = utils.generate_source_response(X=X_target, y=y_target, w=w_target)  # (Ne,Nk,Np)
    A_target = utils.forward_propagation(
        X=X_target,
        y=y_target,
        S=S_target,
        w=w_target
    )  # (Ne,Nc,Nk)

    # solve Procrustes problem & refinement of spatial filters
    w_asp = np.zeros_like(w_source)  # (Ne,Nk,Nc)
    P = np.zeros((n_events, n_components, n_components))  # (Ne,Nk,Nk)
    for ne in range(n_events):
        U, _, Vh = sLA.svd(A_source[ne].T @ A_target[ne])
        P[ne] = U @ Vh
        w_asp[ne] = P[ne].T @ w_target[ne]
    return P, w_asp


def align_covariance(
        X_source: ndarray,
        y_source: ndarray,
        X_target: ndarray,
        w: ndarray,
        unbias: bool = True,
        conditional: bool = False):
    """Align covariance (AC) process.

    Args:
        X_source (ndarray): (Ne*Nt,Nc,Np). Source dataset.
        y_source (ndarray): (Ne*Nt,). Labels for X_source.
        X_target (ndarray): (Nc,Np). Single-trial test data.
        unbias (bool): Unbias estimation. Defaults to False.
            When 'True', the result may fluctuate by 0.05%.
        conditional (bool). Conditional alignment of spatial distribution.
            Defaults to False.

    Returns:
        Q (ndarray): (Ne,Nc,Nc). Transformation matrices.
        w_ac (ndarray): (Ne,Nk,Nc). AC-processed spatial filters.
    """
    # basic information
    n_events = (np.unique(y_source)).shape[0]  # Ne

    # covariance (spatial distribution) of target dataset
    Ct = utils.generate_var(X=X_target[None, ...], y=None, unbias=unbias)  # (Nc,Nc)

    # covariance (spatial distribution) of source dataset & solve CORAL problems
    if not conditional:
        Cs = utils.generate_var(X=X_source, y=None, unbias=unbias)  # (Nc,Nc)
        Q = np.tile(
            A=utils.solve_coral(Cs=Cs, Ct=Ct),
            reps=(n_events, 1, 1)
        )  # (Ne,Nc,Nc)
    else:  # conditional
        Cs = utils.generate_var(X=X_source, y=y_source, unbias=unbias)  # (Ne,Nc,Nc)
        Q = np.zeros_like(Cs)  # (Ne,Nc,Nc)
        for ne in range(n_events):
            Q[ne] = utils.solve_coral(Cs=Cs[ne], Ct=Ct)

    # refinement of spatial filters
    if w.ndim == 3:  # target-domain spatial filters
        w_ac = np.zeros_like(w)  # (Ne,Nk,Nc)
        for ne in range(n_events):
            w_ac[ne] = w[ne] @ Q[ne].T
    elif w.ndim == 2:  # source-domain DSP filter
        w_ac = np.tile(A=np.zeros_like(w), reps=(n_events, 1, 1))  # (Ne,Nk,Nc)
        for ne in range(n_events):
            w_ac[ne] = w @ sLA.inv(Q[ne].T)
    return Q, w_ac


def alpha_align_subspace(
        X_source: ndarray,
        y_source: ndarray,
        X_target: ndarray,
        source_model: Dict[str, ndarray],
        target_model: Dict[str, ndarray]) -> Dict[str, Union[List[ndarray], ndarray]]:
    """Align subspace process of ALPHA.

    Args:
        X_source (ndarray): (Ne*Nt,Nc,Np). Source dataset.
        y_source (ndarray): (Ne*Nt,). Labels for X_source.
        X_target (ndarray): (Nc,Np). Single-trial test data.
        source_model (dict): See details in alpha_source_decomposition().
        target_model (dict). See details in alpha_target_decomposition().

    Returns:
        P (List[ndarray]): (Ne,Nc,Nc). Transformation matrices (P_1 & P_2) of ASP.
        Q (List[ndarray]): (Ne,Nc,Nc). Transformation matrices (Q_1 & Q_2) of AC.
        u_1_asp (ndarray): (Ne,Nk,Nc). ASP spatial filters for target dataset.
        u_4_asp (ndarray): (Ne,Nk,Nc). ASP spatial filters for target dataset.
        u_1_ac (ndarray): (Ne,Nk,Nc). AC spatial filters for source dataset.
        w_ac (ndarray): (Ne,Nk,Nc). AC spatial filters for target dataset.
        uXs_ac (ndarray): (Ne,Nk,Np). The results of X_source_mean filtered by u_1_ac.
        wXt_ac (ndarray): (Ne,Nk,Np). The results of X_target filtered by w_ac.
        uXt_asp_1 (ndarray): (Ne,Nk,Np). The results of X_target filtered by u_1_asp.
        uXt_asp_2 (ndarray): (Ne,Nk,Np). The results of X_target filtered by u_4_asp.
    """
    # subspace alignment (ASP)
    P_1, u_1_asp = align_spatial_pattern(
        X_source=X_source,
        y_source=y_source,
        X_target=X_target,
        w_source=source_model['u_2'],
        w_target=target_model['u_1']
    )  # (Ne,Nk,Nk) & (Ne,Nk,Nc)
    P_2, u_4_asp = align_spatial_pattern(
        X_source=X_source,
        y_source=y_source,
        X_target=X_target,
        w_source=source_model['u_3'],
        w_target=target_model['u_4']
    )  # (Ne,Nk,Nk) & (Ne,Nk,Nc)

    # subspace alignment: AC
    Q_1, u_1_ac = align_covariance(
        X_source=X_source,
        y_source=y_source,
        X_target=X_target,
        w=target_model['u_1']
    )  # (Ne,Nc,Nc) & (Ne,Nk,Nc)
    Q_2, w_ac = align_covariance(
        X_source=X_source,
        y_source=y_source,
        X_target=X_target,
        w=source_model['w']
    )  # (Ne,Nc,Nc) & (Ne,Nk,Nc)

    # basic information
    n_events = P_1.shape[0]  # Ne

    # transformed templates
    uXs_ac = utils.spatial_filtering(w=u_1_ac, X=source_model['X_source_mean'])
    wXt_ac = np.zeros_like(uXs_ac)  # (Ne,Nk,Np)
    uXt_asp_1, uXt_asp_2 = np.zeros_like(uXs_ac), np.zeros_like(uXs_ac)
    for ne in range(n_events):
        wXt_ac[ne] = w_ac[ne] @ X_target
        uXt_asp_1[ne] = u_1_asp[ne] @ X_target
        uXt_asp_2[ne] = u_4_asp[ne] @ X_target
    wXt_ac = utils.fast_stan_3d(wXt_ac)
    uXt_asp_1 = utils.fast_stan_3d(uXt_asp_1)
    uXt_asp_2 = utils.fast_stan_3d(uXt_asp_2)
    return {
        'P': [P_1, P_2], 'Q': [Q_1, Q_2],
        'u_1_asp': u_1_asp, 'u_4_asp': u_4_asp,
        'u_1_ac': u_1_ac, 'w_ac': w_ac,
        'uXs_ac': uXs_ac, 'wXt_ac': wXt_ac,
        'uXt_asp_1': uXt_asp_1, 'uXt_asp_2': uXt_asp_2
    }


def alpha_multi_dims_feature(
        source_model: Dict[str, ndarray],
        target_model: Dict[str, ndarray],
        trans_model: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """The pattern matching process of ALPHA during trianing process of subspace pooling.

    Args:
        source_model (dict): See details in alpha_source_decomposition().
        target_model (dict): See details in alpha_target_decomposition().
        trans_model (dict): See details in alpha_align_subspace().

    Returns:
        rho_temp (ndarray): (Ne,Nk,5). 5-D features.
        rho (ndarray): (Ne,Nk). Intergrated features.
    """
    # load in pre-calculated templates: (Ne,Nk,Np)
    uX_1, vY_1 = target_model['uX_1'], target_model['vY_1']
    uX_2, uX_3 = source_model['uX_2'], source_model['uX_3']
    wX = source_model['wX']
    uXs_ac, wXt_ac = trans_model['uXs_ac'], trans_model['wXt_ac']
    uXt_asp_1, uXt_asp_2 = trans_model['uXt_asp_1'], trans_model['uXt_asp_2']

    # basic information
    n_events = uX_1.shape[0]  # Ne
    n_components = uX_1.shape[1]  # Nk
    # n_points = uX_1.shape[-1]  # Np, unnecessary

    # 5-part discriminant coefficients
    rho_temp = np.zeros((n_events, n_components, 5))  # (Nk,Ne,5)
    for nk in range(n_components):
        rho_temp[:, nk, 0] = utils.fast_corr_2d(X=uX_1[:, nk, :], Y=vY_1[:, nk, :])
        rho_temp[:, nk, 1] = utils.fast_corr_2d(X=uX_1[:, nk, :], Y=uXs_ac[:, nk, :])
        rho_temp[:, nk, 2] = utils.fast_corr_2d(X=uXt_asp_1[:, nk, :], Y=uX_2[:, nk, :])
        rho_temp[:, nk, 3] = utils.fast_corr_2d(X=uXt_asp_2[:, nk, :], Y=uX_3[:, nk, :])
        rho_temp[:, nk, 4] = utils.fast_corr_2d(X=wXt_ac[:, nk, :], Y=wX[:, nk, :])
    # rho_temp /= n_points  # real Pearson correlation coefficient in scale
    features = {
        'rho_temp': rho_temp,
        'rho': utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1],
            rho_temp[..., 2],
            rho_temp[..., 3],
            rho_temp[..., 4],
        ])
    }
    return features


def alpha_feature(
        w_nk: ndarray,
        source_model: Dict[str, ndarray],
        target_model: Dict[str, ndarray],
        trans_model: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """The pattern matching process of ALPHA.

    Args:
        w_nk (ndarray): (Ne,Nk). See details in ALPHA.subspace_pooling().
        source_model (dict): See details in alpha_source_decomposition().
        target_model (dict): See details in alpha_target_decomposition().
        trans_model (dict): See details in alpha_align_subspace().

    Returns:
        rho (ndarray): (Ne,). ALPHA features.
    """
    # basic information
    n_events = w_nk.shape[0]  # Ne
    features = alpha_multi_dims_feature(
        source_model=source_model,
        target_model=target_model,
        trans_model=trans_model
    )['rho']  # (Ne,Nk)

    # construct ALPHA features
    alpha_rho = np.zeros((n_events))  # (Ne,)
    for ne in range(n_events):
        tar_feat = features[ne]  # (Nk,)
        non_tar_feat = np.delete(features, ne, axis=0)  # (Ne-1,Nk)
        alpha_rho[ne] = (tar_feat - np.mean(non_tar_feat, axis=0)) @ w_nk[ne]
    return {'rho': alpha_rho}


class ALPHA(BasicTransfer):
    def subspace_pooling(self):
        """Train projection vectors to integrate multiple dimensions of subspace."""
        # cross-validation on source-domain dataset
        X_feat, y_feat = [], []
        for _, (train_idx, test_idx) in enumerate(self.cv_idx):
            # divide dataset into train/test dataset
            X_train, y_train = self.X_source[train_idx], self.y_source[train_idx]
            X_test, y_test = self.X_source[test_idx], self.y_source[test_idx]

            n_test = X_test.shape[0]
            source_model = alpha_source_decomposition(
                X_source=X_train,
                y_source=y_train,
                sine_template=self.sine_template,
                n_components=self.n_components
            )
            for nte in range(n_test):
                event_idx = list(self.event_type).index(y_test[nte])
                target_model = alpha_target_decomposition(
                    X_source_mean=source_model['X_source_mean'],
                    X_test=X_test[nte],
                    sine_template=self.sine_template,
                    n_components=self.n_components
                )
                trans_model = alpha_align_subspace(
                    X_source=X_train,
                    y_source=y_train,
                    X_target=X_test,
                    source_model=source_model,
                    target_model=target_model
                )
                X_feat.append(alpha_multi_dims_feature(
                    source_model=source_model,
                    target_model=target_model,
                    trans_model=trans_model
                )['rho'][event_idx, :])  # (Nk,)
                y_feat.append(y_test[nte])
        X_feat, y_feat = np.stack(X_feat), np.array(y_feat)

        # solve EPs
        self.w_nk = np.zeros((self.source_info['n_events'], self.n_components))  # (Ne,Nk)
        for ne, et in enumerate(self.event_type):
            X_feat_temp = X_feat[y_feat == et]
            R = X_feat_temp.T @ X_feat_temp  # (Nk,Nk)
            self.w_nk[ne] = utils.solve_ep(A=R, n_components=1)

    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        # 2nd, 3rd CCA model & DSP model
        self.source_model = alpha_source_decomposition(
            X_source=self.X_source,
            y_source=self.y_source,
            sine_template=self.sine_template,
            n_components=self.n_components
        )
        self.subspace_pooling()

    def transfer_learning(self):
        """Transfer learning between source & target datasets."""
        self.trans_model = alpha_align_subspace(
            X_source=self.X_source,
            y_source=self.y_source,
            X_target=self.X_test,
            source_model=self.source_model,
            target_model=self.target_model
        )

    def intra_target_training(self, X_test: ndarray):
        """Intra-domain model training for target dataset."""
        # 1st & 4th CCA model
        self.X_test = X_test
        self.target_model = alpha_target_decomposition(
            X_source_mean=self.source_model['X_source_mean'],
            X_test=X_test,
            sine_template=self.sine_template,
            n_components=self.n_components
        )

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray,
            cv_method: str = 'skf',
            **cv_paras):
        """Train ALPHA source model.

        Args:
            X_source (ndarray): (Ne*Nt,Nc,Np). Source dataset.
            y_source (ndarray): (Ne*Nt,). Labels for X_source.
            sine_template (ndarray): (Ne,2Nh,Np). Source dataset.
            cv_method (str, optional): Defaults to 'skf'.
                'skf': sklearn.model_selection.StratifiedKFold()
                'sss': sklearn.model_selection.StratifiedShuffleSplit()
                'loo': sklearn.model_selection.LeaveOneOut()
            (below are in **kwargs)
            n_splits (int): Number of folds. Must be at least 2.
                See details in StratifiedKFold() & StratifiedShuffleSplit().
            test_size (int or float):
                See details in StratifiedShuffleSplit()
            random_state (int or None):
                refer: https://scikit-learn.org/stable/glossary.html#term-random_state.
        """
        # load in data
        self.X_source = X_train
        self.y_source = y_train
        self.sine_template = sine_template
        self.event_type = np.unique(self.y_source)

        # basic information of source domain
        self.source_info = utils.generate_data_info(X=self.X_source, y=self.y_source)

        # initialization for subspace pooling (cross-validation)
        n_blocks = np.min(self.source_info['n_train'])  # Nt
        self.cv_idx = []
        if cv_method == 'skf':  # leave-one-block-out
            select_model = StratifiedKFold(n_splits=n_blocks)
        elif cv_method == 'sss':  # Monte-Carlo
            try:
                n_splits = cv_paras['n_splits']
                test_size = cv_paras['test_size']
                random_state = cv_paras['random_state']
            except KeyError:
                n_splits = int(2 * n_blocks)
                test_size = self.source_info['n_events']
                random_state = 0
                print('Missing parameters for StratifiedShuffleSplit!')
                print('Using default ones instead: ')
                print('\t n_splits: {}'.format(n_splits))
                print('\t test_size: {}'.format(test_size))
                print('\t random_state: {}'.format(random_state))
            select_model = StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                random_state=random_state
            )
        elif cv_method == 'loo':  # leave-one-(sample)-out
            select_model = LeaveOneOut()
        for _, idx in enumerate(select_model.split(X=self.X_source, y=self.y_source)):
            self.cv_idx.append(idx)  # (train_index, test_index)

        # train source-domain model & subspace pooling
        self.intra_source_training()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            w_nk (ndarray): (Ne,Nk). See details in alpha_feature().
            source_model (dict): See details in alpha_feature().
            target_model (dict): See details in alpha_feature().
            trans_model (dict): See details in alpha_feature().
        """
        self.intra_target_training(X_test=X_test)
        self.transfer_learning()
        return alpha_feature(
            w_nk=self.w_nk,
            source_model=self.source_model,
            target_model=self.target_model,
            trans_model=self.trans_model
        )


class FB_ALPHA(BasicFBTransfer):
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
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=ALPHA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 13. Cross-stimulus transfer method using common impulse response
def common_conv_matrix(
        freqs: List[Union[int, float]],
        phases: List[Union[int, float]],
        n_points: int,
        srate: Union[int, float] = 1000,
        rrate: int = 60,
        len_scale: float = 1.05,
        amp_scale: float = 0.8,
        extract_method: str = 'Square',
        concat_method: str = 'dynamic',
        resize_method: str = 'Lanczos') -> Tuple[ndarray, ndarray]:
    """Create common convolution matrices (multi-event).

    Args:
        freqs (List[Union[int, float]]): Stimulus frequencies.
        phases (List[Union[int, float]]): Stimulus phases (coefficients). 0-2 (pi).
        n_points (int): Data length.
        srate (int or float): Sampling rate. Defaults to 1000 Hz.
        rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
        len_scale (float): The multiplying power when calculating the length of data.
            Defaults to 0.99.
        amp_scale (float): The multiplying power when calculating the amplitudes of data.
            Defaults to 0.8.
        extrac_method (str): 'Square' or 'Cosine'. Defaults to 'Square'.
            See details in utils.extract_periodic_impulse().
        concat_method (str): 'dynamic' or 'static'.
            'static': Concatenated data is starting from 1 s.
            'dynamic': Concatenated data is starting from 1 period.
        resize_method (str): 'nearest', 'linear', cubic', 'area', 'Lanczos',
            'linear-exact', 'inverse-map', 'fill-outliers'.
            Interpolation methods. Defaults to 'Lanczos'.

    Returns:
        H (ndarray): (Ne,Nrl,Np). Convolution matrices.
        H_correct (ndarray): (Ne,Nrl,Np). Corrected H.
    """
    # basic information
    n_events = len(freqs)  # Ne
    common_response_length = round(srate * len_scale / np.min(freqs))  # longest length

    # create convolution matrices for each event
    H = np.zeros((n_events, common_response_length, n_points))
    H_correct = np.zeros_like(H)
    for ne in range(n_events):
        freq, phase = freqs[ne], phases[ne]

        # generate periodic impulse sequence (spikes)
        periodic_impulse = utils.extract_periodic_impulse(
            freq=freq,
            phase=phase,
            n_points=n_points,
            srate=srate,
            rrate=rrate,
            method=extract_method
        )  # (1,Np)

        # config response length (sampling points)
        response_length = round(srate * len_scale / freq)

        # generate & correct convolution matrices
        H_temp = utils.create_conv_matrix(
            periodic_impulse=periodic_impulse,
            response_length=response_length
        )  # (response_length, Np)
        H_correct_temp = utils.correct_conv_matrix(
            H=H_temp,
            freq=freq,
            srate=srate,
            amp_scale=amp_scale,
            concat_method=concat_method
        )  # (response_length, Np)

        # resize convolution matrices
        H[ne] = utils.resize_conv_matrix(
            H=H_temp,
            new_size=(n_points, common_response_length),
            method=resize_method
        )
        H_correct[ne] = utils.resize_conv_matrix(
            H=H_correct_temp,
            new_size=(n_points, common_response_length),
            method=resize_method
        )
    return H, H_correct


def tim_20243374314_kernel(
        w_init: Union[ndarray, None],
        X_mean: ndarray,
        H: ndarray,
        freq: Union[int, float],
        phase: Union[int, float] = 0,
        srate: Union[int, float] = 1000,
        optimize_method: str = 'CCA',
        iter_limit: int = 200,
        err_th: float = 0.00001,
        n_components: int = 1) -> Dict[str, Union[ndarray, int]]:
    """Intra-domain modeling process of algorithm TIM_20243374314.
    Solve the optimization problem: r, w = argmin||rH - wX||_2^2.
    In which r represents for common impulse response, and w for common spatial filter.

    Args:
        w_init (Union[ndarray, None]): (Nk,Nc). Initial spatial filter w.
            Only useful when optimize_method is 'ALS'.
            If optimize_method is 'CCA', w_init could be NoneType.
        X_mean (ndarray): (Ne,Nc,Np). Averaged template.
        H (ndarray): (Ne,Nrl,Np). Convolution matrix.
        freq (Union[int, float]): Frequency of impulse response.
        phase (Union[int, float]): Initial phase of impulse response. Defaults to 0.
        srate (Union[int, float]): Sampling rate. Defaults to 1000 Hz.
        optimize_method (str): 'CCA' or 'ALS'. Defaults to 'CCA'(faster and better).
            If 'CCA', r,w = CCA(H, X);
            If 'ALS', r,w = argmix||rH - wX||.
        cal_method (str): 'lstsq' or 'pinv'. Methods to solve least squares problem:
            x = min ||b - Ax||_2. If 'lstsq', use sLA.lstsq(); If 'pinv', use sLA.pinv().
            Only useful when optimize_method is 'ALS'.
        iter_limit (int): Number of maximum iteration times. Defaults to 200.
            Only useful when optimize_method is 'ALS'.
        err_th (float): The threshold (th) of ALS error. Stop iteration while
            ALS error is smaller than err_th. Defaults to 10^-5.
            Only useful when optimize_method is 'ALS'.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Defaults to 1. Only useful when optimize_method is 'CCA'.

    Returns:
        w (ndarray): (Nk,Nc). Optimized common spatial filter.
        r (ndarray): (Nk,Nrl). Optimized common impulse response.
        wX (ndarray): (Nk,Ne*Np). w @ X_mean (reshaped).
        rH (ndarray): (Nk,Ne*Np). r @ H (reshaped).
    """
    # reshape H (convolution matrices) & X_mean (avg-templates)
    n_events, response_length, n_points = H.shape  # Ne, response_length, Np
    n_chans = X_mean.shape[1]  # Nc
    H = np.reshape(
        a=np.transpose(H, axes=(1, 0, 2)),
        newshape=(response_length, n_events * n_points),
        order='C'
    )  # (response_length, Ne*Np), concatenated H across events axis
    X_mean = np.reshape(
        a=np.transpose(X_mean, axes=(1, 0, 2)),
        newshape=(n_chans, n_events * n_points),
        order='C'
    )  # (Nc, Ne*Np), concatenated X_mean across events axis

    if optimize_method == 'CCA':
        cca_model = cca.cca_kernel(
            X=X_mean,
            Y=utils.centralization(H),
            n_components=n_components
        )
        w, r = cca_model['u'], cca_model['v']
        wX, rH = cca_model['uX'], cca_model['vY']
    elif optimize_method == 'ALS':
        # initial impulse response (r) & template (rH)
        r_init = np.tile(
            A=utils.sin_wave(freq=freq, n_points=H.shape[0], phase=phase, srate=srate),
            reps=(w_init.shape[0], 1)
        )  # (Nk, response_length)
        r_init = np.diag(1 / np.sqrt(np.diag(r_init @ r_init.T))) @ r_init  # ||r(i,:)|| = 1
        rH_init = r_init @ H  # (Nk,Ne*Np)

        # initial spatial filter (w) & template (wX)
        wX_init = w_init @ X_mean  # (Nk,Ne*Np)

        # iteration initialization
        err_init = np.sum((wX_init - rH_init)**2)
        err_old, err_change = err_init, 0
        log_w, log_r, log_err = [w_init], [r_init], [err_init]

        # start iteration
        n_iter = 1
        w_old = w_init
        continue_training = True
        while continue_training:
            wX_temp = w_old @ X_mean  # (Nk,Np)

            # calculate new impulse response (r_new)
            r_new = wX_temp @ H.T @ sLA.pinv(H @ H.T)  # (Nk,Nrl)
            r_new = np.diag(1 / np.sqrt(np.diag(r_new @ r_new.T))) @ r_new  # ||r(i,:)|| = 1

            # calculate new spatial filter (w_new)
            rH_temp = r_new @ H  # (Nk,Ne*Np)
            w_new = rH_temp @ X_mean.T @ sLA.pinv(X_mean @ X_mean.T)  # (Nk,Nc)
            w_new = np.diag(1 / np.sqrt(np.diag(w_new @ w_new.T))) @ w_new  # ||w(i,:)|| = 1

            # update ALS error
            err_new = np.sum((wX_temp - rH_temp)**2)  # faster
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
        # update optimized results
        w, r = log_w[-1], log_r[-1]
        wX, rH = w @ X_mean, r @ H
        _, r, rH = cca.symbol_correction(uX=wX, vY=rH, v=r)  # symbol correction
    return {'w': w, 'r': r, 'wX': wX, 'rH': rH}


def tim_20243374314_feature(
        X_test: ndarray,
        sine_template: ndarray,
        source_model: Dict[str, ndarray],
        trans_model: Dict[str, ndarray],
        n_components: int = 1) -> Dict[str, ndarray]:
    """The pattern matching process of algorithm TIM-20243374314.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        source_model (dict): {'w': ndarray (Nk,Nc)}
            See details in TIM_20243374314.intra_source_training()
        trans_model (dict): {'rH_trans': ndarray (Ne,Nk (or Ne*Nk),Np)}
            See details in: TIM_20243374314.transfer_learning();
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Defaults to 1.

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models
    rH_trans = trans_model['rH_trans']  # (Ne,Nk(or Ne*Nk),Np)
    w = source_model['w']

    # basic information
    n_events = rH_trans.shape[0]
    n_test = X_test.shape[0]  # Ne*Nte
    n_points = X_test.shape[-1]  # Np

    # reshape matrix for faster computing
    rH_trans = np.reshape(a=rH_trans, newshape=(n_events, -1), order='C')  # (Ne,Nk*Np)
    w = np.tile(A=source_model['w'], reps=(n_events, 1, 1))  # (Nk,Nc) -repeat-> (Ne,Nk,Nc)

    # 2-D features
    rho_temp = np.zeros((n_test, n_events, 2))  # (Ne*Nte,Ne,2)
    for nte in range(n_test):
        wX_temp = utils.fast_stan_3d(w @ X_test[nte])  # (Ne,Nk,Np)

        # rho 1: superposition model matching
        rho_temp[nte, :, 0] = utils.fast_corr_2d(
            X=np.reshape(a=wX_temp, newshape=(n_events, -1), order='C'),
            Y=rH_trans
        ) / n_points

        # rho 2: CCA(wX, Y) | SLOW!
        for nem in range(n_events):
            cca_model = cca.cca_kernel(
                X=wX_temp[nem],
                Y=sine_template[nem],
                n_components=n_components
            )
            rho_temp[nte, nem, 1] = cca_model['coef']
    return {
        'rho_temp': rho_temp,
        'rho': utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1]
        ])
    }


class TIM_20243374314(BasicTransfer):
    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        # initialization for optimize_method
        if self.optimize_method == 'ALS':  # {r, w} = argmin||wX - rH||
            init_model = fast_init_model(
                X_train=self.X_source,
                y_train=self.y_source,
                sine_template=self.sine_template_source,
                n_components=self.n_components,
                method=self.source_info['init_model'],
                w=self.source_info['w_init']
            )
            w_init = init_model['w_init'][0]  # common w | (Nk,Nc)
            X_mean = init_model['X_mean']  # (Ne,Nc,Np)
        elif self.optimize_method == 'CCA':  # {r, w} = CCA(H, X)
            w_init = None
            X_mean = utils.generate_mean(X=self.X_source, y=self.y_source)  # (Ne,Nc,Np)

        # initialization for convolution matrice
        if self.correct_conv:
            H = self.H_correct_source
        else:
            H = self.H_source

        # iteration or analytical solution
        self.source_model = {}
        als_model = tim_20243374314_kernel(
            w_init=w_init,
            X_mean=X_mean,
            H=H,
            freq=self.freq_tar,
            phase=self.phase_tar,
            srate=self.srate,
            optimize_method=self.optimize_method,
            iter_limit=self.iter_limit,
            err_th=self.err_th,
            n_components=self.n_components
        )
        self.source_model['w'] = als_model['w']  # (Nk,Nc)
        self.source_model['r'] = als_model['r']  # (Nk,Nrl)
        self.source_model['wX'] = als_model['wX']  # (Nk,Ne*Np)
        self.source_model['rH'] = als_model['rH']  # (Nk,Ne*Np)

    def transfer_learning(self):
        """Transfer learning between exist & missing events."""
        # basic information & initialization
        n_events = self.H_correct.shape[0]  # Ne
        n_points = self.H_correct.shape[-1]  # Np
        self.trans_model = {}

        # superposition model: r @ H
        if self.correct_conv:
            H = self.H_correct
        else:
            H = self.H
        rH_trans = np.zeros((n_events, self.source_model['r'].shape[0], n_points))
        for ne in range(n_events):
            rH_trans[ne] = self.source_model['r'] @ H[ne]
        self.trans_model['rH_trans'] = utils.fast_stan_3d(rH_trans)

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            stim_info: Dict[str, Tuple[Union[float, int], Union[float, int]]],
            n_harmonics: int = 1,
            init_method: Optional[str] = 'DSP',
            w_init: Optional[ndarray] = None,
            srate: Union[float, int] = 1000,
            rrate: int = 60,
            len_scale: float = 1.05,
            extract_method: str = 'Square',
            optimize_method: str = 'CCA',
            correct_conv: bool = True,
            amp_scale: float = 0.8,
            concat_method: str = 'dynamic',
            iter_limit: int = 200,
            err_th: float = 0.00001,
            resize_method: str = 'Lanczos'):
        """Train TIM-20243374314 model.

        Args:
            X_source (ndarray): (Ne(s)*Nt,Nc,Np). Source training dataset.
                Nt>=2, Ne (source) <= Ne (full).
            y_source (ndarray): (Ne(s)*Nt,). Labels for X_source.
            stim_info (dict): {'label': (frequency, phase)}.
            n_harmonics (int): Number of harmonic components for sinusoidal templates.
                Defaults to 1.
            init_method (str): 'msCCA', 'eTRCA' or 'DSP'. Defaults to 'DSP'.
                Only useful when optimize_method is 'ALS'.
            w_init (ndarray): (Nk,Nc). Initial common spatial filter. Defaults to None.
            srate (int or float): Sampling rate. Defaults to 1000 Hz.
            rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
            len_scale (float): The multiplying power when calculating the length of data.
                Defaults to 1.05.
            extract_method (str): 'Square' or 'Cosine'. Defaults to 'Square'.
                See details in utils.extract_periodic_impulse().
            optimize_method (str): 'CCA' or 'ALS'. Defaults to 'CCA'(faster and better).
                If 'CCA', r,w = CCA(H, X);
                If 'ALS', r,w = argmix||rH - wX||.
            correct_conv (bool): Use corrected H as convolution matrices.
                Defaults to False, i.e. H.
            amp_scale (float): The multiplying power when calculating the amplitudes of data.
                Defaults to 0.8.
            concat_method (str): 'dynamic' or 'static'.
                'static': Concatenated data is starting from 1 s.
                'dynamic': Concatenated data is starting from 1 period.
            iter_limit (int): Number of maximum iteration times. Defaults to 200.
            err_th (float): The threshold (th) of ALS error. Stop iteration while
                ALS error is smaller than err_th. Defaults to 10^-5.
            resize_method (str): 'nearest', 'linear', cubic', 'area', 'Lanczos',
                'linear-exact', 'inverse-map', 'fill-outliers'.
                Interpolation methods. Defaults to 'Lanczos'.
        """
        # load in data
        self.X_source = X_train
        self.y_source = y_train
        self.stim_info = stim_info
        self.n_harmonics = n_harmonics
        self.srate = srate
        self.rrate = rrate
        self.len_scale = len_scale
        self.amp_scale = amp_scale
        self.correct_conv = correct_conv
        self.concat_method = concat_method
        self.extract_method = extract_method
        self.optimize_method = optimize_method
        self.iter_limit = iter_limit
        self.err_th = err_th
        self.resize_method = resize_method

        # create sinusoidal templates for full events
        label_list = [int(label) for label in list(stim_info.keys())]
        freq_list = [stim_info[str(label)][0] for label in label_list]
        phase_list = [stim_info[str(label)][1] for label in label_list]

        sorted_pairs = sorted(enumerate(freq_list), key=lambda x: x[1])
        sorted_idx = [sp[0] for sp in sorted_pairs]
        del sorted_pairs

        freqs = [freq_list[si] for si in sorted_idx]
        phases = [phase_list[si] for si in sorted_idx]
        self.event_type = np.array([label_list[si] for si in sorted_idx])
        del freq_list, phase_list, label_list, sorted_idx

        n_events = self.event_type.shape[0]  # Ne (total)
        n_points = self.X_source.shape[-1]  # Np
        self.sine_template = np.zeros((n_events, 2 * self.n_harmonics, n_points))
        for ne, ett in enumerate(self.event_type):
            self.sine_template[ne] = utils.sine_template(
                freq=freqs[ne],
                phase=phases[ne],
                n_points=n_points,
                n_harmonics=self.n_harmonics,
                srate=self.srate
            )
        del ne, ett
        self.freq_tar = freqs[0]  # lowest frequency, longest response
        self.phase_tar = phases[0]

        # create corrected, resized convolution matrices
        self.H, self.H_correct = common_conv_matrix(
            freqs=freqs,
            phases=phases,
            n_points=n_points,
            srate=self.srate,
            rrate=self.rrate,
            len_scale=self.len_scale,
            amp_scale=self.amp_scale,
            extract_method=self.extract_method,
            concat_method=self.concat_method,
            resize_method=self.resize_method
        )  # (Ne,Nrl, Np)
        del freqs, phases

        # basic information of source domain
        self.source_info = utils.generate_data_info(X=self.X_source, y=self.y_source)
        self.source_info['init_model'] = init_method
        self.source_info['w_init'] = w_init
        event_type_source = self.source_info['event_type']

        # config sinusoidal templates & convolution matrices of source domain
        self.sine_template_source, self.H_source, self.H_correct_source = [], [], []
        for ne, etf in enumerate(self.event_type):
            if etf in list(event_type_source):
                self.sine_template_source.append(self.sine_template[ne])
                self.H_source.append(self.H[ne])
                self.H_correct_source.append(self.H_correct[ne])
        self.sine_template_source = np.stack(self.sine_template_source, axis=0)
        self.H_source = np.stack(self.H_source, axis=0)
        self.H_correct_source = np.stack(self.H_correct_source, axis=0)
        del ne, etf

        # main process
        self.intra_source_training()
        self.transfer_learning()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,3). 3-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return tim_20243374314_feature(
            X_test=X_test,
            sine_template=self.sine_template,
            source_model=self.source_model,
            trans_model=self.trans_model,
            n_components=self.n_components
        )


class FB_TIM_20243374314(BasicFBTransfer):
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
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=TIM_20243374314(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 14. Cross-subject calibration-free classifiation based on transfer superposition theory
def construct_prototype_filter(ws: ndarray, preprocessed: bool = True):
    """Construct prototye filters from given ws.
    Refer: https://ieeexplore.ieee.org/document/8616087/.

    Args:
        ws (ndarray): (Ns,Nk,Nc). Spatial filters from Ns source subjects.
        preprocessed (bool): Whether the F-norm of w has been compressed to 1.
            Defaults to True.

    Returns:
        u (ndarray): (Nk,Nc). Prototye filter.
    """
    # basic information
    n_subjects, n_components, n_chans = ws.shape  # Ns, Nk, Nc
    u = np.zeros((n_components, n_chans))  # (Nk,Nc)

    # F-norm normalization
    wn = deepcopy(ws)
    if not preprocessed:
        for ns in range(n_subjects):
            for nk in range(n_components):
                wn[ns, nk, :] /= sLA.norm(ws[ns, nk, :])
        del ns, nk

    # analytic solution
    for nk in range(n_components):
        Cuu = np.zeros((n_chans, n_chans))  # (Nc,Nc)
        for ns in range(n_subjects):
            Cuu += wn[ns, nk, :][:, None] @ wn[ns, nk, :][None, :]
        u[nk] = utils.solve_ep(A=Cuu, n_components=1, mode='Max')
    return u


def tbme_20243406603_feature(
        X_test: ndarray,
        sine_template: ndarray,
        source_model: Dict[str, Any],
        n_components: int = 1) -> Dict[str, ndarray]:
    """The pattern matching process of algorithm TBME-20243406603.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
        source_model (dict): {'w': ndarray (Ns,Ne,Nk,Nc),
                              'rH': ndarray (Ns,Ne,Nk,Np),
                              'Uw': ndarray (Ne,Nk,Nc),
                              'UrH': ndarray (Ne,Nk,Np)}
            See details in TIM_20243374314.intra_source_training()
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Defaults to 1.

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,3). 3-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models
    w = source_model['w']  # (Ns,Ne,Nk,Nc)
    rH = source_model['rH']  # (Ns,Ne,Nk,Np)
    Uw = source_model['Uw']  # (Ne,Nk,Nc)
    UrH = source_model['UrH']  # (Ne,Nk,Np)

    # basic information
    n_subjects = w.shape[0]  # Ns
    n_events = Uw.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte, could be 1 for unsupervised scenario
    n_points = X_test.shape[-1]  # Np

    # reshape matrix for faster computing
    rH = np.reshape(a=rH, newshape=(n_subjects, n_events, -1), order='C')  # (Ns,Ne,Nk*Np)
    UrH = np.reshape(a=UrH, newshape=(n_events, -1), order='C')  # (Ne,Nk*Np)

    # 3-D features
    rho_temp = np.zeros((n_test, n_events, 3))  # (Ne*Nte,Ne,3)
    for nte in range(n_test):
        # rho 1: pattern matching with each source subject
        wX_temp = np.reshape(
            a=utils.fast_stan_4d(w @ X_test[nte]),
            newshape=(n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Nk,Nc) @ (Nc,Np) -flatten-> (Ns,Ne,Nk*Np)
        rho_temp[nte, :, 0] = np.mean(
            a=utils.fast_corr_3d(X=wX_temp, Y=rH),
            axis=0
        ) / n_points  # (Ns,Ne,Nk*Np) -corr-> (Ns,Ne) -avg-> (Ne,)

        # rho 2: pattern matching with prototype model
        UwX_temp = np.reshape(
            a=utils.fast_stan_3d(Uw @ X_test[nte]),
            newshape=(n_events, -1),
            order='C'
        )  # (Ne,Nk,Nc) @ (Nc,Np) -flatten-> (Ne,Nk*Np)
        rho_temp[nte, :, 1] = utils.fast_corr_2d(X=UwX_temp, Y=UrH) / n_points

        # rho 3: CCA | SLOW!
        for nem in range(n_events):
            cca_model = cca.cca_kernel(
                X=X_test[nte],
                Y=sine_template[nem],
                n_components=n_components
            )
            rho_temp[nte, nem, 2] = cca_model['coef']
    return {
        'rho_temp': rho_temp,
        'rho': utils.combine_feature([
            rho_temp[..., 0],
            rho_temp[..., 1],
            rho_temp[..., 2]
        ])
    }


class TBME_20243406603(BasicTransfer):
    def intra_source_training(self):
        """Intra-domain model training for source dataset."""
        # basic information & initialization
        n_events = self.event_type.shape[0]  # Ne
        self.source_model = {}

        # obtain r (impulse responses) & w (spatial filter) from source subjects
        w, r, wX, rH = [], [], [], []
        for nsub in range(self.n_subjects):
            tlcca_model = tlcca_kernel(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                stim_info=self.stim_info,
                H=self.H,
                w_init=None,
                srate=self.srate,
                optimize_method='CCA',
                n_components=self.n_components,
                target_chan_idx=self.target_chan_idx
            )
            w.append(tlcca_model['w'])  # (Ne,Nk,Nc)
            r.append(tlcca_model['r'])  # List[(Nk,Nrl)]
            wX.append(tlcca_model['wX'])  # (Ne,Nk,Np)
            rH.append(tlcca_model['rH'])  # (Ne,Nk,Np)
        del nsub

        # intergrade source model
        self.source_model['w'] = np.stack(w, axis=0)  # (Ns,Ne,Nk,Nc)
        self.source_model['r'] = deepcopy(r)  # List[List[(Nk,Nrl)]]
        self.source_model['wX'] = np.stack(wX, axis=0)  # (Ns,Ne,Nk,Np)
        self.source_model['rH'] = np.stack(rH, axis=0)  # (Ns,Ne,Nk,Np)
        del w, r, wX, rH

        # train prototye filters
        Ur, Uw = [], []
        for ne in range(n_events):
            # prototye filters for impulse response (r): Ur
            rs_temp = [self.source_model['r'][nsub][ne] for nsub in range(self.n_subjects)]
            rs_temp = np.stack(rs_temp, axis=0)  # List[(Nk,Nrl)] -> (Ns,Nk,Nrl)
            Ur.append(construct_prototype_filter(ws=rs_temp, preprocessed=False))  # (Nk,Nrl)

            # prototype filters for spatial filters (w): Uw
            ws_temp = self.source_model['w'][:, ne, ...]  # (Ns,Nk,Nc)
            Uw.append(construct_prototype_filter(ws=ws_temp, preprocessed=True))  # (Nk,Nc)
            del rs_temp, ws_temp
        del ne
        self.source_model['Ur'] = deepcopy(Ur)  # List[(Nk,Nrl)]
        self.source_model['Uw'] = np.stack(Uw, axis=0)  # List[(Nk,Nc)] -> (Ne,Nk,Nc)
        del Ur, Uw

        # train prototype templates: rH
        UrH = np.zeros_like(self.source_model['rH'][0])  # (Ne,Nk,Np)
        for ne in range(n_events):
            UrH[ne] = self.source_model['Ur'][ne] @ self.H_correct[ne]
        self.source_model['UrH'] = deepcopy(UrH)
        del UrH, ne

    def fit(
            self,
            X_train: List[ndarray],
            y_train: List[ndarray],
            sine_template: ndarray,
            stim_info: Dict[str, Tuple[Union[float, int], Union[float, int]]],
            srate: Union[float, int] = 1000,
            rrate: int = 60,
            len_scale: float = 1.05,
            amp_scale: float = 0.8,
            concat_method: str = 'dynamic',
            target_chan_idx: int = 7):
        """Train model.

        Args:
            X_train (ndarray): List[(Ne*Nt,Nc,Np)]. Source training dataset. Nt>=2.
            y_train (ndarray): List[(Ne(s)*Nt,)]. Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
            stim_info (dict): {'label': (frequency, phase)}.
            srate (int or float): Sampling rate. Defaults to 1000 Hz.
            rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
            len_scale (float): The multiplying coefficient for the length of data.
                Defaults to 1.05.
            amp_scale (float): The multiplying coefficient for the amplitudes of data.
                Defaults to 0.8.
            concat_method (str): 'dynamic' or 'static'.
                'static': Concatenated data is starting from 1 s.
                'dynamic': Concatenated data is starting from 1 period.
            target_chan_idx (int): The index of target channel to correct of w & r.
                Recommend to set to the channel 'Oz'. Defaults to 7.
                See details in solve_tlcca_func().
        """
        # load in data
        self.X_source = X_train
        self.y_source = y_train
        self.sine_template = sine_template
        self.stim_info = stim_info
        self.srate = srate
        self.rrate = rrate
        self.len_scale = len_scale
        self.amp_scale = amp_scale
        self.concat_method = concat_method
        self.target_chan_idx = target_chan_idx

        # basic information of source domain
        n_events = self.sine_template.shape[0]  # Ne
        self.n_subjects = len(self.X_source)
        self.event_type = np.array(
            [int(list(stim_info.keys())[ne]) for ne in range(n_events)]
        )
        self.source_info = []
        for nsub in range(self.n_subjects):
            self.source_info.append(utils.generate_data_info(
                X=self.X_source[nsub],
                y=self.y_source[nsub]
            ))

        # config convolution matrices
        n_points = self.X_source[0].shape[-1]  # Np
        self.H, self.H_correct = [], []
        for _, et in enumerate(self.event_type):
            H_temp, H_correct_temp = tlcca_conv_matrix(
                freq=self.stim_info[str(et)][0],
                phase=self.stim_info[str(et)][1],
                n_points=n_points,
                srate=self.srate,
                rrate=self.rrate,
                len_scale=self.len_scale,
                amp_scale=self.amp_scale,
                extract_method='Square',
                concat_method=self.concat_method,
                response_length=None
            )
            self.H.append(H_temp)
            self.H_correct.append(H_correct_temp)
        del H_temp, H_correct_temp, et

        # main process
        self.intra_source_training()

    def transform(self, X_test):
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,3). 3-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return tbme_20243406603_feature(
            X_test=X_test,
            sine_template=self.sine_template,
            source_model=self.source_model,
            n_components=self.n_components
        )


class FB_TBME_20243406603(BasicFBTransfer):
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
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=TBME_20243406603(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )

    def fit(
            self,
            X_train: List[ndarray],
            y_train: List[ndarray],
            bank_weights: Optional[ndarray] = None,
            **kwargs):
        """Load in training dataset and pass it to sub-esimators.
        See details in TBME_20243406603().fit().

        Args:
            X_train (ndarray): List[(Nb,Ne*Nt,Nc,Np)]. Source training dataset. Nt>=2.
            y_train (ndarray): List[(Nb,Ne(s)*Nt,)]. Labels for X_train.
            bank_weights (ndarray, optional): Weights for different filter banks.
                Defaults to None (equal).
        """
        # basic information
        self.Nb = X_train[0].shape[0]  # n_bands
        n_subjects = len(X_train)  # Ns

        # initialization
        self.bank_weights = bank_weights
        if self.version == 'SSVEP' and self.bank_weights is not None:
            self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25 for nb in range(self.Nb)])
        self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]

        # apply in each sub-band
        for nb, se in enumerate(self.sub_estimator):
            se.fit(
                X_train=[X_train[nsub][nb] for nsub in range(n_subjects)],
                y_train=y_train,
                **kwargs
            )
