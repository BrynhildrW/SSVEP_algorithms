# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Task-related component analysis (TRCA) series with dynamic subspace dimension module.
    1. (e)TRCA: https://ieeexplore.ieee.org/document/7904641/
            DOI: 10.1109/TBME.2017.2694818
    2. ms-(e)TRCA: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
            DOI: 10.1088/1741-2552/ab2373
    3. (e)TRCA-R: https://ieeexplore.ieee.org/document/9006809/
            DOI: 10.1109/TBME.2020.2975552
    4. sc-(e)TRCA: https://iopscience.iop.org/article/10.1088/1741-2552/abfdfa
            DOI: 10.1088/1741-2552/abfdfa

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
import trca

from abc import abstractmethod
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
from numpy import ndarray

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from sklearn.model_selection import StratifiedShuffleSplit


# %% Basic TRCA object
class BasicETRCA_DD(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, ratio: Optional[float] = None):
        """Basic configuration.

        Args:
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Only useful when n_components is None.
        """
        # config model
        self.ratio = ratio

    @abstractmethod
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: Optional[ndarray] = None):
        """See details in trca.BasicTRCA().fit()."""
        pass

    @abstractmethod
    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """See details in trca.BasicTRCA().transform()."""
        pass

    def predict(self, X_test: ndarray) -> Union[ndarray, int]:
        """See details in trca.BasicTRCA().predict()."""
        self.features = self.transform(X_test)
        self.y_pred = self.event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


class BasicFBETRCA_DD(utils.FilterBank, ClassifierMixin):
    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """See details in trca.BasicFBTRCA().transform()."""
        if not self.with_filter_bank:  # tranform X_test
            X_test = self.fb_transform(X_test)  # (Nb,Ne*Nte,Nc,Np)
        sub_features = [se.transform(X_test[nse])
                        for nse, se in enumerate(self.sub_estimator)]
        rho_fb = np.stack([sf['rho'] for sf in sub_features], axis=0)
        rho = np.einsum('b,bte->te', self.bank_weights, rho_fb)
        return {'fb_rho': rho_fb, 'rho': rho}

    def predict(self, X_test: ndarray) -> Union[ndarray, int]:
        """See details in trca.BasicFBTRCA().predict()."""
        self.features = self.transform(X_test)
        event_type = self.sub_estimator[0].event_type
        self.y_pred = event_type[np.argmax(self.features['rho'], axis=-1)]
        return self.y_pred


# %% 1. (ensemble) TRCA | (e)TRCA
def solve_etrca_dd_func(
        Q: ndarray,
        S: ndarray,
        ratio: float = 0.1) -> Tuple[List[ndarray], ndarray]:
    """See details in trca.solve_trca_func()."""
    # basic information
    n_events = Q.shape[0]  # Ne

    # solve GEPs
    w = []
    for ne in range(n_events):
        w.append(utils.solve_gep(A=S[ne], B=Q[ne], n_components=None, ratio=ratio))
    ew = np.concatenate(w, axis=0)  # (Ne*Nk,Nc)
    return w, ew


def generate_etrca_dd_template(ew: ndarray, X_mean: ndarray) -> ndarray:
    """See details in trca.generate_trca_template()."""
    return utils.spatial_filtering(w=ew, X_mean=X_mean)  # (Ne,Ne*Nk,Np)


def etrca_dd_kernel(
        X_train: ndarray,
        y_train: ndarray,
        ratio: float = 0.1) -> Dict[str, ndarray]:
    """See details in trca.trca_kernel()."""
    # solve target functions
    Q, S, X_mean = trca.generate_trca_mat(X=X_train, y=y_train)
    w, ew = solve_etrca_dd_func(Q=Q, S=S, ratio=ratio)

    # generate spatial-filtered templates
    ewX = generate_etrca_dd_template(ew=ew, X_mean=X_mean)
    return {'Q': Q, 'S': S, 'w': w, 'ew': ew, 'ewX': ewX}


def etrca_dd_feature(
        X_test: ndarray,
        trca_model: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """See details in trca.trca_feature()."""
    # load in model
    ew, ewX = trca_model['ew'], trca_model['ewX']  # (Ne*Nk,Nc), (Ne,Ne*Nk,Np)
    n_events = ewX.shape[0]  # Ne
    ewX = np.reshape(ewX, (n_events, -1), 'C')  # (Ne,Ne*Nk*Np)

    # pattern matching
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))
    for nte in range(n_test):
        X_temp = utils.fast_stan_2d(ew @ X_test[nte])  # (Ne*Nk,Np)
        X_temp = np.tile(np.reshape(X_temp, -1, 'C'), (n_events, 1))  # (Ne,Ne*Nk*Np)
        rho[nte] = utils.fast_corr_2d(X=X_temp, Y=ewX)
    return {'rho': rho}


class ETRCA(BasicETRCA_DD):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray):
        """See details in trca.TRCA().fit()."""
        # basic information
        self.X_train = utils.fast_stan_3d(X_train)
        self.y_train = y_train
        self.event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'

        # train TRCA filters & templates
        self.training_model = etrca_dd_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            ratio=self.ratio
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """See details in trca.TRCA().transform()."""
        return etrca_dd_feature(X_test=X_test, trca_model=self.training_model)


class ETRCA_DD(BasicETRCA_DD):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            ratio_min: float = 0.1,
            ratio_max: float = 0.6,
            ratio_step: float = 0.05,
            n_splits: int = 10,
            n_valid: int = 2):
        """See details in trca.TRCA().fit()."""
        # basic information
        self.X_train = utils.fast_stan_3d(X_train)
        self.y_train = y_train
        self.event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'

        # train TRCA filters & templates
        if self.ratio is None:
            n_events = self.event_type.shape[0]  # Ne
            sss = StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=int(n_events * n_valid),
                random_state=0
            )
            ratio_list = np.arange(ratio_min, ratio_max + ratio_step, ratio_step)
            acc_valid = np.zeros((ratio_list.shape[0], n_splits))
            for nr, ratio in enumerate(ratio_list):
                for nrep, (tr_idx, va_idx) in enumerate(sss.split(self.X_train, self.y_train)):
                    X_tr, y_tr = self.X_train[tr_idx], self.y_train[tr_idx]
                    X_va, y_va = self.X_train[va_idx], self.y_train[va_idx]

                    model = ETRCA(ratio=np.round(ratio, 2))
                    model.fit(X_train=X_tr, y_train=y_tr)
                    y_pred = model.predict(X_test=X_va)
                    acc_valid[nr, nrep] = utils.acc_compute(y_true=y_va, y_pred=y_pred)
            self.ratio = ratio_list[np.argmax(np.mean(acc_valid, axis=-1))]
        self.training_model = etrca_dd_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            ratio=self.ratio
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """See details in trca.TRCA().transform()."""
        return etrca_dd_feature(X_test=X_test, trca_model=self.training_model)


class FB_ETRCA_DD(BasicFBETRCA_DD):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            ratio: float = 0.1):
        """See details in trca.FB_TRCA().__init__()."""
        super().__init__(
            base_estimator=ETRCA_DD(ratio=ratio),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 2. multi-stimulus (e)TRCA | ms-(e)TRCA
def solve_msetrca_dd_func(
        Q: ndarray,
        S: ndarray,
        event_type: ndarray,
        events_group: Dict[str, List[int]],
        ratio: float = 0.1) -> Tuple[List[ndarray], ndarray]:
    """See details in trca.solve_mstrca_func()."""
    # basic information
    n_events = Q.shape[0]  # Ne

    # solve GEPs with merged data
    w = []
    for ne in range(n_events):
        merged_indices = events_group[str(event_type[ne])]
        Q_temp = np.sum(Q[merged_indices], axis=0)  # (Nc,Nc)
        S_temp = np.sum(S[merged_indices], axis=0)  # (Nc,Nc)
        w.append(utils.solve_gep(A=S_temp, B=Q_temp, n_components=None, ratio=ratio))
    ew = np.concatenate(w, axis=0)  # (Ne*Nk,Nc)
    return w, ew


def msetrca_dd_kernel(
        X_train: ndarray,
        y_train: ndarray,
        events_group: Dict[str, List[int]],
        ratio: float = 0.1) -> Dict[str, ndarray]:
    """See details in trca.mstrca_kernel()."""
    # solve target functions
    Q_total, S_total, X_mean = trca.generate_trca_mat(X=X_train, y=y_train)
    w, ew = solve_msetrca_dd_func(
        Q=Q_total,
        S=S_total,
        event_type=np.unique(y_train),
        events_group=events_group,
        ratio=ratio
    )

    # generate spatial-filtered templates
    ewX = generate_etrca_dd_template(ew=ew, X_mean=X_mean)
    return {'Q': Q_total, 'S': S_total, 'w': w, 'ew': ew, 'ewX': ewX}


class MS_ETRCA(ETRCA):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            events_group: Optional[Dict[str, List[int]]] = None,
            d: int = 2):
        """See details in trca.MS_TRCA().fit()."""
        # basic information
        self.X_train = utils.fast_stan_3d(X_train)
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
        self.training_model = msetrca_dd_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            events_group=self.events_group,
            ratio=self.ratio
        )


class MS_ETRCA_DD(ETRCA):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            events_group: Optional[Dict[str, List[int]]] = None,
            d: int = 2,
            ratio_min: float = 0.1,
            ratio_max: float = 0.6,
            ratio_step: float = 0.05,
            n_splits: int = 10,
            n_valid: int = 2):
        """See details in ETRCA_DD().fit() and trca.MS_TRCA().fit()."""
        # basic information
        self.X_train = utils.fast_stan_3d(X_train)
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
        if self.ratio is None:
            n_events = self.event_type.shape[0]  # Ne
            sss = StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=int(n_events * n_valid),
                random_state=0
            )
            ratio_list = np.arange(ratio_min, ratio_max + ratio_step, ratio_step)
            acc_valid = np.zeros((ratio_list.shape[0], n_splits))
            for nr, ratio in enumerate(ratio_list):
                for nrep, (tr_idx, va_idx) in enumerate(sss.split(self.X_train, self.y_train)):
                    X_tr, y_tr = self.X_train[tr_idx], self.y_train[tr_idx]
                    X_va, y_va = self.X_train[va_idx], self.y_train[va_idx]

                    model = MS_ETRCA(ratio=np.round(ratio, 2))
                    model.fit(
                        X_train=X_tr,
                        y_train=y_tr,
                        events_group=self.events_group,
                        d=self.d
                    )
                    y_pred = model.predict(X_test=X_va)
                    acc_valid[nr, nrep] = utils.acc_compute(y_true=y_va, y_pred=y_pred)
            self.ratio = ratio_list[np.argmax(np.mean(acc_valid, axis=-1))]
        self.training_model = msetrca_dd_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            events_group=self.events_group,
            ratio=self.ratio
        )


# %% 3. similarity constrained (e)TRCA | sc-(e)TRCA
def solve_scetrca_dd_func(
        Q: ndarray,
        S: ndarray,
        n_chans: int,
        ratio: float = 0.1) -> Tuple[List[ndarray], List[ndarray],
                                     ndarray, ndarray]:
    """See details in trca.solve_sctrca_func()."""
    # basic information
    n_events = Q.shape[0]  # Ne

    # solve GEPs
    u, v = [], []  # Ne*(Nk,Nc), Ne*(Nk,2Nh)
    for ne in range(n_events):
        w = utils.solve_gep(A=S[ne], B=Q[ne], n_components=None, ratio=ratio)
        u.append(w[:, :n_chans])  # (Nk,Nc)
        v.append(w[:, n_chans:])  # (Nk,2Nh)
    eu = np.concatenate(u, axis=0)  # (Ne*Nk,Nc)
    ev = np.concatenate(v, axis=0)  # (Ne*Nk,2*Nh)
    return u, v, eu, ev


def generate_scetrca_dd_template(
        eu: ndarray,
        ev: ndarray,
        X_mean: ndarray,
        sine_template: ndarray) -> Tuple[ndarray, ndarray]:
    """See details in trca.generate_sctrca_template()."""
    euX = utils.spatial_filtering(w=eu, X_mean=X_mean)
    evY = utils.spatial_filtering(w=ev, X_mean=sine_template)
    return euX, evY


def scetrca_dd_kernel(
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        ratio: float = 0.1) -> Dict[str, ndarray]:
    """See details in trca.sctrca_kernel()."""
    # basic information
    n_chans = X_train.shape[1]  # Nc

    # solve target functions
    Q, S, X_mean = trca.generate_sctrca_mat(X=X_train, y=y_train, sine_template=sine_template)
    u, v, eu, ev = solve_scetrca_dd_func(Q=Q, S=S, n_chans=n_chans, ratio=ratio)

    # generate spatial-filtered templates
    euX, evY = generate_scetrca_dd_template(
        eu=eu,
        ev=ev,
        X_mean=X_mean,
        sine_template=sine_template
    )
    return {
        'Q': Q, 'S': S,
        'u': u, 'v': v, 'eu': eu, 'ev': ev,
        'euX': euX, 'evY': evY
    }


def scetrca_dd_feature(
        X_test: ndarray,
        sctrca_model: Dict[str, ndarray]) -> Dict[str, ndarray]:
    """See details in trca.sctrca_feature()."""
    # load in model
    eu = sctrca_model['eu']  # (Ne*Nk,Nc)
    euX = sctrca_model['euX']  # (Ne,Ne*Nk,Np)
    evY = sctrca_model['evY']  # (Ne,Ne*Nk,Np)
    n_events = euX.shape[0]  # Ne
    euX = np.reshape(euX, (n_events, -1), 'C')  # (Ne,Ne*Nk*Np)
    evY = np.reshape(evY, (n_events, -1), 'C')  # (Ne,Ne*Nk*Np)

    # pattern matching
    n_test = X_test.shape[0]  # Ne*Nte
    rho_eeg = np.zeros((n_test, n_events))
    rho_sin = np.zeros_like(rho_eeg)  # (Ne*Nte,Ne)
    for nte in range(n_test):
        X_temp = utils.fast_stan_2d(eu @ X_test[nte])  # (Ne*Nk,Np)
        X_temp = np.tile(
            A=np.reshape(X_temp, -1, 'C'),
            reps=(n_events, 1)
        )  # (Ne*Nk,Np) -reshape-> (Ne*Nk*Np,) -repeat-> (Ne,Ne*Nk*Np)
        rho_eeg[nte] = utils.fast_corr_2d(X=X_temp, Y=euX)
        rho_sin[nte] = utils.fast_corr_2d(X=X_temp, Y=evY)
    rho = utils.combine_feature([rho_eeg, rho_sin])
    return {'rho': rho, 'rho_eeg': rho_eeg, 'rho_sin': rho_sin}


class SC_ETRCA(BasicETRCA_DD):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray):
        """See details in trca.SC_TRCA().fit()."""
        # basic information
        self.X_train = utils.fast_stan_3d(X_train)
        self.y_train = y_train
        self.event_type = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'

        # train sc-TRCA models & templates
        self.training_model = scetrca_dd_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=sine_template,
            ratio=self.ratio
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """See details in trca.SC_TRCA().transform()."""
        return scetrca_dd_feature(X_test=X_test, sctrca_model=self.training_model)


class SC_ETRCA_DD(BasicETRCA_DD):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray,
            ratio_min: float = 0.1,
            ratio_max: float = 0.6,
            ratio_step: float = 0.05,
            n_splits: int = 10,
            n_valid: int = 2):
        """See details in trca.SC_TRCA().fit()."""
        # basic information
        self.X_train = utils.fast_stan_3d(X_train)
        self.y_train = y_train
        self.event_type = np.unique(self.y_train)  # [0,1,2,...,Ne-1]
        n_train = np.array([np.sum(self.y_train == et) for et in self.event_type])
        assert np.min(n_train) > 1, 'Insufficient training samples!'

        # train sc-TRCA models & templates
        if self.ratio is None:
            n_events = self.event_type.shape[0]  # Ne
            sss = StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=int(n_events * n_valid),
                random_state=0
            )
            ratio_list = np.arange(ratio_min, ratio_max + ratio_step, ratio_step)
            acc_valid = np.zeros((ratio_list.shape[0], n_splits))
            for nr, ratio in enumerate(ratio_list):
                for nrep, (tr_idx, va_idx) in enumerate(sss.split(self.X_train, self.y_train)):
                    X_tr, y_tr = self.X_train[tr_idx], self.y_train[tr_idx]
                    X_va, y_va = self.X_train[va_idx], self.y_train[va_idx]

                    model = SC_ETRCA(ratio=np.round(ratio, 2))
                    model.fit(X_train=X_tr, y_train=y_tr, sine_template=sine_template)
                    y_pred = model.predict(X_test=X_va)
                    acc_valid[nr, nrep] = utils.acc_compute(y_true=y_va, y_pred=y_pred)
            self.ratio = ratio_list[np.argmax(np.mean(acc_valid, axis=-1))]
        self.training_model = scetrca_dd_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=sine_template,
            ratio=self.ratio
        )

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """See details in trca.SC_TRCA().transform()."""
        return scetrca_dd_feature(X_test=X_test, sctrca_model=self.training_model)
