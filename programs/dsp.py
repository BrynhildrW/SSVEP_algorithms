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


update: 2023/7/4

"""

# %% basic modules
import utils

from abc import abstractmethod, ABCMeta
from typing import Optional, List, Tuple, Any

import numpy as np
from numpy import ndarray
import scipy.linalg as sLA

# %% Basic object
class BasicDSP(metaclass=ABCMeta):
    def __init__(self,
        n_components: int = 1):
        """Basic configuration.

        Args:
            n_components (int): Number of eigenvectors picked as filters.
        """
        # config model
        self.n_components = n_components


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
        """
        pass


    @abstractmethod
    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        self.rho = self.transform(X_test)
        self.y_predict = self.event_type[np.argmax(self.rho, axis=-1)]
        return self.y_predict


class BasicFBDSP(metaclass=ABCMeta):
    def __init__(self,
        n_components: int = 1):
        """Basic configuration.

        Args:
            n_components (int): Number of eigenvectors picked as filters.
        """
        # config model
        self.n_components = n_components


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
        """
        pass


    def predict(self,
        X_test: ndarray) -> Tuple:
        """Using filter-bank DSP algorithms to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        # apply model.predict() method in each sub-band
        self.fb_rho = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.fb_rho[nb] = self.sub_models[nb].transform(X_test=X_test[nb])

        # integration of multi-bands' results
        self.rho = utils.combine_fb_feature(self.fb_rho)
        self.y_predict = self.event_type[np.argmax(self.rho, axis=-1)]
        return self.y_predict


# %% 1. Discriminant Spatial Patterns | DSP
def _dsp_kernel(
    X_train: ndarray,
    y_train: ndarray,
    train_info: dict,
    n_components: int = 1) -> dict:
    """The modeling process of DSP.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int}
        n_components (int, optional): Number of eigenvectors picked as filters. Nk.

    Returns: DSP model (dict)
        Sb (ndarray): (Nc,Nc). Scatter matrix of between-class difference.
        Sw (ndarray): (Nc,Nc). Scatter matrix of within-class difference.
        w (ndarray): (Nk,Nc). Common spatial filter.
        wX (ndarray): (Ne,Nk,Np). DSP templates.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np

    # between-class difference Hb -> scatter matrix Sb
    class_center = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for ne,et in enumerate(event_type):
        class_center[ne] = np.mean(X_train[y_train==et], axis=0)  # (Nc,Np)
    total_center = X_train.mean(axis=0, keepdims=True)  # (1,Nc,Np)
    Hb = class_center - total_center  # (Ne,Nc,Np)
    Sb = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for ne in range(n_events):
        Sb += Hb[ne] @ Hb[ne].T
    Sb /= n_events
    # Sb = np.einsum('ecp,ehp->ch', Hb,Hb)/n_events | clearer but slower

    # within-class difference Hw -> scatter matrix Sw
    Sw = np.zeros_like(Sb)  # (Nc,Nc)
    for ne,et in enumerate(event_type):
        Hw = X_train[y_train==et] - class_center[ne]  # (Nt,Nc,Np)-(Nc,Np)
        for ntr in range(n_train[ne]):  # samples for each event
            Sw += Hw[ntr] @ Hw[ntr].T
    Sw /= X_train.shape[0]
    # Sw = einsum('etcp,ethp->ch', Hw,Hw)/(n_events*n_train) | clearer but slower

    # GEPs | train spatial filter
    w = utils.solve_gep(
        A=Sb,
        B=Sw,
        n_components=n_components,
        ratio=None
    )  # (Nk,Nc)

    # DSP model
    wX = np.einsum('kc,ecp->ekp', w,class_center)  # (Ne,Nk,Np)
    model = {
        'Sb':Sb, 'Sw':Sw,
        'w':w, 'wX':wX
    }
    return model


def _dsp_feature(
    X_test: ndarray,
    dsp_model: dict) -> ndarray:
    """The pattern matching process of DSP.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        dsp_model (dict): See details in _dsp_kernel().

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of DSP.
    """
    w, wX = dsp_model['w'], dsp_model['wX']
    n_events = wX.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))
    for nte in range(n_test):
        temp_X = w @ X_test[nte]
        rho[nte,:] = utils.pearson_corr(X=temp_X, Y=wX, common_filter=True)
    return rho


class DSP(BasicDSP):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        """Train DSP model.

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
            'n_points':self.X_train.shape[-1]
        }

        # train DSP models & templates
        self.training_model = _dsp_kernel(
            X_train=self.X_train,
            y_train = self.y_train,
            train_info = self.train_info,
            n_components=self.n_components,
        )
        return self


    def transform(self,
        X_test: ndarray) -> Tuple:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return: Tuple
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of DSP.
        """
        return _dsp_feature(X_test=X_test, dsp_model=self.training_model)


    def predict(self,
        X_test: ndarray) -> Tuple:
        """Using DSP algorithm to predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            y_predict (ndarray): (Ne*Nte,). Predict labels of DSP.
        """
        self.rho = self.transform(X_test)
        self.y_predict = self.train_info['event_type'][np.argmax(self.rho, axis=-1)]
        return self.y_predict


class FB_DSP(BasicFBDSP):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray):
        """Train filter-bank DSP model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.n_bands = X_train.shape[0]

        # train DSP models & wXs
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = DSP(n_components=self.n_components)
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train
            )
        return self


    def transform(self,
        X_test: ndarray) -> ndarray:
        pass


    def predict(self,
        X_test: ndarray) -> Tuple:
        pass

# %% 2. Discriminant Canonical Pattern Matching | DCPM



# %% 3. Task-discriminant component analysis | TDCA
def _tdca_augmentation(
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
        X_aug2 (ndarray): ((m+1)*n_chans, 2*n_points).
    """
    # basic information
    n_chans = X.shape[0]  # Nc
    n_points = projection.shape[0]  # Np

    # secondary augmented data
    X_aug2 = np.tile(np.zeros_like(X), (extra_length+1, 2))  # ((m+1)*Nc,2*Np)
    if extra_data is not None:  # for training dataset
        X_temp = np.concatenate((X, extra_data), axis=-1)  # with extra length
        for el in range(extra_length+1):
            sp, ep = el*n_chans, (el+1)*n_chans
            X_aug2[sp:ep,:n_points] = X_temp[:,el:n_points+el]
            X_aug2[sp:ep,n_points:] = X_aug2[sp:ep,:n_points] @ projection
    else:  # for test dataset
        for el in range(extra_length+1):
            sp, ep = el*n_chans, (el+1)*n_chans
            X_aug2[sp:ep,:n_points-el] = X[:,el:n_points]
            X_aug2[sp:ep,n_points:] = X_aug2[sp:ep,:n_points] @ projection
    return X_aug2


def _tdca_feature(
    X_test: ndarray,
    tdca_model: dict,
    projection: ndarray,
    extra_length: int) -> ndarray:
    """The pattern matching process of TDCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        tdca_model (dict): See details in _dsp_kernel().
        projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
        extra_length (int): m.

    Returns:
        rho (ndarray): (Ne*Nte,Ne). Discriminant coefficients of TDCA.
    """
    w, wX = tdca_model['w'], tdca_model['wX']
    n_events = wX.shape[0]  # Ne
    n_test = X_test.shape[0]  # Ne*Nte
    rho = np.zeros((n_test, n_events))
    for nte in range(n_test):
        for nem in range(n_events):
            X_test_aug2 = _tdca_augmentation(
                X=X_test[nte],
                projection=projection[nem],
                extra_length=extra_length
            )
            rho[nte,nem] = utils.pearson_corr(X=w@X_test_aug2, Y=wX[nem])
    return rho


class TDCA(BasicDSP):
    def fit(self,
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
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.projection = projection

        # create secondary augmented data | (Ne*Nt,(el+1)*Nc,2*Np)
        self.X_train_aug2 = np.tile(np.zeros_like(self.X_train), (1,(self.extra_length+1),2))
        for ntr,label in enumerate(self.y_train):
            event_idx = list(event_type).index(label)
            self.X_train_aug2[ntr] = _tdca_augmentation(
                X=self.X_train[ntr],
                projection=self.projection[event_idx],
                extra_length=self.extra_length,
                extra_data=self.X_extra[ntr]
            )
        self.train_info = {'event_type':event_type,
                           'n_events':len(event_type),
                           'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
                           'n_chans':self.X_train_aug2.shape[-2],
                           'n_points':self.X_train_aug2.shape[-1]}

        # train DSP models & wXs
        self.training_model = _dsp_kernel(
            X_train=self.X_train_aug2,
            y_train=self.y_train,
            train_info=self.train_info,
            n_components=self.n_components
        )
        return self


    def transform(self,
        X_test: ndarray) -> Tuple:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return: Tuple
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of DSP.
        """
        rho = _tdca_feature(
            X_test=X_test,
            tdca_model=self.training_model,
            projection=self.projection,
            extra_length=self.extra_length
        )
        return rho


    def predict(self,
        X_test: ndarray) -> Tuple:
        """Using TDCA algorithm to predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            y_predict (ndarray): (Ne*Nte,). Predict labels of DSP.
        """
        self.rho = self.transform(X_test)
        self.y_predict = self.train_info['event_type'][np.argmax(self.rho, axis=-1)]
        return self.y_predict


class FB_TDCA(BasicFBDSP):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        projection: ndarray,
        extra_length: int):
        """Train FB-TDCA model.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            projection (ndarray): (Nb,Ne,Np,Np). Orthogonal projection matrices.
            extra_length (int).
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.projection = projection
        self.extra_length = extra_length
        self.n_bands = X_train.shape[0]

        # train TDCA models & wXs
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = TDCA(n_components=self.n_components)
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                projection=self.projection,
                extra_length=self.extra_length
            )
        return self


    def transform(self,
        X_test: ndarray) -> ndarray:
        pass


    def predict(self,
        X_test: ndarray) -> Tuple:
        pass