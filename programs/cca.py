# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Canonical correlation analysis (CCA) series.
    1. CCA: http://ieeexplore.ieee.org/document/4203016/
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

update: 2023/07/04

"""

# %% basic modules
import utils

from abc import abstractmethod, ABCMeta
from typing import Optional, List, Tuple, Any

import numpy as np
from numpy import ndarray
import scipy.linalg as sLA


# %% Basic CCA object
class BasicCCA(metaclass=ABCMeta):
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
            X_train (ndarray): (Ne*Nt,...,Np). Training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
        """
        pass


    @abstractmethod
    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients
            y_predict (ndarray): (Ne*Nte,). Predict labels.
        """
        pass


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
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
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
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int,
                            'events_group':{'event_id':[start index,end index]}}
        n_components (int): Number of eigenvectors picked as filters. Nk.
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
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
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
        """Train ms-eCCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=1.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
            events_group (dict): {'event_id':[start index,end index]}
            d (int): The range of events to be merged.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        self.d = d
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
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
        sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (Ne,),
                            'n_events':int,
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters. Nk.
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

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        train_info (dict): {'n_events':int,
                            'n_train':ndarray (Ne,),
                            'n_chans':int,
                            'n_points':int}
        n_components (int, optional): Number of eigenvectors picked as filters.
            Defaults to 1. Set to 'None' if ratio is not 'None'.
        ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not 'None'.

    Return: MsetCCA1 model (dict)
        R (List[ndarray]): Ne*(Nt*Nc,Nt*Nc). Covariance of original data (various trials).
        S (List[ndarray]): Ne*(Nt*Nc,Nt*Nc). Covariance of original data (same trials).
        w (List[ndarray]): Ne*(Nk, Nt*Nc). Spatial filters for training dataset.
        wX (List[ndarray]): Ne*(Nt*Nk, Np). MsetCCA(1) templates.
    """
    # basic information
    event_type = train_info['event_type']
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np

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


# %%
