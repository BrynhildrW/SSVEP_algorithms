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
    13. stCCA: https://ieeexplore.ieee.org/document/9177172/
            DOI: 10.1109/TNSRE.2020.3019276
    14. 

update: 2022/11/15

"""

# basic modules
import utils

import numpy as np
import scipy.linalg as sLA

from abc import abstractmethod, ABCMeta


# Basic CCA object
class BasicCCA(metaclass=ABCMeta):
    def __init__(self, n_components=1, ratio=None):
        """Config model dimension.

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
    def fit(self, X_train, y_train):
        """Load in training dataset.

        Args:
            X_train (ndarray): (train_trials, ..., n_points). Training dataset.
            y_train (ndarray): (train_trials,). Labels for X_train.
        """
        pass


    @abstractmethod
    def predict(self, X_test, y_test):
        pass


# 1. standard CCA | CCA
def cca_compute(data, template, n_components=1, ratio=None):
    """Canonical correlation analysis.

    Args:
        data (ndarray): (n_chans, n_points). Real EEG data of a single trial.
        template (ndarray): (2*n_harmonics or m, n_points). Artificial sinusoidal template or averaged template.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Eigenvectors are referring to eigenvalues sorted in descend order.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        U (ndarray): (n_components, n_chans). Spatial filter for EEG.
        V (ndarray): (n_components, 2*n_harmonics). Spatial filter for template.
    """
    # GEPs' conditions
    Cxx = data @ data.T  # (Nc,Nc)
    Cyy = template @ template.T  # (2Nh,2Nh)
    Cxy = data @ template.T  # (Nc,2Nh)
    Cyx = Cxy.T  # (2Nh,Nc)

    # EEG part: (n_components(Nk),Nc)
    U = utils.solve_gep(
        A=Cxy @ sLA.solve(Cyy,Cyx),
        B=Cxx,
        n_components=n_components,
        ratio=ratio
    )

    # template part: (Nk,2Nh)
    V = utils.solve_gep(
        A=Cyx @ sLA.solve(Cxx,Cxy),
        B=Cyy,
        n_components=n_components,
        ratio=ratio
    )
    return U, V


class CCA(BasicCCA):
    def fit(self, X_train, y_train):
        """Load in CCA template. CCA is an unsupervised algorithm, 
            so there's no need to train any CCA model.

        Args:
            X_train (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
            y_train (ndarray): (n_events,). Labels for X_train.
        """
        self.X_train = X_train
        self.y_train = y_train
        return self


    def predict(self, X_test):
        """Using CCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.

        Returns:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = len(self.y_train)

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                U, V = cca_compute(
                    data=X_test[nte],
                    template=self.X_train[ne],
                    n_components=self.n_components,
                    ratio=self.ratio
                )
                self.rou[nte,ne] = utils.pearson_corr(
                    X=U @ X_test[nte],
                    Y=V @ self.X_train[ne]
                )
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


class FB_CCA(BasicCCA):
    def fit(self, X_train, y_train):
        """Train filter-bank CCA model.

        Args:
            X_train (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
            y_train (ndarray): (n_events,). Labels for X_train.
        """
        self.X_train = X_train
        self.y_train = y_train
        return self


    def predict(self, X_test):
        """Using filter-bank CCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_bands, n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_bands = X_test.shape[0]
        n_test = X_test.shape[1]

        # apply CCA().predict() in each sub-band
        self.sub_models = [[] for nb in range(n_bands)]
        self.fb_rou = [[] for nb in range(n_bands)]
        self.fb_y_predict = [[] for nb in range(n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = CCA(
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train,
                y_train=self.y_train
            )
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
            self.fb_rou[nb], self.fb_y_predict[nb] = fb_results[0], fb_results[1]

        # integration of multi-bands' results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


# 2. Minimum Energy Combination | MEC
def mec_compute(data, template, n_components=1, ratio=None):
    """Minimum energy combination.

    Args:
        data (ndarray): (n_chans, n_points). Real EEG data of a single trial.
        template (ndarray): (2*n_harmonics or m, n_points).
            Artificial sinusoidal template or averaged template.
        n_components (int): Number of eigenvectors picked as filters. Nk.
            Eigenvectors are referring to eigenvalues sorted in descend order.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.
    """
    # projection matrix: (Np,Np)
    # projection = template.T @ sLA.inv(template @ template.T) @ template  # slow way
    projection = template.T @ template / np.sum(template[0]**2)  # fast way
    X_hat = data - data @ projection  # (Nc,Np)

    # GEP's conditions
    A = X_hat @ X_hat.T

    # spatial filter: (Nk,Nc)
    W = utils.solve_ep(
        A=A,
        n_components=n_components,
        ratio=ratio,
        mode='Min'
    )
    return W


# 3. Maximum Contrast Combination | MCC


# 4. MSI | MSI


# 5. tMSI


# 6. extend-MSI | eMSI


# 7. Extended CCA | eCCA
def ecca_compute(avg_template, sine_template, X_test, n_components=1, ratio=None):
    """CCA with individual calibration data.

    Args:
        avg_template (ndarray): (n_chans, n_points). Trial-averaged data.
        sine_template (ndarray): (2*n_harmonics, n_points). Sinusoidal template.
        X_test (ndarray): (n_chans, n_points). Test-trial EEG.
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (float): feature coefficient.
    """
    # correlation coefficient from CCA process
    U1, V1 = cca_compute(
        data=X_test,
        template=sine_template,
        n_components=n_components,
        ratio=ratio
    )
    r1 = utils.pearson_corr(U1@X_test, V1@sine_template)

    # correlation coefficients between single-trial EEG and SSVEP templates
    U2, V2 = cca_compute(
        data=X_test,
        template=avg_template,
        n_components=n_components,
        ratio=ratio
    )
    r2 = utils.pearson_corr(U2@X_test, U2@avg_template)

    r3 = utils.pearson_corr(U1@X_test, U1@avg_template)

    U3, _ = cca_compute(
        data=avg_template,
        template=sine_template,
        n_components=n_components,
        ratio=ratio
    )
    r4 = utils.pearson_corr(U3@X_test, U3@avg_template)

    # similarity between filters corresponding to single-trial EEG and SSVEP templates
    r5 = utils.pearson_corr(U2@avg_template, V2@avg_template)

    # combined features
    rou = utils.combine_feature([r1, r2, r3, r4, r5])
    return np.real(rou)


class ECCA(BasicCCA):
    def fit(self, X_train, y_train, sine_template):
        """Load in eCCA templates.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
                Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        event_type = np.unique(self.y_train)
        n_events = len(event_type)
        n_chans = X_train.shape[-2]
        n_points = X_train.shape[-1]
        self.train_info = {'event_type':event_type,
                           'n_events':n_events,
                           'n_chans':n_chans,
                           'n_points':n_points}

        # config average template (class center)
        self.avg_template = np.zeros((n_events, n_chans, n_points))
        for ne in range(n_events):
            self.avg_template[ne] = X_train[y_train==ne].mean(axis=0)
        return self


    def predict(self, X_test, y_test):
        """Using eCCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Returns:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = y_test.shape[0]
        n_events = self.train_info['n_events']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                self.rou[nte,ne] = ecca_compute(
                    avg_template=self.avg_template[ne],
                    sine_template=self.sine_template[ne],
                    X_test=X_test[nte],
                    n_components=self.n_components,
                    ratio=self.ratio
                )
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


class FB_ECCA(BasicCCA):
    def fit(self, X_train, y_train, sine_template):
        """Load in filter-bank eCCA templates.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
                Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        return self


    def predict(self, X_test, y_test):
        """Using filter-bank eCCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_bands, n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Returns:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_bands = X_test.shape[0]
        n_test = X_test.shape[1]
        
        # apply ECCA().predict() in each sub-band
        self.sub_models = [[] for nb in range(n_bands)]
        self.fb_rou = [[] for nb in range(n_bands)]
        self.fb_y_predict = [[] for nb in range(n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = ECCA(
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                sine_template=self.sine_template
            )
            fb_results = self.sub_models[nb].predict(
                X_test=X_test[nb],
                y_test=y_test
            )
            self.fb_rou[nb], self.fb_y_predict[nb] = fb_results[0], fb_results[1]

        # integration of multi-bands' results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


# 8-9. Multi-stimulus eCCA | ms-eCCA
# msCCA is only part of ms-eCCA. Personally, i dont like this design
def mscca_compute(avg_template, Q, train_info, n_components=1, ratio=None):
    """Multi-stimulus CCA.

    Args:
        avg_template (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        Q (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        train_info (dict): {'n_events':int,
                            'n_chans':int,
                            'n_points':int}
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        W (ndarray): (n_components, n_chans). Common spatial filter.
        template (ndarray): (n_events, n_components, n_points). TRCA-R templates.
    """
    # basic information
    n_events = train_info['n_events']
    n_chans = train_info['n_chans']

    # GEPs | train common spatial filter
    A = np.zeros((n_chans, n_chans))  # (Nc,Nc)
    for nea in range(n_events):
        for neb in range(n_events):
            A += avg_template[nea] @ Q[nea] @ Q[neb].T @ avg_template[neb].T
    B = np.einsum('ecp,ehp->ch', avg_template, avg_template)
    W = utils.solve_gep(A=A, B=B, n_components=n_components, ratio=ratio)
    
    # signal templats
    template = np.einsum('kc,ecp->ekp', W,avg_template)
    return W, template


class MS_CCA(BasicCCA):
    def fit(self, X_train, y_train, Q):
        """Train ms-CCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            Q (ndarray): (n_events, n_points, 2*n_harmonics).
                QR decomposition of sinusoidal template.
            events_group (dict): {'event_id':[start index,end index]}
            d (int): The range of events to be merged.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.Q = Q
        event_type = np.unique(self.y_train)
        n_events = len(event_type)
        n_chans = self.X_train.shape[-2]
        n_points = self.X_train.shape[-1]
        self.train_info = {'event_type':event_type,
                           'n_events':n_events,
                           'n_chans':n_chans,
                           'n_points':n_points}

        # config average template | (Ne,Nc,Np)
        self.avg_template = np.zeros((n_events, n_chans, n_points))
        for ne in range(n_events):
            self.avg_template[ne] = X_train[y_train==ne].mean(axis=0)

        # train ms-CCA filters & templates
        self.W, self.template = mscca_compute(
            avg_template=self.avg_template,
            Q=self.Q,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        return self


    def predict(self, X_test):
        """Using ms-CCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.

        Returns:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                self.rou[nte,ne] = utils.pearson_corr(
                    X=self.W @ X_test[nte],
                    Y=self.template[ne]
                )
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


class FB_MS_CCA(BasicCCA):
    def fit(self, X_train, y_train, Q):
        """Train filter-bank ms-CCA model.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            Q (ndarray): (n_events, n_points, 2*n_harmonics).
                QR decomposition of sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.Q = Q
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
                Q=self.Q
            )
        return self


    def predict(self, X_test):
        """Using filter-bank ms-CCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_bands, n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.

        Returns:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[1]

        # apply MS_CCA().predict() in each sub-band
        self.fb_rou = [[] for nb in range(self.n_bands)]
        self.fb_y_predict = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
            self.fb_rou[nb], self.fb_y_predict[nb] = fb_results[0], fb_results[1]

        # integration of multi-bands' results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


def msecca_compute(avg_template, sine_template, train_info, n_components=1, ratio=None):
    """Multi-stimulus eCCA.

    Args:
        avg_template (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        train_info (dict): {'n_events':int,
                            'n_chans':int,
                            'n_points':int,
                            'events_group':{'event_id':[start index,end index]}}
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns: | all contained in a tuple
        Czz (ndarray): (n_events, n_chans, n_chans).
            Covariance of averaged EEg template.
        Czy (ndarray): (n_events, n_chans, 2*n_harmonics).
            Covariance between EEG and sinusoidal template.
        Cyy (ndarray): (n_events, 2*n_harmonics, 2*n_harmonics).
            Covariance of sinusoidal template.
        U (list of ndarray): n_events*(n_components, n_chans).
            Spatial filters for EEG.
        V (list of ndarray): n_events*(n_components, 2*n_harmonics).
            Spatial filters for sinusoidal templates.
        template_eeg (ndarray): n_events*(n_components, n_points).
            ms-CCA templates for EEG part.
        template_sin (ndarray): n_events*(n_components, n_points).
            ms-CCA templates for sinusoidal template part.
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    events_group = train_info['events_group']  # dict

    # GEPs' conditions
    total_Czz = np.einsum('ecp,ehp->ech', avg_template,avg_template)  # (Ne,Nc,Nc)
    total_Cyy = np.einsum('ecp,ehp->ech', sine_template,sine_template)  # (Ne,2Nh,2Nh)
    total_Czy = np.einsum('ecp,ehp->ech', avg_template,sine_template)  # (Ne,Nc,2Nh)

    # GEPs with merged data
    U, U_ndim, V, V_ndim = [], [], [], []
    for ne in range(n_events):
        # GEPs' conditions
        idx = str(ne)
        st, ed = events_group[idx][0], events_group[idx][1]
        temp_Czz = np.sum(total_Czz[st:ed], axis=0)  # (Nc,Nc)
        temp_Cyy = np.sum(total_Cyy[st:ed], axis=0)  # (2Nh,2Nh)
        temp_Czy = np.sum(total_Czy[st:ed], axis=0)  # (Nc,2Nh)
        temp_Cyz = temp_Czy.T  # (2Nh,Nc)

        # EEG part: (Nk,Nc)
        spatial_filter_eeg = utils.solve_gep(
            A=temp_Czy @ sLA.solve(temp_Cyy,temp_Cyz),
            B=temp_Czz,
            n_components=n_components,
            ratio=ratio
        )
        U_ndim.append(spatial_filter_eeg.shape[0])
        U.append(spatial_filter_eeg)

        # sinusoidal template part: (Nk,2Nh)
        spatial_filter_sin = utils.solve_gep(
            A=temp_Cyz @ sLA.solve(temp_Czz,temp_Czy),
            B=temp_Cyy,
            n_components=n_components,
            ratio=ratio
        )
        V_ndim.append(spatial_filter_sin.shape[0])
        V.append(spatial_filter_sin)

    # signal templates
    template_eeg, template_sin = [], []
    for ne in range(n_events):
        template_eeg.append(U[ne] @ avg_template[ne])
        template_sin.append(V[ne] @ sine_template[ne])
    model = (
        total_Czz, total_Czy, total_Cyy,
        U, V,
        template_eeg, template_sin
    )
    return model


class MS_ECCA(BasicCCA):
    def fit(self, X_train, y_train, sine_template, events_group=None, d=None):
        """Train ms-eCCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
            events_group (dict): {'event_id':[start index,end index]}
            d (int): The range of events to be merged.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.sine_template = sine_template
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        if events_group:  # given range
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(len(event_type), d)
        n_events = len(event_type)
        n_chans = X_train.shape[-2]
        n_points = X_train.shape[-1]
        self.train_info = {'event_type':event_type,
                           'n_events':n_events,
                           'n_chans':n_chans,
                           'n_points':n_points,
                           'events_group':self.events_group}

        # config average template | (Ne,Nc,Np)
        self.avg_template = np.zeros((n_events, n_chans, n_points))
        for ne in range(n_events):
            self.avg_template[ne] = X_train[y_train==ne].mean(axis=0)

        # train_ms-CCA filters and templates
        results = msecca_compute(
            avg_template=self.avg_template,
            sine_template=self.sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Czz, self.Czy, self.Cyy = results[0], results[1], results[2]
        self.U, self.V = results[3], results[4]
        self.template_eeg, self.template_sin = results[5], results[6]
        return self


    def predict(self, X_test, y_test):
        """Using ms-eCCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Returns:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = y_test.shape[0]
        n_events = self.train_info['n_events']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.rou_eeg = np.zeros_like(self.rou)
        self.rou_sin = np.zeros_like(self.rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            f_test = self.U @ X_test[nte]
            for ne in range(n_events):
                self.rou_eeg[nte,ne] = utils.pearson_corr(
                    X=f_test,
                    Y=self.template_eeg[ne]
                )
                self.rou_sin[nte,ne] = utils.pearson_corr(
                    X=f_test,
                    Y=self.template_sin[ne]
                )
                self.rou[nte,ne] = utils.combine_feature([
                    self.rou_eeg[nte,ne],
                    self.rou_sin[nte,ne]
                ])
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


class FB_MS_ECCA(FB_MS_CCA):
    def fit(self, X_train, y_train, sine_template, events_group=None, d=None):
        """Train filter-bank ms-eCCA model.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
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
                sine_template=self.sine_template,
                events_group=self.events_group,
                d=self.d
            )
        return self


# 10-11 Multiset CCA | MsetCCA1
def msetcca1_compute(X_train, y_train, train_info, n_components=1, ratio=None):
    """Multiset CCA (1).

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        train_info (dict): {'n_events':int,
                            'n_train':ndarray (n_events,),
                            'n_chans':int,
                            'n_points':int}
        n_components (int, optional): Number of eigenvectors picked as filters.
            Defaults to 1. Set to 'None' if ratio is not 'None'.
        ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not 'None'.

    Returns: | all contained in a tuple
        R (list of ndarray): n_events*(n_train*n_chans, n_train*n_chans).
            Covariance of original data (various trials).
        S (list of ndarray): n_events*(n_train*n_chans, n_train*n_chans).
            Covariance of original data (same trials).
        W (list of ndarray): n_events*(1, n_train*n_chans).
            Spatial filters for training dataset.
        template (list of ndarray): n_events*(n_train, n_points).
            MsetCCA(1) templates.
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np

    # GEPs with block-concatenated data
    R, S, W = [], [], []
    for ne in range(n_events):
        n_sample = n_train[ne]
        temp_R = np.zeros((n_sample*n_chans, n_sample*n_chans))  # (Nt*Nc,Nt*Nc)
        temp_S = np.zeros_like(temp_R)
        temp_X_train = X_train[y_train==ne]
        for nsa in range(n_sample):  # loop in columns
            spc, epc = nsa*n_chans, (nsa+1)*n_chans  # start/end point of columns
            temp_S[spc:epc,spc:epc] = temp_X_train[nsa] @ temp_X_train[nsa].T
            for nsm in range(n_sample):  # loop in rows
                spr, epr = nsm*n_chans, (nsm+1)*n_chans  # start/end point of rows
                if nsm < nsa:  # upper triangular district
                    temp_R[spr:epr,spc:epc] = temp_R[spc:epc,spr:epr].T
                else:
                    temp_R[spr:epr,spc:epc] = temp_X_train[nsm] @ temp_X_train[nsa].T
        spatial_filter = utils.solve_gep(
            A=temp_R,
            B=temp_S,
            n_components=n_components,
            ratio=ratio
        )
        R.append(temp_R)
        S.append(temp_S)
        W.append(spatial_filter)

    # signal templates
    template = []
    for ne in range(n_events):
        temp_template = np.zeros((n_train[ne], n_points))
        temp_X_train = X_train[y_train==ne]
        for ntr in range(n_train[ne]):
            temp_template[ntr] = W[ntr][:,ntr*n_chans:(ntr+1)*n_chans] @ temp_X_train[ntr]
        template.append(temp_template)
    model = (R, S, W, template)
    return model


class MSETCCA1(BasicCCA):
    def fit(self, X_train, y_train):
        """Train MsetCCA(1) model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...Ne-1]
        self.train_info = {'event_type':event_type,
                           'n_events':len(event_type),
                           'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
                           'n_chans':X_train.shape[-2],
                           'n_points':X_train.shape[-1]}

        # train MsetCCA(1) filters and templates
        results = msetcca1_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            train_info=self.train_info
        )  # n_components & ratio must be 1, None
        self.R, self.S = results[0], results[1]
        self.W, self.template = results[2], results[3]
        return self


    def predict(self, X_test, y_test):
        """Using MsetCCA(1) algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Returns:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = y_test.shape[0]
        n_events = self.train_info['n_events']

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                U, V = cca_compute(
                    data=X_test[nte],
                    template=self.template[ne],
                    n_components=self.n_components,
                    ratio=self.ratio
                )
                self.rou[nte,ne] = utils.pearson_corr(
                    X=U @ X_test[nte],
                    Y=V @ self.template[ne]
                )
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


def msetcca2_compute():
    pass


class MSETCCA2(BasicCCA):
    pass


# 13. Subject transfer based CCA | stCCA



# Cross-subject transfer learning | CSSFT
def cssft_compute():
    pass


class CSSFT(BasicCCA):
    pass

