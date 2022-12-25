# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Canonical correlation analysis (CCA) series.
    (1) CCA: http://ieeexplore.ieee.org/document/4203016/
            DOI: 10.1109/TBME.2006.889197
    (2) eCCA: http://www.pnas.org/lookup/doi/10.1073/pnas.1508080112
            DOI: 10.1073/pnas.1508080112
    (3) msCCA: https://ieeexplore.ieee.org/document/9006809/
            DOI: 10.1109/TBME.2020.2975552
    (4) ms-eCCA: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
            DOI: 10.1088/1741-2552/ab2373
    (5) MsetCCA1: https://www.worldscientific.com/doi/abs/10.1142/S0129065714500130
            DOI: 10.1142/S0129065714500130
    (6) MsetCCA2: https://ieeexplore.ieee.org/document/8231203/
            DOI: 10.1109/TBME.2017.2785412
    (7) MwayCCA: 
            DOI: 
    (8) stCCA: https://ieeexplore.ieee.org/document/9177172/
            DOI: 10.1109/TNSRE.2020.3019276
    (9)

update: 2022/11/15

"""

# %% basic modules
import utils

import numpy as np

from abc import abstractmethod, ABCMeta


# %% Basic CCA object
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
        pass


    @abstractmethod
    def predict(self, X_test, y_test):
        pass


# %% (1) standard CCA | CCA
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
    U = solve_gep(
        A=Cxy @ sLA.solve(Cyy,Cyx),
        B=Cxx,
        n_components=n_components,
        ratio=ratio
    )

    # template part: (Nk,2Nh)
    V = solve_gep(
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


    def predict(self, X_test, y_test):
        """Using CCA algorithm to predict test data.

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
        n_events = len(self.y_train)

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                self.rou[nte,ne] = utils.pearson_corr(
                    X=X_test[nte],
                    Y=self.X_train[ne]
                )
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


# %% (2) Extended CCA | eCCA
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
    r1 = pearson_corr(U1@X_test, V1@sine_template)

    # correlation coefficients between single-trial EEG and SSVEP templates
    U2, V2 = cca_compute(
        data=X_test,
        template=avg_template,
        n_components=n_components,
        ratio=ratio
    )
    r2 = pearson_corr(U2@X_test, U2@avg_template)

    r3 = pearson_corr(U1@X_test, U1@avg_template)

    U3, _ = cca_compute(
        data=avg_template,
        template=sine_template,
        n_components=n_components,
        ratio=ratio
    )
    r4 = pearson_corr(U3@X_test, U3@avg_template)

    # similarity between filters corresponding to single-trial EEG and SSVEP templates
    r5 = pearson_corr(U2@avg_template, V2@avg_template)

    # combined features
    rou = combine_feature([r1, r2, r3, r4, r5])
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

        # config average template (class center)
        n_events = len(np.unique(self.y_train))
        n_chans = X_train.shape[-2]
        n_points = X_train.shape[-1]
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
        n_events = len(self.y_train)

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


# %% (3-4) Multi-stimulus eCCA | ms-eCCA
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
        w (ndarray): (n_components, n_chans). Common spatial filter.
        template (ndarray): ()
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
    w = utils.solve_gep(A=A, B=B, n_components=1, ratio=None)
    
    # signal templats
    template = np.zeros_like(avg_template)  # (Ne,Nc,Np)
    for ne in range(n_events):
        template[ne] = w @ avg_template[ne]
    return w, template


class MS_CCA(BasicCCA):
    def fit(self, X_train, y_train, Q, events_group=None, d=None):
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
        self.w, self.template = mscca_compute(
            avg_template=self.avg_template,
            Q=self.Q,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        return self


    def predict(self, X_test, y_test):
        """Using ms-CCA algorithm to predict test data.

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
        n_events = len(self.y_train)

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            for ne in range(n_events):
                self.rou[nte,ne] = utils.pearson_corr(
                    X=self.w @ X_test[nte],
                    Y=self.template[ne]
                )
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
        u (list of ndarray): n_events*(n_components, n_chans).
            Spatial filters for EEG.
        v (list of ndarray): n_events*(n_components, 2*n_harmonics).
            Spatial filters for sinusoidal templates.
        template_eeg (ndarray): n_events*(n_components, n_points).
            ms-CCA templates for EEG part.
        template_sin (ndarray): n_events*(n_components, n_points).
            ms-CCA templates for sinusoidal template part.
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    events_group = train_info['events_group']  # dict

    # GEPs' conditions
    total_Czz = einsum('ecp,ehp->ech', avg_template,avg_template)  # (Ne,Nc,Nc)
    total_Cyy = einsum('ecp,ehp->ech', sine_template,sine_template)  # (Ne,2Nh,2Nh)
    total_Czy = einsum('ecp,ehp->ech', avg_template,sine_template)  # (Ne,Nc,2Nh)

    # GEPs with merged data
    u, u_ndim, v, v_ndim = [], [], [], []
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
        u_ndim.append(spatial_filter_eeg.shape[0])
        u.append(spatial_filter_eeg)

        # sinusoidal template part: (Nk,2Nh)
        spatial_filter_sin = utils.solve_gep(
            A=temp_Cyz @ sLA.solve(temp_Czz,temp_Czy),
            B=temp_Cyy,
            n_components=n_components,
            ratio=ratio
        )
        v_ndim.append(spatial_filter_sin.shape[0])
        v.append(spatial_filter_sin)

    # signal templates
    template_eeg, template_sin = [], []
    for ne in range(n_events):
        template_eeg.append(u[ne] @ avg_template[ne])
        template_sin.append(v[ne] @ sine_template[ne])
    model = (
        Czz, Czy, Cyy,
        u, v,
        template_eeg, template_sin
    )
    return model


class MS_ECCA(BasicCCA):
    def fit(self, X_train, y_train, sine_tempalte):
        """Train ms-CCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
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
        return self

        # train_ms-CCA filters and templates
        results = msecca_compute(
            avg_template=self.avg_template,
            sine_template=self.sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Czz, self.Czy, self.Cyy = results[0], results[1], results[2]
        self.u, self.v = results[3], results[4]
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
        n_events = len(self.y_train)

        # pattern matching
        self.rou = np.zeros((n_test, n_events))
        self.rou_eeg = np.zeros_like(self.rou_eeg)
        self.rou_sin = np.zeros_like(self.rou_eeg)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            f_test = self.w @ X_test[nte]
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


# %% (5-6) Multiset CCA | MsetCCA1
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
        w (list of ndarray): n_events*(1, n_train*n_chans).
            Spatial filters for training dataset.
        template (list of ndarray): n_events*(n_train, n_points).
            MsetCCA1 templates.
    """
    # basic information
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np

    # GEPs with block-concatenated data
    R, S, w = [], [], []
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
        w.append(spatial_filter)

    # signal templates
    template = []
    for ne in range(n_events):
        temp_template = np.zeros((n_train[ne], n_points))
        temp_X_train = X_train[y_train==ne]
        for ntr in range(n_train[ne]):
            temp_template[ntr] = w[ntr][:,ntr*n_chans:(ntr+1)*n_chans] @ temp_X_train[ntr]
        template.append(temp_template)
    model = (R, S, w, template)
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
                           'n_chans':X_train.shape[-2],
                           'n_points':X_train.shape[-1]}

        # train MsetCCA(1) filters and templates
        results = msetcca1_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            train_info=self.train_info
        )  # n_components & ratio must be 1, None
        

def msetcca():
    pass


# %% Cross-subject transfer learning | CSSFT
def cssft_compute():
    pass


def cssft():
    pass



# %% Filter-bank CCA series | FB-
def fb_cca(sine_template, test_data, n_components=1, ratio=None):
    """CCA algorithms with filter banks.

    Args:
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_bands = test_data.shape[0]

    # multiple CCA classification
    rou = []
    for nb in range(n_bands):
        rou.append(cca(template=sine_template, test_data=test_data[nb], n_components=n_components, ratio=ratio))
    return combine_fb_feature(rou)


def fb_ecca(train_data, sine_template, test_data, n_components=1, ratio=None):
    """eCCA with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_bands = train_data.shape[0]

    # multiple eCCA classification
    rou = []
    for nb in range(n_bands):
        rou.append(ecca(train_data=train_data[nb], sine_template=sine_template,
            test_data=test_data[nb], n_components=n_components, ratio=ratio))
    return combine_fb_feature(rou)


def fb_mscca(train_data, sine_template, test_data, n_components=1, ratio=None):
    """msCCA with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_bands = train_data.shape[0]

    # multiple msCCA classification
    rou = []
    for nb in range(n_bands):
        rou.append(mscca(
            train_data=train_data[nb],
            sine_template=sine_template,
            test_data=test_data[nb],
            n_components=n_components,
            ratio=ratio
        ))
    return combine_fb_feature(rou)


def fb_msecca(train_data, sine_template, test_data, d, n_components=1, ratio=None, **kwargs):
    """ms-eCCA with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        d (int): The range of events to be merged.
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_bands = train_data.shape[0]
    n_events = train_data.shape[1]
    try:
        events_group = kwargs['events_group']
    except KeyError:
        events_group = augmented_events(n_events, d)

    # multiple mseCCA classification
    rou = []
    for nb in range(n_bands):
        rou.append(msecca(
            train_data=train_data[nb],
            sine_template=sine_template,
            test_data=test_data[nb],
            d=d,
            n_components=n_components,
            ratio=ratio,
            events_group=events_group
        ))
    return combine_fb_feature(rou)