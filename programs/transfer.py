# -*- coding: utf-8 -*-
"""
@ Author: Brynhildr Wu
@ Email: brynhildrwu@gmail.com

Transfer learning based on matrix decomposition.
    (1) SAME: https://ieeexplore.ieee.org/document/9971465/
            DOI: 10.1109/TBME.2022.3227036
    (2) TL-TRCA: (unofficial name): https://ieeexplore.ieee.org/document/10057002/
            DOI: 10.1109/TNSRE.2023.3250953
    (3) stCCA: https://ieeexplore.ieee.org/document/9177172/
            DOI: 10.1109/TNSRE.2020.3019276
    (4) tlCCA: https://ieeexplore.ieee.org/document/9354064/
            DOI: 10.1109/TASE.2021.3054741


update: 2023/07/04

"""

# %% basic modules
import utils
from cca import msecca_compute
from trca import BasicTRCA, BasicFBTRCA

from typing import Optional, List, Tuple
from numpy import ndarray

import numpy as np
import scipy.linalg as sLA


# %% 1. source aliasing matrix estimation, SAME


# %% 2. cross-subject transfer learning TRCA, TL-TRCA
def tltrca_intra_compute(
    X_train: ndarray,
    y_train: ndarray,
    sine_template: ndarray,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict:
    """Intra-subject training in TL-(e)TRCA.

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Dataset of source subjects. train_trials could be 1 if necessary.
        y_train (ndarray): (train_trials,). Labels for X.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: source_model (dict).
        Q (ndarray): (n_events, 2*n_chans+2*n_harmonics, 2*n_chans+2*n_harmonics).
            Covariance of original data & template data.
        S (ndarray): (n_events, 2*n_chans+2*n_harmonics, 2*n_chans+2*n_harmonics).
            Covariance of template data.
        w (List of ndarray): n_events*(n_components, n_chans).
            Spatial filters for original signal.
        u (List of ndarray): n_events*(n_components, n_chans).
            Spatial filters for average template.
        v (List of ndarray): n_events*(n_components, 2*n_harmonics).
            Spatial filters for sinusoidal template.
        w_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for original signal.
        u_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for average template.
        v_concat (ndarray): (n_events*n_components, 2*n_harmonics).
            Concatenated filter for sinusoidal signal.
        I (ndarray): (n_events, n_events*n_components, n_points).
            Transferred individual templates of source subjects.
        R (ndarray): (n_events, n_events*n_components, n_points).
            Transferred reference templates of source subjects.
    """
    # basic information
    event_type = np.unique(y_train)
    n_events = len(event_type)  # Ne
    n_train = np.array([np.sum(y_train==et) for et in event_type])  # [Nt1,Nt2,...]
    n_chans = X_train.shape[-2]  # Nc
    n_points = X_train.shape[-1]  # Np
    n_2harmonics = sine_template.shape[1]  # 2*Nh

    # initialization
    S = np.zeros((n_events, 2*n_chans+n_2harmonics, 2*n_chans+n_2harmonics))
    Q = np.zeros_like(S)
    w, u, v, w_concat, u_concat, v_concat = [], [], [], [], [], []
    I, R = [], []

    # block covariance matrices: S & Q, (Ne,2Nc+2Nh,2Nc+2Nh)
    class_sum = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    class_center = np.zeros_like(class_sum)
    for ne, et in enumerate(event_type):
        train_trials = n_train[ne]  # Nt
        X_temp = X_train[y_train==et]  # (Nt,Nc,Np)
        class_sum[ne], class_center[ne] = X_temp.sum(axis=0), X_temp.mean(axis=0)

        XsXs = class_sum[ne] @ class_sum[ne].T  # S11
        XsXm = class_sum[ne] @ class_center[ne].T  # S12, S21.T
        XmXm = class_center[ne] @ class_center[ne].T  # S22, Q2
        XsY = class_sum[ne] @ sine_template[ne].T  # S13, S31.T
        XmY = class_center[ne] @ sine_template[ne].T  # S23, S32.T
        YY = sine_template[ne] @ sine_template[ne].T  # S33, Q3
        # XX = np.einsum('tcp,thp->ch', X_sub[ne], X_sub[ne])  # (Nc,Nc) | clear but slow
        XX = np.zeros((n_chans, n_chans))  # Q1
        for tt in range(train_trials):
            XX += X_temp[tt] @ X_temp[tt].T

        # block covariance matrices S: [[S11,S12,S13],[S21,S22,S23],[S31,S32,S33]]
        # S11: inter-trial covariance, (Nc,Nc)
        S[ne, :n_chans, :n_chans] = XsXs

        # S12 & S21.T covariance between the SSVEP trials & the individual template, (Nc,Nc)
        S[ne, :n_chans, n_chans:2*n_chans] = XsXm
        S[ne, n_chans:2*n_chans, :n_chans] = XsXm.T

        # S13 & S31.T: similarity between the SSVEP trials & sinusoidal template, (Nc,2Nh)
        S[ne, :n_chans, 2*n_chans:] = XsY
        S[ne, 2*n_chans:, :n_chans] = XsY.T

        # S23 & S32.T: covariance between the individual template & sinusoidal template, (Nc,2Nh)
        S[ne, n_chans:2*n_chans, 2*n_chans:] = XmY
        S[ne, 2*n_chans:, n_chans:2*n_chans] = XmY.T

        # S22 & S33: variance of individual template & sinusoidal template, (Nc,Nc) & (2Nh,2Nh)
        S[ne, n_chans:2*n_chans, n_chans:2*n_chans] = XmXm
        S[ne, 2*n_chans:, 2*n_chans:] = YY

        # block covariance matrices Q: blkdiag(Q1,Q2,Q3)
        # Q1: variance of the single-trial SSVEP, (Nc,Nc)
        Q[ne, :n_chans, :n_chans] = XX

        # Q2 & Q3: variance of individual template & sinusoidal template, (Nc,Nc) & (2Nh,2Nh)
        Q[ne, n_chans:2*n_chans, n_chans:2*n_chans] = XmXm
        Q[ne, 2*n_chans:, 2*n_chans:] = YY

    # GEP | train spatial filters for transfer subjects
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
        u.append(spatial_filter[:,n_chans:2*n_chans])  # (Nk,Nc) | for average template
        v.append(spatial_filter[:,2*n_chans:])  # (Nk,2Nh) | for sine-cosine template
    w_concat = np.zeros((np.sum(ndim), n_chans))  # (Ne*Nk,Nc)
    u_concat = np.zeros_like(w_concat)  # (Ne*Nk,Nc)
    v_concat = np.zeros((np.sum(ndim), n_2harmonics))  # (Ne*Nk,2Nh)
    start_idx = 0
    for ne,dims in enumerate(ndim):
        w_concat[start_idx:start_idx+dims] = w[ne]
        u_concat[start_idx:start_idx+dims] = u[ne]
        v_concat[start_idx:start_idx+dims] = v[ne]
        start_idx += dims

    # transferred individual template & reference template
    I = np.einsum('kc,ecp->ekp', u_concat, class_center)  # (Ne,Ne*Nk,Np)
    R = np.einsum('kh,ehp->ekp', v_concat, sine_template)  # (Ne,Ne*Nk,Np)
    source_model = {
        'Q':Q, 'S':S,
        'w':w, 'u':u, 'v':v,
        'w_concat':w_concat, 'u_concat':u_concat, 'v_concat':v_concat,
        'I':I, 'R':R
    }
    return source_model


def tltrca_source_compute(
    X_source: List[ndarray],
    y_source: List[ndarray],
    sine_template: ndarray,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict:
    """Intra-subject training for source-domain dataset in TL-(e)TRCA.

    Args:
        X_source (List[ndarray]): n_subjects*(n_events*n_train(train_trials), n_chans, n_points).
            Dataset of source subjects. train_trials could be 1 if necessary.
        y_source (List[ndarray]): n_subjects*(train_trials,). Labels for X.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return:
        source_model (list of dict): List of source_model.
    """
    # basic information
    n_subjects = len(X_source)

    # model integration
    source_model = []
    for nsub in range(n_subjects):
        model = tltrca_intra_compute(
            X_train=X_source[nsub],
            y_train=y_source[nsub],
            sine_template=sine_template,
            n_components=n_components,
            ratio=ratio
        )
        source_model.append(model)
        # print('Finish source subject ID: %d' %(nsub+1))
    return source_model


def tltrca_target_compute(
    X_train: ndarray,
    y_train: ndarray,
    source_model: List[dict],
    sine_template: ndarray,
    train_info: dict,
    n_components: Optional[int] = 1,
    ratio: Optional[float] = None) -> dict:
    """Inter-subject training for target-domain dataset in TL-(e)TRCA.

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if neccessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        source_model (List of dict): Details in tltrca_source_compute().
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        train_info (dict): {'event_type':ndarray (n_events,),
                            'n_events':int,
                            'n_train':ndarray (n_events,),
                            'n_chans':int,
                            'n_points':int,
                            'standard':True,
                            'ensemble':True}
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return: all contained in a dict (target_model).
        TI (List of ndarray): n_subjects*(n_events, n_events*n_components, n_chans).
            Transferred spatial filters for individual template.
        TR (List of ndarray): n_subjects*(n_events, n_events*n_components, n_chans).
            Transferred spatial filters for reference template.
        dist_I (ndarray): (n_subjects, n_events).
            Distances between transferred target data and source individual template.
        dist_R (ndarray): (n_subjects, n_events).
            Distances between transferred target data and source reference template.
        cs_I (ndarray): (n_subjects, n_events).
            Contribution scores of dI within source subjects.
        cs_R (ndarray): (n_subjects, n_events).
            Contribution scores of dR within source subjects.
        Q (ndarray): (n_events, 2*n_chans+2*n_harmonics, 2*n_chans+2*n_harmonics).
            Covariance of original data & template data.
        S (ndarray): (n_events, 2*n_chans+2*n_harmonics, 2*n_chans+2*n_harmonics).
            Covariance of template data.
        w (List of ndarray): n_events*(n_components, n_chans).
            Spatial filters for original signal.
        u (List of ndarray): n_events*(n_components, n_chans).
            Spatial filters for average template.
        v (List of ndarray): n_events*(n_components, 2*n_harmonics).
            Spatial filters for sinusoidal template.
        w_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for original signal.
        u_concat (ndarray): (n_events*n_components, n_chans).
            Concatenated filter for average template.
        v_concat (ndarray): (n_events*n_components, 2*n_harmonics).
            Concatenated filter for sinusoidal signal.
        uX (List of ndarray): n_events*(n_components, n_points).
            TL-TRCA average templates of target subject.
        vY (List of ndarray): n_events*(n_components, n_points).
            TL-TRCA sinusoidal templates of target subject.
        euX (ndarray): (n_events, total_components, n_points).
            TL-eTRCA average templates of target subject.
        evY (ndarray): (n_events, total_components, n_points).
            TL-eTRCA sinusoidal templates of target subject.
    """
    # basic information
    event_type = train_info['event_type']
    n_events = train_info['n_events']  # Ne
    n_train = train_info['n_train']  # [Nt1,Nt2,...]
    n_chans = train_info['n_chans']  # Nc
    n_points = train_info['n_points']  # Np
    standard = train_info['standard']  # bool
    ensemble = train_info['ensemble']  # bool
    n_subjects = len(source_model)

    # transferred learning for target subject
    TI, TR = [], []
    dist_I = np.zeros((n_subjects, n_events))  # (Ns,Ne)
    dist_R = np.zeros_like(dist_I)
    cs_I, cs_R = np.zeros_like(dist_I), np.zeros_like(dist_I)
    class_center = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
    for nsub in range(n_subjects):
        I, R = source_model[nsub]['I'], source_model[nsub]['R']  # (Ne,Ne*Nk,Np)
        TI_sub = np.zeros((n_events, I.shape[1], n_chans))  # (Ne,Ne*Nk,Nc)
        TR_sub = np.zeros((n_events, R.shape[1], n_chans))  # (Ne,Ne*Nk,Nc)
        for ne, et in enumerate(event_type):
            X_temp = X_train[y_train==et]  # (Nt,Nc,Np)
            class_center[ne] = X_temp.mean(axis=0)  # (Nc,Np)
            train_trials = n_train[ne]  # Nt

            # transferred spatial filters for individual template & reference template
            for tt in range(train_trials):
                TI_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=I[ne].T)  # (Nc,Ne*Nk)
                TR_temp, _, _, _ = sLA.lstsq(a=X_temp[tt].T, b=R[ne].T)  # (Nc,Ne*Nk)
                TI_sub[ne] += TI_temp
                TR_sub[ne] += TR_temp
            TI_sub[ne] /= train_trials
            TR_sub[ne] /= train_trials

            # distances between transferred target data & source template
            for tt in range(train_trials):
                dist_I[nsub,ne] += utils.pearson_corr(TI_sub[ne] @ X_temp[tt], I[ne])
                dist_R[nsub,ne] += utils.pearson_corr(TR_sub[ne] @ X_temp[tt], R[ne])
        TI.append(TI_sub)
        TR.append(TR_sub)

    # contribution scores
    cs_I = dist_I / np.sum(dist_I, axis=0, keepdims=True)
    cs_R = dist_R / np.sum(dist_R, axis=0, keepdims=True)

    # self-training for target subject
    results = tltrca_intra_compute(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            n_components=n_components,
            ratio=ratio
        )
    u, v = results['u'], results['v']
    u_concat, v_concat = results['u_concat'], results['v_concat']
    
    # self-trained templates: EEG template (u@X) & sinusoidal template (v@Y)
    uX, vY = [], []
    euX = np.zeros((n_events, u_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
    evY = np.zeros_like(euX)
    if standard:
        for ne in range(n_events):
            uX.append(u[ne] @ class_center[ne])  # (Nk,Np)
            vY.append(v[ne] @ sine_template[ne])  # (Nk,Np)
    if ensemble:
        for ne in range(n_events):
            euX[ne] = u_concat @ class_center[ne]  # (Nk*Ne,Np)
            evY[ne] = v_concat @ sine_template[ne]  # (Nk*Ne,Np)
    target_model = {
        'TI':TI, 'TR':TR,
        'dist_I':dist_I, 'dist_R':dist_R,
        'cs_I':cs_I, 'cs_R':cs_R,
        'Q':results['Q'], 'S':results['S'],
        'w':results['w'], 'u':u, 'v':v,
        'w_concat':results['w_concat'], 'u_concat':u_concat, 'v_concat':v_concat,
        'uX':uX, 'vY':vY, 'euX':euX, 'evY':evY
    }
    return target_model


class TL_TRCA(BasicTRCA):
    def fit(self, X_train, y_train, source_model, sine_template):
        """Train TL-(e)TRCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if necessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            source_model (List of dict): Details in tltrca_compute_source().
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
                Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.source_model = source_model
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {
            'event_type':event_type,
            'n_events':len(event_type),
            'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
            'n_chans':self.X_train.shape[-2],
            'n_points':self.X_train.shape[-1],
            'standard':self.standard,
            'ensemble':self.ensemble
        }

        # train transferred model for target subject
        target_model = tltrca_target_compute(
            X_train=self.X_train,
            y_train=self.y_train,
            source_model=self.source_model,
            sine_template=sine_template,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.TI, self.TR = target_model['TI'], target_model['TR']
        self.dist_I, self.dist_R = target_model['dist_I'], target_model['dist_R']
        self.cs_I, self.cs_R = target_model['cs_I'], target_model['cs_R']
        self.w, self.u, self.v = target_model['w'], target_model['u'], target_model['v']
        self.w_concat = target_model['w_concat']
        self.u_concat = target_model['u_concat']
        self.v_concat = target_model['v_concat']
        self.uX, self.vY = target_model['uX'], target_model['vY']
        self.euX, self.evY = target_model['euX'], target_model['evY']
        return self


    def predict(self, X_test):
        """Using TL-(e)TRCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if necessary.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients of TL-TRCA.
                Not empty when self.standard is True.
            y_standard (ndarray): (test_trials,). Predict labels of TL-TRCA.
            erou (ndarray): (test_trials, n_events). Decision coefficients of TL-eTRCA.
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (test_trials,). Predict labels of TL-eTRCA.
        """
        # basic information
        n_test = X_test.shape[0]
        n_events = self.train_info['n_events']

        self.rou = np.zeros((n_test, n_events, 4))
        self.final_rou = np.zeros((n_test, n_events))
        self.erou = np.zeros_like(self.rou)
        self.final_erou = np.zeros_like(self.final_rou)
        self.y_standard = np.empty((n_test))
        self.y_ensemble = np.empty_like(self.y_standard)
        n_subjects = len(self.source_model)

        # rou 1 & 2: transferred pattern matching
        for nte in range(n_test):
            for nem in range(n_events):
                for nsub in range(n_subjects):
                    self.rou[nte,nem,0] += self.cs_I[nsub,nem]*utils.pearson_corr(
                        X=self.TI[nsub][nem] @ X_test[nte],
                        Y=self.source_model[nsub]['I'][nem]
                    )
                    self.rou[nte,nem,1] += self.cs_R[nsub,nem]*utils.pearson_corr(
                        X=self.TR[nsub][nem] @ X_test[nte],
                        Y=self.source_model[nsub]['R'][nem]
                    )

        # rou 3 & 4: self-trained pattern matching (similar to sc-(e)TRCA)
        if self.standard:
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_standard = self.w[nem] @ X_test[nte]
                    self.rou[nte,nem,2] = utils.pearson_corr(
                        X=temp_standard,
                        Y=self.uX[nem]
                    )
                    self.rou[nte,nem,3] = utils.pearson_corr(
                        X=temp_standard,
                        Y=self.vY[nem]
                    )
                    self.final_rou[nte,nem] = utils.combine_feature([
                        self.rou[nte,nem,0],
                        self.rou[nte,nem,1],
                        self.rou[nte,nem,2],
                        self.rou[nte,nem,3],
                    ])
                self.y_standard[nte] = np.argmax(self.final_rou[nte,:])
        if self.ensemble:
            self.erou[...,0], self.erou[...,1] = self.rou[...,0], self.rou[...,1]
            for nte in range(n_test):
                for nem in range(n_events):
                    temp_ensemble = self.w_concat @ X_test[nte]
                    self.erou[nte,nem,2] = utils.pearson_corr(
                        X=temp_ensemble,
                        Y=self.euX
                    )
                    self.erou[nte,nem,3] = utils.pearson_corr(
                        X=temp_ensemble,
                        Y=self.evY
                    )
                    self.final_erou[nte,nem] = utils.combine_feature([
                        self.erou[nte,nem,0],
                        self.erou[nte,nem,1],
                        self.erou[nte,nem,2],
                        self.erou[nte,nem,3],
                    ])
                self.y_ensemble[nte] = np.argmax(self.final_erou[nte,:])
        return self.rou, self.y_standard, self.erou, self.y_ensemble


def fb_tltrca_source_compute(X_source, y_source, sine_template, n_components=1, ratio=None):
    """Intra-subject training for source-domain dataset in FB-TL-(e)TRCA.

    Args:
        X_source (List-like object): n_subjects*(n_bands, train_trials, n_chans, n_points).
            Dataset of source subjects. train_trials could be 1 if necessary.
        y_source (List-like object): n_subjects*(train_trials,). Labels for X.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        n_components (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Return:
        source_model (List of dict): List of source_model.
            Add a dimension of n_bands for each variable in source_model.
    """
    # basic information
    n_subjects = len(X_source)
    n_bands = X_source[0].shape[0]

    # model integration
    fb_source_model = [[] for nb in range(n_bands)]
    for nb in range(n_bands):
        for nsub in range(n_subjects):
            model = tltrca_intra_compute(
                X_train=X_source[nsub][nb],
                y_train=y_source[nsub],
                sine_template=sine_template,
                n_components=n_components,
                ratio=ratio
            )
            fb_source_model[nb].append(model)
    return fb_source_model


class FB_TL_TRCA(BasicFBTRCA):
    def fit(self, X_train, y_train, fb_source_model, sine_template):
        """Train filter-bank TL-(e)TRCA model.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if necessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            fb_source_model (List of dict): Details in tltrca_compute_source().
                Add a dimension of n_bands for each variable in source_model.
            sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.fb_source_model = fb_source_model
        self.n_bands = X_train.shape[0]

        # train TL-TRCA models
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = TL_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                source_model=self.fb_source_model[nb],
                sine_template=sine_template
            )
        return self


# %% 3. subject transfer based CCA, stCCA
def stcca_intra_compute(X_train, y_train, sine_template, n_components=1, ratio=None):
    # basic information
    event_type = np.unique(y_train)
    n_events = len(event_type)  # Ne
    n_train = np.array([np.sum(y_train==et) for et in event_type])  # [Nt1,Nt2,...]
    n_chans = X_train.shape[-2]  # Nc
    n_points = X_train.shape[-1]  # Np
    n_2harmonics = sine_template.shape[1]  # 2*Nh
    
    # GEPs' conditions
    events_group = utils.augmented_events(event_type, n_events)  # 