# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

Other design.
    (1) DSP: https://ieeexplore.ieee.org/document/8930304/
            DOI: 10.1109/TBME.2019.2958641
    (2) DCPM: https://ieeexplore.ieee.org/document/8930304/
            DOI: 10.1109/TBME.2019.2958641
    (3) PT projection: None
            DOI: None
    (4) TDCA: https://ieeexplore.ieee.org/document/9541393/
            DOI: 10.1109/TNSRE.2021.3114340


update: 2022/10/22

"""

# %% basic modules
import utils

import numpy as np

from scipy import linalg as sLA

# %% (1) Discriminant Spatial Patterns | DSP
def dsp_compute(X_train, y_train, train_info, n_components=1, ratio=None):
    """Discriminant Spatial Patterns (DSP).

    Args:
        X_train (ndarray): (n_events*n_train(train_trials), n_chans, n_points).
            Training dataset. train_trials could be 1 if necessary.
        y_train (ndarray): (train_trials,). Labels for X_train.
        train_info (dict): {'event_type':ndarray (n_events,),
                            'n_events':int,
                            'n_train':ndarray (n_events,),
                            'n_chans':int,
                            'n_points':int}
        n_components (int, optional): Number of eigenvectors picked as filters.
            Defaults to 1. Set to 'None' if ratio is not 'None'.
        ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
            Defaults to None when n_component is not 'None'.

    Returns:
        Sb (ndarray): (n_chans, n_chans). Scatter matrix of between-class difference.
        Sw (ndarray): (n_chans, n_chans). Scatter matrix of within-class difference.
        w (ndarray): (n_components, n_chans). Common spatial filter.
        template (ndarray): (n_events, n_components, n_points). DSP templates.
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
        class_center[ne] = X_train[y_train==et].mean(axis=0)  # (:,Nc,Np)
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
    # Sw = einsum('etcp,ethp->ch', Hw,Hw)/(n_events*n_train) | only when events for each type are the same

    # GEPs | train spatial filter
    w = utils.solve_gep(
        A=Sb,
        B=Sw,
        n_components=n_components,
        ratio=ratio
    )  # (Nk,Nc)

    # signal templates
    template = np.einsum('kc,ecp->ekp', w,class_center)  # (Ne,Nk,Np)
    return Sb, Sw, w, template


class DSP(object):
    def __init__(self, n_components=1, ratio=None):
        """Config model dimension.

        Args:
            n_components (int, optional): Number of eigenvectors picked as filters.
                Defaults to 1. Set to 'None' if ratio is not 'None'.
            ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
                Defaults to None when n_component is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio


    def fit(self, X_train, y_train):
        """Train DSP model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if necessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.train_info = {'event_type':event_type,
                           'n_events':len(event_type),
                           'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
                           'n_chans':self.X_train.shape[-2],
                           'n_points':self.X_train.shape[-1]}

        # train DSP models & templates
        self.Sb, self.Sw, self.w, self.template = dsp_compute(
            X_train=self.X_train,
            y_train = self.y_train,
            train_info = self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        return self


    def predict(self, X_test):
        """Using DSP algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if necessary.

        Return:
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
            f_test = self.w @ X_test[nte]  # (Nk,Np)
            for ne in range(n_events):
                self.rou[nte,ne] = utils.pearson_corr(
                    X=f_test,
                    Y=self.template[ne]
                )
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


# %% (2) Discriminant Canonical Pattern Matching | DCPM



# %% (3) PT projection
def pt_proj(X, theta):
    """
    Compute the PT projection matrix

    Args:
        X (ndarray): (n_train, n_chans, n_times)
        theta (float): Hyper-parameter. 0-1.

    Returns:
        projection (ndarray): (n_chans, n_chans)
    """
    # basic information
    n_train = X.shape[0]

    # projection formula
    A = np.einsum('tcp,thp->ch', X,X)  # (Nc,Nc)
    A /= n_train
    Xmean = X.mean(axis=0)  # (Nc,Np)
    B = Xmean @ Xmean.T  # (Nc,Nc)
    projection = sLA.solve(theta*A + (1-2*theta)*B, (1-theta)*B)
    return projection.T


# %% Task-discriminant component analysis | TDCA
def aug_2(data, projection, extra_length, mode='train'):
    """Construct secondary augmented data.

    Args:
        data (ndarray): (n_chans, n_points+m or n_points).
            m must be larger than n_points while mode is 'train'.
        projection (ndarray): (n_points, n_points). Y.T@Y
        extra_length (int): Extra data length.
        mode (str, optional): 'train' or 'test'.

    Returns:
        data_aug2 (ndarray): ((m+1)*n_chans, 2*n_points).
    """
    # basic information
    n_chans = data.shape[0]  # Nc
    n_points = projection.shape[0]  # Np

    # secondary augmented data
    data_aug2 = np.zeros(((extra_length+1)*n_chans, 2*n_points))  # ((m+1)*Nc,2*Np)
    if mode == 'train':
        for el in range(extra_length+1):
            sp, ep = el*n_chans, (el+1)*n_chans
            data_aug2[sp:ep,:n_points] = data[:,el:n_points+el]  # augmented data
            data_aug2[sp:ep,n_points:] = data_aug2[sp:ep,:n_points] @ projection
    elif mode == 'test':
        for el in range(extra_length+1):
            sp, ep = el*n_chans, (el+1)*n_chans
            data_aug2[sp:ep,:n_points-el] = data[:,el:n_points]
            data_aug2[sp:ep,n_points:] = data_aug2[sp:ep,:n_points] @ projection
    return data_aug2


class TDCA(object):
    """Task-discriminant component analysis."""
    def __init__(self, n_components=1, ratio=None):
        """Config model dimension.

        Args:
            n_components (int, optional): Number of eigenvectors picked as filters.
                Defaults to 1. Set to 'None' if ratio is not 'None'.
            ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
                Defaults to None when n_component is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio


    def fit(self, X_train, y_train, projection, extra_length):
        """Train TDCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points+extra_length).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            projection (ndarray): (n_events, n_points, n_points).
                Orthogonal projection matrices.
            extra_length (int).
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        event_type = np.unique(y_train)  # [0,1,2,...,Ne-1]
        self.projection = projection
        self.extra_length = extra_length

        # create secondary augmented data | (Ne*Nt,(el+1)*Nc,2*Np)
        self.X_train_aug2 = np.zeros((len(self.y_train),
                                      (self.extra_length+1)*self.X_train.shape[-2],
                                      2*(self.X_train.shape[-1] - self.extra_length)))
        for ntr in range(len(self.y_train)):
            self.X_train_aug2[ntr] = aug_2(
                data=self.X_train[ntr],
                projection=self.projection[y_train[ntr]],
                extra_length=self.extra_length
            )
        self.train_info = {'event_type':event_type,
                           'n_events':len(event_type),
                           'n_train':np.array([np.sum(self.y_train==et) for et in event_type]),
                           'n_chans':self.X_train_aug2.shape[-2],
                           'n_points':self.X_train_aug2.shape[-1]}

        # train DSP models & templates
        results = dsp_compute(
            X_train=self.X_train_aug2,
            y_train=self.y_train,
            train_info=self.train_info,
            n_components=self.n_components,
            ratio=self.ratio
        )
        self.Sb, self.Sw = results[0], results[1]
        self.w, self.template = results[2], results[3]
        return self


    def predict(self, X_test, y_test):
        """Using TDCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Return:
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
                temp_test_aug2 = aug_2(
                    data=X_test[nte],
                    projection=self.projection[ne],
                    extra_length=self.extra_length,
                    mode='test'
                )
                self.rou[nte,ne] = utils.pearson_corr(
                    X=self.w @ temp_test_aug2,
                    Y=self.template[ne]
                )
                self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


# %% Filter-bank TRCA series | FB-
class FB_DSP(DSP):
    def fit(self, X_train, y_train):
        """Train filter-bank DSP model.

        Args:
            X_train (ndarray): (n_bands, train_trials, n_chans, n_points).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.n_bands = X_train.shape[0]

        # train DSP models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = DSP(
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train
            )
        return self


    def predict(self, X_test, y_test):
        """Using filter-bank DSP algorithm to predict test data.

        Args:
            X_test (ndarray): (n_bands, n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[1]
        
        # apply DSP().predict() in each sub-band
        self.fb_rou = [[] for nb in range(self.n_bands)]
        self.fb_y_predict = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.fb_rou[nb], self.fb_y_predict[nb] = self.sub_models[nb].predict(
                X_test=X_test[nb],
                y_test=y_test
            )

        # integration of multi-bands' results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict


class FB_TDCA(TDCA):
    def fit(self, X_train, y_train, projection, extra_length):
        """Train TDCA model.

        Args:
            X_train (ndarray): (train_trials, n_chans, n_points+extra_length).
                Training dataset. train_trials could be 1 if neccessary.
            y_train (ndarray): (train_trials,). Labels for X_train.
            projection (ndarray): (n_events, n_points, n_points).
                Orthogonal projection matrices.
            extra_length (int).
        """
        # basic information
        self.X_train = X_train
        self.y_train = y_train
        self.projection = projection
        self.extra_length = extra_length
        self.n_bands = X_train.shape[0]

        # train TDCA models & templates
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = TDCA(
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train,
                projection=self.projection,
                extra_length=self.extra_length
            )
        return self


    def predict(self, X_test, y_test):
        """Using filter-bank TDCA algorithm to predict test data.

        Args:
            X_test (ndarray): (n_bands, n_events*n_test(test_trials), n_chans, n_points).
                Test dataset. test_trials could be 1 if neccessary.
            y_test (ndarray): (test_trials,). Labels for X_test.

        Return:
            rou (ndarray): (test_trials, n_events). Decision coefficients.
            y_predict (ndarray): (test_trials,). Predict labels.
        """
        # basic information
        n_test = X_test.shape[1]

        # apply TDCA().predict() in each sub-band
        self.fb_rou = [[] for nb in range(self.n_bands)]
        self.fb_y_predict = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.fb_rou[nb], self.fb_y_predict[nb] = self.sub_models[nb].predict(
                X_test=X_test[nb],
                y_test=y_test
            )

        # integration of multi-bands' results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.y_predict = np.empty((n_test))
        for nte in range(n_test):
            self.y_predict[nte] = np.argmax(self.rou[nte,:])
        return self.rou, self.y_predict