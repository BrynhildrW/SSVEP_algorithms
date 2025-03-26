# -*- coding: utf-8 -*-
"""
@ Author: Brynhildr Wu
@ Email: brynhildrwu@gmail.com

Spatialtemporal domain adaptation based on common latent subspace.

Refers:
    [1]: http://arxiv.org/abs/1612.01939

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
import utils
import cca
import trca
import dsp

from typing import Optional, Tuple, Dict, List
from numpy import ndarray

import numpy as np
import scipy.linalg as sLA
from copy import deepcopy


# %% Pre-functions: marginal distribution alignment
def standardization(X: ndarray) -> ndarray:
    """
    Trial- & channel-level data standardization.

    Parameters
    -------
    X : ndarray, shape (Ne*Nt,Nc,Np).
        Input dataset.

    Returns
    -------
    X_sta : ndarray, shape (Ne*Nt,Nc,Np).
        Data after standardization.
    """
    X_sta = np.zeros_like(X)  # (Ne*Nt,Nc,Np)
    for nt in range(X.shape[0]):
        # trial centralization
        X_sta[nt] = X[nt] - np.mean(X[nt])  # (Nc,Np)

        # channel normalization
        ch_std = np.diag(1 / np.std(X_sta[nt], axis=-1))  # (Nc,Nc)
        X_sta[nt] = ch_std @ X_sta[nt]  # (Nc,Np)
    return X_sta


# %% Pre-functions: conditional distribution alignment
def joint_cca_kernel(
        X_target: ndarray,
        y_target: ndarray,
        X_source: ndarray,
        y_source: ndarray,
        template: Optional[ndarray] = None,
        n_components: int = 1,
        theta: float = 0.5) -> Dict[str, ndarray]:
    """
    Calculate joint-CCA filters to obtain source activity on latent subspace.

    Parameters
    -------
    X_target : ndarray, shape (Ne*Nt(t),Nc,Np).
        Target dataset. Nt>=2.
    y_target : ndarray, shape (Ne*Nt(t),).
        Labels for X_target.
    X_source : ndarray, shape (Ne*Nt(s),Nc,Np).
        Source dataset of one subject.
    y_source : ndarray, shape (Ne*Nt(s),).
        Labels for X_source.
    template : ndarray, shape (Ne,m,Np).
        m could be 2*Nh while template is sinusoidal.
    n_components : int.
        Number of eigenvectors picked as filters (Nk). Defaults to 1.
    theta : float, 0-1.
        Hyper-parameter to control the use of source dataset. Defaults to 0.5.

    Returns
    -------
    Cxx : ndarray, shape (Ne,Nc,Nc).
        Covariance of EEG data.
    Cxy : ndarray, shape (Ne,Nc,m).
        Covariance of EEG & template.
    Cyy : ndarray, shape (Ne,m,m).
        Covariance of template.
    u : ndarray, shape (Ne,Nk,Nc).
        Spatial filter for EEG.
    v : ndarray, shape (Ne,Nk,m).
        Spatial filter for template.
    """
    # basic information
    n_events, n_dims, _ = template.shape  # Ne, m, Np
    n_chans = X_source.shape[-2]  # Nc

    # Cxx, Cxy & Cyy of target- & source-domain dataset (joint)
    X_target_mean = utils.generate_mean(X=X_target, y=y_target)  # (Ne,Nc,Np)
    X_source_mean = utils.generate_mean(X=X_source, y=y_source)  # (Ne,Nc,Np)
    Cxx = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    Cxy = np.zeros((n_events, n_chans, n_dims))  # (Ne,Nc,m)
    Cyy = np.zeros((n_events, n_dims, n_dims))  # (Ne,m,m)
    u = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    v = np.zeros((n_events, n_components, n_dims))  # (Ne,Nk,m)
    for ne in range(n_events):
        Cxx[ne] = (1 - theta) * X_target_mean[ne] @ X_target_mean[ne].T \
            + theta * X_source_mean[ne] @ X_source_mean[ne].T
        Cxy[ne] = (1 - theta) * X_target_mean[ne] @ template[ne].T \
            + theta * X_source_mean[ne] @ template[ne].T
        Cyy[ne] = template[ne] @ template[ne].T

        # GEPs | train spatial filters
        u[ne], v[ne] = cca.solve_cca_func(
            Cxx=Cxx[ne],
            Cxy=Cxy[ne],
            Cyy=Cyy[ne],
            n_components=n_components
        )
    return {'Cxx': Cxx, 'Cxy': Cxy, 'Cyy': Cyy, 'u': u, 'v': v}


def joint_trca_kernel(
        X_target: ndarray,
        y_target: ndarray,
        X_source: ndarray,
        y_source: ndarray,
        template: Optional[ndarray] = None,
        n_components: int = 1,
        theta: float = 0.5) -> Dict[str, ndarray]:
    """
    Calculate joint-TRCA filters to obtain source activity on latent subspace.

    Parameters
    -------
    X_target : ndarray, shape (Ne*Nt(t),Nc,Np).
        Target dataset. Nt>=2.
    y_target : ndarray, shape (Ne*Nt(t),).
        Labels for X_target.
    X_source : ndarray, shape (Ne*Nt(s),Nc,Np).
        Source dataset of one subject.
    y_source : ndarray, shape (Ne*Nt(s),).
        Labels for X_source.
    template : Nonetype.
        Only for code consistency.
    n_components : int.
        Number of eigenvectors picked as filters (Nk). Defaults to 1.
    theta : float, 0-1.
        Hyper-parameter to control the use of source dataset. Defaults to 0.5.

    Returns
    -------
    Q : ndarray, shape (Ne,Nc,Nc).
        Covariance of original data.
    S : ndarray, shape (Ne,Nc,Nc).
        Covariance of template data.
    w : ndarray, shape (Ne,Nk,Nc).
        Spatial filters of joint-TRCA.
    ew : ndarray, shape (Ne*Nk,Nc).
        Common spatial filter of joint-eTRCA.
    """
    # S & Q of target- & source-domain dataset (joint)
    Q_target, S_target, _ = trca.generate_trca_mat(X=X_target, y=y_target)  # (Ne,Nc,Nc)
    Q_source, S_source, _ = trca.generate_trca_mat(X=X_source, y=y_source)  # (Ne,Nc,Nc)
    Q = (1 - theta) * Q_target + theta * Q_source
    S = (1 - theta) * S_target + theta * S_source

    # GEPs | train spatial filters
    w, ew = trca.solve_trca_func(Q=Q, S=S, n_components=n_components)
    return {'Q': Q, 'S': S, 'w': w, 'ew': ew}


def joint_dsp_kernel(
        X_target: ndarray,
        y_target: ndarray,
        X_source: ndarray,
        y_source: ndarray,
        template: Optional[ndarray] = None,
        n_components: int = 1,
        theta: float = 0.5) -> Dict[str, ndarray]:
    """
    Calculate joint-DSP filter to obtain source response on latent subspace.

    Parameters
    -------
    X_target : ndarray, shape (Ne*Nt(t),Nc,Np).
        Target dataset. Nt>=2.
    y_target : ndarray, shape (Ne*Nt(t),).
        Labels for X_target.
    X_source : ndarray, shape (Ne*Nt(s),Nc,Np).
        Source dataset of one subject.
    y_source : ndarray, shape (Ne*Nt(s),).
        Labels for X_source.
    template : Nonetype.
        Only for code consistency.
    n_components : int.
        Number of eigenvectors picked as filters (Nk). Defaults to 1.
    theta : float, 0-1.
        Hyper-parameter to control the use of source dataset. Defaults to 0.5.

    Returns
    -------
    Sb : ndarray, shape (Nc,Nc).
        Scatter matrix of between-class difference.
    Sw : ndarray, shape (Nc,Nc).
        Scatter matrix of within-class difference.
    w : ndarray, shape (Ne,Nk,Nc).
        Spatial filter of joint-DSP.
    """
    # basic information
    n_events = len(np.unique(y_source))  # Ne

    # Sb & Sw of target & source dataset
    Sb_target, Sw_target, _ = dsp.generate_dsp_mat(X=X_target, y=y_target)
    Sb_source, Sw_source, _ = dsp.generate_dsp_mat(X=X_source, y=y_source)
    Sb = (1 - theta) * Sb_target + theta * Sb_source
    Sw = (1 - theta) * Sw_target + theta * Sw_source

    # GEPs | train spatial filter
    w = np.tile(
        A=utils.solve_gep(A=Sb, B=Sw, n_components=n_components),
        reps=(n_events, 1, 1)
    )  # (Ne,Nk,Nc)

    # backward-propagation model
    return {'Sb': Sb, 'Sw': Sw, 'w': w}


def solve_forward_propagation(
        X: ndarray,
        y: ndarray,
        w: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Calculate propagation matrice(A) based on source responses(wX).
    A = argmin||A @ w @ X - X||.

    Parameters
    -------
    X : ndarray, shape (Ne*Nt,Nc,Np).
        Multi-trial dataset. Nt>=2.
    y : ndarray, shape (Ne*Nt,).
        Labels for X.
    w : ndarray, shape (Ne,Nk,Nc).
        Spatial filters.

    Returns
    -------
    A : ndarray, shape (Ne,Nc,Nk).
        Propagation matrices.
    s : ndarray, shape (Ne*Nt,Nk,Np).
        Spatial filtered X (wX).
    """
    # basic information
    event_type = list(np.unique(y))
    n_events, n_components, n_chans = w.shape  # Ne,Nk,Nc
    n_trials, _, n_points = X.shape  # Ne*Nt,Nc,Np

    # construct task-related signal wX (s)
    s = np.zeros((n_trials, n_components, n_points))  # (Ne*Nt,Nk,Np)
    for ntr in range(n_trials):
        event_idx = event_type.index(y[ntr])
        s[ntr] = w[event_idx] @ X[ntr]

    # analytical solution
    X_var = utils.generate_var(X=X, y=y)  # (Ne,Nc,Nc)
    s_var = utils.generate_var(X=s, y=y)  # (Ne,Nk.Nk)
    A = np.zeros((n_events, n_chans, n_components))
    for ne in range(n_events):
        A[ne] = X_var[ne] @ w[ne].T @ sLA.inv(s_var[ne])
    return A, s


def construct_prototype_filter(
        ws: ndarray,
        preprocessed: bool = True) -> ndarray:
    """
    Construct prototye filters from given ws.
    Refer: https://ieeexplore.ieee.org/document/8616087/.

    Parameters
    -------
    ws : ndarray, shape (m,Nk,Nc).
        Spatial filters. m is the total number of filters of shape (Nk,Nc).
    preprocessed : bool.
        Whether the F-norm of w has been compressed to 1. Defaults to True.

    Returns
    -------
    u : ndarray, shape (Nk,Nc).
        Prototye filter.
    """
    # basic information
    n_filters, n_components, n_chans = ws.shape  # m, Nk, Nc
    u = np.zeros((n_components, n_chans))  # (Nk,Nc)

    # F-norm normalization
    wn = deepcopy(ws)
    if not preprocessed:
        for nf in range(n_filters):
            for nk in range(n_components):
                wn[nf, nk, :] /= sLA.norm(ws[nf, nk, :])
        del nf, nk

    # analytic solution
    for nk in range(n_components):
        Cuu = np.zeros((n_chans, n_chans))  # (Nc,Nc)
        for nf in range(n_filters):
            Cuu += wn[nf, nk, :][:, None] @ wn[nf, nk, :][None, :]
        u[nk] = utils.solve_ep(A=Cuu, n_components=1, mode='Max')
    return u


def solve_spatial_alignment(
        X_source: ndarray,
        y_source: ndarray,
        s_target_mean: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Calculate spatial alignment filters for diving into common latent space.
    U = argmin||U @ X - s||_F^2

    Parameters
    -------
    X_source : ndarray, shape (Ne*Nt,Nc(s),Np).
        Source-domain original dataset.
    y_source : ndarray, shape (Ne*Nt,).
        Labels of X_source.
    s_target_mean : ndarray, shape (Ne,Nk,Np).
        Target-domain spatial-filtered templates.

    Returns
    -------
    u_source : ndarray, shape (Ne,Nk,Nc(s)).
        Prototype filters for spatial alignment.
    uX_source : ndarray, shape (Ne*Nt,Nk,Np).
        Spatial aligned X_source.
    """
    # basic information
    event_type = list(np.unique(y_source))
    n_trials, n_chans, n_points = X_source.shape  # Ne*Nt,Nc(s),Np
    n_events, n_dims, _ = s_target_mean.shape  # Ne,Nk,Np

    # analytical solution for u
    us = [[] for ne in range(n_events)]
    for ntr in range(n_trials):
        idx = event_type.index(y_source[ntr])
        us[idx].append(
            s_target_mean[idx] @ X_source[ntr].T @ sLA.inv(X_source[ntr] @ X_source[ntr].T)
        )

    # construct prototype filters
    u_source = np.zeros((n_events, n_dims, n_chans))  # (Ne,Nk,Nc(s))
    for ne in range(n_events):
        u_source[ne] = construct_prototype_filter(
            ws=np.stack(us[ne]),
            preprocessed=False
        )  # (Nk,Nc)

    # spatial alignment for X_source
    uX_source = np.zeros((n_trials, n_dims, n_points))  # (Ne*Nt,Nk,Np)
    for ntr in range(n_trials):
        idx = event_type.index(y_source[ntr])
        uX_source[ntr] = u_source[idx] @ X_source[ntr]
    return u_source, uX_source


def solve_temporal_alignment(
        s_target_mean: ndarray,
        s_source: ndarray,
        y_source: ndarray,
        s_source_mean: ndarray,
        rho: float = 0.1) -> ndarray:
    """
    Calculate time-domain projection matrice P.
    S = argmin|| s_source @ P - s_target ||_F^2 + rho * ||P||_F^2.

    Parameters
    -------
    s_target_mean : ndarray, shape (Ne,Nk,Np).
        Trial-averaged S_target.
    s_source : ndarray, shape (Ne*Nt(s),Nk,Np).
        Source-domain source responses.
    y_source : ndarray, shape (Ne*Nt(s),).
        Labels for S_source.
    s_source_mean : ndarray, shape (Ne,Nk,Np).
        Trial-averaged S_source.
    rho : float, 0-1.
        L2 regularization parameter to control the size of P.

    Returns
    -------
    P : ndarray, shape (Ne,Np,Np).
        Projection matrix.
    """
    # basic information
    n_events = s_source_mean.shape[0]  # Ne
    n_points = s_source_mean.shape[-1]  # Np

    # analytical solution
    var_temp = utils.generate_var(
        X=np.transpose(a=s_source, axes=(0, 2, 1)),
        y=y_source
    )  # (Ne*Nt,Nc,Np) -transpose-> (Ne*Nt,Np,Nc) -var-> (Ne,Np,Np)
    P = np.zeros((n_events, n_points, n_points))  # (Ne,Np,Np)
    for ne in range(n_events):
        temp = (1 - rho) * sLA.inv((1 - rho) * var_temp[ne] + rho * np.eye(n_points))
        P[ne] = temp @ s_source_mean[ne].T @ s_target_mean[ne]
    return P


def solve_noise_estimation(
        X_source: ndarray,
        y_source: ndarray,
        X_target: ndarray,
        y_target: ndarray,
        method: str = 'DSP',
        n_components: int = 1) -> ndarray:
    """
    Assume transformation matrix C satisfies the following conditions: <p>
    (1) Source domain: N = X - A @ w @ X = X - A @ s <p>
    (2) Target domain: X = A @ s + C.T @ N <p>
    By solving problem: <p>
        C = min || (A @ s + C.T @ N) (A @ s + C.T @ N).T - Var ||_F^2
          = min || C.T (N @ N.T) C - Var + A @ s (A @ s).T ||_F^2 <p>
    To get the esimation of the target-domain noise: N (target) = C.T @ N (source)

    Parameters
    -------
    X_source : ndarray, shape (Ne*Nt(s),Nc,Np).
        Source-domain dataset.
    y_source : ndarray, shape (Ne*Nt(s),).
        Labels for X_source.
    X_target : ndarray, shape (Ne*Nt(t),Nc,Np).
        Target-domain dataset.
    y_target : ndarray, shape (Ne*Nt(t),).
        Labels for X_target.
    method : str.
        Models for constructing A & w. Support 'DSP', 'TRCA' and 'eTRCA' now.
    n_components : int.
        Number of eigenvectors picked as filters (Nk). Defaults to 1.

    Returns
    -------
    C : ndarray, shape (Ne,Nc,Nc).
        Transformation matrices.
    N_target : ndarray, shape (Ne*Nt(s),Nc,Np).
        Estimated noise of target-domain data.
    """
    # basic information
    event_type = list(np.unique(y_source))
    n_events = len(event_type)  # Ne
    n_trials, n_chans = X_source.shape[0], X_source.shape[1]  # Ne*Nt,Nc

    # construct source-domain classification models
    if method == 'DSP':
        model = dsp.DSP(n_components=n_components)
        model.fit(X_train=X_source, y_train=y_source)
        w_source = np.tile(model.training_model['w'], (n_events, 1, 1))  # (Ne,Nk,Nc)
        wX_source = model.training_model['wX']  # (Ne,Nk,Np)
    else:  # TRCA or eTRCA
        model = trca.TRCA(n_components=n_components)
        model.fit(X_train=X_source, y_train=y_source)
        if method == 'TRCA':
            w_source = model.training_model['w']  # (Ne,Nk,Nc)
            wX_source = model.training_model['wX']  # (Ne,Nk,Np)
        elif method == 'eTRCA':
            w_source = model.training_model['ew']  # (Ne,Ne*Nk,Nc)
            wX_source = model.training_model['ewX']  # (Ne,Ne*Nk,Np)

    # construct source-domain forward-propagation matrix
    A_source, _ = solve_forward_propagation(
        X=X_source,
        y=y_source,
        w=w_source
    )  # (Ne,Nc,Nk) or (Ne,Nc,Ne*Nk)

    # construct source-domain background noise signal
    X_source_mean = utils.generate_mean(X=X_source, y=y_source)  # (Ne,Nc,Np)
    N_source = np.zeros_like(X_source_mean)  # (Ne,Nc,Np)
    As_source = np.zeros_like(X_source_mean)  # (Ne,Nc,Np)
    for ne in range(n_events):
        As_source[ne] = A_source[ne] @ wX_source[ne]
        N_source[ne] = X_source_mean[ne] - As_source[ne]

    # analytical solution of transformation matrices C
    C = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    X_target_var = utils.generate_var(X=X_target, y=y_target)  # (Ne,Nc,Nc)
    for ne in range(n_events):
        C[ne] = utils.solve_coral(
            Cs=N_source[ne] @ N_source[ne].T,
            Ct=X_target_var[ne] - As_source[ne] @ As_source[ne].T
        )

    # construct transferred noise signal
    N_target = np.zeros_like(X_source)  # (Ne*Nt(s),Nc,Np)
    for ntr in range(n_trials):
        event_idx = event_type.index(y_source[ntr])
        N_target[ntr] = C[event_idx].T @ (X_source[ntr] - As_source[ne])
    return C, N_target


def solve_amplitude_alignment(
        X_aug: ndarray,
        y_aug: ndarray,
        X_target: ndarray,
        y_target: ndarray) -> ndarray:
    """
    By scaling the amplitude, the energy of each channel in the augmented data
    are controlled to match those of the corresponding channels in the target data.

    Parameters
    -------
    X_aug : ndarray, shape (Ne*Nt(a),Nc,Np).
        Augmented data.
    y_aug : ndarray, shape (Ne*Nt(a),).
        Labels for X_aug.
    X_target : ndarray, shape (Ne*Nt(t),Nc,Np).
        Target-domain data.
    y_target : ndarray, shape (Ne*Nt(t),).
        Labels for X_target.

    Returns
    -------
    X_aug_re : ndarray, shape (Ne*Nt(a),Nc,Np).
        Rescaled X_aug.
    """
    # basic information
    event_type = list(np.unique(y_aug))
    n_events = len(event_type)  # Ne

    # generate scaling matrix Lambda
    X_aug_var = utils.generate_var(X=X_aug, y=y_aug)  # (Ne,Nc,Nc)
    X_tar_var = utils.generate_var(X=X_target, y=y_target)  # (Ne,Nc,Nc)
    Lambda = np.zeros_like(X_aug_var)  # (Ne,Nc,Nc)
    for ne in range(n_events):
        Lambda[ne] = np.sqrt(np.diag(X_tar_var[ne]) / np.diag(X_aug_var[ne]))

    # rescale X_aug
    X_aug_re = np.zeros_like(X_aug)  # (Ne*Nt(a),Nc,Np)
    for ntr in range(X_aug.shape[0]):
        idx = event_type.index(y_aug[ntr])
        X_aug_re[ntr] = Lambda[idx] @ X_aug[ntr]
    return X_aug_re


# %% Main class
class CLSDA(object):
    """Domain adaptation based on common latent subspace."""

    joint_kernels = {
        'CCA': joint_cca_kernel,
        'TRCA': joint_trca_kernel,
        'DSP': joint_dsp_kernel
    }

    def __init__(self, n_components: int = 1):
        """
        Basic configuration.

        Parameters
        -------
        n_components : int.
            Number of eigenvectors picked as filters. Defaults to 1.
        """
        # config model
        self.n_components = n_components

    def margi_distr_alignment(self):
        """Marginal distribution alignment between X & identical matrix."""
        # euclidean alignment
        _, X_source_ea = utils.euclidean_alignment(self.X_source)
        _, X_target_ea = utils.euclidean_alignment(self.X_target)

        # standardization
        self.X_source = standardization(X_source_ea)
        self.X_target = standardization(X_target_ea)

    def solve_joint_model(self):
        """Solve spatial filters for projecting data onto a common latent subspace."""
        # check the spatial dimension of two domain
        if self.target_chan_indices is not None:
            X_source = self.X_source[:, self.target_chan_indices, :]
        else:
            X_source = self.X_source

        self.joint_model = self.joint_kernels[self.joint_kernel](
            X_target=self.X_target,
            y_target=self.y_target,
            X_source=X_source,
            y_source=self.y_source,
            template=self.template,
            n_components=self.n_components,
            theta=self.theta
        )

    def forward_propagation(self):
        """Solve propagation matrices for target domain."""
        self.A_target, self.s_target = solve_forward_propagation(
            X=self.X_target,
            y=self.y_target,
            w=self.joint_model['w']
        )  # if joint_kernel is 'eTRCA', w should be self.joint_model['ew']
        self.s_target_mean = utils.generate_mean(X=self.s_target, y=self.y_target)

    def spatial_alignment(self):
        """Spatial alignment for source-domain dataset & common latent subspace."""
        if self.align_space:  # use prototype spatial filters
            self.u_source, self.uX_source = solve_spatial_alignment(
                X_source=self.X_source,
                y_source=self.y_source,
                s_target_mean=self.s_target_mean
            )  # (Ne,Nk,Nc) & (Ne*Nt(t),Nk,Np)
        else:  # use joint spatial filters
            self.u_source = deepcopy(self.joint_model['w'])  # ['ew'] for 'eTRCA'
            self.uX_source = utils.spatial_filtering(
                w=self.u_source,
                X=self.X_source,
                y=self.y_source
            )  # (Ne*Nt(t),Nk,Np)
        self.uX_source_mean = utils.generate_mean(X=self.uX_source, y=self.y_source)

    def temporal_alignment(self):
        """Temporal alignment for source-domain dataset & target templates."""
        self.proj_time = solve_temporal_alignment(
            s_target_mean=self.s_target_mean,
            s_source=self.uX_source,
            y_source=self.y_source,
            s_source_mean=self.uX_source_mean,
            rho=self.rho
        )

    def noise_estimation(self):
        """Generate noise signal by solving a CORAL problem.
        See details in solve_noise_estimation()."""
        # check the spatial dimension of two domain
        if self.target_chan_indices is not None:
            X_source = self.X_source[:, self.target_chan_indices, :]
        else:
            X_source = self.X_source

        self.proj_noise, self.N_target = solve_noise_estimation(
            X_source=X_source,
            y_source=self.y_source,
            X_target=self.X_target,
            y_target=self.y_target,
            method=self.dependent_kernel,
            n_components=self.n_components
        )

    def condi_distr_alignment(self):
        """Conditional distribution alignment between source & target domain."""
        # main process
        self.solve_joint_model()
        self.forward_propagation()
        self.spatial_alignment()
        self.temporal_alignment()
        self.noise_estimation()

        # generate augmented data with estimated noise
        n_trials = self.X_source.shape[0]
        n_chans = self.X_target.shape[1]
        n_points = self.X_target.shape[-1]
        self.X_aug = np.zeros((n_trials, n_chans, n_points))
        self.y_aug = deepcopy(self.y_source)
        for ntr in range(n_trials):
            idx = self.event_type.index(self.y_aug[ntr])
            self.X_aug[ntr] = self.A_target[idx] @ self.uX_source[ntr] @ self.proj_time[idx]\
                + self.N_target[ntr]

    def amplitude_alignment(self):
        """Amplitude alignment between augmented data & real target data."""
        self.X_aug = solve_amplitude_alignment(
            X_aug=self.X_aug,
            y_aug=self.y_aug,
            X_target=self.X_target,
            y_target=self.y_target
        )
        self.X_aug = np.concatenate((self.X_aug, self.X_target), axis=0)
        self.y_aug = np.concatenate((self.y_aug, self.y_target), axis=0)

    def fit(
            self,
            X_source: ndarray,
            y_source: ndarray,
            X_train: ndarray,
            y_train: ndarray,
            template: Optional[ndarray] = None,
            theta: float = 0.7,
            rho: float = 0.5,
            joint_kernel: str = 'TRCA',
            align_space: bool = True,
            target_chan_indices: Optional[List[int]] = None,
            dependent_kernel: str = 'DSP'):
        """
        Train model.

        Parameters
        -------
        X_source : ndarray, shape (Ne*Nt(s),Nc(s),Np).
            Source-domain dataset.
        y_source : ndarray, shape (Ne*Nt(s),).
            Labels for X_source.
        X_train : ndarray, shape (Ne*Nt(t),Nc(t),Np).
            Target-domain training dataset.
        y_train : ndarray, shape (Ne*Nt(t),).
            Labels for X_train.
        template : ndarray, shape (Ne,m,Np).
            m could be 2*Nh while template is sinusoidal.
            Only useful while joint_kernel or dependent_kernel is 'CCA'.
        theta : float, 0-1.
            Defaults to 0.7. See details in joint kernel functions:
            joint_cca_kernel(), joint_trca_kernel() and joint_dsp_kernel().
        rho : float, 0-1.
            Defaults to 0.5. See details in solve_temporal_alignment().
        joint_kernel : str.
            The method of solving the model by integrating data from the source domain
            and the target domain. Supprot 'CCA', 'TRCA' and 'DSP' for now.
        align_space : bool.
            If True, use solve_spatial_alignment() to project the source dataset onto
            the common latent subspace. If False, use joint spatial filters to do so.
        target_chan_indices : List[int], optional.
            The indices of the channels of X_train in the spatial dimension of X_source.
            Must not be None while align_space is True and the Nc(t) is not equal to Nc(s).
        dependent_kernel : str.
            The method of solving the target-domain model.
            Support 'CCA', 'TRCA' and 'DSP' for now.
        """
        # load in data
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_train
        self.y_target = y_train
        self.template = template
        self.theta = theta
        self.rho = rho
        self.joint_kernel = joint_kernel
        self.align_space = align_space
        self.target_chan_indices = target_chan_indices
        self.dependent_kernel = dependent_kernel
        self.event_type = list(np.unique(self.y_source))

        # main process
        self.margi_distr_alignment()
        self.condi_distr_alignment()
        self.amplitude_alignment()


# %% Encapsulated version: takes up more RAM but runs faster
class FastCLSDA(object):
    pass
