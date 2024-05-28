# -*- coding: utf-8 -*-
"""
@ Author: Brynhildr Wu
@ Email: brynhildrwu@gmail.com

Spatialtemporal domain adaptation based on common latent subspace.

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

from typing import Optional, List, Tuple, Dict, Union
from numpy import ndarray

import numpy as np
import scipy.linalg as sLA


# %% Marginal distribution alignment
def covar_alignment(C1: ndarray, C2: ndarray) -> ndarray:
    """Solve problem: Q = min || Q^T C1 Q - C2 ||_F^2.
        i.e. Q = C1^(-1/2) @ C2^(1/2)

    Args:
        C1 (ndarray): (n,n). Covariance matrix.
        C2 (ndarray): (n,n). Covariance matrix.

    Returns:
        Q (ndarray): (n,n). Linear transformation matrix Q.
    """
    return utils.nega_root_matrix(C1) @ utils.root_matrix(C2)


def standardization(X: ndarray) -> ndarray:
    """Data standardization process specially designed for domain adaptation.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Input dataset.

    Returns:
        X (ndarray): (Ne*Nt,Nc,Np). Data after standardization.
    """
    for nt in range(X.shape[0]):
        X[nt] -= np.mean(X[nt])  # trial centralization | (Nc,Np)
        ch_std = np.diag(1 / np.std(X[nt], axis=-1))  # (Nc,Nc)
        X[nt] = ch_std @ X[nt]  # channel normalization | (Nc,Np)
    return X


# %% Backward-propagation model (TRCA filter)
def joint_trca_kernel(
        X_target: ndarray,
        y_target: ndarray,
        X_source: ndarray,
        y_source: ndarray,
        n_components: int = 1,
        theta: float = 0.5) -> Dict[str, ndarray]:
    """Calculate joint-TRCA filters to obtain source activity on latent subspace.

    Args:
        X_target (ndarray): (Ne*Nt(t),Nc,Np). Sklearn-style target dataset. Nt>=2.
        y_target (ndarray): (Ne*Nt(t),). Labels for X_target.
        X_source (ndarray): (Ne*Nt(s),Nc,Np). Source dataset of one subject.
        y_source (ndarray): (Ne*Nt(s),). Labels for X_source.
        n_components (int): Number of eigenvectors picked as filters. Nk.
        theta (float): 0-1. Hyper-parameter to control the use of source dataset.

    Returns: Dict[str, ndarray]
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        w (ndarray): (Ne,Nk,Nc). Spatial filters of joint-TRCA.
        ew (ndarray): (Ne*Nk,Nc). Common spatial filter of joint-eTRCA.
    """
    # S & Q of target- & source-domain dataset (joint)
    Q_target, S_target = trca.generate_trca_mat(X=X_target, y=y_target)  # (Ne,Nc,Nc)
    Q_source, S_source = trca.generate_trca_mat(X=X_source, y=y_source)  # (Ne,Nc,Nc)
    Q = (1 - theta) * Q_target + theta * Q_source
    S = (1 - theta) * S_target + theta * S_source

    # GEPs | train spatial filters
    w, ew = trca.solve_trca_func(Q=Q, S=S, n_components=n_components)

    # backward-propagation model
    return {'Q': Q, 'S': S, 'w': w, 'ew': ew}


def dependent_trca_kernel(
        X: ndarray,
        y: ndarray,
        n_components: int = 1) -> Dict[str, ndarray]:
    """Calculate normal TRCA filters to obtain source activity on latent subspace.

    Args:
        X (ndarray): (Ne*Nt(t),Nc,Np). Sklearn-style dataset. Nt>=2.
        y (ndarray): (Ne*Nt(t),). Labels for X.
        n_components (int): Number of eigenvectors picked as filters. Nk.

    Returns: Dict[str, ndarray]
        Q (ndarray): (Ne,Nc,Nc). Covariance of original data.
        S (ndarray): (Ne,Nc,Nc). Covariance of template data.
        w (ndarray): (Ne,Nk,Nc). Spatial filters of TRCA.
        ew (ndarray): (Ne*Nk,Nc). Common spatial filter of eTRCA.
    """
    # normal TRCA modeling process
    Q, S = trca.generate_trca_mat(X=X, y=y)  # (Ne,Nc,Nc)
    w, ew = trca.solve_trca_func(Q=Q, S=S, n_components=n_components)
    return {'Q': Q, 'S': S, 'w': w, 'ew': ew}


# %% Backward-propagation model (DSP filter)
def joint_dsp_kernel(
        X_target: ndarray,
        y_target: ndarray,
        X_source: ndarray,
        y_source: ndarray,
        n_components: int = 1,
        theta: float = 0.5) -> Dict[str, ndarray]:
    """Calculate DSP filters to obtain source response on latent subspace.

    Args:
        X_target (ndarray): (Ne*Nt(t),Nc,Np). Sklearn-style target dataset. Nt>=2.
        y_target (ndarray): (Ne*Nt(t),). Labels for X_target.
        X_source (ndarray): (Ne*Nt(s),Nc,Np). Source dataset of one subject.
        y_source (ndarray): (Ne*Nt(s),). Labels for X_source.
        n_components (int): Number of eigenvectors picked as filters. Nk.
        theta (float): 0-1. Hyper-parameter to control the use of source dataset.

    Returns: Dict[str, ndarray]
        Sb (ndarray): (Nc,Nc). Scatter matrix of between-class difference.
        Sw (ndarray): (Nc,Nc). Scatter matrix of within-class difference.
        w (ndarray): (Ne,Nk,Nc). Spatial filter of joint-DSP.
    """
    # basic information
    n_events = len(np.unique(y_target))

    # Sb & Sw of target & source dataset
    Sb_target, Sw_target, _ = utils.generate_sb_sw(X=X_target, y=y_target)
    Sb_source, Sw_source, _ = utils.generate_sb_sw(X=X_source, y=y_source)

    Sb = (1 - theta) * Sb_target + theta * Sb_source
    Sw = (1 - theta) * Sw_target + theta * Sw_source

    # GEPs | train spatial filter
    w = utils.solve_gep(A=Sb, B=Sw, n_components=n_components)  # (Nk,Nc)
    w = np.tile(A=w[None, ...], reps=(n_events, 1, 1))  # (Ne,Nk,Nc)

    # backward-propagation model
    return {'Sb': Sb, 'Sw': Sw, 'w': w}


def dependent_dsp_kernel():
    pass


# %% Conditional distribution alignment
def generate_source_response(
        X: ndarray,
        y: ndarray,
        w: ndarray) -> Tuple[ndarray, ndarray]:
    """Calculate wX on latent subspace.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Sklearn-style dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X.
        w (ndarray): (Ne,Nk,Nc). Spatial filters.

    Returns: Tuple[ndarray, ndarray]
        S (ndarray): (Ne*Nt,Nk,Np). Source responses.
        S_mean (ndarray): (Ne,Nk,Np). Trial-avereged S.
    """
    # basic information
    n_trials = X.shape[0]  # Ne*Nt
    n_components = w.shape[-2]  # Nk
    n_points = X.shape[-1]  # Np
    event_type = list(np.unique(y))  # (Ne,)

    # backward-propagation
    S = np.zeros((n_trials, n_components, n_points))
    for nt in range(X.shape[0]):
        event_idx = event_type.index(y[nt])
        S[nt] = w[event_idx] @ X[nt]
    S_mean = utils.generate_mean(X=S, y=y)  # (Ne,Nk,Np)
    return S, S_mean


# Forward-propagation model (aliasing matrix)
def forward_propagation(
        X: ndarray,
        y: ndarray,
        S: ndarray,
        w: ndarray) -> ndarray:
    """Calculate propagation matrice A based on source responses.
        A = argmin||A @ s - X||.

    Args:
        X (ndarray): (Ne*Nt,Nc,Np). Target-domain dataset. Nt>=2.
        y (ndarray): (Ne*Nt,). Labels for X.
        S (ndarray): (Ne*Nt,Nk,Np). Target-domain source responses.
        w (ndarray): (Ne,Nk,Nc). Spatial filters.

    Returns: ndarray
        A (ndarray): (Ne,Nc,Nk). Propagation matrices.
    """
    # basic information
    n_events = len(np.unique(y))  # Ne
    n_chans = X.shape[-2]  # Nc
    n_components = w.shape[-2]  # Nk

    # analytical solution
    X_var = utils.generate_var(X=X, y=y)  # (Ne,Nc,Nc)
    S_var = utils.generate_var(X=S, y=y)  # (Ne,Nk.Nk)
    A = np.zeros((n_events, n_chans, n_components))
    for ne in range(n_events):
        A[ne] = X_var[ne] @ w[ne].T @ sLA.inv(S_var[ne])
    return A


# Time-domain alignment in common latent subspace
def time_alignment(
        S_target_mean: ndarray,
        S_source: ndarray,
        y_source: ndarray,
        S_source_mean: ndarray,
        rho: float = 0.1) -> ndarray:
    """Calculate time-domain projection matrice P.

    Args:
        S_target_mean (ndarray): (Ne,Nk,Np). Trial-averaged S_target.
        S_source (ndarray): (Ne*Nt(s),Nk,Np). Source-domain source responses.
        y_source (ndarray): (Ne*Nt(s),). Labels for S_source.
        S_source_mean (ndarray): (Ne,Nk,Np). Trial-averaged S_source.
        rho (float): 0-1. L2 regularization parameter to control the size of P.

    Returns: ndarray
        P (ndarray): (Ne,Np,Np). Projection matrix.
    """
    # basic information
    n_events = S_source_mean.shape[0]  # Ne
    n_components = S_source_mean.shape[-2]  # Nk
    n_points = S_source_mean.shape[-1]  # Np

    # analytical solution
    S_source_var = utils.generate_var(X=S_source, y=y_source)  # (Ne,Nk,Nk)
    unit_I = np.tile(A=np.eye(n_components), reps=(n_events, 1, 1))  # (Ne,Nk,Nk)
    P = np.zeros((n_events, n_points, n_points))  # (Ne,Np,Np)
    for ne in range(n_events):
        temp = (1 - rho) * sLA.inv((1 - rho) * S_source_var[ne] + rho * unit_I)
        P[ne] = temp @ S_source_mean[ne].T @ S_target_mean[ne]
    return P


# Estimation of target-domain noise (background) signal.
def noise_estimation(
        X_source_mean: ndarray,
        S_source_mean: ndarray,
        A_source: ndarray,
        X_target: ndarray,
        y_target: ndarray) -> Tuple[ndarray, ndarray]:
    """Assume transformation matrix Q_source satisfies the following conditions:
        (1) N_source = X_source - A_source @ S_source
        (2) A_source @ S_source + Q_source @ N_source = X_target
        By solving problem:
        Q = min || (AS + QN) (AS + QN)^T - Var(X_target) ||_F^2
          = min || Q (NN^T) Q^T - Var + AS (AS)^T ||_F^2
        To get the esimation of the noise of target-domain data:
        N_target = Q_source @ N_source
        It should be noted that the calculation process of the input parameter
            S_source_mean & A_source here is independent of the target-domain data,
            which means using X_source & y_source only.

    Args:
        X_source_mean (ndarray): (Ne,Nc,Np). Trial-averaged X_source.
        S_source_mean (ndarray): (Ne,Nk,Np). Spatial filtered X_source_mean.
        A_source (ndarray): (Ne,Nc,Nk). Propagation matrices of S_source.
        X_target (ndarray): (Ne*Nt(t),Nc,Np). Target dataset. Nt>=2.
        y_target (ndarray): (Ne*Nt(t),). Labels for X_target.

    Returns: Tuple[ndarray, ndarray]
        Q_noise (ndarray): (Ne,Nc,Nc). Linear transformation matrix.
        N_target (ndarray): (Ne,Nc,Np). Estimated noise of target-domain data.
    """
    # basic information
    n_events = X_source_mean.shape[0]  # Ne
    n_chans = X_source_mean.shape[1]  # Nc

    # preparation for solution
    # Var(X_target), X_target_var | (Ne,Nc,Nc)
    X_target_var = utils.generate_var(X=X_target, y=y_target)

    # A_source @ S_source, AwX_source_mean | N_source, N_source_mean | (Ne,Nc,Np)
    AwX_source_mean = np.einsum('eck,ekp->ecp', A_source, S_source_mean)
    N_source_mean = X_source_mean - AwX_source_mean  # (Ne,Nc,Np)

    # N @ N^T, N_source_mean | AS @ (AS)^T, AwX_source_var | (Ne,Nc,Nc)
    N_source_var = np.einsum('ecp,ehp->ech', N_source_mean, N_source_mean)
    AwX_source_var = np.einsum('ecp,ehp->ech', AwX_source_mean, AwX_source_mean)

    # analytical solution
    Q_noise = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
    N_target = np.zeros_like(N_source_mean)  # (Ne,Nc,Np)
    for ne in range(n_events):
        Q_noise[ne] = covar_alignment(
            C1=N_source_var[ne],
            C2=X_target_var[ne] - AwX_source_var[ne]
        )  # (Nc,Nc)
        N_target[ne] = Q_noise[ne] @ N_source_mean[ne]
    return Q_noise, N_target


# %% Main classes
class CDA(object):
    """Conditional distribution alignment based on waveform."""
    kernel_methods = {'TRCA': joint_trca_kernel,
                      'DSP': joint_dsp_kernel}

    def __init__(
            self,
            kernel: str = 'DSP',
            joint: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            kernel (str): Backward-propagation method. 'DSP' or 'TRCA'.
            joint (bool): WWhether to use source-domain data in kernel function.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
        """
        # config model
        self.kernel = kernel
        self.joint = joint
        self.n_components = n_components

    def fit(
            self,
            X_target: ndarray,
            y_target: ndarray,
            X_source: ndarray,
            y_source: ndarray,
            sine_template: Optional[ndarray] = None,
            theta: float = 0.5,
            rho: float = 0.1):
        """Train conditional distribution alignment model.

        Args:
            X_target (ndarray): (Ne*Nt(t),Nc,Np). Sklearn-style target dataset. Nt>=2.
            y_target (ndarray): (Ne*Nt(t),). Labels for X_target.
            X_source (ndarray): (Ne*Nt(s),Nc,Np). Source dataset of one subject.
            y_source (ndarray): (Ne*Nt(s),). Labels for X_source.
            sine_template (ndarray, Optional): (Ne,2*Nh,Np). Sinusoidal templates.
            theta (float): 0-1. Hyper-parameter to control the use of source dataset.
            rho (float): 0-1. L2 regularization parameter to control the size of P.
        """
        # basic information
        self.X_target = X_target
        self.y_target = y_target
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template
        self.theta = theta
        self.rho = rho

        # backward-propagation
        self.backward_model = self.kernel_methods[self.kernel](
            X_target=self.X_target,
            y_target=self.y_target,
            X_source=self.X_source,
            y_source=self.y_source,
            n_components=self.n_components,
            theta=self.theta
        )
        self.w = self.backward_model['w']  # (Ne,Nk,Nc)
        self.S_target, self.S_target_mean = generate_source_response(
            X=self.X_target,
            y=self.y_target,
            w=self.w
        )  # (Ne*Nt(t),Nk,Np), (Ne,Nk,Np)

        # forward-propagation
        self.A = forward_propagation(
            X=self.X_target,
            y=self.y_target,
            S=self.S_target,
            w=self.w
        )  # (Ne,Nc,Nk)

        # time-domain alignment
        self.S_source, self.S_source_mean = generate_source_response(
            X=self.X_source,
            y=self.y_source,
            w=self.w
        )  # (Ne*Nt(s),Nk,Np), (Ne,Nk,Np)
        self.P = time_alignment(
            S_target_mean=self.S_target_mean,
            S_source=self.S_source,
            y_source=self.y_source,
            S_source_mean=self.S_source_mean,
            rho=self.rho
        )  # (Ne,Np,Np)
        pass


class CLS_STDA(object):
    """Spatialtemporal domain adaptation (STDA) based on common latent subspace (CLS)."""
    def __init__(
            self,
            X_source: ndarray,
            y_source: ndarray,
            X_target: ndarray,
            y_target: ndarray,
            sine_template: Optional[ndarray] = None,
            joint_kernel: str = 'DSP',
            target_kernel: str = 'TRCA',
            n_components: int = 1,
            theta: float = 0.5,
            rho: float = 0.1):
        """Basic configuration.

        Args:
            X_source (ndarray): (Ne*Nt(s),Nc,Np). Source dataset of one subject.
            y_source (ndarray): (Ne*Nt(s),). Labels for X_source.
            X_target (ndarray): (Ne*Nt(t),Nc,Np). Sklearn-style target dataset. Nt>=2.
            y_target (ndarray): (Ne*Nt(t),). Labels for X_target.
            sine_template (ndarray, Optional): (Ne,2*Nh,Np). Sinusoidal templates.
            joint_kernel (str): Backward-propagation model for domain adaptation.
                Now support 'TRCA' and 'DSP'. Defaults to 'DSP'.
            target_kernel (str): Spatial filtering method for background EEG estimation.
                Now support 'TRCA' and 'DSP'. Defaults to 'TRCA'.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
            theta (float): 0-1. Hyper-parameter to control the use of source dataset.
            rho (float): 0-1. L2 regularization parameter to control the size of P.
        """
        # load in data
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_target
        self.y_target = y_target
        self.sine_template = sine_template
        self.joint_kernel = joint_kernel
        self.target_kernel = target_kernel
        self.n_components = n_components
        self.theta = theta
        self.rho = rho

    def prepare(self):
        """Data initialization for training."""
        pass

    def margi_distr_alignment(self):
        """Marginal distribution alignment between X_source (X_target) & I."""
        # Q_source = min ||Q_source @ Var(X_source) @ Q_source.T - I||_F^2
        self.Q_source = covar_alignment(
            C1=utils.generate_var(X=self.X_source, y=None),
            C2=np.eye(self.X_source.shape[-2])
        )  # (Nc,Nc)
        self.X_source_temp = np.einsum('ch,ehp->ecp', self.Q_source, self.X_source)

        # Q_target = min ||Q_target @ Var(X_target) @ Q_target.T - I||_F^2
        self.Q_target = covar_alignment(
            C1=utils.generate_var(X=self.X_target, y=None),
            C2=np.eye(self.X_target.shape[-2])
        )  # (Nc,Nc)
        self.X_target_temp = np.einsum('ch,ehp->ecp', self.Q_target, self.X_target)

    def data_standardization(self):
        """Data standardization process specially designed for domain adaptation."""
        self.X_source_temp = standardization(X=self.X_source_temp)
        self.X_target_temp = standardization(X=self.X_target_temp)

    def condi_distr_alignment(self):
        """Conditional distribution alignment based on waveform."""
        
        pass
