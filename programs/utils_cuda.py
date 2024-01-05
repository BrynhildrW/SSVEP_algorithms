# -*- coding: utf-8 -*-
"""
@ author: Brynhildr Wu
@ email: brynhildrwu@gmail.com

GPU version of utils.py

1. Data preprocessing:
    (1-1) centralization()
    (1-2) normalization()
    (1-3) standardization()

5. spatial distances
    (5-1) pearson_corr()


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

update: 2023/10/16

"""


# %% basic moduls
<<<<<<< HEAD
from typing import Optional, List, Tuple, Union
import cupy as cp


# %% 1. Data preprocessing
def centralization(X: cp.ndarray) -> cp.ndarray:
    """Transform vector x into y, s.t. mean(y) = 0.

    Args:
        X (cp.ndarray): (...,Np).

    Returns:
        Y (cp.ndarray): Data after centralization.
=======
from typing import Optional, List, Tuple, Any
import cupy as cp

# %% 1. Data preprocessing
def centralization(
    X: cp.ndarray) -> cp.ndarray:
    """Transform vector x into y, s.t. mean(y) = 0.
    
    Args:
        X (ndarray): (...,Np).
    
    Returns:
        Y (ndarray): Data after centralization.
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
    """
    return X - X.mean(axis=-1, keepdims=True)


<<<<<<< HEAD
def normalization(X: cp.ndarray) -> cp.ndarray:
    """Transform vector x into y, s.t. y = (x - min(x)) / (max(x) - min(x)).
        The range of y is [0,1].

    Args:
        X (cp.ndarray): (...,Np).

    Returns:
        Y (cp.ndarray): Data after normalization.
    """
    X_min = cp.min(X, axis=-1, keepdims=True)  # (...,1)
    X_max = cp.max(X, axis=-1, keepdims=True)  # (...,1)
    return (X - X_min) / (X_max - X_min)


def standardization(X: cp.ndarray) -> cp.ndarray:
    """Transform vector x into y, s.t. mean(y) = 0, var(y) = 1.

    Args:
        X (cp.ndarray): (...,Np).

    Returns:
        Y (cp.ndarray): Data after standardization.
    """
    X = centralization(X)
    return X / cp.std(X, axis=-1, keepdims=True)
=======
def normalization(
    X: cp.ndarray) -> cp.ndarray:
    """Transform vector x into y, s.t. y = (x - min(x)) / (max(x) - min(x)).
        The range of y is [0,1].
    
    Args:
        X (ndarray): (...,Np).
    
    Returns:
        Y (ndarray): Data after normalization.
    """
    X_min = cp.min(X, axis=-1, keepdims=True)  # (...,1)
    X_max = cp.max(X, axis=-1, keepdims=True)  # (...,1)
    return (X-X_min)/(X_max-X_min)


def standardization(
    X: cp.ndarray) -> cp.ndarray:
    """Transform vector x into y, s.t. mean(y) = 0, var(y) = 1.
    
    Args:
        X (ndarray): (...,Np).
    
    Returns:
        Y (ndarray): Data after standardization.
    """
    return (X - cp.mean(X, axis=-1, keepdims=True)) / (cp.std(X, axis=-1, keepdims=True))
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138


# %% 5. spatial distances
def pearson_corr(
<<<<<<< HEAD
        X: cp.ndarray,
        Y: cp.ndarray,
        parallel: bool = False) -> Union[float, cp.ndarray]:
    """Pearson correlation coefficients.

    Args:
        X (cp.ndarray): (m,n).
            e.g. Spatial filtered single-trial data (Nk,Np).
        Y (cp.ndarray): (l,m,n) or (m,n).
            e.g. Templates while parallel=True (Ne,Nk,Np) or False (Nk,Np).
        parallel (bool): An accelerator. Defaults to False.
            If False, X could only be compared with Y of shape (m,n);
            If True, X could be compared with Y of shape (l,m,n).

    Returns:
        corr_coef (Union[float, cp.ndarray]):
            float: X (m,n) & Y (m,n).
            ndarray (l,): X (m,n) & Y (l,m,n).
    """
    # X, Y must be standardized
    X, Y = standardization(X), standardization(Y)
    n = X.shape[-1]

    # reshape data into vector-style: reshape() is 5 times faster than flatten()
    X = cp.reshape(X, -1, order='C')  # (m*n,)
    if parallel:  # Y: (l,m,n)
        Y = cp.reshape(Y, (Y.shape[0], -1), order='C')  # (l,m*n)
    else:  # Y: (m,n)
        Y = cp.reshape(Y, -1, order='C')  # (m*n,)
    return Y @ X / n
=======
    X: cp.ndarray,
    Y: cp.ndarray,
    common_filter: bool = False) -> cp.ndarray:
    """Pearson correlation coefficients.

    Args:
        X (ndarray): (Ne*Nk,Np) or (Nk,Np). Spatial filtered single-trial data.
        Y (ndarray): (Ne,Ne*Nk,Np) or (Ne,Nk,Np). Templates.

    Returns:
        corr_coef (ndarray): (Ne,) or float.
    """
    # X, Y must be standardized
    X, Y = standardization(X), standardization(Y)

    # reshape data into vector-style: reshape() is 5 times faster than flatten()
    X = cp.reshape(X, -1, order='C')  # (Ne*Nk*Np,) or (Nk*Np,)
    if common_filter:
        Y = cp.reshape(Y, (Y.shape[0], -1), order='C')  # (Ne,Ne*Nk*Np) or (Ne,Nk*Np)
    else:
        Y = cp.reshape(Y, -1, order='C')  # (Nk*Np,)
    return Y @ X
>>>>>>> 144b677c7c84e6a2b8cba9067520257956199138
