# import os
# import pickle
# from copy import deepcopy

import numpy as np
from numpy import ndarray

import scipy
import scipy.io
import scipy.stats
from scipy import signal

# from sklearn.linear_model import LogisticRegression

import warnings
# import time

from pyriemann.utils.base import invsqrtm
# from pyriemann.utils.base import sqrtm
from pyriemann.utils.mean import mean_riemann, mean_logeuclid
# from pyriemann.utils.mean import mean_euclid
# from pyriemann.estimation import Covariances
# from pyriemann.classification import MDM, FgMDM, TSclassifier
from pyriemann.estimation import Shrinkage
# from pyriemann.estimation import ERPCovariances

# import seaborn as sn
# import matplotlib.pyplot as plt

# from rpa import transfer_learning as TL

from typing import Optional, Tuple, Any


# %% utils of SSTA
def regularization_cov(
        data_set: ndarray,
        mu: float = 0.01) -> ndarray:
    """
    Regularization of ill-conditioned covariance matrix.

    Parameters
    -------
    data_set : ndarray, shape (Nt,Nc,Nc).
        Covariance matrix.
    mu : float.
        Regularization coefficient. Defaults to 0.01.

    Returns
    -------
    out : ndarray, shape (Nt,Nc,Nc).
        Regularized data_set.
    """
    n_trial, n_chan, _ = data_set.shape  # Nt,Nc,Nc
    unit = np.eye(n_chan, dtype=data_set.dtype)
    return data_set + np.stack([unit * mu] * n_trial)


def z_score_trial(data: ndarray) -> ndarray:
    numerator = data - data.mean(axis=(1, 2), keepdims=True)
    denominator = data.std(axis=(1, 2), keepdims=True)
    return np.array(numerator / denominator, dtype=np.float32)


def z_score(data):
    numerator = data - data.mean(axis=(1, 2), keepdims=True)
    denominator = data.std(axis=2, keepdims=True)
    return np.array(numerator / denominator, dtype=np.float32)


def z_score_m(data):
    numerator = data - data.mean(axis=(0, 1), keepdims=True)
    denominator = data.std(axis=1, keepdims=True)
    return np.array(numerator / denominator, dtype=np.float32)


def get_active_pattern(C_x, W, C_s):
    try:
        A = C_x @ W @ np.linalg.inv(C_s)
        if np.sum(np.isnan(A)) > 0:
            A = C_x @ W @ np.linalg.pinv(C_s)
        if np.sum(np.isinf(A)) > 0:
            A = C_x @ W @ np.linalg.pinv(C_s)
    except:
        A = C_x @ W @ np.linalg.pinv(C_s)
    return A


def inverse_filtered_data(A: ndarray, fdata: ndarray):
    return np.einsum('ck, tks-> tcs', A, fdata)


def get_filtered_data(W, data):
    return np.einsum('ck, tcs-> tks', W, data)


def get_self_dot(data):
    return np.einsum('tcs, tas-> tca', data, data)


def get_filtered_element(target_data, source_data, W):
    ftarget = get_filtered_data(W, target_data)
    fsource = get_filtered_data(W, source_data)
    Cs = get_self_dot(fsource).mean(axis=0)
    Ct = get_self_dot(ftarget).mean(axis=0)
    return ftarget, fsource, Cs, Ct


def get_component_number(eig_value_source, threshould=0.99, min_n=None, max_n=None):
    percent = 0
    percent_list = []
    for component_num0 in range(eig_value_source.size):
        percent = eig_value_source[0:component_num0 + 1].sum() / eig_value_source.sum()
        percent_list.append(percent)
        if percent >= threshould:
            component_num = component_num0
            if min_n is None:
                pass
            else:
                if component_num < min_n:
                    component_num = min_n
            if max_n is None:
                pass
            else:
                if component_num > max_n:
                    component_num = max_n
            break
    return component_num


def fix_eigvalue(svd_value):
    """fix inf value in eigvalue

    Returns:
        _type_: _description_
    """
    if np.sum(np.isinf(svd_value)) > 0:
        denote_idx = np.argsort(svd_value)
        denote_idx = np.flip(denote_idx)
        inf_idx = np.where(np.isinf(svd_value))
        for eig_idx in denote_idx:
            if eig_idx not in inf_idx[0]:
                break
        svd_value[inf_idx] = svd_value[eig_idx] * 5
    else:
        svd_value = svd_value
    return svd_value


def get_main_component(
        source_data: ndarray,
        source_tmp: ndarray,
        source_cov: ndarray,
        threshould: float = 0.9,
        min_n: Optional[int] = None,
        max_n: Optional[int] = None):
    svd_value, right_vector = scipy.linalg.eig(np.array(source_tmp @ source_tmp.T), np.array(source_cov))
    svd_value = fix_eigvalue(svd_value)
    denote_idx = np.argsort(svd_value)
    denote_idx = np.flip(denote_idx)
    eig_value_source = np.real(svd_value[denote_idx])
    W_s = right_vector[:, denote_idx]
    component_num = get_component_number(
        eig_value_source,
        threshould,
        min_n=min_n,
        max_n=max_n
    )
    Ws_all_signal = np.array(np.real(W_s[:, 0:component_num]))
    fsource_data_signel = get_filtered_data(Ws_all_signal, source_data)
    Cs_signal = get_self_dot(fsource_data_signel).mean(axis=0)
    As_signal = get_active_pattern(source_cov, Ws_all_signal, Cs_signal)
    singnal_source = inverse_filtered_data(As_signal, fsource_data_signel)
    return singnal_source


def get_DSP_matrix(data: np.array, label: np.array, shrinkage=0.01):
    """_summary_

    Args:
        data (np.array): _description_
        label (np.array): _description_
        shrinkage (float, optional): _description_. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    classes = np.unique(label)
    class_tmp_array = np.dstack([data[label == class_i, ...].mean(axis=0) for class_i in classes])
    data_inner_class = np.zeros_like(data)
    for idx, class_i in enumerate(classes):
        data_inner_class[label == class_i, ...] = data[label == class_i, ...] - class_tmp_array[..., idx]

    Sb = 0
    Sw = 0
    for idx, class_i in enumerate(classes):
        # between class
        template_all = np.mean(class_tmp_array, axis=-1)
        template_class_i = class_tmp_array[..., idx]
        Sb += (template_class_i - template_all) @ (template_class_i - template_all).T

        # innter class
        inner_class_cov = np.einsum('tcp, tap-> tca', data_inner_class, data_inner_class)
        cov_estimator = Shrinkage(shrinkage=shrinkage)
        inner_class_cov_sk = cov_estimator.transform(inner_class_cov)
        Sw += inner_class_cov_sk.mean(axis=0)
    Sb = Sb / len(classes)
    Sw = Sw / len(classes)
    return Sb, Sw


def Combine_DSP(target_data, target_label, source_data, source_label, mix_coffe=0.5):
    Sb_tar, Sw_tar = get_DSP_matrix(target_data, target_label, shrinkage=0.01)
    Sb_sor, Sw_sor = get_DSP_matrix(source_data, source_label, shrinkage=0.01)

    # solve the optimizatino problem
    aim = np.linalg.pinv((1 - mix_coffe) * Sw_tar + mix_coffe * Sw_sor) @ ((1 - mix_coffe) * Sb_tar + mix_coffe * Sb_sor)
    svd_value, right_vector = np.linalg.eig(aim)
    denote_idx = np.argsort(svd_value)
    denote_idx = np.flip(denote_idx)
    # sorted_V = svd_value[denote_idx]
    sorted_W = right_vector[:, denote_idx]
    return sorted_W


def Combine_SNRmaxing(source_data, target_data, mix_coffe):
    mix_coffe = mix_coffe
    source_tmp = source_data.mean(axis=0)
    target_tmp = target_data.mean(axis=0)
    source_cov_single = get_self_dot(source_data)
    source_cov = source_cov_single.mean(axis=0)
    target_cov = get_self_dot(target_data).mean(axis=0)
    mixed_tmp = target_tmp * (1 - mix_coffe) + source_tmp * mix_coffe
    mixed_tmp_cov = mixed_tmp @ mixed_tmp.T
    # diff_tmp = target_tmp
    tmp_cov = mixed_tmp_cov
    mixed_cov = target_cov * (1 - mix_coffe) + source_cov * mix_coffe
    eig_val, eig_vec = scipy.linalg.eig(np.array(tmp_cov), np.array(mixed_cov))
    denote_idx = np.argsort(eig_val)
    denote_idx = np.flip(denote_idx)
    W = np.array(np.real(eig_vec[:, denote_idx]), dtype=np.float32)
    return W


class SSTS_kernal():
    def __init__(
            self,
            cmp_num_tar: int = 5,
            regu_coffe: float = 0.1,
            mix_coffe: float = 0.7):
        """
        Initialization.

        Parameters
        -------
        cmp_num_tar : int.
            Dimension of subspaces. Defaults to 5.
        regu_coffe : float.
            Regularization coefficient for temporal alignment (range: 0-1).
            Defaults to 0.1.
        mix_coffe : float.
            The weights of the source domain when constructing the spatial filter.
            Defaults to 0.7.
        """
        self.cmp_num_tar = cmp_num_tar
        self.regu_coffe = regu_coffe
        self.mix_coffe = mix_coffe
        self.n_trial_source, self.n_chan_source, self.n_time_source = None, None, None
        self.source_tmp = None
        self.target_tmp = None
        self.source_cov = None
        self.target_cov = None
        self.mixed_cov = None
        self.W = None
        self.A = None
        self.ftarget, self.fsource, self.Cs, self.Ct = None, None, None, None
        self.P = None
        self.source_scaling_matirx = None

    def get_common_subspace(
            self,
            source_data,
            target_data,
            W) -> Tuple[ndarray, ndarray]:
        self.n_trial_source, self.n_chan_source, self.n_time_source = source_data.shape
        mix_coffe = self.mix_coffe
        cmp_num_tar = self.cmp_num_tar

        self.source_tmp = source_data.mean(axis=0)
        self.target_tmp = target_data.mean(axis=0)

        source_cov_single = get_self_dot(source_data)
        self.source_cov = source_cov_single.mean(axis=0)
        self.target_cov = get_self_dot(target_data).mean(axis=0)

        self.mixed_cov = self.target_cov * (1 - mix_coffe) + self.source_cov * mix_coffe
        self.W = np.array(np.real(W[:, 0:cmp_num_tar]), dtype=np.float32)
        self.ftarget, self.fsource, self.Cs, self.Ct = get_filtered_element(target_data, source_data, self.W)
        self.A = get_active_pattern(self.mixed_cov, self.W, self.Ct)
        return self.W, self.A

    def get_temporal_transfer_matrix(self):
        regu_coffe = self.regu_coffe
        W = self.W
        fsource_data = self.fsource
        ftarget_data = self.ftarget

        # 时间迁移滤波器求解（正交强迫一致）-latent
        fsource_ave = fsource_data.mean(axis=0)
        ftarget_ave = ftarget_data.mean(axis=0)
        Cs_time = np.zeros((fsource_data.shape[2], fsource_data.shape[2]))
        for i in range(fsource_data.shape[0]):
            Cs_time += fsource_data[i].T @ fsource_data[i]
        Cs_time = Cs_time / fsource_data.shape[0]
        Cs_time_regu = (1 - regu_coffe) * Cs_time + regu_coffe * np.trace(Cs_time) * np.eye(Cs_time.shape[0]) / Cs_time.shape[0]
        self.P = np.linalg.inv(Cs_time_regu) @ fsource_ave.T @ ftarget_ave
        # Pfsource = np.einsum('ps, tkp -> tks', self.P, fsource_data)
        Pfsource_ave = fsource_ave @ self.P
        self.scaling_coffe = np.trace(Pfsource_ave @ ftarget_ave.T + ftarget_ave @ Pfsource_ave.T) / (2 * np.trace(Pfsource_ave @ Pfsource_ave.T))
        proportion = np.array([self.scaling_coffe] * W.shape[1])
        self.source_scaling_matirx = np.diag(proportion)
        return self.P, self.source_scaling_matirx

    def get_subordinate_component(self, source_data, threshould=0.99, min_n=2):
        mix_coffe = self.mix_coffe
        source_cov = self.source_cov
        target_cov = self.target_cov
        Cs = self.Cs
        W = self.W
        fsource_data = self.fsource
        A_source = get_active_pattern(source_cov, W, Cs)
        Diff_comp = inverse_filtered_data(A_source, fsource_data)
        Rest_comp = source_data - Diff_comp
        Rest_tmp = Rest_comp.mean(axis=0)
        Rest_cov = get_self_dot(Rest_comp).mean(axis=0) * mix_coffe + target_cov * (1 - mix_coffe)
        subordinate_component = get_main_component(
            source_data=Rest_comp,
            source_tmp=Rest_tmp,
            source_cov=Rest_cov,
            threshould=threshould,
            min_n=min_n
        )
        return subordinate_component

    def get_auto_noise(self, source_data, min_n=None, max_n=None):
        source_tmp = self.source_tmp
        source_cov = self.source_cov
        Singnal_source = get_main_component(
            source_data=source_data,
            source_tmp=source_tmp,
            source_cov=source_cov,
            threshould=0.9,
            min_n=min_n,
            max_n=max_n
        )
        Noise_source = source_data - Singnal_source
        return Noise_source, Singnal_source

    def get_wite_noise(self, mixed_cov, f_1=1, f_2=10, fs=128):
        n_chan = self.n_chan_source
        n_trial = self.n_trial_source
        n_time = self.n_time_source
        Cov = np.eye((n_chan)) * np.sqrt(np.trace(mixed_cov) / n_chan) * 0.5

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            noise_data = np.random.multivariate_normal(
                mean=np.zeros((n_chan,)),
                cov=Cov,
                size=(n_trial, n_time + 200)
            )
        noise_data = noise_data.transpose((0, 2, 1))

        # 对噪声滤波
        wn = [f_1 * 2 / fs, f_2 * 2 / fs]
        b, a = signal.butter(3, wn, 'bandpass')

        # Time consuming is: 38.0044 ms(gpu to cpu 5~25ms)
        noise_data = signal.filtfilt(
            b=b.astype('float32'),
            a=a.astype('float32'),
            x=np.array(noise_data),
            axis=2
        )
        Noise_data = noise_data[..., 100:100 + n_time]
        return Noise_data

    def noise_align(self, Signal_data, Noise_data, mixed_cov, sk_coffe=0.6):
        signal_tmp = Signal_data.mean(axis=0)
        erp_cov = signal_tmp @ signal_tmp.T
        noise_cov = get_self_dot(Noise_data).mean(axis=0)
        Cov_target = (np.trace(mixed_cov)) / mixed_cov.shape[0] * np.eye(mixed_cov.shape[0])

        # 计算特征值和特征向量
        eigvals2, eigvecs2 = np.linalg.eigh(Cov_target - erp_cov + 0.000001 * np.eye(noise_cov.shape[0]))
        # 计算Λ^-1/2
        L2 = np.diag(np.sign(eigvals2) * np.sqrt(np.abs(eigvals2)))
        # 计算B
        C_t = eigvecs2 @ L2 @ eigvecs2.T
        eigvals3, eigvecs3 = np.linalg.eigh(noise_cov + 0.000001 * np.eye(noise_cov.shape[0]))

        # 计算Λ^-1/2
        L3 = np.linalg.inv(np.diag(np.sign(eigvals3) * np.sqrt(np.abs(eigvals3))))

        # 计算B
        C_s = eigvecs3 @ L3 @ eigvecs3.T

        # C_t = sqrtm(Cov_target - erp_cov + 0.001 * np.eye(noise_cov.shape[0]))
        # C_s = invsqrtm(noise_cov + 0.001 * np.eye(noise_cov.shape[0]))

        noise_data1 = np.einsum('ac, tcs-> tas', C_t @ C_s, Noise_data) * sk_coffe

        # A1 = (np.tile(signal_tmp, (noise_data1.shape[0],1,1)) + noise_data1)
        # C_A1 =  get_self_dot(A1).mean(axis = 0)

        # A0 = (np.tile(signal_tmp, (Noise_data.shape[0],1,1)) + Noise_data)
        # C_A0 =  get_self_dot(A0).mean(axis = 0)

        # J0 = np.linalg.norm(C_A0- Cov_target)
        # J1 = np.linalg.norm(C_A1- Cov_target)
        return noise_data1

    def subspace_align(self):
        P = self.P
        fsource_data = self.fsource
        temporal_transfored_source_data = np.einsum('sp, tks-> tkp', P, fsource_data)
        return temporal_transfored_source_data

    def spatial_align(self, temporal_transfored_source_data, scaling=False):
        A = self.A
        if scaling:
            scaling = self.source_scaling_matirx
            return np.einsum('ck, tks-> tcs', A @ scaling, temporal_transfored_source_data)
        else:
            return np.einsum('ck, tks-> tcs', A, temporal_transfored_source_data)


# %% SSTA
class SSTA_intra_class_AN():
    def __init__(
            self,
            cmp_num_tar: int = 5,
            regu_coffe: float = 0.1,
            mix_coffe: float = 0.7,
            sclaing: bool = False,
            mask_class: Optional[Any] = None):
        """
        Initialization.

        Parameters
        -------
        cmp_num_tar : int.
            Dimension of subspaces. Defaults to 5.
        regu_coffe : float.
            Regularization coefficient for temporal alignment (range: 0-1).
            Defaults to 0.1.
        mix_coffe : float.
            The weights of the source domain when constructing the spatial filter.
            Defaults to 0.7.
        scaling : bool.
            Whether to scale and align the energy of different channels.
            Defaults to True.
        """
        self.cmp_num_tar = cmp_num_tar
        self.regu_coffe = regu_coffe
        self.mix_coffe = mix_coffe
        self.sclaing = sclaing
        self.mask_class = mask_class
        if sclaing:
            self.sk_coffe = 0.6
        else:
            self.sk_coffe = 1

    def single_class_transformer(
            self,
            source_data: ndarray,
            target_data: ndarray) -> ndarray:
        W = Combine_SNRmaxing(
            source_data=source_data,
            target_data=target_data,
            mix_coffe=self.mix_coffe
        )
        kernal = SSTS_kernal(
            cmp_num_tar=self.cmp_num_tar,
            regu_coffe=self.regu_coffe,
            mix_coffe=self.mix_coffe
        )
        kernal.get_common_subspace(
            source_data=source_data,
            target_data=target_data,
            W=W)
        kernal.get_temporal_transfer_matrix()
        temporal_transfored_source_data = kernal.subspace_align()
        spatial_temporal_transfered_reduced_data = kernal.spatial_align(
            temporal_transfored_source_data,
            scaling=self.sclaing
        )
        noise, signal = kernal.get_auto_noise(source_data)
        noise_align = kernal.noise_align(
            signal,
            noise,
            kernal.mixed_cov,
            sk_coffe=self.sk_coffe
        )
        return spatial_temporal_transfered_reduced_data + noise_align

    def transform(self, X_s, y_s, X_tl, y_tl, X_tu):
        # Marginal distribution alignment
        ea = Alignment()
        ea.get_center(X_tl, y_tl, metric='euclid')
        X_tl_trans = ea.alignment(X_tl)
        X_tu_trans = ea.alignment(X_tu)

        ea = Alignment()
        ea.get_center(X_s, y_s, metric='euclid')
        X_s = ea.alignment(X_s)

        X_tl_trans = z_score(X_tl_trans)
        X_tu_trans = z_score(X_tu_trans)
        X_s = z_score(X_s)

        # Conditional distribution alignment
        classes = np.unique(y_tl)
        if self.mask_class is not None:
            # 使用np.isin来创建一个布尔索引
            mask = np.isin(classes, self.mask_class, invert=True)

            # 使用这个布尔索引来选择不在mask_class中的元素
            filtered_classes = classes[mask]
        else:
            filtered_classes = classes

        S_trans_list = []
        label_list = []
        for idx, class_i in enumerate(filtered_classes):
            source_data_c = X_s[y_s == class_i, ...]
            target_data_c = X_tl_trans[y_tl == class_i, ...]
            S_trans = self.single_class_transformer(source_data_c, target_data_c)
            S_trans_list.append(S_trans)
            label_list.append(y_s[y_s == class_i])

        S_trans_array = np.concatenate(S_trans_list, axis=0)
        y_s_trans = np.hstack(label_list)
        return S_trans_array, y_s_trans, X_tl_trans, y_tl, X_tu_trans


class SSTA_between_class_S_AN():
    def __init__(
            self,
            cmp_num_tar=5,
            regu_coffe=0.1,
            mix_coffe=0.7,
            sclaing=False,
            mask_class=None) -> None:
        self.cmp_num_tar = cmp_num_tar
        self.regu_coffe = regu_coffe
        self.mix_coffe = mix_coffe
        self.sclaing = sclaing
        self.mask_class = mask_class
        if sclaing:
            self.sk_coffe = 0.6
        else:
            self.sk_coffe = 1

    def single_class_transformer(self, source_data, target_data, W):
        mix_coffe = self.mix_coffe
        kernal = SSTS_kernal(
            self.cmp_num_tar,
            self.regu_coffe,
            self.mix_coffe
        )
        kernal.get_common_subspace(source_data, target_data, W)
        kernal.get_temporal_transfer_matrix()
        temporal_transfored_source_data = kernal.subspace_align()
        spatial_temporal_transfered_reduced_data = kernal.spatial_align(
            temporal_transfored_source_data,
            scaling=self.sclaing
        )
        signal_subordinate_component = kernal.get_subordinate_component(
            source_data,
            threshould=0.99,
            min_n=2
        )
        noise, _ = kernal.get_auto_noise(source_data)
        transformed_ERP = (1 - mix_coffe) * spatial_temporal_transfered_reduced_data + mix_coffe * signal_subordinate_component
        noise_align = kernal.noise_align(
            transformed_ERP,
            noise,
            kernal.mixed_cov,
            sk_coffe=self.sk_coffe
        )
        return transformed_ERP + noise_align

    def transform(self, X_s, y_s, X_tl, y_tl, X_tu):
        # Marginal distribution alignment
        ea = Alignment()
        ea.get_center(X_tl, y_tl, metric='euclid', shrinkage=0.01)
        X_tl_trans = ea.alignment(X_tl)
        X_tu_trans = ea.alignment(X_tu)

        ea = Alignment()
        ea.get_center(X_s, y_s, metric='euclid', shrinkage=0.01)
        X_s = ea.alignment(X_s)

        X_tl_trans = z_score(X_tl_trans)
        X_tu_trans = z_score(X_tu_trans)
        X_s = z_score(X_s)

        # Conditional distribution alignment
        W = Combine_DSP(X_tl_trans, y_tl, X_s, y_s, mix_coffe=self.mix_coffe)
        classes = np.unique(y_tl)
        if self.mask_class is not None:
            # 使用np.isin来创建一个布尔索引
            mask = np.isin(classes, self.mask_class, invert=True)

            # 使用这个布尔索引来选择不在mask_class中的元素
            filtered_classes = classes[mask]
        else:
            filtered_classes = classes
        S_trans_list = []
        label_list = []
        for class_i in filtered_classes:
            source_data_c = X_s[y_s == class_i, ...]
            target_data_c = X_tl_trans[y_tl == class_i, ...]
            S_trans = self.single_class_transformer(source_data_c, target_data_c, W)
            S_trans_list.append(S_trans)
            label_list.append(y_s[y_s == class_i])
        S_trans_array = np.concatenate(S_trans_list, axis=0)
        y_s_trans = np.hstack(label_list)
        return S_trans_array, y_s_trans, X_tl_trans, y_tl, X_tu_trans


# %% Alignment method
class get_embeded_data(object):
    def __init__(self, classes = 1) -> None:
        self.classes = classes

    def fit(self, dataset, labels):
        self.data_tmp = dataset[labels == self.classes, ...].mean(axis=0, keepdims=True)
        return self

    def transform(self, dataset):
        n_tirla = dataset.shape[0]
        data_tmp_tile = np.repeat(self.data_tmp, axis=0, repeats=n_tirla)
        return np.concatenate([dataset, data_tmp_tile], axis = 1)


class Alignment(object):
    def __init__(self):
        pass

    # def fit_transform(
    #         self,
    #         Xs,
    #         ys,
    #         Xt_l,
    #         yt,
    #         Xt_u,
    #         metric='euclid',
    #         weighted=None,
    #         shrinkage=0.01):
    #     self.get_center(Xs, ys, metric=metric, weighted=weighted, shrinkage=shrinkage)
    #     Xs_new = self.alignment(Xs)

    #     self.get_center(Xt_l, y_tl, metric=metric, weighted=weighted, shrinkage=shrinkage)
    #     Xt_l_new = self.alignment(Xt_l)
    #     Xt_u_new = self.alignment(Xt_u)
    #     return Xs_new, ys, Xt_l_new, yt, Xt_u_new

    def get_center(self, X, y, metric='euclid', weighted=None, shrinkage=0.01):
        """计算数据的中心

        Args:
            X (ndarray): 试次*导联*时间

        return: self
        """
        classes = np.unique(y)
        if weighted is None:
            weight_v = np.zeros((y.size, ), dtype=X.dtype)
        else:
            weight_v = np.zeros((y.size, ), dtype=X.dtype)
            for classes_i in classes:
                weight_v[y == classes_i] = weight_v.size / np.sum(y == classes_i)

        Covs = np.einsum('tcp, tap-> tca', X, X)
        cov_estimator = Shrinkage(shrinkage=shrinkage)
        Cov = cov_estimator.transform(Covs)

        if metric == "riemann":
            try:
                M = mean_riemann(Cov, sample_weight=weight_v)
            except:
                Covs = np.einsum('tcp, tap-> tca', X, X)
                cov_estimator = Shrinkage(shrinkage=0.1)
                Cov = cov_estimator.transform(Covs)
                M = mean_riemann(Cov, sample_weight=weight_v)
        elif metric == "euclid":
            M = np.mean(Cov, axis=0)
        elif metric == "log-euclid":
            try:
                M = mean_logeuclid(Cov, sample_weight=weight_v)
            except:
                Covs = np.einsum('tcp, tap-> tca', X, X)
                cov_estimator = Shrinkage(shrinkage=0.1)
                Cov = cov_estimator.transform(Covs)
                M = mean_logeuclid(Cov, sample_weight=weight_v)
        self.center = invsqrtm(M)
        return self

    def alignment(self, X):
        ntrial = X.shape[0]
        alignmented_X = np.zeros_like(X)
        for trial_idx in range(ntrial):
            alignmented_X[trial_idx, ...] = self.center @ X[trial_idx, ...]
        return alignmented_X


class Riemannian_Procrustes_Analysis():
    def __init__(self):
        pass

    def fit_transform(self, Xs, ys, Xt_l, yt, Xt_u, shrinkage=0.01, type='STL', return_type='covs'):
        try:
            # n_chans = Xs.shape[1]
            sk = Shrinkage(shrinkage=shrinkage)
            source_org = {}
            Xs_spd = np.einsum('tcp, tap -> tca', Xs, Xs)
            # source_org['covs'] = regularization_cov(Xs_spd)
            source_org['covs'] = sk.transform(Xs_spd)
            source_org['trials'] = Xs
            source_org['labels'] = ys

            target_org_train = {}
            Xt_l_spd = np.einsum('tcp, tap -> tca', Xt_l, Xt_l)
            # target_org_train['covs'] = regularization_cov(Xt_l_spd)
            target_org_train['covs'] = sk.transform(Xt_l_spd)
            target_org_train['trials'] = Xt_l
            target_org_train['labels'] = yt

            target_org_test = {}
            Xt_u_spd = np.einsum('tcp, tap -> tca', Xt_u, Xt_u)
            # target_org_test['covs'] = regularization_cov(Xt_u_spd)
            target_org_test['covs'] = sk.transform(Xt_u_spd)
            target_org_test['trials'] = Xt_u

            # get the score with the re-centered matrices
            source_rct, target_rct_train, target_rct_test = TL.RPA_recenter(source_org, target_org_train, target_org_test)

            # get the score with the stretched matrices
            source_str, target_str_train, target_str_test = TL.RPA_stretch(source_rct, target_rct_train, target_rct_test, source_org, target_org_train, target_org_test, paradigm = type)

            # rotate the re-centered-stretched matrices using information from classes
            source_rpa, target_rpa_train, target_rpa_test = TL.RPA_rotate(source_str, target_str_train, target_str_test, paradigm=type)
        except:
            try:
                # n_chans = Xs.shape[1]
                sk = Shrinkage(shrinkage=0.1)
                source_org = {}
                Xs_spd = np.einsum('tcp, tap -> tca', Xs, Xs)
                # source_org['covs'] = regularization_cov(Xs_spd)
                source_org['covs'] = sk.transform(Xs_spd)
                source_org['trials'] = Xs
                source_org['labels'] = ys

                target_org_train = {}
                Xt_l_spd = np.einsum('tcp, tap -> tca', Xt_l, Xt_l)
                # target_org_train['covs'] = regularization_cov(Xt_l_spd)
                target_org_train['covs'] = sk.transform(Xt_l_spd)
                target_org_train['trials'] = Xt_l
                target_org_train['labels'] = yt

                target_org_test = {}
                Xt_u_spd = np.einsum('tcp, tap -> tca', Xt_u, Xt_u)
                # target_org_test['covs'] = regularization_cov(Xt_u_spd)
                target_org_test['covs'] = sk.transform(Xt_u_spd)
                target_org_test['trials'] = Xt_u

                # get the score with the re-centered matrices
                source_rct, target_rct_train, target_rct_test = TL.RPA_recenter(source_org, target_org_train, target_org_test)
                # get the score with the stretched matrices
                source_str, target_str_train, target_str_test = TL.RPA_stretch(source_rct, target_rct_train, target_rct_test, source_org, target_org_train, target_org_test, paradigm = type)
                # rotate the re-centered-stretched matrices using information from classes
                source_rpa, target_rpa_train, target_rpa_test = TL.RPA_rotate(source_str, target_str_train, target_str_test, paradigm=type)
            except:
                # n_chans = Xs.shape[1]
                sk = Shrinkage(shrinkage=0.3)
                source_org = {}
                Xs_spd = np.einsum('tcp, tap -> tca', Xs, Xs)
                # source_org['covs'] = regularization_cov(Xs_spd)
                source_org['covs'] = sk.transform(Xs_spd)
                source_org['trials'] = Xs
                source_org['labels'] = ys

                target_org_train = {}
                Xt_l_spd = np.einsum('tcp, tap -> tca', Xt_l, Xt_l)
                # target_org_train['covs'] = regularization_cov(Xt_l_spd)
                target_org_train['covs'] = sk.transform(Xt_l_spd)
                target_org_train['trials'] = Xt_l
                target_org_train['labels'] = yt

                target_org_test = {}
                Xt_u_spd = np.einsum('tcp, tap -> tca', Xt_u, Xt_u)
                # target_org_test['covs'] = regularization_cov(Xt_u_spd)
                target_org_test['covs'] = sk.transform(Xt_u_spd)
                target_org_test['trials'] = Xt_u

                # get the score with the re-centered matrices
                source_rct, target_rct_train, target_rct_test = TL.RPA_recenter(source_org, target_org_train, target_org_test)

                # get the score with the stretched matrices
                source_str, target_str_train, target_str_test = TL.RPA_stretch(source_rct, target_rct_train, target_rct_test, source_org, target_org_train, target_org_test, paradigm=type)
                # rotate the re-centered-stretched matrices using information from classes
                source_rpa, target_rpa_train, target_rpa_test = TL.RPA_rotate(source_str, target_str_train, target_str_test, paradigm=type)

        # # rotate the re-centered-stretched matrices using information from classes
        # source_rpa, target_rpa_train, target_rpa_test = TL.RPA_rotate(source_rct, target_rct_train, target_rct_test, paradigm = type)
        # source_rpa, target_rpa_train, target_rpa_test = TL.RPA_recenter(source_org, target_org_train, target_org_test)

        if return_type == 'covs':
            return source_rpa['covs'], source_rpa['labels'], target_rpa_train['covs'], target_rpa_train['labels'], target_rpa_test['covs']
        elif return_type == 'trials':
            return source_rpa['trials'], source_rpa['labels'], target_rpa_train['trials'], target_rpa_train['labels'], target_rpa_test['trials']
        else:
            return (source_rpa['trials'], source_rpa['covs']), source_rpa['labels'], (target_rpa_train['trials'], target_rpa_train['covs']), target_rpa_train['labels'], (target_rpa_test['trials'], target_rpa_test['covs'])
