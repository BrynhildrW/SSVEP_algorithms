# -*- coding: utf-8 -*-
"""
@ Author: Brynhildr Wu
@ Email: brynhildrwu@gmail.com

Transfer learning based on matrix decomposition. Demo.

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
    response length: Nrl

"""

# %% Basic modules
import utils

import cca
import trca
import dsp
import transfer

from typing import Optional, List, Tuple, Dict, Union, Any
from numpy import ndarray

import numpy as np
import scipy.linalg as sLA
# from copy import deepcopy

from sklearn.base import clone
# from sklearn.model_selection import StratifiedShuffleSplit

import pickle


# %% 1. Common impulse response component analysis, CIRCA
def circa_kernel(
        X_train: ndarray,
        y_train: ndarray,
        events_group: Dict[str, List[int]],
        freq_list: List[Union[float, int]],
        phase_list: List[Union[float, int]],
        srate: Union[float, int] = 1000,
        rrate: int = 60,
        len_scale: float = 1.05,
        extract_method: str = 'Square',
        amp_scale: float = 0.8,
        concat_method: str = 'dynamic',
        resize_method: str = 'Lanczos',
        n_components: int = 1) -> Dict[str, Any]:
    """The modeling process of CIRCA.

    Args:
        X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset. Nt>=2.
        y_train (ndarray): (Ne*Nt,). Labels for X_train.
        events_group (Dict[str, List[int]]): {'event_id':[idx_1,idx_2,...]}.
            Event indices being emerged for each event.
        freq_list (list): Stimulus frequencies.
        phase_list (list): Stimulus phases.
        srate (int or float): Sampling rate. Defaults to 1000 Hz.
        rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
        len_scale (float): The multiplying power when calculating the length of data.
            Defaults to 1.05.
        extract_method (str): 'Square' or 'Cosine'. Defaults to 'Square'.
            See details in utils.extract_periodic_impulse().
        amp_scale (float): The multiplying power when calculating the amplitudes of data.
            Defaults to 0.8.
        concat_method (str): 'dynamic' or 'static'.
            'static': Concatenated data is starting from 1 s.
            'dynamic': Concatenated data is starting from 1 period.
        resize_method (str): 'nearest', 'linear', cubic', 'area', 'Lanczos',
            'linear-exact', 'inverse-map', 'fill-outliers'.
            Interpolation methods. Defaults to 'Lanczos'.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.

    Returns:
        w (ndarray): (Ne,Nk,Nc). Optimized common spatial filter.
        r (List[ndarray]): List[(Nk,Nrl)]. Optimized common impulse response.
        wX (ndarray): (Ne,Nk,Np). w @ X_mean.
        rH (ndarray): (Ne,Nk,Np). r @ H_correct.
    """
    # basic information of source domain
    X_mean = utils.generate_mean(X=X_train, y=y_train)  # (Ne(s),Nc,Np)
    n_chans, n_points = X_mean.shape[1], X_mean.shape[2]  # Nc, Np
    event_type_src = list(np.unique(y_train))

    # basic information of target domain
    event_type_tar = list(np.array([int(label) for label in events_group.keys()]))
    n_events = len(event_type_tar)  # Ne(full)

    # extract common impulse response & spatial filters
    w = np.zeros((n_events, n_components, n_chans))  # (Ne,Nk,Nc)
    r, H_correct = [], []
    for ne, label in enumerate(event_type_tar):
        # select ms- group
        merged_indices_tar = events_group[str(label)]
        merged_indices_src = [event_type_src.index(mit) for mit in merged_indices_tar]

        # config parameters for constructing convolution matrices
        tar_idx = merged_indices_src.index(label)
        freqs_temp = [freq_list[mi] for mi in merged_indices_tar]
        phase_temp = [phase_list[mi] for mi in merged_indices_tar]
        basic_idx = np.argmin(freqs_temp)  # lowest frequency idx
        basic_freq, basic_phase = freq_list[basic_idx], phase_list[basic_idx]

        # generate resized convolution matrices
        H_temp, H_correct_temp = transfer.common_conv_matrix(
            freqs=freqs_temp,
            phases=phase_temp,
            n_points=n_points,
            srate=srate,
            rrate=rrate,
            len_scale=len_scale,
            amp_scale=amp_scale,
            extract_method=extract_method,
            concat_method=concat_method,
            resize_method=resize_method
        )  # (Ne(group),Nrl,Np), (Ne(group),Nrl,Np)

        # analytical solution: {r,w} = CCA(rH, wX)
        circa_model = transfer.tim_20243374314_kernel(
            w_init=None,
            X_mean=X_mean[merged_indices_src],
            H=H_temp,
            freq=basic_freq,
            phase=basic_phase,
            srate=srate,
            n_components=n_components
        )
        w[ne] = circa_model['w']
        r.append(circa_model['r'])  # List[(Nk,Nrl)]
        H_correct.append(H_correct_temp[tar_idx])  # List[(Nrl,Np)]

    # calculate templates: wX & rH
    wX = np.zeros((n_events, n_components, n_points))  # (Ne,Nk,Np)
    rH = np.zeros_like(wX)  # (Ne,Nk,Np)
    for ne, ett in enumerate(event_type_tar):
        rH[ne] = r[ne] @ H_correct[ne]
        if ett in event_type_src:
            wX[ne] = w[ne] @ X_mean[ne]
        else:
            wX[ne] = rH[ne]
    return {'w': w, 'r': r, 'wX': wX, 'rH': rH}


def circa_feature(
        X_test: ndarray,
        circa_model: Dict[str, ndarray],
        pattern_list: List[str] = ['1', '2'],
        fusion_method: str = 'DirSum') -> Dict[str, ndarray]:
    """The pattern matching process of CIRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        circa_model (Dict[str, ndarray]): See details in circa_kernel().
        pattern_list (List[str]): Different coefficient.
            '1': corr(w @ X_test, wX)
            '2': corr(w @ X_test, rH)
        fusion_method (str): 'DirSum' or 'SignSum'.
            'DirSum': rho = rho1 + rho2. Recommended for filter-bank scenario.
            'SignSum': rho = sign(rho1)*rho1^2 + sign(rho2)*rho2^2. Better for single-band.

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in spatial filter & basic information
    w = circa_model['w']  # (Ne,Nk,Nc)
    wX, rH = circa_model['wX'], circa_model['rH']  # (Ne,Nk,Np), (Ne,Nk,Np)
    n_test = X_test.shape[0]  # Nte
    n_events = wX.shape[0]  # Ne

    # reshape templates for fastser computing
    wX = utils.fast_stan_2d(np.reshape(wX, (n_events, -1), 'C'))  # (Ne,Nk*Np)
    rH = utils.fast_stan_2d(np.reshape(rH, (n_events, -1), 'C'))  # (Ne,Nk*Np)

    # pattern matching: 2-D features
    rho_temp = np.zeros((n_test, n_events, 2))
    for nte in range(n_test):
        X_temp = np.reshape(
            a=utils.fast_stan_3d(w @ X_test[nte]),
            newshape=(n_events, -1),
            order='C'
        )  # (Ne,Nk,Np) -reshape-> (Ne,Nk*Np)
        coef = X_temp.shape[-1]

        if '1' in pattern_list:  # corr(w @ X_test, wX)
            rho_temp[nte, :, 0] = utils.fast_corr_2d(X=X_temp, Y=wX) / coef
        if '2' in pattern_list:  # corr(w @ X_test, rH)
            rho_temp[nte, :, 1] = utils.fast_corr_2d(X=X_temp, Y=rH) / coef

    # integration of features
    if fusion_method == 'DirSum':
        rho = np.zeros((n_test, n_events))
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho += rho_temp[..., pl]
    elif fusion_method == 'SignSum':
        rho = []
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho.append(rho_temp[..., pl])
        if len(rho) == 1:  # only 1 coefficient
            rho = rho[0]
        else:  # more than 1 coefficient
            rho = utils.combine_feature(rho)
    return {'rho_temp': rho_temp, 'rho': rho}


class CIRCA(cca.BasicCCA):
    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            stim_info: Dict[str, Tuple[Union[float, int], Union[float, int]]],
            srate: Union[float, int] = 1000,
            rrate: int = 60,
            len_scale: float = 1.05,
            extract_method: str = 'Square',
            amp_scale: float = 0.8,
            concat_method: str = 'dynamic',
            resize_method: str = 'Lanczos',
            events_group: Optional[Dict[str, List[int]]] = None,
            d: int = 2,
            pattern_list: List[str] = ['1', '2'],
            fusion_method: str = 'DirSum'):
        """Train CIRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            stim_info (dict): {'label': (frequency, phase)}.
            srate (int or float): Sampling rate. Defaults to 1000 Hz.
            rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
            len_scale (float): The multiplying power when calculating the length of data.
                Defaults to 1.05.
            extract_method (str): 'Square' or 'Cosine'. Defaults to 'Square'.
                See details in utils.extract_periodic_impulse().
            amp_scale (float): The multiplying power when calculating the amplitudes of data.
                Defaults to 0.8.
            concat_method (str): 'dynamic' or 'static'.
                'static': Concatenated data is starting from 1 s.
                'dynamic': Concatenated data is starting from 1 period.
            resize_method (str): 'nearest', 'linear', cubic', 'area', 'Lanczos',
                'linear-exact', 'inverse-map', 'fill-outliers'.
                Interpolation methods. Defaults to 'Lanczos'.
            events_group (Dict[str, List[int]]): {'event_id':[idx_1,idx_2,...]}.
                Event indices being emerged for each event.
            d (int): The range of events to be merged. Defaults to 2.
            pattern_list (List[str]): Different coefficient.
            fusion_method (str): 'DirSum' or 'SignSum'.
                'DirSum': rho = sum(rho(i)). Better for filter-bank.
                'SignSum': rho = sum(sign(rho(i))*rho(i)^2). Better for single-band.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.stim_info = stim_info
        self.srate = srate
        self.rrate = rrate
        self.len_scale = len_scale
        self.extract_method = extract_method
        self.amp_scale = amp_scale
        self.concat_method = concat_method
        self.resize_method = resize_method
        self.pattern_list = pattern_list
        self.fusion_method = fusion_method

        # basic information
        self.event_type = np.array([int(label) for label in list(self.stim_info.keys())])
        self.freq_list = [stim_info[str(et)][0] for et in self.event_type]
        self.phase_list = [stim_info[str(et)][1] for et in self.event_type]

        # config information of ms- groups
        if events_group is not None:
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(
                event_type=self.event_type,
                d=d
            )

        # main process
        self.training_model = circa_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            events_group=self.events_group,
            freq_list=self.freq_list,
            phase_list=self.phase_list,
            srate=self.srate,
            rrate=self.rrate,
            len_scale=self.len_scale,
            extract_method=self.extract_method,
            amp_scale=self.amp_scale,
            concat_method=self.concat_method,
            resize_method=self.resize_method,
            n_components=self.n_components
        )

    def transform(self, X_test) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return circa_feature(
            X_test=X_test,
            circa_model=self.training_model,
            pattern_list=self.pattern_list,
            fusion_method=self.fusion_method
        )


class FB_CIRCA(cca.BasicFBCCA):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=CIRCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% 2. Cross-subject IRCA with accuracy-based subject selcetion, ASS-CSIRCA
# def irca_feature(
#         X_test: ndarray,
#         sine_template: ndarray,
#         source_model: Dict[str, ndarray],
#         pattern_list: List[str] = ['1', '2', '3', '4', '5'],
#         n_components: int = 1) -> Dict[str, ndarray]:
#     """The pattern matching process of algorithm CS-IRCA.

#     Args:
#         X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
#         sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
#         source_model (dict): {'wd': ndarray (Ns,Nk,Nc),
#                               'rHd': ndarray (Ns,Ne,Nk,Np),
#                               'Uwd': ndarray (Nk,Nc),
#                               'UrHd': ndarray (Ne,Nk,Np)}
#             See details in CIRT.intra_source_training()
#         pattern_list (List[str]): Different coefficient. Labeled as '1' to '5'.
#         n_components (int): Number of eigenvectors picked as filters. Nk.
#             Defaults to 1.

#     Returns:
#         rho_temp (ndarray): (Ne*Nte,Ne,5). 5-D features.
#         rho (ndarray): (Ne*Nte,Ne). Intergrated features.
#     """
#     # load in models
#     wd, wg = source_model['wd'], source_model['wg']  # (Ns,Nk,Nc), (Ns,Ne,Nk,Nc)
#     rHd, rHg = source_model['rHd'], source_model['rHg']  # (Ns,Ne,Nk,Np), (Ns,Ne,Nk,Np)
#     Uwd, Uwg = source_model['Uwd'], source_model['Uwd']  # (Nk,Nc), (Ne,Nk,Nc)
#     UrHd, UrHg = source_model['UrHd'], source_model['UrHg']  # (Ne,Nk,Np), (Ne,Nk,Np)

#     # basic information
#     n_subjects = wd.shape[0]  # Ns
#     n_events = UrHd.shape[0]  # Ne
#     n_test = X_test.shape[0]  # Ne*Nte, could be 1 for unsupervised scenario
#     n_points = X_test.shape[-1]  # Np

#     # reshape matrix for faster computing
#     rHd = np.reshape(a=rHd, newshape=(n_subjects, n_events, -1), order='C')  # (Ns,Ne,Nk*Np)
#     rHg = np.reshape(a=rHg, newshape=(n_subjects, n_events, -1), order='C')  # (Ns,Ne,Nk*Np)
#     UrHd = np.reshape(a=UrHd, newshape=(n_events, -1), order='C')  # (Ne,Nk*Np)
#     UrHg = np.reshape(a=UrHg, newshape=(n_events, -1), order='C')  # (Ne,Nk*Np)

#     # multi-dimension features
#     rho_temp = np.zeros((n_test, n_events, 5))  # (Ne*Nte,Ne,5)
#     for nte in range(n_test):
#         if '1' in pattern_list:  # discriminative pattern matching with each source subject
#             wdX_temp = np.reshape(
#                 a=np.tile(
#                     A=utils.fast_stan_3d(wd @ X_test[nte])[:, None, ...],
#                     reps=(1, n_events, 1, 1)
#                 ),
#                 newshape=(n_subjects, n_events, -1),
#                 order='C'
#             )  # (Ns,Nk,Np) -repeat-> (Ns,Ne,Nk,Np) -reshape-> (Ns,Ne,Nk*Np)
#             rho_temp[nte, :, 0] = np.mean(
#                 a=utils.fast_corr_3d(X=wdX_temp, Y=rHd),
#                 axis=0
#             ) / wdX_temp.shape[-1]  # (Ns,Ne,Nk*Np) -corr-> (Ns,Ne) -avg-> (Ne,)

#         if '2' in pattern_list:  # discriminative pattern matching with prototype model
#             UwdX_temp = np.reshape(
#                 a=np.tile(
#                     A=utils.fast_stan_2d(Uwd @ X_test[nte]),
#                     reps=(n_events, 1, 1)
#                 ),
#                 newshape=(n_events, -1),
#                 order='C'
#             )  # (Nk,Np) -repeat-> (Ne,Nk,Np) -reshape-> (Ne,Nk*Np)
#             rho_temp[nte, :, 1] = utils.fast_corr_2d(X=UwdX_temp, Y=UrHd) / UwdX_temp.shape[-1]

#         if '3' in pattern_list:  # CCA
#             for nem in range(n_events):
#                 cca_model = cca.cca_kernel(
#                     X=X_test[nte],
#                     Y=sine_template[nem],
#                     n_components=n_components
#                 )
#                 rho_temp[nte, nem, 2] = cca_model['coef']

#         if '4' in pattern_list:  # generative pattern matching with each source subject
#             wgX_temp = np.reshape(
#                 a=utils.fast_stan_4d(wg @ X_test[nte]),
#                 newshape=(n_subjects, n_events, -1),
#                 order='C'
#             )  # (Ns,Ne,Nk,Np) -reshape-> (Ns,Ne,Nk*Np)
#             rho_temp[nte, :, 3] = np.mean(
#                 a=utils.fast_corr_3d(X=wgX_temp, Y=rHg),
#                 axis=0
#             ) / n_points  # (Ns,Ne,Nk*Np) -corr-> (Ns,Ne) -avg-> (Ne,)

#         if '5' in pattern_list:  # generative pattern matching with prototype model
#             UwgX_temp = np.reshape(
#                 a=np.tile(
#                     A=utils.fast_stan_2d(Uwg @ X_test[nte]),
#                     reps=(n_events, 1, 1)
#                 ),
#                 newshape=(n_events, -1),
#                 order='C'
#             )  # (Nk,Np) -repeat-> (Ne,Nk,Np) -reshape-> (Ne,Nk*Np)
#             rho_temp[nte, :, 4] = utils.fast_corr_2d(X=UwgX_temp, Y=UrHg) / n_points

#     # integration of features
#     rho = []
#     for pl in range(5):
#         if str(pl + 1) in pattern_list:
#             rho.append(rho_temp[..., pl])
#     return {'rho_temp': rho_temp, 'rho': utils.combine_feature(rho)}


# class CIRCA_ASS(transfer.BasicASS):
#     """Accuracy-based subject selection (ASS) for CIRCA."""
#     def __init__(
#             self,
#             X_source: List[ndarray],
#             y_source: List[ndarray],
#             stim_info: Dict[str, Tuple[Union[float, int], Union[float, int]]],
#             srate: Union[float, int] = 1000,
#             rrate: int = 60,
#             len_scale: float = 1.05,
#             amp_scale: float = 0.8,
#             concat_method: str = 'dynamic',
#             pattern_list: List[str] = ['1', '2'],
#             n_components: int = 1,
#             n_splits: int = 10,
#             train_size: Union[float, int] = 0.8,
#             thred: Union[float, int] = 0.5,
#             **kwargs):
#         """Basic configuration.

#         Args:
#             X_source (List[ndarray]): List[(Ne*Nt,Nc,Np)]. Source dataset.
#             y_source (List[ndarray]): List[(Ne*Nt,)]. Labels for X_source.
#             stim_info (dict): {'label': (frequency, phase)}.
#             srate (int or float): Sampling rate. Defaults to 1000 Hz.
#             rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
#             len_scale (float): The multiplying power when calculating the length of data.
#                 Defaults to 1.05.
#             amp_scale (float): The multiplying power when calculating the amplitudes of data.
#                 Defaults to 0.8.
#             concat_method (str): 'dynamic' or 'static'.
#                 'static': Concatenated data is starting from 1 s.
#                 'dynamic': Concatenated data is starting from 1 period.
#             pattern_list (List[str]): Different coefficient.
#             n_splits (int): The number of folds for cross-validation.
#             train_size (float or int). See details in StratifiedShuffleSplit().
#             thred (float or int). When float: the percentage of the total subjects.
#                 When int: the specific number of the selected subjects.
#         """
#         # config model
#         self.X_source = X_source
#         self.y_source = y_source
#         self.stim_info = stim_info
#         self.srate = srate
#         self.rrate = rrate
#         self.len_scale = len_scale
#         self.amp_scale = amp_scale
#         self.concat_method = concat_method
#         self.pattern_list = pattern_list
#         self.n_components = n_components
#         self.n_splits = n_splits
#         self.train_size = train_size
#         self.thred = thred

#     def evaluation(self):
#         """Calculate the CIRCA classification accuracy for each source subject."""
#         # basic information
#         self.n_subjects = len(self.X_source)

#         # apply CIRCA classification
#         self.acc_list = np.zeros((self.n_subjects))
#         for nsub in range(self.n_subjects):
#             # initialization for cross-validation
#             X_temp, y_temp = self.X_source[nsub], self.y_source[nsub]
#             sss = StratifiedShuffleSplit(
#                 n_splits=self.n_splits,
#                 train_size=self.train_size,
#                 random_state=0
#             )

#             # apply CIRCA model for each source subject
#             model = CIRCA(n_components=self.n_components)
#             for _, (train_idx, test_idx) in enumerate(sss.split(X_temp, y_temp)):
#                 X_train, y_train = X_temp[train_idx], y_temp[train_idx]
#                 X_test, y_test = X_temp[test_idx], y_temp[test_idx]

#                 model.fit(
#                     X_train=X_train,
#                     y_train=y_train,
#                     stim_info=self.stim_info,
#                     srate=self.srate,
#                     rrate=self.rrate,
#                     len_scale=self.len_scale,
#                     amp_scale=self.amp_scale,
#                     concat_method=self.concat_method,
#                     pattern_list=self.pattern_list
#                 )
#                 y_pred = model.predict(X_test=X_test)
#                 self.acc_list[nsub] += utils.acc_compute(y_true=y_test, y_pred=y_pred)
#         self.acc_list /= self.n_splits

#     def select_subjects(self) -> List[int]:
#         """Main process.

#         Returns:
#             subject_indices (List[int]).
#         """
#         self.evaluation()
#         self.sort_subject_list()

#         if isinstance(self.thred, int):
#             return self.sorted_idx[:self.thred]
#         elif isinstance(self.thred, float):
#             selected_num = int(np.ceil(self.n_subjects * self.thred))
#             return self.sorted_idx[:selected_num]


# class FB_CIRCA_ASS(CIRCA_ASS):
#     """Accuracy-based subject selection (ASS) for FB-CIRCA."""
#     def evaluation(self):
#         """Calculate the FB-CIRCA classification accuracy for each source subject."""
#         # basic information
#         self.n_subjects = len(self.X_source)

#         # apply FB-CIRCA classification
#         self.acc_list = np.zeros((self.n_subjects))
#         for nsub in range(self.n_subjects):
#             # initialization for cross-validation
#             X_temp, y_temp = self.X_source[nsub], self.y_source[nsub]
#             sss = StratifiedShuffleSplit(
#                 n_splits=self.n_splits,
#                 train_size=self.train_size,
#                 random_state=0
#             )

#             # apply FB-CIRCA model for each source subject
#             model = FB_CIRCA(n_components=self.n_components)
#             for _, (train_idx, test_idx) in enumerate(sss.split(X_temp[0], y_temp)):
#                 X_train, y_train = X_temp[:, train_idx, ...], y_temp[train_idx]
#                 X_test, y_test = X_temp[:, test_idx, ...], y_temp[test_idx]

#                 model.fit(
#                     X_train=X_train,
#                     y_train=y_train,
#                     stim_info=self.stim_info,
#                     srate=self.srate,
#                     rrate=self.rrate,
#                     len_scale=self.len_scale,
#                     amp_scale=self.amp_scale,
#                     concat_method=self.concat_method,
#                     pattern_list=self.pattern_list
#                 )
#                 y_pred = model.predict(X_test=X_test)
#                 self.acc_list[nsub] += utils.acc_compute(y_true=y_test, y_pred=y_pred)
#         self.acc_list /= self.n_splits


# class ASS_IRCA(transfer.BasicTransfer):
#     def intra_source_training(self):
#         """Intra-domain model training for source dataset."""
#         # basic information & initialization
#         n_events = self.event_type.shape[0]  # Ne
#         self.source_model = {}

#         # train discriminative model: common impulse responses
#         if any(pattern in ['1', '2'] for pattern in self.pattern_list):
#             wd, rd, wXd, rHd = [], [], [], []
#             for nsub in range(self.n_subjects):
#                 X_mean = utils.generate_mean(X=self.X_source[nsub], y=self.y_source[nsub])
#                 common_model = transfer.tim_20243374314_kernel(
#                     w_init=None,
#                     X_mean=X_mean,
#                     H=self.Hd,
#                     freq=self.freq_tar,
#                     phase=self.phase_tar,
#                     optimize_method='CCA',
#                     n_components=1
#                 )
#                 wd.append(common_model['w'])  # (Nk,Nc)
#                 rd.append(common_model['r'])  # (Nk,Nrl)
#                 wXd.append(np.reshape(
#                     a=common_model['wX'],
#                     newshape=(self.n_components, n_events, X_mean.shape[-1]),
#                     order='C'
#                 ).transpose(1, 0, 2))  # (Nk,Ne*Np) -reshape & transpose-> (Ne,Nk,Np)
#                 rHd.append(np.reshape(
#                     a=common_model['rH'],
#                     newshape=(self.n_components, n_events, X_mean.shape[-1]),
#                     order='C'
#                 ).transpose(1, 0, 2))  # (Nk,Ne*Np) -reshape & transpose-> (Ne,Nk,Np)
#             del nsub, common_model, X_mean

#             # integrade source model
#             self.source_model['wd'] = np.stack(wd, axis=0)  # (Ns,Nk,Nc)
#             self.source_model['rd'] = np.stack(rd, axis=0)  # (Ns,Nk,Nrl)
#             self.source_model['wXd'] = np.stack(wXd, axis=0)  # (Ns,Ne,Nk,Np)
#             self.source_model['rHd'] = np.stack(rHd, axis=0)  # (Ns,Ne,Nk,Np)
#             del wd, rd, wXd, rHd

#             # train prototypes Urd & Uwd for r & w
#             self.source_model['Urd'] = transfer.construct_prototype_filter(
#                 ws=self.source_model['rd'],
#                 preprocessed=False
#             )  # (Nk,Nrl)
#             self.source_model['Uwd'] = transfer.construct_prototype_filter(
#                 ws=self.source_model['wd'],
#                 preprocessed=True
#             )  # (Nk,Nc)

#             # train prototype for common superposition templates: UrHd
#             UrHd = np.zeros_like(self.source_model['rHd'][0])  # (Ne,Nk,Np)
#             for ne in range(n_events):
#                 UrHd[ne] = self.source_model['Urd'] @ self.Hd_correct[ne]
#             self.source_model['UrHd'] = utils.fast_stan_3d(UrHd)
#             del UrHd, ne

#         # train generative model: frequency-specific impulse responses
#         if any(pattern in ['4', '5'] for pattern in self.pattern_list):
#             wg, rg, wXg, rHg = [], [], [], []
#             for nsub in range(self.n_subjects):
#                 tlcca_model = transfer.tlcca_kernel(
#                     X_train=self.X_source[nsub],
#                     y_train=self.y_source[nsub],
#                     stim_info=self.stim_info,
#                     H=self.Hg,
#                     w_init=None,
#                     srate=self.srate,
#                     optimize_method='CCA',
#                     n_components=self.n_components,
#                     target_chan_idx=self.target_chan_idx
#                 )

#                 # update source model
#                 wg.append(tlcca_model['w'])  # (Ne,Nk,Nc)
#                 rg.append(tlcca_model['r'])  # List[(Nk,Nrl)]
#                 wXg.append(tlcca_model['wX'])  # (Ne,Nk,Np)
#                 rHg.append(tlcca_model['rH'])  # (Ne,Nk,Np)
#             del nsub

#             # integrade source model
#             self.source_model['wg'] = np.stack(wg, axis=0)  # (Ns,Ne,Nk,Nc)
#             self.source_model['rg'] = deepcopy(rg)  # List[List[(Nk,Nrl)]]
#             self.source_model['wXg'] = np.stack(wXg, axis=0)  # (Ns,Ne,Nk,Np)
#             self.source_model['rHg'] = np.stack(rHg, axis=0)  # (Ns,Ne,Nk,Np)
#             del wg, rg, wXg, rHg

#             # train prototye filters: Urg & Uwg for rg & wg
#             Urg, Uwg = [], []
#             for ne in range(n_events):
#                 # prototye filters for generative impulse response (r): Urg
#                 rs_temp = [self.source_model['rg'][nsub][ne]
#                            for nsub in range(self.n_subjects)]
#                 rs_temp = np.stack(rs_temp, axis=0)  # List[(Nk,Nrl)] -> (Ns,Nk,Nrl)
#                 Urg.append(transfer.construct_prototype_filter(
#                     ws=rs_temp,
#                     preprocessed=False
#                 ))  # List[(Nk,Nrl)]

#                 # prototype filters for generative spatial filters (w): Uwg
#                 ws_temp = self.source_model['wg'][:, ne, ...]  # (Ns,Nk,Nc)
#                 Uwg.append(transfer.construct_prototype_filter(
#                     ws=ws_temp,
#                     preprocessed=True
#                 ))  # List[(Nk,Nc)]
#                 del rs_temp, ws_temp
#             del ne
#             self.source_model['Urg'] = deepcopy(Urg)  # List[(Nk,Nrl)]
#             self.source_model['Uwg'] = np.stack(Uwg, axis=0)  # (Ne,Nk,Nc)
#             del Urg, Uwg

#             # train prototype templates: UrHg
#             UrHg = np.zeros_like(self.source_model['rHg'][0])  # (Ne,Nk,Np)
#             for ne in range(n_events):
#                 UrHg[ne] = self.source_model['Urg'][ne] @ self.Hg_correct[ne]
#             self.source_model['UrHg'] = deepcopy(UrHg)
#             del UrHg, ne

#     def fit(
#             self,
#             X_train: List[ndarray],
#             y_train: List[ndarray],
#             sine_template: ndarray,
#             stim_info: Dict[str, Tuple[Union[float, int], Union[float, int]]],
#             srate: Union[float, int] = 1000,
#             rrate: int = 60,
#             len_scale: float = 1.05,
#             amp_scale: float = 0.8,
#             concat_method: str = 'dynamic',
#             target_chan_idx: int = 7,
#             pattern_list: List[str] = ['1', '2', '3', '4', '5'],
#             selection: Optional[List[int]] = None,
#             n_splits: int = 10,
#             train_size: Union[float, int] = 0.8,
#             thred: Union[float, int] = 0.5):
#         """Train IRCA model.

#         Args:
#             X_train (ndarray): List[(Ne*Nt,Nc,Np)]. Source training dataset. Nt>=2.
#             y_train (ndarray): List[(Ne(s)*Nt,)]. Labels for X_train.
#             sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
#             stim_info (dict): {'label': (frequency, phase)}.
#             srate (int or float): Sampling rate. Defaults to 1000 Hz.
#             rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
#             len_scale (float): The multiplying coefficient for the length of data.
#                 Defaults to 1.05.
#             amp_scale (float): The multiplying coefficient for the amplitudes of data.
#                 Defaults to 0.8.
#             concat_method (str): 'dynamic' or 'static'.
#                 'static': Concatenated data is starting from 1 s.
#                 'dynamic': Concatenated data is starting from 1 period.
#             target_chan_idx (int): The index of target channel to correct of w & r.
#                 Recommend to set to the channel 'Oz'. Defaults to 7.
#                 See details in solve_tlcca_func().
#             pattern_list (List[str]): Different coefficient. Labeled as '1' to '5'.
#             selection (List[int], optional): If None, use CIRCA_ASS() to select subjects.
#             n_splits (int): The number of folds for cross-validation.
#             train_size (float or int). See details in StratifiedShuffleSplit().
#             min_acc (float, optional). The minimum accuracy for selected subjects.
#             max_n_sub (int, optional). The maximum number of selected subjects.
#                 Defaults to None.
#         """
#         # load in parameters
#         self.sine_template = sine_template
#         self.stim_info = stim_info
#         self.srate = srate
#         self.rrate = rrate
#         self.len_scale = len_scale
#         self.amp_scale = amp_scale
#         self.concat_method = concat_method
#         self.target_chan_idx = target_chan_idx
#         self.pattern_list = pattern_list
#         self.n_splits = n_splits
#         self.train_size = train_size
#         self.thred = thred

#         # select best source subjects
#         if isinstance(selection, list):  # recommended
#             self.selection = selection
#         else:  # unrecommended
#             ass = CIRCA_ASS(
#                 X_source=X_train,
#                 y_source=y_train,
#                 stim_info=self.stim_info,
#                 srate=self.srate,
#                 rrate=self.rrate,
#                 len_scale=self.len_scale,
#                 amp_scale=self.amp_scale,
#                 concat_method=self.concat_method,
#                 pattern_list=['1', '2'],
#                 n_components=self.n_components,
#                 n_splits=self.n_splits,
#                 train_size=self.train_size,
#                 thred=self.thred
#             )
#             self.selection = ass.select_subjects()
#         self.X_source, self.y_source = [], []
#         for se in self.selection:
#             self.X_source.append(X_train[se])
#             self.y_source.append(y_train[se])

#         # basic information of source domain
#         label_lst = [int(label) for label in list(stim_info.keys())]
#         freq_lst = [stim_info[str(label)][0] for label in label_lst]
#         phase_lst = [stim_info[str(label)][1] for label in label_lst]

#         sorted_pairs = sorted(enumerate(freq_lst), key=lambda x: x[1])
#         sorted_idx = [sp[0] for sp in sorted_pairs]
#         del sorted_pairs

#         freqs = [freq_lst[si] for si in sorted_idx]
#         phases = [phase_lst[si] for si in sorted_idx]
#         self.event_type = np.array([label_lst[si] for si in sorted_idx])
#         del freq_lst, phase_lst, label_lst, sorted_idx
#         self.freq_tar = freqs[0]  # lowest frequency, longest response
#         self.phase_tar = phases[0]

#         self.n_subjects = len(self.X_source)  # Ns
#         self.source_info = []
#         for nsub in range(self.n_subjects):
#             self.source_info.append(utils.generate_data_info(
#                 X=self.X_source[nsub],
#                 y=self.y_source[nsub]
#             ))

#         # config convolution matrices for discriminative model
#         n_points = self.X_source[0].shape[-1]  # Np
#         self.Hd, self.Hd_correct = transfer.common_conv_matrix(
#             freqs=freqs,
#             phases=phases,
#             n_points=n_points,
#             srate=self.srate,
#             rrate=self.rrate,
#             len_scale=self.len_scale,
#             amp_scale=self.amp_scale,
#             extract_method='Square',
#             concat_method=self.concat_method,
#             resize_method='Lanczos'
#         )  # (Ne, longest Nrl, Np)
#         del freqs, phases

#         # config convolution matrices for generative model
#         self.Hg, self.Hg_correct = [], []
#         for _, et in enumerate(self.event_type):
#             H_temp, H_correct_temp = transfer.tlcca_conv_matrix(
#                 freq=self.stim_info[str(et)][0],
#                 phase=self.stim_info[str(et)][1],
#                 n_points=n_points,
#                 srate=self.srate,
#                 rrate=self.rrate,
#                 len_scale=self.len_scale,
#                 amp_scale=self.amp_scale,
#                 extract_method='Square',
#                 concat_method=self.concat_method,
#                 response_length=None
#             )
#             self.Hg.append(H_temp)
#             self.Hg_correct.append(H_correct_temp)
#         del H_temp, H_correct_temp, et

#         # main process
#         self.intra_source_training()

#     def transform(self, X_test):
#         """Transform test dataset to features.

#         Args:
#             X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

#         Returns:
#             rho_temp (ndarray): (Ne*Nte,Ne,3). 3-D features.
#             rho (ndarray): (Ne*Nte,Ne). Intergrated features.
#         """
#         return irca_feature(
#             X_test=X_test,
#             sine_template=self.sine_template,
#             source_model=self.source_model,
#             pattern_list=self.pattern_list,
#             n_components=self.n_components
#         )


# class FB_ASS_IRCA(transfer.BasicFBTransfer):
#     def __init__(
#             self,
#             filter_bank: Optional[List] = None,
#             with_filter_bank: bool = True,
#             n_components: int = 1):
#         """Basic configuration.

#         Args:
#             filter_bank (List[ndarray], optional):
#                 See details in utils.generate_filter_bank(). Defaults to None.
#             with_filter_bank (bool): Whether the input data has been FB-preprocessed.
#                 Defaults to True.
#             n_components (int): Number of eigenvectors picked as filters.
#                 Defaults to 1.
#         """
#         self.n_components = n_components
#         super().__init__(
#             base_estimator=ASS_IRCA(n_components=self.n_components),
#             filter_bank=filter_bank,
#             with_filter_bank=with_filter_bank,
#             version='SSVEP'
#         )

#     def fit(
#             self,
#             X_train: List[ndarray],
#             y_train: List[ndarray],
#             bank_weights: Optional[ndarray] = None,
#             **kwargs):
#         """Load in training dataset and pass it to sub-esimators.
#         See details in TBME_20243406603().fit().

#         Args:
#             X_train (ndarray): List[(Nb,Ne*Nt,Nc,Np)]. Source training dataset. Nt>=2.
#             y_train (ndarray): List[(Nb,Ne(s)*Nt,)]. Labels for X_train.
#             bank_weights (ndarray, optional): Weights for different filter banks.
#                 Defaults to None (equal).
#         """
#         # basic information
#         self.Nb = X_train[0].shape[0]  # n_bands
#         n_subjects = len(X_train)  # Ns

#         # initialization
#         self.bank_weights = bank_weights
#         if self.version == 'SSVEP' and self.bank_weights is None:
#             self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25 for nb in range(self.Nb)])
#         self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]

#         # select best source subjects
#         if isinstance(kwargs['selection'], list):  # recommended
#             self.selection = kwargs['selection']  # actually equals to 'pass'
#         else:  # recommended
#             ass = FB_CIRCA_ASS(X_source=X_train, y_source=y_train, **kwargs)
#             kwargs['selection'] = ass.select_subjects()

#         # apply in each sub-band
#         for nb, se in enumerate(self.sub_estimator):
#             se.fit(
#                 X_train=[X_train[nsub][nb] for nsub in range(n_subjects)],
#                 y_train=y_train,
#                 **kwargs
#             )


# %% 3. Subject transfer based CIRCA, st-CIRCA
def stcirca_source_training(
        filter_bank_idx: int,
        data_path: List[str],
        chan_indices: List[int],
        time_range: Tuple[int, int],
        events_group: Dict[str, List[int]],
        freq_list: List[Union[float, int]],
        phase_list: List[Union[float, int]],
        srate: Union[float, int] = 1000,
        rrate: int = 60,
        len_scale: float = 1.05,
        extract_method: str = 'Square',
        amp_scale: float = 0.8,
        concat_method: str = 'dynamic',
        resize_method: str = 'Lanczos',
        n_components: int = 1,
        with_filter_banks: bool = False) -> Dict[str, ndarray]:
    """The modeling process of ST-CIRCA for source dataset.

    Args:
        filter_bank_idx (int): The index of filter banks to be used in model training process.
            Could be any value when 'with_filter_banks' is False.
            Setting to be the 1st parameter for parallely computing.
        data_path (List[str]): List of source data file paths.
        chan_indices (List[int]): Indices of selected channels.
        time_range (Tuple[int, int]): Start & end indices of time points.
        events_group (Dict[str, List[int]]): {'event_id':[idx_1,idx_2,...]}.
            Event indices being emerged for each event.
        freq_list (list): Stimulus frequencies.
        phase_list (list): Stimulus phases.
        srate (int or float): Sampling rate. Defaults to 1000 Hz.
        rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
        len_scale (float): The multiplying power when calculating the length of data.
            Defaults to 1.05.
        extract_method (str): 'Square' or 'Cosine'. Defaults to 'Square'.
            See details in utils.extract_periodic_impulse().
        amp_scale (float): The multiplying power when calculating the amplitudes of data.
            Defaults to 0.8.
        concat_method (str): 'dynamic' or 'static'.
            'static': Concatenated data is starting from 1 s.
            'dynamic': Concatenated data is starting from 1 period.
        resize_method (str): 'nearest', 'linear', cubic', 'area', 'Lanczos',
            'linear-exact', 'inverse-map', 'fill-outliers'.
            Interpolation methods. Defaults to 'Lanczos'.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.
        with_filter_banks (bool): Defaults to False, i.e. the shape of data is (Ne*Nte,Nc,Np).

    Returns:
        w_source (ndarray): (Ns,Ne,Nk,Nc). w.
        wX (ndarray): (Ns,Ne,Nk,Np). w @ X_mean.
        rH (ndarray): (Ns,Ne,Nk,Np). r @ H_correct.
    """
    # basic information
    n_subjects = len(data_path)  # Ns

    # main process
    w_source, wX_source, rH_source = [], [], []
    for nsub in range(n_subjects):
        # load in source data
        with open(data_path[nsub], 'rb') as file:
            eeg = pickle.load(file)
        X_source = eeg['X'][..., chan_indices, time_range[0]:time_range[1]]
        y_source = eeg['y']
        del eeg, file

        if with_filter_banks:
            X_source = X_source[filter_bank_idx]

        # train CIRCA model
        model = circa_kernel(
            X_train=X_source,
            y_train=y_source,
            events_group=events_group,
            freq_list=freq_list,
            phase_list=phase_list,
            srate=srate,
            rrate=rrate,
            len_scale=len_scale,
            extract_method=extract_method,
            amp_scale=amp_scale,
            concat_method=concat_method,
            resize_method=resize_method,
            n_components=n_components
        )
        w_source.append(model['w'])  # List[(Ne,Nk,Nc)]
        wX_source.append(model['wX'])  # List[(Ne,Nk,Np)]
        rH_source.append(model['rH'])  # List[(Ne,Nk,Np)]
    return {
        'w_source': np.stack(w_source),
        'wX_source': np.stack(wX_source),
        'rH_source': np.stack(rH_source)
    }


def stcirca_feature(
        X_test: ndarray,
        trans_model: Dict[str, ndarray],
        target_model: Dict[str, Any],
        pattern_list: List[str] = ['1', '2', '3', '4'],
        fusion_method: str = 'DirSum') -> Dict[str, ndarray]:
    """The pattern matching process of ST-CIRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        trans_model (Dict): {'wX_trans': ndarray (Ne,Nk,Np),
                             'rH_trans': ndarray (Ne,Nk,Np)}
            See details in ST_CIRCA.tranfer_learning().
        target_model (Dict): {'w': ndarray (Ne,Nk,Nc),
                              'wX': ndarray (Ne,Nk,Np),
                              'rH': ndarray (Ne,Nk,Np)}
            See details in ST_CIRCA.intra_target_training().
        pattern_list (List[str]): Different coefficient.
            '1': corr(w @ X_test, wX_target)
            '2': corr(w @ X_test, rH_target)
            '3': corr(w @ X_test, wX_trans)
            '4': corr(w @ X_test, rH_trans)
        fusion_method (str): 'DirSum' or 'SignSum'.
            'DirSum': rho = rho1 + rho2. Recommended for filter-bank scenario.
            'SignSum': rho = sign(rho1)*rho1^2 + sign(rho2)*rho2^2. Better for single-band.

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,4). 2-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models
    w = target_model['w']  # (Ne,Nk,Nc)
    wX, rH = target_model['wX'], target_model['rH']  # (Ne,Nk,Np)
    wX_trans, rH_trans = trans_model['wX_trans'], trans_model['rH_trans']  # (Ne,Nk,Np)
    n_test = X_test.shape[0]
    n_events = wX.shape[0]  # Ne

    # reshape & standardize templates for faster computing
    wX = utils.fast_stan_2d(np.reshape(wX, (n_events, -1), 'C'))  # (Ne,Nk*Np)
    rH = utils.fast_stan_2d(np.reshape(rH, (n_events, -1), 'C'))  # (Ne,Nk*Np)
    wX_trans = utils.fast_stan_2d(np.reshape(wX_trans, (n_events, -1), 'C'))  # (Ne,Nk*Np)
    rH_trans = utils.fast_stan_2d(np.reshape(rH_trans, (n_events, -1), 'C'))  # (Ne,Nk*Np)

    # pattern matching: 4-d features
    rho_temp = np.zeros((n_test, n_events, 4))
    for nte in range(n_test):
        X_temp = np.reshape(
            a=utils.fast_stan_3d(w @ X_test[nte]),
            newshape=(n_events, -1),
            order='C'
        )  # (Ne,Nk,Np) -reshape-> (Ne,Nk*Np)
        coef = X_temp.shape[-1]

        if '1' in pattern_list:  # corr(w @ X_test, wX_target)
            rho_temp[nte, :, 0] = utils.fast_corr_2d(X=X_temp, Y=wX) / coef
        if '2' in pattern_list:  # corr(w @ X_test, rH_target)
            rho_temp[nte, :, 1] = utils.fast_corr_2d(X=X_temp, Y=rH) / coef
        if '3' in pattern_list:  # corr(w @ X_test, wX_trans)
            rho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp, Y=wX_trans) / coef
        if '4' in pattern_list:  # corr(w @ X_test, rH_trans)
            rho_temp[nte, :, 3] = utils.fast_corr_2d(X=X_temp, Y=rH_trans) / coef

    # integration of features
    if fusion_method == 'DirSum':
        rho = np.zeros((n_test, n_events))
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho += rho_temp[..., pl]
    elif fusion_method == 'SignSum':
        rho = []
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho.append(rho_temp[..., pl])
        if len(rho) == 1:  # only 1 coefficient
            rho = rho[0]
        else:  # more than 1 coefficient
            rho = utils.combine_feature(rho)
    return {'rho_temp': rho_temp, 'rho': rho}


class ST_CIRCA(transfer.BasicTransfer):
    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        self.target_model = circa_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            events_group=self.events_group,
            freq_list=self.freq_list,
            phase_list=self.phase_list,
            srate=self.srate,
            rrate=self.rrate,
            len_scale=self.len_scale,
            extract_method=self.extract_method,
            amp_scale=self.amp_scale,
            concat_method=self.concat_method,
            resize_method=self.resize_method,
            n_components=self.n_components
        )

    def weight_calc(self):
        """Optimize the transfer weights."""
        # basic information
        event_type_target = self.event_type_target.tolist()
        event_type_source = self.event_type.tolist()  # Ne (full)
        select_indices = [event_type_source.index(ett) for ett in event_type_target]

        # solve LST problem for wX | DO NOT standardize AwX for better performance
        self.bwX = np.reshape(
            a=self.target_model['wX'],
            newshape=(len(event_type_target), -1),
            order='C'
        )  # (Ne(t),Nk,Np) -reshape-> (Ne(t),Nk*Np)
        self.bwX = np.reshape(utils.fast_stan_2d(self.bwX), -1, 'C')  # (Ne(t)*Nk*Np,)
        self.AwX = np.reshape(
            a=self.source_model['wX_source'][:, select_indices, :],
            newshape=(self.n_subjects, len(event_type_source), -1),
            order='C'
        )  # (Ns,Ne,Nk,Np) -select-> (Ns,Ne(t),Nk,Np) -reshape-> (Ns,Ne(t),Nk*Np)
        self.AwX = np.reshape(
            a=self.AwX,
            newshape=(self.n_subjects, -1),
            order='C'
        ).T  # (Ns,Ne(t),Nk*Np) -reshape-> (Ns,Ne(t)*Nk*Np) -tranpose-> (Ne(t)*Nk*Np,Ns)
        self.weight_wX, _, _, _ = sLA.lstsq(a=self.AwX, b=self.bwX)  # (Ns,)

        # solve LST problem for rH | DO NOT standardize ArH for better performance
        self.brH = np.reshape(
            a=self.target_model['rH'],
            newshape=(len(event_type_target), -1),
            order='C'
        )  # (Ne(t),Nk,Np) -reshape-> (Ne(t),Nk*Np)
        self.brH = np.reshape(utils.fast_stan_2d(self.brH), -1, 'C')  # (Ne(t)*Nk*Np,)
        self.ArH = np.reshape(
            a=self.source_model['rH_source'][:, select_indices, :],
            newshape=(self.n_subjects, len(event_type_source), -1),
            order='C'
        )  # (Ns,Ne,Nk,Np) -select-> (Ns,Ne(t),Nk,Np) -reshape-> (Ns,Ne(t),Nk*Np)
        self.ArH = np.reshape(
            a=self.ArH,
            newshape=(self.n_subjects, -1),
            order='C'
        ).T  # (Ns,Ne(t),Nk*Np) -reshape-> (Ns,Ne(t)*Nk*Np) -tranpose-> (Ne(t)*Nk*Np,Ns)
        self.weight_rH, _, _, _ = sLA.lstsq(a=self.ArH, b=self.brH)  # (Ns,)

    def transfer_learning(self):
        """Transfer learning process."""
        wX_trans = np.einsum('s,sekp->ekp', self.weight_wX, self.source_model['wX_source'])
        rH_trans = np.einsum('s,sekp->ekp', self.weight_rH, self.source_model['rH_source'])
        self.trans_model = {'wX_trans': wX_trans, 'rH_trans': rH_trans}

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            stim_info: Dict[str, Tuple[Union[float, int], Union[float, int]]],
            srate: Union[float, int] = 1000,
            rrate: int = 60,
            len_scale: float = 1.05,
            extract_method: str = 'Square',
            amp_scale: float = 0.8,
            concat_method: str = 'dynamic',
            resize_method: str = 'Lanczos',
            events_group: Optional[Dict[str, List[int]]] = None,
            d: int = 2,
            pattern_list: List[str] = ['1', '2', '3', '4'],
            fusion_method: str = 'DirSum',
            source_model: Optional[Dict[str, ndarray]] = None):
        """Train ST-CIRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            stim_info (dict): {'label': (frequency, phase)}.
            srate (int or float): Sampling rate. Defaults to 1000 Hz.
            rrate (int or float): Refresh rate of stimulus devices. Defaults to 60 Hz.
            len_scale (float): The multiplying power when calculating the length of data.
                Defaults to 1.05.
            extract_method (str): 'Square' or 'Cosine'. Defaults to 'Square'.
                See details in utils.extract_periodic_impulse().
            amp_scale (float): The multiplying power when calculating the amplitudes of data.
                Defaults to 0.8.
            concat_method (str): 'dynamic' or 'static'.
                'static': Concatenated data is starting from 1 s.
                'dynamic': Concatenated data is starting from 1 period.
            resize_method (str): 'nearest', 'linear', cubic', 'area', 'Lanczos',
                'linear-exact', 'inverse-map', 'fill-outliers'.
                Interpolation methods. Defaults to 'Lanczos'.
            events_group (Dict): {'event_id':[idx_1,idx_2,...]}.
                Event indices being emerged for each event.
            d (int): The range of events to be merged. Defaults to 2.
            pattern_list (List[str]): Different coefficient.
            fusion_method (str): 'DirSum' or 'SignSum'. See details in stcirca_feature().
            source_model (Dict): See details in stcirca_source_training().
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.stim_info = stim_info
        self.srate = srate
        self.rrate = rrate
        self.len_scale = len_scale
        self.extract_method = extract_method
        self.amp_scale = amp_scale
        self.concat_method = concat_method
        self.resize_method = resize_method
        self.pattern_list = pattern_list
        self.fusion_method = fusion_method

        # basic information of stimuli
        self.event_type = np.array([int(label) for label in list(self.stim_info.keys())])
        self.event_type_target = np.unique(y_train)
        self.freq_list = [self.stim_info[str(et)][0] for et in self.event_type]
        self.phase_list = [self.stim_info[str(et)][1] for et in self.event_type]

        # config information of ms- groups
        if events_group is not None:
            self.events_group = events_group
        else:
            self.events_group = utils.augmented_events(
                event_type=self.event_type,
                d=d
            )

        # main process
        self.source_model = source_model
        self.n_subjects = self.source_model['wX_source'].shape[0]  # Ns
        self.intra_target_training()
        self.weight_calc()
        self.transfer_learning()

    def transform(self, X_test) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features.
        """
        return stcirca_feature(
            X_test=X_test,
            trans_model=self.trans_model,
            target_model=self.target_model,
            pattern_list=self.pattern_list,
            fusion_method=self.fusion_method
        )


class FB_ST_CIRCA(cca.BasicFBCCA):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=ST_CIRCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            source_model: List[Dict[str, ndarray]],
            bank_weights: Optional[ndarray] = None,
            **kwargs):
        """Load in training dataset and pass it to sub-esimators.

        Args:
            X_train (ndarray): (Ne*Nt,...,Np) or (Nb,Ne*Nt,...,Np) (with_filter_bank=True).
                Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            bank_weights (ndarray, optional): Weights for different filter banks.
                Defaults to None (equal).
        """
        if self.with_filter_bank:  # X_train has been filterd: (Nb,Ne*Nt,...,Np)
            self.Nb = X_train.shape[0]
        else:  # tranform X_train into shape: (Nb,Ne*Nt,...,Np)
            self.Nb = len(self.filter_bank)
            X_train = self.fb_transform(X_train)

        self.bank_weights = bank_weights
        if self.version == 'SSVEP':
            self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25 for nb in range(self.Nb)])

        self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]
        for nse, se in enumerate(self.sub_estimator):
            se.fit(
                X_train=X_train[nse],
                y_train=y_train,
                source_model=source_model[nse],
                **kwargs
            )


# %% 4. Subject transfer based eTRCA, st-eTRCA
def sttrca_source_training(
        filter_bank_idx: int,
        data_path: List[str],
        chan_indices: List[int],
        time_range: Tuple[int, int],
        n_components: int = 1,
        with_filter_banks: bool = False) -> Dict[str, ndarray]:
    """The modeling process of st-(e)TRCA for source dataset.

    Args:
        filter_bank_idx (int): The index of filter banks to be used in model training process.
            Could be any value when 'with_filter_banks' is False.
            Setting to be the 1st parameter for parallely computing.
        data_path (List[str]): List of source data file paths.
        chan_indices (List[int]): Indices of selected channels.
        time_range (Tuple[int, int]): Start & end indices of time points.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.
        with_filter_banks (bool): Defaults to False, i.e. the shape of data is (Ne*Nte,Nc,Np).

    Returns:
        w_source (ndarray): (Ns,Ne,Nk,Nc). TRCA filters.
        wX_source (ndarray): (Ns,Ne,Nt,Nk,Np). TRCA-filtered data.
        ew_source (ndarray): (Ns,Ne,Ne*Nk,Nc). eTRCA filters.
        ewX_source (ndarray): (Ns,Ne,Nt,Ne*Nk,Np). eTRCA-filtered data.
    """
    # basic information
    n_subjects = len(data_path)  # Ns

    # main process
    w_list, ew_list, wX_list, ewX_list = [], [], [], []
    for nsub in range(n_subjects):
        # load in source data
        with open(data_path[nsub], 'rb') as file:
            eeg = pickle.load(file)
        X_source = eeg['X'][..., chan_indices, time_range[0]:time_range[1]]
        y_source = eeg['y']
        event_type = list(np.unique(y_source))
        del eeg, file

        if with_filter_banks:
            X_source = X_source[filter_bank_idx]

        # train TRCA model
        Q, S, X_mean = trca.generate_trca_mat(X=X_source, y=y_source)
        w, ew = trca.solve_trca_func(Q=Q, S=S, n_components=n_components)
        w_list.append(w)  # List[(Ne,Nk,Nc)]
        ew_list.append(ew)  # List[(Ne*Nk,Nc)]

        # construct spatial filtered data without trial-averaging
        n_points = X_mean.shape[-1]  # Np
        wX_source = np.zeros((X_source.shape[0], w.shape[1], n_points))
        ewX_source = np.zeros((X_source.shape[0], ew.shape[0], n_points))
        for ntr in range(X_source.shape[0]):
            X_temp = X_source[ntr]  # (Nc,Np)
            event_idx = event_type.index(y_source[ntr])
            wX_source[ntr] = w[event_idx] @ X_temp
            ewX_source[ntr] = ew @ X_temp
        wX_source = utils.reshape_dataset(
            data=wX_source,
            labels=y_source,
            target_style='public'
        )  # (Ne,Nt,Nk,Np)
        ewX_source = utils.reshape_dataset(
            data=ewX_source,
            labels=y_source,
            target_style='public'
        )  # (Ne,Nt,Ne*Nk,Np)
        wX_list.append(wX_source)  # List[(Ne,Nt,Nk,Np)]
        ewX_list.append(ewX_source)  # List[(Ne,Nt,Ne*Nk,Np)]
    return {
        'w_source': np.stack(w_list), 'wX_source': np.stack(wX_list),
        'ew_source': np.stack(ew_list), 'ewX_source': np.stack(ewX_list)
    }


def sttrca_feature(
        X_test: ndarray,
        trans_model: Dict[str, ndarray],
        target_model: Dict[str, Any],
        pattern_list: List[str] = ['1', '2'],
        standard: bool = True,
        ensemble: bool = True,
        fusion_method: str = 'DirSum') -> Dict[str, ndarray]:
    """The pattern matching process of st-(e)TRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        trans_model (Dict): {'wX_trans': ndarray (Ne,Nk,Np),
                             'ewX_trans': ndarray (Ne,Ne*Nk,Np)}
            See details in ST_TRCA.tranfer_learning().
        target_model (Dict): {'w': ndarray (Ne,Nk,Nc),
                              'wX': ndarray (Ne,Nk,Np),
                              'ewX': ndarray (Ne,Ne*Nk,Np)}
            See details in ST_TRCA.intra_target_training().
        pattern_list (List[str]): Different coefficient.
            '1': corr(w @ X_test, wX) or corr(ew @ X_test, ewX)
            '2': corr(w @ X_test, wX_trans) or corr(ew @ X_test, ewX_trans)
        fusion_method (str): 'DirSum' or 'SignSum'.
            'DirSum': rho = rho1 + rho2. Recommended for filter-bank scenario.
            'SignSum': rho = sign(rho1)*rho1^2 + sign(rho2)*rho2^2. Better for single-band.
    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features of st-TRCA.
        erho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features of st-eTRCA.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features of TRCA.
        erho (ndarray): (Ne*Nte,Ne). Intergrated features of eTRCA.
    """
    # load in models & basic information
    w, ew = target_model['w'], target_model['ew']  # (Ne,Nk,Nc), (Ne*Nk,Nc)
    n_test = X_test.shape[0]
    n_events = w.shape[0]  # Ne
    if standard:  # reshape & standardize templates for faster computing
        wX, wX_trans = target_model['wX'], trans_model['wX_trans']  # (Ne,Nk,Np)
        wX = utils.fast_stan_2d(np.reshape(wX, (n_events, -1), 'C'))  # (Ne,Nk*Np)
        wX_trans = utils.fast_stan_2d(np.reshape(wX_trans, (n_events, -1), 'C'))
    if ensemble:  # the same for ensemble scenario
        ewX, ewX_trans = target_model['ewX'], trans_model['ewX_trans']  # (Ne,Ne*Nk,Np)
        ewX = utils.fast_stan_2d(np.reshape(ewX, (n_events, -1), 'C'))  # (Ne,Ne*Nk*Np)
        ewX_trans = utils.fast_stan_2d(np.reshape(ewX_trans, (n_events, -1), 'C'))

    # pattern matching: 2-d features
    rho_temp = np.zeros((n_test, n_events, 2))
    erho_temp = np.zeros_like(rho_temp)
    if standard:
        for nte in range(n_test):
            X_temp = np.reshape(w @ X_test[nte], (n_events, -1), 'C')  # (Ne,Nk*Np)
            X_temp = utils.fast_stan_2d(X_temp)
            coef = X_temp.shape[-1]

            if '1' in pattern_list:  # corr(w @ X_test, wX_target)
                rho_temp[nte, :, 0] = utils.fast_corr_2d(X=X_temp, Y=wX) / coef
            if '2' in pattern_list:  # corr(w @ X_test, wX_trans)
                rho_temp[nte, :, 1] = utils.fast_corr_2d(X=X_temp, Y=wX_trans) / coef
    if ensemble:
        for nte in range(n_test):
            X_temp = np.tile(
                A=np.reshape(ew @ X_test[nte], -1, 'C'),
                reps=(n_events, 1)
            )  # (Ne*Nk,Np) -reshape-> (Ne*Nk*Np,) -repeat-> (Ne,Ne*Nk*Np)
            X_temp = utils.fast_stan_2d(X_temp)
            coef = X_temp.shape[-1]

            if '1' in pattern_list:  # corr(w @ X_test, wX_target)
                erho_temp[nte, :, 0] = utils.fast_corr_2d(X=X_temp, Y=ewX) / coef
            if '2' in pattern_list:  # corr(w @ X_test, wX_trans)
                erho_temp[nte, :, 1] = utils.fast_corr_2d(X=X_temp, Y=ewX_trans) / coef

    # integration of features
    if fusion_method == 'DirSum':
        rho, erho = np.zeros((n_test, n_events)), np.zeros((n_test, n_events))
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho += rho_temp[..., pl]
                erho += erho_temp[..., pl]
    elif fusion_method == 'SignSum':
        rho, erho = [], []
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho.append(rho_temp[..., pl])
                erho.append(erho_temp[..., pl])
        if len(rho) == 1:  # only 1 coefficient
            rho = rho[0]
            erho = erho[0]
        else:  # more than 1 coefficient
            rho = utils.combine_feature(rho)
            erho = utils.combine_feature(erho)
    return {
        'rho_temp': rho_temp, 'rho': rho,
        'erho_temp': erho_temp, 'erho': erho
    }


class ST_TRCA(transfer.BasicTransfer):
    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        self.target_model = trca.trca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            standard=self.standard,
            ensemble=self.ensemble,
            n_components=self.n_components
        )

    def weight_calc(self):
        """Optimize the transfer weights."""
        # basic information
        n_events = len(self.event_type)

        if self.standard:  # solve LST problem for wX
            self.bwX = np.reshape(
                a=self.target_model['wX'],
                newshape=(n_events, -1),
                order='C'
            )  # (Ne,Nk,Np) -reshape-> (Ne,Nk*Np)
            self.bwX = np.reshape(utils.fast_stan_2d(self.bwX), -1, 'C')  # (Ne*Nk*Np,)
            self.AwX = np.reshape(
                a=self.source_model['wX_source'].mean(axis=2),
                newshape=(self.n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Nk,Np) -reshape-> (Ns,Ne,Nk*Np)
            self.AwX = np.reshape(
                a=self.AwX,
                newshape=(self.n_subjects, -1),
                order='C'
            ).T  # (Ns,Ne,Nk*Np) -reshape-> (Ns,Ne*Nk*Np) -tranpose-> (Ne*Nk*Np,Ns)
            self.weight_wX, _, _, _ = sLA.lstsq(a=self.AwX, b=self.bwX)  # (Ns,)

        if self.ensemble:  # solve LST problem for ewX
            self.bewX = np.reshape(
                a=self.target_model['ewX'],
                newshape=(n_events, -1),
                order='C'
            )  # (Ne,Ne*Nk,Np) -reshape-> (Ne,Ne*Nk*Np)
            self.bewX = np.reshape(utils.fast_stan_2d(self.bewX), -1, 'C')  # (Ne*Ne*Nk*Np,)
            self.AewX = np.reshape(
                a=self.source_model['ewX_source'].mean(axis=2),
                newshape=(self.n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Ne*Nk,Np) -reshape-> (Ns,Ne,Ne*Nk*Np)
            self.AewX = np.reshape(
                a=self.AewX,
                newshape=(self.n_subjects, -1),
                order='C'
            ).T  # (Ns,Ne,Ne*Nk*Np) -reshape-> (Ns,Ne*Ne*Nk*Np) -tranpose-> (Ne*Ne*Nk*Np,Ns)
            self.weight_ewX, _, _, _ = sLA.lstsq(a=self.AewX, b=self.bewX)  # (Ns,)

    def transfer_learning(self):
        """Transfer learning process."""
        self.trans_model = {}
        if self.standard:
            wX_trans = np.einsum(
                's,sekp->ekp',
                self.weight_wX,
                self.source_model['wX_source'].mean(axis=2)
            )
            self.trans_model['wX_trans'] = wX_trans
        if self.ensemble:
            ewX_trans = np.einsum(
                's,sekp->ekp',
                self.weight_ewX,
                self.source_model['ewX_source'].mean(axis=2)
            )
            self.trans_model['ewX_trans'] = ewX_trans

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            pattern_list: List[str] = ['1', '2'],
            fusion_method: str = 'DirSum',
            source_model: Optional[Dict[str, ndarray]] = None):
        """Train st-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            pattern_list (List[str]): Different coefficient.
            fusion_method (str): 'DirSum' or 'SignSum'. See details in sttrca_feature().
            source_model (Dict): See details in stcirca_source_training().
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.event_type = np.unique(self.y_train)
        self.pattern_list = pattern_list
        self.fusion_method = fusion_method
        self.source_model = source_model
        if self.standard:
            self.n_subjects = self.source_model['wX_source'].shape[0]  # Ns
        elif self.ensemble:
            self.n_subjects = self.source_model['ewX_source'].shape[0]  # Ns

        # main process
        self.intra_target_training()
        self.weight_calc()
        self.transfer_learning()

    def transform(self, X_test) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features of st-TRCA.
            erho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features of st-eTRCA.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features of TRCA.
            erho (ndarray): (Ne*Nte,Ne). Intergrated features of eTRCA.
        """
        return sttrca_feature(
            X_test=X_test,
            trans_model=self.trans_model,
            target_model=self.target_model,
            pattern_list=self.pattern_list,
            standard=self.standard,
            ensemble=self.ensemble,
            fusion_method=self.fusion_method
        )

    def predict(self, X_test: ndarray) -> Union[Tuple[ndarray, ndarray],
                                                Tuple[int, int]]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            y_standard (ndarray or int): (Ne*Nte,). Predict label(s).
            y_ensemble (ndarray or int): (Ne*Nte,). Predict label(s) (ensemble).
        """
        self.features = self.transform(X_test)
        self.y_standard = self.event_type[np.argmax(self.features['rho'], axis=-1)]
        self.y_ensemble = self.event_type[np.argmax(self.features['erho'], axis=-1)]
        return self.y_standard, self.y_ensemble


class FB_ST_TRCA(trca.BasicFBTRCA):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            standard: bool = True,
            ensemble: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble
        super().__init__(
            base_estimator=ST_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components
            ),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            source_model: List[Dict[str, ndarray]],
            bank_weights: Optional[ndarray] = None,
            **kwargs):
        """Load in training dataset and pass it to sub-esimators.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np) or (Nb,Ne*Nt,Nc,Np) (with_filter_bank=True).
                Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            source_model (List[Dict[str, ndarray]]): See details in sttrca_source_training().
            bank_weights (ndarray, optional): Weights for different filter banks.
                Defaults to None (equal).
        """
        if self.with_filter_bank:  # X_train has been filterd: (Nb,Ne*Nt,...,Np)
            self.Nb = X_train.shape[0]
        else:  # tranform X_train into shape: (Nb,Ne*Nt,...,Np)
            self.Nb = len(self.filter_bank)
            X_train = self.fb_transform(X_train)

        self.bank_weights = bank_weights
        if self.version == 'SSVEP':
            self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25 for nb in range(self.Nb)])

        self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]
        for nse, se in enumerate(self.sub_estimator):
            se.fit(
                X_train=X_train[nse],
                y_train=y_train,
                source_model=source_model[nse],
                **kwargs
            )


# %% 5. Subject transfer based TDCA, st-TDCA
def sttdca_source_training(
        filter_bank_idx: int,
        data_path: List[str],
        chan_indices: List[int],
        time_range: Tuple[int, int],
        extra_length: int,
        srate: int = 250,
        n_harmonics: int = 5,
        n_components: int = 1,
        with_filter_banks: bool = False) -> Dict[str, ndarray]:
    """The modeling process of st-TDCA for source dataset.

    Args:
        filter_bank_idx (int): The index of filter banks to be used in model training process.
            Could be any value when 'with_filter_banks' is False.
            Setting to be the 1st parameter for parallely computing.
        data_path (List[str]): List of source data file paths.
        chan_indices (List[int]): Indices of selected channels.
        time_range (Tuple[int, int]): Start & end indices of time points.
        extra_length (int): Hyper-parameter for TDCA algorithm.
            See details in dsp.tdca_feature().
        srate (int): Sampling rate. Defaults to 250 Hz.
        n_harmonics (int): Nh. Defaults to 5.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.
        with_filter_banks (bool): Defaults to False, i.e. the shape of data is (Ne*Nte,Nc,Np).

    Returns:
        w_source (ndarray): (Ns,Nk,(m+1)*Nc). TDCA filters.
        wX_source (ndarray): (Ns,Ne,Nk,2*Np). TDCA templates.
    """
    # basic information
    n_subjects = len(data_path)  # Ns

    # main process
    w_source, wX_source = [], []
    for nsub in range(n_subjects):
        # load in source data
        with open(data_path[nsub], 'rb') as file:
            eeg = pickle.load(file)
        X_source = eeg['X'][..., chan_indices, time_range[0]:time_range[1] + extra_length]
        y_source = eeg['y']
        freq_list, phase_list = eeg['freqs'], eeg['phases']
        del eeg, file

        if with_filter_banks:
            X_source = X_source[filter_bank_idx]

        # basic information
        X_train = X_source[..., :-extra_length]
        X_extra = X_source[..., -extra_length:]
        event_type = np.unique(y_source)
        n_events = len(event_type)  # Ne
        n_points = X_train.shape[-1]  # Np

        # config projection matrices
        projection = np.zeros((n_events, n_points, n_points))  # (Ne,Np,Np)
        for ne in range(n_events):
            sine_templates = utils.sine_template(
                freq=freq_list[ne],
                phase=phase_list[ne],
                n_points=n_points,
                n_harmonics=n_harmonics,
                srate=srate
            )  # (2*Nh,Np)
            projection[ne] = utils.qr_projection(sine_templates.T)

        # create secondary augmented data | (Ne*Nt,(m+1)*Nc,2*Np)
        X_train_aug2 = np.tile(
            A=np.zeros_like(X_train),
            reps=(1, (extra_length + 1), 2)
        )
        for ntr, label in enumerate(y_source):
            event_idx = list(event_type).index(label)
            X_train_aug2[ntr] = dsp.tdca_augmentation(
                X=X_train[ntr],
                projection=projection[event_idx],
                extra_length=extra_length,
                X_extra=X_extra[ntr]
            )

        # train DSP models & templates
        model = dsp.dsp_kernel(
            X_train=X_train_aug2,
            y_train=y_source,
            n_components=n_components
        )
        w_source.append(model['w'])  # List[(Nk,Nc)]
        wX_source.append(model['wX'])  # List[(Ne,Nk,2*Np)]
    return {'w_source': np.stack(w_source), 'wX_source': np.stack(wX_source)}


def sttdca_feature(
        X_test: ndarray,
        trans_model: Dict[str, ndarray],
        target_model: Dict[str, Any],
        projection: ndarray,
        extra_length: int,
        pattern_list: List[str] = ['1', '2'],
        fusion_method: str = 'DirSum') -> Dict[str, ndarray]:
    """The pattern matching process of st-TDCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        trans_model (Dict): {'wX_trans': ndarray (Ne,Nk,Np)}
            See details in ST_TDCA.tranfer_learning().
        target_model (Dict): {'w': ndarray (Nk,Nc),
                              'wX': ndarray (Ne,Nk,Np)}
            See details in ST_TDCA.intra_target_training().
        projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
        extra_length (int): m.
        pattern_list (List[str]): Different coefficient.
            '1': corr(w @ X_test, wX_target)
            '2': corr(w @ X_test, wX_trans)
        fusion_method (str): 'DirSum' or 'SignSum'.
            'DirSum': rho = rho1 + rho2. Recommended for filter-bank scenario.
            'SignSum': rho = sign(rho1)*rho1^2 + sign(rho2)*rho2^2. Better for single-band.

    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features.
    """
    # load in models & basic information
    w = target_model['w']  # (Nk,Nc)
    wX, wX_trans = target_model['wX'], trans_model['wX_trans']  # (Ne,Nk,2*Np)
    n_test = X_test.shape[0]
    n_events = wX.shape[0]  # Ne

    # pattern matching: 2-d features
    rho_temp = np.zeros((n_test, n_events, 2))
    for nte in range(n_test):
        for nem in range(n_events):
            X_temp = w @ dsp.tdca_augmentation(
                X=X_test[nte],
                projection=projection[nem],
                extra_length=extra_length
            )  # (Nk,2*Np)
            if '1' in pattern_list:  # corr(w @ X_test, wX_target)
                rho_temp[nte, nem, 0] = utils.pearson_corr(X=X_temp, Y=wX[nem])
            if '2' in pattern_list:  # corr(w @ X_test, wX_trans)
                rho_temp[nte, nem, 1] = utils.pearson_corr(X=X_temp, Y=wX_trans[nem])

    # integration of features
    if fusion_method == 'DirSum':
        rho = np.zeros((n_test, n_events))
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho += rho_temp[..., pl]
    elif fusion_method == 'SignSum':
        rho = []
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho.append(rho_temp[..., pl])
        if len(rho) == 1:  # only 1 coefficient
            rho = rho[0]
        else:  # more than 1 coefficient
            rho = utils.combine_feature(rho)
    return {'rho_temp': rho_temp, 'rho': rho}


class ST_TDCA(transfer.BasicTransfer):
    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        # create secondary augmented data | (Ne*Nt,(m+1)*Nc,2*Np)
        self.X_train_aug2 = np.tile(
            A=np.zeros_like(self.X_train),
            reps=(1, (self.extra_length + 1), 2)
        )
        for ntr, label in enumerate(self.y_train):
            event_idx = list(self.event_type).index(label)
            self.X_train_aug2[ntr] = dsp.tdca_augmentation(
                X=self.X_train[ntr],
                projection=self.projection[event_idx],
                extra_length=self.extra_length,
                X_extra=self.X_extra[ntr]
            )

        # train DSP models & templates
        self.target_model = dsp.dsp_kernel(
            X_train=self.X_train_aug2,
            y_train=self.y_train,
            n_components=self.n_components
        )

    def weight_calc(self):
        """Optimize the transfer weights."""
        # basic information
        n_events = len(self.event_type)

        # solve LST problem for wX
        self.bwX = np.reshape(
            a=self.target_model['wX'],
            newshape=(n_events, -1),
            order='C'
        )  # (Ne,Nk,2*Np) -reshape-> (Ne,Nk*2*Np)
        self.bwX = np.reshape(utils.fast_stan_2d(self.bwX), -1, 'C')  # (Ne*Nk*2*Np,)
        self.AwX = np.reshape(
            a=self.source_model['wX_source'],
            newshape=(self.n_subjects, n_events, -1),
            order='C'
        )  # (Ns,Ne,Nk,2*Np) -reshape-> (Ns,Ne,Nk*2*Np)
        self.AwX = np.reshape(
            a=self.AwX,
            newshape=(self.n_subjects, -1),
            order='C'
        ).T  # (Ns,Ne,Nk*2*Np) -reshape-> (Ns,Ne*Nk*2*Np) -tranpose-> (Ne*Nk*2*Np,Ns)
        self.weight_wX, _, _, _ = sLA.lstsq(a=self.AwX, b=self.bwX)  # (Ns,)

    def transfer_learning(self):
        """Transfer learning process."""
        wX_trans = np.einsum('s,sekp->ekp', self.weight_wX, self.source_model['wX_source'])
        self.trans_model = {'wX_trans': wX_trans}

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            extra_length: int,
            projection: ndarray,
            pattern_list: List[str] = ['1', '2'],
            fusion_method: str = 'DirSum',
            source_model: Optional[Dict[str, ndarray]] = None):
        """Train st-TDCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np+m). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            extra_length (int): m.
            projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
            pattern_list (List[str]): Different coefficient.
            fusion_method (str): 'DirSum' or 'SignSum'. See details in sttdca_feature().
            source_model (Dict): See details in stcirca_source_training().
        """
        # load in data
        self.X_train = X_train[..., :-extra_length]  # (Ne*Nt,Nc,Np)
        self.X_extra = X_train[..., -extra_length:]  # (Ne*Nt,Nc,m)
        self.extra_length = extra_length  # m
        self.y_train = y_train
        self.event_type = np.unique(y_train)
        self.projection = projection
        self.pattern_list = pattern_list
        self.fusion_method = fusion_method
        self.source_model = source_model
        self.n_subjects = self.source_model['wX_source'].shape[0]  # Ns

        # main process
        self.intra_target_training()
        self.weight_calc()
        self.transfer_learning()

    def transform(self, X_test: ndarray) -> Dict[str, ndarray]:
        """Transform test dataset to discriminant features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Return:
            rho (ndarray): (Ne*Nte,Ne). Decision coefficients of DSP.
        """
        return sttdca_feature(
            X_test=X_test,
            trans_model=self.trans_model,
            target_model=self.target_model,
            projection=self.projection,
            extra_length=self.extra_length,
            pattern_list=self.pattern_list,
            fusion_method=self.fusion_method
        )


class FB_ST_TDCA(dsp.BasicFBDSP):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        self.n_components = n_components
        super().__init__(
            base_estimator=ST_TDCA(n_components=self.n_components),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            extra_length: int,
            source_model: List[Dict[str, ndarray]],
            bank_weights: Optional[ndarray] = None,
            **kwargs):
        """Load in training dataset and pass it to sub-esimators.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np+m) or (Nb,Ne*Nt,Nc,Np+m) (with_filter_bank=True).
                Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            extra_length (int): m.
            source_model (List[Dict[str, ndarray]]): See details in sttrca_source_training().
            bank_weights (ndarray, optional): Weights for different filter banks.
                Defaults to None (equal).
        """
        if self.with_filter_bank:  # X_train has been filterd: (Nb,Ne*Nt,...,Np)
            self.Nb = X_train.shape[0]
        else:  # tranform X_train into shape: (Nb,Ne*Nt,...,Np)
            self.Nb = len(self.filter_bank)
            X_train = self.fb_transform(X_train)

        self.bank_weights = bank_weights
        if self.version == 'SSVEP':
            self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25 for nb in range(self.Nb)])

        self.sub_estimator = [clone(self.base_estimator) for nb in range(self.Nb)]
        for nse, se in enumerate(self.sub_estimator):
            se.fit(
                X_train=X_train[nse],
                y_train=y_train,
                extra_length=extra_length,
                source_model=source_model[nse],
                **kwargs
            )


# %% 6. Subject transfer based sc-eTRCA, st-sc-eTRCA
def stsctrca_source_training(
        filter_bank_idx: int,
        data_path: List[str],
        chan_indices: List[int],
        time_range: Tuple[int, int],
        sine_template: ndarray,
        n_components: int = 1,
        with_filter_banks: bool = False) -> Dict[str, ndarray]:
    """The modeling process of st-sc-(e)TRCA for source dataset.

    Args:
        filter_bank_idx (int): The index of filter banks to be used in model training process.
            Could be any value when 'with_filter_banks' is False.
            Setting to be the 1st parameter for parallely computing.
        data_path (List[str]): List of source data file paths.
        chan_indices (List[int]): Indices of selected channels.
        time_range (Tuple[int, int]): Start & end indices of time points.
        sine_template (ndarray): (Ne,2*Nh,Np). Artificial templates.
        n_components (int): Number of eigenvectors picked as filters.
            Defaults to 1.
        with_filter_banks (bool): Defaults to False, i.e. the shape of data is (Ne*Nte,Nc,Np).

    Returns:
        u_source (ndarray): (Ns,Ne,Nk,Nc). sc-TRCA filters (EEG).
        uX_source (ndarray): (Ns,Ne,Nk,Np). sc-TRCA templates (EEG).
        v_source (ndarray): (Ns,Ne,Nk,m). sc-TRCA filters (templates).
        vY_source (ndarray): (Ns,Ne,Nk,Np). sc-TRCA templates (templates).
        eu_source (ndarray): (Ns,Ne*Nk,Nc). sc-eTRCA filters (EEG).
        euX_source (ndarray): (Ns,Ne,Ne*Nk,Np). sc-eTRCA templates (EEG).
        ev_source (ndarray): (Ns,Ne*Nk,m). sc-eTRCA filters (templates).
        evY_source (ndarray): (Ns,Ne,Ne*Nk,Np). sc-eTRCA templates (templates).
    """
    # basic information
    n_subjects = len(data_path)  # Ns

    # main process
    u_source, uX_source, v_source, vY_source = [], [], [], []
    eu_source, euX_source, ev_source, evY_source = [], [], [], []
    for nsub in range(n_subjects):
        # load in source data
        with open(data_path[nsub], 'rb') as file:
            eeg = pickle.load(file)
        X_source = eeg['X'][..., chan_indices, time_range[0]:time_range[1]]
        y_source = eeg['y']
        del eeg, file

        if with_filter_banks:
            X_source = X_source[filter_bank_idx]

        # train CIRCA model
        model = trca.sctrca_kernel(
            X_train=X_source,
            y_train=y_source,
            sine_template=sine_template,
            standard=True,
            ensemble=True,
            n_components=n_components
        )
        u_source.append(model['u'])  # List[(Ne,Nk,Nc)]
        uX_source.append(model['uX'])  # List[(Ne,Nk,Np)]
        v_source.append(model['v'])  # List[(Ne,Nk,m)]
        vY_source.append(model['vY'])  # List[(Ne,Nk,Np)]
        eu_source.append(model['eu'])  # List[(Ne*Nk,Nc)]
        euX_source.append(model['euX'])  # List[(Ne,Ne*Nk,Np)]
        ev_source.append(model['ev'])  # List[(Ne*Nk,m)]
        evY_source.append(model['evY'])  # List[(Ne,Ne*Nk,Np)]
    return {
        'u_source': np.stack(u_source), 'uX_source': np.stack(uX_source),
        'v_source': np.stack(v_source), 'vY_source': np.stack(vY_source),
        'eu_source': np.stack(eu_source), 'euX_source': np.stack(euX_source),
        'ev_source': np.stack(ev_source), 'evY_source': np.stack(evY_source),
    }


def stsctrca_feature(
        X_test: ndarray,
        trans_model: Dict[str, ndarray],
        target_model: Dict[str, Any],
        pattern_list: List[str] = ['1', '2', '3', '4'],
        standard: bool = True,
        ensemble: bool = True,
        fusion_method: str = 'DirSum') -> Dict[str, ndarray]:
    """The pattern matching process of st-sc-(e)TRCA.

    Args:
        X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.
        trans_model (Dict): {'uX_trans': ndarray (Ne,Nk,Np),
                             'euX_trans': ndarray (Ne,Ne*Nk,Np),
                             'vY_trans': ndarray (Ne,Nk,Np),
                             'evY_trans': ndarray (Ne,Ne*Nk,Np)}
            See details in ST_SC_TRCA.tranfer_learning().
        target_model (Dict): {'u': ndarray (Ne,Nk,Nc),
                              'uX': ndarray (Ne,Nk,Np),
                              'vY': ndarray (Ne,Nk,Np),
                              'eu': ndarray (Ne*Nk,Nc),
                              'euX': ndarray (Ne,Ne*Nk,Np),
                              'evY': ndarray (Ne,Ne*Nk,Np)}
            See details in ST_SC_TRCA.intra_target_training().
        pattern_list (List[str]): Different coefficient.
            '1': corr(u @ X_test, uX) or corr(eu @ X_test, euX)
            '2': corr(u @ X_test, vY) or corr(eu @ X_test, evY)
            '3': corr(u @ X_test, uX_trans) or corr(eu @ X_test, euX_trans)
            '4': corr(u @ X_test, vY_trans) or corr(eu @ X_test, evY_trans)
        fusion_method (str): 'DirSum' or 'SignSum'.
            'DirSum': rho = sum(rho(i)). Recommended for filter-bank scenario.
            'SignSum': rho = sum(sign(rho(i))*rho(i)^2). Better for single-band.
    Returns:
        rho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features of st-TRCA.
        erho_temp (ndarray): (Ne*Nte,Ne,2). 2-D features of st-eTRCA.
        rho (ndarray): (Ne*Nte,Ne). Intergrated features of TRCA.
        erho (ndarray): (Ne*Nte,Ne). Intergrated features of eTRCA.
    """
    # load in models & basic information
    u, eu = target_model['u'], target_model['eu']  # (Ne,Nk,Nc), (Ne*Nk,Nc)
    n_test = X_test.shape[0]
    n_events = u.shape[0]  # Ne
    if standard:  # reshape & standardize templates for faster computing
        uX, uX_trans = target_model['uX'], trans_model['uX_trans']  # (Ne,Nk,Np)
        vY, vY_trans = target_model['vY'], trans_model['vY_trans']  # (Ne,Nk,Np)
        uX = utils.fast_stan_2d(np.reshape(uX, (n_events, -1), 'C'))  # (Ne,Nk*Np)
        uX_trans = utils.fast_stan_2d(np.reshape(uX_trans, (n_events, -1), 'C'))
        vY = utils.fast_stan_2d(np.reshape(vY, (n_events, -1), 'C'))  # (Ne,Nk,Np)
        vY_trans = utils.fast_stan_2d(np.reshape(vY_trans, (n_events, -1), 'C'))
    if ensemble:  # the same for ensemble scenario
        euX, euX_trans = target_model['euX'], trans_model['euX_trans']  # (Ne,Ne*Nk,Np)
        evY, evY_trans = target_model['evY'], trans_model['evY_trans']  # (Ne,Ne*Nk,Np)
        euX = utils.fast_stan_2d(np.reshape(euX, (n_events, -1), 'C'))  # (Ne,Ne*Nk*Np)
        euX_trans = utils.fast_stan_2d(np.reshape(euX_trans, (n_events, -1), 'C'))
        evY = utils.fast_stan_2d(np.reshape(evY, (n_events, -1), 'C'))  # (Ne,Ne*Nk*Np)
        evY_trans = utils.fast_stan_2d(np.reshape(evY_trans, (n_events, -1), 'C'))

    # pattern matching: 4-d features
    rho_temp = np.zeros((n_test, n_events, 4))
    erho_temp = np.zeros_like((rho_temp))
    if standard:
        for nte in range(n_test):
            X_temp = np.reshape(u @ X_test[nte], (n_events, -1), 'C')  # (Ne,Nk*Np)
            X_temp = utils.fast_stan_2d(X_temp)
            coef = X_temp.shape[-1]

            if '1' in pattern_list:  # corr(u @ X_test, uX)
                rho_temp[nte, :, 0] = utils.fast_corr_2d(X=X_temp, Y=uX) / coef
            if '2' in pattern_list:  # corr(u @ X_test, uX_trans)
                rho_temp[nte, :, 1] = utils.fast_corr_2d(X=X_temp, Y=vY) / coef
            if '3' in pattern_list:  # corr(u @ X_test, uX_trans)
                rho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp, Y=uX_trans) / coef
            if '4' in pattern_list:  # corr(u @ X_test, vY_trans)
                rho_temp[nte, :, 3] = utils.fast_corr_2d(X=X_temp, Y=vY_trans) / coef
    if ensemble:
        for nte in range(n_test):
            X_temp = np.tile(
                A=np.reshape(eu @ X_test[nte], -1, 'C'),
                reps=(n_events, 1)
            )  # (Ne*Nk,Np) -reshape-> (Ne*Nk*Np,) -repeat-> (Ne,Ne*Nk*Np)
            X_temp = utils.fast_stan_2d(X_temp)
            coef = X_temp.shape[-1]

            if '1' in pattern_list:  # corr(eu @ X_test, euX)
                erho_temp[nte, :, 0] = utils.fast_corr_2d(X=X_temp, Y=euX) / coef
            if '2' in pattern_list:  # corr(eu @ X_test, euX_trans)
                erho_temp[nte, :, 1] = utils.fast_corr_2d(X=X_temp, Y=evY) / coef
            if '3' in pattern_list:  # corr(eu @ X_test, euX_trans)
                erho_temp[nte, :, 2] = utils.fast_corr_2d(X=X_temp, Y=euX_trans) / coef
            if '4' in pattern_list:  # corr(eu @ X_test, evY_trans)
                erho_temp[nte, :, 3] = utils.fast_corr_2d(X=X_temp, Y=evY_trans) / coef

    # integration of features
    if fusion_method == 'DirSum':
        rho, erho = np.zeros((n_test, n_events)), np.zeros((n_test, n_events))
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho += rho_temp[..., pl]
                erho += erho_temp[..., pl]
    elif fusion_method == 'SignSum':
        rho, erho = [], []
        for pl in range(rho_temp.shape[-1]):
            if str(pl + 1) in pattern_list:
                rho.append(rho_temp[..., pl])
                erho.append(erho_temp[..., pl])
        if len(rho) == 1:  # only 1 coefficient
            rho = rho[0]
            erho = erho[0]
        else:  # more than 1 coefficient
            rho = utils.combine_feature(rho)
            erho = utils.combine_feature(erho)
    return {
        'rho_temp': rho_temp, 'rho': rho,
        'erho_temp': erho_temp, 'erho': erho
    }


class ST_SC_TRCA(transfer.BasicTransfer):
    def intra_target_training(self):
        """Intra-domain model training for target dataset."""
        self.target_model = trca.sctrca_kernel(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            standard=self.standard,
            ensemble=self.ensemble,
            n_components=self.n_components
        )

    def weight_calc(self):
        """Optimize the transfer weights."""
        # basic information
        n_events = len(self.event_type)

        # analytical solution
        if self.standard:
            # solve LST problem for uX
            self.buX = np.reshape(
                a=self.target_model['uX'],
                newshape=(n_events, -1),
                order='C'
            )  # (Ne,Nk,Np) -reshape-> (Ne,Nk*Np)
            self.buX = np.reshape(utils.fast_stan_2d(self.buX), -1, 'C')  # (Ne*Nk*Np,)
            self.AuX = np.reshape(
                a=self.source_model['uX_source'],
                newshape=(self.n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Nk,Np) -reshape-> (Ns,Ne,Nk*Np)
            self.AuX = np.reshape(
                a=self.AuX,
                newshape=(self.n_subjects, -1),
                order='C'
            ).T  # (Ns,Ne,Nk*Np) -reshape-> (Ns,Ne*Nk*Np) -tranpose-> (Ne*Nk*Np,Ns)
            self.weight_uX, _, _, _ = sLA.lstsq(a=self.AuX, b=self.buX)  # (Ns,)

            # solve LST problem for vY
            self.bvY = np.reshape(
                a=self.target_model['vY'],
                newshape=(n_events, -1),
                order='C'
            )  # (Ne,Nk,Np) -reshape-> (Ne,Nk*Np)
            self.bvY = np.reshape(utils.fast_stan_2d(self.bvY), -1, 'C')  # (Ne*Nk*Np,)
            self.AvY = np.reshape(
                a=self.source_model['vY_source'],
                newshape=(self.n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Nk,Np) -reshape-> (Ns,Ne,Nk*Np)
            self.AvY = np.reshape(
                a=self.AvY,
                newshape=(self.n_subjects, -1),
                order='C'
            ).T  # (Ns,Ne,Nk*Np) -reshape-> (Ns,Ne*Nk*Np) -tranpose-> (Ne*Nk*Np,Ns)
            self.weight_vY, _, _, _ = sLA.lstsq(a=self.AvY, b=self.bvY)  # (Ns,)
        if self.ensemble:  # solve LST problem for ewX
            # solve LST problem for euX
            self.beuX = np.reshape(
                a=self.target_model['euX'],
                newshape=(n_events, -1),
                order='C'
            )  # (Ne,Ne*Nk,Np) -reshape-> (Ne,Ne*Nk*Np)
            self.beuX = np.reshape(utils.fast_stan_2d(self.beuX), -1, 'C')  # (Ne*Ne*Nk*Np,)
            self.AeuX = np.reshape(
                a=self.source_model['euX_source'],
                newshape=(self.n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Ne*Nk,Np) -reshape-> (Ns,Ne,Ne*Nk*Np)
            self.AeuX = np.reshape(
                a=self.AeuX,
                newshape=(self.n_subjects, -1),
                order='C'
            ).T  # (Ns,Ne,Ne*Nk*Np) -reshape-> (Ns,Ne*Ne*Nk*Np) -tranpose-> (Ne*Ne*Nk*Np,Ns)
            self.weight_euX, _, _, _ = sLA.lstsq(a=self.AeuX, b=self.beuX)  # (Ns,)

            # solve LST problem for evY
            self.bevY = np.reshape(
                a=self.target_model['evY'],
                newshape=(n_events, -1),
                order='C'
            )  # (Ne,Ne*Nk,Np) -reshape-> (Ne,Ne*Nk*Np)
            self.bevY = np.reshape(utils.fast_stan_2d(self.bevY), -1, 'C')  # (Ne*Ne*Nk*Np,)
            self.AevY = np.reshape(
                a=self.source_model['evY_source'],
                newshape=(self.n_subjects, n_events, -1),
                order='C'
            )  # (Ns,Ne,Ne*Nk,Np) -reshape-> (Ns,Ne,Ne*Nk*Np)
            self.AevY = np.reshape(
                a=self.AevY,
                newshape=(self.n_subjects, -1),
                order='C'
            ).T  # (Ns,Ne,Ne*Nk*Np) -reshape-> (Ns,Ne*Ne*Nk*Np) -tranpose-> (Ne*Ne*Nk*Np,Ns)
            self.weight_evY, _, _, _ = sLA.lstsq(a=self.AevY, b=self.bevY)  # (Ns,)

    def transfer_learning(self):
        """Transfer learning process."""
        self.trans_model = {}
        if self.standard:
            self.trans_model['uX_trans'] = np.einsum(
                's,sekp->ekp',
                self.weight_uX,
                self.source_model['uX_source']
            )  # (Ne,Nk,Np)
            self.trans_model['vY_trans'] = np.einsum(
                's,sekp->ekp',
                self.weight_vY,
                self.source_model['vY_source']
            )  # (Ne,Nk,Np)
        if self.ensemble:
            self.trans_model['euX_trans'] = np.einsum(
                's,sekp->ekp',
                self.weight_euX,
                self.source_model['euX_source']
            )
            self.trans_model['evY_trans'] = np.einsum(
                's,sekp->ekp',
                self.weight_evY,
                self.source_model['evY_source']
            )  # (Ne,Nk,Np)

    def fit(
            self,
            X_train: ndarray,
            y_train: ndarray,
            sine_template: ndarray,
            pattern_list: List[str] = ['1', '2'],
            fusion_method: str = 'DirSum',
            source_model: Optional[Dict[str, ndarray]] = None):
        """Train st-sc-(e)TRCA model.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Sklearn-style training dataset.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal templates.
            pattern_list (List[str]): Different coefficient.
            fusion_method (str): 'DirSum' or 'SignSum'. See details in sttrca_feature().
            source_model (Dict): See details in stcirca_source_training().
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.event_type = np.unique(self.y_train)
        self.sine_template = sine_template
        self.pattern_list = pattern_list
        self.fusion_method = fusion_method
        self.source_model = source_model
        if self.standard:
            self.n_subjects = self.source_model['uX_source'].shape[0]  # Ns
        elif self.ensemble:
            self.n_subjects = self.source_model['euX_source'].shape[0]  # Ns

        # main process
        self.intra_target_training()
        self.weight_calc()
        self.transfer_learning()

    def transform(self, X_test) -> Dict[str, ndarray]:
        """Transform test dataset to features.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features of st-sc-TRCA.
            erho_temp (ndarray): (Ne*Nte,Ne,4). 4-D features of st-sc-eTRCA.
            rho (ndarray): (Ne*Nte,Ne). Intergrated features of st-sc-TRCA.
            erho (ndarray): (Ne*Nte,Ne). Intergrated features of st-sc-eTRCA.
        """
        return stsctrca_feature(
            X_test=X_test,
            trans_model=self.trans_model,
            target_model=self.target_model,
            pattern_list=self.pattern_list,
            standard=self.standard,
            ensemble=self.ensemble,
            fusion_method=self.fusion_method
        )

    def predict(self, X_test: ndarray) -> Union[Tuple[ndarray, ndarray],
                                                Tuple[int, int]]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            y_standard (ndarray or int): (Ne*Nte,). Predict label(s).
            y_ensemble (ndarray or int): (Ne*Nte,). Predict label(s) (ensemble).
        """
        self.features = self.transform(X_test)
        self.y_standard = self.event_type[np.argmax(self.features['rho'], axis=-1)]
        self.y_ensemble = self.event_type[np.argmax(self.features['erho'], axis=-1)]
        return self.y_standard, self.y_ensemble


class FB_ST_SC_TRCA(FB_ST_TRCA):
    def __init__(
            self,
            filter_bank: Optional[List] = None,
            with_filter_bank: bool = True,
            standard: bool = True,
            ensemble: bool = True,
            n_components: int = 1):
        """Basic configuration.

        Args:
            filter_bank (List[ndarray], optional):
                See details in utils.generate_filter_bank(). Defaults to None.
            with_filter_bank (bool): Whether the input data has been FB-preprocessed.
                Defaults to True.
            standard (bool): Standard model. Defaults to True.
            ensemble (bool): Ensemble model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Defaults to 1.
        """
        self.n_components = n_components
        self.standard = standard
        self.ensemble = ensemble
        super().__init__(
            base_estimator=ST_SC_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components
            ),
            filter_bank=filter_bank,
            with_filter_bank=with_filter_bank,
            version='SSVEP'
        )


# %% Accuracy-based subject selection
class FB_CIRCA_ASS(transfer.BasicASS):
    def __init__(
            self,
            source_model: List[Dict[str, ndarray]],
            X_target: ndarray,
            y_target: ndarray):
        """Basic configuration.

        Args:
            source_model (List[Dict[str, ndarray]]): See details in stcirca_source_training().
            y_source (List[ndarray]): List[(Ne*Nt,)]. Labels for X_source.
            X_target (ndarray): (Nb,Ne*Nt,Nc,Np). Target dataset with filter banks.
            y_target (ndarray): (Ne*Nt,). Labels for X_target.
        """
        # load in model
        self.X_target = X_target
        self.y_target = y_target
        self.source_model = source_model

        # basic information
        self.n_bands = self.X_target.shape[0]  # Nb
        self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25 for nb in range(self.n_bands)])
        self.event_type = np.unique(self.y_target)

    def evaluation_1st(self):
        """Calculate the cross-subject classification accuracy for each source subject."""
        # load in source model
        self.n_subjects = self.source_model[0]['w_source'].shape[0]  # Ns

        # apply FB-CIRCA classification
        self.acc_list = np.zeros((self.n_subjects))
        self.rho_list = []
        for nsub in range(self.n_subjects):
            rho_fb = []

            # pattern matching with pre-trained models
            for nb in range(self.n_bands):
                circa_model = {
                    'w': self.source_model[nb]['w_source'][nsub],
                    'wX': self.source_model[nb]['wX_source'][nsub],
                    'rH': self.source_model[nb]['rH_source'][nsub],
                }
                rho_fb.append(circa_feature(
                    X_test=self.X_target[nb],
                    circa_model=circa_model,
                    pattern_list=['1', '2'],
                    fusion_method='DirSum'
                )['rho'])  # List[(Ne*Nte,Ne)]
            rho_fb = np.stack(rho_fb)  # (Nb,Ne*Nte,Ne)
            rho = np.einsum('b,bte->te', self.bank_weights, utils.sign_sta(rho_fb))
            y_pred = self.event_type[np.argmax(rho, axis=-1)]
            self.acc_list[nsub] = utils.acc_compute(y_true=self.y_target, y_pred=y_pred)
            self.rho_list.append(rho_fb)  # List[(Nb,Ne*Nte,Ne)]

    def evaluation_2nd(self):
        """Find the best group of source subjects."""
        # forward iteration
        self.iter_list = np.zeros((self.n_subjects))
        descend_rho_list = [self.rho_list[si] for si in self.sorted_idx]
        for nsub in range(self.n_subjects):
            rho_fb = np.mean(np.stack(descend_rho_list[:nsub + 1]), axis=0)  # (Nb,Ne*Nte,Ne)
            rho = np.einsum('b,bte->te', self.bank_weights, utils.sign_sta(rho_fb))
            y_pred = self.event_type[np.argmax(rho, axis=-1)]
            self.iter_list[nsub] = utils.acc_compute(y_true=self.y_target, y_pred=y_pred)

        # select best subjects group
        self.max_idx = np.argmax(self.iter_list)

    def select_subjects(self) -> List[int]:
        """Main process.

        Returns:
            subject_indices (List[int]).
        """
        self.evaluation_1st()
        self.sort_subject_list()
        self.evaluation_2nd()
        return self.sorted_idx[:self.max_idx + 1]


class FB_TRCA_ASS(FB_CIRCA_ASS):
    def evaluation_1st(self):
        """Calculate the cross-subject classification accuracy for each source subject."""
        # load in source model
        self.n_subjects = self.source_model[0]['w'].shape[0]  # Ns

        # apply FB-CIRCA classification
        self.acc_list = np.zeros((self.n_subjects))
        self.rho_list = []
        for nsub in range(self.n_subjects):
            rho_fb = []

            # pattern matching with pre-trained models
            for nb in range(self.n_bands):
                etrca_model = {
                    'ew': self.source_model[nb]['ew_source'][nsub],
                    'ewX': self.source_model[nb]['ewX_source'][nsub]
                }
                rho_fb.append(trca.trca_feature(
                    X_test=self.X_target[nb],
                    trca_model=etrca_model,
                    standard=False,
                    ensemble=True
                )['erho'])  # List[(Ne*Nte,Ne)]
            rho_fb = np.stack(rho_fb)  # (Nb,Ne*Nte,Ne)
            rho = np.einsum('b,bte->te', self.bank_weights, utils.sign_sta(rho_fb))
            y_pred = self.event_type[np.argmax(rho, axis=-1)]
            self.acc_list[nsub] = utils.acc_compute(y_true=self.y_target, y_pred=y_pred)
            self.rho_list.append(rho_fb)  # List[(Nb,Ne*Nte,Ne)]


class FB_TDCA_ASS(FB_CIRCA_ASS):
    def __init__(
            self,
            source_model: List[Dict[str, ndarray]],
            X_target: ndarray,
            y_target: ndarray,
            extra_length: int,
            projection: ndarray):
        """Basic configuration.

        Args:
            source_model (List[Dict[str, ndarray]]): See details in sttrca_source_training().
            y_source (List[ndarray]): List[(Ne*Nt,)]. Labels for X_source.
            X_target (ndarray): (Nb,Ne*Nt,Nc,Np+m). Target dataset with filter banks.
            y_target (ndarray): (Ne*Nt,). Labels for X_target.extra_length (int): m.
            projection (ndarray): (Ne,Np,Np). Orthogonal projection matrices.
        """
        # load in model
        self.X_target = X_target[..., :-extra_length]  # (Ne*Nt,Nc,Np)
        self.X_extra = X_target[..., -extra_length:]  # (Ne*Nt,Nc,m)
        self.extra_length = extra_length
        self.y_target = y_target
        self.projection = projection
        self.source_model = source_model

        # basic information
        self.n_bands = self.X_target.shape[0]  # Nb
        self.bank_weights = np.array([(nb + 1)**(-1.25) + 0.25 for nb in range(self.n_bands)])
        self.event_type = np.unique(self.y_target)

    def evaluation_1st(self):
        """Calculate the cross-subject classification accuracy for each source subject."""
        # load in source model
        self.n_subjects = self.source_model[0]['w'].shape[0]  # Ns

        # apply FB-CIRCA classification
        self.acc_list = np.zeros((self.n_subjects))
        self.rho_list = []
        for nsub in range(self.n_subjects):
            rho_fb = []

            # pattern matching with pre-trained models
            for nb in range(self.n_bands):
                tdca_model = {
                    'w': self.source_model[nb]['w_source'][nsub],
                    'wX': self.source_model[nb]['wX_source'][nsub]
                }
                rho_fb.append(dsp.tdca_feature(
                    X_test=self.X_target[nb],
                    tdca_model=tdca_model,
                    projection=self.projection,
                    extra_length=self.extra_length
                )['rho'])  # List[(Ne*Nte,Ne)]
            rho_fb = np.stack(rho_fb)  # (Nb,Ne*Nte,Ne)
            rho = np.einsum('b,bte->te', self.bank_weights, utils.sign_sta(rho_fb))
            y_pred = self.event_type[np.argmax(rho, axis=-1)]
            self.acc_list[nsub] = utils.acc_compute(y_true=self.y_target, y_pred=y_pred)
            self.rho_list.append(rho_fb)  # List[(Nb,Ne*Nte,Ne)]
