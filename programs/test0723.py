# %% basic modules
from abc import abstractmethod, ABCMeta

import utils
import cca
from cca import BasicCCA, BasicFBCCA, cca_compute, msecca_compute
import trca
# from trca import BasicTRCA, BasicFBTRCA

from transfer import BasicTransfer

from typing import Optional, List, Tuple, Any
from numpy import ndarray

import numpy as np
import scipy.linalg as sLA

# %%
class BasicTransfer(metaclass=ABCMeta):
    def __init__(self,
        standard: Optional[bool] = True,
        ensemble: Optional[bool] = True,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None):
        """Basic configuration.

        Args:
            standard (bool, optional): Standard TRCA model. Defaults to True.
            ensemble (bool, optional): Ensemble TRCA model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None' when n_components is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio
        self.standard = standard
        self.ensemble = ensemble


    @abstractmethod
    def source_intra_training(self,):
        """Intra-subject model training for source dataset."""
        pass


    @abstractmethod
    def transfer_learning(self,):
        """Transfer learning for source datasets."""
        pass


    @abstractmethod
    def data_augmentation(self,):
        """Data augmentation for target dataset."""
        pass


    @abstractmethod
    def dist_calculation(self,):
        """Calculate spatial distance of target & source datasets."""
        pass


    @abstractmethod
    def weight_optimization(self,):
        """Optimize the transfer weight for each source subject."""
        pass


    @abstractmethod
    def target_intra_training(self,):
        """Intra-subject model training for target dataset."""
        pass    


    @abstractmethod
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        stim_info: Optional[dict] = None,
        sine_template: Optional[ndarray] = None,
        ):
        """Load data and train classification models.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset (target domain). Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            stim_info (Optional[dict]): Information of stimulus.
                {'event_type':ndarray, (Ne,),
                 'freqs':List or ndarray, (Ne,),
                 'phases':List or ndarray, (Ne,), etc}
            sine_template (Optional[ndarray]): (Ne,2*Nh,Np). Sinusoidal template.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.stim_info = stim_info
        self.sine_template = sine_template

        # main process
        self.transfer_learning()
        self.data_augmentation()
        self.target_intra_training()
        self.dist_calculation()
        self.weight_optimization()
        return self


    @abstractmethod
    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Predict test data.

        Args:
            X_test (ndarray): (Ne*Nte,...,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            y_standard (ndarray): (Ne*Nte,). Predict labels.
            erou (ndarray): (Ne*Nte,Ne). Decision coefficients (ensemble).
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels (ensemble).
        """
        pass


class BasicFBTransfer(metaclass=ABCMeta):
    def __init__(self,
        standard: Optional[bool] = True,
        ensemble: Optional[bool] = True,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None):
        """Basic configuration.

        Args:
            standard (bool, optional): Standard TRCA model. Defaults to True.
            ensemble (bool, optional): Ensemble TRCA model. Defaults to True.
            n_components (int): Number of eigenvectors picked as filters.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None' when n_components is not 'None'.
        """
        # config model
        self.n_components = n_components
        self.ratio = ratio
        self.standard = standard
        self.ensemble = ensemble


    @abstractmethod
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        stim_info: Optional[dict] = None,
        sine_template: Optional[ndarray] = None,
        ):
        """Load data and train classification models.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training target dataset. Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Nb,Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            stim_info (Optional[dict]): Information of stimulus.
                {'event_type':ndarray, (Ne,),
                 'freqs':List or ndarray, (Ne,),
                 'phases':List or ndarray, (Ne,), etc}
            sine_template (Optional[ndarray]): (Ne,2*Nh,Np). Sinusoidal template.
        """
        pass


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using filter-bank algorithms to predict test data.

        Args:
            X_test (ndarray): (Nb,Ne*Nte,Nc,Np). Test dataset.

        Return:
            rou (ndarray): (Ne*Nte,Ne). Decision coefficients.
                Not empty when self.standard is True.
            y_standard (ndarray): (Ne*Nte,). Predict labels.
            erou (ndarray): (Ne*Nte,Ne). Decision coefficients (ensemble).
                Not empty when self.ensemble is True.
            y_ensemble (ndarray): (Ne*Nte,). Predict labels (ensemble).
        """
        # basic information
        n_test = X_test.shape[1]

        # apply model.predict() method in each sub-band
        self.fb_rou = [[] for nb in range(self.n_bands)]
        self.fb_y_standard = [[] for nb in range(self.n_bands)]
        self.fb_erou = [[] for nb in range(self.n_bands)]
        self.fb_y_ensemble = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            fb_results = self.sub_models[nb].predict(X_test=X_test[nb])
            self.fb_rou[nb], self.fb_y_standard[nb] = fb_results[0], fb_results[1]
            self.fb_erou[nb], self.fb_y_ensemble[nb] = fb_results[2], fb_results[3]

        # integration of multi-bands' results
        self.rou = utils.combine_fb_feature(self.fb_rou)
        self.erou = utils.combine_fb_feature(self.fb_erou)
        self.y_standard = np.empty((n_test))
        self.y_ensemble = np.empty_like(self.y_standard)
        for nte in range(n_test):
            self.y_standard[nte] = np.argmax(self.rou[nte,:])
            self.y_ensemble[nte] = np.argmax(self.erou[nte,:])
        return self.rou, self.y_standard, self.erou, self.y_ensemble


# %%
class TL_TRCA(BasicTransfer):
    def source_intra_training(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> dict:
        """Intra-subject training for source dataset.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
            n_components (int): Number of eigenvectors picked as filters. Nk.
                Set to 'None' if ratio is not 'None'.
            ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
                Defaults to be 'None'.

        Returns: dict
            Q (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Covariance matrices.
            S (ndarray): (Ne,2*Nc+2*Nh,2*Nc+2*Nh). Variance matrices. Q^{-1}Sw = lambda w.
            w (List[ndarray]): Ne*(Nk,Nc). Spatial filters for original signal.
            u (List[ndarray]): Ne*(Nk,Nc). Spatial filters for averaged template.
            v (List[ndarray]): Ne*(Nk,2*Nh). Spatial filters for sinusoidal template.
            w_concat (ndarray): (Ne*Nk,Nc). Concatenated filter for w.
            u_concat (ndarray): (Ne*Nk,Nc). Concatenated filter for u.
            v_concat (ndarray): (Ne*Nk,2*Nh). Concatenated filter for v.
            uX (ndarray): (Ne,Ne*Nk,Np). Filtered averaged templates.
            vY (ndarray): (Ne,Ne*Nk,Np). Filtered sinusoidal templates.
        """
        # basic information
        event_type = np.unique(y_train)
        n_events = len(event_type)  # Ne of source dataset
        n_train = np.array([np.sum(y_train==et) for et in event_type])  # [Nt1,Nt2,...]
        n_chans = X_train.shape[-2]  # Nc
        n_points = X_train.shape[-1]  # Np
        n_2harmonics = sine_template.shape[1]  # 2*Nh

        # initialization
        S = np.zeros((n_events, 2*n_chans+n_2harmonics, 2*n_chans+n_2harmonics))
        Q = np.zeros_like(S)
        w, u, v, w_concat, u_concat, v_concat = [], [], [], [], [], []
        uX, vY = [], []

        # block covariance matrices: S & Q, (Ne,2Nc+2Nh,2Nc+2Nh)
        class_sum = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
        class_center = np.zeros_like(class_sum)
        for ne,et in enumerate(event_type):
            train_trials = n_train[ne]  # Nt
            assert train_trials>1, 'The number of training samples is too small!'
            X_temp = X_train[y_train==et]
            class_sum[ne] = np.sum(X_temp, axis=0)
            class_center[ne] = np.mean(X_temp, axis=0)
            XsXs = class_sum[ne] @ class_sum[ne].T  # (Nc,Nc)
            XsXm = class_sum[ne] @ class_center[ne].T  # (Nc,Nc), (Nc,Nc)
            XmXm = class_center[ne] @ class_center[ne].T  # (Nc,Nc), (Nc,Nc)
            XsY = class_sum[ne] @ sine_template[ne].T  # (Nc,2Nh), (2Nh,Nc)
            XmY = class_center[ne] @ sine_template[ne].T  # (Nc,2Nh), (2Nh,Nc)
            YY = sine_template[ne] @ sine_template[ne].T  # (2Nh,2Nh), (2Nh,2Nh)
            XX = np.zeros((n_chans, n_chans))  # (Nc,Nc)
            for tt in range(train_trials):
                XX += X_temp[tt] @ X_temp[tt].T
            # XX = np.einsum('tcp,thp->ch', X_sub[ne], X_sub[ne]) # clear but slow

            # block covariance matrices S: [[S11,S12,S13],[S21,S22,S23],[S31,S32,S33]]
            # S11: inter-trial covariance
            S[ne, :n_chans, :n_chans] = XsXs

            # S12 & S21.T covariance between the SSVEP trials & the individual template
            S[ne, :n_chans, n_chans:2*n_chans] = XsXm
            S[ne, n_chans:2*n_chans, :n_chans] = XsXm.T

            # S13 & S31.T: similarity between the SSVEP trials & sinusoidal template
            S[ne, :n_chans, 2*n_chans:] = XsY
            S[ne, 2*n_chans:, :n_chans] = XsY.T

            # S23 & S32.T: covariance between the individual template & sinusoidal template
            S[ne, n_chans:2*n_chans, 2*n_chans:] = XmY
            S[ne, 2*n_chans:, n_chans:2*n_chans] = XmY.T

            # S22 & S33: variance of individual template & sinusoidal template
            S[ne, n_chans:2*n_chans, n_chans:2*n_chans] = XmXm
            S[ne, 2*n_chans:, 2*n_chans:] = YY

            # block covariance matrices Q: blkdiag(Q1,Q2,Q3)
            # Q1: variance of the single-trial SSVEP
            Q[ne, :n_chans, :n_chans] = XX

            # Q2 & Q3: variance of individual template & sinusoidal template
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
            u.append(spatial_filter[:,n_chans:2*n_chans])  # (Nk,Nc) | for averaged template
            v.append(spatial_filter[:,2*n_chans:])  # (Nk,2Nh) | for sinusoidal template
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
        uX = np.zeros((n_events, u_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
        vY = np.zeros((n_events, v_concat.shape[0], n_points))  # (Ne,Ne*Nk,Np)
        for ne in range(n_events):  # ensemble version
            uX[ne] = u_concat @ class_center[ne]
            vY[ne] = v_concat @ sine_template[ne]

        # Intra-subject model
        intra_source_model = {
            'Q':Q, 'S':S,
            'w':w, 'u':u, 'v':v,
            'w_concat':w_concat, 'u_concat':u_concat, 'v_concat':v_concat,
            'uX':uX, 'vY':vY
        }
        return intra_source_model


    def transfer_learning(self):
        """Transfer learning process.

        Updates: object attributes
            n_subjects (int): The number of source subjects.
            source_intra_model (List[object]): See details in source_intra_training().
            partial_transfer_model (dict[str, List]):{
                'source_uX': List[ndarray]: Ns*(Ne,Ne*Nk,Np). uX of each source subject.
                'source_vY': List[ndarray]: Ns*(Ne,Ne*Nk,Np). vY of each source subject.
                'trans_uX': List[ndarray]: Ns*(Ne,Ne*Nk,Nc). Transfer matrices for uX.
                'trans_vY': List[ndarray]: Ns*(Ne,Ne*Nk,Nc). Transfer matrices for vY.
            }
        """
        # basic information
        self.n_subjects = len(self.X_source)  # Ns
        self.source_intra_model = []
        self.event_type = np.unique(self.y_train)
        self.n_events = len(self.event_type)
        self.n_chans = self.X_train.shape[-2]  # Nc for target dataset

        # obtain partial transfer model
        trans_uX, trans_vY = [], []
        source_uX, source_vY = [], []
        for nsub in range(self.n_subjects):
            intra_model = self.source_intra_training(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.source_intra_model.append(intra_model)
            uX, vY = intra_model['uX'], intra_model['vY']
            source_uX.append(uX)  # (Ne,Ne*Nk,Np)
            source_vY.append(vY)  # (Ne,Ne*Nk,Np)

            # LST alignment
            trans_uX.append(np.zeros((self.n_events, uX.shape[1], self.n_chans)))  # (Ne,Ne*Nk,Nc)
            trans_vY.append(np.zeros((self.n_events, vY.shape[1], self.n_chans)))  # (Ne,Ne*Nk,Nc)
            for ne,et in enumerate(self.event_type):
                X_temp = self.X_train[self.y_train==et]  # (Nt,Nc,Np)
                train_trials = X_temp.shape[0]
                for tt in range(train_trials):
                    trans_uX_temp, _, _, _ = sLA.lstsq(
                        a=X_temp[tt].T,
                        b=uX[ne].T
                    )  # b * a^T * (a * a^T)^{-1}
                    trans_vY_temp, _, _, _ = sLA.lstsq(
                        a=X_temp[tt].T,
                        b=vY[ne].T
                    )
                    trans_uX[nsub][ne] += trans_uX_temp.T
                    trans_vY[nsub][ne] += trans_vY_temp.T
                trans_uX[nsub][ne] /= train_trials
                trans_vY[nsub][ne] /= train_trials
        self.part_trans_model = {
            'source_uX':source_uX, 'source_vY':source_vY,
            'trans_uX':trans_uX, 'trans_vY':trans_vY
        }


    def data_augmentation(self,):
        """Do nothing."""
        pass


    def dist_calculation(self):
        """Calculate the spatial distances between source and target domain.

        Updates:
            dist_uX, dist_vY (ndarray): (Ns,Ne).
        """
        self.dist_uX = np.zeros((self.n_subjects, self.n_events))  # (Ns,Ne)
        self.dist_vY = np.zeros_like(self.dist_uX)
        for nsub in range(self.n_subjects):
            for ne,et in enumerate(self.event_type):
                X_temp = self.X_train[self.y_train==et]
                train_trials = X_temp.shape[0]
                for tt in range(train_trials):
                    self.dist_uX[nsub,ne] += utils.pearson_corr(
                        X=self.part_trans_model['trans_uX'][nsub][ne] @ X_temp[tt],
                        Y=self.part_trans_model['source_uX'][nsub][ne]
                    )
                    self.dist_vY[nsub,ne] += utils.pearson_corr(
                        X=self.part_trans_model['trans_vY'][nsub][ne] @ X_temp[tt],
                        Y=self.part_trans_model['source_vY'][nsub][ne]
                    )


    def weight_optimization(self):
        """Optimize the transfer weights.

        Updates:
            weight_uX, weight_vY (ndarray): (Ns,Ne)
        """
        self.weight_uX = self.dist_uX / np.sum(self.dist_uX, axis=0, keepdims=True)
        self.weight_vY = self.dist_vY / np.sum(self.dist_vY, axis=0, keepdims=True)


    def target_intra_training(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> dict:
        """Intra-subject training for target dataset.

        Args:
            See details in source_intra_training().

        Returns: dict
            See details in source_intra_training().
        """
        self.target_model = self.source_intra_training(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            n_components=n_components,
            ratio=ratio
        )


    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        stim_info: Optional[dict] = None,
        sine_template: Optional[ndarray] = None):
        """Load data and train TL-TRCA models.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset (target domain). Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            stim_info (Optional[dict]): Information of stimulus.
                {'event_type':ndarray, (Ne,),
                 'freqs':List or ndarray, (Ne,),
                 'phases':List or ndarray, (Ne,), etc}. No need here.
            sine_template (Optional[ndarray]): (Ne,2*Nh,Np). Sinusoidal template.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.stim_info = stim_info
        self.sine_template = sine_template

        # main process
        self.transfer_learning()
        self.dist_calculation()
        self.weight_optimization()
        self.target_intra_training(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            n_components=self.n_components,
            ratio=self.ratio
        )
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using TL-(e)TRCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rou (ndarray): (Ne*Nte,Ne,4). Decision coefficients of TL-TRCA.
            y_predict (ndarray): (Ne*Nte,). Predict labels of TL-TRCA.
        """
        # basic information
        n_test = X_test.shape[0]

        self.rou = np.zeros((n_test, self.n_events, 4))
        self.final_rou = np.zeros((n_test, self.n_events))
        self.y_predict = np.empty((n_test))

        # rou 1 & 2: transferred pattern matching
        for nte in range(n_test):
            for nsub in range(self.n_subjects):
                trans_uX = self.part_trans_model['trans_uX'][nsub]  # (Ne,Ne*Nk,Nc)
                trans_vY = self.part_trans_model['trans_vY'][nsub]  # (Ne,Ne*Nk,Nc)
                source_uX = self.part_trans_model['source_uX'][nsub]  # (Ne,Ne*Nk,Np)
                source_vY = self.part_trans_model['source_vY'][nsub]  # (Ne,Ne*Nk,Np)
                for nem in range(self.n_events):
                    self.rou[nte,nem,0] += self.weight_uX[nsub,nem]*utils.pearson_corr(
                        X=trans_uX[nem] @ X_test[nte],
                        Y=source_uX[nem]
                    )
                    self.rou[nte,nem,1] += self.weight_vY[nsub,nem]*utils.pearson_corr(
                        X=trans_vY[nem] @ X_test[nte],
                        Y=source_vY[nem]
                    )

        # rou 3 & 4: self-trained pattern matching (similar to sc-(e)TRCA)
        for nte in range(n_test):
            for nem in range(self.n_events):
                temp_standard = self.target_model['w'][nem] @ X_test[nte]  # (Nk,Np)
                self.rou[nte,nem,2] = utils.pearson_corr(
                    X=temp_standard,
                    Y=self.target_model['uX'][nem]
                )
                self.rou[nte,nem,3] = utils.pearson_corr(
                    X=temp_standard,
                    Y=self.target_model['vY'][nem]
                )
                self.final_rou[nte,nem] = utils.combine_feature([
                    self.rou[nte,nem,0],
                    self.rou[nte,nem,1],
                    self.rou[nte,nem,2],
                    self.rou[nte,nem,3],
                ])
            self.y_predict[nte] = self.event_type[np.argmax(self.final_rou[nte,:])]
        return self.rou, self.y_predict


class FB_TL_TRCA(BasicFBTransfer):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        stim_info: Optional[dict] = None,
        sine_template: Optional[ndarray] = None):
        """Load data and train FB-TL-TRCA models.

        Args:
            X_train (ndarray): (Nb,Ne*Nt,Nc,Np). Training target dataset. Typically Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Nb,Ne*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne*Nt,). Labels for X_source.
            stim_info (Optional[dict]): Information of stimulus.
                {'event_type':ndarray, (Ne,),
                 'freqs':List or ndarray, (Ne,),
                 'phases':List or ndarray, (Ne,), etc}. No need here.
            sine_template (Optional[ndarray]): (Nb,Ne,2*Nh,Np). Sinusoidal template.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.stim_info = stim_info
        self.sine_template = sine_template
        self.n_bands = self.X_train.shape[0]
        
        # train TL-TRCA models in each band
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = TL_TRCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            fb_X_source = [data[nb] for data in self.X_source]
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train[nb],
                X_source=fb_X_source,
                y_source=self.y_source,
                stim_info=self.stim_info,
                sine_template=self.sine_template[nb]
            )
        return self


# %% subject-based transfer CCA
class STCCA(BasicTransfer):
    def source_intra_training(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> dict[str, Any]:
        """Intra-subject training for source dataset.

        Args:
            X_train (ndarray): (Ne*Nt,Nc,Np). Training dataset. Nt>=2.
            y_train (ndarray): (Ne*Nt,). Labels for X_train.
            sine_template (ndarray): (Ne,2*Nh,Np). Sinusoidal template.
            n_components (int): Number of eigenvectors picked as filters. Nk.
                Nk of each subject must be same (given by input number or 1).
                i.e. ratio must be None.
            ratio (float, optional): The ratio of the sum of eigenvalues to the total (0-1).
                Defaults to None when n_component is not 'None'.

        Returns: dict
            Cxx (ndarray): (Nc,Nc). Variance matrices of EEG.
            Cxy (ndarray): (Nc,2*Nh). Covariance matrices.
            Cyy (ndarray): (2*Nh,2*Nh). Variance matrices of sinusoidal template.
            u (ndarray): (Nk,Nc). Common spatial filter for averaged template.
            v (ndarray): (Nk,2*Nh). Common spatial filter for sinusoidal template.
            uX (ndarray): (Ne,Nk,Np). Filtered averaged templates.
            vY (ndarray): (Ne,Nk,Np). Filtered sinusoidal templates.
            event_type (ndarray): (Ne,). Event id for current dataset.
        """
        # basic information
        event_type = np.unique(y_train)
        n_events = len(event_type)
        n_train = np.array([np.sum(y_train==et) for et in event_type])
        n_chans = X_train.shape[-2]  # Nc
        n_points = X_train.shape[-1]  # Np
        n_2harmonics = sine_template.shape[1]  # 2*Nh

        # obtain averaged template
        avg_template = np.zeros((n_events, n_chans, n_points))  # (Ne,Nc,Np)
        for ne,et in enumerate(event_type):
            train_trials = n_train[ne]
            assert train_trials > 1, 'The number of training samples is too small!'
            avg_template[ne] = np.mean(X_train[y_train==et], axis=0)  # (Nc,Np)

        # initialization
        Cxx = np.zeros((n_events, n_chans, n_chans))  # (Ne,Nc,Nc)
        Cyy = np.zeros((n_events, n_2harmonics, n_2harmonics))  # (Ne,2Nh,2Nh)
        Cxy = np.zeros((n_events, n_chans, n_2harmonics))  # (Ne,Nc,2Nh)

        # covariance matrices
        for ne in range(n_events):
            Cxx[ne] = avg_template[ne] @ avg_template[ne].T
            Cxy[ne] = avg_template[ne] @ sine_template[ne].T
            Cyy[ne] = sine_template[ne] @ sine_template[ne].T
        Cxx = np.sum(Cxx, axis=0)
        Cxy = np.sum(Cxy, axis=0)
        Cyy = np.sum(Cyy, axis=0)

        # solve GEPs
        u = utils.solve_gep(
            A=Cxy @ sLA.solve(Cyy, Cxy.T),
            B=Cxx,
            n_components=n_components,
            ratio=ratio
        )  # (Nk,Nc)
        v = utils.solve_gep(
            A=Cxy.T @ sLA.solve(Cxx, Cxy),
            B=Cyy,
            n_components=n_components,
            ratio=ratio
        )  # (Nk,2Nh)

        # intra-subject templates
        uX = np.zeros((n_events, u.shape[0], n_points))  # (Ne,Nk,Np)
        vY = np.zeros((n_events, v.shape[0], n_points))  # (Ne,Nk,Np)
        for ne in range(n_events):
            uX[ne] = u @ avg_template[ne]
            vY[ne] = v @ sine_template[ne]

        # intra-subject model
        intra_source_model = {
            'Cxx':Cxx, 'Cxy':Cxy, 'Cyy':Cyy,
            'u':u, 'v':v,
            'uX':uX, 'vY':vY, 'event_type':event_type
        }
        return intra_source_model


    def transfer_learning(self):
        """Transfer learning process. Actually there is no so-called transfer process.
            This function is only used for intra-subject training for source dataset.

        Updates: object attributes
            n_subjects (int): The number of source subjects.
            source_intra_model (List[object]): See details in source_intra_training().
            event_type (ndarray): (Ne,).
            n_events: (int). Total number of stimuli.
            n_chans: (int). Total number of channels.
        """
        # basic information
        self.n_subjects = len(self.X_source)  # Ns
        self.source_intra_model = []
        self.event_type = np.unique(self.y_train)
        self.n_events = len(self.event_type)
        self.n_chans = self.X_train.shape[-2]  # Nc for target dataset

        # intra-subject training for all source subjects
        for nsub in range(self.n_subjects):
            intra_model = self.source_intra_training(
                X_train=self.X_source[nsub],
                y_train=self.y_source[nsub],
                sine_template=self.sine_template,
                n_components=self.n_components,
                ratio=self.ratio
            )
            self.source_intra_model.append(intra_model)


    def data_augmentation(self):
        """Do nothing."""
        pass


    def dist_calculation(self):
        """Do nothing."""
        pass


    def weight_optimization(self):
        """Optimize the transfer weights.

        Updates:
            n_points (int).
            u (ndarray): (Nk,Nc). Spatial filters for averaged template (target).
            v (ndarray): (Nk,2Nh). Spatial filters for sinusoidal template (target).
            uX (ndarray): (Ne(t), Nk, Np). Filtered averaged templates (target).
            vY (ndarray): (Ne(t), Nk, Np). Filtered sinusoidal templates (target).
            buX (ndarray): (Nk, Ne(t)*Np). Concatenated uX (target).
            AuX (ndarray): (Ns*Nk, Ne(s)*Np). Concatenated uX (source).
            bvY (ndarray): (Nk, Ne(t)*Np). Concatenated vY (target).
            AvY (ndarray): (Ns*Nk, Ne(s)*Np). Concatenated vY (source).
            weight_uX (ndarray): (Ns,Nk). Transfer weights of averaged templates (source).
            weight_vY (ndarray): (Ns,Nk). Transfer weights of sinusoidal templates (source).
            wuX (ndarray): (Ne(s), Nk, Np). Transferred averaged templates (full-event).
            wvY (ndarray): (Ne(s), Nk, Np). Transferred sinusoidal templates (full-event).
        """
        # basic information
        self.n_points = self.X_train.shape[-1]  # Np

        # intra-subject training for target domain
        self.target_intra_model = self.target_intra_training(
            X_train=self.X_train,
            y_train=self.y_train,
            sine_template=self.sine_template,
            n_components=self.n_components,
            ratio=self.ratio
        )
        u, v = self.target_model['u'], self.target_model['v']
        uX, vY = self.target_model['uX'], self.target_model['vY']

        # transfer weights training based on LST: min||b-Aw||_2^2
        self.buX = np.zeros((u.shape[0], self.n_events*self.n_points))  # (Nk,Ne*Np)
        # self.bvY = np.zeros((v.shape[0], self.n_events*self.n_points))  # (Nk,Ne*Np)
        for ne in range(self.n_events):  # Ne for target dataset
            self.buX[:, ne*self.n_points:(ne+1)*self.n_points] = uX[ne]
            # self.bvY[:, ne*self.n_points:(ne+1)*self.n_points] = vY[ne]
        uX_total_Nk = 0
        # vY_total_Nk = 0
        for nsub in range(self.n_subjects):
            uX_total_Nk += self.source_intra_model[nsub]['u'].shape[0]
            # vY_total_Nk += self.source_intra_model[ns]['v'].shape[0]
        self.AuX = np.zeros((uX_total_Nk, self.n_events*self.n_points))  # (Ns*Nk,Ne*Np)
        # self.AvY = np.zeros((vY_total_Nk, self.n_events*self.n_points))  # (Ns*Nk,Ne*Np)
        row_uX_idx = 0
        # row_vY_idx = 0
        for nsub in range(self.n_subjects):
            uX_Nk = self.source_intra_model[nsub]['u'].shape[0]
            # vY_Nk = self.source_intra_model[nsub]['v'].shape[0]
            source_uX = self.source_intra_model[nsub]['uX']
            # source_vY = self.source_intra_model[nsub]['vY']
            source_event_type = self.source_intra_model[nsub]['event_type'].tolist()
            for ne,et in enumerate(self.event_type):
                event_id = source_event_type.index(et)
                self.AuX[row_uX_idx:row_uX_idx+uX_Nk, ne*self.n_points:(ne+1)*self.n_points] = source_uX[event_id]
                # AvY[row_vY_idx:row_vY_idx+vY_Nk, ne*self.n_points:(ne+1)*self.n_points] = source_vY[event_id]
            row_uX_idx += uX_Nk
            # row_vY_idx += vY_Nk
        self.weight_uX, _, _, _ = sLA.lstsq(a=self.AuX.T, b=self.buX.T)  # (Ns,Nk)
        # self.weight_vY, _, _, _ = sLA.lstsq(a=self.AvY.T, b=self.bvY.T)  # (Ns,Nk)

        # cross subject averaged templates
        self.wuX = np.zeros_like(self.source_intra_model[0]['uX'])  # (Ne(s),Nk,Np)
        # self.wvY = np.zeros_like(self.source_intra_model[0]['vY'])
        for nsub in range(self.n_subjects):
            self.wuX += np.einsum('k,ekp->ekp', self.weight_uX[nsub], self.source_intra_model[nsub]['uX'])
            # self.wvY += np.einsum('k,ekp->ekp', self.weight_vY[nsub], self.source_intra_model[nsub]['vY'])
        self.wuX /= self.n_subjects
        # self.wvY /= self.n_subjects


    def target_intra_training(self,
        X_train: ndarray,
        y_train: ndarray,
        sine_template: ndarray,
        n_components: Optional[int] = 1,
        ratio: Optional[float] = None) -> dict[str, Any]:
        """Intra-subject training for target dataset.

        Args:
            See details in source_intra_training().

        Returns: dict
            See details in source_intra_training().
        """
        self.target_model = self.source_intra_training(
            X_train=X_train,
            y_train=y_train,
            sine_template=sine_template,
            n_components=n_components,
            ratio=ratio
        )


    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        sine_template: Optional[ndarray] = None):
        """Load data and train stCCA model.

        Args:
            X_train (ndarray): (Ne(t)*Nt,Nc,Np). Target training dataset. Typically Nt>=2.
            y_train (ndarray): (Ne(t)*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Ne(s)*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne(s)*Nt,). Labels for X_source.
            sine_template (ndarray): (Ne(t), 2*Nh, Np). Sinusoidal template.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template
        
        # main process
        self.transfer_learning()
        self.weight_optimization()
        return self


    def predict(self,
        X_test: ndarray) -> Tuple[ndarray]:
        """Using stCCA algorithm to compute decision coefficients.

        Args:
            X_test (ndarray): (Ne*Nte,Nc,Np). Test dataset.

        Returns:
            rou (ndarray): (Ne*Nte,Ne,4). Decision coefficients of TL-TRCA.
            y_predict (ndarray): (Ne*Nte,). Predict labels of TL-TRCA.
        """
        # basic information
        n_test = X_test.shape[0]
        self.rou = np.zeros((n_test, self.n_events, 2))
        self.final_rou = np.zeros((n_test, self.n_events))
        self.y_predict = np.empty((n_test))

        # rou 1 & 2
        for nte in range(n_test):
            temp = self.target_model['u'] @ X_test[nte]  # (Nk,Np)
            for nem in range(self.n_events):
                self.rou[nte,nem,0] = utils.pearson_corr(
                    X=temp,
                    Y=self.target_model['v'] @ self.sine_template[nem]
                )
                self.rou[nte,nem,1] = utils.pearson_corr(
                    X=temp,
                    Y=self.wuX[nem]
                )
                # self.rou[nte,nem,2] = utils.pearson_corr(
                #     X=temp,
                #     Y=self.wvY[nem]
                # )
                self.final_rou[nte,nem] = utils.combine_feature([
                    self.rou[nte,nem,0],
                    self.rou[nte,nem,1],
                    # self.rou[nte,nem,2]
                ])
            self.y_predict[nte] = self.event_type[np.argmax(self.final_rou[nte,:])]
        return self.rou, self.y_predict


class FB_STCCA(BasicFBTransfer):
    def fit(self,
        X_train: ndarray,
        y_train: ndarray,
        X_source: List[ndarray],
        y_source: List[ndarray],
        sine_template: Optional[ndarray] = None):
        """Load data and train FB-stCCA models.

        Args:
            X_train (ndarray): (Nb,Ne(t)*Nt,Nc,Np). Target training dataset. Typically Nt>=2.
            y_train (ndarray): (Ne(t)*Nt,). Labels for X_train.
            X_source (List[ndarray]): Ns*(Nb,Ne(s)*Nt,Nc,Np). Source dataset.
            y_source (List[ndarray]): Ns*(Ne(s)*Nt,). Labels for X_source.
            sine_template (ndarray): (Nb,Ne(t), 2*Nh, Np). Sinusoidal template.
        """
        # load in data
        self.X_train = X_train
        self.y_train = y_train
        self.X_source = X_source
        self.y_source = y_source
        self.sine_template = sine_template
        self.n_bands = self.X_train.shape[0]

        # train stCCA models in each band
        self.sub_models = [[] for nb in range(self.n_bands)]
        for nb in range(self.n_bands):
            self.sub_models[nb] = STCCA(
                standard=self.standard,
                ensemble=self.ensemble,
                n_components=self.n_components,
                ratio=self.ratio
            )
            fb_X_source = [data[nb] for data in self.X_source]
            self.sub_models[nb].fit(
                X_train=self.X_train[nb],
                y_train=self.y_train[nb],
                X_source=fb_X_source,
                y_source=self.y_source,
                sine_template=self.sine_template[nb]
            )
        return self

# %%
