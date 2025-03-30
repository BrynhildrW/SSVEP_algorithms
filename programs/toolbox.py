'''
Author:
    RuixinLuo  ruixin_luo@tju.edu.cn
'''

# load modules
import numpy as np
import numpy.matlib
import math
import scipy.io as sio
import warnings
from scipy import signal
# warnings.filterwarnings('default')
from sklearn.cross_decomposition import CCA


def corr2(a, b):
    a = a - np.sum(a) / np.size(a)
    b = b - np.sum(b) / np.size(b)
    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r


def acc_calculate(predict):
    """
    :param predict:  (n_trial,n_event)
    :return: acc
    """
    [nTrials, nEvents] = predict.shape
    label_target = np.ones((nTrials, 1)) * np.arange(0, nEvents, 1, int)
    logical_right = (label_target == predict)
    acc_num = np.sum(logical_right != 0)
    acc = acc_num / nTrials / nEvents
    return acc


# msCCA
def cal_CCA(X, Y):
    """ CCA count
    :param X: (num of sample points * num of channels1 )
    :param Y: (num of sample points * num of channels2 )
    :return:
    """

    #  Center the variables
    X = X - np.tile(np.mean(X, 0), (X.shape[0], 1))
    if Y.ndim == 1:
        Ynew = Y.reshape(Y.shape[0], 1)  # reshape（num of sample points）to（num of sample points * 1）
        del Y
        Y = Ynew
        Y = Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1))
    else:
        Y = Y - np.tile(np.mean(Y, 0), (Y.shape[0], 1))

    #  Calculate corr_cca
    P_U = Y @ np.linalg.inv(Y.T @ Y) @ Y.T
    b_U = (np.linalg.inv(X.T @ X)) @ (X.T @ P_U @ X)
    [eig_value_U, eig_U] = np.linalg.eig(b_U)  # Calculate U
    U = eig_U[:, 0]
    P_V = X @ np.linalg.inv(X.T @ X) @ X.T
    b_V = (np.linalg.inv(Y.T @ Y)) @ (Y.T @ P_V @ Y)
    [eig_value_V, eig_V] = np.linalg.eig(b_V)  # Calculate V
    V = eig_V[:, 0]

    corr = np.corrcoef(U.T @ X.T, V.T @ Y.T)
    corr_cca = corr[0, 1]
    return U.real, V.real, corr_cca.real


def ms_eCCA_spatialFilter(
        mean_temp_all,
        iEvent,
        nTemplates,
        fs,
        f_list,
        phi_list,
        Nh):
    """ spatialFilter for multi-stimulus extended canonical correlation analysis (ms-eCCA)
    adopted from https://github.com/edwin465/SSVEP-MSCCA-MSTRCA

    :param mean_temp_all: ndarray(n_channels, n_times, n_events)
    :param iEvent:        the i-th event for the selection of neighboring frequencies
    :param nTemplates:    the number of neighboring frequencies
    :param fs:            the sample rate
    :param f_list:        the all frequency of reference signal
    :param phi_list:      the all phase of reference signal
    :param Nh:            the number of harmonics

    :return:  U, V        Spatial Filters
    """

    mean_temp_all = np.transpose(mean_temp_all, [1, 0, 2])

    [nTimes, nChannels, nEvents] = mean_temp_all.shape
    d0 = nTemplates / 2
    d1 = nEvents

    n = iEvent + 1
    if n <= d0:
        template_st = 1
        template_ed = nTemplates
    elif (n > d0) & (n < (d1 - d0 + 1)):
        template_st = n - d0
        template_ed = n + (nTemplates - d0 - 1)
    else:
        template_st = (d1 - nTemplates + 1)
        template_ed = d1
    template_st = int(template_st - 1)
    template_ed = int(template_ed)

    #  Concatenation of the templates (or sine-cosine references)
    mscca_ref = np.zeros((nTemplates * nTimes, 2 * Nh))
    mscca_template = np.zeros((nTemplates * nTimes, nChannels))

    index = 0
    for j in range(template_st, template_ed, 1):
        # sine-cosine references
        f = f_list[j]
        phi = phi_list[j]
        Ts = 1 / fs
        n = np.arange(nTimes) * Ts
        Yf = np.zeros((nTimes, Nh * 2))
        for iNh in range(Nh):
            y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n + (iNh + 1) * np.pi * phi)
            Yf[:, iNh * 2] = y_sin
            y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n + (iNh + 1) * np.pi * phi)
            Yf[:, iNh * 2 + 1] = y_cos
        mscca_ref[index * nTimes: (index + 1) * nTimes, :] = Yf
        # templates
        ss = mean_temp_all[:, :, j]
        # ss = ss - np.tile(np.mean(ss, 0), (ss.shape[0], 1))
        mscca_template[index * nTimes:(index + 1) * nTimes, :] = ss
        index = index + 1

    # calculate U and V
    U, V, r = cal_CCA(mscca_template, mscca_ref)
    if r < 0:   # Symbol Control
        V = -V
    return U, V


# tlCCA
def square(t, duty=50):
    """
    adopted from Matlab square
    :param t:     time
    :param duty:  duty rate
    :return:
    """
    # Compute values of t normalized to (0,2*pi)
    tem = np.mod(t, 2 * np.pi)
    # Compute normalized frequency for breaking up the interval (0,2*pi)
    w0 = 2 * np.pi * duty / 100
    # Assign 1 values to normalized t between (0,w0), 0 elsewhere
    nodd = np.array((tem < w0))
    # The actual square wave computation
    s = 2 * nodd - 1
    return s


def _get_convH(f, ph, Fs, tw, refresh_rate, erp_period):
    """ obtain impulse response H
    adopted from: https://github.com/edwin465/SSVEP-Impulse-Response

    :parameter:
    f: frequency
    ph: phase
    Fs: sample rate
    tw: time length
    refresh_rate: refresh_rate of LCD
    erp_period:   length of r

    :return:
    H0 impulse response H
    """
    sig_len = int(np.floor(tw * Fs))
    t = np.arange(sig_len) / Fs
    h0 = np.cos(2 * np.pi * f * t + ph * np.pi) + 1  # init h0 as cos

    # sel_idx represents the number of sample points per frame
    # refresh_rate*tw = tol frames
    sel_idx = np.round(Fs / refresh_rate * np.arange(
        refresh_rate * tw) + 0.001) + 1  # +0.001 Guaranteed rounding upwards when .5, the same as matlab
    h_val = h0[0]  # when time=0

    # Obtain the sample signal sampled at the refresh rate
    cn = 0
    h = np.zeros_like(h0)
    for m in range(h0.shape[0]):
        if m + 1 == sel_idx[cn]:
            h_val = h0[m]
            if cn + 1 >= sel_idx.shape[0]:
                pass
            else:
                cn = cn + 1
        else:
            pass
        h[m] = h_val

    #  Obtain the impulse response (all tw)
    hs = square(2 * np.pi * f * t + ph * np.pi, duty=20) + 1
    hs = hs.astype(float)  # change int to float
    count_thres = np.floor(0.9 * Fs / f)  # < points of a cycle
    count = count_thres + 1
    for m in range(h0.shape[0]):
        if hs[m] == 0:
            count = count_thres + 1
        else:
            if count >= count_thres:
                hs[m] = h[m]
                count = 1
            else:
                count = count + 1
                hs[m] = 0

    # Obtain the impulse response H
    erp_len = int(np.round(erp_period * Fs))
    H = np.zeros((erp_len, erp_len + sig_len - 1))
    for k in range(erp_len):
        H[k, k:k + sig_len] = hs

    H = H[:, :sig_len]

    return H


def decompose_tlCCA(mean_temp, fre_list, ph_list, Fs, Oz_loc):
    """ decompose SSVEPs according to tlCCA
    source code: https://github.com/edwin465/SSVEP-Impulse-Response
    from matlab to python

    :parameter
    mean_temp ndarray(n_channels, n_times,n_events)
    fre_list  ndarray(n_events,)
    ph_list   ndarray(n_events,)
    Fs  sample rate
    Oz_loc  the location of Oz

    :return:
    w_all list(n_events) spatially filters of all events
    r_all list(n_events) inpalse responses of all events
    """
    n_channel, n_times, n_event = mean_temp.shape
    tw = n_times / Fs

    w_all = []
    r_all = []
    for i in range(n_event):
        dat = mean_temp[..., i]
        dat_oz = dat[Oz_loc, :]  # for symbol correction
        # get w, r
        fre_period = 1.05 * (1 / fre_list[i])
        H0 = _get_convH(
            f=fre_list[i],
            ph=ph_list[i],
            Fs=Fs,
            tw=tw,
            refresh_rate=60,
            erp_period=fre_period
        )
        w, r, corr = cal_CCA(dat.T, H0.T)
        if corr < 0:
            w = -w
        # symbol correction
        r_oz = dat_oz @ H0.T @ np.linalg.inv(H0 @ H0.T)
        cr = np.corrcoef(r, r_oz)
        if cr[0, 1] < 0:
            w = -w
            r = -r
        w_all.append(w)
        r_all.append(r)
    return w_all, r_all


def reconstruct_tlCCA(r, fre_list_target, fre_list_souce, ph_list, Fs, tw):
    """ reconstruct SSVEPs according to tlCCA
    source code: https://github.com/edwin465/SSVEP-Impulse-Response
    from matlab to python

    :parameter
    r_all list(n_events) impulse responses of all events
    fre_list  ndarray(n_events,)
    ph_list   ndarray(n_events,)
    Fs  sample rate

    :return:
    Hr  ndarray(n_times,n_events)   reconstructed SSVEP
    """
    n_event = fre_list_target.size
    Hr = np.zeros((int(tw * Fs), n_event))
    for i in range(n_event):
        # get H0
        fre_period = 1.05 * (1 / fre_list_souce[i])
        H1 = _get_convH(
            f=fre_list_target[i],
            ph=ph_list[i],
            Fs=Fs,
            tw=tw,
            refresh_rate=60,
            erp_period=fre_period
        )
        y_hat = r[i] @ H1
        ind0 = np.where(y_hat == 0)[0]
        n_ind0 = ind0.shape[0]
        y_hat[:n_ind0] = 0.8 * y_hat[Fs:Fs + n_ind0]
        y_hat = y_hat - np.mean(y_hat)
        y_hat = y_hat / np.std(y_hat, ddof=1)
        Hr[:, i] = y_hat
    return Hr


def reconstruct_tlCCA_classification(r, fre_list_target, fre_list_souce, ph_list, Fs, tw):
    """ reconstruct SSVEPs according to tlCCA
    source code: https://github.com/edwin465/SSVEP-Impulse-Response
    from matlab to python

    :parameter
    r_all list(n_events) impulse responses of all events
    fre_list  ndarray(n_events,)
    ph_list   ndarray(n_events,)
    Fs  sample rate

    :return:
    Hr  ndarray(n_times,n_events)   reconstructed SSVEP
    """
    n_event = fre_list_target.size
    Hr = np.zeros((int(tw * Fs), n_event))
    for i in range(n_event):
        # get H0
        fre_period = 1.05 * (1 / fre_list_souce[i])
        H1 = _get_convH(f=fre_list_target[i], ph=ph_list[i], Fs=Fs, tw=tw, refresh_rate=60, erp_period=fre_period)
        y_hat = r[i] @ H1
        # ind0 = np.where(y_hat == 0)[0]
        # n_ind0 = ind0.shape[0]
        # y_hat[:n_ind0] = 0.8 * y_hat[Fs:Fs + n_ind0]
        y_hat = y_hat - np.mean(y_hat)
        y_hat = y_hat / np.std(y_hat, ddof=1)
        Hr[:, i] = y_hat
    return Hr


def SS_tlCCA_test(test_data_S, hr, u, v, w, fs, f, phi, Nh):
    """ test data under singel trials using tlCCA
    source code: https://github.com/edwin465/SSVEP-Impulse-Response
    from matlab to python

    :parameter
    test_Data_S   ndarray(n_channels,n_times)
    hr            ndarray(n_times,)
    u             ndarray(n_channels,)
    v             ndarray(2*n_harmonics,)
    w             ndarray(n_channels,)
    fs            sample rate
    f             the frequency of Yf
    phi           the phase of Yf
    Nh            the number of harmonics

    :return:
    corr_tlCCA
    """

    # calculate corr_tlCCA
    nTimes = test_data_S.shape[-1]

    Ts = 1 / fs
    n = np.arange(nTimes) * Ts
    Yf_cca = np.zeros((nTimes, Nh * 2))
    for iNh in range(Nh):
        y_sin = np.sin(2 * np.pi * f * (iNh + 1) * n + (iNh + 1) * np.pi * phi)
        Yf_cca[:, iNh * 2] = y_sin
        y_cos = np.cos(2 * np.pi * f * (iNh + 1) * n + (iNh + 1) * np.pi * phi)
        Yf_cca[:, iNh * 2 + 1] = y_cos

    rr1 = np.corrcoef(u.T @ test_data_S, v.T @ Yf_cca.T)
    rr2 = np.corrcoef(w.T @ test_data_S, hr)
    # _,_,rr3 = cal_CCA(np.expand_dims (w.T @ test_data_S,1), Yf_cca) # When inputting single channel data, the output has problems
    cca = CCA(n_components=1)
    cca.fit(np.expand_dims(w.T @ test_data_S, 1), Yf_cca)
    X_train_r, Y_train_r = cca.transform(np.expand_dims(w.T @ test_data_S, 1), Yf_cca)
    rr3 = np.corrcoef(X_train_r[:, 0], Y_train_r[:, 0])[0, 1]  # The result of CCA is positive
    rr1_abs = rr1[0, 1]
    rr2_abs = rr2[0, 1]

    corr_tlCCA = np.sign(rr1_abs) * (rr1_abs)**2 + np.sign(rr2_abs) * (rr2_abs)**2 + (rr3)**2
    return corr_tlCCA


def tlCCA_test(test_data, HR, U, V, W, fs, f_list, phi_list, Nh):
    """ test data under mulple trials using tlCCA

    :parameter
    test_Data   ndarray(n_channels,n_times,n_test)
    HR            ndarray(n_times,n_events)
    U             ndarray(n_channels,,n_events)
    V             ndarray(2*n_harmonics,,n_events)
    W             ndarray(n_channels,,n_events)
    fs            sample rate
    f_list        the all frequency of Yf
    phi_list      the all phase of Yf
    Nh            the number of harmonics

    :return:
    rr            ndarray(n_test,n_events)
    """
    [nChannels, nTimes, nEvents] = test_data.shape
    rr = np.zeros((nEvents, nEvents))
    for m in range(nEvents):  # the m-th test data
        r = np.zeros(nEvents)
        for n in range(nEvents):  # the n-th train data
            test = test_data[:, :, m]
            train = HR[:, n]
            r[n] = SS_tlCCA_test(test, train, u=U[:, n], v=V[:, n], w=W[:, n], fs=fs, f=f_list[n], phi=phi_list[n], Nh=Nh)
        rr[m, :] = r
    return rr


class PreProcessing():
    """Adopted from OrionHH
    """
    CHANNELS = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
        'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
        'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FC8', 'T7',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
        'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
        'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
        'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]  # M1: 33. M2: 43.

    def __init__(self, filepath, t_begin, t_end, n_classes=40, fs_down=250, chans=None, num_filter=1):

        self.filepath = filepath
        self.fs_down = fs_down
        self.t_begin = t_begin
        self.t_end = t_end
        self.chans = chans
        self.n_classes = n_classes
        self.num_filter = num_filter

    def load_data(self):
        '''
        Application: load data and selected channels by chans.

        :param chans: list | None
        :return: raw_data: 4-D, numpy
            n_chans * n_samples * n_classes * n_trials
        :return: event: 2-D, numpy
            event[0, :]: label
            event[1, :]: latency

        '''
        raw_mat = sio.loadmat(self.filepath)
        raw_data = raw_mat['data']  # (64, 1500, 40, 6)
        # begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)

        idx_loc = list()
        if isinstance(self.chans, list):
            for _, char_value in enumerate(self.chans):
                idx_loc.append(self.CHANNELS.index(char_value.upper()))

        raw_data = raw_data[idx_loc, ...] if idx_loc else raw_data

        self.raw_fs = 250  # .mat sampling rate
        return raw_data

    def resample_data(self, raw_data):
        '''
        :param raw_data: from method load_data.
        :return: raw_data_resampled, 4-D, numpy
            n_chans * n_samples * n_classes * n_trials
        '''
        if self.raw_fs > self.fs_down:
            raw_data_resampled = signal.resample(
                raw_data,
                round(self.fs_down * raw_data.shape[1] / self.raw_fs),
                axis=1
            )
        elif self.raw_fs < self.fs_down:
            warnings.warn('You are up-sampling, no recommended')
            raw_data_resampled = signal.resample(
                raw_data,
                round(self.fs_down * raw_data.shape[1] / self.raw_fs),
                axis=1
            )
        else:
            raw_data_resampled = raw_data

        return raw_data_resampled

    def _get_iir_sos_band(self, w_pass, w_stop):
        '''
        Get second-order sections (like 'ba') of Chebyshev type I filter.
        :param w_pass: list, 2 elements
        :param w_stop: list, 2 elements
        :return: sos_system
            i.e the filter coefficients.
        '''
        if len(w_pass) != 2 or len(w_stop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if w_pass[0] > w_pass[1] or w_stop[0] > w_stop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

        if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')

        wp = [2 * w_pass[0] / self.fs_down, 2 * w_pass[1] / self.fs_down]
        ws = [2 * w_stop[0] / self.fs_down, 2 * w_stop[1] / self.fs_down]
        gpass = 4
        gstop = 30  # dB

        N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
        sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')

        return sos_system

    def filtered_data_iir(self, w_pass_2d, w_stop_2d, data):
        '''
        filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data: 4-d, numpy, from method load_data or resample_data.
            n_chans * n_samples * n_classes * n_trials
        :return: filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans * n_samples * n_classes * n_trials.
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.num_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)
        sos_system = dict()
        filtered_data = dict()
        for idx_filter in range(self.num_filter):
            sos_system['filter' + str(idx_filter + 1)] = self._get_iir_sos_band(
                w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                w_stop=[w_stop_2d[0, idx_filter], w_stop_2d[1, idx_filter]]
            )
            filter_data = signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=1)
            filtered_data['bank' + str(idx_filter + 1)] = filter_data[:, begin_point:end_point, :, :]

        return filtered_data

    def filtered_data_iir111(self, w_pass_2d, w_stop_2d, data):
        '''
        filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data: 4-d, numpy, from method load_data or resample_data.
            n_chans * n_samples * n_classes * n_trials
        :return: filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans * n_samples * n_classes * n_trials.
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.num_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)
        data = data[:, begin_point:end_point, :, :]

        sos_system = dict()
        filtered_data = dict()
        for idx_filter in range(self.num_filter):
            sos_system['filter' + str(idx_filter + 1)] = self._get_iir_sos_band(
                w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                w_stop=[w_stop_2d[0, idx_filter], w_stop_2d[1, idx_filter]]
            )
            filter_data = signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=1)
            filtered_data['bank' + str(idx_filter + 1)] = filter_data

        return filtered_data


class PreProcessing_BETA():
    """Adopted from OrionHH
    """
    CHANNELS = [
        'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3',
        'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
        'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FC8', 'T7',
        'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
        'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4',
        'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
        'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
        'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2'
    ]  # M1: 33. M2: 43.

    def __init__(self, filepath, t_begin, t_end, n_classes=40, fs_down=250, chans=None, num_filter=1):

        self.filepath = filepath
        self.fs_down = fs_down
        self.t_begin = t_begin
        self.t_end = t_end
        self.chans = chans
        self.n_classes = n_classes
        self.num_filter = num_filter

    def load_data(self):
        '''
        Application: load data and selected channels by chans.

        :param chans: list | None
        :return: raw_data: 4-D, numpy
            n_chans * n_samples * n_classes * n_trials
        :return: event: 2-D, numpy
            event[0, :]: label
            event[1, :]: latency

        '''
        raw_mat = sio.loadmat(self.filepath)
        raw_data11 = raw_mat['data']  # (64, 1500, 40, 6)
        data = raw_data11[0, 0]['EEG']
        raw_data = np.transpose(data, [0, 1, 3, 2])

        idx_loc = list()
        if isinstance(self.chans, list):
            for _, char_value in enumerate(self.chans):
                idx_loc.append(self.CHANNELS.index(char_value.upper()))

        raw_data = raw_data[idx_loc, ...] if idx_loc else raw_data
        self.raw_fs = 250  # .mat sampling rate
        return raw_data

    def resample_data(self, raw_data):
        '''
        :param raw_data: from method load_data.
        :return: raw_data_resampled, 4-D, numpy
            n_chans * n_samples * n_classes * n_trials
        '''
        if self.raw_fs > self.fs_down:
            raw_data_resampled = signal.resample(
                raw_data,
                round(self.fs_down * raw_data.shape[1] / self.raw_fs),
                axis=1
            )
        elif self.raw_fs < self.fs_down:
            warnings.warn('You are up-sampling, no recommended')
            raw_data_resampled = signal.resample(
                raw_data,
                round(self.fs_down * raw_data.shape[1] / self.raw_fs),
                axis=1
            )
        else:
            raw_data_resampled = raw_data
        return raw_data_resampled

    def _get_iir_sos_band(self, w_pass, w_stop):
        '''
        Get second-order sections (like 'ba') of Chebyshev type I filter.
        :param w_pass: list, 2 elements
        :param w_stop: list, 2 elements
        :return: sos_system
            i.e the filter coefficients.
        '''
        if len(w_pass) != 2 or len(w_stop) != 2:
            raise ValueError('w_pass and w_stop must be a list with 2 elements.')

        if w_pass[0] > w_pass[1] or w_stop[0] > w_stop[1]:
            raise ValueError('Element 1 must be greater than Element 0 for w_pass and w_stop.')

        if w_pass[0] < w_stop[0] or w_pass[1] > w_stop[1]:
            raise ValueError('It\'s a band-pass iir filter, please check the values between w_pass and w_stop.')

        wp = [2 * w_pass[0] / self.fs_down, 2 * w_pass[1] / self.fs_down]
        ws = [2 * w_stop[0] / self.fs_down, 2 * w_stop[1] / self.fs_down]
        gpass = 4
        gstop = 30  # dB

        N, wn = signal.cheb1ord(wp, ws, gpass=gpass, gstop=gstop)
        sos_system = signal.cheby1(N, rp=0.5, Wn=wn, btype='bandpass', output='sos')
        return sos_system

    def filtered_data_iir(self, w_pass_2d, w_stop_2d, data):
        '''
        filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data: 4-d, numpy, from method load_data or resample_data.
            n_chans * n_samples * n_classes * n_trials
        :return: filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans * n_samples * n_classes * n_trials.
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.num_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)
        sos_system = dict()
        filtered_data = dict()
        for idx_filter in range(self.num_filter):
            sos_system['filter' + str(idx_filter + 1)] = self._get_iir_sos_band(
                w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                w_stop=[w_stop_2d[0, idx_filter], w_stop_2d[1, idx_filter]]
            )
            filter_data = signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=1)
            filtered_data['bank' + str(idx_filter + 1)] = filter_data[:, begin_point:end_point, :, :]

        return filtered_data

    def filtered_data_iir111(self, w_pass_2d, w_stop_2d, data):
        '''
        filter data by IIR, which parameters are set by method _get_iir_sos_band in BasePreProcessing class.
        :param w_pass_2d: 2-d, numpy,
            w_pass_2d[0, :]: w_pass[0] of method _get_iir_sos_band,
            w_pass_2d[1, :]: w_pass[1] of method _get_iir_sos_band.
        :param w_stop_2d: 2-d, numpy,
            w_stop_2d[0, :]: w_stop[0] of method _get_iir_sos_band,
            w_stop_2d[1, :]: w_stop[1] of method _get_iir_sos_band.
        :param data: 4-d, numpy, from method load_data or resample_data.
            n_chans * n_samples * n_classes * n_trials
        :return: filtered_data: dict,
            {'bank1': values1, 'bank2': values2, ...,'bank'+str(num_filter): values}
            values1, values2,...: 4-D, numpy, n_chans * n_samples * n_classes * n_trials.
        e.g.
        w_pass_2d = np.array([[5, 14, 22, 30, 38, 46, 54],[70, 70, 70, 70, 70, 70, 70]])
        w_stop_2d = np.array([[3, 12, 20, 28, 36, 44, 52],[72, 72, 72, 72, 72, 72, 72]])
        '''
        if w_pass_2d.shape != w_stop_2d.shape:
            raise ValueError('The shape of w_pass_2d and w_stop_2d should be equal.')
        if self.num_filter > w_pass_2d.shape[1]:
            raise ValueError('num_filter should be less than or equal to w_pass_2d.shape[1]')

        begin_point, end_point = int(np.ceil(self.t_begin * self.fs_down)), int(np.ceil(self.t_end * self.fs_down) + 1)
        data = data[:, begin_point:end_point, :, :]

        sos_system = dict()
        filtered_data = dict()
        for idx_filter in range(self.num_filter):
            sos_system['filter' + str(idx_filter + 1)] = self._get_iir_sos_band(
                w_pass=[w_pass_2d[0, idx_filter], w_pass_2d[1, idx_filter]],
                w_stop=[w_stop_2d[0, idx_filter], w_stop_2d[1, idx_filter]]
            )
            filter_data = signal.sosfiltfilt(sos_system['filter' + str(idx_filter + 1)], data, axis=1)
            filtered_data['bank' + str(idx_filter + 1)] = filter_data

        return filtered_data
