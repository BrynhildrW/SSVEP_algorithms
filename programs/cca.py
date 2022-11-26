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
    (5) MsetCCA: https://www.worldscientific.com/doi/abs/10.1142/S0129065714500130
            DOI: 10.1142/S0129065714500130
    (6) MwayCCA: 
            DOI: 
    (7) CCA-M3: https://www.worldscientific.com/doi/abs/10.1142/S0129065720500203
            DOI: 10.1142/S0129065720500203
    (8)

update: 2022/11/15

"""

# %% basic modules
from utils import *


# %% (1) standard CCA | CCA
def cca_compute(data, template, Nk=1, ratio=None):
    """Canonical correlation analysis.

    Args:
        data (ndarray): (n_chans, n_points). Real EEG data of a single trial.
        template (ndarray): (2*n_harmonics or m, n_points). Artificial sinusoidal template or averaged template.
        Nk (int): Number of eigenvectors picked as filters.
            Eigenvectors are referring to eigenvalues sorted in descend order.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        U (ndarray): (Nk, n_chans). Spatial filter for EEG.
        V (ndarray): (Nk, 2*n_harmonics). Spatial filter for template.
    """
    # GEPs' conditions
    Cxx = data @ data.T  # (Nc,Nc)
    Cyy = template @ template.T  # (2Nh,2Nh)
    Cxy = data @ template.T  # (Nc,2Nh)
    Cyx = Cxy.template  # (2Nh,Nc)
    A = sLA.solve(Cxx,Cxy) @ sLA.solve(Cyy,Cyx)  # AU = lambda*U
    B = sLA.solve(Cyy,Cyx) @ sLA.solve(Cxx,Cxy)  # BV = lambda*V

    # EEG part
    U = solve_ep(A, Nk, ratio)  # (Nk,Nc)

    # template part
    V = solve_ep(B, Nk, ratio)  # (Nk,2Nh)
    return U, V


def cca(template, test_data, Nk=1, ratio=None):
    """Using CCA to compute decision coefficients.

    Args:
        template (ndarray): (n_events, 2*n_harmonics, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_events = test_data.shape[0]
    n_test = test_data.shape[1]

    # CCA classification
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real, Nt, Ne model)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                U, V = cca_compute(data=temp, template=template[nem,...], Nk=Nk, ratio=ratio)
                rou[ner,nte,nem] = pearson_corr(U@temp, V@template[nem,...])
    return rou


# %% (2) Extended CCA | eCCA
def ecca_compute(avg_template, sine_template, test_data, Nk=1, ratio=None):
    """CCA with individual calibration data.

    Args:
        avg_template (ndarray): (n_chans, n_points). Trial-averaged data.
        sine_template (ndarray): (2*n_harmonics, n_points). Sinusoidal template.
        test_data (ndarray): (n_chans, n_points). Test-trial EEG.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (float): feature coefficient.
    """
    # correlation coefficient from CCA process
    U1, V1 = cca_compute(data=test_data, template=sine_template, Nk=Nk, ratio=ratio)
    r1 = pearson_corr(U1@test_data, V1@sine_template)

    # correlation coefficients between single-trial EEG and SSVEP templates
    U2, V2 = cca_compute(data=test_data, template=avg_template, Nk=Nk, ratio=ratio)
    r2 = pearson_corr(U2@test_data, U2@avg_template)

    r3 = pearson_corr(U1@test_data, U1@avg_template)

    U3, _ = cca_compute(data=avg_template, template=sine_template, Nk=Nk, ratio=ratio)
    r4 = pearson_corr(U3@test_data, U3@avg_template)

    # similarity between filters corresponding to single-trial EEG and SSVEP templates
    r5 = pearson_corr(U2@avg_template, V2@avg_template)

    # combined features
    rou = combine_feature([r1, r2, r3, r4, r5])
    return np.real(rou)


def ecca(train_data, sine_template, test_data, Nk=1, ratio=None):
    """Use eCCA to compute decision coefficient.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                train_mean = train_data[nem,...].mean(axis=0)  # (Nc,Np)
                rou[ner,nte,nem] = ecca_compute(avg_template=train_mean,
                    sine_template=sine_template[nem,...], test_data=temp, Nk=Nk, ratio=ratio)
    return rou


def ecca_sp_compute(Xmean, Y, data):
    """CCA with individual calibration data. | More coefficient.

    Args:
        Xmean (ndarray): (n_chans, n_points). Trial-averaged SSVEP template.
        Y (ndarray): (2*n_harmonics, n_points). Sinusoidal SSVEP template.
        data (ndarray): (n_chans, n_points). Test-trial EEG.

    Returns:
        rou (float): feature coefficient.
    """
    U1, V1 = cca(test_data=data, template=Y)
    U2, V2 = cca(test_data=data, template=Xmean)
    U3, V3 = cca(test_data=Xmean, template=Y)
    pass


# %% (3-4) Multi-stimulus eCCA | ms-eCCA
# msCCA is only part of ms-eCCA. Personally, i dont like this design
def mscca_compute(avg_template, sine_template, Nk=1, ratio=None):
    """Multi-stimulus CCA.

    Args:
        avg_template (ndarray): (n_events, n_chans, n_points). Trial-averaged data.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        w (ndarray): (Nk, n_chans). Common spatial filter.
    """
    # GEPs' conditions
    Czz = einsum('ecp,ehp->ch', avg_template,avg_template)  # (Nc,Nc)
    Cyy = einsum('ecp,ehp->ch', sine_template,sine_template)  # (2Nh,2Nh)
    Czy = einsum('ecp,ehp->ch', avg_template,sine_template)  # (Nc,2Nh)
    Cyz = Czy.T  # (2Nh,Nc)
    A = sLA.solve(Czz,Czy) @ sLA.solve(Cyy,Cyz)  # AU = lambda*U

    # ms-CCA part
    w = solve_ep(A, Nk, ratio)  # (Nk,Nc)
    return w


def mscca(train_data, sine_template, test_data, Nk=1, ratio=None):
    """Use msCCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]

    # training models & filters
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    w = mscca_compute(Xmean=train_mean, sine_template=sine_template,
                      Nk=Nk, ratio=ratio)  # (Nk,Nc)
    model = einsum('kc,ecp->ekp', w,train_mean)  # (Ne,Nk,Np)

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                rou[ner,nte,nem] = pearson_corr(w@temp, model[nem,...])
    return rou


def msecca_compute(avg_template, sine_template, events_group=None, Nk=1, ratio=None):
    """Multi-stimulus eCCA.

    Args:
        avg_template (ndarray): (n_events, n_chans, n_points). Averaged template.
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        events_group (dict): {'events':[start index,end index]}
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        U (list of ndarray): n_events * (Nk, n_chans). Spatial filters for EEG.
        V (list of ndarray): n_events * (Nk, 2*n_harmonics). Spatial filters for templates.
    """
    # basic information
    n_events = avg_template.shape[0]

    # GEPs' conditions
    Czz_total = einsum('ecp,ehp->ech', avg_template,avg_template)  # (Ne,Nc,Nc)
    Cyy_total = einsum('ecp,ehp->ech', sine_template,sine_template)  # (Ne,2Nh,2Nh)
    Czy_total = einsum('ecp,ehp->ech', avg_template,sine_template)  # (Ne,Nc,2Nh)

    # GEPs with merged data
    U, V = [], []
    for ne in range(n_events):
        # GEPs' conditions
        idx = str(ne)
        st, ed = events_group[idx][0], events_group[idx][1]
        Czz = np.sum(Czz_total[st:ed], axis=0)  # (Nc,Nc)
        Cyy = np.sum(Cyy_total[st:ed], axis=0)  # (2Nh,2Nh)
        Czy = np.sum(Czy_total[st:ed], axis=0)  # (Nc,2Nh)
        Cyz = Czy.T  # (2Nh,Nc)
        A = sLA.solve(Czz,Czy) @ sLA.solve(Cyy,Cyz)  # AU = lambda*U
        B = sLA.solve(Cyy,Cyz) @ sLA.solve(Czz,Czy)  # AU = lambda*U

        # solve GEPs
        U.append(solve_ep(A, Nk, ratio))  # EEG part: (Nk,Nc)
        V.append(solve_ep(B, Nk, ratio))  # template part: (Nk,2Nh)
    return U, V


def msecca(train_data, sine_template, test_data, d, Nk=1, ratio=None, **kwargs):
    """Using ms-eCCA to compute decision coefficients.

    Args:
        train_data (ndarray): (n_events, n_train, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        test_data (ndarray): (n_events, n_test, n_chans, n_points).
        d (int): The range of events to be merged.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_events = train_data.shape[0]
    n_test = test_data.shape[1]
    try:
        events_group = kwargs['events_group']
    except KeyError:
        events_group = augmented_events(n_events, d)

    # training models & filters
    train_mean = train_data.mean(axis=1)  # (Ne,Nc,Np)
    U,V = msecca_compute(Xmean=train_mean, sine_template=sine_template,
        events_group=events_group, Nk=Nk, ratio=ratio)
    model_eeg, model_template = [], []
    for ne in range(n_events):
        model_eeg.append(U[ne] @ train_mean[ne])
        model_template.append(V[ne] @ sine_template[ne])

    # pattern matching
    rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
    for ner in range(n_events):
        for nte in range(n_test):
            temp = test_data[ner,nte,...]  # (Nc,Np)
            for nem in range(n_events):
                r1 = pearson_corr(U[nem]@temp, model_template[nem])
                r2 = pearson_corr(U[nem]@temp, model_eeg[nem])
                rou[ner,nte,nem] = np.real(combine_feature([r1, r2]))
    return rou


# %% (5) Multiset CCA | MsetCCA
def msetcca_compute():
    pass


def msetcca():
    pass


# %% Cross-subject transfer learning | CSSFT
def cssft_compute():
    pass


def cssft():
    pass



# %% Filter-bank CCA series | FB-
def fbcca(sine_template, test_data, Nk=1, ratio=None):
    """CCA algorithms with filter banks.

    Args:
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
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
        rou.append(cca(template=sine_template, test_data=test_data[nb], Nk=Nk, ratio=ratio))
    return combine_fb_feature(rou)


def fbecca(train_data, sine_template, test_data, Nk=1, ratio=None):
    """eCCA with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_bands = train_data.shape[0]
    n_events = train_data.shape[1]
    n_test = test_data.shape[2]

    # pattern matching
    rou = []
    for nb in range(n_bands):
        temp_rou = np.zeros((n_events, n_test, n_events))
        for ner in range(n_events):
            for nte in range(n_test):
                temp = test_data[nb,ner,nte,...]  # (Nc,Np)
                for nem in range(n_events):
                    Xmean = train_data[nb,nem,...].mean(axis=0)  # (Nc,Np)
                    temp_rou[ner,nte,nem] = ecca_compute(Xmean=Xmean,
                        Y=sine_template[nem,...], test_data=temp, Nk=Nk, ratio=ratio)
        rou.append(temp_rou)
    return combine_fb_feature(rou)


def fbmscca(train_data, sine_template, test_data, Nk=1, ratio=None):
    """msCCA with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_bands = train_data.shape[0]
    n_events = train_data.shape[1]
    n_test = test_data.shape[2]

    # training models & filters
    train_mean = train_data.mean(axis=2)  # (Nb,Ne,Nc,Np)
    w, model = [], []
    for nb in range(n_bands):
        w.append(mscca_compute(Xmean=train_mean[nb], sine_template=sine_template,
                               Nk=Nk, ratio=ratio))  # (Nk,Nc)
        model.append(einsum('kc,ecp->ekp', w[nb],train_mean[nb]))  # (Ne,Nk,Np)

    # pattern matching
    rou = []
    for nb in range(n_bands):
        temp_rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
        for ner in range(n_events):
            for nte in range(n_test):
                temp = test_data[nb,ner,nte,...]  # (Nc,Np)
                for nem in range(n_events):
                    temp_rou[ner,nte,nem] = pearson_corr(w[nb]@temp, model[nb][nem])
        rou.append(temp_rou)
    return combine_fb_feature(rou)


def fbmsecca(train_data, sine_template, test_data, d, Nk=1, ratio=None, **kwargs):
    """ms-eCCA with filter banks.

    Args:
        train_data (ndarray): (n_bands, n_events, n_train, n_chans, n_points).
        sine_template (ndarray): (n_events, 2*n_harmonics, n_points). Sinusoidal template.
        test_data (ndarray): (n_bands, n_events, n_test, n_chans, n_points).
        d (int): The range of events to be merged.
        Nk (int): Number of eigenvectors picked as filters.
            Set to 'None' if ratio is not 'None'.
        ratio (float): 0-1. The ratio of the sum of eigenvalues to the total.
            Defaults to be 'None'.

    Returns:
        rou (ndarray): (n_events for real, n_test, n_events for model).
    """
    # basic information
    n_bands = train_data.shape[0]
    n_events = train_data.shape[1]
    n_test = test_data.shape[2]
    try:
        events_group = kwargs['events_group']
    except KeyError:
        events_group = augmented_events(n_events, d)

    # training models & filters
    train_mean = train_data.mean(axis=2)  # (Nb,Ne,Nc,Np)
    U = [[] for nb in range(n_bands)]
    V = [[] for nb in range(n_bands)]
    model_eeg = [[] for nb in range(n_bands)]
    model_template = [[] for nb in range(n_bands)]
    for nb in range(n_bands):
        U[nb], V[nb] = msecca_compute(Xmean=train_mean[nb], sine_template=sine_template,
                                      events_group=events_group, Nk=Nk, ratio=ratio)
        for ne in range(n_events):
            model_eeg[nb].append(U[nb][ne] @ train_mean[nb,ne,...])
            model_template[nb].append(V[nb][ne] @ sine_template[ne])

    # pattern matching
    rou = []
    for nb in range(n_bands):
        temp_rou = np.zeros((n_events, n_test, n_events))  # (Ne real,Nt,Ne model)
        for ner in range(n_events):
            for nte in range(n_test):
                temp = test_data[nb,ner,nte,...]  # (Nc,Np)
                for nem in range(n_events):
                    r1 = pearson_corr(U[nb][nem]@temp, model_template[nb][nem])
                    r2 = pearson_corr(U[nb][nem]@temp, model_eeg[nb][nem])
                    temp_rou[ner,nte,nem] = np.real(combine_feature([r1, r2]))
        rou.append(temp_rou)
    return combine_fb_feature(rou)