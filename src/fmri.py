import numpy as np
from scipy.special import gammaln
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set(style='whitegrid', font_scale=1.5)


def spm_hrf(TR, P=None, fMRI_T=16):
    """ Python implentation of spm hrf function. From Poldrack Lab:
    https://github.com/poldracklab/poldracklab-base/blob/master/fmri/spm_hrf.py
    """
    p = np.array([6, 16, 1, 1, 6, 0, 32], dtype=float)
    if P is not None:
        p[0:len(P)] = P

    def _spm_Gpdf(x, h, l):
        tmp1 = h * np.log(l)
        tmp2 = (h - 1) * np.log(x)
        tmp3 = (l * x) - gammaln(h)
        return np.exp(tmp1 + tmp2 - tmp3)

    # modelled hemodynamic response function - {mixture of Gammas}
    dt = TR / float(fMRI_T)
    u = np.arange(0, int(p[6] / dt + 1)) - p[5] / dt
    with np.errstate(divide='ignore'):  # Known division-by-zero
        tmp1 = _spm_Gpdf(u, p[0] / p[2], dt / p[2])
        tmp2 = _spm_Gpdf(u, p[1] / p[3], dt / p[3]) / p[4]
        hrf = tmp1 - tmp2

    idx = np.arange(0, int((p[6] / TR) + 1)) * fMRI_T
    hrf = hrf[idx]
    hrf = hrf / np.sum(hrf)
    return hrf


def generate_task_regressors(trial_type, onsets, durations, rts,
                             tr, num_acq, dt):
    """
    """

    # upsampled times to construct regressors with greater hrf resolution
    scan_time = num_acq * tr
    times = np.round(np.arange(0, scan_time + dt, dt), 3)
    num_times = times.shape[0]

    # make base boxcars
    boxcar_base = np.zeros(num_times)
    for onset, duration in zip(onsets, durations):
        mask = np.logical_and(times >= onset, times <= onset + duration)
        boxcar_base[mask] = 1

    # make rt variable boxcars
    boxcar_rt = np.zeros(num_times)
    for onset, rt in zip(onsets, rts):
        mask = np.logical_and(times >= onset, times <= onset + rt)
        boxcar_rt[mask] = 1

    # perform convolutions
    hrf = spm_hrf(dt)
    bold_base = np.convolve(boxcar_base, v=hrf)
    bold_base = bold_base[:boxcar_base.shape[-1]]
    bold_rt = np.convolve(boxcar_rt, v=hrf)
    bold_rt = bold_rt[:boxcar_rt.shape[-1]]

    # downsample to tr sampling resolution
    tr_indices = np.where(np.mod(times, tr) == 0)[0][:-1]
    bold_base = bold_base[tr_indices]
    bold_rt = bold_rt[tr_indices]

    # normalize (max height = 1)
    bold_base /= bold_base.max()
    bold_rt /= bold_rt.max()

    return [bold_base, bold_rt]


def plot_task_regressors(regressors, labels, subject):

    def plot_vif(regs, labs, ax):
        n_reg = regs.shape[1]
        vif = [variance_inflation_factor(regs, i) for i in range(n_reg)]
        ticks = np.arange(len(vif)) + .8
        ax.bar(ticks, vif)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels[2:])
        ax.set_ylabel('VIF')
        return ax

    fig = plt.figure(figsize=(20, 12))

    for i, label in enumerate(labels):
        reg = regressors[:, i]

        ax = plt.subplot2grid((4, 6), (i, 0), colspan=4)
        ax.plot(reg)
        ax.set_title(label)
        ax.set_xlabel('TR')
        ax.set_xlim((1, len(reg)))

    # plot the base vif
    ax = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2)
    ax = plot_vif(regressors[:, :2], labels[:2], ax)
    ax.set_title('Base Regressors')

    # plot the rt vif
    ax = plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=2)
    ax = plot_vif(regressors[:, 2:], labels[2:], ax)
    ax.set_title('RT Regressors')

    plt.suptitle('%s Task Regressors' % subject)
    sns.despine()
    plt.tight_layout()

    return fig

