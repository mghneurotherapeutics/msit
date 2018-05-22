import os
import glob
import pickle
from mne.io import Raw
from mne import read_epochs
from mne import Epochs
from mne import find_events
from scipy.special import lpmn
from numpy.linalg import inv
from mne import pick_types
from mne import read_evokeds, combine_evoked
from mne.time_frequency import read_tfrs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from mne.preprocessing import create_eog_epochs, create_ecg_epochs, ICA

sns.set(style='whitegrid', font_scale=1.5)


CH_NAMES = ['Fp1', 'Fpz', 'Fp2',
            'AF7', 'AF3', 'AFz',
            'AF4', 'AF8', 'F7',
            'F5', 'F3', 'F1', 'Fz',
            'F2', 'F4', 'F6',
            'F8', 'FT9', 'FT7',
            'FC5', 'FC3', 'FC1',
            'FCz', 'FC2', 'FC4',
            'FC6', 'FT8', 'FT10',
            'T9', 'T7', 'C5', 'C3',
            'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'T10',
            'TP9', 'TP7', 'CP5',
            'CP3', 'CP1', 'CPz',
            'CP2', 'CP4', 'CP6',
            'TP8', 'TP10', 'P9',
            'P7', 'P5', 'P3',
            'P1', 'Pz', 'P2', 'P4',
            'P6', 'P8', 'P10',
            'PO7', 'PO3', 'P0z',
            'PO4', 'PO8', 'O1',
            'Oz', 'O2', 'Iz']


def mark_bad_channels(raw_file, bad_ch_file, config, sub_behavior):
    """
    Plots the raw data for a subject allowing interactive manual
    updating of bad channels in the raw by clicking on the bad
    channel timecourses. Additionally plots epochs butterfly plot
    to assist with detecting bad channels.

    Parameters
    ----------
    raw_file : str
        filepath to the raw mne object fif file
    bad_ch_file : str
        filepath to the bad channels text file
    config: dict
        dictionary containing configuration parameters for making the epochs
    sub_behavior: DataFrame
        dataframe containing the subject's behavior for making the epochs

    Returns
    -------
    MNE Raw Object
        the update MNE Raw object with updated bad channels stored in its info
    """

    # load in the raw and existing bad channels if present
    raw = Raw(raw_file, verbose=False, preload=True)
    if os.path.exists(bad_ch_file):
        raw.load_bad_channels(bad_ch_file)

    # iteratively mark bad channels and check until satisfied
    ok = 'n'
    while ok != 'y':

        # make butterfly plots to check outlier channels
        epochs = construct_epochs(raw, config['epochs_info'], sub_behavior,
                                  add_buffer=False)
        fig = plot_butterfly(epochs, h_freq=40)
        fig.show()

        # plot the raw for manual bad channel marking
        raw.plot(show=True, block=True)

        ok = raw_input('Ok to move on? (y or n):')

    return raw


def update_events(eeg_events, epo_type, behavior):
    """
    Syncs the behavioral and eeg events by removing non-response trials,
    handling misalignment between the eeg and behavior events, and encoding
    trial type in the eeg events.

    Parameters
    ----------
    eeg_events: MNE events numpy array
        MNE events array containing the eeg derived events
    epo_type: str
        the type of epoch being processed (e.g. stimulus, response, ...)
    behavior: DataFrame
        dataframe containing the subject's behavior

    Returns
    -------
    MNE events numpy array
        the updated events array aligned with the behavior, with non-response
        trials removed, and with trial type encoded
    """

    subject = behavior.participant_id.iloc[0]

    # pp012 is missing the first stimulus trigger since
    # recording was started after the first stimulus presentation
    if subject == 'sub-pp012':
        behavior = behavior.iloc[1:, :].reset_index(drop=True)
        if epo_type == 'response':
            eeg_events = eeg_events[1:, :]

    # assert event counts match and reduce only to response trials
    response_ix = np.array(~behavior.response_time.isnull())
    if epo_type == 'stimulus':
        assert(behavior.shape[0] == eeg_events.shape[0])
        behavior = behavior[response_ix]
        eeg_events = eeg_events[response_ix]
    elif epo_type == 'response':
        behavior = behavior[response_ix]
        assert(behavior.shape[0] == eeg_events.shape[0])

    # remove too fast trials, error trials, and post-error trials from the eeg
    exclude = ['fast_rt', 'error', 'post_error']
    behavior.loc[:, 'exclude'] = np.where(np.sum(behavior[exclude],
                                                 axis=1) > 0, 1, 0)
    keep_ix = np.array(behavior.exclude == 0)
    behavior = behavior[keep_ix]
    eeg_events = eeg_events[keep_ix, :]

    # encode trial type in the eeg events
    behavior.loc[behavior.trial_type == 'incongruent',
                 'trial_type'] = 1
    behavior.loc[behavior.trial_type == 'congruent',
                 'trial_type'] = 0
    behavior_events = behavior.trial_type
    eeg_events[:, -1] = np.array(behavior_events)

    return eeg_events


def construct_epochs(raw, epochs_info, behavior, add_buffer=True,
                     baseline_correct=True):
    """
    Constructs MNE epochs objects from the raw data given behavioral data
    and epoch configuration information.

    Parameters
    ----------
    raw: MNE Raw Object
        MNE events array containing the eeg derived events
    epochs_info: dict
        configuration dictionary containing important epoch info including:
            - the event channels to find the events on
            - the epoch types
            - the time boundaries for the epochs
            - information for extracting the baseline period
    behavior: DataFrame
        dataframe containing the subject's behavior
    add_buffer: boolean, optional
        boolean indicating whether to add an extra buffer period to the front
        and back of the epoch to allow for further filtering
    baseline_correct: boolean, optional
        boolean indicating whether to baseline correct the epoch by subtracting
        the average of the baseline period from every epoch

    Returns
    -------
    list of MNE Epochs objects
        list containing an MNE Epochs object for each epoch type specified in
        epochs_info
    """

    epochs = {}

    for i, epo_type in enumerate(epochs_info['epo_types']):

        # extract events
        events = find_events(raw, stim_channel=epochs_info['event_chs'][i],
                             output='onset', verbose=False)

        # verify and update events
        events = update_events(events, epo_type, behavior)

        # make epochs
        tmin = epochs_info['epo_boundaries'][i][0]
        tmax = epochs_info['epo_boundaries'][i][1]
        if add_buffer:
            tmin -= epochs_info['filter_buffer']
            tmax += epochs_info['filter_buffer']
        epochs[epo_type] = Epochs(raw, events, baseline=None, proj=False,
                                  event_id=epochs_info['event_ids'],
                                  tmin=tmin, tmax=tmax, verbose=False,
                                  detrend=None, preload=True)

    # baseline correct
    if baseline_correct:

        # extract baseline data
        baseline_data = epochs[epochs_info['baseline_container']].copy()
        tmin = epochs_info['baseline_boundary'][0]
        tmax = epochs_info['baseline_boundary'][1]
        baseline_data = baseline_data.crop(tmin, tmax).get_data()

        for epo_type in epochs_info['epo_types']:
            epochs[epo_type] = epochs_baseline_correct(epochs[epo_type],
                                                       baseline_data)

    return epochs


def epochs_baseline_correct(epochs, baseline):
    """
    Baseline correct epochs object by subtracting average of baseline period
    from each epoch.

    Parameters
    ----------
    epochs: MNE Epochs Object
        MNE epochs object containing the epochs data to be corrected
    baseline: MNE Epochs Object
        MNE epochs object containing the baseline data to be used
        for correction

    Returns
    -------
    MNE Epochs Object
        MNE epochs object containing the corrected epochs data
    """

    # extract data out from mne objects
    epochs_data = epochs.get_data()

    # collapse baseline along time
    baseline = baseline.mean(axis=-1)

    # subtract baseline average from every time point
    epochs_data = np.array([arr - baseline.T for arr in epochs_data.T]).T

    # replace the data in the epochs object with baseline corrected data
    epochs._data = epochs_data

    return epochs


def fit_ica(raw, aux_chs, n_components=None, seed=50):
    """
    Compute ICA on the raw data and automatically extract out bad components
    highly correlated with EOG and ECG signals.

    Parameters
    ----------
    raw: MNE Raw Object
        MNE raw object containing the raw data to be perform ICA on
    aux_chs: dict
        dictionary mapping EOG and ECG channel names to channel types

    Returns
    -------
    MNE ICA Object
        MNE ICA object containing the fitted ICA results and auto-marked
        bad components
    """

    # highpass filter the raw to assist with fitting
    raw.filter(l_freq=1, h_freq=None)

    # extract number of components
    if not n_components:
        n_components = raw.estimate_rank()

    # fit the ICA
    ica = ICA(n_components=n_components, random_state=seed)
    ica.fit(raw, verbose=False)

    # auto-detect bad eog and ecg components
    for ch in aux_chs:
        if 'eog' in ch.lower():
            epochs = create_eog_epochs(raw, ch_name=ch, verbose=False)
            ix, scores = ica.find_bads_eog(epochs, ch_name=ch)
        else:
            epochs = create_ecg_epochs(raw, ch_name=ch, verbose=False)
            ix, scores = ica.find_bads_ecg(epochs, ch_name=ch)

        ica.labels_[ch] = ix
        ica.exclude += ix

    ica.labels_ = {ch: ica.labels_[ch] for ch in aux_chs}
    return ica


def verify_ica(ica, raw, sub_behavior, config):
    """
    Manually verify ICA components marked for exclusion.

    Plots various metrics including:
      - epochs before and after ica bad components removal
      - components time-locked to eog and ecg events
      - data before and after ica, time-locked to eog and ecg events
      - interactive ICA components plot with ability to update bad components
        by clicking on time courses and plot topomaps of component weightings
        by clicking on component channel names

    Parameters
    ----------
    ica: MNE ICA Object
        MNE ICA object containing the ICA whose components are going
        to be checked
    raw: MNE Raw Object
        MNE raw object containing the raw data the ICA was fit to
    sub_behavior: DataFrame
        Dataframe containing subject behavior used to create epochs
    config: dict
        configuration dictionary containing important epoch info

    Returns
    -------
    MNE ICA Object
        MNE ICA object with updated bad components
    """

    subject = sub_behavior.participant_id.iloc[0]

    # plot before and after ICA butterfly plots
    fig, axs = plt.subplots(2, 2, figsize=(20, 8))
    before_epo = construct_epochs(raw, config['epochs_info'],
                                  sub_behavior, add_buffer=False)
    after_epo = construct_epochs(ica.apply(raw.copy()),
                                 config['epochs_info'],
                                 sub_behavior, add_buffer=False)
    plot_butterfly(before_epo, title='Before ICA', axs=axs[0, :])
    plot_butterfly(after_epo, title='After ICA', axs=axs[1, :])

    # plot before and after ica component diagnostics
    plot_ica_rejection_results(raw, ica, config['aux_chs'])

    # plot component time courses
    # for manual bad component updating
    ica.plot_sources(raw, block=True, title=subject)

    return ica


def plot_ica_rejection_results(raw, ica, aux_chs):
    """
    Plots ICA components time-locked to eog and ecg events and
    plots raw data before and after ICA, time-locked to eog and ecg events.

    Parameters
    ----------
    ica: MNE ICA Object
        MNE ICA object containing the ICA whose components are going to
        be visualized
    raw: MNE Raw Object
        MNE raw object containing the raw data the ICA was fit to
    aux_chs: dict
        dictionary mapping EOG and ECG channel names to channel types


    Returns
    -------
    None
        Shows the plots described above
    """

    raw.filter(l_freq=1, h_freq=None)

    for i, ch in enumerate(aux_chs):
        # create artifact aligned epochs
        if 'eog' in ch.lower():
            epochs = create_eog_epochs(raw, ch_name=ch, verbose=False)
        else:
            epochs = create_ecg_epochs(raw, ch_name=ch, verbose=False)

        # plot data before and after ica epoched to artifact events
        f = ica.plot_overlay(epochs.average())
        f.suptitle('%s Before & After' % ch)

        # plot source time courses epoched to artifact events
        f = ica.plot_sources(epochs.average(), exclude=ica.exclude)
        f.suptitle('%s ICA Time Courses' % ch)


def plot_butterfly(epochs, title='', axs=None, ylim=(-20, 20),
                   tmins=None, tmaxs=None, l_freq=None, h_freq=None):
    """
    Makes a butterfly plot of the evoked data for a given MNE Epochs object.

    Parameters
    ----------
    epochs: list of MNE Epochs Objects
        MNE Epochs objects whose data will be visualized
    title: str
        The title for the plot
    axs: list of Axes Objects
        List of Matplotlib Axes objects to plot to. Default is none, in which
        case will plot to a new set of axes
    ylim: tuple
        Y axis limits for the plots. Default is -20 to 20.
    tmins: list
        List of starting times to crop the data to.
        Must be the same length as epochs.
    tmaxs: list
        List of ending times to crop the data to.
        Must be the same length as epochs.
    l_freq: int
        Highpass frequency to filter the data. Default is None, no highpass
    h_freq: int
        Lopass frequency to filter the data. Default is None, no lowpass

    Returns
    -------
    Matplotlib Figure Object
        The figure object containing the evoked butterfly plot
    """

    fig = None
    n_epochs = len(epochs.keys())
    if axs is None:
        fig, axs = plt.subplots(1, n_epochs,
                                figsize=(8 * n_epochs, 8))
    for i, epo_type in enumerate(epochs.keys()):

        evoked = epochs[epo_type].average()

        if tmins:
            tmin = tmins[i]
        else:
            tmin = evoked.times[0]
        if tmaxs:
            tmax = tmaxs[i]
        else:
            tmax = evoked.times[-1]
        evoked.crop(tmin, tmax)

        if l_freq or h_freq:
            evoked.filter(h_freq=h_freq, l_freq=l_freq)

        evoked.plot(spatial_colors=True, axes=axs[i], show=False)
        axs[i].set_ylim(ylim)
        axs[i].set_title('%s %s locked' % (title, epo_type))

    return fig


def plot_autoreject_summary(ar, subject, epo_type, ch_names):
    """
    Makes a sumamry plot of fitted autoreject results for a given subject
    and epochs type. This summary plot includes:
      - Heatmap of bad (red) and interpolated (blue) epoch segments
      - Distribution of channel amplitude rejection thresholds
      - Heatmap of cv scores for the consensus % and # of interpolation
        parameters with the highest score highlighted

    Parameters
    ----------
    ar: Autoreject object
        The fitted autoreject object to be summarized
    subject: str
        The subject code for the subject being plotted
    epo_type: str
        The type of epoch being plotted
    ch_names: list
        List of channel names

    Returns
    -------
    Matplotlib Figure Object
        The figure object containing the autoreject summary plot
    """

    fig = plt.figure(figsize=(24, 12))

    # plot the fix Log
    ax = plt.subplot2grid((4, 6), loc=(0, 0), rowspan=4, colspan=3)

    # sub-select to good eeg channels
    fix_log = ar.fix_log.T[ar.picks, :]

    # mark bad epochs
    if len(ar.bad_epochs_idx) > 0:
        fix_log[:, ar.bad_epochs_idx] = 1

    # re-code interpolations
    fix_log[fix_log == 2] = -1

    # heatmap with touches
    sns.set(style='white')
    sns.heatmap(fix_log, ax=ax, cbar=False, center=0, cmap=plt.cm.bwr)
    ax.set_yticklabels(np.array(ch_names)[ar.picks][::-1], rotation=0)
    ax.set_xticks([])
    ax.set_xlabel('Epoch')
    ax.set_title('%d Bad Epochs' % len(ar.bad_epochs_idx))

    # plot the channel thresholds
    ax = plt.subplot2grid((4, 6), loc=(0, 3), rowspan=2, colspan=3)
    sns.distplot(np.array(ar.threshes_.values()) * 1e6, ax=ax,
                 norm_hist=False, kde=False)
    ax.set_ylabel('# Channels')
    ax.set_xlabel('Amplitude Threshold')

    # plot the cross-validation for k and row
    ax = plt.subplot2grid((4, 6), loc=(2, 3), rowspan=2, colspan=3)

    # extract average validation score
    loss = ar.loss_['eeg'].mean(axis=-1)

    # heatmap with touches
    ax.matshow(loss.T * 1e6, cmap=plt.get_cmap('Blues_r'))
    ax.set_xticks(np.arange(len(ar.consensus_percs)))
    ax.set_yticks(np.arange(len(ar.n_interpolates)))
    ax.set_xticklabels(ar.consensus_percs)
    ax.set_yticklabels(ar.n_interpolates)
    ax.set_xlabel('Consensus %')
    ax.set_ylabel('# Interpolate')
    ax.xaxis.set_ticks_position('bottom')

    # highlight selected values
    idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
    rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # add title
    plt.tight_layout()
    plt.suptitle('%s %s autoreject summary' % (subject, epo_type), fontsize=24)
    plt.subplots_adjust(top=0.93)

    return fig


def extract_bad_ch_group_info(pipeline_root, ch_names):
    """
    Extracts bad channel summary information across subjects.

    Parameters
    ----------
    pipeline_root: str
        Filepath to directory containing eeg preprocessing results
    ch_names: list
        List of channel names in the data

    Returns
    -------
    DataFrame
        Dataframe where each column is a channel, each row is a subject,
        and each cell is either 1 or 0 denoting if that channel
        was marked bad for the given subject
    """

    # construct dictionary that will turn into bad ch info dataframe
    bad_ch_info = {}
    for ch in ch_names:
        bad_ch_info[ch] = []
    bad_ch_info['participant_id'] = []

    # iterate through subject folders
    subject_dirs = glob.glob('%s/sub-*' % pipeline_root)
    for sd in subject_dirs:

        bad_ch_info['participant_id'].append(sd[-9:])

        # retrieve the bad channels
        f = '%s/%s_task-msit_stimulus-epo.fif' % (sd, sd[-9:])
        epochs = read_epochs(f, verbose=False, preload=False)
        bad_chs = epochs.info['bads']

        # fill in indicators for each channel if bad or not
        for ch in ch_names:
            if ch in bad_chs:
                bad_ch_info[ch].append(1)
            else:
                bad_ch_info[ch].append(0)

    return pd.DataFrame(bad_ch_info)


def extract_bad_epochs_group_info(pipeline_root, ch_names):
    """
    Extracts bad epochs summary information across subjects.

    Parameters
    ----------
    pipeline_root: str
        Filepath to directory containing eeg preprocessing results
    ch_names: list
        List of channel names in the data

    Returns
    -------
    DataFrame
        Dataframe with columns denoting subject, epoch type, and number of
        bad epochs for the given subject and epoch type
    """

    # construct dictionary that will turn into bad ch info dataframe
    bad_epoch_info = {}
    bad_epoch_info['participant_id'] = []
    bad_epoch_info['epoch_type'] = []
    bad_epoch_info['num_bad'] = []

    # iterate through subject folders
    subject_dirs = glob.glob('%s/sub-*' % pipeline_root)
    for sd in subject_dirs:

        bad_epoch_info['participant_id'] += [sd[-9:]] * 2

        # retrieve the bad channels
        for typ in ['stimulus', 'response']:
            f = '%s/%s_task-msit_%s_ar.pkl' % (sd, sd[-9:], typ)
            with open(f, 'r') as fid:
                ar = pickle.load(fid)

            num_bad_epochs = len(ar.bad_epochs_idx)
            bad_epoch_info['epoch_type'].append(typ)
            bad_epoch_info['num_bad'].append(num_bad_epochs)

    df = pd.DataFrame(bad_epoch_info)
    return df.sort_values(by='num_bad')


def plot_bad_chs_group_summary(bad_ch_info, ch_names):
    """
    Makes a summary plot of bad channels across all subjects including:
        - heatmap of channels by subject indicating if channel was marked bad
        - Bad channel % for each participant
        - % of subjects with channel marked bad for each subject

    Parameters
    ----------
    bad_ch_info: DataFrame
        Bad channel summary dataframe as extracted by extract_bad_ch_group_info
    ch_names: list
        List of channel names in the data

    Returns
    -------
    Matplotlib Figure Object
        The figure object containing the bad channels group summary plot
    """

    fig = plt.figure(figsize=(24, 12))
    sns.set(font_scale=1, style='white')

    # extract bad channel data
    ch_data = bad_ch_info.loc[:, bad_ch_info.columns != 'participant_id']
    bad_ch_freqs = ch_data.as_matrix().mean(axis=1)
    subjects = bad_ch_info.participant_id.as_matrix()

    # order by # bad channels
    freq_sorted_ix = np.argsort(bad_ch_freqs)
    ch_data = ch_data.iloc[freq_sorted_ix, :]
    subjects = subjects[freq_sorted_ix]
    bad_ch_freqs = bad_ch_freqs[freq_sorted_ix]

    # Plot heatmap of bad channel indicators
    ax = plt.subplot2grid((2, 2), loc=(0, 0), rowspan=2)
    sns.heatmap(ch_data.T, xticklabels=subjects,
                yticklabels=list(ch_data.columns), ax=ax)

    # Plot bad channel percentages
    ax = plt.subplot2grid((2, 2), loc=(0, 1))
    sns.barplot(x=subjects, y=bad_ch_freqs, ax=ax, order=subjects)
    ax.set_xticklabels(subjects, rotation=90)
    ax.set_ylabel('Bad Channel %')

    # Plot bad ch participant frequency counts
    ax = plt.subplot2grid((2, 2), loc=(1, 1))

    ch_freqs = []
    for ch in ch_names:
        ch_freqs.append(bad_ch_info[ch].mean())
    ax.bar(np.arange(len(ch_freqs)), ch_freqs)
    ax.set_xticks(np.arange(len(ch_freqs)) + .5)
    ax.set_xticklabels(CH_NAMES, rotation=90)
    ax.set_ylabel('% of Participants')

    sns.despine()
    plt.tight_layout()
    return fig


def surface_laplacian(inst, x, y, z, inst_type, m='auto', leg_order='auto',
                      smoothing=1e-5):
    """Compute Surface Laplacian using Perrin 1989 Spherical Splines Method.

    This is a translation of Mike Cohen's surface laplacian matlab code from
    the textbook Analyzing Neural Time Series Data.

    Parameters
    ----------
    inst : instance of MNE Epochs or Raw or Evoked
        The data to be transformed
    x: numpy array
        the x coordinates of the channels
    y: numpy array
        the y coordinates of the channels
    z: numpy array
        the z coordinates of the channels
    inst_type: str
        the type of MNE object. epochs or raw or evoked

    Returns
    -------
    lap_inst : instance of MNE Epochs or Raw or Evoked
        The modified instance with surface laplacian transformed EEG data.

    """

    inst = inst.copy()

    # Get indices of EEG data
    eeg_ix = pick_types(inst.info, eeg=True, meg=False, exclude=[])
    eeg_ch_names = np.array(inst.ch_names)[eeg_ix]

    # drop out bads from the spherical locations
    good_ix = np.array([ix for ix in range(len(eeg_ch_names))
                        if eeg_ch_names[ix] not in inst.info['bads']])
    x = x[good_ix]
    y = y[good_ix]
    z = z[good_ix]
    eeg_ix = pick_types(inst.info, eeg=True, meg=False, exclude='bads')

    if inst_type == 'evoked':
        data = inst.data[eeg_ix, :]
    elif inst_type == 'epochs':
        data = inst._data[:, eeg_ix, :]
        num_epochs, num_chs, num_time_points = data.shape
        data = np.concatenate([data[i, :, :] for i in range(num_epochs)],
                              axis=-1)

    # Compute the G & H matrices
    G, H, _ = _compute_GH(x, y, z)

    # Compute the surface laplacian transform
    lap = _compute_perrin_surf_laplacian(data, G, H, smoothing)

    # Insert the laplacian transformed data back into the MNE object
    if inst_type == 'evoked':
        inst.data[eeg_ix, :] = lap
    elif inst_type == 'epochs':
        lap = np.split(lap, num_epochs, axis=-1)
        lap = np.concatenate([d[np.newaxis, :, :] for d in lap], axis=0)
        inst._data[:, eeg_ix, :] = lap
    return inst


def _transform_unit_sphere(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    max_r = np.max(r)
    x /= max_r
    y /= max_r
    z /= max_r
    return x, y, z


@np.vectorize
def _vector_legendre(n, x):
    return lpmn(0, n, x)[0][-1, -1]


def _compute_GH(x, y, z):

    num_electrodes = len(x)
    if num_electrodes > 90:
        m = 3
        leg_order = 40
    else:
        m = 4
        leg_order = 20

    x, y, z = _transform_unit_sphere(x, y, z)
    G = np.zeros((num_electrodes, num_electrodes))
    H = np.zeros((num_electrodes, num_electrodes))
    cos_dist = np.zeros((num_electrodes, num_electrodes))

    # Compute the Cosine Distance Between Electrodes
    for i in range(num_electrodes):
        for j in range(i + 1, num_electrodes):
            cos_dist[i, j] = 1 - (((x[i] - x[j])**2 + (y[i] - y[j])**2 +
                                   (z[i] - z[j])**2) / 2.)

    cos_dist = cos_dist + cos_dist.T + np.identity(num_electrodes)

    # Compute the Legendre Polynomials
    leg_poly = np.zeros((leg_order, num_electrodes, num_electrodes))
    for n in range(1, leg_order + 1):
        leg_poly[n - 1, :, :] = _vector_legendre(n, cos_dist)

    for i in range(num_electrodes):
        for j in range(i, num_electrodes):

            g = 0
            h = 0

            for n in range(1, leg_order + 1):
                g += ((2 * n + 1) * leg_poly[n - 1, i, j]) / ((n * (n + 1)) ** m)
                h -= ((2 * n + 1) * leg_poly[n - 1, i, j]) / ((n * (n + 1)) ** (m - 1))

            G[i, j] = g / (4 * np.pi)
            H[i, j] = -h / (4 * np.pi)

    G += G.T
    H += H.T

    # ??
    G -= np.identity(num_electrodes) * G[0, 0] / 2.
    H -= np.identity(num_electrodes) * H[0, 0] / 2.

    return G, H, cos_dist


def _compute_perrin_surf_laplacian(data, G, H, smoothing):

    num_electrodes, num_time_points = data.shape

    Gs = G + np.identity(num_electrodes) * smoothing

    GsinvS = np.sum(inv(Gs), axis=1)

    # Python lstsq gives different answer than MATLAB \
    d = np.linalg.lstsq(Gs.T, data)[0].T

    tmp = np.sum(d, axis=1) / np.sum(GsinvS)
    C = d - tmp[:, np.newaxis].dot(GsinvS[np.newaxis, :])

    surf_lap = C.dot(H.T).T

    return surf_lap


def plot_waveforms(trans_type, epo_type, data, stats, ch, subject, ax,
                   time, behavior, colors=['#e41a1c', '#377eb8'],
                   threshold=.05):
    """
    Plots ERP or band power waveforms split by condition with
    statistical significance shading.

    Parameters
    ----------
    trans_type: str
        The type of transform. laplacian or base
    epo_type: str
        The type of epoch. stimulus or response
    data: dict
        Dictionary containing the waveform data and metadata
    stats: dict
        Dictionary containing the statistical info
    ch: str
        The channel name to plot
    subject: str
        The subject whose data to plot
    ax: Matplotlib axis
        The axis to plot to
    time: float
        The highlighted time point to plot
    behavior: Pandas DataFrame
        The subject behavioral data
    colors: list
        List of colors to use for each condition
    threshold: float
        The statistical significance threshold


    Returns
    -------
    ax: Matplotlib Axis
        The axis with the above plotted
    """

    # remove bad trials from behavior
    exclusions = ['fast_rt', 'no_response', 'error', 'post_error']
    behavior = behavior.loc[np.where(np.sum(behavior[exclusions],
                                            axis=1) == 0)[0], :]
    if subject != 'group':
        behavior = behavior.loc[behavior.participant_id == subject, :]

    # load data
    chs = data['chs']
    evo = data['data']
    conditions = data['conditions']
    times = data['times']
    subjects = data['subjects']

    # reduce to channel of interest
    ch_ix = list(chs).index(ch)
    evo = evo[:, :, ch_ix, :].squeeze()

    # reduce to subject of interest
    if subject != 'group':
        sub_ix = list(subjects).index(subject)
        evo = evo[:, sub_ix, :].squeeze()

    # plot tfce perm significance shading
    if stats is not None:
        pvals = stats['p_vals'][ch_ix, :]
        for i, p in enumerate(pvals):
            if p < threshold:
                ax.axvline(times[i], color='k', alpha=.1, label='_nolegend_',
                           linewidth=3)

    for i, c in enumerate(conditions):

        con_data = evo[i, :]

        if subject == 'group':
            mean = con_data.mean(axis=0)
            std_err = con_data.std(axis=0) / np.sqrt(con_data.shape[0])
        else:
            mean = con_data

        ax.plot(times, mean, color=colors[i])

        if subject == 'group':
            ax.fill_between(times, mean - std_err, mean + std_err,
                            alpha=0.5, color=colors[i])

    bottom = ax.get_ylim()[0]
    for i, c in enumerate(conditions):
        if epo_type == 'stimulus':
            rt = behavior[behavior.trial_type == c].response_time
            ax.hist(rt, color=colors[i], alpha=0.2,
                    normed=True, bottom=bottom)

    # x-axis ticks
    if epo_type == 'stimulus':
        ax.set_xticks(np.arange(-.5, 1.8, .25))
        ax.set_xlim((-.5, 1.75))
    else:
        ax.set_xticks(np.arange(-1, 1.1, .25))

    ax.set_xlabel('Time [s]')

    ax.axvline(0, color='k', label='_nolegend_')
    ax.axhline(0, color='k', label='_nolegend_')
    ax.axvline(time, color='k', linestyle='--', label='_nolegend')
    ax.legend(conditions, loc='best')

    return ax


def visualize_erps(deriv_dir, trans_type, epo_type, time_ix,
                   ch, threshold, behavior):
    """
    Plots interactive ERP summary to be used with ipywidgets interact.

    Parameters
    ----------
    deriv_dir: str
        Path to derivatives directory holding data
    trans_type: str
        The type of transform. laplacian or base
    epo_type: str
        The type of epoch. stimulus or response
    time_ix: int
        The time index to plot for the topomap
    ch: str
        The channel name to plot
    behavior: Pandas DataFrame
        The subject behavioral data
    threshold: float
        The statistical significance threshold


    Returns
    -------
    None
        Plots the summary plot
    """

    subject = 'group'
    # load the erp data
    if trans_type == 'laplacian':
        f = '%s/%s/%s_task-msit_%s_laplacian-ave.npz'
    else:
        f = '%s/%s/%s_task-msit_%s-ave.npz'
    data = np.load(f % (deriv_dir, subject, subject, epo_type))

    # load the topo data
    f = f.replace('.npz', '.fif')
    evokeds = read_evokeds(f % (deriv_dir, subject, subject, epo_type),
                           verbose=False)
    evo = combine_evoked(evokeds, [1, -1])

    # load the stats
    f = f.replace('-ave.fif', '_tfce.npz')
    f = f % (deriv_dir, 'group', 'group', epo_type)
    if os.path.exists(f):
        stats = np.load(f)
    else:
        stats = None

    f, axs = plt.subplots(1, 3, figsize=(28, 6))

    time = data['times'][time_ix]
    ax = plot_waveforms(trans_type, epo_type, data, stats, ch, subject,
                        axs[0], time, behavior, threshold=threshold)
    # axis labels
    if trans_type == 'base':
        ax.set_ylabel('$\mu V$')
    else:
        ax.set_ylabel('$\mu V / mm^2$')

    # plot topomap
    if stats is not None:
        mask = stats['p_vals'] < threshold
    else:
        mask = None

    if trans_type == 'laplacian':
        limit = 7
    else:
        limit = 1

    import matplotlib
    matplotlib.rcParams.update({'font.size': 22, 'font.weight': 'bold'})
    evo.plot_topomap(times=time, mask=mask, show=False, show_names=True,
                     axes=axs[1], vmin=-limit, vmax=limit, colorbar=True,
                     size=2, mask_params=dict(markersize=0))
    plt.suptitle('%s %s %s %s-locked ERPS' % (subject, ch, trans_type,
                                              epo_type), fontsize=25, y=1.1)
    sns.despine()
    plt.show()


def visualize_band_power(deriv_dir, trans_type, epo_type, time_ix, band,
                         ch, threshold, behavior):
    """
    Plots interactive band power summary to be used with ipywidgets interact.

    Parameters
    ----------
    deriv_dir: str
        Path to derivatives directory holding data
    trans_type: str
        The type of transform. laplacian or base
    epo_type: str
        The type of epoch. stimulus or response
    time_ix: int
        The time index to plot for the topomap
    band: str
        The power band to plot. theta or alpha or beta
    ch: str
        The channel name to plot
    behavior: Pandas DataFrame
        The subject behavioral data
    threshold: float
        The statistical significance threshold


    Returns
    -------
    None
        Plots the summary plot
    """

    subject = 'group'
    # load the erp data
    if trans_type == 'laplacian':
        f = '%s/%s/%s_task-msit_%s_laplacian_band-tfr.npz'
    else:
        f = '%s/%s/%s_task-msit_%s_band-tfr.npz'
    data = np.load(f % (deriv_dir, subject, subject, epo_type))
    band_data = data[band]

    # create the topo data
    if trans_type == 'laplacian':
        f = '%s/%s/%s_task-msit_%s_laplacian-ave.fif'
    else:
        f = '%s/%s/%s_task-msit_%s-ave.fif'
    evokeds = read_evokeds(f % (deriv_dir, subject, subject, epo_type),
                           verbose=False)
    evokeds[0].data = band_data[0].squeeze().mean(axis=0)
    evokeds[1].data = band_data[1].squeeze().mean(axis=0)
    evo = combine_evoked(evokeds, [1, -1])

    # load the stats
    if trans_type == 'laplacian':
        f = '%s/group/group_task-msit_%s_%s_laplacian_tfce.npz'
    else:
        f = '%s/group/group_task-msit_%s_%s_tfce.npz'
    f = f % (deriv_dir, epo_type, band)
    if os.path.exists(f):
        stats = np.load(f)
    else:
        stats = None

    f, axs = plt.subplots(1, 3, figsize=(28, 6))

    time = data['times'][time_ix]
    data = {'times': data['times'],
            'data': data[band],
            'chs': data['chs'],
            'conditions': data['conditions'],
            'subjects': data['subjects']}
    ax = plot_waveforms(trans_type, epo_type, data, stats, ch, subject,
                        axs[0], time, behavior, threshold=threshold)
    ax.set_ylabel('DB Change from Baseline')

    # plot topomap
    if stats is not None:
        mask = stats['p_vals'] < threshold
    else:
        mask = None

    if trans_type == 'laplacian':
        limit = 1
    else:
        limit = 1

    import matplotlib
    matplotlib.rcParams.update({'font.size': 22, 'font.weight': 'bold'})
    evo.plot_topomap(times=time, mask=mask, show=False, show_names=True,
                     axes=axs[1], vmin=-limit, vmax=limit, colorbar=True,
                     size=2, mask_params=dict(markersize=0),
                     scalings=dict(eeg=1), units='mV')
    plt.suptitle('%s %s %s %s-locked %s Band Power' % (subject, ch, trans_type,
                                                       epo_type, band),
                 fontsize=25, y=1.1)
    sns.despine()
    plt.show()


def baseline_normalize(tfr, baseline, func=np.mean, method='classic'):
    """
    Baseline normalizes power data using a decibel normalization.

    Parameters
    ----------
    tfr: MNE EpochsTFR object
        MNE object holding the raw time frequency power
    baseline: tuple or numpy array
        If tuple, it provides the bounds for the given data that should
        be used as baseline. If numpy array, it contains the baseline data
        itself. The baseline data gets passed in for the response epochs
        since the stimulus epochs contain the baseline period.
    func: numpy summary function
        The summary function. Should be either np.mean or np.median
    method: str
        The baseline normalization method. classic or grandchamp. grandchamp
        refers to the technique discussed in Grandchamp and Delorme, 2013.
        classic refers to the standard decibel normalization technique.

    Returns
    -------
    tfr: MNE EpochsTFR object
        MNE object holding the raw time frequency power
    baseline: numpy array
        The extracted baseline data
    """

    data = tfr.data
    times = tfr.times

    if method == 'grandchamp':
        # scale by individual trial median
        trial_summ = func(data, axis=-1)[:, :, :, np.newaxis]
        data /= trial_summ

    # collapse across trials
    data = func(data, axis=0)

    # divide by baseline period collapsed across time
    if len(baseline) == 2:
        baseline_ix = np.where(np.logical_and(times >= baseline[0],
                                              times <= baseline[1]))[0]
        baseline = data[:, :, baseline_ix]

    baseline_ave = func(baseline, axis=-1)[:, :, np.newaxis]
    data /= baseline_ave

    # decibel transformation
    data = 10 * np.log10(data)

    # stick back into mne object
    tfr = tfr.average()
    tfr.data = data

    return tfr, baseline


def power_heatmap(power, ax, lim, rts=None, rt_colors=None):
    """
    Plots a TFR power heatmap.

    Parameters
    ----------
    power: MNE AverageTFR object
        The power data to be plotted
    ax: Matplotlib axis
        The axis to plot to
    lim: float
        The limit for the the colorbar
    rts: list
        list of response times for each condition
    rt_colors: list
        list of colors to use for each condition when plotting the rts

    Returns
    -------
    ax: Matplotlib Axis
        The axis with the above plotted
    """

    # heatmap data
    ax.contourf(power.times, np.arange(len(power.freqs)),
                power.data.squeeze(), 40,
                cmap='jet', vmin=-lim, vmax=lim)

    # frequency axis labeling
    freq_ticks = [2, 4, 7.5, 12.5, 30, 60]
    ys = []
    for ft in freq_ticks:
        ys.append(np.argmin(np.abs(power.freqs - ft)))
        ax.axhline(ys[-1], color='k', linestyle='--')
    ax.set_yticks(ys)
    ax.set_yticklabels(freq_ticks)

    # axis flourishes
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.axvline(0, color='k')

    # add rt dist to bottom
    if rts:
        for rt, col in zip(rts, rt_colors):
            ax.hist(rt, color=col, normed=True, bottom=-4, alpha=0.7)

    return ax


def visualize_tfr_heatmap(deriv_dir, subject, ch, lim, trans_type, behavior,
                          config):
    """
    Plots interactive tfr heatmaps for both conditions and epoch type.

    Parameters
    ----------
    deriv_dir: str
        Path to derivatives directory holding data
    subject: str
        subjectcode of subject to plot
    ch: str
        The channel name to plot
    lim: float
        The limit for the the colorbar
    trans_type: str
        The type of transform. laplacian or base
    behavior: Pandas DataFrame
        The subject behavioral data
    config: dict
        Configuration dictionary holding analysis parameters


    Returns
    -------
    None
        Plots the summary plot
    """

    exclusions = ['fast_rt', 'no_response', 'error', 'post_error']
    sub_behavior = behavior.loc[np.where(np.sum(behavior[exclusions],
                                                axis=1) == 0)[0], :]
    if subject != 'group':
        sub_behavior = sub_behavior.loc[sub_behavior.participant_id == subject,
                                        :]
    sns.set(style='white', font_scale=2)
    plt.close('all')

    fig, axs = plt.subplots(2, 3, figsize=(24, 16))

    for i, epo_type in enumerate(config['epochs_info']['epo_types']):

        if trans_type == 'laplacian':
            f = '%s/%s/%s_task-msit_%s_laplacian_norm-tfr.h5'
        else:
            f = '%s/%s/%s_task-msit_%s_norm-tfr.h5'
        tfrs = read_tfrs(f % (deriv_dir, subject, subject, epo_type))

        for j, c in enumerate(config['event_types']):

            power = tfrs[j]
            power.pick_channels([ch])
            ax = axs[i, j]
            rts = [sub_behavior[sub_behavior.trial_type == c].response_time]
            rt_colors = [config['colors'][j]]
            if epo_type == 'stimulus':
                ax = power_heatmap(power, ax, lim, rts, rt_colors)
            else:
                ax = power_heatmap(power, ax, lim)
            ax.set_title('%s %s-locked' % (c, epo_type))

        ax = axs[i, 2]
        power = tfrs[0] - tfrs[1]
        power.pick_channels([ch])
        rts = [sub_behavior[sub_behavior.trial_type == c].response_time
               for c in config['event_types']]
        rt_colors = [config['colors'][0], config['colors'][1]]
        if epo_type == 'stimulus':
            ax = power_heatmap(power, ax, lim, rts, rt_colors)
        else:
            ax = power_heatmap(power, ax, lim)

    plt.tight_layout()
    plt.subplots_adjust(top=.92)
    plt.suptitle('%s %s TFR Heatmaps (Color Limit: +- %s)' % (subject,
                                                              ch, lim),
                 fontsize=24)
    plt.show()
