from grabbit import Layout
import os
import glob
import pickle
from mne.io import Raw
from mne import read_epochs
from mne import Epochs
from mne import find_events
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from mne.preprocessing import create_eog_epochs, create_ecg_epochs, ICA
from utils import CH_NAMES

sns.set(style='whitegrid', font_scale=1.5)


# Behavior Functions


def create_tidy_subject_df(filename, subject, modality, fast_rt_thresh=.2):
    """
    Creates a cleaned tidy subject dataframe. This includes removing fixation
    trials and encoding iti in separate column, encoding error and posterror trials,
    encoding n-1 trial sequences, and marking fast rts.

    Parameters
    ----------
    filename : string 
        filepath to the subject tsv file
    subject : string
        participant id
    modality : string
        the modality of the data (eeg or fmri)
    fast_rt_thresh: float, optional
        threshold for denoting fast rt in seconds, by default 200 ms

    Returns
    -------
    DataFrame 
        tidy subject pandas DataFrame 
    """
    
    df = pd.read_csv(filename, sep='\t', na_values='n/a')
    df['participant_id'] = subject
    df['modality'] = modality
    
    # extract out itis and remove fixation triasl
    df = extract_itis(df)
    df = df[df.trial_type != 'fixation'].reset_index(drop=True)
    
    # encode error and post-error trials
    df['error'] = 1 - df.response_accuracy
    df = encode_post_error(df)
    
    # encode no-response trials
    df['no_response'] = 0
    df.loc[df.chosen_response.isnull(), 'no_response'] = 1

    # Encode condition sequence effects
    df = encode_trial_type_sequence(df)

    # encode fast response times
    df['fast_rt'] = np.where(df.response_time < fast_rt_thresh, 1, 0)

    # encode trial numbers
    df['trial'] = np.arange(df.shape[0]) + 1
    
    return df


def extract_itis(df):
    """
    Extracts itis from a subject's behavior dataframe in a separate column.

    Parameters
    ----------
    df : DataFrame 
        the subject's dataframe with fixation trials 

    Returns
    -------
    DataFrame 
        the subject's dataframe with iti column added 
    """

    df['iti'] = 0

    # Get the indices where there was a previous iti
    iti_ix = np.where(np.roll(df.trial_type, 1) == 'fixation')[0]

    # Get the iti duration values
    itis = np.array(df[df.trial_type == 'fixation'].duration)

    # Ignore the wrap around
    if df.trial_type.as_matrix()[-1] == 'fixation':
        itis = np.delete(itis, 0)
        iti_ix = np.delete(iti_ix, 0)

    # update the datframe
    df.loc[iti_ix, 'iti'] = itis

    return df


def encode_trial_type_sequence(df):
    """
    Extracts n-1 trial type sequences from a subject's behavior dataframe 
    in a separate column

    Parameters
    ----------
    df : DataFrame 
        the subject's dataframe

    Returns
    -------
    DataFrame 
        the subject's dataframe with n-1 trial type sequence column added 
    """

    df['trial_type_sequence'] = 'cc'

    for seq in ['ci', 'ic', 'ii']:
        prev_ix = np.roll(df.trial_type.str.startswith(seq[0]), 1)
        curr_ix = df.trial_type.str.startswith(seq[1])
        seq_ix = np.where(np.logical_and(prev_ix, curr_ix))[0]
        df.loc[seq_ix, 'trial_type_sequence'] = seq

    df.loc[0, 'trial_type_sequence'] = np.NaN

    return df


def encode_post_error(df):
    """
    Extracts trials immediately following an error in a separate column

    Parameters
    ----------
    df : DataFrame 
        the subject's dataframe

    Returns
    -------
    DataFrame 
        the subject's dataframe with post_error column added 
    """

    df['post_error'] = 0
    post_error_ix = np.where(np.roll(df.error, 1) == 1)[0]

    # Ignore potential wraparound from last trial
    if len(post_error_ix) != 0 and post_error_ix[0] == 0:
        post_error_ix = np.delete(post_error_ix, 0)

    df.loc[post_error_ix, 'post_error'] = 1
    return df


# EEG Functions


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
    Syncs the behavioral and eeg events by removing non-response trials, handling
    misalignment between the eeg and behavior events, and encoding trial
    type in the eeg events.
    
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
        the updated events array aligned with the behavior, with non-response trials
        removed, and with trial type encoded
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
    behavior.loc[:, 'exclude'] = np.where(np.sum(behavior[exclude], axis=1) > 0, 1, 0)
    keep_ix = np.array(behavior.exclude == 0)
    behavior = behavior[keep_ix]
    eeg_events = eeg_events[keep_ix, :]
        
    # encode trial type in the eeg events
    behavior_events = np.array(behavior.trial_type.astype('category').cat.codes)
    eeg_events[:, -1] = behavior_events
    
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
        MNE epochs object containing the baseline data to be used for correction 
    
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
        dictionary mapping EOG and ECG channel names to channel types (eog or ecg)
    
    Returns
    -------
    MNE ICA Object 
        MNE ICA object containing the fitted ICA results and auto-marked bad components 
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
        MNE ICA object containing the ICA whose components are going to be checked
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
        MNE ICA object containing the ICA whose components are going to be visualized 
    raw: MNE Raw Object 
        MNE raw object containing the raw data the ICA was fit to
    aux_chs: dict 
        dictionary mapping EOG and ECG channel names to channel types (eog or ecg)
    
    
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
        List of starting times to crop the data to. Must be the same length as epochs.
    tmaxs: list
        List of ending times to crop the data to. Must be the same length as epochs.
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
      - Heatmap of cross validation scores for the consensus % and # of interpolation
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
    sns.heatmap(ch_data.T, xticklabels=subjects, yticklabels=list(ch_data.columns),
                ax=ax)

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
