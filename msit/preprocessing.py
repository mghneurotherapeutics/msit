from grabbit import Layout
import os
import glob
import pickle
from mne.io import Raw
from mne import read_epochs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from mne.preprocessing import create_eog_epochs, create_ecg_epochs

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


def make_eeg_prep_derivatives_folder(datapath):
    """ Creates a skeleton file structure for the eeg preprocessing pipeline
    built as part of this project.

    :param datapath: Path to a BIDS-compatible dataset.
    :type datapath: str.

    :returns: None -- Creates the skeleton within the folder provided
    by datapath.
    """

    pipeline_root = '%s/derivatives/eeg_preprocessing' % datapath
    if not os.path.exists(pipeline_root):
        os.makedirs(pipeline_root)

    layout = Layout(datapath, '%s/grabbit_config.json' % datapath)
    subjects = layout.get(target='subject', modality='eeg', return_type='id')

    for subject in subjects:
        for step in ['ica', 'epochs', 'autoreject']:
            step_path = '%s/%s/%s' % (pipeline_root, subject, step)
            if not os.path.exists(step_path):
                os.makedirs(step_path)

        bad_chs_file = '%s/%s/bad_chs.txt' % (pipeline_root, subject)
        open(bad_chs_file, 'a').close()


def manually_mark_bad_channels(subject_file):
    """ Plots the raw data for a subject and allows manual updating of bad
    channels via interactive plot.

    """

    raw = Raw(subject_file.filename, verbose=False, preload=True)

    raw = load_bad_channels(subject_file.subject, raw)

    raw.plot(title=subject_file.subject, show=True, block=True)

    write_bad_channels(subject_file.subject, raw)


def load_bad_channels(subject, mne_object):

    bad_ch_file = layout.get(subject=subject, modality='eeg',
                             derivative='eeg_preprocessing',
                             extensions='bad_chs.txt')[0].filename

    mne_object.load_bad_channels(bad_ch_file)
    return mne_object


def write_bad_channels(subject, mne_object):

    bad_ch_file = layout.get(subject=subject, modality='eeg',
                             derivative='eeg_preprocessing',
                             extensions='bad_chs.txt')[0].filename
    with open(bad_ch_file, 'w') as fid:
        for bad in mne_object.info['bads']:
            fid.write('%s\n' % bad)


# Epochs

def load_epochs(subject, layout):

    epoch_files = layout.get(subject=subject,
                             derivative='eeg_preprocessing',
                             extensions='uncleaned-epo.fif')
    ar_files = layout.get(subject=subject,
                          derivative='eeg_preprocessing',
                          extensions='ar.pkl')
    epochs = [read_epochs(f.filename, verbose=False) for f in epoch_files]
    ar = [pickle.load(open(f.filename, 'r')) for f in ar_files]

    # apply autoreject correction to the data
    ar_epochs = [ar.transform(epoch) for epoch in epochs]

    return epochs, ar_epochs


def plot_evoked_butterfly(epochs, config, ylim=(-15, 15)):

    fig, axs = plt.subplots(1, 2, figsize=(24, 8))
    for i, epo_type in enumerate(config['epoch_types']):
        epochs[i].average().plot(ylim=ylim, spatial_colors=True,
                                 axes=axs[i])
        axs[i].set_title('%s locked' % epo_type)
    return fig


def visually_verify_epochs(subject, epochs, config, ylim=(-15, 15)):
    """ Iteratively plots epochs with and without autoreject cleaning
    and average re-referencing and allows bad channel selection until
    satisfied.

    :param epochs: Non-autorejected epochs.
    :type epochs: MNE Epochs object.
    :param ar_epochs: Auto-rejected epochs
    :type ar_epochs: MNE Epochs object.
    :param ylim: y axis limits for butterfly plots.
    :type ylim: tuple of two floats

    :returns: ar_epochs -- The autorejected epochs with updated bad channels.
    """

    # crop for visualization
    epochs = [epochs[i].copy().crop(config['epoch_times'][i][0],
                                    config['epoch_times'][i][1])
              for i in range(len(epochs))]

    ok = 'n'
    while ok != 'y':

        # average re-reference
        avg = [epoch.copy().set_eeg_reference().apply_proj()
               for epoch in epochs]

        # visualize evoked butterfly
        plot_evoked_butterfly(avg, config)
        plt.suptitle(subject)

        # plot corrected epochs for bad channel updating
        epochs[0].plot(block=True)
        epochs[1].info['bads'] = epochs[0].info['bads']

        ok = raw_input('OK to move on? (enter y or n):')

    return epochs[0].info['bads']


def extract_bad_ch_group_info(pipeline_root, ch_names):

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
        f = '%s/epochs/%s_stimulus_cleaned-epo.fif' % (sd, sd[-9:])
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
            f = '%s/autoreject/%s_%s_ar.pkl' % (sd, sd[-9:], typ)
            with open(f, 'r') as fid:
                ar = pickle.load(fid)

            num_bad_epochs = len(ar.bad_epochs_idx)
            bad_epoch_info['epoch_type'].append(typ)
            bad_epoch_info['num_bad'].append(num_bad_epochs)

    df = pd.DataFrame(bad_epoch_info)
    return df.sort_values(by='num_bad')


def plot_epoched_ica_artifact_info(raw, ica, subject, pipeline_root):

    for i, ch in enumerate(['VEOG', 'HEOG', 'ECG']):
        # create artifact aligned epochs
        if 'EOG' in ch:
            epochs = create_eog_epochs(raw, ch_name=ch)
        else:
            epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)

        # plot data before and after ica epoched to artifact events
        f = ica.plot_overlay(epochs.average())
        f.suptitle('%s %s' % (subject, ch))
        f.savefig('%s/%s/ica/%s_%s_before_after.png' % (pipeline_root,
                                                        subject,
                                                        subject,
                                                        ch))

        # plot source time courses epoched to artifact events
        f = ica.plot_sources(epochs.average(), exclude=ica.exclude)
        f.suptitle('%s %s' % (subject, ch))
        f.savefig('%s/%s/ica/%s_%s_components.png' % (pipeline_root,
                                                      subject,
                                                      subject,
                                                      ch))


def verify_events(events, behavior, epo_type):
    num_eeg_events = events.shape[0]
    if epo_type == 'response':
        num_behavior_events = behavior[behavior.response_time.notnull()].shape[0]
    else:
        num_behavior_events = behavior.shape[0]

    if num_eeg_events != num_behavior_events:
        raise ValueError('Mismatching EEG & Behavior %s Events: %s EEG Events, %s Behavior Events' % (epo_type, num_eeg_events, num_behavior_events))


def handle_event_exceptions(subject, epo_type, eeg_events, behavior_events):

    if subject == 'sub-pp012':
        if epo_type == 'stimulus':
            behavior_events = behavior_events[1:]
        elif epo_type == 'response':
            eeg_events = eeg_events[1:, :]
    else:
        raise ValueError('Unhandled Subject Exception.')

    return eeg_events, behavior_events


def epoch_baseline_correct(epochs, baseline):

    # extract data out from mne objects
    epochs_data = epochs.get_data()

    # collapse baseline along time
    baseline = baseline.mean(axis=-1)

    # subtract baseline average from every time point
    epochs_data = np.array([arr - baseline.T for arr in epochs_data.T]).T

    # replace the data in the epochs object with baseline corrected data
    epochs._data = epochs_data

    return epochs


def plot_autoreject_summary(ar, subject, epo_type, pipeline_root, ch_names):
    plt.figure(figsize=(24, 12))

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

    # save out figure
    plt.tight_layout()
    plt.suptitle('%s %s autoreject summary' % (subject, epo_type), fontsize=24)
    plt.subplots_adjust(top=0.93)
    plt.savefig('%s/%s/autoreject/%s_%s_ar_summary.png' % (pipeline_root,
                                                           subject,
                                                           subject,
                                                           epo_type))
    plt.close('all')


def plot_bad_chs_group_summary(bad_ch_info):

    fig = plt.figure(figsize=(24, 12))

    ch_data = bad_ch_info.loc[:, bad_ch_info.columns != 'participant_id']
    bad_ch_freqs = ch_data.as_matrix().mean(axis=1)
    subjects = bad_ch_info.participant_id.as_matrix()

    freq_sorted_ix = np.argsort(bad_ch_freqs)
    ch_data = ch_data.iloc[freq_sorted_ix, :]
    subjects = subjects[freq_sorted_ix]

    # Plot heatmap of bad channel indicators
    ax = plt.subplot2grid((2, 2), loc=(0, 0), rowspan=2)
    sns.heatmap(ch_data.T, xticklabels=subjects, ax=ax)

    # Plot distribution of bad channel percentages
    ax = plt.subplot2grid((2, 2), loc=(0, 1))
    sns.distplot(bad_ch_freqs, norm_hist=False, kde=False, ax=ax)
    ax.set_xlabel('Bad Channel %')
    ax.set_ylabel('# Participants')

    # Plot bad ch participant frequency counts
    ax = plt.subplot2grid((2, 2), loc=(1, 1))

    ch_freqs = []
    for ch in CH_NAMES:
        ch_freqs.append(bad_ch_info[ch].mean())
    ax.bar(np.arange(len(ch_freqs)), ch_freqs)
    ax.set_xticks(np.arange(len(ch_freqs)) + .5)
    ax.set_xticklabels(CH_NAMES, rotation=90)
    ax.set_ylabel('% of Participants')

    sns.despine()
    plt.tight_layout()
    return fig


def extract_itis(df):

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

    df['trial_type_sequence'] = 'cc'

    for seq in ['ci', 'ic', 'ii']:
        prev_ix = np.roll(df.trial_type.str.startswith(seq[0]), 1)
        curr_ix = df.trial_type.str.startswith(seq[1])
        seq_ix = np.where(np.logical_and(prev_ix, curr_ix))[0]
        df.loc[seq_ix, 'trial_type_sequence'] = seq

    df.loc[0, 'trial_type_sequence'] = np.NaN

    return df


def encode_post_error(df):

    df['post_error'] = 0
    post_error_ix = np.where(np.roll(df.error, 1) == 1)[0]

    # Ignore potential wraparound from last trial
    if len(post_error_ix) != 0 and post_error_ix[0] == 0:
        post_error_ix = np.delete(post_error_ix, 0)

    df.loc[post_error_ix, 'post_error'] = 1
    return df


def encode_target_location(df):

    rs = []
    for s, cr in zip(df.stimulus, df.correct_response):
        if int(s[0]) == cr:
            rs.append('left')
        elif int(s[2]) == cr:
            rs.append('right')
        else:
            rs.append('middle')

    df['target_location'] = rs
    return df

