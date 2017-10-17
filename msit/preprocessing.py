from grabbit import Layout
import os
import pickle
import matplotlib.pyplot as plt
from mne.io import Raw
from mne import read_epochs


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


def load_epochs(subject, epo_type, layout):

    epoch_file = layout.get(subject=subject,
                            derivative='eeg_preprocessing',
                            extensions='%s-epo.fif' % epo_type)
    ar_file = layout.get(subject=subject,
                         derivative='eeg_preprocessing',
                         extensions='%s_ar.pkl' % epo_type)
    epochs = read_epochs(epoch_file[0].filename, verbose=False)
    with open(ar_file[0].filename, 'r') as inp:
        ar = pickle.load(inp)

    # apply autoreject correction to the data
    ar_epochs = ar.transform(epochs)

    return epochs, ar_epochs


def visually_verify_epochs(epochs, ar_epochs, ylim=(-15, 15)):
    """ Iteratively plots epochs with and without autoreject cleaning
    and average re-referencing and allows bad channel selection until
    satisfied.

    :param epochs: Non-autorejected epochs.
    :type epochs: MNE Epochs object.
    :param ar_epochs: Auto-rejected epochs
    :type ar_epochs: MNE Epochs object.
    :param ylim: y axis limits for butterfly plots.
    :type ylim: tuple of two floats

    :returns: ar_epochs -- The
    """

    epochs.crop(epochs.times[0] + 1, epochs.times[-1] - 1)

    ok = 'n'
    while ok != 'y':

        # crop and average re-reference for visualization
        ar_viz_epochs = ar_epochs.copy().crop(ar_epochs.times[0] + 1,
                                              ar_epochs.times[-1] + 1)

        # before-after autoreject comparison butterfly plots
        epochs.plot(ylim=ylim, spatial_colors=True)
        plt.title('No AutoReject')
        ar_viz_epochs.average().plot(ylim=ylim, spatial_colors=True)
        plt.title('AutoReject')

        # plot corrected epochs for bad channel updating
        ar_epochs.plot(block=True)

        ok = raw_input('OK to move on? (enter y or n):')

    return ar_epochs

