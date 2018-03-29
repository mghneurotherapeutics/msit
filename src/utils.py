import numpy as np
import pandas as pd
import pickle

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


def exclude_subjects(subjects, mod, reset=False):
    """
    :param subjects: List of participants to be excluded.
    :type subjects: list
    :param mod: The modality to exclude for (behavior_eeg, behavior_fmri,
                fmri, or eeg).
    :type mod: string
    :param reset: Whether to clear existing exclusions prior to updating.
    :type reset: boolean

    :returns: None -- Updates the participants.tsv file with the exclusions
    """

    f = '../data/participants.tsv'
    sub_info = pd.read_csv(f, sep='\t')
    if reset:
        sub_info['%s_exclude' % mod] = 0
    sub_info.loc[sub_info.participant_id.isin(subjects),
                 '%s_exclude' % mod] = 1
    sub_info.to_csv(f, sep='\t', index=False)


def select_subjects(mod, exclude=None):
    """
    :param mod: The modality to select subjects for
    :type mod: str
    :param exclude: Subjects to exclude. If None, defaults to all currently
    specified in the participants.tsv file
    :type exclude: list

    :returns: subject -- List of subjects not excluded for the given modality
    """
    f = '../data/participants.tsv'
    sub_info = pd.read_csv(f, sep='\t')
    
    # reduce to subjects who completed the given modality
    sub_info = sub_info[sub_info[mod] == 1]

    # retrieve existing exclusions if no custom list is provided
    if not exclude:
        tmp1 = sub_info[sub_info['%s_exclude' % mod] == 1].participant_id
        tmp2 = sub_info[sub_info['behavior_%s_exclude' % mod] == 1].participant_id
        exclude = np.unique(list(tmp1) + list(tmp2))
        
    # reduce to non-excluded subjects
    if len(exclude) > 0:
        keep_ix = ~sub_info.participant_id.isin(exclude)
        subjects = list(sub_info[keep_ix].participant_id)
    else:
        subjects = list(sub_info.participant_id)
        
    return subjects


def drop_bad_trials(subject, behavior, epochs, layout, epo_type):

    # drop non-responses from behavior
    behavior = behavior[behavior.no_response != 1].reset_index()

    # skip the first behavior response since eeg was started late
    if subject == 'sub-pp012':
        behavior = behavior.iloc[1:, :].reset_index()

    # drop bad eeg trials from behavior
    ar_file = layout.get(subject=subject,
                         derivative='eeg_preprocessing',
                         extensions='%s_ar.pkl' % epo_type)[0].filename
    ar = pickle.load(open(ar_file, 'r'))
    if len(ar.bad_epochs_idx) > 0:
        behavior = behavior.drop(ar.bad_epochs_idx).reset_index()

    # drop bad behavior trials from eeg and behavior
    behavior_exclude = np.where(np.sum(behavior[['fast_rt', 'error',
                                                 'post_error']],
                                       axis=1))[0]
    behavior = behavior.drop(behavior_exclude)
    epochs.drop(behavior_exclude)

    return behavior, epochs
