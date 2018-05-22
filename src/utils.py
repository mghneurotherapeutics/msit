import numpy as np
import pandas as pd


def exclude_subjects(subjects, mod, reset=False):
    """
    Utility function to denote excluded subjects

    Parameters
    ----------
    subjects: list of str
        The subjects to be excluded
    mod: str
        The modality they are being excluded base on. fmri, eeg, eeg_behavior,
        or fmri_behavior
    reset: boolean
        Whether to remove past exclusions or append to them

    Returns
    -------
    None
        Updates the participants.tsv file with the exclusions
    """

    f = '../data/participants.tsv'
    sub_info = pd.read_csv(f, sep='\t')
    if reset:
        sub_info['%s_exclude' % mod] = 0
    sub_info.loc[sub_info.participant_id.isin(subjects),
                 '%s_exclude' % mod] = 1
    sub_info.to_csv(f, sep='\t', index=False)


def select_subjects(mod, exclude='auto'):
    """
    Utility function to denote select out subjects by modality.

    Parameters
    ----------
    mod: str
        The modality to select subjects for. fmri, eeg, or both
    exclude: list or str
        If list, all subjects who completed the given modality will be selected
        if they are not in the provided list. 'auto' will only select subjects
        who are not denoted as excluded in the participants.tsv file.

    Returns
    -------
    list
        The list of subjects who meet the modality and exclusion criteria
    """
    f = '../data/participants.tsv'
    sub_info = pd.read_csv(f, sep='\t')

    # reduce to subjects who completed the given modality
    if mod == 'both':
        ix = np.array(sub_info['eeg'] + sub_info['fmri']) == 2
        sub_info = sub_info[ix]
    else:
        sub_info = sub_info[sub_info[mod] == 1]

    # retrieve existing exclusions if no custom list is provided
    if exclude == 'auto':
        if mod == 'both':
            exclude = []
            for mod in ['eeg', 'fmri']:
                tmp1 = sub_info[sub_info['%s_exclude' % mod] == 1].participant_id
                tmp2 = sub_info[sub_info['behavior_%s_exclude' % mod] == 1].participant_id
                exclude += list(np.unique(list(tmp1) + list(tmp2)))
        else:
            tmp1 = sub_info[sub_info['%s_exclude' % mod] == 1].participant_id
            tmp2 = sub_info[sub_info['behavior_%s_exclude' % mod] == 1].participant_id
            exclude = np.unique(list(tmp1) + list(tmp2))

    # reduce to non-excluded subjects
    if len(exclude) > 1:
        keep_ix = ~sub_info.participant_id.isin(exclude)
        subjects = list(sub_info[keep_ix].participant_id)
    else:
        subjects = list(sub_info.participant_id)

    return sorted(subjects)
