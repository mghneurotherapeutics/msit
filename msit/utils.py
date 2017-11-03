import numpy as np
import pandas as pd


def select_subjects(layout, modality, start=None, end=None, exclude=[]):

    subjects = np.array(sorted(layout.get(target='subject',
                                          modality=modality,
                                          return_type='id')))

    # perform index based selection
    start_ix, end_ix = 0, len(subjects) + 1
    if start:
        start_ix = np.where(subjects == start)[0]
    if end:
        end_ix = np.where(subjects == end)[0]
    subjects = subjects[start_ix:end_ix]

    # determines exclusions
    if type(exclude) == str:
        ex = pd.read_csv('../data/participants.tsv', sep='\t')
        tmp1 = ex[ex['%s_exclude' % exclude] == 1].participant_id
        tmp2 = ex[ex['behavior_%s_exclude' % exclude] == 1].participant_id
        exclude = np.unique(list(tmp1) + list(tmp2))

    return [sub for sub in subjects if sub not in exclude]


def denote_exclusions(excluded_subjects, modality, overwrite=True):
    """

    :param excluded_subjects: List of participants to be excluded.
    :type epochs: list
    :param modality: The modality subjects are being excluded based on.
    :type modality: string
    :param overwrite: Whether to clear existing exclusions or update
    :type ylim: boolean

    :returns: None -- Updates the participants.tsv file with the exclusions
    """

    exclusions = pd.read_csv('../data/participants.tsv', sep='\t')

    # clear existing exclusions if overwrite is true
    if overwrite:
        exclusions['%s_exclude' % modality] = 0

    for s in excluded_subjects:
        exclusions.loc[np.where(exclusions.participant_id == s)[0],
                       '%s_exclude' % modality] = 1
    exclusions.to_csv('../data/participants.tsv', sep='\t', index=False)
