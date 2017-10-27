import numpy as np


def select_subjects(layout, modality, start=None, end=None, exclude=[]):

    subjects = np.array(sorted(layout.get(target='subject',
                                          modality=modality,
                                          return_type='id')))

    # perform selection
    start_ix, end_ix = 0, len(subjects) + 1
    if start:
        start_ix = np.where(subjects == start)[0]
    if end:
        end_ix = np.where(subjects == end)[0]
    subjects = subjects[start_ix:end_ix]

    return [sub for sub in subjects if sub not in exclude]
