import numpy as np


def baseline_normalize(tfr, baseline):

    data = tfr.data
    times = tfr.times

    # scale by individual trial median
    # trial_median = np.median(data, axis=-1)[:, :, :, np.newaxis]
    # data /= trial_median

    # collapse across trials
    # data = np.median(data, axis=0)
    data = np.mean(data, axis=0)

    # divide by median baseline period
    if type(baseline) == tuple:
        baseline_ix = np.where(np.logical_and(times >= baseline[0],
                                              times <= baseline[1]))[0]
        baseline = data[:, :, baseline_ix]

    # baseline_median = np.median(baseline, axis=-1)[:, :, np.newaxis]
    baseline_summ = np.mean(baseline, axis=-1)[:, :, np.newaxis]
    # data /= baseline_median
    data /= baseline_summ

    # decibel transformation
    data = 10 * np.log10(data)

    # stick back into mne object
    tfr = tfr.average()
    tfr.data = data

    return tfr, baseline


def add_events(epochs, behavior, event_type, events):

    if event_type == 'condition':
        epochs.event_id = {'incongruent': 1, 'congruent': 0}
        events = behavior.trial_type.astype('category').cat.codes
    elif event_type == 'laterality':
        drop_ix = behavior.target_location == 'middle'
        behavior = behavior[~drop_ix]
        epochs.drop(drop_ix)
        epochs.event_id = {'incongruent-left': 0,
                           'incongruent-right': 1,
                           'congruent-left': 2,
                           'congruent-right': 3}
        events = []
        for tt, loc in zip(behavior.trial_type, behavior.target_location):
            if tt == 'incongruent' and loc == 'left':
                events.append(0)
            elif tt == 'incongruent' and loc == 'right':
                events.append(1)
            elif tt == 'congruent' and loc == 'left':
                events.append(2)
            else:
                events.append(3)

    else:
        raise ValueError('Unknown Event Type')

    epochs.events[:, -1] = events
    return epochs
