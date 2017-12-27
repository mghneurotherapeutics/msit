import numpy as np


def baseline_normalize(tfr, baseline, func=np.mean, method='classic'):

    data = tfr.data
    times = tfr.times

    if method == 'grandchamp':
        # scale by individual trial median
        trial_median = np.median(data, axis=-1)[:, :, :, np.newaxis]
        data /= trial_median

    # collapse across trials
    data = func(data, axis=0)

    # divide by baseline period collapsed across time
    if type(baseline) == tuple:
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


def add_events(mne_object, behavior):
    mne_object.event_id = {'incongruent': 1, 'congruent': 0}
    mne_object.events[:, -1] = behavior.trial_type.astype('category').cat.codes
    return mne_object


def power_heatmap(power, ax, lim, rts=None, rt_colors=None):

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

