import pystan
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import re
from scipy.stats import linregress

sns.set(style='whitegrid', font_scale=2)


def create_tidy_subject_df(filename, subject, modality, fast_rt_thresh=.2):
    """
    Creates a cleaned tidy subject dataframe. This includes removing fixation
    trials and encoding iti in separate column, encoding error and
    posterror trials, encoding n-1 trial sequences, and marking fast rts.

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


def wald(x, db, dr, ndt):
    """
    Computes the wald pdf distribution.

    Parameters
    ----------
    x: numpy array
        input data to compute the pdf over
    db: float
        decision boundary parameter.
    dr: float
        drift rate parameter.
    ndt: float
        non-decision time parameter.

    Returns
    -------
    numpy array
        calculated wald pdf of same shape as x
    """
    mu = db / dr
    lam = db * db
    first = lam / np.sqrt(2 * np.pi * np.power(x - ndt, 3))
    second_num = -lam * np.power(x - ndt - mu, 2)
    second_denom = (2 * mu**2 * (x - ndt))
    second = np.exp(second_num / second_denom)
    return first * second


def detrend_data(data):
    """
    Removes the linear slope (not intercept) for each subject. Done separately
    for each trial type and modality pairing.

    Parameters
    ----------
    data: DataFrame
        subject group behavioral dataframe

    Returns
    -------
    DataFrame
        subject group dataframe with the detrending applied
    """
    for sub in data.participant_id.unique():
        sub_ix = data.participant_id == sub
        for mod in data.modality.unique():
            mod_ix = data.modality == mod
            for tt in data.trial_type.unique():
                tt_ix = data.trial_type == tt
                ix = (mod_ix) & (tt_ix) & (sub_ix)
                rt = data[ix].response_time.as_matrix()
                trial = data[ix].trial.as_matrix()
                slope = linregress(trial, rt).slope
                detrended = rt - slope * trial
                data.loc[ix, 'response_time'] = detrended
    return data


def extract_samples(fit, exclude=None):
    """
    Extracts posterior samples from a pystan fitted model object into
    a pandas DataFrame.

    Parameters
    ----------
    fit: Pystan Fit Object
        The fitted pystan model to extract the samples from
    exclude: list
        List of parameters to ignore when extracting samples

    Returns
    -------
    DataFrame
        Dataframe with a row for each sample and a column for each parameter.
        Additionally has a column denoting the chain for each sample and
        a column denoting if a sample is warmup or not.
    """

    df = {}

    # extract stan data
    niter = fit.stan_args[0]['ctrl']['sampling']['iter']
    nwarmup = fit.stan_args[0]['ctrl']['sampling']['warmup']
    nchains = len(fit.stan_args)

    # create chain and warmup indicators
    tmp = [[i] * niter for i in range(1, nchains + 1)]
    df['chain'] = np.array(tmp).ravel()
    tmp = [1] * nwarmup + [0] * (niter - nwarmup)
    df['warmup'] = np.array(tmp * nchains)
    df['sample'] = np.tile(np.arange(1, niter + 1), nchains)

    # extract param names
    pars = fit.model_pars
    if exclude:
        pars = [p for p in pars if p not in exclude]

    # extract model parameter samples
    sim = fit.sim['samples']
    var_names = list(sim[0]['chains'].keys())
    for p in pars:
        p_vars = [v for v in var_names if re.match('^%s(\[\d+\])*$' % p, v)]
        for pv in p_vars:
            ch_s = [sim[i]['chains'][pv] for i in range(nchains)]
            df[pv] = np.array(ch_s).ravel()

    return pd.DataFrame(df)


def param_plot(param, samples, summary, inc_warmup=False):
    """
    Plot individual parameter diagnostic plot. This plot includes a
    posterior KDE plot, a traceplot, and an autocorrelation plot.
    Plots are split by chain.

    Parameters
    ----------
    param: str
        The parameter to be plotted
    samples: DataFrame
        dataframe of extracted samples as retrieved with the extract_samples
        function
    summary: DataFrame
        dataframe of extracted summary info as retrieved with the
        extract_summary function
    inc_warmup: boolean
        indicator on whether to include warmup samples in plot

    Returns
    -------
    pyplot Figure
        Matplotlib figure as described above
    """

    nchains = len(samples.chain.unique())
    niter = int(samples.shape[0] / nchains)
    nwarmup = int(samples.warmup.sum() / nchains)

    if not inc_warmup:
        samples = samples[samples.warmup == 0]

    # extract summary info
    rhat = float(summary.loc[param, 'Rhat'])
    neff = int(summary.loc[param, 'n_eff'])
    mean = summary.loc[param, 'mean']

    fig = plt.figure(figsize=(16, 10))

    # traceplot
    ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    if inc_warmup:
        samples['sample'] = np.tile(np.arange(1, niter + 1), nchains)
    else:
        samples['sample'] = np.tile(np.arange(1, niter - nwarmup + 1), nchains)
    samples['unit'] = 0
    sns.tsplot(data=samples, time='sample', unit='unit',
               value=param, condition='chain', ax=ax)
    if inc_warmup:
        ax.fill_between(np.arange(1, nwarmup + 1), ax.get_ylim()[0] - .5,
                        ax.get_ylim()[1] + .5, alpha=0.1, color='k')
    ax.set_title('Rhat: %.3f' % rhat)

    # density plot
    ax = plt.subplot2grid((2, 2), (0, 0))
    for chain in samples.chain.unique():
        data = samples[samples.chain == chain][param]
        sns.kdeplot(data=data, ax=ax)
    ax.legend(())
    ax.axvline(mean, color='k')

    # autocorrelation plot
    ax = plt.subplot2grid((2, 2), (0, 1))
    for chain in samples.chain.unique():
        data = samples[samples.chain == chain][param]
        pd.plotting.autocorrelation_plot(data, ax=ax)
    ess_ratio = float(neff) / ((niter - nwarmup) * nchains)
    ax.set_title('ESS: %d (%.2f)' % (neff, ess_ratio))
    ax.set_xlim((1, 20))

    return fig


def extract_summary(fit, exclude=None):
    """
    Extracts summary info from a pystan fitted model object into
    a pandas DataFrame.

    Parameters
    ----------
    fit: Pystan Fit Object
        The fitted pystan model to extract the samples from
    exclude: list
        List of parameters to ignore when extracting samples

    Returns
    -------
    DataFrame
        Dataframe with a row for each parameter and a column for
        each summary value.
    """

    # extract param names
    pars = fit.model_pars
    if exclude:
        pars = [p for p in pars if p not in exclude]

    summary = pystan.misc.stansummary(fit, pars=pars)
    summary = summary.splitlines()

    with open('tmp.txt', 'w+') as fid:
        for line in summary[4:-4]:
            fid.write(line + '\n')

    df = pd.read_csv('tmp.txt', sep='\s+', index_col=0, header=0)

    os.remove('tmp.txt')
    return df


def summary_plot(metric, summary, param_type):
    """
    Plot a summary value for each parameter of a given parameter type.

    Parameters
    ----------
    metric: str
        Rhat or n_eff. The metric to be plotted
    summary: DataFrame
        dataframe of extracted summary info as retrieved with the
        extract_summary function
    param_type: str
        The family of params to plot. One of db, dr, or ndt.

    Returns
    -------
    pyplot Figure
        boxplot with summary metric for each parameter
    """
    plt.figure(figsize=(20, 20))
    summary = summary.filter(like=param_type, axis=0)
    summary['param'] = summary.index
    fig = sns.barplot(y='param', x=metric, data=summary)
    if metric == 'Rhat':
        plt.xlim((.9, 1.1))
    return fig


def plot_ppc(subject, behavior, samples):
    """
    Plot posterior predictive checks. Plots generated data overlaid on the
    actual data using a random sample of posterior draws.

    Parameters
    ----------
    subject: str
        The subject to plot generated data for
    behavior: DataFrame
        The dataframe containing the behavioral data
    samples: DataFrame
        dataframe of extracted samples as retrieved with the extract_samples
        function

    Returns
    -------
    pyplot Figure
        Matplotlib figure as described above
    """

    # organize and extract samples
    samples = samples[samples.warmup == 0]
    samples = samples[[c for c in samples.columns if 'rt_pred' in c]]
    cols = [int(s[s.index('[') + 1:s.index(']')]) for s in samples.columns]
    ix = np.argsort(cols)
    pred = np.array(samples)
    pred = pred[:, ix]

    sub_ix = behavior.participant_id == subject
    pred = pred[:, np.where(sub_ix)[0]]
    behavior = behavior[sub_ix]

    f, axs = plt.subplots(1, 2, figsize=(16, 6))
    for i, mod in enumerate(['eeg', 'fmri']):
        mod_ix = behavior.modality == mod
        mod_behavior = behavior[mod_ix]
        mod_pred = pred[:, np.where(mod_ix)[0]]

        ax = axs[i]
        colors = ['b', 'r']
        for j, con in enumerate(['incongruent', 'congruent']):
            con_ix = mod_behavior.trial_type == con
            con_behavior = mod_behavior[con_ix]
            con_pred = mod_pred[:, np.where(con_ix)[0]]

            # plot behavior
            sns.distplot(con_behavior.response_time, color=colors[j], ax=ax)

            # plot ppc samples
            ix = np.random.choice(con_pred.shape[0], 100, replace=False)
            for i in ix:
                sns.kdeplot(con_pred[i, :], color=colors[j], alpha=0.07, ax=ax)

        ax.set_xlim((0, 1.75))
        ax.set_ylim((0, 5))
        ax.legend(['incongruent', 'congruent'])
        ax.set_title(mod)

    plt.show()


def plot_posteriors(param_type, samples, subjects):
    """
    Plot posterior distributions for a given parameter type.

    Parameters
    ----------
    param_type: str
        The family of params to plot. One of db, dr, or ndt.
    subjects: list
        List of subjects to plot
    samples: DataFrame
        dataframe of extracted samples as retrieved with the extract_samples
        function

    Returns
    -------
    pyplot Figure
        violin plot of all of the posterior distributions
    """
    samples = samples[samples.warmup == 0]
    num_samples = samples.shape[0]

    f, axs = plt.subplots(2, 1, figsize=(20, 12))
    if param_type == 'ndt':
        labels = ['fmri', 'eeg']
    else:
        labels = ['intercept', 'beta']

    for i, label in enumerate(labels):

        ax = axs[i]

        cols = [col for col in samples.columns if label in col and
                'sigma' not in col]
        param = '%s_%s' % (param_type, label)
        df = {param: [], 'subject': []}
        for col in sorted(cols):

            if param_type in col:
                if 'group' in col:
                    df['subject'] += ['group'] * num_samples
                else:
                    sub = subjects[int(col[col.index('[') + 1:col.index(']')])]
                    df['subject'] += [sub] * num_samples
                df[param] += list(samples[col])

        df = pd.DataFrame(df)
        data = df.groupby(['subject'], as_index=False)[param].mean()
        order = data.sort_values(by=param).subject
        sns.violinplot(x='subject', y=param, data=df, order=order, ax=ax)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)

    plt.tight_layout()
    plt.show()


def plot_corr(params, samples, subjects):
    """
    Plot parameter correlations (across subjects) for a given model.

    Parameters
    ----------
    params: list
        The list of params to plot
    subjects: list
        List of subjects to plot
    samples: DataFrame
        dataframe of extracted samples as retrieved with the extract_samples
        function

    Returns
    -------
    pyplot Figure
        violin plot of all of the posterior distributions
    """
    samples = samples[samples.warmup == 0]
    num_samples = samples.shape[0]

    df = {'subject': []}
    for param in params:
        df[param] = []
    for col in samples.columns:
        for i, param in enumerate(params):
            if param in col:
                if i == 0:
                    sub = subjects[int(col[col.index('[') + 1:col.index(']')])]
                    df['subject'] += [sub] * num_samples
                df[param] += list(samples[col])
    df = pd.DataFrame(df)
    data = df.groupby(['subject'], as_index=False)[params].mean()

    sns.pairplot(data, kind='reg')

    plt.tight_layout()
    plt.show()


def plot_group_es(samples, summary):
    """
    Plot a group effect size summary. Plots the db and dr group parameter
    posteriors and then plots the difference between the incongruent and
    congruent response time distributions resulting.

    Parameters
    ----------
    samples: DataFrame
        dataframe of extracted samples as retrieved with the extract_samples
        function
    summary: DataFrame
        dataframe of extracted summary info as retrieved with the
        extract_summary function

    Returns
    -------
    pyplot Figure
        plot as described above
    """
    samples = samples[samples.warmup == 0]

    cols = [c for c in samples.columns if 'group_mu' in c]
    samples = samples[cols]
    summary = summary.loc[cols]
    print(summary)

    f, axs = plt.subplots(2, 3, figsize=(20, 10))
    for i, label in enumerate(['db', 'dr']):
        ax = axs[i, 0]
        param = '%s_group_mu_intercept' % label
        sns.distplot(samples[param], ax=ax)
        ax.axvline(summary.loc[param, '2.5%'], color='k')
        ax.axvline(summary.loc[param, '97.5%'], color='k')
        ax.axvline(summary.loc[param, 'mean'], color='k')

        ax = axs[i, 1]
        param = '%s_group_mu_beta' % label
        sns.distplot(samples[param], ax=ax)
        ax.axvline(summary.loc[param, '2.5%'], color='k')
        ax.axvline(summary.loc[param, '97.5%'], color='k')
        ax.axvline(summary.loc[param, 'mean'], color='k')

        ax = axs[i, 2]
        if label == 'dr':
            other = summary.loc['db_group_mu_intercept', 'mean']
        else:
            other = summary.loc['dr_group_mu_intercept', 'mean']

        con = summary.loc['%s_group_mu_intercept' % label, 'mean']
        incon = con + summary.loc['%s_group_mu_beta' % label, 'mean']

        x = np.arange(0, 1.75, .01)
        if label == 'db':
            ax.plot(x, wald(x, con, other, 0), color='r')
            ax.plot(x, wald(x, incon, other, 0), color='b')
        else:
            ax.plot(x, wald(x, other, con, 0), color='r')
            ax.plot(x, wald(x, other, incon, 0), color='b')
        ax.legend(['congruent', 'incongruent'])

    plt.tight_layout()
    plt.show()
