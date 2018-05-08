import pystan
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os
import re
from sklearn.neighbors import KernelDensity
from scipy.stats import norm

sns.set(style='whitegrid', font_scale=2)


def weibull(x, alpha, sigma, shift):
    p1 = (alpha / sigma)
    p2 = np.power((x - shift) / sigma, alpha - 1)
    p3 = np.exp(-np.power((x - shift) / sigma, alpha))
    return p1 * p2 * p3


def wald(x, alpha, gamma, theta):
    first = alpha / np.sqrt(2 * np.pi * np.power(x - theta, 3))
    second = np.exp(-(np.power(alpha - gamma * (x - theta), 2)) / (2 * (x - theta)))
    return first * second

def extract_samples(fit, exclude=None):

    df = {}

    # extract stan data
    niter = fit.stan_args[0]['ctrl']['sampling']['iter']
    nwarmup = fit.stan_args[0]['ctrl']['sampling']['warmup']
    nchains = len(fit.stan_args)

    # create chain and warmup indicators
    df['chain'] = np.array([[i] * niter for i in range(1, nchains + 1)]).ravel()
    df['warmup'] = np.array(([1] * nwarmup + [0] * (niter - nwarmup)) * nchains)
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

    # extract nuts sampling parameters
    nuts = fit.get_sampler_params()
    for k in nuts[0].keys():
        t = []
        for i in range(nchains):
            t.append(nuts[i][k])
        df[k[:-2]] = np.ravel(t)

    return pd.DataFrame(df)


def param_plot(param, samples, summary, inc_warmup=False):

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
    ax.set_title('ESS: %d (%.2f)' % (neff, float(neff) / ((niter - nwarmup) * nchains)))
    ax.set_xlim((1, 20))

    return fig


def extract_summary(fit, exclude=None):

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


def display_elapsed_time(msg, start_time):
    curr_time = time.time()
    elapsed_time = curr_time - start_time
    et_min = elapsed_time // 60
    et_sec = elapsed_time % 60
    print('%s took %d min. %d sec.' % (msg, et_min, et_sec))
    return curr_time


def find_map(samples):

    rang = samples.max() - samples.min()
    bw = rang / 10

    X = np.linspace(samples.min() - rang, samples.max() + rang, 1000).reshape((-1, 1))

    kde = KernelDensity(bandwidth=bw)
    kde = kde.fit(samples.reshape(-1, 1))

    log_pdf = kde.score_samples(X)
    return X[np.argsort(log_pdf)][-1][0]


def extract_maps(samples):

    maps = {}
    for param in samples.keys():
        samps = samples[param]
        if len(samps.shape) == 3:
            maps[param] = []
            for i in range(samps.shape[0]):
                samp = samps[i, :, :].flatten()
                maps[param].append(find_map(samp))
        else:
            samp = samps.flatten()
            maps[param] = find_map(samp)

    return maps


def fit_model(model_name, model_path, data, n_iter, n_chains, init, seed=10,
              keep_params=None):

    print('Starting Model Fit...')
    start_time = time.time()

    # Initialize model fit dictionary
    model_fit = {}
    model_fit['n_chains'] = n_chains
    model_fit['n_iter'] = n_iter

    print('Compiling Model...')
    f = '../models/%s/%s.stan' % (model_path, model_name)
    model = pystan.StanModel(file=f, model_name=model_name)
    curr_time = display_elapsed_time('Compiling', start_time)

    print('Sampling from Posterior...')
    fit = model.sampling(data=data,
                         iter=n_iter,
                         chains=n_chains,
                         init='random',
                         seed=seed)
    msg = 'Drawing %s Posterior Samples' % n_iter
    curr_time = display_elapsed_time(msg, curr_time)

    print('Extracting Samples...')
    model_fit['samples'] = extract_samples(fit,
                                           keep_params,
                                           n_chains, n_iter)
    curr_time = display_elapsed_time('Extracting samples', curr_time)

    print('Computing MAP Estimates...')
    model_fit['maps'] = extract_maps(model_fit['samples'])
    curr_time = display_elapsed_time('Finding MAP estimates', curr_time)

    print('Extracting Fit Summary...')
    summary = pystan.misc._summary(fit, pars=keep_params,
                                   probs=(0.02, 0.25, 0.5, 0.75, 0.98),
                                   digits_summary=5)
    model_fit['summary'] = pd.DataFrame(summary['summary'],
                                        index=summary['summary_rownames'],
                                        columns=summary['summary_colnames'])

    curr_time = display_elapsed_time('Extracting fit summary', curr_time)

    print('Pickling Model Fit...')
    f = f.replace('.stan', '.pkl')
    if os.path.exists(f):
        os.remove(f)
    pickle.dump(model_fit, open(f, 'w'))

    curr_time = display_elapsed_time('Pickling model fit', curr_time)

    curr_time = display_elapsed_time('Total Time:', start_time)

    print('Finished')

    return model_fit


def plot_hierarchical_wald_fit(model_fit, behavior, subject, subjects):
    plt.close('all')
    f, ax = plt.subplots(1, 1, figsize=(16, 8))
    mapp = model_fit['maps']
    colors = ['#e41a1c', '#377eb8']
    conditions = ['congruent', 'incongruent']

    if subject != 'group':
        behavior = behavior[behavior.participant_id == subject]
        sub_ix = subjects.index(subject)

    for i, c in enumerate(conditions):

        cond_rt = behavior[behavior.trial_type == c].response_time

        gamma = mapp['dr_%s' % c][sub_ix]
        alpha = mapp['db_%s' % c][sub_ix]
        theta = mapp['ndt_%s' % c][sub_ix]

        x = np.arange(theta, 1.75, .01)
        sns.distplot(cond_rt, color=colors[i], ax=ax, kde=False,
                     norm_hist=True)

        ax.plot(x, wald(x, alpha, gamma, theta), color=colors[i])

    plt.title(subject)
    plt.legend(conditions)
    plt.xlabel('Response Time (s)')
    plt.xlim((0, 1.75))
    plt.ylim((0, 4))
    plt.show()


def plot_weibull_subject_fit(model_name, model_fit, behavior, subject,
                             subjects):
    plt.close('all')
    f, ax = plt.subplots(1, 1, figsize=(16, 8))
    mapp = model_fit['map']
    colors = ['#e41a1c', '#377eb8']
    conditions = ['congruent', 'incongruent']

    if subject != 'group':
        behavior = behavior[behavior.participant_id == subject]
        sub_ix = subjects.index(subject)

    for i, c in enumerate(conditions):

        cond_rt = behavior[behavior.trial_type == c].response_time

        if model_name == 'weibull_additive_hierarchical':
            if c == 'congruent':
                shape = mapp['beta0_shape'][sub_ix]
                scale = mapp['beta0_scale'][sub_ix]
                shift = mapp['beta0_shift'][sub_ix]
            else:
                shape = mapp['beta0_shape'][sub_ix] + mapp['beta1_shape'][sub_ix]
                scale = mapp['beta0_scale'][sub_ix] + mapp['beta1_scale'][sub_ix]
                shift = (mapp['beta0_shift'][sub_ix] + mapp['beta1_shift'][sub_ix])
        else:
            shape = mapp['shape_%s' % c][sub_ix]
            scale = mapp['scale_%s' % c][sub_ix]
            shift = mapp['shift_%s' % c][sub_ix]

        x = np.arange(shift, 1.75, .01)
        sns.distplot(cond_rt, color=colors[i], ax=ax, kde=False,
                     norm_hist=True)

        ax.plot(x, weibull(x, shape, scale, shift), color=colors[i])

    plt.title(subject)
    plt.legend(conditions)
    plt.xlabel('Response Time (s)')
    plt.xlim((0, 1.75))
    plt.ylim((0, 4))
    plt.show()


def plot_group_wald_parameters(param, model_fit):

    f, axs = plt.subplots(2, 2, figsize=(18, 12))
    colors = ['#e41a1c', '#377eb8']
    conditions = ['incongruent', 'congruent']


    for i, c in enumerate(conditions):


        ax = axs[0, 1]
        mu = '%s_group_mu_%s' % (param, c)
        samples = model_fit['samples'][mu].flatten()
        sns.distplot(samples, ax=ax, kde=False, norm_hist=True, color=colors[i])
        mu = model_fit['maps'][mu]
        ax.legend(conditions)
        ax.axvline(mu, color=colors[i], label='_nolegend_')
        ax.set_title('Mu')

        ax = axs[1, 1]
        sig = '%s_group_sigma_%s' % (param, c)
        samples = model_fit['samples'][sig].flatten()
        sns.distplot(samples, ax=ax, kde=False, norm_hist=True, color=colors[i])
        sig = model_fit['maps'][sig]
        ax.axvline(sig, color=colors[i])
        ax.set_title('Sigma')

        ax = axs[0, 0]
        maps = model_fit['maps']['%s_%s' % (param, c)]
        sns.distplot(maps, kde=False, norm_hist=True, ax=ax, color=colors[i])
        if param == 'ndt':
            xlim = (0, 1)
        else:
            xlim = (0, 8)

        ax.set_xlim(xlim)
        x = np.linspace(xlim[0], xlim[1], 1000)
        ax.plot(x, norm.pdf(x, mu, sig), color=colors[i])
        ax.axvline(mu, color=colors[i], label='_nolegend_')

    ax = axs[1, 0]
    mu = '%s_group_beta' % (param)
    maps = model_fit['maps']['%s_beta' % (param)]
    sns.distplot(maps, ax=ax, kde=False, norm_hist=True, color=colors[i])
    mu = model_fit['maps'][mu]
    ax.axvline(mu, color=colors[i], label='_nolegend_')
    ax.set_title('Beta')

    sns.despine()
    plt.show()


def plot_posterior(param, model_fit, subject, subjects):

    sub_ix = subjects.index(subject)

    samples = model_fit['samples'][param].squeeze()
    mapp = model_fit['maps'][param]

    f, axs = plt.subplots(1, 2, figsize=(20, 8))

    if len(samples.shape) > 2:

        plt.suptitle('%s %s' % (subject, param), y=1.01)

        # extract summary info
        mean = model_fit['summary'].loc['%s[%d]' % (param, sub_ix), 'mean']
        low = model_fit['summary'].loc['%s[%d]' % (param, sub_ix), '2%']
        high = model_fit['summary'].loc['%s[%d]' % (param, sub_ix), '98%']
        eff_ss = model_fit['summary'].loc['%s[%d]' % (param, sub_ix), 'n_eff']
        r = model_fit['summary'].loc['%s[%d]' % (param, sub_ix), 'Rhat']

        sub_samples = samples[sub_ix, :, :]

        # plot map distribution
        ax = axs[0, 0]
        sns.distplot(mapp, ax=ax)
        ax.axvline(mapp[sub_ix], color='k')
        ax.set_title('All Subject MAPs')

        # plot subject posterior
        ax = axs[0, 1]
        sns.distplot(sub_samples.flatten(), ax=ax)
        ax.axvline(mapp[sub_ix], color='b')
        ax.axvline(mean, color='k')
        ax.axvline(low, color='k', linestyle='--')
        ax.axvline(high, color='k', linestyle='--')
        ax.set_title('Posterior Density')

        # plot trace plot
        ax = axs[1, 0]
        for i in range(sub_samples.shape[0]):
            ax.plot(sub_samples[i, :])
        ax.set_title('Trace Plot: Rhat = %.2f' % (r))

        # plot auto-correlation
        ax = axs[1, 1]
        for i in range(sub_samples.shape[0]):
            acf_s = pd.Series(sub_samples[i, :])
            acf = []
            lags = np.arange(1, 51, 1)
            for lag in lags:
                acf.append(acf_s.autocorr(lag))
            ax.plot(lags, acf)
        ax.set_title('Effective Sample Size = %.1f' % eff_ss)
        ax.set_ylabel('Auto-Correlation')
        ax.set_xlabel('Lag')

    else:
        plt.suptitle('%s' % (param), y=1.01)

        mean = model_fit['summary'].loc['%s' % (param), 'mean']
        low = model_fit['summary'].loc['%s' % (param), '2%']
        high = model_fit['summary'].loc['%s' % (param), '98%']
        eff_ss = model_fit['summary'].loc['%s' % (param), 'n_eff']
        r = model_fit['summary'].loc['%s' % (param), 'Rhat']

        # plot map distribution
        ax = axs[0]
        sns.distplot(samples.flatten(), ax=axs[0])
        ax.axvline(mapp, color='b')
        ax.axvline(mean, color='k')
        ax.axvline(low, color='k', linestyle='--')
        ax.axvline(high, color='k', linestyle='--')
        axs[0].set_title('Posterior Density')

        # plot trace plot
        for i in range(samples.shape[0]):
            axs[1].plot(samples[i, :])
        axs[1].set_title('Trace Plot: Rhat = %.2f' % (r))

        ax = axs[2]
        for i in range(samples.shape[0]):
            acf_s = pd.Series(samples[i, :])
            acf = []
            lags = np.arange(1, 51, 1)
            for lag in lags:
                acf.append(acf_s.autocorr(lag))
            ax.plot(lags, acf)
        ax.set_title('Effective Sample Size = %.1f' % eff_ss)
        ax.set_ylabel('Auto-Correlation')
        ax.set_xlabel('Lag')

    plt.tight_layout()
    plt.show()
