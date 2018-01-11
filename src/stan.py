import pystan
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time
import os
import re

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


def extract_samples(fit, params, n_chains, n_iter):
    sim = fit.sim['samples']
    var_names = list(sim[0]['chains'].keys())

    ss = n_iter / 2

    samples = {}
    for p in params:
        p_vars = [v for v in var_names if re.match('^%s(\[\d\])*$' % p, v)]

        p_data = []
        for pv in p_vars:
            ch_s = [sim[i]['chains'][pv][ss:] for i in range(n_chains)]
            p_data.append(ch_s)

        samples[p] = np.array(p_data)
        samples[p] = samples[p].squeeze()

    return samples


def display_elapsed_time(msg, start_time):
    curr_time = time.time()
    elapsed_time = curr_time - start_time
    et_min = elapsed_time // 60
    et_sec = elapsed_time % 60
    print('%s took %d min. %d sec.' % (msg, et_min, et_sec))
    return curr_time


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

    print('Computing MAP Estimates...')
    model_fit['map'] = model.optimizing(data=data,
                                        seed=seed,
                                        init=init)
    curr_time = display_elapsed_time('Finding MAP estimates', curr_time)

    if not keep_params:
        keep_params = model_fit['map'].keys()

    print('Sampling from Posterior...')
    init = [model_fit['map']] * n_chains
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
    mapp = model_fit['map']
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


def plot_posterior(param, model_fit, subject, subjects):

    sub_ix = subjects.index(subject)

    samples = model_fit['samples'][param].squeeze()
    mapp = model_fit['map'][param]

    if len(samples.shape) > 2:

        f, axs = plt.subplots(2, 2, figsize=(18, 12))
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
        f, axs = plt.subplots(1, 3, figsize=(24, 8))
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

