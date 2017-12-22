import pystan
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

sns.set(style='whitegrid', font_scale=2)


def weibull(x, alpha, sigma, shift):
    p1 = (alpha / sigma)
    p2 = np.power((x - shift) / sigma, alpha - 1)
    p3 = np.exp(-np.power((x - shift) / sigma, alpha))
    return p1 * p2 * p3


def fit_model(model_name, data, n_iter, n_chains, init, seed=10,
              parameters=None):

    print('Starting Model Fit...')
    model_fit = {}

    print('Compiling Model...')
    start_time = time.time()
    f = '../models/%s/%s.stan' % (model_name, model_name)
    model_fit['model'] = pystan.StanModel(file=f, model_name=model_name)
    compile_time = time.time()
    print('Compiling took %s min. %s sec' % ((compile_time - start_time) // 60,
          (compile_time - start_time) % 60))

    print('Computing MAP Estimates...')
    model_fit['mapp'] = model_fit['model'].optimizing(data=data,
                                                      seed=seed,
                                                      init=init)
    map_time = time.time()
    m = 'Finding MAP estimates took %s min. %s sec'
    print(m % ((map_time - compile_time) // 60,
               (map_time - compile_time) % 60))
    pickle.dump(model_fit, open(f.replace('.stan', '.pkl'), 'w'))

    print('Sampling from Posterior...')
    model_fit['samples'] = model_fit['model'].sampling(data=data,
                                                       iter=n_iter,
                                                       chains=n_chains,
                                                       init=[model_fit['mapp']] * n_chains,
                                                       seed=seed)
    sample_time = time.time()
    m = 'Drawing %s Posterior Samples took %s min. %s sec'
    print(m % (n_chains * n_iter,
               (sample_time - map_time) // 60,
               (sample_time - map_time) % 60))

    print('Pickling Model Fit...')
    pickle.dump(model_fit, open(f.replace('.stan', '.pkl'), 'w'))

    pickle_time = time.time()
    m = 'Pickling model took %s min. %s sec'
    print(m % ((pickle_time - sample_time) // 60,
               (pickle_time - sample_time) % 60))

    print('Total Time: %s min. %s sec' % ((pickle_time - start_time) // 60,
                                          (pickle_time - start_time) % 60))
    print('Finished')

    return model_fit


def plot_weibull_subject_fit(model_fit, behavior, subject, subjects):
    plt.close('all')
    f, ax = plt.subplots(1, 1, figsize=(16, 8))
    mapp = model_fit['mapp']
    colors = ['#e41a1c', '#377eb8']
    conditions = ['congruent', 'incongruent']

    if subject != 'group':
        behavior = behavior[behavior.participant_id == subject]
        sub_ix = subjects.index(subject)

    for i, c in enumerate(conditions):

        cond_rt = behavior[behavior.trial_type == c].response_time

        if subject != 'group':
            if c == 'congruent':
                shape = mapp['beta0_shape'][sub_ix]
                scale = mapp['beta0_scale'][sub_ix]
                shift = mapp['beta0_shift'][sub_ix]
            else:
                shape = mapp['beta0_shape'][sub_ix] + mapp['beta1_shape'][sub_ix]
                scale = mapp['beta0_scale'][sub_ix] + mapp['beta1_scale'][sub_ix]
                shift = (mapp['beta0_shift'][sub_ix] + mapp['beta1_shift'][sub_ix])
        else:
            if c == 'congruent':
                shape = mapp['beta0_shape'][sub_ix]
                scale = mapp['beta0_scale'][sub_ix]
                shift = mapp['beta0_shift'][sub_ix]
            else:
                shape = mapp['beta0_shape'][sub_ix] + mapp['beta1_shape']
                scale = mapp['beta0_scale'][sub_ix] + mapp['beta1_scale']
                shift = mapp['beta0_shift'][sub_ix] + mapp['beta1_shift']



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

    samples = model_fit['samples'][param]
    mapp = model_fit['mapp'][param]
    print(samples.shape)

    if len(samples.shape) > 1:

        f, axs = plt.subplots(1, 3, figsize=(24, 6))
        sub_samples = samples[:, sub_ix]

        # plot map distribution
        sns.distplot(mapp, ax=axs[0])
        axs[0].axvline(mapp[sub_ix], color='k')
        axs[0].set_title('%s Subject MAPs' % param)

        # plot subject posterior
        sns.distplot(sub_samples, ax=axs[1])
        axs[1].axvline(mapp[sub_ix], color='k')
        axs[1].set_title('%s %s Posterior' % (subject, param))

        # plot trace plot
        axs[2].plot(sub_samples)
        axs[2].set_title('%s Trace Plot' % param)
    else:
        f, axs = plt.subplots(1, 2, figsize=(16, 8))

        # plot map distribution
        sns.distplot(samples, ax=axs[0])
        axs[0].axvline(mapp, color='k')
        axs[0].set_title('%s Posterior' % param)

        # plot trace plot
        axs[1].plot(samples)
        axs[1].set_title('%s Trace Plot' % param)

    plt.show()


def plot_map_estimates(model_fit, param):
    plt.close('all')
    values = model_fit['mapp'][param]
    if values.shape:
        values = np.unique(values)
        f, ax = plt.subplots(1, 1, figsize=(16, 8))
        sns.distplot(values)
        plt.xlabel(param)
        plt.show()
    else:
        print(values)
