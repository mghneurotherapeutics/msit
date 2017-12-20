import pystan
import pickle


def fit_model(model_name, data, n_iter, n_chains, init, seed=10):

    model_fit = {}

    print('Compiling Model...')
    f = '../models/%s/%s.stan' % (model_name, model_name)
    model_fit['model'] = pystan.StanModel(file=f, model_name=model_name)

    print('Computing MAP Estimates...')
    model_fit['mapp'] = model_fit['model'].optimizing(data=data,
                                                      seed=seed,
                                                      init=init)

    print('Sampling from Posterior...')
    model_fit['samples'] = model_fit['model'].sampling(data=data,
                                                       iter=n_iter,
                                                       chains=n_chains,
                                                       init=init,
                                                       seed=seed)

    print('Pickling Model Fit...')
    pickle.dump(model_fit, open(f.replace('.stan', '.pkl'), 'w'))

    print('Finished')

    return model_fit
