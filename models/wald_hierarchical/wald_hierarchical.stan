functions {

  // shifted wald log density
  real wald_lpdf(vector x, vector mu, vector lambda, vector shift) {
    vector[num_elements(x)] prob;
    vector[num_elements(x)] sx;
    sx = x - shift;
    prob = sqrt(lambda ./ (2 * pi() * sx .* sx .* sx)) .* exp(-(lambda .* (sx - mu) .* (sx - mu)) ./ (2 * mu .* mu .* sx));
    return sum(log(prob));
  }

  // shifted wald log likelihood
  vector wald_ll(vector x, vector mu, vector lambda, vector shift) {
    vector[num_elements(x)] prob;
    vector[num_elements(x)] sx;
    sx = x - shift;
    prob = sqrt(lambda ./ (2 * pi() * sx .* sx .* sx)) .* exp(-(lambda .* (sx - mu) .* (sx - mu)) ./ (2 * mu .* mu .* sx));
    return log(prob);
  }

  // shifted wald random number generator
  // algorithm from the inverse gaussian wikipedia page
  real wald_rng(real mu, real lambda, real shift) {
    real y;
    real x1;
    real x2;
    real x;
    real un_samp;

    y = square(normal_rng(0, 1));
    x1 = mu + (square(mu) * y) / (2 * lambda);
    x2 = (mu / (2 * lambda)) * sqrt(4 * mu * lambda * y + square(mu) * square(y));
    x = x1 - x2;
    un_samp = uniform_rng(0, 1);

    if (un_samp <= mu / (mu + x)) {
      return x + shift;
    } else {
      return (square(mu) / x) + shift;
    }

  }
}

data {
  int<lower=0> Nt;
  int<lower=0> Ns;
  vector<lower=0>[Nt] rt;
  vector<lower=0>[Ns] min_rt_eeg;
  vector<lower=0>[Ns] min_rt_fmri;
  int sub_ix[Nt];
  vector[Nt] tt;
  vector[Nt] mod;
}

parameters {

  // Group Parameters

  // Drift Rate Parameters
  real<lower=0> dr_group_mu_intercept;
  real<lower=0> dr_group_sigma_intercept;
  real dr_group_mu_beta;
  real<lower=0> dr_group_sigma_beta;

  // Decision Boundary Parameters
  real<lower=0> db_group_mu_intercept;
  real<lower=0> db_group_sigma_intercept;
  real db_group_mu_beta;
  real<lower=0> db_group_sigma_beta;

  // Subject Parameters
  vector<lower=0, upper=1>[Ns] ndt_eeg_tmp;
  vector<lower=0, upper=1>[Ns] ndt_fmri_tmp;
  vector<lower=0>[Ns] db_intercept;
  vector<lower=0>[Ns] db_incongruent;
  vector<lower=0>[Ns] dr_intercept;
  vector<lower=0>[Ns] dr_incongruent;
}

transformed parameters {
  vector[Ns] db_beta;
  vector[Ns] dr_beta;
  vector[Ns] ndt_eeg;
  vector[Ns] ndt_fmri;
  vector[Nt] ndt;
  vector[Nt] db;
  vector[Nt] dr;
  vector[Nt] mu;
  vector[Nt] lambda;

  ndt_eeg = ndt_eeg_tmp .* min_rt_eeg;
  ndt_fmri = ndt_fmri_tmp .* min_rt_fmri;

  db_beta = db_incongruent - db_intercept;
  dr_beta = dr_incongruent - dr_intercept;

  db = db_intercept[sub_ix] + db_beta[sub_ix] .* tt;
  dr = dr_intercept[sub_ix] + dr_beta[sub_ix] .* tt;

  ndt = ndt_fmri[sub_ix] .* mod + ndt_eeg[sub_ix] .* (1 - mod);

  mu = db ./ dr;
  lambda = db .* db;
}

model {

  // Drift Rate Priors
  dr_group_mu_intercept ~ lognormal(1.5, 1);
  dr_group_sigma_intercept ~ normal(0, 1);
  dr_group_mu_beta ~ normal(0, 1);
  dr_group_sigma_beta ~ normal(0, 1);

  dr_intercept ~ normal(dr_group_mu_intercept,
                        square(dr_group_sigma_intercept));
  dr_beta ~ normal(dr_group_mu_beta,
                   square(dr_group_sigma_beta));

  // Decision Boundary Priors
  db_group_sigma_intercept ~ normal(0, 1);
  db_group_sigma_beta ~ normal(0, 1);
  db_group_mu_beta ~ normal(0, 1);
  db_group_mu_intercept ~ lognormal(1.5, 1);

  db_intercept ~ normal(db_group_mu_intercept,
                        square(db_group_sigma_intercept));
  db_beta ~ normal(db_group_mu_beta,
                   square(db_group_sigma_beta));

  // wald likelihood
  rt ~ wald(mu, lambda, ndt);

}

generated quantities {
  vector[Nt] log_lik;
  vector[Nt] rt_pred;

  // Posterior Predictions
  for (n in 1:Nt) {
    rt_pred[n] = wald_rng(mu[n], lambda[n], ndt[n]);
  }

  // Log likelihood for loo and waic
  log_lik = wald_ll(rt, mu, lambda, ndt);
}

