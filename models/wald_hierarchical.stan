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
  vector<lower=0>[Ns] min_rt;
  int sub_ix[Nt];
  vector[Nt] tt;
  vector[Nt] mod;
}

parameters {

  // Group Parameters

  // Drift Rate Parameters
  real<lower=0> dr_group_mu_intercept;
  real<lower=0> dr_group_sigma_intercept;
  real dr_group_mu_beta_mod;
  real<lower=0> dr_group_sigma_beta_mod;
  real dr_group_mu_beta_tt;
  real<lower=0> dr_group_sigma_beta_tt;

  // Decision Boundary Parameters
  real<lower=0> db_group_mu_intercept;
  real<lower=0> db_group_sigma_intercept;
  real db_group_mu_beta_mod;
  real<lower=0> db_group_sigma_beta_mod;
  real db_group_mu_beta_tt;
  real<lower=0> db_group_sigma_beta_tt;

  // Non-Decision Time parameters
  real<lower=0> ndt_group_mu_intercept;
  real<lower=0> ndt_group_sigma_intercept;
  real ndt_group_mu_beta_mod;
  real<lower=0> ndt_group_sigma_beta_mod;
  real ndt_group_mu_beta_tt;
  real<lower=0> ndt_group_sigma_beta_tt;

  // Subject Parameters
  vector<lower=0, upper=1>[Ns] ndt_sub_tmp;
  vector[Ns] ndt_beta_mod;
  vector[Ns] ndt_beta_tt;
  vector<lower=0>[Ns] db_intercept;
  vector[Ns] db_beta_mod;
  vector[Ns] db_beta_tt;
  vector<lower=0>[Ns] dr_intercept;
  vector[Ns] dr_beta_mod;
  vector[Ns] dr_beta_tt;
}

transformed parameters {
  vector[Ns] ndt_intercept;
  vector[Nt] db;
  vector[Nt] dr;
  vector[Nt] ndt_tmp;
  vector[Nt] ndt;
  vector[Nt] mu;
  vector[Nt] lambda;

  ndt_intercept = ndt_sub_tmp .* min_rt;

  db = db_intercept[sub_ix] .* exp(db_beta_tt[sub_ix] .* tt) .* exp(db_beta_mod[sub_ix] .* mod);
  dr = dr_intercept[sub_ix] .* exp(dr_beta_tt[sub_ix] .* tt) .* exp(dr_beta_mod[sub_ix] .* mod);

  ndt_tmp = inv_logit(ndt_intercept[sub_ix] + ndt_beta_tt[sub_ix] .* tt + ndt_beta_mod[sub_ix] .* mod);
  ndt = ndt_tmp .* min_rt[sub_ix];

  mu = db ./ dr;
  lambda = db .* db;
}

model {

  // Drift Rate Priors
  dr_group_sigma_intercept ~ normal(0, 1);
  dr_group_sigma_beta_tt ~ normal(0, 1);
  dr_group_mu_beta_tt ~ normal(0, 1);
  dr_group_sigma_beta_mod ~ normal(0, 1);
  dr_group_mu_beta_mod ~ normal(0, 1);
  dr_group_mu_intercept ~ lognormal(3.25, .8);

  dr_intercept ~ normal(dr_group_mu_intercept,
                        square(dr_group_sigma_intercept));
  dr_beta_tt ~ normal(dr_group_mu_beta_tt,
                      square(dr_group_sigma_beta_tt));
  dr_beta_mod ~ normal(dr_group_mu_beta_mod,
                      square(dr_group_sigma_beta_mod));

  // Decision Boundary Priors
  db_group_sigma_intercept ~ normal(0, 1);
  db_group_sigma_beta_tt ~ normal(0, 1);
  db_group_mu_beta_tt ~ normal(0, 1);
  db_group_sigma_beta_mod ~ normal(0, 1);
  db_group_mu_beta_mod ~ normal(0, 1);
  db_group_mu_intercept ~ lognormal(0.3, .8);

  db_intercept ~ normal(db_group_mu_intercept,
                        square(db_group_sigma_intercept));
  db_beta_tt ~ normal(db_group_mu_beta_tt,
                      square(db_group_sigma_beta_tt));
  db_beta_mod ~ normal(db_group_mu_beta_mod,
                      square(db_group_sigma_beta_mod));

  // Non-Decision Time Priors
  ndt_group_sigma_intercept ~ normal(0, 1);
  ndt_group_sigma_beta_tt ~ normal(0, 1);
  ndt_group_mu_beta_tt ~ normal(0, 1);
  ndt_group_sigma_beta_mod ~ normal(0, 1);
  ndt_group_mu_beta_mod ~ normal(0, 1);
  ndt_group_mu_intercept ~ lognormal(0.3, .3);

  ndt_intercept ~ normal(ndt_group_mu_intercept,
                         square(ndt_group_sigma_intercept));
  ndt_beta_tt ~ normal(ndt_group_mu_beta_tt,
                       square(ndt_group_sigma_beta_tt));
  ndt_beta_mod ~ normal(ndt_group_mu_beta_mod,
                        square(ndt_group_sigma_beta_mod));

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

