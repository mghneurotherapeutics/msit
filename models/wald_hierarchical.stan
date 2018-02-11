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
  int<lower=0> Ni;
  int<lower=0> Nc;
  int<lower=0> Ns;
  vector<lower=0>[Ni] rt_i;
  vector<lower=0>[Nc] rt_c;
  int<lower=0> ll_c[Nc];
  int<lower=0> ll_i[Ni];
  vector<lower=0>[Ns] min_rt_i;
  vector<lower=0>[Ns] min_rt_c;
}

parameters {

  // Group Parameters

  // Drift Rate Parameters
  real<lower=.01, upper=10> dr_group_mu_congruent;
  real<lower=.01, upper=3> dr_group_sigma_congruent;
  real<lower=.01, upper=10> dr_group_mu_incongruent;
  real<lower=0.001, upper=3> dr_group_sigma_incongruent;

  // Decision Boundary Parameters
  real<lower=.01, upper=10> db_group_mu_congruent;
  real<lower=0.001, upper=3> db_group_sigma_congruent;
  real<lower=.01, upper=10> db_group_mu_incongruent;
  real<lower=0.001, upper=3> db_group_sigma_incongruent;

  // Non-Decision Time Parameters
  real<lower=0.1, upper=1> ndt_group_mu_congruent;
  real<lower=0.001, upper=.5> ndt_group_sigma_congruent;
  real<lower=0.1, upper=1> ndt_group_mu_incongruent;
  real<lower=0.001, upper=.5> ndt_group_sigma_incongruent;

  // Subject Parameters

  vector<lower=0.01, upper=1>[Ns] ndt_con;
  vector<lower=0.01, upper=1>[Ns] ndt_inc;
  vector<lower=0.01>[Ns] dr_congruent;
  vector<lower=0.01>[Ns] dr_incongruent;
  vector<lower=0.01>[Ns] db_congruent;
  vector<lower=0.01>[Ns] db_incongruent;
}

transformed parameters {
  vector[Ns] ndt_incongruent;
  vector[Ns] ndt_congruent;

  vector[Ni] mu_i;
  vector[Nc] mu_c;
  vector[Ni] lambda_i;
  vector[Nc] lambda_c;
  vector[Ni] shift_i;
  vector[Nc] shift_c;

  // transform constrained variable to actual non-decision time
  ndt_congruent = ndt_con .* min_rt_c;
  ndt_incongruent = ndt_inc .* min_rt_i;

  // Up-sampled transformed parameters to every trial
  mu_i = db_incongruent[ll_i] ./ dr_incongruent[ll_i];
  mu_c = db_congruent[ll_c] ./ dr_congruent[ll_c];
  lambda_i = db_incongruent[ll_i] .* db_incongruent[ll_i];
  lambda_c = db_congruent[ll_c] .* db_congruent[ll_c];
  shift_i = ndt_incongruent[ll_i];
  shift_c = ndt_congruent[ll_c];
}

model {

  // Drift Rate Priors
  dr_group_sigma_congruent ~ normal(0, 1);
  dr_group_sigma_incongruent ~ normal(0, 1);

  dr_congruent ~ normal(dr_group_mu_congruent,
                        square(dr_group_sigma_congruent));
  dr_incongruent ~ normal(dr_group_mu_incongruent,
                          square(dr_group_sigma_incongruent));

  // decision boundary priors
  db_group_sigma_congruent ~ normal(0, 1);
  db_group_sigma_incongruent ~ normal(0, 1);

  db_congruent ~ normal(db_group_mu_congruent,
                        square(db_group_sigma_congruent));
  db_incongruent ~ normal(db_group_mu_incongruent,
                          square(db_group_sigma_incongruent));

  // non-decision time priors
  ndt_group_sigma_congruent ~ normal(0, .1);
  ndt_group_sigma_incongruent ~ normal(0, .1);

  ndt_congruent ~ normal(ndt_group_mu_congruent,
                         square(ndt_group_sigma_congruent));
  ndt_incongruent ~ normal(ndt_group_mu_incongruent,
                           square(ndt_group_sigma_incongruent));

  // wald likelihood
  rt_c ~ wald(mu_c, lambda_c, shift_c);
  rt_i ~ wald(mu_i, lambda_i, shift_i);
}

generated quantities {
  real db_group_beta;
  real dr_group_beta;
  real ndt_group_beta;

  vector[Ns] db_beta;
  vector[Ns] dr_beta;
  vector[Ns] ndt_beta;

  vector[Nc + Ni] log_lik;

  vector[Ni] rt_i_pred;
  vector[Nc] rt_c_pred;

  // Posterior Predictions
  for (n in 1:Ni) {
    rt_i_pred[n] = wald_rng(mu_i[n], lambda_i[n], shift_i[n]);
  }
  for (n in 1:Nc) {
    rt_c_pred[n] = wald_rng(mu_c[n], lambda_c[n], shift_c[n]);
  }

  // Log likelihood for loo and waic
  log_lik = append_row(wald_ll(rt_c, mu_c, lambda_c, shift_c),
                       wald_ll(rt_i, mu_i, lambda_i, shift_i));

  // Group mean parameter differences between conditions
  db_group_beta = db_group_mu_incongruent - db_group_mu_congruent;
  dr_group_beta = dr_group_mu_incongruent - dr_group_mu_congruent;
  ndt_group_beta = ndt_group_mu_incongruent - ndt_group_mu_congruent;

  // Subject mean parameter differences between conditions
  db_beta = db_incongruent - db_congruent;
  dr_beta = dr_incongruent - dr_congruent;
  ndt_beta = ndt_incongruent - ndt_congruent;
}

