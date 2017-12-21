data {
  int<lower=0> Nt;
  int<lower=0> Ns;
  vector<lower=0>[Nt] rt;
  vector[Nt] tt;
  int<lower=0> ll[Nt];
  vector<lower=0>[Ns] min_rt_i;
  vector<lower=0>[Ns] min_rt_c;
}

parameters {

  // Group Parameters

  // Scale Parameters
  real<lower=0> k_beta0_scale;
  real<lower=0> theta_beta0_scale;
  real mu_beta1_scale;
  real<lower=0> sigma_beta1_scale;

  // Shape Parameters
  real<lower=0> k_beta0_shape;
  real<lower=0> theta_beta0_shape;
  real mu_beta1_shape;
  real<lower=0> sigma_beta1_shape;

  // Subject Parameters
  vector<lower=0, upper=1>[Ns] beta_con_shift;
  vector<lower=0, upper=1>[Ns] beta_inc_shift;
  vector<lower=0>[Ns] beta0_scale;
  vector[Ns] beta1_scale;
  vector<lower=0>[Ns] beta0_shape;
  vector[Ns] beta1_shape;

}

transformed parameters {
  vector<lower=0>[Nt] shape;
  vector<lower=0>[Nt] scale;
  vector<lower=0>[Nt] shift;

  shape = beta0_shape[ll] + beta1_shape[ll] .* tt;
  scale = beta0_scale[ll] + beta1_scale[ll] .* tt;
  shift = beta_inc_shift[ll] .* min_rt_i[ll] .* tt + beta_con_shift[ll] .* min_rt_c[ll] .* (1 - tt);
}

model {

  // Scale Priors

  k_beta0_scale ~ gamma(1.8, 1.3);
  theta_beta0_scale ~ gamma(1.8, 0.2);
  beta0_scale ~ gamma(k_beta0_scale, theta_beta0_scale);

  mu_beta1_scale ~ normal(0, 1);
  sigma_beta1_scale ~ normal(0, 1);
  beta1_scale ~ normal(mu_beta1_scale, sigma_beta1_scale);

  // Shape Priors

  k_beta0_shape ~ gamma(2.5, 1.0);
  theta_beta0_shape ~ gamma(2.0, 0.7);
  beta0_shape ~ gamma(k_beta0_shape, theta_beta0_shape);

  mu_beta1_shape ~ normal(0, 1);
  sigma_beta1_shape ~ normal(0, 1);
  beta1_shape ~ normal(mu_beta1_shape, sigma_beta1_shape);

  // Likelihood

  rt - shift ~ weibull(shape, scale);
}

generated quantities {
  vector[Ns] beta1_shift;
  vector<lower=0>[Ns] beta0_shift;
  beta0_shift = min_rt_c .* beta_con_shift;
  beta1_shift = min_rt_i .* beta_inc_shift - min_rt_c .* beta_con_shift;
}
