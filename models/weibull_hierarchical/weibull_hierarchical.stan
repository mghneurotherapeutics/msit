data {
  int<lower=0> Nt;
  int<lower=0> Ns;
  vector[Nt] rt;
  vector[Nt] tt;
  int<lower=0> ll[Nt];
  vector[Nt] min_rts;
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

  // Shift Parameters
  real <lower=0, upper=1> mu_beta0_shift;
  real <lower=0> sigma_beta0_shift;
  real mu_beta1_shift;
  real<lower=0> sigma_beta1_shift;

  // Subject Parameters
  vector<lower=0, upper=1>[Ns] beta0_shift;
  vector[Ns] beta1_shift;
  vector<lower=0>[Ns] beta0_scale;
  vector[Ns] beta1_scale;
  vector<lower=0>[Ns] beta0_shape;
  vector[Ns] beta1_shape;

}

transformed parameters {
  vector<lower=0>[Nt] shape;
  vector<lower=0>[Nt] scale;
  vector<lower=0, upper=1>[Nt] shift_base;
  vector<lower=0>[Nt] shift;

  shape = beta0_shape[ll] + beta1_shape[ll] .* tt;
  scale = beta0_scale[ll] + beta1_scale[ll] .* tt;
  shift_base = beta0_shift[ll] + beta1_shift[ll] .* tt;
  shift = shift_base .* min_rts;
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

  // Shift Priors

  sigma_beta0_shift ~ normal(0, 1);
  beta0_shift ~ normal(mu_beta0_shift, sigma_beta0_shift);

  mu_beta1_shift ~ normal(0, 1);
  sigma_beta1_shift ~ normal(0, 1);
  beta1_shift ~ normal(mu_beta1_shift, sigma_beta1_shift);

  // Likelihood

  rt - shift ~ weibull(shape, scale);
}
