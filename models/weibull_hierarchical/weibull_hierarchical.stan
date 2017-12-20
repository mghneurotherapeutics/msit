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
  real<lower=0> beta0_scale_group1;
  real<lower=0> beta0_scale_group2;
  real<lower=0> beta0_shape_group1;
  real<lower=0> beta0_shape_group2;
  real<lower=0, upper=1> beta0_shift_group;
  real beta1_scale_group;
  real beta1_shape_group;
  real beta1_shift_group;

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

  beta0_scale_group1 ~ gamma(1.8, 1.3);
  beta0_scale_group2 ~ gamma(1.8, 0.2);
  beta0_scale ~ gamma(beta0_scale_group1, beta0_scale_group2);

  beta1_scale_group_mu ~ normal(0, 1);
  // prior for group sigma should go here
  // beta1_scale_group_sigma ~ gamma(?, ?)
  beta1_scale_group ~ normal(beta1_scale_group_mu, 1);

  // Shape Priors

  beta0_shape_group1 ~ gamma(2.5, 1.0);
  beta0_shape_group2 ~ gamma(2.0, 0.7);
  beta0_shape ~ gamma(beta0_shape_group1, beta0_shape_group2);

  beta1_shape_group_mu ~ normal(0, 1);
  // prior for group sigma should go here
  // beta1_shape_group_sigma ~ gamma(?, ?)
  beta1_shape ~ normal(beta1_shape_group_mu, 1);

  // Shift Priors

  // Likelihood

  rt - shift ~ weibull(shape, scale);
}
