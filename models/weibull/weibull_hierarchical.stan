functions {
  real calc_mode(real k, real theta) {
    real mode;
    mode = (k - 1) / theta;
    return mode;
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

  // Scale Parameters

  real<lower=.001> scale_group_k_congruent;
  real<lower=.001> scale_group_theta_congruent;
  real<lower=.001> scale_group_k_incongruent;
  real<lower=.001> scale_group_theta_incongruent;

  real<lower=.001> shape_group_k_congruent;
  real<lower=.001> shape_group_theta_congruent;
  real<lower=.001> shape_group_k_incongruent;
  real<lower=.001> shape_group_theta_incongruent;

  real<lower=.001> shift_group_k_congruent;
  real<lower=.001> shift_group_theta_congruent;
  real<lower=.001> shift_group_k_incongruent;
  real<lower=.001> shift_group_theta_incongruent;

  // Subject Parameters

  vector<lower=.001, upper=1>[Ns] shift_inc;
  vector<lower=.001, upper=1>[Ns] shift_con;
  vector<lower=.001>[Ns] scale_congruent;
  vector<lower=.001>[Ns] scale_incongruent;
  vector<lower=.001>[Ns] shape_congruent;
  vector<lower=.001>[Ns] shape_incongruent;

}

transformed parameters {

  vector<lower=0>[Ns] shift_incongruent;
  vector<lower=0>[Ns] shift_congruent;

  shift_incongruent = shift_inc .* min_rt_i;
  shift_congruent = shift_con .* min_rt_c;

}

model {

  vector[Nc] shape_c;
  vector[Nc] scale_c;
  vector[Nc] shift_c;
  vector[Ni] shape_i;
  vector[Ni] scale_i;
  vector[Ni] shift_i;


  // Shape Priors

  shape_group_k_congruent ~ gamma(51, 7);
  shape_group_k_incongruent ~ gamma(51, 7);

  shape_group_theta_congruent ~ gamma(11, 3.3);
  shape_group_theta_incongruent ~ gamma(11, 3.3);

  shape_incongruent ~ gamma(shape_group_k_incongruent, shape_group_theta_incongruent);
  shape_congruent ~ gamma(shape_group_k_congruent, shape_group_theta_congruent);

  // Scale Priors

  scale_group_k_congruent ~ gamma(9, 2);
  scale_group_k_incongruent ~ gamma(9, 2);

  scale_group_theta_congruent ~ gamma(22.2, 2.4);
  scale_group_theta_incongruent ~ gamma(22.2, 2.4);

  scale_incongruent ~ gamma(scale_group_k_incongruent, scale_group_theta_incongruent);
  scale_congruent ~ gamma(scale_group_k_congruent, scale_group_theta_congruent);

  // Shift Priors

  shift_group_k_congruent ~ gamma(18, 2.8);
  shift_group_k_incongruent ~ gamma(18, 2.8);

  shift_group_theta_congruent ~ gamma(22.2, 2.4);
  shift_group_theta_incongruent ~ gamma(22.2, 2.4);

  shift_incongruent ~ gamma(shift_group_k_incongruent, shift_group_theta_incongruent);
  shift_congruent ~ gamma(shift_group_k_congruent, shift_group_theta_congruent);


  // Likelihood

  shape_c = shape_congruent[ll_c];
  shape_i = shape_incongruent[ll_i];
  scale_c = scale_congruent[ll_c];
  scale_i = scale_incongruent[ll_i];
  shift_i = shift_incongruent[ll_i];
  shift_c = shift_congruent[ll_c];
  rt_i - shift_i ~ weibull(shape_i, scale_i);
  rt_c - shift_c ~ weibull(shape_c, scale_c);
}

generated quantities {
  vector[Ns] beta_shift;
  vector[Ns] beta_shape;
  vector[Ns] beta_scale;

  real scale_group_mode_incongruent;
  real scale_group_mode_congruent;
  real shape_group_mode_incongruent;
  real shape_group_mode_congruent;
  real shift_group_mode_incongruent;
  real shift_group_mode_congruent;

  real group_beta_shape;
  real group_beta_scale;
  real group_beta_shift;

  scale_group_mode_incongruent = calc_mode(scale_group_k_incongruent, scale_group_theta_incongruent);
  scale_group_mode_congruent = calc_mode(scale_group_k_congruent, scale_group_theta_congruent);
  shape_group_mode_incongruent = calc_mode(shape_group_k_incongruent, shape_group_theta_incongruent);
  shape_group_mode_congruent = calc_mode(shape_group_k_congruent, shape_group_theta_congruent);
  shift_group_mode_incongruent = calc_mode(shift_group_k_incongruent, shift_group_theta_incongruent);
  shift_group_mode_congruent = calc_mode(shift_group_k_congruent, shift_group_theta_congruent);

  group_beta_scale = scale_group_mode_incongruent - scale_group_mode_congruent;
  group_beta_shape = shape_group_mode_incongruent - shape_group_mode_congruent;
  group_beta_shift = shift_group_mode_incongruent - shift_group_mode_congruent;

  beta_shift = shift_incongruent - shift_congruent;
  beta_scale = scale_incongruent - scale_congruent;
  beta_shape = shape_incongruent - shape_congruent;
}

