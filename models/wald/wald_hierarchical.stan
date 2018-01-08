functions {
  real wald_log(vector x, vector gamma, vector alpha, vector theta) {
    vector[num_elements(x)] prob;
    vector[num_elements(x)] sx;
    sx = x - theta;
    prob = (alpha ./ sqrt(2 * pi() * sx .* sx .* sx)) .* exp(-((alpha - gamma .* sx) .* (alpha - gamma .* sx)) ./ (2 * sx));
    return sum(log(prob));
  }
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

  // Drift Rate Parameters
  real<lower=0.001> dr_group_k_congruent;
  real<lower=0.001> dr_group_theta_congruent;
  real<lower=0.001> dr_group_k_incongruent;
  real<lower=0.001> dr_group_theta_incongruent;

  // Decision Boundary Parameters
  real<lower=0.001> db_group_k_congruent;
  real<lower=0.001> db_group_theta_congruent;
  real<lower=0.001> db_group_k_incongruent;
  real<lower=0.001> db_group_theta_incongruent;

  // Non-Decision Time Parameters
  real<lower=0.001> ndt_group_k_congruent;
  real<lower=0.001> ndt_group_theta_congruent;
  real<lower=0.001> ndt_group_k_incongruent;
  real<lower=0.001> ndt_group_theta_incongruent;

  // Subject Parameters

  vector<lower=0.001, upper=1>[Ns] ndt_inc;
  vector<lower=0.001, upper=1>[Ns] ndt_con;
  vector<lower=0.001>[Ns] dr_congruent;
  vector<lower=0.001>[Ns] dr_incongruent;
  vector<lower=0.001>[Ns] db_congruent;
  vector<lower=0.001>[Ns] db_incongruent;

}

transformed parameters {
  vector[Ns] ndt_incongruent;
  vector[Ns] ndt_congruent;

  ndt_incongruent = ndt_inc .* min_rt_i;
  ndt_congruent = ndt_con .* min_rt_c;
}

model {


  // Up-sampled parameters to full trial count for entry into likelihood
  vector[Nc] gamma_c;
  vector[Nc] alpha_c;
  vector[Nc] theta_c;
  vector[Ni] gamma_i;
  vector[Ni] alpha_i;
  vector[Ni] theta_i;

  // Drift Rate Priors

  dr_group_k_congruent ~ gamma(12.5, 1.8);
  dr_group_k_incongruent ~ gamma(12.5, 1.8);

  dr_group_theta_congruent ~ gamma(1.8, 0.7);
  dr_group_theta_incongruent ~ gamma(1.8, 0.7);

  dr_congruent ~ gamma(dr_group_k_congruent, dr_group_theta_congruent);
  dr_incongruent ~ gamma(dr_group_k_incongruent, dr_group_theta_incongruent);

  // Decision Boundary Priors

  db_group_k_congruent ~ gamma(3.3, 0.9);
  db_group_k_incongruent ~ gamma(3.3, 0.9);

  db_group_theta_congruent ~ gamma(1.35, 0.6);
  db_group_theta_incongruent ~ gamma(1.35, 0.6);

  db_congruent ~ gamma(db_group_k_congruent, db_group_theta_congruent);
  db_incongruent ~ gamma(db_group_k_incongruent, db_group_theta_incongruent);

  // Non-Decision Time Priors

  ndt_group_k_congruent ~ gamma(2.6, .8);
  ndt_group_k_incongruent ~ gamma(2.6, .8);

  ndt_group_theta_congruent ~ gamma(6.9, 1.3);
  ndt_group_theta_incongruent ~ gamma(6.9, 1.3);

  ndt_congruent ~ gamma(ndt_group_k_congruent, ndt_group_theta_congruent);
  ndt_incongruent ~ gamma(ndt_group_k_incongruent, ndt_group_theta_incongruent);

  // Wald Likelihood

  gamma_c = dr_congruent[ll_c];
  gamma_i = dr_incongruent[ll_i];
  alpha_c = db_congruent[ll_c];
  alpha_i = db_incongruent[ll_i];
  theta_i = ndt_incongruent[ll_i];
  theta_c = ndt_congruent[ll_c];
  rt_i ~ wald(gamma_i, alpha_i, theta_i);
  rt_c ~ wald(gamma_c, alpha_c, theta_c);
}

generated quantities {

  vector[Ns] beta_ndt;
  vector[Ns] beta_db;
  vector[Ns] beta_dr;

  real dr_group_mode_incongruent;
  real dr_group_mode_congruent;
  real db_group_mode_incongruent;
  real db_group_mode_congruent;
  real ndt_group_mode_incongruent;
  real ndt_group_mode_congruent;

  real group_beta_ndt;
  real group_beta_db;
  real group_beta_dr;

  dr_group_mode_incongruent = calc_mode(dr_group_k_incongruent, dr_group_theta_incongruent);
  dr_group_mode_congruent = calc_mode(dr_group_k_congruent, dr_group_theta_congruent);
  db_group_mode_incongruent = calc_mode(db_group_k_incongruent, db_group_theta_incongruent);
  db_group_mode_congruent = calc_mode(db_group_k_congruent, db_group_theta_congruent);
  ndt_group_mode_incongruent = calc_mode(ndt_group_k_incongruent, ndt_group_theta_incongruent);
  ndt_group_mode_congruent = calc_mode(ndt_group_k_congruent, ndt_group_theta_congruent);

  beta_ndt = ndt_incongruent - ndt_congruent;
  beta_dr = dr_incongruent - dr_congruent;
  beta_db = db_incongruent - db_congruent;

  group_beta_ndt = ndt_group_mode_incongruent - ndt_group_mode_congruent;
  group_beta_dr = dr_group_mode_incongruent - dr_group_mode_congruent;
  group_beta_db = db_group_mode_incongruent - db_group_mode_congruent;

}




