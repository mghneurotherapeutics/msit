data {
  int<lower=0> Nt;
  int<lower=0> Ns;
  vector[Nt] rt;
  vector[Nt] tt;
  int<lower=0> ll[Nt];
  vector[Ns] min_rt_i;
  vector[Ns] min_rt_c;
}

parameters {

  real<lower=0> scale_hyp1;
  real<lower=0> scale_hyp2;
  real<lower=0> shape_hyp1;
  real<lower=0> shape_hyp2;
  real scale_group_beta;
  real shape_group_beta;

  vector<lower=0, upper=1>[Ns] shift_base_i;
  vector<lower=0, upper=1>[Ns] shift_base_c;
  vector[Ns] scale_base;
  vector[Ns] scale_beta;
  vector[Ns] shape_base;
  vector[Ns] shape_beta;

}

transformed parameters {
  vector<lower=0>[Nt] shape;
  vector<lower=0>[Nt] scale;
  vector<lower=0>[Nt] shift;
//  for (n in 1:Nt) {
//    shape[n] = shape_base[ll[n]] + shape_beta[ll[n]] * tt[n];
//    scale[n] = scale_base[ll[n]] + scale_beta[ll[n]] * tt[n];
//    shift[n] = shift_base[tt_ix[n], ll[n]] .* min_rts[tt_ix[n], ll[n]];
//  }
  shape = shape_base[ll] + shape_beta[ll] .* tt;
  scale = scale_base[ll] + scale_beta[ll] .* tt;
  shift = shift_base_i[ll] .* tt .* min_rt_i[ll] + shift_base_c[ll] .* min_rt_c[ll] .* (1 - tt);
}

model {

  scale_hyp1 ~ gamma(1.8, 1.3);
  scale_hyp2 ~ gamma(1.8, 0.2);
  scale_base ~ gamma(scale_hyp1, scale_hyp2);

  scale_group_beta ~ normal(0, 1);
  scale_beta ~ normal(scale_group_beta, 1);

  shape_hyp1 ~ gamma(2.5, 1.0);
  shape_hyp2 ~ gamma(2.0, 0.7);
  shape_base ~ gamma(shape_hyp1, shape_hyp2);

  shape_group_beta ~ normal(0, 1);
  shape_beta ~ normal(shape_group_beta, 1);

  //for (n in 1:Nt) {
  //  rt[n] - shift[n] ~ weibull(shape[n], scale[n]);
  //}
  rt - shift ~ weibull(shape, scale);
}
