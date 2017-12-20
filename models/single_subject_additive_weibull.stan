data {
    int<lower=0> N;
    vector[N] rt;
    vector[N] tt;
}

parameters {
  real<lower=0, upper=min(rt)> shift;
  real beta1_scale;
  real beta0_scale;
  real beta1_shape;
  real beta0_shape;
}

transformed parameters {
  vector<lower=0>[N] shape;
  vector<lower=0>[N] scale;
  shape = beta0_shape + beta1_shape * tt;
  scale = beta0_scale + beta1_scale * tt;
}

model {
  rt - shift ~ weibull(shape, scale);
}
