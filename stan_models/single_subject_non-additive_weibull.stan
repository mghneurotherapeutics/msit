data {
    int<lower=0> Nc;
    int<lower=0> Ni;
    vector[Nc] rt_c;
    vector[Ni] rt_i;
}

parameters {
  real<lower=0, upper=min(rt_c)> shift_c;
  real<lower=0, upper=min(rt_i)> shift_i;
  real<lower=0> shape_c;
  real<lower=0> scale_c;
  real<lower=0> shape_i;
  real<lower=0> scale_i;
}

model {
  shift_c ~ uniform(0, min(rt_c));
  shape_c ~ uniform(1, 10);
  scale_c ~ uniform(0, 5);
  shift_i ~ uniform(0, min(rt_i));
  shape_i ~ uniform(1, 10);
  scale_i ~ uniform(0, 5);
  rt_c - shift_c ~ weibull(shape_c, scale_c);
  rt_i - shift_i ~ weibull(shape_i, scale_i);
}
