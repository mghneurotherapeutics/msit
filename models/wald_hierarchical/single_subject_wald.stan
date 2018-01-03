functions {
  real wald_log(vector x, real gamma, real alpha, real theta) {
    real lprob;
    lprob = 0;
    for (i in 1:num_elements(x)) {
      lprob = lprob + log(alpha / sqrt(2 * pi() * pow((x[i] - theta), 3)) * exp(-(square(alpha - gamma * (x[i] - theta))) / (2 * (x[i] - theta))));
    }
    return lprob;
  }
}

data {
    int<lower=0> Nc;
    int<lower=0> Ni;
    vector[Nc] rt_c;
    vector[Ni] rt_i;
}

parameters {
  real<lower=0, upper=min(rt_c)> ndt_c;
  real<lower=0, upper=min(rt_i)> ndt_i;
  real<lower=0> dr_c;
  real<lower=0> dr_i;
  real<lower=0> db_c;
  real<lower=0> db_i;
}

model {
  rt_c ~ wald(dr_c, db_c, ndt_c);
  rt_i ~ wald(dr_i, db_i, ndt_i);
}
