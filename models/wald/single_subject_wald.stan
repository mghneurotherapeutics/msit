functions {
  real wald_log(vector x, real gamma, real alpha, real theta) {
    vector[num_elements(x)] prob;
    vector[num_elements(x)] sx;
    sx = x - theta;
    prob = (alpha ./ sqrt(2 * pi() * sx .* sx .* sx)) .* exp(-((alpha - gamma .* sx) .* (alpha - gamma .* sx)) ./ (2 .* sx));
    return sum(log(prob));
  }
}

data {
    int<lower=0> Nc;
    int<lower=0> Ni;
    vector[Nc] rt_c;
    vector[Ni] rt_i;
    real mini;
}

parameters {
  real<lower=0, upper=mini> ndt;
  real<lower=0.001, upper=100> dr_c;
  real<lower=0.001, upper=100> dr_i;
  real<lower=0.001, upper=100> db_c;
  real<lower=0.001, upper=100> db_i;
}

model {
  rt_c ~ wald(dr_c, db_c, ndt);
  rt_i ~ wald(dr_i, db_i, ndt);
}
