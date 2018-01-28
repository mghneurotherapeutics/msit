# Create Makevars file for rstan
# Installation instructions linked here: https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
# Below is for macos using clang. You will need to change based on operating system and c compiler

dotR <- file.path(Sys.getenv("HOME"), ".R")
if (!file.exists(dotR)) dir.create(dotR)
M <- file.path(dotR, "Makevars")
if (!file.exists(M)) file.create(M)

cat("\nCXXFLAGS=-O3 -mtune=native -march=native",
    "CXXFLAGS= -Wno-unused-variable -Wno-unused-function  -Wno-macro-redefined",
    file = M, sep = "\n", append = TRUE)

cat("\nCC=clang",
    "CXX=clang++ -arch x86_64 -ftemplate-depth-256",
    file = M, sep = "\n", append = TRUE)

# Install extra stan packages not available as conda packages
install.packages("bayesplot")
install.packages("loo")
