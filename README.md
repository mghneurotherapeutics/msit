msit
==============================

Matt's analysis of the multisource interference task (MSIT) data for the DARPA funded TRANSFORM DBS project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- fMRI & eeg data folder according to BIDS. EEG is in the spirit of BIDS, fMRI is compliant.
    │   ├── derivatives    <- Contains pre-processed and transformed versions of the data.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Installation

### Python & R Packages

Python and R packages were managed using the Anaconda 5.0.1 distribution for Python 3.6.

With anaconda installed (https://docs.anaconda.com/anaconda/install/) you can create a copy of the msit environment provided in the environment.yml file with the following command:
  conda env create -f environment.yml

The new environment can then be accessed by typing:
  source activate msit

This will reproduce all the R and Python packages with the versions used. You should have the msit environment active before running all scripts and notebooks.

This environment includes the jupyter notebook extensions. I highly recommend enabling the Table of Contents extension so that you can navigate the notebooks with the interactive table of contents.

The final installation step is to run the r_install.R script in the src/ folder. This script finishes installation for some of the r stan packages. It first creates a Makevars file that allows stan to run with the proper c++ compiler. It then installs the loo and bayesplot packages via the r installer since they are not available as conda packages.

