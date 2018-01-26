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

Python and R were managed using the Anaconda 5.0.1 distribution for Python 3.6.

A new environment was created:
  conda create -n msit python=2.7

Python & Jupyter packages were installed:
  - conda install jupyter mayavi ipywidgets scipy pandas matplotlib seaborn statsmodels
  - pip install nibabel
  - pip install pysurfer
  - pip install mne

R packages were installed:
  - conda install -c r r-essentials
  - conda install -c mittner r-rstan
  - The loo and bayesplot stan extension packages were installed by running install.packages("bayesplot") and install.packages("loo") in a jupyter notebook running the irkernel.

Notebook Extensions were installed:
  - conda install -c conda-forge jupyter_contrib_nbextensions
  - I highly recommend enabling the table of contents so you can make use of the hierarchical structure and navigate sections easier.

An environment file was exported to environment.yml
