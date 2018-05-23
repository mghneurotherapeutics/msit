MSIT
==============================

[Matt Boggess's](https://github.com/mattboggess) analysis of the multisource interference task (MSIT) data for the [DARPA-funded TRANSFORM DBS project](https://transformdbs.partners.org/). This analysis serves as a starting point for looking at the MSIT data with the intention of supporting and seeding future analyses of this data for publication.

Project Organization
------------

    ├── LICENSE
    │   
    ├── README.md
    │   
    ├── data               <- fMRI & eeg data folder according to BIDS. EEG is in the spirit of BIDS, fMRI is compliant.
    │   ├── derivatives    <- Contains pre-processed and transformed versions of the data.
    │   
    ├── environment.yml    <- Anaconda environment file for reproducing python environment.
    │   
    ├── notebooks                              <- Analysis notebooks for the projects. 
    │   ├── experiment_config.json             <- Contains analysis parameters
    │   ├── B1-behavior_preprocessing.ipynb    <- Notebook doing QC and preprocessing of the behavior data. 
    │   ├── B2-behavior_EDA.ipynb              <- Exploratory analysis of the behavior data. B1, E1, and F1 need to be run prior. 
    │   ├── B3-behavior_modeling.ipynb         <- Bayesian modeling of the behavior response time distributions. B1, E1, and F1 need to be run prior. B2 should be run prior. 
    │   ├── E1-eeg_preprocessing.ipynb         <- QC and preprocessing of the EEG data. 
    │   ├── E2-eeg_analysis.ipynb              <- ERP and TFR Power analyses of the EEG data. B1, E1, and F1 need to be run prior. 
    │   ├── F1-fmri_preprocessing.ipynb        <- QC and preprocessing of the fMRI data. 
    │   ├── F2-fmri_analysis.ipynb             <- GLM analysis of the fmri data. 
    │   
    ├── src                <- Source code referenced in the notebooks.
    │   ├── eeg.py         <- Contains helper functions for the EEG analysis.
    │   ├── behavior.py    <- Contains helper functions for the behavior analysis.
    │   ├── fmri.py        <- Contains helper functions for the fMRI analysis. 
    │   ├── utils.py       <- Contains utility functions for managing subject exclusions.
    │   
    ├── models                      <- Stan model files for the behavior response time models. 
    │   ├── wald_hierarchial        <- Contains stan model file (wald_hierarchical.stan) and kruschke diagram of model (wald_hierachical.png).


Computing Environment & Software Installation
--------

### Computing Environment

The project was completed on a Linux machine running Centos 7 with 64 GB of RAM and 24 cores available. The full dataset, including all derived outputs, requires 900 GB of space. This project is quite resource intensive and many parts of the project still take hours or even a few days to run making use of this RAM and core count.

### Python Environment

Python packages were managed using the Anaconda 5.0.1 distribution for Python 3.6.

With [anaconda installed](https://docs.anaconda.com/anaconda/install/) you can create a copy of the msit environment provided in the environment.yml file with the following command:

  conda env create -f environment.yml

The new environment can then be accessed by typing:

  source activate msit

This will reproduce all the Python packages with the versions used. You should have the msit environment active before running all notebooks.

I also highly recommend installing the [jupyter notebook extensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions). These notebooks were written with the intention of using the tables of contents notebook extension to navigate through the different sections.

### Other Software 

[Docker](https://www.docker.com/) was used for QC and preprocessing of the fMRI data. It would need to be installed prior to running these commands.

[Freesurfer 6.0](https://surfer.nmr.mgh.harvard.edu/) was used for the fmri first levels analysis. Specifically, the [fsfast toolkit](https://surfer.nmr.mgh.harvard.edu/fswiki/FsFast) packaged with Freesurfer 6.0 was used. 

Data
--------

The data for this project is not currently publicly available.
