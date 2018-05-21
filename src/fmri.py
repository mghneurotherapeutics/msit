import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import coo_matrix
from mne import read_surface, spatial_tris_connectivity
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set(style='whitegrid', font_scale=1.5)


def wls(X, Y, W):
    """
    Computes a weighted least squares estimate. Returns the intercept
    and F-statistic for the intercept.

    Parameters
    ----------
    X: numpy array
        The design matrix (n_obs x n_pred)
    Y: numpy array
        The data to be predicted (n_obs x 1)
    W: numpy array
        Diagonal matrix of weights for each observation (n_obs x n_obs)

    Returns
    -------
    B[0], F: float
        The intercept beta term and associated F-statistic
    """
    B = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
    ssr = W.dot(np.power(Y - np.dot(X, B), 2)).sum()
    scale = ssr / (Y.shape[0] - X.shape[1])
    cov_p = np.linalg.inv(X.T.dot(W).dot(X)) * scale
    F = np.power(B[0], 2) * np.power(cov_p[0, 0], -1)
    return B[0], F


def compute_connectivity(space, subjects_dir, voxels):
    """
    Computes a connectivity matrix for a given fmri space.

    Parameters
    ----------
    space: str
        The fMRI space to compute connectivity for. lh, rh or mni305
    subjects_dir: str
        Path to freesurfer recons directory
    voxels: numpy array
        Voxel indices for the mni305 space data

    Returns
    -------
    coo: sparse matrix
        sparse matrix containing the connectivity for the given space
    """

    if space == 'mni305':
        n_voxels = voxels.shape[0]
        coo = np.zeros([n_voxels, n_voxels], dtype=int)
        # 6-lattice connectivity (up,down,forward,backward,left,right)
        for n in range(n_voxels):
            diff = np.linalg.norm(voxels - voxels[n], axis=1)
            M, = np.where(diff == 1.)
            for m in M:
                coo[n, m] = 1
        coo = coo_matrix(coo)
    else:
        f = '%s/fsaverage/surf/%s.white' % (subjects_dir, space)
        _, tris = read_surface(f)
        coo = spatial_tris_connectivity(tris, remap_vertices=True)

    return coo


def plot_design_matrix(fsfast_path, subject, typ):
    """
    Plots the first levels design matrix for a given subject.

    Parameters
    ----------
    fsfast_path: str
        Path to the fsfast analysis directory
    subject: str
        Subjectcode of subject to plot
    typ: str
        The type of design matrix (base or rt)

    Returns
    -------
    fig: matplotlib Figure
        The figure with multiple design matrix summary plots.
    """
    X = np.loadtxt('%s/%s/msit/%s.lh/X.dat' % (fsfast_path, subject, typ))
    reg_labels = []

    fig = plt.figure(figsize=(25, 20))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.suptitle('%s Design Matrix' % subject, y=1)
    ix = 0

    # Plot design matrix
    ax = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
    sns.heatmap(X, axes=ax, vmin=-2.5, vmax=2.5, cmap='RdBu_r')
    ax.set_yticks(())
    ax.set_xticks(())

    # Plot task regressors
    for i, c in enumerate(['Congruent', 'Incongruent']):
        ax = plt.subplot2grid((4, 4), (i, 3))
        ax.plot(X[:, i])
        ax.set_xlabel('Acquisition')
        ax.set_title(c)
        ax.set_xlim((0, len(X[:, i].T)))
        reg_labels.append(c)
        ix += 1

    # Add demean regressor
    reg_labels.append('Demean')
    ix += 1

    # Add HPF regressor
    # Number is determined by peaking at design matrix
    for k in range(2):
        reg_labels.append('HPF %s' % (k + 1))
    ix += 1

    # Plot highpass regressors
    ax = plt.subplot2grid((4, 4), (2, 3))
    for k in range(2):
        ax.plot(X[:, k + 3])
        ax.set_xlim((0, len(X[:, i].T)))
    ax.legend(reg_labels[-2:])
    ax.set_xlabel('Acquisition')
    ax.set_title('Highpass Filter')

    # Add initial volume skips
    for k in range(4):
        reg_labels.append('Skip %s' % (k + 1))
    ix += 4

    # Add scrubbing regressors
    try:
        scrubs = np.loadtxt('%s/%s/msit/001/fd_censor.par' % (fsfast_path,
                                                              subject))
        for k in range(len(scrubs)):
            reg_labels.append('Scrub %s' % k + 1)
            ix += 1
    except:
        print('No acquisitions with fd exceeding threshold')

    # Computer and plot variance inflation factor
    ax = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    ax.bar(np.arange(len(vif)), vif)
    ax.set_xticks(np.arange(len(vif)) + .4)
    ax.set_xticklabels(reg_labels)
    ax.set_ylabel('VIF')

    fig.tight_layout()
    return fig
