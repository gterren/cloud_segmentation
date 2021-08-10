import pickle, glob, sys, csv, warnings
import numpy as np

from feature_extraction_utils import _load_file, _save_file, _get_node_info
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn import mixture

from scipy.ndimage.filters import median_filter
from scipy.stats import multivariate_normal

from time import time

from utils import *

# Do not display warnings in the output file
warnings.filterwarnings('ignore')

def _LOO(X_tr_, Z_tr_, W_tr_, theta_):
    def __validation_dataset(X_, Z_, W_, i):
        return X_[..., i], Z_[..., i], W_[..., i]
    def __training_dataset(X_, Z_, W_, i):
        x_ = np.delete(X_, i, axis = 2)
        z_ = np.delete(Z_, i, axis = 1)
        w_ = np.delete(W_, i, axis = 1)
        x_, z_, w_ = _samples_dataset(x_, z_, w_)
        #x_, z_, w_ = _subsample_dataset(x_, z_, w_, seed = 0)
        return x_, z_, w_
    try:
        n  = X_tr_.shape[-1]
        e_ = np.zeros((n,))
        for i in range(n):
            x_tr_, z_tr_, w_tr_ = __training_dataset(X_tr_, Z_tr_, W_tr_, i)
            x_val_, z_val_, w_val_ = __validation_dataset(X_tr_, Z_tr_, W_tr_, i)
            model_ = _train(x_tr_, w_tr_, theta_)[0]
            e_[i]  = _test(x_val_, w_val_, theta_, model_)
            #print('>>> LOO: {}--{} j-stat: {}'.format(i, n, e_[i]))
        return e_.mean()
    except:
        return -1.

# Fit Multivariate Normal Distribution to each class samples
def _class_distribution(X_, Y_, gamma):
    N, D = X_.shape
    # Find each class elements
    idx_0_ = Y_ == 0
    idx_1_ = Y_ == 1
    # sample mean for each class
    mu_0_ = np.mean(X_[idx_0_], axis = 0)
    mu_1_ = np.mean(X_[idx_1_], axis = 0)
    # sample covariance for each class
    E_0_ = np.cov(X_[idx_0_].T) + np.eye(D) * gamma
    E_1_ = np.cov(X_[idx_1_].T) + np.eye(D) * gamma
    # Define Normal Distribution for each clasee
    _N_0 = multivariate_normal(mu_0_, E_0_)
    _N_1 = multivariate_normal(mu_1_, E_1_)
    prior_0_ = 0.5
    prior_1_ = 0.5
    return _N_0, _N_1, prior_0_, prior_1_

# Test Results for Computing Time
def _predict(X_, theta_, model_):
    # Loop over frames
    n = X_.shape[0]
    # Initial time
    t_init = time()
    # Do the segmentation
    Z_hat_ = _predict_proba(X_, model_)
    W_hat_ = _classify(Z_hat_, prob = theta_[0], invert_label = model_[-1])
    # Get this frame time
    tm = time() - t_init
    return W_hat_, tm

def _train(X_, W_, theta_):
    t_init = time()
    # Model Fit
    _N_0, _N_1, prior_0_, prior_1_ = _class_distribution(X_, W_, gamma = theta_[-1])    # Invert labels?
    tm = time() - t_init
    return [[_N_0, _N_1, prior_0_, prior_1_], False], tm

def _test(X_, W_, theta_, model_):
    # Predict Probabilities
    Z_hat_ = _predict_proba(X_, model_)
    W_hat_ = _classify(Z_hat_, prob = theta_[0], invert_label = model_[-1])
    return _scores(W_, W_hat_)[-1]

def _CV_GDA(X_, Z_, W_, P, G):
    # Variables Initialization
    prob_ = np.linspace(0.01, 0.99, P)
    gamm_ = np.logspace(-8, 2, G)
    err_  = np.zeros((P, G))
    # loop Over Probabilities
    for i in range(P):
        for j in range(G):
            # To-CV-Parameters
            theta_ = [prob_[i], gamm_[j]]
            # Fit Model to save
            err_[i, j] = _LOO(X_, Z_, W_, theta_)
            print('>>> CV Prob: {} Gamma: {} J-stat: {}'.format(prob_[i], gamm_[j], err_[i, j]))
    x_, y_ = np.where(err_ == err_.max())
    return [prob_[x_[0]], gamm_[y_[0]]], err_.max()

def _predict_proba(X_, model_):
    _N_0, _N_1, prior_0_, prior_1_ = model_[0]
    Z_ = np.zeros((X_.shape[0], 2))
    Z_[:, 0] = _N_0.pdf(X_) * prior_0_
    Z_[:, 1] = _N_1.pdf(X_) * prior_1_
    return Z_ / np.tile(np.sum(Z_, axis = 1)[..., np.newaxis], (1, 2))


# Nodes and jobs information for communication from MPI
i_job, N_jobs, comm = _get_node_info(verbose = True)
# Experiment Configuration
_degree = int(sys.argv[1])
_vars   = int(sys.argv[2])
_shape  = int(sys.argv[3])
name    = r'{}{}{}'.format(_degree, _vars, _shape)

X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_ = _load_dataset(_degree, _vars, _shape, files_path = r'/users/terren/cloud_segmentation', sample = r'v241')
x_tr_, z_tr_, w_tr_, x_ts_, z_ts_, w_ts_ = _form_dataset(X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_)

# Cross-Validate Model
theta_, e_val = _CV_GDA(X_tr_, Z_tr_, W_tr_, P = 99, G = 5)
print(theta_, e_val)
# Train Model
model_, t_tr = _train(x_tr_, w_tr_, theta_)
print(t_tr)
# Test Model
n_ts  = X_ts_.shape[-1]
e_ts_ = np.zeros((n_ts, ))
t_ts_ = np.zeros((n_ts, ))
W_ts_hat_ = np.zeros(W_ts_.shape)
for i in range(n_ts):
    # Calculate Test Scores
    W_ts_hat_[..., i], t_ts_[i] = _predict(X_ts_[..., i], theta_, model_)
    e_ts_[i] = _scores(W_ts_[..., i],  W_ts_hat_[..., i])[-1]
e_ts = e_ts_.mean()
t_ts = t_ts_.mean()
# Save Data
path = r'/users/terren/cloud_segmentation/logs/{}'
save_name = r'gda.csv'.format(path)
x_ = [name, e_val, e_ts, t_tr, t_ts, theta_[0], theta_[1]] + e_ts_.tolist()
print(x_)
#_write_file(x_, path = path, name = save_name)
# Save Model
path = r'/users/terren/cloud_segmentation/models/{}'
save_name = r'gda_{}.pkl'.format(name)
_save_model(C_ = [model_, theta_, W_ts_hat_], path = path, name = save_name)
