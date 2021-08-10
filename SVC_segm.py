import pickle, glob, sys, csv, warnings

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
from sklearn.svm import LinearSVC

from feature_extraction_utils import _load_file, _save_file, _get_node_info
from scipy.stats import multivariate_normal
from scipy.ndimage.filters import median_filter
from time import time
from mpi4py import MPI

import numpy as np

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
    def __parallel(i, n, X_tr_, Z_tr_, W_tr, theta_):
        e = np.zeros(1)
        x_tr_, z_tr_, w_tr_ = __training_dataset(X_tr_, Z_tr_, W_tr_, i)
        x_val_, z_val_, w_val_ = __validation_dataset(X_tr_, Z_tr_, W_tr_, i)
        x_tr_, x_val_ = _scaling(x_tr_, x_val_, _scaler)
        x_tr_, x_val_ = _polynomial(x_tr_,  x_val_, _degree)
        model_ = _train(x_tr_, w_tr_, theta_)[0]
        error  = _test(x_val_, w_val_, theta_, model_)
        print('>>> LOO: {}--{} j-stat: {}'.format(i, n, error))
        comm.Barrier()
        comm.Reduce(np.array(error), e, op = MPI.SUM, root = 0)
        if i_job == 0:
            return e/N_jobs
        else:
            return None
    try:
        n  = X_tr_.shape[-1]
        e_ = np.zeros((n,))
        for i in range(n):
            x_tr_, z_tr_, w_tr_ = __training_dataset(X_tr_, Z_tr_, W_tr_, i)
            x_val_, z_val_, w_val_ = __validation_dataset(X_tr_, Z_tr_, W_tr_, i)
            #x_tr_, x_val_ = _scaling(x_tr_, x_val_, _scaler)
            x_tr_, x_val_ = _polynomial(x_tr_,  x_val_, _degree)
            model_ = _train(x_tr_, w_tr_, theta_)[0]
            e_[i]  = _test(x_val_, w_val_, theta_, model_)
            print('>>> LOO: {}--{} j-stat: {}'.format(i, n, e_[i]))
        e = e_.mean()
        #e = __parallel(i_job, n, X_tr_, Z_tr_, W_tr_, theta_)
        return e
    except:
        return -1.

def _train(X_, W_, theta_):
    t_init = time()
    # Model Fit
    _SVC = LinearSVC(C = theta_[-1], penalty = 'l2', dual = False, loss = 'squared_hinge', max_iter = 1000, tol = 1e-5).fit(X_, W_)
    tm = time() - t_init
    return [_SVC, False], tm

def _test(X_, W_, theta_, model_):
    # Predict Probabilities
    Z_hat_ = _predict_proba(X_, model_)
    W_hat_ = _classify(Z_hat_, prob = theta_[0], invert_label = model_[1])
    return _scores(W_, W_hat_)[-1]

def _predict_proba(X_, model_):
    _SVC, invert_label = model_
    Z_hat_ = np.zeros((X_.shape[0], 2))
    z_ = _SVC.decision_function(X_)
    Z_hat_[:, 1] = 1./(1. + np.exp(- z_))
    Z_hat_[:, 0] = 1. - Z_hat_[:, 1]
    return Z_hat_

# Test Results for Computing Time
def _predict(X_, theta_, model_):
    X_, X_ = _scaling(X_, X_, _scaler)
    X_, _  = _polynomial(X_, X_, _degree)
    # Initial time
    t_init = time()
    # Do the segmentation
    Z_hat_ = _predict_proba(X_, model_)
    W_hat_ = _classify(Z_hat_, prob = theta_[0], invert_label = model_[-1])
    # Get this frame time
    tm = time() - t_init
    return W_hat_, tm

# No. of Validations
def _CV_SVC(X_, Z_, W_, P, C):
    # Variables Initialization
    prob_ = np.linspace(0.26, 0.74, P)
    comp_ = np.logspace(-5, -1, C)
    err_   = np.zeros((P, C))
    error_ = np.zeros((C))
    # loop Over Probabilities
    i = i_job
    for j in range(C):
        t_init = time()
        # To-CV-Parameters
        theta_ = [prob_[i], comp_[j]]
        # Fit Model to save
        error_[j] = _LOO(X_, Z_, W_, theta_)
        print('>>> CV Prob: {} Complexity: {} J-stat: {} Time: {}'.format(prob_[i], comp_[j], error_[j], time() - t_init))
    # Parallelization
    comm.Barrier()
    comm.Gatherv(sendbuf = error_, recvbuf = (err_, C), root = 0)
    if i_job == 0:
        x_, y_ = np.where(err_ == err_.max())
        return [prob_[x_[0]], comp_[y_[0]]], err_.max()
    else:
        return None, None

# Nodes and jobs information for communication from MPI
i_job, N_jobs, comm = _get_node_info(verbose = True)
# Experiment Configuration
_degree = int(sys.argv[1])
_vars   = int(sys.argv[2])
_shape  = int(sys.argv[3])
name    = r'{}{}{}'.format(_degree, _vars, _shape)

X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_ = _load_dataset(_degree, _vars, _shape, files_path = r'/users/terren/cloud_segmentation', sample = r'v241')
x_tr_, z_tr_, w_tr_, x_ts_, z_ts_, w_ts_ = _form_dataset(X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_)
# Trasform Dataset
_scaler = _get_scaler(x_tr_)
x_tr_, _ = _scaling(x_tr_, x_ts_, _scaler)
x_tr_, _ = _polynomial(x_tr_, x_ts_, _degree)

# Cross-Validate Model parameters
theta_, e_val = _CV_SVC(X_tr_, Z_tr_, W_tr_, P = N_jobs, C = 9)
if i_job == 0:
    print(theta_, e_val)
    # Train Model
    model_, t_tr = _train(x_tr_, w_tr_, theta_)
    # Test Model
    n_ts  = X_ts_.shape[-1]
    e_ts_ = np.zeros((n_ts, ))
    t_ts_ = np.zeros((n_ts, ))
    W_ts_hat_ = np.zeros(W_ts_.shape)
    for i in range(n_ts):
        # Calculate Test Scores
        W_ts_hat_[..., i], t_ts_[i] = _predict(X_ts_[..., i], theta_, model_)
        e_ts_[i] = _scores(W_ts_[..., i], W_ts_hat_[..., i])[-1]
    e_ts = e_ts_.mean()
    t_ts = t_ts_.mean()
    # Save Data
    path = r'/users/terren/cloud_segmentation/logs/{}'
    save_name = r'svc.csv'.format(path)
    x_ = [name, e_val, e_ts, t_tr, t_ts, theta_[0], theta_[1]] + e_ts_.tolist()
    print(x_)
    _write_file(x_, path = path, name = save_name)
    # Save Model
    path = r'/users/terren/cloud_segmentation/models/{}'
    save_name = r'svc_{}.pkl'.format(name)
    _save_model(C_ = [model_, theta_, _scaler, W_ts_hat_], path = path, name = save_name)
