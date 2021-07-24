import pickle, glob, sys, csv, warnings
import numpy as np

from feature_extraction_utils import _load_file, _save_file, _get_node_info
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
from sklearn.cluster import KMeans

from scipy.ndimage.filters import median_filter
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
            x_tr_, x_val_ = _scaling(x_tr_, x_val_, _scaler)
            model_ = _train(x_tr_, w_tr_, theta_)[0]
            e_[i]  = _test(x_val_, w_val_, theta_, model_)
            print('>>> LOO: {}--{} j-stat: {}'.format(i, n, e_[i]))
        return e_.mean()
    except:
        return -1.

def _train(X_, W_, theta_):
    t_init = time()
    # Fit Model to save
    _KMS = KMeans(n_clusters = 2, n_init = 7, algorithm = 'full', tol = 1e-3, init = 'random')
    _KMS.fit(X_)
    # Invert labels?
    W_hat_ = _KMS.predict(X_)
    # Find out if intertion is needed
    if accuracy_score(W_, W_hat_) > accuracy_score(W_, 1. - W_hat_):
       invert_label = False
    else:
       invert_label = True
    tm = time() - t_init
    return [_KMS, invert_label], tm

def _test(X_, W_, theta_, model_):
    # Predict Probabilities
    Z_hat_ = _predict_proba(X_, model_)
    W_hat_ = _classify(Z_hat_, prob = theta_[0], invert_label = model_[1])
    return _scores(W_, W_hat_)[-1]

def _predict_proba(X_, model_):
    _KMS, invert_label = model_
    Z_hat_ = _KMS.transform(X_)
    K_ = np.tile(np.sum(Z_hat_, axis = 1)[..., np.newaxis], (1, 2))
    Z_hat_ = 1. - (Z_hat_ / K_)
    return Z_hat_

def _CV_kmeans(X_, Z_, W_, P):
    # Variables Initialization
    prob_ = np.linspace(0.01, 0.99, P)
    err_  = np.zeros((P, ))
    # loop Over Probabilities
    for i in range(P):
        # To-CV-Parameters
        theta_ = [prob_[i]]
        # Fit Model to save
        err_[i] = _LOO(X_, Z_, W_, theta_)
        print('>>> CV Prob: {} J-stat: {}'.format(prob_[i], err_[i]))
    x_ = np.where(err_ == err_.max())
    return [prob_[x_[0]]], err_.max()


# Test Results for Computing Time
def _predict(X_, theta_, model_):
    X_, X_ = _scaling(X_, X_, _scaler)
    # Initial time
    t_init = time()
    # Do the segmentation
    Z_hat_ = _predict_proba(X_, model_)
    W_hat_ = _classify(Z_hat_, prob = theta_[0], invert_label = model_[-1])
    # Get this frame time
    tm = time() - t_init
    return W_hat_, tm

# Nodes and jobs information for communication from MPI
i_job, N_jobs, comm = _get_node_info(verbose = True)
# Experiment Configuration
_degree = int(sys.argv[1])
_vars   = int(sys.argv[2])
_shape  = int(sys.argv[3])
_init   = int(sys.argv[4])
name    = r'{}{}{}'.format(_degree, _vars, _shape)

X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_ = _load_dataset(_degree, _vars, _shape, files_path = r'/users/terren/cloud_segmentation', sample = r'v241')
x_tr_, z_tr_, w_tr_, x_ts_, z_ts_, w_ts_ = _form_dataset(X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_)
# Trasform Dataset
_scaler = _get_scaler(x_tr_)
x_tr_, x_ts_ = _scaling(x_tr_, x_ts_, _scaler)

# Cross-Validate Model
theta_, e_val = _CV_kmeans(X_tr_, Z_tr_, W_tr_, P = 99)
print(theta_, e_val)
# Train Model
model_, t_tr = _train(x_tr_, w_tr_, theta_)
print(t_tr)
# Train Model
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
save_name = r'kms-scaled.csv'.format(path)
x_ = [name, _init, e_val, e_ts, t_tr, t_ts, theta_[0][0]] + e_ts_.tolist()
print(x_)
_write_file(x_, path = path, name = save_name)
# Save Model
path = r'/users/terren/cloud_segmentation/models/{}'
save_name = r'kms_{}-{}.pkl'.format(name, _init)
_save_model(C_ = [model_, theta_, _scaler, W_ts_hat_], path = path, name = save_name)
