import pickle, glob, sys, csv, warnings

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve

from feature_extraction_utils import _load_file, _save_file, _get_node_info
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.ndimage.filters import median_filter
from scipy import linalg
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
        x_tr_, x_val_ = _polynomial(x_tr_, x_val_, _degree)
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
            t_init = time()
            x_tr_, z_tr_, w_tr_ = __training_dataset(X_tr_, Z_tr_, W_tr_, i)
            x_val_, z_val_, w_val_ = __validation_dataset(X_tr_, Z_tr_, W_tr_, i)
            x_tr_, x_val_ = _polynomial(x_tr_,  x_val_, _degree)
            model_ = _train(x_tr_, w_tr_, theta_)[0]
            e_[i]  = _test(x_val_, w_val_, theta_, model_)
            print('>>> LOO: {}--{} j-stat: {} Time: {}'.format(i, n, e_[i], time() - t_init))
        e = e_.mean()
        #e = __parallel(i_job, n, X_tr_, Z_tr_, W_tr_, theta_)
        return e
    except:
        return -1.

# Regularized Sigmoid functon
def _sigmoid(w_, X_, epsilon = 1e-16):
    # sigmoid function
    z_ = 1./(1. + np.exp(- X_ @ w_))
    # Regularization
    z_[z_ < epsilon] = epsilon
    z_[z_ > 1. - epsilon] = 1. - epsilon
    return z_

# Probit Approximation of Sigmoid Function
def _probit(s_):
    return 1./np.sqrt( 1. + np.pi*s_/8. )

# Optimize Linear Gaussian Procecss a.k.a. Bayesian Logistic Regression
def _Gaussian_Process_Fit(X_, y_, gamma, n_init):
    # Compute Multivariate Noraml Prior
    def __prior(w_, mu_, Sigma_, d):
        return - d*np.log(2.*np.pi)/2. - np.log(np.linalg.det(Sigma_))/2. - (w_ - mu_).T @ np.linalg.inv(Sigma_) @ (w_ - mu_)/2.
    # Compute Binomal Likelihood
    def __likelihood(y_, y_hat_):
        return y_.T @ np.log(y_hat_) + (1. - y_).T @ np.log(1. - y_hat_)
    # Compute negative marginal log-likelihood
    def __f(theta_, X_, y_, mu_, Sigma_):
        # Get Constants
        N, d = X_.shape
        # Get Variables
        w_  = theta_
        mu_ = theta_
        # Compute Prediction
        y_hat_ = _sigmoid(w_, X_)
        # Compute Sum of log-Probability
        f = __likelihood(y_, y_hat_) + __prior(w_, mu_, Sigma_, d)
        if np.isnan(f): f = - np.inf
        return - f    # Compute Jacobian Matrix
    def __j(theta_, X_, y_, mu_, Sigma_):
        # Get Constants
        N, d = X_.shape
        # Get Variables
        w_  = theta_
        # Compute Prediction
        y_hat_  = _sigmoid(w_, X_)
        # Inverte Prior Covariance
        iSigma_ = np.linalg.inv(Sigma_)
        # Compute Gradiantes
        dw_ = w_.T @ iSigma_ - mu_.T @ iSigma_ + X_.T @ (y_hat_ - y_)
        return dw_
    # Numerical Gradient Optimization
    def __optimize(X_, y_, gamma):
        # Constants Initialization
        N, d = X_.shape
        # Variables Initialization
        mu_    = np.zeros((d,))
        Sigma_ = np.eye(d)*gamma
        w_     = multivariate_normal(mu_, Sigma_).rvs(size = 1)
        # Run optimization
        return minimize(fun = __f, jac = __j, x0 = w_, args = (X_, y_, mu_, Sigma_), method = 'L-BFGS-B')
    # Compute Posterior Covariance
    def __posterior_cov(X_, Sigma_, w_hat_):
        # Constants Initialization
        iSigma_ = np.linalg.inv(Sigma_)
        y_hat_  = _sigmoid(w_hat_, X_)
        # Variable Initialization
        Q_ = np.zeros((d, d))
        # loop Over Samples
        for i in range(y_hat_.shape[0]):
            q_ = ( y_hat_[i]*(1. - y_hat_[i]) ) * ( X_[i, :][np.newaxis].T @ X_[i, :][np.newaxis] )
            Q_ += q_
        iSigma_n_ = iSigma_ + Q_
        return np.linalg.inv(iSigma_n_)

    N, d = X_.shape
    # Results Variable Initialization
    f_ = []
    x_ = []
    # loop Over Iteration
    for i in range(n_init):
        # Run optimization
        opt_ = __optimize(X_, y_, gamma)
        f_.append(opt_['fun'])
        x_.append(opt_['x'])
    # Get Best Results Optimal Parameters
    theta_  = x_[np.argmin(f_)]
    mu_hat_ = theta_
    Sigma_  = np.eye(d)*gamma
    mu_hat_    = theta_
    Sigma_hat_ = __posterior_cov(X_, Sigma_, mu_hat_)
    return mu_hat_, Sigma_hat_

def _train(X_, W_, theta_, n_init = 3):
    t_init = time()
    # Model Fit
    w_, Sigma_ = _Gaussian_Process_Fit(X_, W_, gamma = theta_[-1], n_init = n_init)
    tm = time() - t_init
    return [[w_, Sigma_], False], tm

def _test(X_, W_, theta_, model_):
    # Predict Probabilities
    Z_hat_ = _predict_proba(X_, model_)
    W_hat_ = _classify(Z_hat_, prob = theta_[0], invert_label = model_[1])
    return _scores(W_, W_hat_)[-1]

def _predict_proba(X_, model_):
    w_, Sigma_ = model_[0]
    # Variable Initialization
    Z_hat_ = np.zeros((X_.shape[0], 2))
    # Predictive Posterior
    s_ = np.diagonal(X_ @ Sigma_ @ X_.T)
    q_ = _probit(s_)
    z_ = X_ @ w_
    # Probability of each class
    Z_hat_[:, 1] = 1./(1. + np.exp(- q_*z_))
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
def _CV_GPC(X_, Z_, W_, P, G):
    # Variables Initialization
    prob_ = np.linspace(0.26, 0.74, P)
    gamm_ = np.logspace(-5, -1, G)
    err_   = np.zeros((P, G))
    error_ = np.zeros((G))
    # loop Over Probabilities
    i = i_job
    for j in range(G):
        t_init = time()
        # To-CV-Parameters
        theta_ = [prob_[i], gamm_[j]]
        # Fit Model to save
        error_[j] = _LOO(X_, Z_, W_, theta_)
        print('>>> CV Prob: {} Gamma: {} J-stat: {} Time: {}'.format(prob_[i], gamm_[j], error_[j], time() - t_init))
    # Parallelization
    comm.Barrier()
    comm.Gatherv(sendbuf = error_, recvbuf = (err_, G), root = 0)
    if i_job == 0:
        x_, y_ = np.where(err_ == err_.max())
        return [prob_[x_[0]], gamm_[y_[0]]], err_.max()
    else:
       return None, None

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
x_tr_, _ = _scaling(x_tr_, x_ts_, _scaler)
x_tr_, _ = _polynomial(x_tr_, x_ts_, _degree)

# Cross-Validate Model
theta_, e_val = _CV_GPC(X_tr_, Z_tr_, W_tr_, P = N_jobs, G = 3)
if i_job == 0:
    print(theta_, e_val)
    # Train Model
    model_, t_tr = _train(x_tr_, w_tr_, theta_, n_init = 9)
    print(t_tr)
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
    save_name = r'gpc_v1-MPI.csv'.format(path)
    x_ = [name, _init, e_val, e_ts, t_tr, t_ts, theta_[0], theta_[1]] + e_ts_.tolist()
    print(x_)
    _write_file(x_, path = path, name = save_name)
    # Save Model
    path = r'/users/terren/cloud_segmentation/models/{}'
    save_name = r'gpc_{}-{}.pkl'.format(name, _init)
    _save_model(C_ = [model_, theta_, _scaler, W_ts_hat_], path = path, name = save_name)
