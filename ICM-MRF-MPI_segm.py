import pickle, glob, sys, csv, warnings, multiprocessing

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve

from feature_extraction_utils import _load_file, _save_file, _get_node_info
from scipy.stats import multivariate_normal
from scipy.ndimage.filters import median_filter
from time import time
from joblib import Parallel, delayed
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
        return x_, z_, w_
    try:
        n = X_tr_.shape[-1]
        e_ = np.zeros((n,))
        for i in range(n):
            x_tr_, z_tr_, w_tr_ = __training_dataset(X_tr_, Z_tr_, W_tr_, i)
            x_val_, z_val_, w_val_ = __validation_dataset(X_tr_, Z_tr_, W_tr_, i)
            model_ = _train(x_tr_, w_tr_, theta_)[0]
            e_[i]  = _test(x_val_, w_val_, theta_, model_)
        return e_.mean()
    except:
        return -1.

def _train(X_, W_, theta_, n_init = 3):
    t_init = time()
    # Model Fit
    X_ = X_.reshape(60, 80, X_.shape[-2], X_.shape[-1])
    _N_0, _N_1 = _Iterated_Conditional_Modes(X_, W_, _cliques, gamma = theta_[1], beta = theta_[2], n_eval = 100, n_init = n_init)
    tm = time() - t_init
    return [[_N_0, _N_1], False], tm

def _test(X_, W_, theta_, model_):
    # Predict Probabilities
    X_ = X_.reshape(60, 80, X_.shape[-1])[..., np.newaxis]
    Z_hat_ = _predict_proba(X_, _cliques, beta = theta_[-1], _N_0 = model_[0][0], _N_1 = model_[0][1], n_eval = 100)
    Z_hat_ = Z_hat_[..., 0].reshape(Z_hat_.shape[0]*Z_hat_.shape[1], Z_hat_.shape[2])
    W_hat_ = _classify(Z_hat_, prob = theta_[0], invert_label = model_[1])
    return _scores(W_, W_hat_)[-1]

# Test Results for Computing Time
def _predict(X_, theta_, model_):
    X_ = X_.reshape(60, 80, X_.shape[-1])[..., np.newaxis]
    # Initial time
    t_init = time()
    # Do the segmentation
    Z_hat_ = _predict_proba(X_, _cliques, beta = theta_[2], _N_0 = model_[0][0], _N_1 = model_[0][1], n_eval = 100)
    Z_hat_ = np.swapaxes(Z_hat_, 2, 3)
    Z_hat_ = Z_hat_.reshape(Z_hat_.shape[0]*Z_hat_.shape[1]*Z_hat_.shape[2], Z_hat_.shape[3])
    W_hat_ = _classify(Z_hat_, prob = theta_[0], invert_label = model_[-1])
    # Get this frame time
    tm = time() - t_init
    return W_hat_, tm

# ICM Optimization for Markov Random Field Model
def _Iterated_Conditional_Modes(X_, W_, _cliques, beta, gamma, n_eval, n_init):
    # Fit Multivariate Normal Distribution to each class samples
    def __class_distribution(X_, W_, M, N, D, n, gamma):
        x_ = X_.swapaxes(2, 3).reshape(M*N*n, D)
        w_ = W_.reshape(M*N*n)
        # Find each class elements
        idx_0_ = w_ == 0
        idx_1_ = w_ == 1
        # sample mean for each class
        mu_0_ = np.mean(x_[idx_0_, :], axis = 0)
        mu_1_ = np.mean(x_[idx_1_, :], axis = 0)
        # sample covariance for each class
        E_0_ = np.cov(x_[idx_0_, :].T) + np.eye(D)*gamma
        E_1_ = np.cov(x_[idx_1_, :].T) + np.eye(D)*gamma
        # Define Normal Distribution for each clasee
        return multivariate_normal(mu_0_, E_0_), multivariate_normal(mu_1_, E_1_)
    # Evaluate Likelihood
    def __likelihood(X_, _N_0, _N_1, M, N, D):
        x_ = X_.reshape(M*N, D)
        return _N_0.logpdf(x_), _N_1.logpdf(x_)
    # Energy Potential Function
    def __prior(W_, cliques_, beta, M, N):
        # Prior based on neigborhood class
        def ___neigborhood(w, W_, i, j, cliques_, beta, M, N):
            prior = 0
            # Loop over neigbors
            for clique_ in cliques_:
                k = i + clique_[0]
                m = j + clique_[1]
                if k < 0 or m < 0 or k >= M or m >= N:
                    pass
                else:
                    if w == W_[k, m]:
                        prior += beta
                    else:
                        prior -= beta
            return prior
        # Variable Initialization
        prior_0_ = np.zeros((M, N))
        prior_1_ = np.zeros((M, N))
        # Loop over Pixels in an Image
        for i in range(M):
            for j in range(N):
                # Energy function Value and Prior Probability
                prior_0_[i, j] = ___neigborhood(0, W_, i, j, cliques_, beta, M, N)
                prior_1_[i, j] = ___neigborhood(1, W_, i, j, cliques_, beta, M, N)
        return prior_0_, prior_1_
    # Evaluate Energy and Classification
    def __energy(lik_, pri_, M, N):
        # Variables Initialization
        W_ = np.zeros((M, N))
        Z_ = np.zeros((M, N, 2))
        Z_[..., 0] = lik_[0].reshape(M, N) + pri_[0]
        Z_[..., 1] = lik_[1].reshape(M, N) + pri_[1]
        # Maximum Energy Classification
        idx_ = Z_[..., 0] < Z_[..., 1]
        W_[idx_] = 1
        # Maximum Pixel's Energy
        U_ = Z_[..., 0].copy()
        U_[idx_] = Z_[idx_, 1]
        return U_, W_
    # Random Initialization
    def __rand_init(X_):
        M, N, D, n = X_.shape
        W_init_ = np.zeros((M*N, n))
        for i in range(n):
            idx_ = np.random.randint(0, M*N - 1, size = M*N//2)
            W_init_[idx_, i] = 1.
        return W_init_.reshape(M, N, n)

    def __run_icm(X_, W_, cliques_, beta, gamma, n_eval):
        # Constants Initialization
        M, N, D, n = X_.shape
        # Stopping criteria Initialization
        u_k = - np.inf
        U_k_1_ = np.zeros((M, N, n))
        W_k_1_ = np.zeros((M, N, n))
        W_k_   = __rand_init(X_)
        # Class Distribution Inference
        _N_0, _N_1 = __class_distribution(X_, W_k_, M, N, D, n, gamma)
        # Loop over No. of evaluation
        for k in range(n_eval):
            # loop over No. of samples
            for i in range(n):
                # Evaluate likelihood function
                lik_   = __likelihood(X_[..., i], _N_0, _N_1, M, N, D)
                # Evaluate prior function
                prior_ = __prior(W_k_[..., i], cliques_, beta, M, N)
                # Current Evaluation Weights Initialization
                U_k_1_[..., i], W_k_1_[..., i] = __energy(lik_, prior_, M, N)
            # Current Evaluation Total Energy
            u_k_1 = U_k_1_.sum()
            #print('>>> No Iter.: {} -- {} Prev. Energy: {} Energy: {}'.format(k, n_eval, u_k, u_k_1))
            # Stop if it is a minima
            if u_k >= u_k_1:
                break
            # If not keep optimizing
            else:
                # Update next iteration objective functions
                W_k_ = W_k_1_.copy()
                u_k  = u_k_1.copy()
                # Class Distribution Inference
                _N_0, _N_1 = __class_distribution(X_, W_k_, M, N, D, n, gamma)
        return _N_0, _N_1, u_k, _scores(W_.flatten(), W_k_.reshape(W_.shape).flatten())[-1], _scores(W_.flatten(), 1. - W_k_.reshape(W_.shape).flatten())[-1]

    # Variables Initialization
    cliques_ = [C_0_, C_1_, C_1_ + C_2_][_cliques]
    j_  = np.zeros((n_init,))
    ij_ = np.zeros((n_init,))
    u_  = np.zeros((n_init,))
    class_ = []
    for i in range(n_init):
        #print('>>> No. Init.: {} -- {}'.format(i, n_init))
        _N_0, _N_1, u_[i], j_[i], ij_[i] = __run_icm(X_, W_, cliques_, beta, gamma, n_eval)
        class_.append([_N_0, _N_1])
    i = np.argmax(u_)
    # Invert Labels?
    if j_[i] < ij_[i]:
        return [class_[i][1], class_[i][0]]
    else:
        return [class_[i][0], class_[i][1]]

# Markov Random Field Model
def _predict_proba(X_, _cliques, beta, _N_0, _N_1, n_eval):
    # Evaluate Likelhood
    def __likelihood(X_, _N_0, _N_1, M, N, D):
        x_ = X_.reshape(M*N, D)
        return _N_0.logpdf(x_), _N_1.logpdf(x_)
    # Energy Potential Function
    def __prior(W_, cliques_, beta, M, N):
        # Prior based on neigborhood class
        def ___neigborhood(w, W_, i, j, cliques_, beta, M, N):
            prior = 0
            # Loop over neigbors
            for clique_ in cliques_:
                k = i + clique_[0]
                m = j + clique_[1]
                if k < 0 or m < 0 or k >= M or m >= N:
                    pass
                else:
                    if w == W_[k, m]:
                        prior += beta
                    else:
                        prior -= beta
            return prior
        # Variable Initialization
        prior_0_ = np.zeros((M, N))
        prior_1_ = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                # Energy function Value and Prior Probability
                prior_0_[i, j] = ___neigborhood(0, W_, i, j, cliques_, beta, M, N)
                prior_1_[i, j] = ___neigborhood(1, W_, i, j, cliques_, beta, M, N)
        return prior_0_.flatten(), prior_1_.flatten()
    # Compute softmax values for each sets of scores in x
    def __softmax(x_):
        z_ = np.exp(x_)
        return z_ / np.tile(np.sum(z_, axis = 1)[:, np.newaxis], (1, 2))
    # Compute the sum of the values to add to 1
    def __hardmax(x_):
        x_ = x_ - x_.min()
        return np.nan_to_num(x_ / np.tile(np.sum(x_, axis = 1)[:, np.newaxis], (1, 2)))
    # Compute Pixels' Energy
    def __energy(lik_, pri_, M, N):
        # Variables Initialization
        Z_ = np.zeros((M*N, 2))
        W_ = np.zeros((M*N))
        # Labels Energy per pixel
        Z_[..., 0] = lik_[0] + pri_[0]
        Z_[..., 1] = lik_[1] + pri_[1]
        # Maximum Energy Classification
        idx_ = Z_[..., 0] < Z_[..., 1]
        # Compute the total energy
        U_ = Z_[..., 0].copy()
        U_[idx_] = Z_[idx_, 1]
        W_[idx_] = 1
        Z_ =__hardmax(Z_)
        return Z_.reshape(M, N, 2), W_.reshape(M, N), U_.sum()
    # Maximum Likelihood classification
    def __ML(X_, _N_0, _N_1):
        # Evaluate Likelyhood
        def __likelihood(X_, _N_0, _N_1):
            M, N, D = X_.shape
            x_ = X_.reshape(M*N, D)
            return _N_0.logpdf(x_), _N_1.logpdf(x_)
        # Classify by maximum likelihood
        def __classify(W_hat_, lik_):
            M, N = W_hat_.shape
            index_ = (lik_[0] < lik_[1]).reshape(M, N)
            W_hat_[index_] = 1
            return W_hat_
        # Variable Initialization
        M, N, D, n = X_.shape
        W_hat_ = np.zeros((M, N, n))
        # loop over Images
        for i in range(n):
            W_hat_[..., i] = __classify(W_hat_[..., i], lik_ = __likelihood(X_[..., i], _N_0, _N_1))
        return W_hat_

    # Variables Initialization
    cliques_ = [C_0_, C_1_, C_1_ + C_2_][_cliques]
    W_init_  = __ML(X_, _N_0, _N_1)
    # Constants Initialization
    M, N, D, n = X_.shape
    Z_hat_ = np.zeros((M, N, 2, n))
    W_hat_ = W_init_.copy()
    # loop over samples
    for i in range(n):
        u_k = - np.inf
        # If Inference of the distribution is necessary
        lik_ = __likelihood(X_[..., i], _N_0, _N_1, M, N, D)
        for j in range(n_eval):
            # Current Evaluation Weights Initialization
            pri_ = __prior(W_hat_[..., i], cliques_, beta, M, N)
            # Compute Probability
            Z_hat_[..., i], W_hat_[..., i], u_k_1 = __energy(lik_, pri_, M, N)
            #print('>>> No. Images.: {} -- {} No Iter.: {} -- {} Prev. Energy: {} Energy: {}'.format(i, n, j, n_eval, u_k, u_k_1))
            if u_k_1 <= u_k:
                break
            else:
                u_k = u_k_1.copy()
    return Z_hat_

# No. of Validations
def _CV_ICM_MRF(X_, Z_, W_, P, B, gamma):
    # Variables Initialization
    prob_ = np.linspace(0.26, 0.74, P)
    beta_ = np.linspace(0.1, 25., B)
    err_   = np.zeros((P, B))
    error_ = np.zeros((B))
    # loop Over Probabilities
    i = i_job
    for j in range(B):
        t_init = time()
        # To-CV-Parameters
        theta_ = [prob_[i], gamma, beta_[j]]
        # Fit Model to save
        error_[j] = _LOO(X_, Z_, W_, theta_)
        print('>>> CV Prob: {} Gamma: {} Beta: {} J-stat: {} Time: {}'.format(prob_[i], gamma, beta_[j], error_[j], time() - t_init))
    comm.Barrier()
    comm.Gatherv(sendbuf = error_, recvbuf = (err_, B), root = 0)
    if i_job == 0:
        x_, y_ = np.where(err_ == err_.max())
        return [prob_[x_[0]], gamma, beta_[y_[0]]], err_.max()
    else:
        return None, None


# Cliques
C_0_ = [[0, 0]]
C_1_ = [[0, 1], [0, -1], [1, 0], [-1, 0]]
C_2_ = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
# Nodes and jobs information for communication from MPI
i_job, N_jobs, comm = _get_node_info(verbose = True)
# Experiment Configuration
_cliques = int(sys.argv[1])
_vars    = int(sys.argv[2])
_shape   = int(sys.argv[3])
_init    = int(sys.argv[4])
name     = r'{}{}{}'.format(_cliques, _vars, _shape)

X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_ = _load_dataset(_cliques, _vars, _shape, files_path = r'/users/terren/cloud_segmentation', sample = r'v241')

# Cross-Validate Model
theta_, e_val = _CV_ICM_MRF(X_tr_, Z_tr_, W_tr_, P = N_jobs, B = 21, gamma = 1.)
if i_job == 0:
    print(theta_, e_val)
    model_, t_tr = _train(X_tr_, W_tr_, theta_, n_init = 9)
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
    save_name = r'icm-mrf-energy-MPI.csv'.format(path)
    x_ = [name, _init, e_val, e_ts, t_tr, t_ts, theta_[0], theta_[1], theta_[2]] + e_ts_.tolist()
    print(x_)
    _write_file(x_, path = path, name = save_name)
    # Save Model
    path = r'/users/terren/cloud_segmentation/models/{}'
    save_name = r'icm-mrf_{}-{}.pkl'.format(name, _init)
    _save_model(C_ = [model_, theta_, W_ts_hat_], path = path, name = save_name)
