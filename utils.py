import pickle, glob, sys, csv

from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve

from feature_extraction_utils import _load_file, _save_file, _get_node_info
from scipy.stats import multivariate_normal
from scipy.ndimage.filters import median_filter
from sklearn import preprocessing

from time import time

import numpy as np

# Save model in path with name
def _save_model(C_, path, name):
    with open(path.format(name), 'wb') as f:
        pickle.dump(C_, f)

def _write_file(C_, path, name):
    with open(path.format(name), 'a', newline = '\n') as f:
        writer = csv.writer(f)
        writer.writerow(C_)

# Reviver operating caracteristics
def _ROC(W_, W_hat_):
    C_ = confusion_matrix(W_hat_, W_, labels = [0, 1])
    TP, FP, FN, TN = C_[0, 0], C_[0, 1], C_[1, 0], C_[1, 1]
    return TP, FP, FN, TN

# Compute the sum of the values to add to 1
def _hardmax(x_):
    x_ = x_ - x_.min()
    return np.nan_to_num(x_ / np.tile(np.sum(x_, axis = 1)[:, np.newaxis], (1, 2)))

def _classify(Z_, prob, invert_label):
    # Variables Initizalization
    labels_ = np.zeros((Z_.shape[0]))
    # Maximum Likelihood Classification
    idx_ = Z_[:, 0] * prob < Z_[:, 1] * (1. - prob)
    labels_[idx_] = 1.
    if invert_label:
        return 1. - labels_
    else:
        return labels_

# Test Scores
def _scores(W_, W_hat_):
    # False Positve Range and True Positive Rate
    def __FPR_TPR(roc_):
        # Coordinates TP, FP, FN, TN
        sen_ = roc_[0]/(roc_[0] + roc_[2])
        spe_ = roc_[3]/(roc_[3] + roc_[1])
        return np.nan_to_num(sen_, nan = 1.), np.nan_to_num(spe_, nan = 1.)
    e = accuracy_score(W_, W_hat_)
    roc_ = _ROC(W_, W_hat_)
    sen_, spe_ = __FPR_TPR(roc_)
    return e, roc_, sen_ + spe_ - 1.

# Select Image Samples
def _select_samples(X_, Z_, W_, sample_idx_):
    return X_[..., sample_idx_], Z_[..., sample_idx_], W_[..., sample_idx_]

def _get_dataset(X_, Z_, W_, var_idx_, str_idx_):
    def __median_padding(x_, M, N, D, n):
        I_ = np.zeros((M + 2, N + 2, D, n))
        # Zero-pad image
        I_[1:-1,1:-1, ...] = x_
        # loop over images
        for i in range(n):
            # loop over dimension
            for j in range(D):
                # Aply median filter
                I_prime_ = median_filter(I_[..., j, i], size = 7, mode = 'reflect')
                # Median fileter Papped Image
                I_[0, :, j, i]  = I_prime_[0, :]
                I_[-1, :, j, i] = I_prime_[-1, :]
                I_[:, 0, j, i]  = I_prime_[:, 0]
                I_[:, -1, j, i] = I_prime_[:, -1]
        return I_
    # Grab Data only desired Features
    x_ = X_[..., var_idx_, :].copy()
    # Variables Initialization
    M, N, D, n = x_.shape
    X_ = np.zeros((0, len(var_idx_)*np.sum(str_idx_), n))
    # Image padding with madian filter
    I_ = __median_padding(x_, M, N, D, n)
    # Loop over Pixels
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            # Grab the entired 2nd order neiborhood
            w_ = I_[i - 1:i + 2, j - 1:j + 2, ...].copy()
            # Grab desired Neiborhood
            x_ = w_[str_idx_, ...]
            # Concatenate Pixel Features Vector
            X_ = np.concatenate((X_, x_.reshape(x_.shape[0]*x_.shape[1], x_.shape[2])[np.newaxis, ...]), axis = 0)
    # Reshape Segmentation and Objective Matrix
    Z_ = Z_.reshape(M*N, n)
    W_ = W_.reshape(M*N, n)
    return X_, Z_, W_

# Set all images in a single dataset
def _samples_dataset(X_, Z_, W_):
    N, D, n = X_.shape
    return X_.swapaxes(1, 2).reshape(N*n, D), Z_.reshape(N*n), W_.reshape(N*n)

# Data Normalization Between 0 and 1
def _get_norm(X_, Z_, W_tr):
    X_, Z_, W_ = _samples_dataset(X_, Z_, W_)
    return preprocessing.MinMaxScaler().fit(X_)

# Data Standardization
def _get_scaler(X_, Z_, W_):
    X_, Z_, W_ = _samples_dataset(X_, Z_, W_)
    return preprocessing.StandardScaler().fit(X_)

# Apply Scaling
def _scaling(x_tr_, x_ts_, _scale):
    _scale.transform(x_tr_)
    _scale.transform(x_ts_)
    return x_tr_, x_ts_

# Load images data
def _load_images(file_names_, M = 60, N = 80, D = 10):
    n = len(file_names_)
    # Variables Initialization
    x_ = np.zeros((M, N, D, n))
    z_ = np.zeros((M, N, n))
    w_ = np.zeros((M, N, n))
    # Loop over files
    for file_, i in zip(file_names_, range(n)):
        print(i, file_)
        # Load Image Files
        ii, X_, Z_, W_ = _load_file(file_)[0]
        # Concatenate All image files
        x_[..., i] = np.concatenate((X_[0][..., np.newaxis], X_[1][..., np.newaxis], X_[2][..., np.newaxis], X_[3][..., np.newaxis],
                                     X_[4][..., np.newaxis] - 273.15, X_[5][..., np.newaxis]/1000., # Kelvin to Celsius and meters to km
                                     X_[6][..., np.newaxis] - 273.15, X_[7][..., np.newaxis]/1000.,
                                     X_[8][..., np.newaxis] - 273.15, X_[9][..., np.newaxis]/1000.), axis = 2)
        z_[..., i] = Z_
        w_[..., i] = W_ > 0
        # Make Sure
        if (1. - w_[..., i]).sum() == 1.:
            w_[..., i] = np.ones(W_.shape)
        if w_[..., i].sum() == 1.:
            w_[..., i] = np.zeros(W_.shape)
    return x_, z_, w_

def _load_dataset(_degree, _vars, _shape, files_path, sample):
    # Pixels Structures
    idx_0_ = np.matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype = bool)
    idx_1_ = np.matrix([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = bool)
    idx_2_ = np.matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype = bool)
    # I_2_norm_, M_, I_scatter_, I_diffuse_, K_0_, H_0_, K_1_, H_1_, K_2_, H_2_
    var_idx_ = [[4, 5], [6, 7], [3, 9], [0, 1, 3]][_vars]
    #var_idx_ = [[0], [0, 3], [0, 3, 9], [0, 3, 9, 1]][_vars]
    #var_idx_ = [[3], [0, 3], [0, 1, 3], [0, 1, 3, 9], [0, 1, 3, 8, 9]][_vars]
    #var_idx_ = [[9], [0, 9], [0, 3, 9], [0, 3, 8, 9], [0], [3], [8], [9]][_vars]
    #var_idx_ = [[9], [0, 9], [3, 9], [8, 9], [0, 3, 9], [0, 8, 9], [3, 8, 9], [0, 3, 8, 9], [0], [3], [8], [9]][_vars]
    str_idx_ = [idx_0_, idx_1_, idx_2_][_shape]
    print('_Degree: {} Variable: {} Neiborhood Structure: {}'.format(_degree, var_idx_, str_idx_))

    name = r'{}/samples/{}/*.pkl'.format(files_path, sample)
    # load Images
    x_, z_, w_ = _load_images(file_names_ = sorted(glob.glob(name)))
    #print(x_.shape, z_.shape, w_.shape)
    # Process Dataset
    X_, Z_, W_ = _get_dataset(x_, z_, w_, var_idx_, str_idx_)
    #print(X_.shape, Z_.shape, W_.shape)
    # 16-Images
    # Sequential Order: [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13] - [3, 4, 5, 14, 15]
    # Minimum KL {0, 3, 8, 9} : [0, 1, 2, 4, 6, 9, 10, 12, 13, 14, 15] - [3, 5, 7, 8, 11]
    # 15-Images
    # Minimum KL {0, 1, 3, 8, 9} : [0, 2, 4, 6, 9, 10, 12, 13, 14, 15] - [3, 5, 7, 8, 11]
    # Minimum KL {0, 1, 3, 8, 9} : [1, 3, 4, 6, 7, 8, 9, 10, 12, 13] - [0, 2, 5, 11, 15]
    # 8-8-Images
    # Minimum KL {0, 1, 3, 9} : [1, 2, 6, 8, 11, 12, 13, 14] - [0, 3, 4, 5, 7, 9, 10, 15]
    # 10-6-Images
    # Minimum KL {0, 1, 3, 9} : [0, 2, 4, 6, 10, 11, 12, 13, 14, 15] - [1, 3, 5, 7, 8, 9]
    # [2, 8, 12, 0, 3, 5, 9, 7] - [4, 15, 10, 1, 14]
    # [2, 6, 8, 12, 3, 9, 7] - [15, 10, 1, 4, 14]
    X_tr_, Z_tr_, W_tr_ = _select_samples(X_, Z_, W_, sample_idx_ = [2, 6, 8, 12, 3, 9, 7])
    print(X_tr_.shape, Z_tr_.shape, W_tr_.shape)
    # Select Image Samples for test
    X_ts_, Z_ts_, W_ts_ = _select_samples(X_, Z_, W_, sample_idx_ = [15, 10, 1, 4, 14])
    print(X_ts_.shape, Z_ts_.shape, W_ts_.shape)
    return X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_

def _form_dataset(X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_):
    # Define Training Dataset
    X_tr_, Z_tr_, W_tr_ = _samples_dataset(X_tr_, Z_tr_, W_tr_)
    print(X_tr_.shape, Z_tr_.shape, W_tr_.shape)
    # Define Validation Dataset
    # X_val_, Z_val_, W_val_ = _samples_dataset(X_val_, Z_val_, W_val_)
    # print(X_val_.shape, Z_val_.shape, W_val_.shape)
    # Define Test Dataset
    X_ts_, Z_ts_, W_ts_ = _samples_dataset(X_ts_, Z_ts_, W_ts_)
    print(X_ts_.shape, Z_ts_.shape, W_ts_.shape)
    return X_tr_, Z_tr_, W_tr_, X_ts_, Z_ts_, W_ts_

def _polynomial(x_tr_, x_ts_, degree):
    x_tr_ = PolynomialFeatures(degree).fit_transform(x_tr_)
    x_ts_ = PolynomialFeatures(degree).fit_transform(x_ts_)
    return x_tr_, x_ts_

__all__ = ['_save_model', '_write_file', '_ROC', '_hardmax', '_classify', '_scores', '_load_dataset', '_form_dataset', '_polynomial',
           '_select_samples', '_get_dataset', '_samples_dataset', '_get_norm', '_scaling', '_load_images', '_get_scaler']
