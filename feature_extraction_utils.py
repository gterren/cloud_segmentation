import pickle, warnings, scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.interpolate import interp1d
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from scipy.ndimage.filters import median_filter
from scipy.ndimage.morphology import binary_fill_holes

from skimage.measure import label
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from mpi4py import MPI
from datetime import datetime

# Transfsorm Velocity Vectors in Cartenian to Polar Coordiantes
def _cart_to_polar(x, y):
    # Vector Modulus
    psi_ = np.nan_to_num(np.sqrt(x**2 + y**2))
    # Vector Direction
    phi_ = np.nan_to_num(np.arctan2(y, x))
    # Correct for domain -2pi to 2pi
    phi_[phi_ < 0.] += 2*np.pi
    return psi_, phi_

# Load all variable in a pickle file
def _load_file(name):
    def __load_variable(files = []):
        while True:
            try:
                files.append(pickle.load(f))
            except:
                return files
    with open(name, 'rb') as f:
        files = __load_variable()
    return files

# Group down together the entired dataset in predictions and covariates
def _save_file(X_, name):
    with open(name, 'wb') as f:
        pickle.dump(X_, f)
    print(name)

# Get node information
def _get_node_info(verbose = False):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    if verbose:
        print('>> MPI: Name: {} Rank: {} Size: {}'.format(name, rank, size) )
    return int(rank), int(size), comm

# Define a euclidian frame of coordenate s
def _frame_euclidian_coordiantes(N_x, N_y): return np.meshgrid(np.linspace(0, N_x - 1, N_x), np.linspace(0, N_y - 1, N_y))

# Find Sun coordinates on a given frame
def _sun_position_coordinates(I_, I_sun_ = 45057.):
    x_sun = 40.
    y_sun = 30.
    x_ = np.where(I_ >= I_sun_)

    if ( x_[0].size != 0 ) and ( x_[1].size != 0 ):
        x_sun = np.mean(x_[1])
        y_sun = np.mean(x_[0])

    return np.array((x_sun, y_sun))[:, np.newaxis]

# Keep static the Sun postions when it is occluded
def _sun_occlusion_coordinates(x_1_, x_2_):
    # It wasn't occuluded ...
    if (x_1_[0] != 40.) and (x_1_[1] != 30.):
        # It's occuluded ...
        if (x_2_[0] == 40.) and (x_2_[1] == 30.):
            x_2_ = x_1_.copy()
    return x_2_

# Normalize an infrared image of 16 bits to 8 bits
def _normalize_infrared_image(I_, I_min = 0., I_max = 2**13): return 255*(I_ - I_.min())/I_max

# Label Clouds by Connected Components labelling
def _cloud_pixels_labeling(I_, fill = False):
    if fill: return label(binary_fill_holes(I_), background = 0)
    else:    return label(I_, background = 0)

# Generating grid values and set them with reference to the Sun
# to denote their distance to the center of the image
def _polar_coordinates_transformation(x_sun_, X, Y):
    X_prime_ = X - x_sun_[0]
    Y_prime_ = Y - x_sun_[1]
    theta_   = np.sqrt( (X_prime_)**2 + (Y_prime_)**2 ) + 1e-25
    alpha_   = np.arccos(X_prime_/theta_)
    alpha_[:30, :] = np.pi - alpha_[:30, :] + np.pi
    alpha_ = np.fliplr(np.flipud(alpha_))
    return theta_, alpha_

# Interpolate the pixels on the circumsolar area for smoothing atmospheric effecs removal
def _sun_pixels_interpolation(I_, MA_, X_, Y_, N_y, N_x, radius):
    # Index of the circumsolar pixel
    idx_ = MA_[0] >= radius
    # Defining grid of pixels to interpolate by using the pixels selected and do interpolation
    xy       = np.concatenate((X_[idx_][:, np.newaxis], Y_[idx_][:, np.newaxis]), axis = 1)
    I_prime_ = griddata(xy, I_[idx_], (X_.flatten(), Y_.flatten()), method = 'nearest').reshape([N_y, N_x])
    # Removing remaining deformities: positives and negatives by appying a median filter
    I_prime_[~idx_] = median_filter(I_prime_, size = 5, mode = 'reflect')[~idx_]
    return I_prime_

# Applying a geometric transformation to account for the distance of the clouds on the background
def _geometric_coordinates_transformation(I_, A_sun_, x_sun_, X_, Y_, N_y, N_x, FOV = 50.):
    # Calculating each pixels elevation angle and applying transformation to x-axis
    def _x_axis_optical_depth(X_, n_x_prime_, x_, theta):
        # Transforming coordiantes system to a non-linear coordinates system
        X_prime_ = np.ones(X_.shape) * n_x_prime_
        X_prime_ = np.cumsum(X_prime_, axis = 0) - n_x_prime_
        X_prime_ = X_ + X_prime_*X_
        x_prime_ = np.ones((N_y, 1)) * n_x_prime_
        x_prime_ = np.cumsum(x_prime_, axis = 0) - n_x_prime_
        x_prime_ = x_sun_[0] + x_prime_*x_sun_[0]
        # Distance Normalization depending on the elevation angle
        X_norm_  = X_prime_ - x_prime_
        x_prime_ = x_ + n_x_prime_*x_**2
        return X_norm_, X_prime_, x_prime_
    # Calculating each pixels elevation angle and applying transformation to y-axis
    def _y_axis_optical_depth(Y_, beta, y_, theta):
        # Transforming coordiantes system to a non-linear coordinates system
        Y_prime_ = Y_/( np.sin( np.radians(beta) )**2 )
        y_prime_ = y_/( np.sin( np.radians(theta) )**2 )
        # Distance Normalization depending on the elevation angle
        Y_norm_  = Y_prime_ - y_prime_
        return Y_norm_, Y_prime_, y_prime_
    # Normalization with respect to maximum and minimum distance
    def _normalize_coordiantes(X_, Y_):
        X_ = (X_ / (X_.max() - X_.min())) * (N_x - 1)
        Y_ = (Y_ / Y_.max()) * (N_y - 1)
        Z_ = np.sqrt( X_**2 + Y_**2)
        return X_, Y_, Z_
    # Normaliza image with respect to the Suns circular shape!
    def _circular_cirmcumsolar_area_normalization(X_norm_, Y_norm_, x_, y_):
        # Calculate Numerical Gradient
        X_grad_, Y_grad_ = np.gradient(X_prime_, axis = 1), np.gradient(Y_prime_, axis = 0)
        # Prapare Grid of points as NxD matrix
        XY_ = np.concatenate((X_.flatten()[..., np.newaxis], Y_.flatten()[..., np.newaxis]), axis = 1)
        # Interpolate gradient to Sun coordiantes
        x_prime_ = griddata(XY_, X_grad_.flatten(), (x_, y_), method = 'linear')
        y_prime_ = griddata(XY_, Y_grad_.flatten(), (x_, y_), method = 'linear')
        # Apply Scale
        return Y_norm_ * (x_prime_/y_prime_)
    # Sun Coordinates and Elevation
    x_ = x_sun_[0]
    y_ = x_sun_[1]
    theta = A_sun_[0, 0]
    # Calculating increasing proportion of Field of View in an Image
    n_x_prime_ = 1. / (N_x * np.sin( np.radians(theta) ))
    # Calculating the Field of View angles for each pixel
    angle_per_pixel = FOV/N_x
    beta  = theta + ( angle_per_pixel * (y_ - Y_) )
    # Normalizing x,y, and z-axes with respect to the FOV on current frame
    X_norm_, X_prime_, x_prime_ = _x_axis_optical_depth(X_, n_x_prime_, x_, theta)
    Y_norm_, Y_prime_, y_prime_ = _y_axis_optical_depth(Y_, beta, y_, theta)
    # Circumsolar area has to be circular
    Y_norm_ = _circular_cirmcumsolar_area_normalization(X_norm_, Y_norm_, x_, y_)
    # Calculate relative distances from each pixel to the Sun
    Z_norm_  = np.sqrt( X_norm_**2 + Y_norm_**2)
    # Transformation of the camerea trjectory to pixels on the images
    A_sun_[0, :] = (A_sun_[0, :] - A_sun_[0, 0]) * angle_per_pixel
    A_sun_[1, :] = (A_sun_[1, :] - A_sun_[1, 0]) * angle_per_pixel
    # Concatenate data and return
    XYZ_ = np.concatenate((X_norm_[..., np.newaxis], Y_norm_[..., np.newaxis], Z_norm_[..., np.newaxis]), axis = 2)
    return XYZ_, A_sun_

# Finding mass center for each segmented object on a frame
# https://stackoverflow.com/questions/37519238/python-find-center-of-object-in-an-image
def _clouds_mass_center(I_seg_, N_x, N_y):
    # Initialize positions matrix
    M_ = np.zeros((N_y, N_x), dtype = bool)
    # loop for each segmented object on an image
    for l in np.unique(I_seg_):
        # Selecting only the object to me analized
        I_bin_ = np.zeros((N_y, N_x), dtype = bool)
        I_bin_[I_seg_ == l] = True
        # Calculating object mass
        mass = I_bin_ / np.sum(I_bin_)
        # Finding the center of mass
        c_x = int( np.around(np.sum(np.sum(mass, 0) * np.arange(N_x))) )
        c_y = int( np.around(np.sum(np.sum(mass, 1) * np.arange(N_y))) )
        M_[c_y, c_x] = True
    return M_

# Generate the Fpheric effect for the Sun frame position coordiantes and horizon angles
def _atmospheric_effect(I, a_, m_, C_, t_, XX, YY, N_y = 60, N_x = 80, x_sun_ = 45057.):
	# Obtaining coefficients for the athmospheric models from a polynomial model
    def __polynomial_model(X_, C_, n):
        return np.matmul(PolynomialFeatures(n).fit_transform(X_.T), C_)
	# Applying atmospheric models to estimate effects of scatter radiation on the images
    def __F(w_, I):
        # When there is sun on the image apply Sun model and Background model
        if (I > x_sun_ - 2.).any():
            f1 = w_[0] * np.exp( (X_[:, 1] - m_[1]) / w_[1])[:, np.newaxis]
            f2 = w_[2] * ( (w_[3]**2) /( (X_[:, 0] - m_[0])**2 + (X_[:, 1] - m_[1])**2 + w_[3]**2 )**1.5 )[:, np.newaxis]
            f  = f1  + f2
            f[f > x_sun_] = x_sun_

            # fig = plt.figure(figsize = (15, 10))
            # ax = plt.subplot(111, projection='3d')
            # ax.plot_surface(XX, YY, f1.reshape((60, 80)), cmap = 'inferno')
            # ax.set_xlabel('X axis', fontsize = 25)
            # ax.set_ylabel('Y axis', fontsize = 25)
            # ax.set_zlabel('Z axis', fontsize = 25)
            # plt.show()
            #
            # fig = plt.figure(figsize = (15, 10))
            # ax = plt.subplot(111, projection='3d')
            # ax.plot_surface(XX, YY, f2.reshape((60, 80)), cmap = 'inferno')
            # ax.set_xlabel('X axis', fontsize = 25)
            # ax.set_ylabel('Y axis', fontsize = 25)
            # ax.set_zlabel('Z axis', fontsize = 25)
            # plt.show()

            return f

        # When the Sun on the images is covered apply only the background model
        else: return w_[0] * np.exp( (X_[:, 1] - m_[1]) / w_[1])[:, np.newaxis]

    X_ = np.concatenate([XX.flatten()[:, np.newaxis], YY.flatten()[:, np.newaxis]], axis = 1)
    A_ = np.concatenate((a_, t_))
    # Atmospheric models parameters
    w_ = __polynomial_model(A_, C_[0], n = 5)
    return __F(w_[0], I).reshape([N_y, N_x])

# Caculate potential lines function
def _potential_lines(u, v):
    return .5*( np.cumsum(u, axis = 1) + np.cumsum(v, axis = 0) )

# Caculate stramelines function
def _streamlines(u, v):
    return .5*( np.cumsum(u, axis = 0) - np.cumsum(v, axis = 1) )

# Calculate vorticity approximating hte veloctiy gradient by numerical diffenciation
def _vorticity(u, v):
    return np.gradient(u)[1] - np.gradient(v)[0]

# Calculate divergence approximating hte veloctiy gradient by numerical diffenciation
def _divergence(u, v):
    return np.gradient(u)[1] + np.gradient(v)[0]

# Calculate the magnitude of a vector
def _magnitude(u, v):
    return np.sqrt(u**2 + v**2)

# Finding index on a field by thresholding magnitude
def _index_velocity_vectors(u, v, tau):
    return _magnitude(u, v) > tau

# Bayesian approach for selecting the most likely vector a posteriori
# def _vector_selection(X_, u_, v_, n_tr, n_ts):
#     # Bayesian Sample Selection with Full, Spheric or Diagonal Covariance Function options
#     def __bayesian_sample_selection(u_, v_, n_tr, n_ts, _cov = 'full'):
#         # Calculating the prior data convariance matrix
#         def ___prior(Y):
#             Sp = np.zeros((Y.shape[1], Y.shape[1]))
#             # Loop removing each sample from the dataset
#             # and calculating the covariance matrix for the entire set
#             for i in range(Y.shape[0]):
#                 Z = np.delete(Y, i, axis = 0) - Y[i, :]
#                 Sp += np.matmul(Z.T, Z)/Y.shape[0]
#             return Sp/Y.shape[0]
#         # Calculating the posterior probabilities
#         def ___posterior(Y, pl, Sp):
#             # Variable Initialization
#             pp_ = np.zeros((Y.shape[0]))
#             # Loop centering likelihood in each sample
#             for y, i in zip(Y, range(Y.shape[0])):
#                 # Posteriot probabilities
#                 pp_[i] = (pl + multivariate_normal(y, Sp).logpdf(Y)).sum()
#             return pp_
#         Y_  = np.concatenate((u_[..., np.newaxis], v_[..., np.newaxis]), axis = 1)
#         dim = Y_.shape[1]
#         # Defining Gaussian function parameters likelihood
#         print(Y_)
#         m_ = np.mean(Y_, axis = 0)
#         S_ = np.cov(Y_.T) + np.eye(dim)*1e-5
#         print(S_)
#         pl_ = multivariate_normal(m_, S_).logpdf(Y_)        # Defining Gaussian function parameters for the prior
#         if 'full':    Sp = ___prior(Y_)                       # Full Covariance Matrix
#         if 'spheric': Sp = np.eye(dim)                        # Spheric Covariance Matrix
#         if 'diag':    Sp = np.eye(dim) * np.var(Y_, axis = 0) # Diagonal Covariance Matrix
#         # Finding for each sample as prior mean the posterior likelihoods
#         pp_ = ___posterior(Y_, pl_, Sp)
#         pp_ += abs(pp_.min()) + 1e-10
#         # Select test index randomly from the index remaining..
#         idx_ = np.random.choice(np.argsort(pp_)[::-1], size = n_tr + n_ts, p = pp_/pp_.sum(), replace = False)
#         idx_ = idx_[np.random.permutation(idx_.shape[0])]
#         # Finding maximum likihoods for training
#         return idx_[:n_tr], idx_[-n_ts:]

    # Do Bayesian Selection on the velocity vectors per each row
    def __select_vector_per_distance(X_, u_, v_):
        # Index of vectors
        idx_ = np.arange(u_.shape[0], dtype = int)
        # Variables Initialization
        index_tr_, index_ts_ = [], []
        # Loop over distance data
        for idx_dist_, n_tr, n_ts in zip(X_[0], X_[1], X_[2]):
            # Select Velocity Vectors
            idx_tr_, idx_ts_ = __bayesian_sample_selection(u_[idx_dist_], v_[idx_dist_], n_tr, n_ts)
            # Save selected Vectors
            index_tr_.append( idx_[idx_dist_][idx_tr_] )
            index_ts_.append( idx_[idx_dist_][idx_ts_] )
        # Stack all selected velocity vectors in an array
        return np.concatenate(index_tr_, axis = 0), np.concatenate(index_ts_, axis = 0)

    # Adapt Sample number selected to the actual number of samples available
    def __adaptative_sample_number(X_, n_tr, n_ts, n_samples, per = 3):
        # No. of maximum vector per row
        n_tot = n_tr + n_ts
        n_row = n_tot//6
        # Variables Initialization
        idx_, n_tr_, n_ts_ = [], [], []
        # Check if there is enough Vectors
        if n_samples > n_tot//per:
            # loop over distancs available
            for y in np.unique(X_[:, 1]):
                idx_.append(X_[:, 1] == y)
                # How many vectors there are with that distance?
                n = idx_[-1].sum()//2
                # Get as much vectors as there are availables
                if n < n_row:
                    n_tr_.append( n )
                    n_ts_.append( n )
                else:
                    n_tr_.append( n_row )
                    n_ts_.append( n_row )
            # Flag - There is enough data
            return [idx_, n_tr_, n_ts_], True
        # Flag - There is not enough data
        else: return [idx_, n_tr_, n_ts_], False

    # Adapt Sample number selected to the actual number of samples available
    n_samples = u_.shape[0]
    D_, flag = __adaptative_sample_number(X_, n_tr, n_ts, n_samples)
    # Sufficient amount of vectors so select vectors
    if flag: idx_tr_, idx_ts_ = __select_vector_per_distance(D_, u_, v_)
    # Do not select any vector
    else: idx_tr_, idx_ts_ = [], []
    # Return Only training and test Selected Samples
    return X_[idx_tr_, :], u_[idx_tr_], v_[idx_tr_], X_[idx_ts_, :], u_[idx_ts_], v_[idx_ts_]

# Redece the dimensions of the clouds velocity field
def _cloud_velocity_field_processing(F_, M_, X_, Y_, U_, V_, x_, y_, step_size, lag, tau = 1e-2):
    # Applying an average windown over a vector field
    def __mean_velocity_field(F_, M_, tau, step_size):
        # Average only between velotiy vectors
        def __find_velocity_field_(g_, m_):
            if m_.sum() != 0: return np.mean(g_[m_])
            else: return 0.
        D, N, K = F_.shape
        # Variable Initialization
        M_ = M_ > tau
        f_ = np.zeros((D//step_size, N//step_size, K))
        k  = step_size//2
        # Loop over field compoenent
        for i in np.arange(K):
            # Loop over step window throughout y-axis
            for ii, d in zip(np.arange(k, D + k, step_size), np.arange(D//step_size)):
                # Loop over step window throughout X-axis
                for iii, n in zip(np.arange(k, N + k, step_size), np.arange(N//step_size)):
                    # Mean window pixel
                    f_[d, n, i] = __find_velocity_field_(g_ = F_[(ii - k):(ii + k), (iii - k):(iii + k), i], m_ = M_[(ii - k):(ii + k), (iii - k):(iii + k)])
        # Return Average Velocity in vector form
        return f_[..., 0].flatten(), f_[..., 1].flatten()
    # Lagged list of consecutive vectors
    def __lag_data(X_lag, Y_lag, u_lag, v_lag, xy_, uv_, lag):
        # Keep the desire number of lags on the list by removing the last and aadding at the bigging
        if len(X_lag) == lag:
            X_lag.pop(0)
            Y_lag.pop(0)
            u_lag.pop(0)
            v_lag.pop(0)
        # Keep adding until we have the desired number of lag time stamps
        X_lag.append(xy_[0])
        Y_lag.append(xy_[1])
        u_lag.append(uv_[0])
        v_lag.append(uv_[1])
        return X_lag, Y_lag, u_lag, v_lag
    # Applying mean window to reduce velocity field dimensions
    u_, v_ = __mean_velocity_field(F_, M_, tau, step_size)
    # Index of thresholding velocity vectors to remove noisy vectors
    idx_ = _magnitude(u_, v_) > tau
    # Lagging data for wind velocity field estimation
    return __lag_data(X_, Y_, U_, V_, xy_ = [x_[idx_], y_[idx_]], uv_ = [u_[idx_], v_[idx_]], lag = lag)

# Finding index of pixels expecting to intercept the Sun for each horizon (k)
def _pixels_selection(XYZ_, U_, V_, Phi_, Psi_, x_sun_, A_sun_, N_y, N_x, G_, X_, Y_, radius_1):
    # Estimate circule center for selection according to estimate arrivale time of a pixel
    def __estimate_time(XYZ_, U_, V_, idx_2):
        # Initialize variables, identify pixels on the streamline, and proximity sorting index definition
        idx_3 = idx_2 > 0
        i_    = np.argsort(idx_2[idx_3] - 1)
        # Space distance on the x and y axis on a non-linear-metric
        z_ = XYZ_[idx_3, 2][i_]
        y_ = XYZ_[idx_3, 1][i_]
        # Numerical differenciation on a non-linear frame-metric
        dz_ = np.gradient(z_)
        dy_ = np.gradient(y_)
        # Calculate the velocity components for each streamline pixel
        w_ = np.sqrt(U_[idx_3][i_]**2 + (dy_*V_[idx_3][i_])**2)
        # Integrating and solving for time for each component
        t_ = np.cumsum(dz_/(w_ + 1e-25))
        # Organizing time instants on the matrix
        idx_   = np.argsort(idx_2.flatten())
        t_hat_ = np.zeros(idx_.shape)
        t_hat_[idx_[-t_.shape[0]:]] = t_
        return t_hat_.reshape(idx_2.shape)
    # Selecting pixels by streamlines and potential lines that intercept the Sun
    def __select_intercepting_potential_line(Psi_mean):
        return Psi_ > Psi_mean
    # Connected components form the Sun streamline alon equipotential streamlines
    def __select_intercepting_streamline(Phi_, Psi_, x_sun, y_sun, Phi_mean, idx_1_):
        # Finding connected pixel
        def ___next_pixel_coordiantes(Phi_, i, j, idx_sun, idx_):
            # Defining errors matrix
            E_ = np.zeros((3, 3))
            # loop over nerbouring pixels
            for k in [-1, 0, 1]:
                for m in [-1, 0, 1]:
                    c = idx_[i + k - 1: i + k + 2, j + m - 1: j + m + 2].sum()
                    if idx_[i + k, j + m] or (k == 0 and m == 0) or c > 2:
                        E_[1 + k, 1 + m] = np.inf
                    else:
                        E_[1 + k, 1 + m] = ( Phi_mean - Phi_[i + k, j + m])**2
            # Unravel error matrix coordiantes of min value error
            k_, m_ = np.where(E_ == E_.min())
            # Updating new streamline pixel coordiantes
            i_new, j_new = i + k_[0] - 1, j + m_[0] - 1
            return i_new, j_new
        # Variables initialization
        i, j, idx_2_   = y_sun, x_sun, np.zeros((N_y, N_x), dtype = int)
        # Position Initialization
        count = 1
        idx_1_[i, j], idx_2_[i, j] = True, count
        # Loop to the edge of the frame
        while True:
            count += 1
            # Finding next pixel on the streamline
            i, j = ___next_pixel_coordiantes(Phi_, i, j, Phi_mean, idx_1)
            # Updating positon
            idx_1_[i, j], idx_2_[i, j] = True, count
            # Finding the edge
            if (i >= 59 or i <= 0) or (j >= 79 or j <= 0):
                break
        return idx_2_
    # Define patch shape and distance away from the sun for horizon (k)
    def __define_sector_coordinates(t_hat_, radius, g_, idx_2, x_sun, y_sun):
        # Estimate circle center
        def ___calcualte_center(XYZ_, inv_e_, idx):
            # Variables initialization
            x_ = XYZ_[idx, 0]
            y_ = XYZ_[idx, 1]
            o_ = np.ones(x_.shape)
            # Wighted mean ...
            x_ = np.matmul(inv_e_, x_.T)/np.matmul(o_, inv_e_.T)
            y_ = np.matmul(inv_e_, y_.T)/np.matmul(o_, inv_e_.T)
            return x_, y_
        # Finding streamline pixels closest to the horizon (k)
        idx_3  = idx_2 > 0
        inv_e_ = 1./ np.exp(np.sqrt( (t_hat_[idx_3] - g_)**2 ))
        # Robustness in center when is not possible to calculate
        if inv_e_.sum() != 0:
            inv_e_ /= inv_e_.sum()
            x_, y_  = ___calcualte_center(XYZ_, inv_e_, idx_3)
        else:
            x_, y_ = XYZ_[y_sun, x_sun, 0], XYZ_[y_sun, x_sun, 1]
        # Return Selected Pixels
        return __circle_function(XYZ_, x_, y_, radius)
    # Circle function Affecting Sun pixels indixes
    def __circle_function(XYZ_, x_, y_, radius):
        return np.sqrt( (XYZ_[..., 0] - x_)**2 + (XYZ_[..., 1] - y_)**2 ) < radius
    # Initialize List of selected sectors
    index_ = []
    # Loop over groups ..
    for g_, i_ in zip(G_, range(len(G_))):
        # No incrementes due to camera movements...
        #x_prime, y_prime = 0., 0.
        # Sun position absolute increments by elevation and azimuth
        # Make it robust in case of a day begining or ending
        if A_sun_.shape[1] - 1 < g_:
            x_prime = A_sun_[1, -1]
            y_prime = A_sun_[0, -1]
        else:
            x_prime = A_sun_[1, g_]
            y_prime = A_sun_[0, g_]
        # Sun position on integer value to use it as indexing value
        x_sun = int(np.around(x_sun_[0] + x_prime))
        y_sun = int(np.around(x_sun_[1] - y_prime))
        # Sun-interceptig streamline
        Psi_sun = Psi_[y_sun, x_sun]
        Phi_sun = Phi_[y_sun, x_sun]
        # Selecting Sun intercepting streamline index
        idx_1 = __select_intercepting_potential_line(Psi_sun)
        idx_2 = __select_intercepting_streamline(Phi_, Psi_, x_sun, y_sun, Phi_sun, idx_1)
        # Time estimation for intercepting the Sun for each pixel on the streamline
        t_hat_ = __estimate_time(XYZ_, U_, V_, idx_2)
        # Set of pixels selected for each horizon-group (k)
        idx_ = __define_sector_coordinates(t_hat_, radius_1, g_, idx_2, x_sun, y_sun)
        index_.append(idx_)
    #t_hat_[t_hat_ > 0] = 1
    return index_, t_hat_

# Functions to transform the features selected to histogram counts of groups done by distances to the Sun
def _calculate_features_statistics(I_segm_, I_, M_, D_, V_, index_, N_y, N_x, dim):
    # Variables Initialization
    Xi_  = np.empty((0, 2, dim))
    idx_ = np.zeros((N_y, N_x))
    # Loop over the date on each group
    for idx, g in zip(index_, range(len(index_))):
        # Iff were pixels selected...
        if idx.sum() > 0:
            # Adjusting normal distribution to each group data
            mu_0, var_0 = norm.fit(I_[idx])
            mu_1, var_1 = norm.fit(M_[idx])
            mu_2, var_2 = norm.fit(D_[idx])
            mu_3, var_3 = norm.fit(V_[idx])
            # Calculating standard deviation from variance
            std_0 = np.sqrt(var_0)
            std_1 = np.sqrt(var_1)
            std_2 = np.sqrt(var_2)
            std_3 = np.sqrt(var_3)
        else:
            # if not selected statistics are 0.
            mu_0, std_0 = 0., 0.
            mu_1, std_1 = 0., 0.
            mu_2, std_2 = 0., 0.
            mu_3, std_3 = 0., 0.
        # Organize on a vector Dx1 the data from each group D = N_groups x dim
        xi_0 = np.asarray((mu_0, std_0))[:, np.newaxis]
        xi_1 = np.asarray((mu_1, std_1))[:, np.newaxis]
        xi_2 = np.asarray((mu_2, std_2))[:, np.newaxis]
        xi_3 = np.asarray((mu_3, std_3))[:, np.newaxis]
        # Concatenate Statistics on a tensor for later defining the dataset
        xi_ = np.concatenate((xi_0, xi_1, xi_2, xi_3), axis = 1)[np.newaxis, ...]
        Xi_ = np.concatenate((Xi_, xi_), axis = 0)
        # Representation of sectors by interger labels
        idx_[idx] = g + 1
    xi_ = Xi_[..., np.newaxis]
    return xi_, idx_


__all__ = ['_load_file', '_save_file', '_get_node_info', '_frame_euclidian_coordiantes',
'_sun_position_coordinates', '_sun_occlusion_coordinates', '_normalize_infrared_image', '_cloud_pixels_labeling', '_polar_coordinates_transformation', '_cart_to_polar',
'_sun_pixels_interpolation', '_geometric_coordinates_transformation', '_clouds_mass_center', '_atmospheric_effect', '_potential_lines', '_streamlines', '_vorticity', '_divergence',
'_magnitude', '_index_velocity_vectors', '_cloud_velocity_field_processing', '_pixels_selection', '_calculate_features_statistics']
