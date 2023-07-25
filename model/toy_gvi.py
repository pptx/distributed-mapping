import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import cProfile
import functools
from scipy.stats import qmc, multivariate_normal

def profile(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            retval = func(*args, **kwargs)
        finally:
            filename = 'profile.prof'  # You can change this if needed
            profiler.dump_stats(filename)
            profiler.disable()
            # with open('profile.txt', 'w') as profile_file:
                # sortby = 'cumulative'
                # stats = pstats.Stats(
                    # profiler, stream=profile_file).sort_stats(sortby)
                # stats.print_stats()
        return retval
    return inner
    
def generate_data(M):
    sampler = qmc.LatinHypercube(d=2)
    samples = sampler.random(n=M)
    # y = [0 if x[1]>x[0]**1.5 else 1 for x in samples]
    # y = [1 if (x[0]-0.5)**2+(x[1]-0.75)**2>0.25**2 else 0 for x in samples]
    y = [1 if (x[0]-0)**2+(x[1]-0.5)**2>0.5**2 else 0 for x in samples]
    return samples, y

def plot_categorical_data(X, y):
    fig, ax = plt.subplots()
    indices = np.where(np.array(y)==0)
    n_indices = np.where(np.array(y)==1)
    ax.scatter(X[indices, 0], X[indices, 1], marker='^')
    ax.scatter(X[n_indices, 0], X[n_indices, 1], marker='o')
    return ax
    
def plot_predict_cat(X, y):
    fig, ax = plt.subplots()
    indices = np.where(np.array(y)<=0.5)
    n_indices = np.where(np.array(y)>0.5)
    ax.scatter(X[indices, 0], X[indices, 1], marker='^')
    ax.scatter(X[n_indices, 0], X[n_indices, 1], marker='o')
    return ax
    
# @jit(nopython=True)
def feature_RBF(x, fpoints):
    """
    Type: Specify poly or RBF
    args: power of function or feature points
    """
    lscale = 0.25
    fpoints = np.vstack((x, fpoints)) # Create bias vector
    nf = fpoints.shape[0]
    # dist = np.array([np.linalg.norm(fpoints[idf] - x) for idf in range(nf)]).reshape(-1,1)
    dist = np.linalg.norm(fpoints - np.tile(x, (nf, 1)) , axis = 1)
    return np.exp(-1*dist/(2*lscale**2))
    
# @jit(nopython=True)
def sigmoid(x):
    return 1./(1+np.exp(-x))
    
# @jit(nopython=True)
def forward_model(x, theta, theta_cov, fpoints):
    Phi_X = feature_RBF(x, fpoints)
    den = (1+Phi_X@theta_cov.dot(Phi_X.reshape(-1,1)))**(0.5)
    return sigmoid(Phi_X.dot(theta)/den)
    
# @jit(nopython=True)
def gradients(x, y, fpoints, theta):
    Phi_X = feature_RBF(x, fpoints)
    val = Phi_X.dot(theta)
    sig = sigmoid(val)
    val2 = sig*np.exp(-val)
    nu = y - sig
    der = (1./(1-sig))*val2*nu*Phi_X
    dder = (nu*sig)*np.exp(-val) - (sig**2)*np.exp(-val) - nu + ((sig**2)*nu/(2*(1-sig)))
    dder = (sig/(1-sig))*dder*np.exp(-val)*(Phi_X.reshape(-1,1).dot(Phi_X.reshape(1,-1)))
    return der, dder
    
def gradients_s(Phi_X, y, fpoints, theta):
    sig = sigmoid(Phi_X.dot(theta))
    der = (y-sig)*Phi_X
    # dder = -sig*(1-sig)*(np.outer(Phi_X, Phi_X))
    dder_diag = -(sig*(1-sig))*(np.diag(Phi_X**2))
    return der, dder_diag
    
# Implement autograd
# def log_likelihood(x, y, fpoints, theta)
    
# Use snakeviz profile.prof later
# @profile
def gvi(T, M, B, X, y, X_p, y_p, fpoints):
    """
    Updates follow Eqn. 30-31 in https://hal.inria.fr/hal-03086627/document
    
    Implement extended Kalman filter in this function.
    message_mu is in agent-neighbor format
    (i,j) represents message from agent i to neighbor j
    
    :param B: Batch size of collected data
    """
    d_dim = fpoints.shape[0]+1
    mu_update = np.zeros((T+1, 1, d_dim))
    cov_update_t = np.zeros((T+1, 1, d_dim, d_dim))
    cov_update_t[0, 0,:,:] = 0.3*np.eye(d_dim)
    # Running the Bayesian propagation algorithm for T steps
    error = []
    for t in range(T):
        samples = multivariate_normal.rvs(mu_update[t, 0,:], cov_update_t[t, 0,:,:], size=M)
        n_sigma_ = np.zeros((d_dim, d_dim))
        delta_mu = np.zeros((d_dim))
        der = np.random.rand(d_dim)
        err = 0
        for db in range(B):
            db_id = [random.randint(0, T-1) if db != 0 else t][0]# Since T = #data points.
            Phi_X = feature_RBF(X[db_id,:], fpoints)
            for s in range(M):
                prev_der = der
                der, d_der = gradients_s(Phi_X, y[db_id], fpoints, samples[s])
                n_sigma_ += d_der
                delta_mu += der
                err = err + np.linalg.norm(prev_der-der)
        
        # Generate likelihood update
        delta_omega_ = (1./(M*B))*n_sigma_
        n_omega_ = np.linalg.pinv(cov_update_t[t,0,:,:]) - delta_omega_
        n_sigma_ = np.linalg.pinv(n_omega_)
        
        delta_mu = (1./(M*B))*n_sigma_@delta_mu
        n_mu = mu_update[t, 0,:]+delta_mu
        # print (n_mu, n_sigma_)
        mu_update[t+1,0,:] = n_mu.flatten()
        cov_update_t[t+1,0,:,:] = n_sigma_
        
        # Compute verification error
        if (t%100==0):
            print ('Average relative gradient error is', err/(M*B), 'at iteration', t)
            err = 0
            for i, x in enumerate(X_p):
                y_pred = forward_model(x, mu_update[t+1,0,:], cov_update_t[t+1,0,:,:], fpoints)
                err += -(y_p[i]*np.log(y_pred) + (1-y_p[i])*np.log(1-y_pred))
            error.append(err)
    return mu_update, cov_update_t, error

def create_dataset(N, Np, nf):
    X, y = generate_data(T)
    # Prediction set
    y_pred = []
    X_p, y_p = generate_data(300)
    
    sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    fpoints = sampler.random(n=nf)
    return X, y, X_p, y_p, fpoints

"""
Analytical vs Autograd gradients: Try batch simulation
Increase number of feature points: Does work for 100 feature points in 
"""
if __name__ == '__main__':
    T = 1000
    nf = 150
    X, y, X_p, y_p, fpoints = create_dataset(T, 300, nf)
    # plt.scatter(fpoints[:,0], fpoints[:,1])
    # plt.show(); quit()
    # plot_categorical_data(X, y)
    M = 150
    B = 1
    mu_update, cov_update_t, error = gvi(T, M, B, X, y, X_p, y_p, fpoints)
    fig, ax = plt.subplots()
    mu_t = np.squeeze(mu_update)
    print (mu_t.shape)
    for id_f in range(1, nf):
        ax.plot(mu_t[:, id_f], label='weights')
    ax.plot(mu_t[:,0], '--', label='bias')
    ax.legend()
    ax.grid()
    fig, ax = plt.subplots()
    ax.plot(error)
    
    y_pred = []
    for i, x in enumerate(X_p):
        y_pred.append(forward_model(x, mu_update[-1,0,:], cov_update_t[-1,0,:,:], fpoints))
    plot_categorical_data(X_p, y_p)
    plot_predict_cat(X_p, y_pred)
    plt.show()