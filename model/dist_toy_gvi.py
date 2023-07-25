import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import cProfile
import functools
import sys
from scipy.stats import qmc, multivariate_normal

from network_functions import generate_connected_graph

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
    
# Generate similar data in each class
def generate_homogenous_data(n, M, type_=2):
    d=2
    Xn = np.zeros((n, M, d))
    yn = np.zeros((n, M))
    for i in range(n):
        sampler = qmc.LatinHypercube(d)
        samples = sampler.random(n=M)
        match type_:
            case 1:
                y = [0 if x[1]>x[0]**1.5 else 1 for x in samples]
            case 2:
                y = [1 if (x[0]-0.5)**2+(x[1]-0.75)**2>0.25**2 else 0 for x in samples]
            case 3: 
                y = [1 if (x[0]-0)**2+(x[1]-0.5)**2>0.5**2 else 0 for x in samples]
        Xn[i,:,:] = samples 
        yn[i,:] = y
    return Xn, yn
    
def generate_heterogenous_data(n, M, type_=2):
    d=2
    Xn = np.zeros((n, M, d))
    yn = np.zeros((n, M))
    for i in range(n):
        sampler = qmc.LatinHypercube(d)
        samples = sampler.random(n=M)
        samples[0,:] = i/n + (1./n)*samples[0,:]
        match type_:
            case 1:
                y = [0 if x[1]>x[0]**1.5 else 1 for x in samples]
            case 2:
                y = [1 if (x[0]-0.5)**2+(x[1]-0.75)**2>0.25**2 else 0 for x in samples]
            case 3: 
                y = [1 if (x[0]-0)**2+(x[1]-0.5)**2>0.5**2 else 0 for x in samples]
        Xn[i,:,:] = samples 
        yn[i,:] = y
    return Xn, yn

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
    
def gradients_s(Phi_X, y, fpoints, theta):
    val = Phi_X.dot(theta)
    sig = sigmoid(val)
    der = (y-sig)*Phi_X
    dder = -sig*(1-sig)*(np.outer(Phi_X, Phi_X))
    return der, dder
    
# Use snakeviz profile.prof later
# @profile
def dgvi(T, M, n, A, B, X, y, X_p, y_p, fpoints):
    """
    Updates follow Eqn. 30-31 in https://hal.inria.fr/hal-03086627/document
    
    Implement extended Kalman filter in this function.
    message_mu is in agent-neighbor format
    (i,j) represents message from agent i to neighbor j
    
    :param M: Number of samples used to compute derivatives
    :param n, A: Number of agents and the communication network 
    :param X, y: Input data and class value for multiple agents
    :param B: Batch size of collected data
    """
    d_dim = fpoints.shape[0]+1
    mu_update = np.zeros((T+1, n, d_dim))
    omega_update = np.zeros((T+1, n, d_dim, d_dim))
    for i in range(n):
        omega_update[0, i,:,:] = 0.3*np.eye(d_dim)
    # Running the Bayesian propagation algorithm for T steps
    error = []
    for t in range(T):
        for agent in range(n):
            # Compute geometric mixing update with neighbors
            omega_received_sum = np.einsum('i,ijk->jk', A[:,agent], omega_update[t,:,:,:])
            weighted_omega = np.einsum('i,ijk->ijk', A[:,agent], omega_update[t,:,:,:])
            weighted_mu = np.einsum('ijk,ik->j', weighted_omega, mu_update[t,:,:])
            weighted_mu = np.linalg.solve(omega_received_sum, weighted_mu)
            weighted_sigma = np.linalg.pinv(omega_received_sum)
            
            samples = multivariate_normal.rvs(weighted_mu, weighted_sigma, size=M)
            n_sigma_ = np.zeros((d_dim, d_dim))
            delta_mu = np.zeros((d_dim))
            der = np.random.rand(d_dim)
            err = 0
            for db in range(B):
                db_id = [random.randint(0, T-1) if db != 0 else t][0]# Since T = #data points.
                Phi_X = feature_RBF(X[i, db_id,:], fpoints)
                for s in range(M):
                    prev_der = der
                    der, d_der = gradients_s(Phi_X, y[i, db_id], fpoints, samples[s])
                    n_sigma_ += d_der
                    delta_mu += der
                    err = err + np.linalg.norm(prev_der-der)
            
            # Generate likelihood update
            delta_omega_ = (1./(M*B))*n_sigma_
            n_omega_ = omega_received_sum - delta_omega_
            n_sigma_ = np.linalg.pinv(n_omega_)
            
            delta_mu = (1./(M*B))*n_sigma_@delta_mu
            n_mu = weighted_mu + delta_mu
            # print (n_mu, n_sigma_)
            mu_update[t+1,agent,:] = n_mu.flatten()
            omega_update[t+1,agent,:,:] = n_omega_
        
        # Compute verification error
        if (t%100==0):
            sys.stdout.flush()
            print ('Average relative gradient error is', err/(M*B), 'at iteration', t)
            err = 0
            for idx, x in enumerate(X_p):
                y_pred = forward_model(x, mu_update[t+1,0,:], np.linalg.pinv(omega_update[t+1,0,:,:]), fpoints)
                err += -(y_p[idx]*np.log(y_pred) + (1-y_p[idx])*np.log(1-y_pred))
            error.append(err)
    return mu_update, omega_update, error



"""
Analytical vs Autograd gradients: Try batch simulation
Increase number of feature points: Does work for 100 feature points in 
"""
if __name__ == '__main__':
    T = 1000
    n = 4
    A = generate_connected_graph(n)
    X, y = generate_heterogenous_data(n, T, type_=3)
    # Prediction set
    y_pred = []
    X_p, y_p = generate_homogenous_data(1, 300, 3)
    X_p, y_p = X_p[0,:,:], y_p[0,:]
    
    sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    nf = 150
    fpoints = sampler.random(n=nf)
    # plt.scatter(fpoints[:,0], fpoints[:,1])
    # plt.show(); quit()
    theta = np.random.rand(nf+1)
    # print (gradients(X[0,:], y[0], fpoints, theta))
    # plot_categorical_data(X, y)
    M = 150
    B = 1
    mu_update, omega_update, error = dgvi(T, M, n, A, B, X, y, X_p, y_p, fpoints)
    fig, ax = plt.subplots()
    mu_t = np.squeeze(mu_update[:,0,:])
    for id_f in range(1, nf):
        ax.plot(mu_t[:, id_f], label='weights')
    ax.plot(mu_t[:,0], '--', label='bias')
    ax.legend()
    ax.grid()
    fig, ax = plt.subplots()
    ax.plot(error)
    
    
    for i, x in enumerate(X_p):
        y_pred.append(forward_model(x, mu_update[-1,0,:], np.linalg.pinv(omega_update[-1,0,:,:]), fpoints))
    plot_categorical_data(X_p, y_p)
    plot_predict_cat(X_p, y_pred)
    plt.show()