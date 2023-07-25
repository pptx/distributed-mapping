import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
import pandas as pd
import sys
from scipy.stats import qmc, multivariate_normal, norm
import time
import pickle
import scipy.sparse.linalg as sla

def plot_categorical_data(X, y):
    fig, ax = plt.subplots()
    indices = np.where(np.array(y)==0)
    n_indices = np.where(np.array(y)==1)
    ax.scatter(X[indices, 0], X[indices, 1], marker='^',s=1)
    ax.scatter(X[n_indices, 0], X[n_indices, 1], marker='o',s=1)
    return ax
    
def plot_predict_cat(X, y):
    fig, ax = plt.subplots()
    indices = np.where(np.array(y)<=0.5)
    n_indices = np.where(np.array(y)>0.5)
    ax.scatter(X[indices, 0], X[indices, 1], marker='^', s=1)
    ax.scatter(X[n_indices, 0], X[n_indices, 1], marker='o', s=1)
    return ax
    
def feature_RBF(x, fpoints, lscale):
    """
    Type: Specify poly or RBF
    args: power of function or feature points
    """
    fpoints = np.vstack((x, fpoints)) # Create bias vector, maybe append lscale here
    nf = fpoints.shape[0]
    # dist = np.linalg.norm(fpoints - np.tile(x, (nf, 1)) , axis = 1)
    dist = np.linalg.norm(fpoints - np.tile(x, (nf, 1)), ord=1, axis = 1) # Order 1 norm
    # dist = np.sum( (fpoints - np.tile(x, (nf, 1)))**2, axis=1)
    return np.exp(-1*dist/(2*lscale**2))
    
def sigmoid(x):
    return 1./(1+np.exp(-x))
    
# @jit(nopython=True)
def forward_model(x, theta, phi_cov_phi, fpoints, lscale):
    Phi_X = feature_RBF(x, fpoints, lscale)
    den = (1+phi_cov_phi)**(0.5)
    return sigmoid(Phi_X.dot(theta)/den)
    
def gradients_s(Phi_X, y, fpoints, theta):
    sig = sigmoid(Phi_X.dot(theta))
    der = (y-sig)*Phi_X
    dder_diag = -(sig*(1-sig))*(Phi_X**2)
    return der, dder_diag
    
import networkx as nx

def plot_network(X_test, A):

    assert A.shape[0] == 4
    n = 4
    # Create a new graph
    G = nx.Graph()

    # Add four nodes to the graph
    nodes = ['1', '2', '3', '4']
    G.add_nodes_from(nodes)

    # Add edges to connect the nodes
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                G.add_edge(nodes[i], nodes[j])

    # Define node colors and sizes
    node_colors = ['#9ACD32', '#9ACD32', '#9ACD32', '#9ACD32']
    node_sizes = [400, 400, 400, 400]

    # Draw the scatter plot in the background
    fig, ax = plt.subplots()
    ax.scatter(X_test[:,0], X_test[:,1], alpha=0.2, c='#aaaaaa')
    
    # Get the bounds of the scatter plot
    x_min, x_max = np.percentile(X_test[:,0], 2), np.max(X_test[:,0])
    y_min, y_max = np.min(X_test[:,1]), np.max(X_test[:,1])
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Draw the graph
    pos = nx.circular_layout(G)
    # pos = {'1': ((pos['1'][0]-x_min)/x_range, (pos['1'][1]-y_min)/y_range),
       # '2': ((pos['2'][0]-x_min)/x_range, (pos['2'][1]-y_min)/y_range),
       # '3': ((pos['3'][0]-x_min)/x_range, (pos['3'][1]-y_min)/y_range),
       # '4': ((pos['4'][0]-x_min)/x_range, (pos['4'][1]-y_min)/y_range)}
    theta = np.linspace(0, 2*np.pi, len(G.nodes), endpoint=False)
    pos = {'1': ((np.cos(theta[0])+1)/2*(x_range) + x_min, (np.sin(theta[0])+1)/2*(y_range) + y_min),
       '2': ((np.cos(theta[1])+1)/2*(x_range) + x_min, (np.sin(theta[1])+1)/2*(y_range) + y_min),
       '3': ((np.cos(theta[2])+1)/2*(x_range) + x_min, (np.sin(theta[2])+1)/2*(y_range) + y_min),
       '4': ((np.cos(theta[3])+1)/2*(x_range) + x_min, (np.sin(theta[3])+1)/2*(y_range) + y_min)}

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos, font_size=18, font_family='sans-serif')

    # Remove the axis ticks and labels
    b = 2
    ax.set_xlim([x_min-3*b, x_max+b])
    ax.set_ylim([y_min-b, y_max+b])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    return 0
    
def gvi(T, M, B, X, y, X_p, y_p, fpoints, lscale):
    """
    Updates follow Eqn. 30-31 in https://hal.inria.fr/hal-03086627/document
    
    Implement extended Kalman filter in this function.
    message_mu is in agent-neighbor format
    (i,j) represents message from agent i to neighbor j
    
    :param B: Batch size of collected data
    Learning from experiments: 
    Randomized data is better, maybe sample from range (0,t) to simulate reality.
    Lengthscales of 0.3 and 3 were bad for predictions.
    """
    # T = len(X)
    d_dim = fpoints.shape[0]+1
    mu_update = np.zeros((T+1, 1, d_dim))
    cov_update_t = np.zeros((T+1, 1, d_dim))
    cov_update_t[0, 0,:] = 0.3*np.ones(d_dim)
    # Running the Bayesian propagation algorithm for T steps
    error = []
    for t in range(T):
        if t%100 == 0:
            print (t) ;sys.stdout.flush()
        samples = multivariate_normal.rvs(mu_update[t, 0,:], np.diag(cov_update_t[t, 0, :]), size=M)
        n_sigma_ = np.zeros((d_dim))
        delta_mu = np.zeros((d_dim))
        err = 0
        
        ##################### Only for B = 1
        for db in range(B):
            # db_id = [random.randint(0, len(X)-1) if db != 0 else t][0]# Since T = #data points.
            db_id = random.randint(0, min(t+1, len(X))-1) 
            Phi_X = feature_RBF(X[db_id,:], fpoints, lscale)
            
            sig_sum, ssig_sum = 0, 0
            for s in range(M):
                # Compute gradients
                # der, d_der = gradients_s(Phi_X, y[db_id], fpoints, samples[s])
                sig = sigmoid(Phi_X.dot(samples[s]))
                sig_sum += sig
                ssig_sum += sig*(1-sig)
                
            # Using a diagonal estimate of the double derivative
            der = (y[db_id] - (1./M)*sig_sum)*Phi_X
            dder_diag = -(1./M)*ssig_sum*(Phi_X**2)
        #####################
        
        # Generate likelihood update
        delta_omega_ = (1./(B))*dder_diag
        n_omega_ = cov_update_t[t,0,:]**(-1) - delta_omega_
        n_sigma_ = n_omega_**(-1)
        
        delta_mu = (1./(B))*np.diag(n_sigma_).dot(der)
        n_mu = mu_update[t, 0,:]+delta_mu
        # print (n_mu, n_sigma_)
        mu_update[t+1,0,:] = n_mu.flatten()
        cov_update_t[t+1,0,:] = n_sigma_
        
        # Compute verification error
        # if (t%100==0):
            # print ('Average relative gradient error is', err/(M*B), 'at iteration', t)
            # err = 0
            # for i, x in enumerate(X_p):
                # y_pred = forward_model(x, mu_update[t+1,0,:], phi_cov_phi, fpoints, lscale)
                # err += -(y_p[i]*np.log(y_pred) + (1-y_p[i])*np.log(1-y_pred))
            # error.append(err)
    return mu_update, cov_update_t, error

def dist_gvi_eprobit(n, A, T, B, lik_factor, X, y, X_p, y_p, fpoints, lscale):
    """
    n: Number of agents
    T: # of iterations
    X: Input data of dimensions (n, Nx, d)
    y: Input data of dimensions (n, Nx, 1)
    fpoints: Feature points
    """
    # T = len(X)
    nv = len(X_p)
    xi = 0.61
    d_dim = fpoints.shape[0]+1
    mu_update = np.zeros((T+1, n, d_dim))
    
    n_sigma_ = np.zeros((n, d_dim, d_dim))
    omega_ = np.zeros((n, d_dim, d_dim))
    n_omega_ = np.zeros((n, d_dim, d_dim))
    for i in range(n):
        omega_[i,:,:] = 1e-3*np.eye(d_dim)
    # Running the Bayesian propagation algorithm for T steps
    error = [[] for i in range(n)]
    l_error = [[] for i in range(n)]
    train_idx = np.random.randint(0, X.shape[1]-1, T)
    for t in range(T):
        if t%100 == 0:
            print (t) ;sys.stdout.flush()
        
        for i in range(n):
            mu = np.zeros(d_dim)
            omega_i = np.zeros((d_dim, d_dim))
            
            for j in range(n):
                omega_i = omega_i + A[i, j]*omega_[j,:,:]
                mu = mu + A[i, j]*(omega_[j,:,:]@mu_update[t,j,:])
                
            db_id = train_idx[t]
            Phi_X = feature_RBF(X[i,db_id,:], fpoints, lscale)
            
            # cov_i = np.linalg.pinv(omega_i); mu = cov_i@mu
            # https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html#module-scipy.sparse.linalg
            omega_i = scipy.sparse.csc_matrix(omega_i).tocsc()
            solved = sla.spsolve(omega_i, np.vstack([mu, Phi_X]).T )
            mu = solved[:,0]
            
            # cov_phi = cov_i@Phi_X
            cov_phi = solved[:,1]
            phi_cov_phi = Phi_X@cov_phi
            phi_mu = Phi_X@mu
        
            beta = 1 + (xi**2)*phi_cov_phi
            op = np.outer(Phi_X, Phi_X)
            xi2_beta = xi**2/beta
            gamma = (xi2_beta/(2*np.pi))**(1./2)
            gamma = gamma*np.exp(- (xi2_beta/2.)*(phi_mu*phi_mu))
            dder = gamma*op
        
            der = xi*(phi_mu/(beta**(0.5)))
            der_coeff = (y[i, db_id] - norm.cdf(der))
            der = der_coeff*Phi_X
            
            # Generate likelihood update
            n_omega_[i,:,:] = n_omega_[i,:,:] + lik_factor*dder
            # n_sigma_[i,:,:] =  cov_i  - gamma*(1./(1.+gamma*phi_cov_phi))*np.outer(cov_phi, cov_phi)
        
            term = der_coeff*cov_phi - gamma*(1./(1.+gamma*phi_cov_phi))*(cov_phi@der)*cov_phi
            n_mu = mu + lik_factor*term
            # print (n_mu, n_sigma_)
            mu_update[t+1,i,:] = n_mu.flatten()
        omega_ = n_omega_
        
        # Compute verification error
        if (t%500==0):
            err, lerr = 0, 0
            for i, x in enumerate(X_p):
                y_pred = forward_model(x, n_mu.flatten(), phi_cov_phi, fpoints, lscale)
                err += -(y_p[i]*np.log(y_pred) + (1-y_p[i])*np.log(1-y_pred))
                lerr += np.abs(y_p[i] - y_pred)
            error.append(err)
            l_error.append(lerr)
            print('Average error is', err/nv, 'Percentage error is ', lerr/nv)
    return mu_update, n_sigma_, [error, l_error]
    
def dist_gvi_diag_eprobit(n, A, T, B, lik_factor, X, y, X_p, y_p, fpoints, lscale):
    """
    n: Number of agents
    T: # of iterations
    X: Input data of dimensions (n, Nx, d)
    y: Input data of dimensions (n, Nx, 1)
    fpoints: Feature points
    """
    # T = len(X)
    nv = len(X_p)
    xi = 0.61
    d_dim = fpoints.shape[0]+1
    mu_update = np.zeros((T+1, n, d_dim))
    
    alfa = 5.
    n_sigma_ = alfa*np.ones((n, d_dim))
    omega_ = (1./alfa)*np.ones((n, d_dim))
    n_omega_ = (1./alfa)*np.ones((n, d_dim))
    
    # Running the Bayesian propagation algorithm for T steps
    error = []#[[] for i in range(n)]
    l_error = []#[[] for i in range(n)]
    train_idx = np.reshape(np.array([np.random.randint(0, X[i].shape[0]-1, T) for i in range(n)]), (n, T))

    for t in range(T):
        if t%100 == 0:
            print (t) ;sys.stdout.flush()
        
        for i in range(n):
            
            # Compute geometric average as (mu, cov_i)
            mu = np.zeros(d_dim)
            omega_i = np.zeros(d_dim)
            
            for j in range(n):
                omega_i = omega_i + A[i, j]*omega_[j,:]
                mu = mu + A[i, j]*(omega_[j,:]*mu_update[t,j,:])
            cov_i = omega_i**(-1)
            mu = cov_i*mu
            
            # Compute derivative and double derivative terms
            db_id = train_idx[i, t]
            # db_id = [random.randint(0, len(X)-1) if db != 0 else t][0]# Since T = #data points.
            # db_id = random.randint(0, min(t+1, len(X))-1) 
            Phi_X = feature_RBF(X[i][db_id,:], fpoints, lscale)
            
            cov_phi = cov_i*Phi_X
            phi_cov_phi = Phi_X@cov_phi
            phi_mu = Phi_X@mu
        
            beta = 1 + (xi**2)*phi_cov_phi
            op = Phi_X*Phi_X
            xi2_beta = xi**2/beta
            gamma = (xi2_beta/(2*np.pi))**(1./2)
            gamma = gamma*np.exp(- (xi2_beta/2.)*(phi_mu*phi_mu))
            dder = gamma*op
        
            der = xi*(phi_mu/(beta**(0.5)))
            der = (y[i][db_id] - norm.cdf(der))*Phi_X
                
            # Generate likelihood update
            n_omega_[i,:] = n_omega_[i,:] + lik_factor*dder
            n_sigma_[i,:] = n_omega_[i,:]**(-1)
        
            n_mu = mu + lik_factor*n_sigma_[i,:]*der
            # print (n_mu, n_sigma_)
            mu_update[t+1,i,:] = n_mu.flatten()
        omega_ = n_omega_
        
        # Compute verification error
        if (t%500==0):
            err, lerr = 0, 0
            for i, x in enumerate(X_p):
                y_pred = forward_model(x, n_mu.flatten(), phi_cov_phi, fpoints, lscale)
                err += -(y_p[i]*np.log(y_pred) + (1-y_p[i])*np.log(1-y_pred))
                lerr += np.abs(y_p[i] - y_pred)
            error.append(err)
            l_error.append(lerr)
            print('Average error is', err/nv, 'Percentage error is ', lerr/nv)
            
        final_sigma = np.zeros((n, d_dim, d_dim))
        for i in range(n):
            final_sigma[i,:,:] = np.diag(n_sigma_[i, :])
    return mu_update, final_sigma, [error, l_error]
    
def gvi_eprobit(T, B, X, y, X_p, y_p, fpoints, lscale):
    """
    Updates follow Eqn. 30-31 in https://hal.inria.fr/hal-03086627/document
    
    Implement extended Kalman filter in this function.
    message_mu is in agent-neighbor format
    (i,j) represents message from agent i to neighbor j
    
    :param B: Batch size of collected data
    Learning from experiments: 
    Randomized data is better, maybe sample from range (0,t) to simulate reality.
    Lengthscales of 0.3 and 3 were bad for predictions.
    """
    # T = len(X)
    xi = 0.61
    d_dim = fpoints.shape[0]+1
    mu_update = np.zeros((T+1, 1, d_dim))
    # cov_update_t = np.zeros((T+1, 1, d_dim, d_dim))
    cov = 5*np.eye(d_dim)
    n_omega_ = np.linalg.pinv(cov)
    # Running the Bayesian propagation algorithm for T steps
    error = []
    l_error = []
    train_idx = np.random.randint(0, len(X)-1, T)
    for t in range(T):
        if t%100 == 0:
            print (t) ;sys.stdout.flush()
        
        mu = mu_update[t, 0,:]
        ##################### Only for B = 1
        for db in range(B):
            db_id = train_idx[t]
            # db_id = [random.randint(0, len(X)-1) if db != 0 else t][0]# Since T = #data points.
            # db_id = random.randint(0, min(t+1, len(X))-1) 
            Phi_X = feature_RBF(X[db_id,:], fpoints, lscale)
            
            cov_phi = cov@Phi_X
            phi_cov_phi = Phi_X@cov_phi
            phi_mu = Phi_X@mu
            
            beta = 1 + (xi**2)*phi_cov_phi
            op = np.outer(Phi_X, Phi_X)
            xi2_beta = xi**2/beta
            gamma = (xi2_beta/(2*np.pi))**(1./2)
            gamma = gamma*np.exp(- (xi2_beta/2.)*(phi_mu*phi_mu))
            dder = gamma*op
            
            der = xi*(Phi_X@mu/(beta**(0.5)))
            der = (y[db_id] - norm.cdf(der))*Phi_X
        
        # Generate likelihood update
        n_omega_ = n_omega_ + dder
        n_sigma_ = cov - gamma*(1./(1.+gamma*phi_cov_phi))*np.outer(cov_phi, cov_phi)
        
        n_mu = mu + n_sigma_@der
        # print (n_mu, n_sigma_)
        mu_update[t+1,0,:] = n_mu.flatten()
        cov = n_sigma_
        
        # Compute verification error
        if (t%500==0):
            err, lerr = 0, 0
            for i, x in enumerate(X_p):
                y_pred = forward_model(x, mu_update[t+1,0,:], phi_cov_phi, fpoints, lscale)
                err += -(y_p[i]*np.log(y_pred) + (1-y_p[i])*np.log(1-y_pred))
                lerr += np.abs(y_p[i] - y_pred)
            error.append(err)
            l_error.append(lerr)
            print('Error values', err, lerr)
    return mu_update, cov, [error, l_error]
    
def gvi_diag_eprobit(T, B, X, y, X_p, y_p, fpoints, lscale):
    """
    Updates follow Eqn. 30-31 in https://hal.inria.fr/hal-03086627/document
    
    Implement extended Kalman filter in this function.
    message_mu is in agent-neighbor format
    (i,j) represents message from agent i to neighbor j
    
    :param B: Batch size of collected data
    Learning from experiments: 
    Randomized data is better, maybe sample from range (0,t) to simulate reality.
    Lengthscales of 0.3 and 3 were bad for predictions.
    """
    # T = len(X)
    xi = 0.61
    d_dim = fpoints.shape[0]+1
    mu_update = np.zeros((T+1, 1, d_dim))
    
    alfa = 5
    cov = alfa*np.ones(d_dim)
    n_omega_ = (1./alfa)*np.ones(d_dim)
    # Running the Bayesian propagation algorithm for T steps
    error = []
    l_error = []
    train_idx = np.random.randint(0, len(X)-1, T)
    for t in range(T):
        if t%100 == 0:
            print (t) ;sys.stdout.flush()
        
        mu = mu_update[t, 0,:]
        ##################### Only for B = 1
        for db in range(B):
            db_id = train_idx[t]
            # db_id = [random.randint(0, len(X)-1) if db != 0 else t][0]# Since T = #data points.
            # db_id = random.randint(0, min(t+1, len(X))-1) 
            Phi_X = feature_RBF(X[db_id,:], fpoints, lscale)
            
            cov_phi = cov*Phi_X
            phi_cov_phi = Phi_X@cov_phi
            phi_mu = Phi_X@mu
            
            beta = 1 + (xi**2)*phi_cov_phi
            op = Phi_X**2
            xi2_beta = xi**2/beta
            gamma = (xi2_beta/(2*np.pi))**(1./2)
            gamma = gamma*np.exp(- (xi2_beta/2.)*(phi_mu*phi_mu))
            dder = gamma*op
            
            der = xi*(Phi_X@mu/(beta**(0.5)))
            der = (y[db_id] - norm.cdf(der))*Phi_X
        
        # Generate likelihood update
        n_omega_ = n_omega_ + dder
        n_sigma_ = n_omega_**(-1)
        
        n_mu = mu + n_sigma_*der
        # print (n_mu, n_sigma_)
        mu_update[t+1,0,:] = n_mu.flatten()
        cov = n_sigma_
        
        # Compute verification error
        if (t%500==0):
            err, lerr = 0, 0
            for i, x in enumerate(X_p):
                Phi_X = feature_RBF(x, fpoints, lscale)
                phi_cov_phi = Phi_X@(cov*Phi_X)
                y_pred = forward_model(x, mu_update[t+1,0,:], phi_cov_phi, fpoints, lscale)
                err += -(y_p[i]*np.log(y_pred) + (1-y_p[i])*np.log(1-y_pred))
                lerr += np.abs(y_p[i] - y_pred)
            error.append(err)
            l_error.append(lerr)
            print('Error values', err, lerr)
    return mu_update, np.diag(cov), [error, l_error]
    
    
def load_sbkm_parameters():
    parameters = {'intel': \
                      ('./../data/intel/intel.csv',
                       (0.2, 0.2), # grid resolution for occupied samples and free samples, respectively
                       (-20, 20, -25, 10),  # area to be mapped [x1_min, x1_max, x2_min, x2_max]
                       1, # skip
                       6.71,  # gamma: kernel parameter
                       25, # k_nearest: picking K nearest relevance vectors
                       20 # max_iter: maximum number of iterations
                       ),
                  }
    return parameters['intel']

def sbkm_data(g, complete=0, n=1):
    n_test = 1000
    if n == 1:
        # Homogenous partial
        if complete==0:
            # Random 10% for training
            X_train = np.float_(g[::9, 1:3])
            Y_train = np.float_(g[::9, 3][:, np.newaxis]).ravel()  # * 2 - 1
            
            # 10% for testing
            X_test = np.float_(g[::10, 1:3])
            Y_test = np.float_(g[::10, 3][:, np.newaxis]).ravel() 
                
        # Homogenous complete
        elif complete==1:
            # 90% for training
            X_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 1:3])
            Y_train = np.float_(g[np.mod(np.arange(len(g)), 10) != 0, 3][:, np.newaxis]).ravel()  # * 2 - 1
            
            # 10% for testing
            X_test = np.float_(g[::10, 1:3])
            Y_test = np.float_(g[::10, 3][:, np.newaxis]).ravel() 
            
        else:
            raise ValueError('Use 0 or 1.')
    else:
        # Distributed homogenous
        if complete == 0:
            assert n <= 50
            # Lower bound of the number len(g)
            size_ = int(len(g)/n)
            data_len = n*size_
            # Consists of numbers from 0 to n-1
            remainder_list = np.mod(np.arange(len(g)), n)
            remainder_list[data_len:] = n
            
            X_train = np.zeros((n, size_, 2))
            Y_train = np.zeros((n, size_))
            for idx in range(n):
                X_train[idx, :, :] = np.float_(g[remainder_list == idx, 1:3])
                Y_train[idx, :] = np.float_(g[remainder_list == idx, 3][:, np.newaxis]).ravel()  # * 2 - 1
                
            # 10% for testing
            X_test = np.float_(g[::n, 1:3])
            Y_test = np.float_(g[::n, 3][:, np.newaxis]).ravel() 
        
        else:
            assert n <= 10
            # Lower bound of the number len(g)
            size_ = int(len(g)/n)
            data_len = n*size_
            
            X_train = np.zeros((n, size_, 2))
            Y_train = np.zeros((n, size_))
            for idx in range(n):
                X_train[idx, :, :] = np.float_(g[idx*size_:(idx+1)*size_, 1:3])
                Y_train[idx, :] = np.float_(g[idx*size_:(idx+1)*size_, 3][:, np.newaxis]).ravel()  # * 2 - 1
                
            # 10% for testing
            X_test = np.float_(g[::n, 1:3])
            Y_test = np.float_(g[::n, 3][:, np.newaxis]).ravel() 
        
    n_test = min(n_test, X_test.shape[0])
    test_idx = np.random.randint(0, X_test.shape[0], n_test)
    X_ver = X_test[test_idx, :]
    Y_ver = Y_test[test_idx]
        
    print(len(g), len(Y_train), len(Y_test))
    return X_train, Y_train, X_test, Y_test, X_ver, Y_ver
    
    
def sbkm_fpoints(g, nf, type_ = 'random'):
    x_min, x_max = -10, 20
    y_min, y_max = -25, 5
    
    # First value is 1 for bias term
    lscale = np.ones(nf+1)
    if type_ == 'random':
        sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
        fpoints = sampler.random(n=nf)
        fpoints[:,0] = x_min+(x_max-x_min)*fpoints[:,0]
        fpoints[:,1] = y_min+(y_max-y_min)*fpoints[:,1]
        
        return fpoints, lscale
    else:
        f_idx = np.random.randint(0, len(g), nf)
        fpoints = g[f_idx, 1:3]
        for r_idx, idx in enumerate(f_idx):
            if g[idx, 3] != 1:
                lscale[r_idx+1] = 0.5
            else:
                lscale[r_idx+1] = 0.5                
        return fpoints, lscale

"""
Analytical vs Autograd gradients: Try batch simulation
Find the proper number of feature points
"""
if __name__ == '__main__':
    from network_functions import generate_connected_graph
    T = 100000
    n = 4
    A = generate_connected_graph(n)
    # LOAD PARAMS
    
    fn_train, res, cell_max_min, skip, gamma, k_nearest, max_iter = load_sbkm_parameters()
    # read data
    g = pd.read_csv(fn_train, delimiter=',').values 
    g = g[80000:370000,:]
    X_train, Y_train, X_test, Y_test, X_ver, Y_ver = sbkm_data(g, complete=1, n=n)
    # plot_network(X_test, A)   
    # plt.show()
    
    # Plot the dataset accumulated from all time steps.
    # x_min, x_max = -13, 20
    # y_min, y_max = -25, 8
    # nrow, ncol = 2,2
    # fig, ax = plt.subplots(2,2, figsize=(n, n))
    # for idx in range(nrow):
        # for idxc in range(ncol):
            # didx = 2*idx+ idxc
            # ax[idx, idxc].scatter(X_train[didx,:, 0], X_train[didx,:, 1], c=Y_train[didx,:], s=1)
            # ax[idx, idxc].set_xlim([x_min, x_max])
            # ax[idx, idxc].set_ylim([y_min, y_max])
    
    B = 1 # Keep at 1 until batch implemented.
    M = 100
    lik_factor = 1
    
    data = {}
    for nf in [1500]: #[200, 400, 600, 800, 1200, 2000]:
        fpoints, lscale = sbkm_fpoints(g, nf, type_ = 'samples')
        type_ = 'gvip' # 'gvi' 'gvip' 'gvidp' 'dgvip' 'dgvidp'
        if type_ == 'gvi':
            mu_update, cov_update_t, error = gvi(T, M, B, X_train, Y_train, X_test, Y_test, fpoints, lscale)
        elif type_ == 'gvip':
            mu_update, cov_update_t, error = gvi_eprobit(T, B, X_train, Y_train, X_ver, Y_ver, fpoints, lscale)
        elif type_ == 'gvidp':
            mu_update, cov_update_t, error = gvi_diag_eprobit(T, B, X_train, Y_train, X_ver, Y_ver, fpoints, lscale)
        elif type_ == 'dgvip':
            data["A"] = A
            mu_update, cov_update_t, error = dist_gvi_eprobit(n, A, T, B, lik_factor, X_train, Y_train, X_ver, Y_ver, fpoints, lscale)
        elif type_ == 'dgvidp':
            data["A"] = A
            mu_update, cov_update_t, error = dist_gvi_diag_eprobit(n, A, T, B, lik_factor, X_train, Y_train, X_ver, Y_ver, fpoints, lscale)
        else:
            raise(NotImplementedError)
        
        data["train"] = [X_train, Y_train]
        data["test"] = [X_test, Y_test]
        data["verify"] = [X_ver, Y_ver]
        data["features"] = fpoints
        data["mu"] = mu_update
        data["cov"] = cov_update_t
        data["error"] = error
        data["lscale"] = lscale
        
        filename = 'slam_algo_'+type_+'_nf_'+str(nf)+'.pcl'
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if mu_update.shape[1] > 1:
        mu_update = mu_update[:, 0, :]
    else:
        mu_update = np.squeeze(mu_update)
    # fig, ax = plt.subplots()
    # for id_f in range(1, mu_update.shape[1]):
        # ax.plot(mu_update[:, id_f], label='weights')
    # ax.plot(mu_update[:,0], '--', label='bias')
    # ax.legend()
    # ax.grid()
    
    fig, ax = plt.subplots()
    ax.plot(np.array(error[0]).flatten())
    
    y_pred = []
    if len(cov_update_t.shape) == 3:
        cov = np.diag(cov_update_t[-1,0,:])
    elif len(cov_update_t.shape) == 2:
        cov = cov_update_t
    else:
        cov = cov_update_t[-1,0,:, :]
    
    for i, x in enumerate(X_test):
        Phi_X = feature_RBF(x, fpoints, lscale)
        phi_cov_phi = Phi_X@(cov@Phi_X)
        y_pred.append(forward_model(x, mu_update[-1,:], phi_cov_phi, fpoints, lscale))
    plot_categorical_data(X_test, Y_test)
    plot_predict_cat(X_test, y_pred)
    plt.show()
