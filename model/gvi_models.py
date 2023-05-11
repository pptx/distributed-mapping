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
from .utils import feature_RBF, forward_model, plot_categorical_data, plot_predict_cat

"""
This file has full and diagonalized covariance implementation 
of Gaussian variational inference and distributed Gaussian variational inference.
"""


class Model:
    def __init__(self, xi, fpoints, lscale):
        self.xi = xi
        self.fpoints = fpoints 
        self.lscale = lscale
        
    def feature_RBF(self, x):
        """
        Type: Specify poly or RBF
        args: power of function or feature points
        """
        fpoints_ = np.vstack((x, self.fpoints)) # Create bias vector, maybe append lscale here
        nf = fpoints_.shape[0]
        dist = np.linalg.norm(fpoints_ - np.tile(x, (nf, 1)), ord=1, axis = 1) # Order 1 norm
        # dist = np.sum( (self.fpoints - np.tile(x, (nf, 1)))**2, axis=1)
        return np.exp(-1*dist/(2*self.lscale**2))
        
    def sigmoid(self, x):
        return 1./(1+np.exp(-x))
        
    # @jit(nopython=True)
    def forward_model(self, x, theta, phi_cov_phi):
        Phi_X = self.feature_RBF(x)
        den = (1+phi_cov_phi)**(0.5)
        return self.sigmoid(Phi_X.dot(theta)/den)
        
    def der_dder_dsig(self, y, cov_phi, Phi_X, mu):
        """
        Return the first and second derivatives wrt xi, and the first derivative wrt cov_phi and mu
        """
        #Dot products of cov_phi and mu with Phi_X
        phi_cov_phi = Phi_X@cov_phi
        phi_mu = Phi_X@mu

        op = Phi_X*Phi_X # Calculate outer product of Phi_X with itself
        # Calculate parameters for second derivative
        beta = 1 + (self.xi**2)*phi_cov_phi
        xi2_beta = self.xi**2/beta
        gamma = (xi2_beta/(2*np.pi))**(1./2)
        gamma = gamma*np.exp(- (xi2_beta/2.)*(phi_mu*phi_mu))
        dder = gamma*op # Calculate second derivative 

        der = self.xi*(phi_mu/(beta**(0.5))) # Calculate first derivative
        der_coeff = (y - norm.cdf(der)) # Calculate derivative coefficient for first derivative
        der = der_coeff*Phi_X

        dsig = der_coeff*cov_phi - gamma*(1./(1.+gamma*phi_cov_phi))*(cov_phi@der)*cov_phi
        return der, dder, dsig

    def verify_error(self, X_p, y_p, n_mu, phi_cov_phi):
        err, lerr = 0, 0
        for i, x in enumerate(X_p):
            y_pred = self.forward_model(x, n_mu, phi_cov_phi)
            err += -(y_p[i]*np.log(y_pred) + (1-y_p[i])*np.log(1-y_pred))
            lerr += np.abs(y_p[i] - y_pred)
            
        return err, lerr
        
    def plot_categorical_data(self, X, y):
        fig, ax = plt.subplots()
        indices = np.where(np.array(y)==0)
        n_indices = np.where(np.array(y)==1)
        ax.scatter(X[indices, 0], X[indices, 1], marker='^',s=1)
        ax.scatter(X[n_indices, 0], X[n_indices, 1], marker='o',s=1)
        return ax
        
    def plot_predict_cat(self, X, y):
        fig, ax = plt.subplots()
        indices = np.where(np.array(y)<=0.5)
        n_indices = np.where(np.array(y)>0.5)
        ax.scatter(X[indices, 0], X[indices, 1], marker='^', s=1)
        ax.scatter(X[n_indices, 0], X[n_indices, 1], marker='o', s=1)
        return ax

    
class DistributedClassification(Model):
    
    def __init__(self, xi, fpoints, lscale, A, lik_factor, X, y, X_p, y_p):
        """
        :Inputs:
        n: Number of agents
        A: Doubly stochastic matrix representing connected communication graph
        X: Training set of dimensions (n, Nx, d)
        y: Binary training output of dimensions (n, Nx, 1)
        lik_factor: Weight on likelihood terms
        X: Verification set of dimensions (n, Np, d)
        y: Binary verification output of dimensions (n, Np, 1)
        """
        self.xi = xi
        self.fpoints = fpoints 
        self.lscale = lscale
        
        self.A = A
        self.n = np.shape(self.A)[0]
        self.lik_factor = lik_factor
        self.X = X
        self.y = y 
        self.X_p = X_p
        self.y_p = y_p
        
        self.nv = len(self.X_p)
        super(DistributedClassification, self).__init__(xi, fpoints, lscale)

    def dist_gvi_eprobit(self, T):
        """
        Implement distributed Gaussian variational inference algorithm 
        :Inputs:
        n: Number of agents
        T: Number of iterations
        :Outputs:
        mu: Mean value over the model weights
        cov: Covariance over the weight parameters
        error: Binary cross entropy error over the verification data set during training
        """
        d_dim = self.fpoints.shape[0]+1
        mu_update = np.zeros((T+1, self.n, d_dim))
        
        n_sigma_ = np.zeros((self.n, d_dim, d_dim))
        omega_ = np.zeros((self.n, d_dim, d_dim))
        n_omega_ = np.zeros((self.n, d_dim, d_dim))
        for i in range(n):
            omega_[i,:,:] = 1e-3*np.eye(d_dim)
        # Running the Bayesian propagation algorithm for T steps
        error = [[] for i in range(self.n)]
        l_error = [[] for i in range(self.n)]
        train_idx = np.random.randint(0, self.X.shape[1]-1, T)
        for t in range(T):
            if t%100 == 0:
                print (t) ;sys.stdout.flush()
            
            for i in range(self.n):
                mu = np.zeros(d_dim)
                omega_i = np.zeros((d_dim, d_dim))
                
                for j in range(self.n):
                    omega_i = omega_i + self.A[i, j]*omega_[j,:,:]
                    mu = mu + self.A[i, j]*(omega_[j,:,:]@mu_update[t,j,:])
                    
                db_id = train_idx[t]
                Phi_X = feature_RBF(self.X[i,db_id,:])
                
                omega_i = scipy.sparse.csc_matrix(omega_i).tocsc()
                solved = sla.spsolve(omega_i, np.vstack([mu, Phi_X]).T )
                mu = solved[:,0]
                # cov_phi = cov_i@Phi_X
                cov_phi = solved[:,1]

                der, dder, dsig = self.der_dder_dsig(self.y[i, db_id], cov_phi, Phi_X, mu)
                # Generate likelihood update
                n_omega_[i,:,:] = n_omega_[i,:,:] + self.lik_factor*dder
                n_mu = mu + self.lik_factor*dsig
                # print (n_mu, n_sigma_)
                mu_update[t+1,i,:] = n_mu.flatten()
            omega_ = n_omega_
            
            # Compute verification error
            if (t%500==0):
                err, lerr = 0, 0
                for i, x in enumerate(self.X_p):
                    y_pred = forward_model(x, n_mu.flatten(), phi_cov_phi)
                    err += -(self.y_p[i]*np.log(y_pred) + (1-self.y_p[i])*np.log(1-y_pred))
                    lerr += np.abs(self.y_p[i] - y_pred)
                error.append(err)
                l_error.append(lerr)
                print('Average error is', err/self.nv, 'Percentage error is ', lerr/self.nv)
        return mu_update, n_sigma_, [error, l_error]
        
    def dist_gvi_diag_eprobit(self, T):
        
        """
        Implement distributed Gaussian variational inference algorithm 
        :Inputs:
        n: Number of agents
        T: Number of iterations
        :Outputs:
        mu: Mean value over the model weights
        cov: Covariance over the weight parameters
        error: Binary cross entropy error over the verification data set during training
        """
        # T = len(X)
        d_dim = self.fpoints.shape[0]+1
        mu_update = np.zeros((T+1, self.n, d_dim))
        
        alfa = 5.
        n_sigma_ = alfa*np.ones((self.n, d_dim))
        omega_ = (1./alfa)*np.ones((self.n, d_dim))
        n_omega_ = (1./alfa)*np.ones((self.n, d_dim))
        
        # Running the Bayesian propagation algorithm for T steps
        error = [[] for i in range(self.n)]
        l_error = [[] for i in range(self.n)]
        train_idx = np.reshape(np.array([np.random.randint(0, self.X[i].shape[0]-1, T) for i in range(self.n)]), (self.n, T))

        for t in range(T):
            if t%100 == 0:
                print (t) ;sys.stdout.flush()
            
            for i in range(self.n):
                
                # Compute geometric average as (mu, cov_i)
                mu = np.zeros(d_dim)
                omega_i = np.zeros(d_dim)
                
                for j in range(self.n):
                    omega_i = omega_i + self.A[i, j]*omega_[j,:]
                    mu = mu + self.A[i, j]*(omega_[j,:]*mu_update[t,j,:])
                cov_i = omega_i**(-1)
                mu = cov_i*mu
                
                # Compute derivative and double derivative terms
                db_id = train_idx[i, t]
                # db_id = [random.randint(0, len(X)-1) if db != 0 else t][0]# Since T = #data points.
                # db_id = random.randint(0, min(t+1, len(X))-1) 
                Phi_X = self.feature_RBF(self.X[i][db_id,:])
                cov_phi = cov_i*Phi_X
                
                # Compute verification error
                if (t%500==0):
                    phi_cov_phi = Phi_X@cov_phi
                    err, lerr = self.verify_error(self.X_p, self.y_p, mu, phi_cov_phi)
                    error[i].append(err)
                    l_error[i].append(lerr)
                    print('For agent', i, ', average error is', err/self.nv, ', and percentage error is ', lerr/self.nv)

                der, dder, dsig = self.der_dder_dsig(self.y[i][db_id], cov_phi, Phi_X, mu)
                    
                # Generate likelihood update
                n_omega_[i,:] = n_omega_[i,:] + self.lik_factor*dder
                n_sigma_[i,:] = n_omega_[i,:]**(-1)
            
                n_mu = mu + self.lik_factor*n_sigma_[i,:]*der
                # print (n_mu, n_sigma_)
                mu_update[t+1,i,:] = n_mu.flatten()
            omega_ = n_omega_
                
            final_sigma = np.zeros((self.n, d_dim, d_dim))
            for i in range(self.n):
                final_sigma[i,:,:] = np.diag(n_sigma_[i, :])
        return mu_update, final_sigma, [error, l_error]
    

if __name__ == '__main__':
    from network_functions import generate_connected_graph
    T = 100000
    n = 4
    A = generate_connected_graph(n)
    # LOAD PARAMS
    
    fn_train, res, cell_max_min, skip, gamma, k_nearest, max_iter = load_sbkm_parameters()
    # read data
    g = pd.read_csv(fn_train, delimiter=',').values 
    print (len(g), A)
    g = g[80000:370000,:]
    X_train, Y_train, X_test, Y_test, X_ver, Y_ver = sbkm_data(g, complete=1, n=n)
    
    B = 1 # Keep at 1 until batch implemented.
    M = 100
    lik_factor = n
    
    data = {}
    for nf in [1500]: #[200, 400, 600, 800, 1200, 2000]:
        fpoints, lscale = sbkm_fpoints(g, nf, type_ = 'samples')
        type_ = 'dgvidp' # 'gvi' 'gvip' 'gvidp' 'dgvip' 'dgvidp'
        if type_ == 'dgvip':
            mu_update, cov_update_t, error = dist_gvi_eprobit(n, A, T, B, lik_factor, X_train, Y_train, X_ver, Y_ver, fpoints, lscale)
        elif type_ == 'dgvidp':
            mu_update, cov_update_t, error = dist_gvi_diag_eprobit(n, A, T, B, lik_factor, X_train, Y_train, X_ver, Y_ver, fpoints, lscale)
        else:
            raise(NotImplementedError)
        
        
        data["A"] = A
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

    plt.show()