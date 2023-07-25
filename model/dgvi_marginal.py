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

"""
This file has full and diagonalized covariance implementation 
of Gaussian variational inference and distributed Gaussian variational inference.
"""


class Model:
    def __init__(self, xi, fpoints, lscale):
        self.xi = xi
        self.fpoints = fpoints 
        self.lscale = lscale

    @staticmethod
    def convert_to_ds(A):
        err = 0.1
        while err > 1e-3:
            A_prev = A
            # Make row stochastic
            A = A/np.sum(A, axis=1).reshape(-1,1)
            # Make column stochastic
            A = np.transpose(A.T/np.sum(A, axis=0).reshape(-1,1))
            err = np.linalg.norm(A-A_prev)
        return A
        
    def feature_RBF(self, agent_id, x):
        """
        Type: Specify poly or RBF
        args: power of function or feature points
        """
        relevant_indices = np.where(self.member[:,agent_id]>0)[0]-1 # Move to __init__
        fpoints_ = np.vstack((x, self.fpoints[relevant_indices[1:], :])) # Create bias vector, maybe append lscale here
        nf = fpoints_.shape[0]
        dist = np.linalg.norm(fpoints_ - np.tile(x, (nf, 1)), ord=1, axis = 1) # Order 1 norm
        # dist = np.sum( (self.fpoints - np.tile(x, (nf, 1)))**2, axis=1)
        l_indices = relevant_indices # l_scale contains extra first element 1.
        lscale_ = self.lscale[l_indices]
        return np.exp(-1*dist/(2*lscale_**2))
        
    def sigmoid(self, x):
        return 1./(1+np.exp(-x))
        
    # @jit(nopython=True)
    def forward_model(self, agent_id, x, theta, phi_cov_phi):
        Phi_X = self.feature_RBF(agent_id, x)
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
        agent_id = 0
        print ("Are you using the right data set?")
        for i, x in enumerate(X_p):
            y_pred = self.forward_model(x, agent_id, n_mu, phi_cov_phi)
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

    
class DistributedMarginalClassification(Model):
    
    def __init__(self, xi, fpoints, lscale, FC, A, member, lik_factor, X, y, X_p, y_p):
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
        self.member = member
        self.lscale = lscale
        self.FC = FC
        
        self.A = A
        self.n = np.shape(self.A)[0]
        self.lik_factor = lik_factor
        self.X = X
        self.y = y 
        self.X_p = X_p
        self.y_p = y_p
        
        self.nv = len(self.X_p)
        super(DistributedMarginalClassification, self).__init__(xi, fpoints, lscale)

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
        raise NotImplementedError
        return 0
        
    def training_indices(self, T):
        ndata = [self.X[i].shape[0] for i in range(self.n)]
        T_min = min(ndata)
        
        if T<T_min:
            train_idx = np.reshape(np.array([np.random.choice(range(self.X[i].shape[0]), T, replace=False) \
                                        for i in range(self.n)]), (self.n, T))
            idtype = 'No_replacement'
        else:
            train_idx = np.reshape(np.array([np.hstack\
                                            ((np.random.choice(range(self.X[i].shape[0]), T_min, replace=False),\
                                            np.random.randint(0, self.X[i].shape[0], T-T_min))) \
                                            for i in range(self.n)]), (self.n, T))
            idtype = 'With_replacement'
        return train_idx, idtype
        
    def dist_gvi_diag_eprobit(self, T, reindexer, invindexer):
        
        """
        Implement distributed Gaussian variational inference algorithm 
        :Inputs:
        n: Number of agents
        T: Number of iterations
        reindexer: 
        invindexer: 
        :Outputs:
        mu: Mean value over the model weights
        cov: Covariance over the weight parameters
        error: Binary cross entropy error over the verification data set during training
        """
        d_dim = [int(self.FC[int(i),int(i)]) for i in range(self.n)]
        mu_update = [np.zeros((T+1, d_dim[i])) for i in range(self.n)]
        
        alfa = 5.
        n_sigma_ = [alfa*np.ones((dim_i)) for dim_i in d_dim]
        omega_ = [(1./alfa)*np.ones((dim_i)) for dim_i in d_dim]
        n_omega_ = [(1./alfa)*np.ones((dim_i)) for dim_i in d_dim]
        
        # Running the Bayesian propagation algorithm for T steps
        error = [[] for i in range(self.n)]
        l_error = [[] for i in range(self.n)]
        # Compute list of T-training indices for agents
        train_idx, idtype = self.training_indices(T)
        # print (train_idx.shape, idtype)
        
        for t in range(T):
            if t%100 == 0:
                print (t) ;sys.stdout.flush()
            
            for i in range(self.n):
                
                # Compute geometric average as (mu, cov_i)
                mu = np.zeros(d_dim[i])
                omega_i = np.zeros(d_dim[i])
                
                for j in range(self.n):
                    if self.A[i, j] > 0:
                        # Marginal from agent j
                        omega_ji = reindexer[i][j]@omega_[j]
                        mu_ji = reindexer[i][j]@mu_update[j][t,:]
                        # print (omega_ji)
                        # Conditional for agent j
                        omega_i_j = invindexer[i][j]@omega_[i] 
                        mu_i_j = invindexer[i][j]@mu_update[i][t,:]
                        # print (omega_i_j)
                        # Conditional marginal product
                        omega_i_cp = omega_i_j+omega_ji 
                        nu_i_cp = omega_i_j*mu_i_j+omega_ji*mu_ji 
                        # print(omega_i_cp)
                        omega_i += self.A[i, j]*(omega_i_cp)
                        
                        mu = mu + self.A[i, j]*(nu_i_cp)
                cov_i = omega_i**(-1)
                mu = cov_i*mu
                
                # Compute derivative and double derivative terms
                db_id = train_idx[i, t]
                Phi_X = self.feature_RBF(i, self.X[i][db_id,:])
                cov_phi = cov_i*Phi_X
                der, dder, dsig = self.der_dder_dsig(self.y[i][db_id], cov_phi, Phi_X, mu)
                
                # Compute verification error
                # if (t%500==0):
                    # phi_cov_phi = Phi_X@cov_phi
                    # err, lerr = self.verify_error(self.X_p, self.y_p, mu, phi_cov_phi)
                    # error[i].append(err)
                    # l_error[i].append(lerr)
                    # print('For agent', i, ', average error is', err/self.nv, ', and percentage error is ', lerr/self.nv)
                    
                # Generate likelihood update
                n_omega_[i] = n_omega_[i] + self.lik_factor*dder
                n_sigma_[i] = n_omega_[i]**(-1)
            
                n_mu = mu + self.lik_factor*n_sigma_[i]*der
                # print (n_mu, n_sigma_)
                mu_update[i][t+1,:] = n_mu.flatten()
            omega_ = n_omega_
                
        final_sigma = [np.zeros((d_dim_i, d_dim_i)) for d_dim_i in d_dim]
        for i in range(self.n):
            final_sigma[i] = np.diag(n_sigma_[i])
        return mu_update, final_sigma, [error, l_error]
    

if __name__ == '__main__':
    from network_functions import generate_connected_graph
    T, n = 200000, 7
    xi = 0.61
    
    filename = './../data/marginal_dinno_r1k.pcl'
    dbfile = open(filename, 'rb')
    db = pickle.load(dbfile)
    X_train, Y_train, X_test, Y_test, X_ver, Y_ver = db['data']
    FC = db['comm'] #Communication matrix containing #(common parameters) 
    fpoints, lscale = db['model_param']
    member, reindexer, invindexer = db['sharing_info']
    print ('Train set', [X.shape[0] for X in X_train], 
            'Test ', X_test[0].shape[0], 'Verify ', X_ver[0].shape[0])
    for i in range(n):
        for j in range(n):
            print (invindexer[i][j].shape, reindexer[i][j].shape)
    lik_factor = n
    A = Model.convert_to_ds(FC)
    np.set_printoptions(precision=3)
    print (FC, A)
    
    marginal_classifier = DistributedMarginalClassification(xi, fpoints, lscale, FC, A, member, lik_factor, \
                                                            X_train, Y_train, X_ver, Y_ver)
    data = {}
    type_ = 'dgvidp' # 'dgvip' 'dgvidp'
    if type_ == 'dgvip':
        mu_update, cov_update_t, error = marginal_classifier.dist_gvi_eprobit(T)
    elif type_ == 'dgvidp':
        mu_update, cov_update_t, error = marginal_classifier.dist_gvi_diag_eprobit(T, reindexer, invindexer)
    else:
        raise(NotImplementedError)
        
    data["A"] = A
    data["train"] = [X_train, Y_train]
    data["test"] = [X_test, Y_test]
    data["verify"] = [X_ver, Y_ver]
    data["features"] = fpoints
    data["member"] = member
    data["mu"] = mu_update
    data["cov"] = cov_update_t
    data["error"] = error
    data["lscale"] = lscale
    
    filename = 'marginal_dinno_'+type_+'.pcl'
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.show()