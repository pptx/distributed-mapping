"""
This file computes the update for Gaussian estimates in a variational form.
"""

"""
Note: The mean value of the particles and the resampled set is the same.
Therefore, resampling process is immaterial to the accuracy of the our estimates.
Rather, we want to fix the points that are initially sampled.

An attempt to sample from the mixed probability makes the sampling worse.
It is worse as the particles in a smaller domain are sampled. This is caused 
by fewer points being sampled in the entire space.
"""

import bayespy
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

import collections

import torch

# generate weights
def gen_weights(n):
    # weight = [np.random.rand() for i in range(n)]
    weight = [1. for i in range(n)]
    sum_weight = np.sum(weight)
    weight = [i/sum_weight for i in weight]
    return weight

def resampler(x, w):
    """
    Implement stratified resampling algorithm. 
    """
    resampled_x = np.zeros(x.shape)
    j = 0
    c = w[0]
    for i in range(len(w)):
        u = (1./len(w))*np.random.rand()
        beta = u + i/len(w)
        while beta>c:
            j = j + 1
            c = c + w[j]
        resampled_x[i,:] = x[j,:]
    return resampled_x
    
def create_lik_samples(weights, param_dict, obs, obs_mat, M=1e3):
    """
    Generate samples and compute the mixed and updated probabilities followed 
    by resampling relevant particles.
    """
    N_gen = [int(w*M) for w in weights]
    Nd = np.sum(N_gen)
    y_mixed = np.random.rand(Nd, param_dict['n_agent'])
    mult_sigma = np.zeros((param_dict['dim'], param_dict['dim']))
    for i in range(param_dict['n_agent']):
        mult_sigma += weights[i]*np.linalg.pinv(param_dict['parameters'][i][1])
    mult_sigma = np.linalg.inv(mult_sigma)
    mult_mu = [w*np.linalg.pinv(param_dict['parameters'][i][1]).dot(param_dict['parameters'][i][0].reshape(-1,1)) \
               for (w,i) in zip(weights, np.arange(param_dict['n_agent']))]
    mult_mu = np.array([np.sum([mult_mu[i][0] for i in range(param_dict['n_agent'])]), np.sum([mult_mu[i][1] for i in range(param_dict['n_agent'])])])
    mult_mu = mult_sigma.dot(mult_mu.reshape(-1,1))
    
    x = np.random.multivariate_normal(mult_mu.flatten(), mult_sigma, size=Nd)
    for i in range(param_dict['n_agent']):
        y_mixed[:, i] = multivariate_normal.logpdf(x, mean=param_dict['parameters'][i][0], cov=param_dict['parameters'][i][1]) 
    y = np.dot(y_mixed, np.array(weights).reshape(-1,1))
    
    H, Sigma_z = obs_mat
    prob_likelihood = [np.exp(y[i]+multivariate_normal.logpdf(obs, mean=x_i, cov=Sigma_z)) \
                        for i, x_i in enumerate(x)]
    
    y_sum = np.sum(prob_likelihood)
    w = [i/y_sum for i in prob_likelihood]
    
    resampled_x = resampler(x, w)
    return N_gen, x, w, resampled_x
    
import random
import copy
def convex_hull_samples(N_gen, x, param_dict):
    """
    Replace half the data with points inside the convex hull
    """
    Nd = np.sum(N_gen)
    x_mix = np.zeros(x.shape)
    avail_agents = [idx for idx, val in enumerate(N_gen) if val>0]
    for d_idx in range(Nd):
        if np.random.random_sample(1)>1./len(avail_agents):
            x_mix[d_idx,:] = x[d_idx,:]
        else:
            idx_list = [random.randint(int(np.sum(N_gen[:a_idx])), int(np.sum(N_gen[:a_idx+1])-1)) \
                        for a_idx in avail_agents]
            prob = np.random.rand(len(avail_agents))
            prob = (1./np.sum(prob))*prob
            x_mix[d_idx,:] = np.sum(np.array([p*x[idx,:] for p,idx in zip(prob, idx_list)]), axis = 0)
    return x_mix

def create_samples(weights, param_dict, obs, obs_mat, step, M=1e3):
    """
    Generate samples and compute the mixed and updated probabilities followed 
    by resampling relevant particles.
    """    
    N_gen = [int(w*M) for w in weights]
    Nd = np.sum(N_gen)
    x = np.random.rand(Nd, param_dict['dim'])
    y_mixed = np.random.rand(Nd, param_dict['n_agent'])
    for i in range(param_dict['n_agent']):
        # Increasing covariance by a unit may increase noise: +np.eye(param_dict['dim'])
        x[int(np.sum(N_gen[:i])):int(np.sum(N_gen[:i+1])),:] = \
        np.random.multivariate_normal(param_dict['parameters'][i][0], param_dict['parameters'][i][1]+(0.5/(step+1))*np.eye(param_dict['dim']), size=N_gen[i])
    x_mix = convex_hull_samples(N_gen, x, param_dict)
    for i in range(param_dict['n_agent']):
        y_mixed[:, i] = multivariate_normal.logpdf(x_mix, mean=param_dict['parameters'][i][0], cov=param_dict['parameters'][i][1]) 
    y = np.dot(y_mixed, np.array(weights).reshape(-1,1))
    
    H, Sigma_z = obs_mat
    prob_likelihood = [np.exp(y[i]+multivariate_normal.logpdf(obs.flatten(), mean=H.dot(x_i.reshape(-1,1)).flatten(), cov=Sigma_z)) \
                        for i, x_i in enumerate(x_mix)]
    y_sum = np.sum(prob_likelihood)
    w = [i/y_sum for i in prob_likelihood]
    
    resampled_x = resampler(x_mix, w)
    return N_gen, x, x_mix, w, resampled_x


from bayespy.nodes import Gaussian, Wishart
from bayespy.inference import VB

def gaussian_inference(resampled_x):
    Nd = resampled_x.shape[0]
    dim = resampled_x.shape[1]
    mu = Gaussian(np.random.rand(1,dim).flatten(), 1e-3*np.eye(dim))
    Lambda = Wishart(dim, 1e-3*np.eye(dim))
    X = Gaussian(mu, Lambda, plates=(Nd,), name='x')

    X.initialize_from_random()
    X.observe(resampled_x)

    Q = VB(mu, Lambda, X)
    Q.update(repeat=200)
    
    return mu, Lambda, X, Q

from pprint import pprint
def mu_random(mu_, n):
    """
    Sampling mean
    :.u: contains the list of moments
    :.phi: List of array of natural parameters
    """
    mean = mu_.u[0]
    cov = np.linalg.pinv(-2*mu_.phi[1])
    # object_methods = [method_name for method_name in dir(object)
                  # if callable(getattr(mu_, method_name))]
    # print (object_methods)
    # print (mu_.__dict__.keys())
    # pprint (vars(mu_))#dir(mu_))#, mu_.__dict__)mu_.get_parameters()
    # pprint (mu_)
    # U = np.linalg.cholesky(-2*mu_.phi[1])
    # print (np.linalg.pinv(np.dot(U.T, U)))
    # print (mu_)
    # mean, cov = mu.get_moments()[0], mu.get_moments()[1]
    return np.random.multivariate_normal(mean=mean.reshape(len(mean),), cov=cov, size=n)
    # return mu_.random()
    
from scipy.stats import wishart as wishart_
# https://www.bayespy.org/dev_guide/writingnodes.html
def cov_random(Lambda, n):
    """
    Sampling covariance
    Wishart prior provides precision matrix, i.e. inverse covariance
    """
    df = int(2*Lambda.phi[1])
    scale = np.linalg.pinv(-2*Lambda.phi[0])
    # print (df, scale, Lambda.u)
    return wishart_.rvs(df, scale, size=n)
    
    
def sample_pdf(mu, Lambda):
    """
    Generate samples from a density function with hyperparameters
    mu sampled from a Gaussian and Lambda sampled from Wishart
    """
    Ns = 1000
    mu_samples = mu_random(mu, Ns)
    cov_samples = cov_random(Lambda, Ns)
    # print (mu_samples, cov_samples)
    samples = [np.random.multivariate_normal(mu_1, np.linalg.pinv(cov_1), 1) \
                for mu_1, cov_1 in zip(mu_samples, cov_samples)]
    samples = np.array(samples).reshape(-1,2)
    # NotImplementedError
    # print (mu.random(), Lambda.random() )
    
    return samples
    
def gaussian_mixture_2d(X, alpha=None, scale=2, fill=False, axes=None, **kwargs):
    """
    Plot Gaussians as ellipses in 2-D

    Parameters
    ----------

    X : Gaussian node

    scale : float (optional)
       Scale for the covariance ellipses (by default, 2)
    """

    if axes is None:
        axes = plt.gca()

    mu_Lambda = X._ensure_moments(X.parents[1], GaussianWishartMoments)

    (mu, _, Lambda, _) = mu_Lambda.get_moments()
    mu = np.linalg.solve(Lambda, mu)

    if len(mu_Lambda.plates) != 1:
        raise NotImplementedError("Not yet implemented for more plates")

    K = mu_Lambda.plates[0]

    width = np.zeros(K)
    height = np.zeros(K)
    angle = np.zeros(K)

    for k in range(K):
        m = mu[k]
        L = Lambda[k]
        (u, W) = scipy.linalg.eigh(L)
        u[0] = np.sqrt(1/u[0])
        u[1] = np.sqrt(1/u[1])
        width[k] = 2*u[0]
        height[k] = 2*u[1]
        angle[k] = np.arctan(W[0,1] / W[0,0])

    angle = 180 * angle / np.pi
    mode_height = 1 / (width * height)

    # Use cluster probabilities to adjust alpha channel
    if alpha is not None:
        # Compute the normalized probabilities in a numerically stable way
        logsum_p = misc.logsumexp(alpha.u[0], axis=-1, keepdims=True)
        logp = alpha.u[0] - logsum_p
        p = np.exp(logp)
        # Visibility is based on cluster mode peak height
        visibility = mode_height * p
        visibility /= np.amax(visibility)
    else:
        visibility = np.ones(K)

    for k in range(K):
        ell = mpl.patches.Ellipse(mu[k], scale*width[k], scale*height[k],
                                  angle=(180+angle[k]),
                                  fill=fill,
                                  alpha=visibility[k],
                                  **kwargs)
        axes.add_artist(ell)

    plt.axis('equal')

    # If observed, plot the data too
    if np.any(X.observed):
        mask = np.array(X.observed) * np.ones(X.plates, dtype=np.bool)
        y = X.u[0][mask]
        plt.plot(y[:,0], y[:,1], 'r.')

    return
    
if __name__ == '__main__':
    # Set up the data operation
    n_agent = 5
    dim = 2
    # Agent's observation model
    H = 0.5*np.eye(dim)
    Sigma_z = 0.1*np.eye(dim)
    
    prob_type = ['Normal' for i in range(n_agent)]
    n_params = [2 for i in range(n_agent)]
    # List of tuples for the Normal parameters
    params = [(4*np.random.rand(dim), (1./4)*np.random.rand()*np.eye(dim)) for i in range(n_agent)]

    param_dict = {'n_agent':n_agent, 'dim':dim, 'prob_type':prob_type, 'n_params':n_params, 'parameters': params}
    weights = gen_weights(n_agent)

    obs = 7*np.random.rand(1, dim)
    M = 1e5
    N_gen, x, x_mix, w, resampled_x = create_samples(weights, param_dict, obs, (H, Sigma_z), M)
    # N_gen, x, w, resampled_x = create_lik_samples(weights, param_dict, obs, (H, Sigma_z), M)
    print (x.shape, len(w), N_gen, np.array(w).reshape(1,-1).dot(x))
    mu, Lambda, X, Q = gaussian_inference(resampled_x)
    print (mu, Lambda, Lambda.u)
    samples = sample_pdf(mu, Lambda)
    
    fig, ax = plt.subplots()
    for i in range(n_agent):
        ax.scatter(x[int(np.sum(N_gen[:i])):int(np.sum(N_gen[:i+1])),0], x[int(np.sum(N_gen[:i])):int(np.sum(N_gen[:i+1])),1], s=5, alpha=0.3)
    plt.savefig('sample_comp.pdf', bbox_inches='tight')
    
    # Mixed particle
    fig, ax = plt.subplots()
    ax.scatter(x_mix[:,0], x_mix[:,1], s = 5, alpha=0.5)
    ax.set_xlim([min(x[:,0]), max(x[:,0])])
    ax.set_ylim([min(x[:,1]), max(x[:,1])])
    
    # Should look the same as before due to the sampling
    fig, ax = plt.subplots()
    ax.scatter(resampled_x[:,0], resampled_x[:,1], s = 5, alpha=0.5)
    ax.set_xlim([min(x[:,0]), max(x[:,0])])
    ax.set_ylim([min(x[:,1]), max(x[:,1])])
    
    
    Nd = resampled_x.shape[0]
    # Compare the estimated parameters
    mult_sigma = np.dot(H.T, np.linalg.pinv(Sigma_z).dot(H))
    for i in range(n_agent):
        mult_sigma += weights[i]*np.linalg.pinv(param_dict['parameters'][i][1])
    mult_sigma = np.linalg.inv(mult_sigma)
    mult_mu = [w*np.linalg.pinv(param_dict['parameters'][i][1]).dot(param_dict['parameters'][i][0].reshape(-1,1)) \
               for (w,i) in zip(weights, np.arange(n_agent))]
    mult_mu = np.array([np.sum([mult_mu[i][0] for i in range(n_agent)]), np.sum([mult_mu[i][1] for i in range(n_agent)])])
    mult_mu = mult_sigma.dot(mult_mu.reshape(-1,1)+np.dot(H.T, np.linalg.pinv(Sigma_z).dot(obs.reshape(-1,1))))
    print ('The analytical mean is', mult_mu.flatten(), 'and the mean of resampled points is', np.mean(resampled_x, axis=0))
    print ('The analytical covariance is', mult_sigma, 'estimated is', np.linalg.pinv(Lambda.u[0]), 'and the covariance of resampled points is', np.cov(resampled_x.T))

    # mu.show()
    # Lambda.show()#, X.show()
    
    mu_ = mu.get_moments()[0]
    cov_ = np.linalg.pinv(Lambda.u[0])

    from scipy.stats import multivariate_normal
    distr = multivariate_normal(mean = mu_, cov = cov_)

    mean_1, mean_2 = mu_[0], mu_[1]
    sigma_1, sigma_2 = cov_[0,0], cov_[1,1]
    
    # x_1 = np.linspace(np.min(x[:,0]), np.max(x[:,0]), num=100)
    # x_2 = np.linspace(np.min(x[:,1]), np.max(x[:,1]), num=100)
    x_1 = np.linspace(min(min(samples[:,0]), min(resampled_x[:,0])), max(max(samples[:,0]), max(resampled_x[:,0])), num=100)
    x_2 = np.linspace(min(min(samples[:,1]), min(resampled_x[:,1])), max(max(samples[:,1]), max(resampled_x[:,1])), num=100)
    X, Y = np.meshgrid(x_1, x_2)

    pdf = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pdf[i,j] = distr.pdf([X[i,j], Y[i,j]])

    fig, ax = plt.subplots()
    ax.contourf(X, Y, pdf, cmap='GnBu')
    ax.scatter(resampled_x[:,0], resampled_x[:,1], color='chocolate', s = 5, alpha=0.3, label='Resampled data')
    ax.scatter(samples[:,0], samples[:,1], s=5, c ='b', marker='^', alpha=0.1, label='Estimate samples')
    ax.scatter(mult_mu[0], mult_mu[1], s=50, c ='black', marker='s', label='Analytical mean')
    ax.scatter(mu_[0], mu_[1], s=50, c ='red', marker='^', label='Estimate mean')
    ax.legend()
    plt.savefig('resampled_estimates.pdf', bbox_inches='tight')
    plt.show()