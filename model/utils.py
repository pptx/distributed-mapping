import numpy as np
import matplotlib.pyplot as plt

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