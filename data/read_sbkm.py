import numpy as np
import pandas as pd

def load_sbkm_parameters(filename):
    parameters = {'intel': \
                      (filename,
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
        # Distributed heterogenous
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
            for idx in range(n-1):
                X_train[idx, :, :] = np.float_(g[remainder_list == idx, 1:3])
                Y_train[idx, :] = np.float_(g[remainder_list == idx, 3][:, np.newaxis]).ravel()  # * 2 - 1
                
            # 10% for testing
            X_test = np.float_(g[::n, 1:3])
            Y_test = np.float_(g[::n, 3][:, np.newaxis]).ravel() 
            return X_train, Y_train, X_test, Y_test
        
        else:
            assert n <= 10
            raise NotImplementedError('Heterogenous distribution to be done by scans')
            return X_train, Y_train, X_test, Y_test
        
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


def read_data(filename, complete=1, n=1):
    fn_train, res, cell_max_min, skip, gamma, k_nearest, max_iter = load_sbkm_parameters(filename)
    # read data
    g = pd.read_csv(fn_train, delimiter=',').values 
    X_train, Y_train, X_test, Y_test, X_ver, Y_ver = sbkm_data(g, complete, n)
    return X_train, Y_train, X_test, Y_test, X_ver, Y_ver
   