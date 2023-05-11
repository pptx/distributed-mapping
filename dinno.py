import numpy as np
import glob
import os
import pickle
import random

# from data.floorplans.lidar.lidar import Lidar2D, OnlineTrajectoryLidarDataset
from data.read_sbkm import read_data
from model.network_functions import generate_connected_graph
from model.gvi_models import Model, DistributedClassification
from model.utils import feature_RBF, forward_model, plot_categorical_data, plot_predict_cat
import matplotlib.pyplot as plt

xi = 0.61
T = 100000
n = 7
lik_factor = 1

# dinnoc = ReadDinno()
# X_train, Y_train, X_test, Y_test, X_ver, Y_ver = dinnoc.train_test_set()
# fpoints, lscale = dinnoc.dinno_fpoints(np.hstack((X_test, Y_test)), nf, type_ = 'random')

dbfile = open('./data/dinno_r1k.pcl', 'rb')
db = pickle.load(dbfile)
X_train, Y_train, X_test, Y_test, X_ver, Y_ver = db['data']
print ('Train set', [X.shape[0] for X in X_train], 
        'Test ', X_test.shape[0], 'Verify ', X_ver.shape[0])
            
fpoints, lscale = db['param']
A = generate_connected_graph(n)
modelc = Model(xi, fpoints, lscale)
classify = DistributedClassification(xi, fpoints, lscale, A, lik_factor, X_train, Y_train, X_ver, Y_ver)

for nf in [1000]: # [200, 400, 600, 800, 1200, 2000]
    type_ = 'dgvidp' # 'gvi' 'gvip' 'gvidp' 'dgvip' 'dinno_'
    if type_ == 'dgvip':
        mu_update, cov_update_t, error = classify.dist_gvi_eprobit(T)
    elif type_ == 'dgvidp':
        mu_update, cov_update_t, error = classify.dist_gvi_diag_eprobit(T)
    else:
        raise(NotImplementedError)
        

data = {'A':A, 'train':[X_train, Y_train], 'verify':[X_ver, Y_ver], 'test':[X_test, Y_test],
'features':fpoints, 'mu':mu_update, 'cov':cov_update_t, 'error':error, 'lscale':lscale}

filename = './results/dinno_'+type_+'_nf_'+str(nf)+'l1'+'.pcl'
with open(filename, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if mu_update.shape[1] > 1:
    mu_update = mu_update[:, 0, :]
else:
    mu_update = np.squeeze(mu_update)


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