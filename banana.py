import numpy as np
import pickle
import random

# from data.floorplans.lidar.lidar import Lidar2D, OnlineTrajectoryLidarDataset
from data.read_banana import ReadBanana
from model.network_functions import generate_connected_graph
from model.slam_td import gvi_eprobit, feature_RBF, forward_model, plot_categorical_data, plot_predict_cat
import matplotlib.pyplot as plt


T = 20000
B = 1 # Keep at 1 until batch implemented.
M = 100
n = 1
lik_factor = 1

nf = 100
dbfile = open('./data/banana_f'+str(nf)+'.pcl', 'rb')
db = pickle.load(dbfile)
X_train, Y_train, X_test, Y_test, X_ver, Y_ver = db['data']
print ('Train set', X_train.shape[0], 
        'Test ', X_test.shape[0], 'Verify ', X_ver.shape[0])
            
fpoints, lscale = db['param']

type_ = 'gvip' # 'gvi' 'gvip' 'gvidp' 'dgvip' 'dgvidp'
if type_ == 'gvi':
    mu_update, cov_update_t, error = gvi(T, M, B, X_train, Y_train, X_test, Y_test, fpoints, lscale)
elif type_ == 'gvip':
    mu_update, cov_update_t, error = gvi_eprobit(T, B, X_train, Y_train, X_ver, Y_ver, fpoints, lscale)
elif type_ == 'gvidp':
    mu_update, cov_update_t, error = gvi_diag_eprobit(T, B, X_train, Y_train, X_ver, Y_ver, fpoints, lscale)
else:
    raise(NotImplementedError)

data = {'train':[X_train, Y_train], 'verify':[X_ver, Y_ver], 'test':[X_test, Y_test],
'features':fpoints, 'mu':mu_update, 'cov':cov_update_t, 'error':error, 'lscale':lscale}
        
filename = './results/banana'+type_+'_nf_'+str(nf)+'.pcl'
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