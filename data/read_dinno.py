#!/usr/local/bin/python39
import glob
import numpy as np
import os
from scipy.stats import qmc
import pickle
import random

class ReadDinno(object):
    def __init__(self):
        # Settings
        self.data_dir = "./floorplans/32_data"
        self.waypoint_subdir = "some_overlap" #minimal_overlap, some_overlap, tight_paths
        img_path = os.path.join(self.data_dir, "floor_img.png")
        num_beams = 20
        beam_samps = 10 # Number of points selected from each beam
        beam_length = 0.2
        samp_distribution_factor = 1.0
        collision_samps = 50
        fine_samps = 3
        self.num_scans_in_window = 30
        self.spline_res = 30

        # Setup Lidar Object
        self.lidar = Lidar2D(img_path, num_beams, beam_length, beam_samps,
            samp_distribution_factor, collision_samps, fine_samps, border_width=30)

        # Waypoint Settings
        self.wp_names = glob.glob1(os.path.join(self.data_dir, self.waypoint_subdir), "*.npy") 
        self.datasets = []
        

    def create_datasets(self):
        
        for wp_name in self.wp_names:
            wp_path = os.path.join(self.data_dir, self.waypoint_subdir, wp_name)
            wps = np.load(wp_path)
            ds = OnlineTrajectoryLidarDataset(self.lidar, wps, 
                self.spline_res, self.num_scans_in_window)
            self.datasets.append(ds)
        
    def train_test_set(self):
        self.create_datasets()
        # 7 datasets 
        n = 7
        X_train = []
        Y_train = []
        test_set = [[] for i in range(n)]
        ver_set = [[] for i in range(n)]
        # For 25 beam_samps, (12, 61, 317)
        train_f = 3
        test_f = 11
        ver_f = 81
        for idx in range(n):
            scan_points = np.array(self.datasets[idx].scans[::train_f, :])
            X_train.append(scan_points[:,0:2])
            Y_train.append(scan_points[:,2])
            scan_points = np.array(self.datasets[idx].scans[::test_f, :])
            test_set[idx].append(scan_points)
            scan_points = np.array(self.datasets[idx].scans[::ver_f, :])
            ver_set[idx].append(scan_points)
            
        test_set = np.concatenate([arr[0] for arr in test_set], axis=0)
        ver_set = np.concatenate([arr[0] for arr in ver_set], axis=0)

        return X_train, Y_train, test_set[:,0:2], test_set[:,2]\
        , ver_set[:,0:2], ver_set[:,2]
        
    def dinno_fpoints(self, X_test, Y_test, nf, type_ = 'random'):
        x_min, x_max = -1100, 1100
        y_min, y_max = -800, 800
        
        # First value is 1 for bias term
        lscale = 8*np.ones(nf+1)
        sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
        if type_ == 'random':
            fpoints = sampler.random(n=nf)
            fpoints[:,0] = x_min+(x_max-x_min)*fpoints[:,0]
            fpoints[:,1] = y_min+(y_max-y_min)*fpoints[:,1]
            
            return fpoints, lscale
        else:
            # Three classes, 0s, 1s and random samples.
            nf1 = int(nf/3.)
            # Indices with value 1 representing obstacles
            idx1 = np.where(Y_test == 1)[0] 
            # Sample indices without replacement
            sel_idx1 = random.sample(idx1.tolist(), nf1) 
            nf2 = 2*nf1
            idx0 = np.where(Y_test == 0)[0] 
            sel_idx0 = random.sample(idx0.tolist(), nf2-nf1) 
            
            rpoints = sampler.random(n=nf-nf2)
            rpoints[:,0] = x_min+(x_max-x_min)*rpoints[:,0]
            rpoints[:,1] = y_min+(y_max-y_min)*rpoints[:,1]
            
            fpoints = np.zeros((nf, 2))
            fpoints[:nf1, :] = X_test[sel_idx1, :]
            fpoints[nf1:nf2, :] = X_test[sel_idx0, :]
            fpoints[nf2:, :] = rpoints
                         
            lscale[:nf1] = 0.5*lscale[:nf1]
            lscale[nf2:] = 3*lscale[nf2:]
            return fpoints, lscale

if __name__ == '__main__':
    from floorplans.lidar.lidar import Lidar2D, OnlineTrajectoryLidarDataset
    
    dinnoc = ReadDinno()
    X_train, Y_train, X_test, Y_test, X_ver, Y_ver = dinnoc.train_test_set()
    fpoints, lscale = dinnoc.dinno_fpoints(X_test, Y_test, nf=1000, type_ = 'sample')
    
    dataset = {'data':[X_train, Y_train, X_test, Y_test, X_ver, Y_ver],
    'param':[fpoints, lscale]}
    # Its important to use binary mode
    dbfile = open('dinno_r1k.pcl', 'wb')
    # source, destination
    pickle.dump(dataset, dbfile)                     
    dbfile.close()