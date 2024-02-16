#!/usr/local/bin/python39
from scipy.io import arff
import pandas as pd
import numpy as np
from scipy.stats import qmc
import pickle
import random


class ReadBanana(object):
    def __init__(self):
        # Settings
        self.data_dir = './banana/banana.arff'
        data = arff.loadarff(self.data_dir)
        self.df = pd.DataFrame(data[0])
        
    def train_test_set(self):
        X = np.array([self.df['V1'], self.df['V2']])
        y = np.array([int(x)-1 for x in self.df['Class']])

        train_f = 2
        test_f = 5
        ver_f = 8

        X_train = np.array([X[0][::train_f], X[1][::train_f]]).T
        X_test = np.array([X[0][::test_f], X[1][::test_f]]).T
        X_ver = np.array([X[0][::ver_f], X[1][::ver_f]]).T

        Y_train = y[::train_f]
        Y_test = y[::test_f]
        Y_ver = y[::ver_f]

        return X_train, Y_train, X_test, Y_test, X_ver, Y_ver
        
    def dinno_fpoints(self, X_test, Y_test, nf):
        x_min, x_max = min(X_test[:,0]), max(X_test[:,0])
        y_min, y_max = min(X_test[:,1]), max(X_test[:,1])
        
        # First value is 1 for bias term
        lscale = 0.3*np.ones(nf+1)
        sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
        
        fpoints = sampler.random(n=nf)
        fpoints[:,0] = x_min+(x_max-x_min)*fpoints[:,0]
        fpoints[:,1] = y_min+(y_max-y_min)*fpoints[:,1]
        
        return fpoints, lscale

if __name__ == '__main__':
    
    bdata = ReadBanana()
    X_train, Y_train, X_test, Y_test, X_ver, Y_ver = bdata.train_test_set()
    nf = 100
    fpoints, lscale = bdata.dinno_fpoints(X_test, Y_test, nf=nf)
    
    dataset = {'data':[X_train, Y_train, X_test, Y_test, X_ver, Y_ver],
    'param':[fpoints, lscale]}
    dbfile = open('banana_f'+str(nf)+'.pcl', 'wb')
    # Its important to use binary mode
    # source, destination
    pickle.dump(dataset, dbfile)                     
    dbfile.close()