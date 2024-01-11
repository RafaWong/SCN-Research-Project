from cProfile import label
import os
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import pickle
from utils import pkl_load, pad_nan_to_target
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import scipy.io as scio
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric.nn as pyg_nn

date = '0726'

#### load data
def load_SCN():
    # scn_data_path = f'D:/lab/scn/3d/20220516/20210918/INTERP/0918_6-29.mat'
    # scn_data_path = f'D:/lab/scn/Data_0712/20210602_CT02-11/20210602_data.mat'
    # scn_data_path = f'D:/lab/scn/3d/20220516/20210916/INTERP/0916_int_7-30.mat' # 0916_int_7-30
    # scn_data_path = f'D:/lab/scn/3d/spike_int.mat' # 0916
    # scn_data_path = f'D:/lab/scn/3d/20221010/spike_int.mat'
    # scn_data_path = f'D:/lab/scn/3d/2022{date_name}/2022{date_name}_data.mat'
    # scn_data_path = f'D:/lab/scn/3d/20220516/2021{date}/INTERP/2021{date}_baseData.mat' # baseline
    # scn_data_path = f'D:/lab/scn/3d/data_0509/2021{date}_non/{date}_hand_not.mat' # non_scn (12 hours)
    scn_data_path = f'E:/scn/2022{date}/2022{date}_data.mat'

    scn_data = scio.loadmat(scn_data_path)

    trace = scn_data['trace'].T # trace

    trace = trace[:,0:4800] # standard 4800, time sample 2400, non scn 2400

    ## TODO normalize
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler().fit(trace)
    # trace = scaler.transform(trace)

    train = np.reshape(trace, (trace.shape[0], 24, 200)) # standard 24 200, sample 24 100, non scn 12 200

    # ## TODO shuffle periods for all neuron
    # t_ind = np.random.permutation(24)
    # train = train[:, t_ind,:]
    # way 2 shuffle for each neuron
    # for i in range(trace.shape[0]):
    #     t_ind = np.random.permutation(24)
    #     train[i] = train[i, t_ind,:]
    # # scio.savemat(f'./TC-results/exp-0610/shuffle_periods.mat', {'dff':train})
    # ##

    # ## TODO shuffle time points
    # for i in range(trace.shape[0]):
    #     for t in range(24):
    #         s_ind = np.random.permutation(200)
    #         train[i,t,:] = train[i, t, s_ind]
    # # scio.savemat(f'./TC-results/exp-0610/shuffle_time.mat', {'dff':train})
    # #

    ## TODO: change to gaussian noise
    # for i in range(train.shape[0]):
    #     for t in range(24):
    #         sample = train[i,t,:]
    #         miu = np.mean(sample)
    #         sigma = np.std(sample)
    #         # random_noise = np.random.normal(miu,sigma,sample.shape)
    #         random_noise = np.random.normal(0,sigma,sample.shape)
    #         train[i,t,:] = random_noise
    # scio.savemat(f'./TC-results/exp-0610/gaussian_time.mat', {'dff':train})
    ##

    ## TODO: change 24h to gaussian noise
    # for i in range(trace.shape[0]):
    #     sample = trace[i,:]
    #     miu = np.mean(sample)
    #     sigma = np.std(sample)
    #     random_noise = np.random.normal(miu,sigma,sample.shape)
    #     trace[i,:] = random_noise
    # train = np.reshape(trace, (trace.shape[0], 24, 200))
    # scio.savemat(f'./TC-results/exp-0610/gaussian_period.mat', {'dff':train})
    ##

    ## TODO: all to noise
    # miu = np.mean(train)
    # sigma = np.std(train)
    # random_noise = np.random.normal(miu,sigma,train.shape)
    # train = random_noise
    # scio.savemat(f'./TC-results/exp-0610/gaussian_all.mat', {'dff':train})
    ##

    # train = scn_data['s_i'] # trace, s_o

    # TODO pca
    # from sklearn.decomposition import PCA
    # tmp1 = np.reshape(train, (trace.shape[0]*24, 200))
    # n_components = 2
    # pca = PCA(n_components)
    # tmp2 = pca.fit_transform(tmp1)
    # train = np.reshape(tmp2, (trace.shape[0], 24, n_components))

    return train