import numpy as np
import scipy.io as scio
import torch

date = '20210916'

# load data
def load_SCN(scn_data_path, task):

    scn_data = scio.loadmat(scn_data_path)
    trace = scn_data['trace'].T # trace
    poi = torch.FloatTensor(scn_data['POI'])

    if task == 'standard' or task == 'pc-sample':
        trace = trace[:,0:4800]
        train = np.reshape(trace, (trace.shape[0], 24, 200))
    elif task == 'time-sample':
        trace = trace[:,0:2400]
        train = np.reshape(trace, (trace.shape[0], 24, 100))
    elif task == '1_3-sample':
        trace = trace[:,0:1600]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
    elif task == '2_3-sample':
        trace = trace[:,1600:3200]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
    elif task == '3_3-sample':
        trace = trace[:,3200:4800]
        train = np.reshape(trace, (trace.shape[0], 8, 200))
        
    return train, poi