from torch.utils.data import Dataset
import torch
import os
import numpy as np
import scipy.io as scio 

class NeuronData(Dataset):
    def __init__(self, path, average=1, class_num=24):
        super(NeuronData, self).__init__()
        
        data = scio.loadmat(path, variable_names=['dff_set'])['dff_set']
        _data = list()
        label = list()
        time_class = data.shape[1]
        num_neuron = data.shape[0]
        for i in range(data.shape[0]): # neuron
            for j in range(data.shape[1]): # time
                _data.append(data[i, j])
                label.append(j)
        self.data = np.concatenate(_data, axis=0).reshape(num_neuron, time_class, -1).astype(np.float32)
        self.data = np.mean(self.data, axis=-1)
        self.label = np.array(label).reshape(num_neuron, time_class).astype(np.int64)[0, :]  # 24
        
    
    def __len__(self):
        return 24

    def __getitem__(self, idx):
        data = self.data[:,idx]  # n * 200
        label = self.label[idx].reshape(-1)
        
        trace_data = torch.from_numpy(data).float()
        return trace_data, label
