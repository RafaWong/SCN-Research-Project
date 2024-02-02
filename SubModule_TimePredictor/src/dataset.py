from torch.utils.data import Dataset
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, x, y, num_neuron):
        super(TrainDataset, self).__init__()
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        self.x = x[idx]
        self.y = y[idx]
        self.num_neuron = num_neuron
        self.location = [
                np.argwhere(self.y == label)[..., 0] for label in range(self.y.max()+1)
        ]
    def __getitem__(self, index):
        label = self.y[index]
        if self.num_neuron >= self.x.shape[0]:
            self.num_neuron = self.x.shape[0] -1 
        random_idx = np.random.choice(self.location[label], self.num_neuron, replace=False)
        return self.x[random_idx], label
    
    def __len__(self):
        return self.x.shape[0]

class TestDataset(Dataset):
    def __init__(self, x, y, num_neuron):
        super(TestDataset, self).__init__()
        self.x = x
        self.y = y
        self.num_neuron = num_neuron

    def __getitem__(self, index):
        label = self.y[index]
        if self.num_neuron >= self.x.shape[0]:
            self.num_neuron = self.x.shape[0] -1 
        random_idx = np.random.choice(self.x.shape[0], self.num_neuron, replace=False)
        return self.x[random_idx], label

    def __len__(self):
        return self.x.shape[0]
