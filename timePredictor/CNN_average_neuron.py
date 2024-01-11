from scipy.io import loadmat
from scipy.stats import zscore
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pickle, sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

class CNN(nn.Module):
    def __init__(self, time_len, num_seq, num_class):
        super(CNN, self).__init__()
        self.time_len = time_len
        self.num_seq = num_seq
        self.num_class = num_class
        
        self.conv = nn.Sequential(*[
            nn.Conv1d(self.num_seq, 32, kernel_size=7, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5),
            nn.Conv1d(32, 32, kernel_size=3, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5),
            nn.Flatten(start_dim=1, end_dim=-1),
        ])
        with torch.no_grad():
            x = torch.zeros(1, self.num_seq, self.time_len)
            x = self.conv(x)
            self.fc_dim = x.reshape(1, -1).shape[-1]
        self.fc = nn.Sequential(*[
            nn.Linear(self.fc_dim, self.num_class)
        ])
    
    def forward(self, x):
        return self.fc(self.conv(x))
    
class TrainDataset(Data.Dataset):
    def __init__(self, x, y, average_num):
        super(TrainDataset, self).__init__()
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        self.x = x[idx]
        self.y = y[idx]
        self.average_num = average_num
        self.location = [
                np.argwhere(self.y == label)[..., 0] for label in range(self.y.max()+1)
        ]
    def __getitem__(self, index):
        label = self.y[index]
        if self.average_num >= self.x.shape[0]:
            self.average_num = self.x.shape[0] -1 
        random_idx = np.random.choice(self.location[label], self.average_num, replace=False)
        return np.mean(self.x[random_idx], axis=0)[None], label

    def __len__(self):
        return self.x.shape[0]

class TestDataset(Data.Dataset):
    def __init__(self, x, y, average_num):
        super(TestDataset, self).__init__()
        self.x = x
        self.y = y
        self.average_num = average_num

    def __getitem__(self, index):
        label = self.y[index]
        if self.average_num >= self.x.shape[0]:
            self.average_num = self.x.shape[0] -1 
        random_idx = np.random.choice(self.x.shape[0], self.average_num, replace=False)
        return np.mean(self.x[random_idx], axis=0), label

    def __len__(self):
        return self.x.shape[0]

filename = sys.argv[1]
data = loadmat(filename, variable_names=['dff_set'])['dff_set']
_data = list()
label = list()
time_class = data.shape[1]
num_neuron = data.shape[0]
for i in range(data.shape[0]): # neuron
    for j in range(data.shape[1]): # time
        _data.append(data[i, j])
        label.append(j)
data = np.concatenate(_data, axis=0).reshape(num_neuron, time_class, -1).astype(np.float32)
label = np.array(label).reshape(num_neuron, time_class).astype(np.int64)
# print(f'data shape: {data.shape}; label shape: {label.shape}')
time_len = data.shape[-1]

seed = int(sys.argv[2])
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 32
train_size = int(data.shape[0] * 0.6)
val_size = int(data.shape[0] * 0.1)
test_size = data.shape[0] - train_size - val_size
# print(train_size, val_size, test_size)

train_loss_list = list()
val_loss_list = list()
test_loss_list = list()
test_acc_list = list()
criterion = nn.CrossEntropyLoss()
cuda = torch.cuda.is_available()
# print(f'cuda available: {cuda}')

average_num = int(sys.argv[3])

#! Kaiwen's random split for train, validation and test sets.
# shuffle_idx = np.arange(num_neuron)
# np.random.shuffle(shuffle_idx)
# data = data[shuffle_idx]
# label = label[shuffle_idx]
# train_dataloader = Data.DataLoader(TrainDataset(data[:train_size].reshape(-1, time_len), label[:train_size].reshape(-1), average_num), batch_size=batch_size, shuffle=True)
# val_dataloader = Data.DataLoader(TrainDataset(data[train_size:train_size+val_size].reshape(-1, time_len), label[train_size:train_size+val_size].reshape(-1), average_num), batch_size=batch_size, shuffle=True)
# test_dataloader = Data.DataLoader(TestDataset(data[train_size+val_size:], label[train_size+val_size:], average_num), batch_size=batch_size, shuffle=True)

#! train, validation and test sets split within a specific class
X_train_val, X_test, y_train_val, y_test = train_test_split(data, label, test_size=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/7, random_state=0)
train_dataloader = Data.DataLoader(TrainDataset(X_train.reshape(-1, time_len), y_train.reshape(-1), average_num), batch_size=batch_size, shuffle=True)
#! val_size: 604, 778, 782 for three datastes.
if average_num > 604 and '0916' in filename:
    val_average_num = 604
elif average_num > 778 and '0918' in filename:
    val_average_num = 778
elif average_num > 782 and '0922' in filename:
    val_average_num = 782
else:
    val_average_num = average_num
val_dataloader = Data.DataLoader(TrainDataset(X_val.reshape(-1, time_len), y_val.reshape(-1), val_average_num), batch_size=batch_size, shuffle=True)
test_dataloader = Data.DataLoader(TestDataset(X_test, y_test, average_num), batch_size=batch_size, shuffle=True)

cnn = CNN(time_len, 1, label.max()+1)
if cuda:
    cnn.cuda()
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)

save_filename = os.path.join('training_log', f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{seed}.pkl")
try:
    with open(save_filename, 'rb') as f:
        res = pickle.load(f)
    train_loss_list = res['train_loss_list']
    val_loss_list = res['val_loss_list']
    test_loss_list = res['test_loss_list']
    test_acc_list = res['test_acc_list']
    st = len(train_loss_list[-1])
    cnn.load_state_dict(res['model'])
    optimizer.load_state_dict(res['optimizer'])
except FileNotFoundError:
    train_loss_list.append(list())
    val_loss_list.append(list())
    test_loss_list.append(list())
    test_acc_list.append(list())
    st = 0
no_improve = 10
last_val_loss = 1e8
for epoch in range(st, 100):  # loop over the dataset multiple times

    cnn.train()
    train_loss = 0.0
    for i, (x, y) in enumerate(train_dataloader):

        if cuda:
            x = x.cuda()
            y = y.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = cnn(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item() * x.shape[0]

    train_loss /= train_size * 24
    cnn.eval()
    val_loss = 0.
    with torch.no_grad():
        for x, y in val_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            outputs = cnn(x)
            val_loss += criterion(outputs, y).item() * x.shape[0]
    val_loss /= val_size * 24

    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for x, y in test_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            x = x.reshape(-1, 1, time_len)
            y = y.reshape(-1)
            outputs = cnn(x)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            test_acc += (y.cpu().numpy() == pred).sum()
            test_loss += criterion(outputs, y).item() * x.shape[0]
    test_loss /= test_size * 24
    test_acc /= test_size * 24
    
    if epoch % 20 == 0:
        print(f'epoch {epoch} train_loss: {np.round(train_loss, 5)}; val_loss: {np.round(val_loss, 5)}; test_loss: {np.round(test_loss, 5)}; test_acc: {np.round(test_acc, 3)}')
    train_loss_list[-1].append(train_loss)
    val_loss_list[-1].append(val_loss)
    test_loss_list[-1].append(test_loss)
    test_acc_list[-1].append(test_acc)

    # earlystopping
    # if val_loss > last_val_loss:
    #     no_improve += 1
    #     if no_improve >= 10:
    #         print(f'Early stop at epoch {epoch}')
    #         break
    # else:
    #     no_improve = 0

    pickle.dump({
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'test_loss_list': test_loss_list,
        'test_acc_list': test_acc_list,
        'model': cnn.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, open(save_filename, 'wb'))
