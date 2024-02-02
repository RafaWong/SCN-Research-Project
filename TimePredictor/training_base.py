
import numpy as np
import pickle, sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.lr_updater import CosineLrUpdater
from src.model import CNN

from src.dataset import TrainDataset, TestDataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

filename = sys.argv[1]
data = loadmat(filename, variable_names=['dff_set'])['dff_set']
_data = list()
label = list()
time_class = data.shape[1]
num_neuron = data.shape[0]
for i in range(data.shape[0]): 
    for j in range(data.shape[1]):
        _data.append(data[i, j])
        label.append(j)
data = np.concatenate(_data, axis=0).reshape(num_neuron, time_class, -1).astype(np.float32)
label = np.array(label).reshape(num_neuron, time_class).astype(np.int64)
time_len = data.shape[-1]

seed = int(sys.argv[2])
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 32
train_size = int(data.shape[0] * 0.6)
val_size = int(data.shape[0] * 0.1)
test_size = data.shape[0] - train_size - val_size
train_acc_list = list()
train_loss_list = list()
val_loss_list = list()
test_loss_list = list()
test_acc_list = list()
criterion = nn.CrossEntropyLoss()
cuda = torch.cuda.is_available()


num_neuron = int(sys.argv[3])


X_train_val, X_test, y_train_val, y_test = train_test_split(data, label, test_size=0.3, random_state=74)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/7, random_state=74)
train_dataloader = DataLoader(TrainDataset(X_train.reshape(-1, time_len), y_train.reshape(-1), num_neuron), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(TrainDataset(X_val.reshape(-1, time_len), y_val.reshape(-1), num_neuron), batch_size=batch_size, num_workers=4, shuffle=True)
test_dataloader = DataLoader(TestDataset(X_test, y_test, num_neuron), batch_size=batch_size, shuffle=True)

cnn = CNN(time_len, num_neuron, label.max()+1)
if cuda:
    cnn.cuda()

optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
scheduler = CosineLrUpdater(optimizer, 
                                periods=[100],
                                by_epoch=True,
                                warmup='linear',
                                warmup_iters=200,
                                warmup_ratio=1.0 / 3,
                                warmup_by_epoch=False,
                                restart_weights=[1],
                                min_lr=[1e-7],
                                min_lr_ratio=None)

os.makedirs('./training_log', exist_ok=True)
save_filename = os.path.join('training_log', f"{os.path.basename(filename).split('_SCNProject')[0]}_log_num_neuron{num_neuron}_seed{seed}.pkl")
start_iter = 0
try:
    with open(save_filename, 'rb') as f:
        res = pickle.load(f)
    train_acc_list = res['train_acc_list']
    train_loss_list = res['train_loss_list']
    val_loss_list = res['val_loss_list']
    test_loss_list = res['test_loss_list']
    test_acc_list = res['test_acc_list']
    st = len(train_loss_list[-1])
    cnn.load_state_dict(res['model'])
    optimizer.load_state_dict(res['optimizer'])
    start_iter=res['step']
except FileNotFoundError:
    train_acc_list.append(list())
    train_loss_list.append(list())
    val_loss_list.append(list())
    test_loss_list.append(list())
    test_acc_list.append(list())
    st = 0

no_improve = 10
last_val_loss = 1e8
epoch = 100
current_iters=start_iter
scheduler.before_run()
for current_epoch in range(st, epoch):  
    scheduler.before_train_epoch(train_size, current_epoch, current_iters)        
    cnn.train()
    train_loss = 0.0
    train_acc = 0.0
    for i, (x, y) in enumerate(train_dataloader):
        scheduler.before_train_iter(current_epoch, current_iters)

        if cuda:
            x = x.cuda()
            y = y.cuda()
        optimizer.zero_grad()
        outputs = cnn(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        current_iters+=1
        train_loss += loss.item() * x.shape[0]
        pred = torch.argmax(outputs, dim=1).cpu().numpy()
        train_acc += (y.cpu().numpy() == pred).sum()
    train_loss /= train_size * 24
    train_acc /= train_size * 24
    if num_neuron < val_size:
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
    else:
        val_loss = 0.

    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for x, y in test_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            x = x.permute(0,2,1,3).reshape(-1, num_neuron, time_len)
            
            y = y.reshape(-1)
            outputs = cnn(x)
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            test_acc += (y.cpu().numpy() == pred).sum()
            test_loss += criterion(outputs, y).item() * x.shape[0]
    test_loss /= test_size * 24
    test_acc /= test_size * 24
    
    if current_epoch % 20 == 0:
        if num_neuron < val_size:
            print(f'epoch {current_epoch} train_loss: {np.round(train_loss, 5)}; train_acc: {np.round(train_acc, 5)}; val_loss: {np.round(val_loss, 5)}; test_loss: {np.round(test_loss, 5)}; test_acc: {np.round(test_acc, 3)}')
        else:
            print(f'epoch {current_epoch} train_loss: {np.round(train_loss, 5)}; train_acc: {np.round(train_acc, 5)}; test_loss: {np.round(test_loss, 5)}; test_acc: {np.round(test_acc, 3)}')
    train_acc_list[-1].append(train_acc)
    train_loss_list[-1].append(train_loss)
    val_loss_list[-1].append(val_loss)
    test_loss_list[-1].append(test_loss)
    test_acc_list[-1].append(test_acc)

    # earlystopping
    if num_neuron < val_size:
        if val_loss > last_val_loss:
            no_improve += 1
            if no_improve >= 10:
                print(f'Early stop at epoch {epoch}')
                break
        else:
            no_improve = 0

    pickle.dump({
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,
        'val_loss_list': val_loss_list,
        'test_loss_list': test_loss_list,
        'test_acc_list': test_acc_list,
        'model': cnn.state_dict(),
        'optimizer': optimizer.state_dict(),
        "step": current_iters,
    }, open(save_filename, 'wb'))
