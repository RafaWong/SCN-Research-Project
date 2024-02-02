
import numpy as np
import pickle, sys, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.lr_updater import CosineLrUpdater
from src.model import CNN

from src.dataset import NeuronData
from scipy.io import loadmat

def accuracy(output, target, topk=(1)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

filename = sys.argv[1]
data = loadmat(filename, variable_names=['dff_set'])['dff_set']


seed = int(sys.argv[2])
np.random.seed(seed)
torch.manual_seed(seed)

batch_size = 1

train_acc_list = list()
train_loss_list = list()

test_loss_list = list()
test_acc_list = list()
criterion = nn.CrossEntropyLoss()
cuda = torch.cuda.is_available()


num_neuron = int(sys.argv[3])

train_dataloader = DataLoader(NeuronData(filename, 24), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(NeuronData(filename, 24), batch_size=batch_size, shuffle=True)

cnn = CNN(num_class=24, num_seq=num_neuron, base_channel=16)
        
if cuda:
    cnn.cuda()

optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
scheduler = CosineLrUpdater(optimizer, 
                                periods=[500],
                                by_epoch=True,
                                warmup='linear',
                                warmup_iters=1,
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
    test_loss_list = res['test_loss_list']
    test_acc_list = res['test_acc_list']
    st = len(train_loss_list[-1])
    cnn.load_state_dict(res['model'])
    optimizer.load_state_dict(res['optimizer'])
    start_iter=res['step']
except FileNotFoundError:
    train_acc_list.append(list())
    train_loss_list.append(list())
    test_loss_list.append(list())
    test_acc_list.append(list())
    st = 0

epoch = 500
current_iters=start_iter
scheduler.before_run()
for current_epoch in range(st, epoch):  
    scheduler.before_train_epoch(24, current_epoch, current_iters)        
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
        loss = criterion(outputs, y.squeeze(-1).to(torch.int64))
        

        loss.backward()
        optimizer.step()

        current_iters+=1
        train_loss += loss.item()
        train_acc += accuracy(outputs, y.squeeze(-1), topk=(1, 2))[0].cpu().numpy()
    train_loss /= 24
    train_acc /= 24


    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for x, y in test_dataloader:
            if cuda:
                x = x.cuda()
                y = y.cuda()
            outputs = cnn(x)
            loss = criterion(outputs, y.squeeze(-1).to(torch.int64))
            
            current_iters+=1
            test_loss += loss.item()
            test_acc += accuracy(outputs, y.squeeze(-1), topk=(1, 2))[0].cpu().numpy()
    test_loss /= 24
    test_acc /= 24
    if current_epoch % 20 == 0:
        print(f'epoch {current_epoch} train_loss: {np.round(train_loss, 5)}; train_acc: {np.round(train_acc, 5)}; test_loss: {np.round(test_loss, 5)}; test_acc: {np.round(test_acc, 3)}')
    train_acc_list[-1].append(train_acc)
    train_loss_list[-1].append(train_loss)
    test_loss_list[-1].append(test_loss)
    test_acc_list[-1].append(test_acc)


    pickle.dump({
        'train_loss_list': train_loss_list,
        'train_acc_list': train_acc_list,
        'test_loss_list': test_loss_list,
        'test_acc_list': test_acc_list,
        'model': cnn.state_dict(),
        'optimizer': optimizer.state_dict(),
        "step": current_iters,
    }, open(save_filename, 'wb'))
