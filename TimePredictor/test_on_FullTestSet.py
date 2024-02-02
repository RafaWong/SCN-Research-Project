
import pickle, os
import numpy as np
from matplotlib import ticker
from sklearn.model_selection import train_test_split
import scipy.io as scio
import torch
from src.model import CNN
from src.dataset import TestDataset
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

filename_list = [
        '../SCNData/Dataset1_SCNProject.mat',
        '../SCNData/Dataset2_SCNProject.mat',
        '../SCNData/Dataset3_SCNProject.mat',
        '../SCNData/Dataset4_SCNProject.mat',
        '../SCNData/Dataset5_SCNProject.mat',
        '../SCNData/Dataset6_SCNProject.mat',
        ]   


#! this part is for visualizing the 'test-set' accuracy with different num_neuron number.
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)

for filename in filename_list:
    
    num_neuron_list = [1, 10, 30, 50, 100, 300, 500, 600,  700,  750,  800,  850,  900,  950, 1000, 1500]
    
    all_s=[]

    data = scio.loadmat(filename, variable_names=['dff_set'])['dff_set']
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
    time_len = data.shape[-1]
    train_size = int(data.shape[0] * 0.6)
    val_size = int(data.shape[0] * 0.1)
    test_size = data.shape[0] - train_size - val_size
    _, X_test_all, _, y_test_all = train_test_split(data, label, test_size=0.3, random_state=74)
        
    all_mean=[]
    all_std=[]
    for i, num_neuron_num in enumerate(num_neuron_list):
        test_acc_list = list() 
        model = CNN(time_len, num_neuron_num, label.max()+1).cuda()
        for seed in range(5):
            torch.manual_seed(seed)
            save_filename = f"{os.path.basename(filename).split('_SCNProject')[0]}_log_num_neuron{num_neuron_num}_seed{seed}.pkl"
            with open(os.path.join('./training_log', save_filename), 'rb') as f:
                res = pickle.load(f)
            model.load_state_dict(res['model'])
            model.eval()
            for np_seed in range(1000): #! retest 100 times for each training_seed.
                np.random.seed(np_seed)
                test_dataloader = DataLoader(TestDataset(X_test_all, y_test_all, num_neuron_num), batch_size=1, shuffle=False)
                with torch.no_grad():
                    test_acc=0.0
                    for x, y in test_dataloader:                    
                        x = x.cuda()
                        y = y.cuda()
                        # x = x.reshape(-1, num_neuron_num, time_len)
                        x = x.permute(0,2,1,3).reshape(-1, num_neuron_num, time_len)
                        y = y.reshape(-1)
                        outputs = model(x)
                        pred = torch.argmax(outputs, dim=1).cpu().numpy()
                        test_acc += (y.cpu().numpy() == pred).sum()  
                    test_acc /= test_size * 24
                    test_acc_list.append(test_acc)
                    print('ACC', test_acc)

        test_acc_list = np.array(test_acc_list)

        mean = np.mean(test_acc_list)
        std = np.std(test_acc_list)
        all_mean.append(mean)
        all_std.append(std)
        
    plt.plot(list(np.arange(len(num_neuron_list))), all_mean, 'o-')
    axs.fill_between(list(np.arange(len(num_neuron_list))), np.array(all_mean) - np.array(all_std), np.array(all_mean) + np.array(all_std), alpha=0.2)
    axs.set_xticks(np.arange(len(num_neuron_list)))
    axs.set_xticklabels(num_neuron_list)
    axs.set_xlabel('num_neuron traces', fontname='Arial', fontsize=18, fontweight='bold')
    axs.set_ylabel('Accuracy', fontname='Arial', fontsize=18, fontweight='bold')
    threshold1 = 1.0
    axs.axhline(threshold1, color='grey', lw=1, alpha=0.7, linestyle ='--')
    threshold2 = 1/24
    axs.axhline(threshold2, color='grey', lw=1, alpha=0.7, linestyle ='--')
    threshold3 = 0.99
    axs.axhline(threshold3, color='grey', lw=1, alpha=0.7, linestyle ='--')
    axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.985), prop = {'size':9})
    plt.tight_layout()
    plt.yticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.xticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    os.makedirs('./test_results', exist_ok=True)
    plt.savefig(f"./test_results/{save_filename.split('_seed')[0]}_res.png", dpi=400, bbox_inches='tight')
    axs.cla()
