import pickle, os
import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
import numpy as np
from src.model import CNN
from src.dataset import TestDataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# we show the example of general time predictor test on the submodule in the full testset of 'Dataset1_SCNProject'
filename = '../SCNData/Dataset1_SCNProject.mat'
full_data_for_analysis={}
subname = os.path.basename(filename).split('_SCNProject')[0]

# load data 
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
batch_size = 1
train_size = int(data.shape[0] * 0.6)
val_size = int(data.shape[0] * 0.1)
test_size = data.shape[0] - train_size - val_size

# we have 5 submodules in the full testset.
num_submodule = 5
num_neuron_list = [1, 10, 30, 50, 100, 143-1, 238-1, 300, 400-1, 448-1, 500, 586-1]

#! Note: for sub-module test, make sure that you have got the spatial label as '5-class.mat' before this test.    
index_for_spatial_label = (loadmat(f'../SCNData/{num_submodule}-class.mat',variable_names=['index'])['index']).squeeze(-1)
indices = np.arange(data.shape[0]) 
indices_train, indices_test, _, _ = train_test_split(indices, data, test_size=0.3, random_state=74)

X_test_all = data[indices_test, ...]
y_test_all = label[indices_test, ...]
spatial_label = index_for_spatial_label[indices_test] - 1 #! from 1-5 to 0-4

for spatial_class_class in range(0, num_submodule):
    spatial_class_index = list(np.where(spatial_label==spatial_class_class)[0])
    print(f'In processing Spatial class spatial_class {spatial_class_class+1} withn mumber {len(spatial_class_index)}')
    sub_X_test_all = X_test_all[spatial_class_index, ...]
    sub_y_test_all = y_test_all[spatial_class_index, ...]        
    print(f'spatail class {spatial_class_class+1} has',sub_X_test_all.shape[0])
    full_data_for_analysis[f'spatail class {spatial_class_class+1}']=[]

    for i, num_neuron in enumerate(num_neuron_list):
        model = CNN(200, num_neuron, 24).cuda()
        all_test_acc = []
        for seed in range(5):
            if len(spatial_class_index) > num_neuron:
                torch.manual_seed(seed)
                weight_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_num_neuron{num_neuron}_seed{seed}.pkl"
                with open(os.path.join('./training_log', weight_filename), 'rb') as f:
                    res = pickle.load(f)
                # set the model to evaluation mode
                model.load_state_dict(res['model'])
                model.eval()

                for np_seed in range(1000): #! retest 1000 times for each training_seed.
                    np.random.seed(np_seed)
                    test_dataloader = DataLoader(TestDataset(sub_X_test_all, sub_y_test_all, num_neuron), batch_size=batch_size, shuffle=False)
                    with torch.no_grad():
                        test_acc=0.0
                        for x, y in test_dataloader:                    
                            x = x.cuda()
                            y = y.cuda()
                            x = x.permute(0,2,1,3).reshape(-1, num_neuron, time_len)
        
                            y = y.reshape(-1)
                            outputs = model(x)
                            pred = torch.argmax(outputs, dim=1).cpu().numpy()
                            test_acc += (y.cpu().numpy() == pred).sum()  
                        test_acc /= test_size * 24
                        all_test_acc.append(test_acc)
                        print(f'spatail class {spatial_class_class+1}, num_neuron: {num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')
        if len(spatial_class_index) >=num_neuron:
            all_test_acc=np.array(all_test_acc)
            full_data_for_analysis[f'spatail class {spatial_class_class+1}'].append([np.mean(all_test_acc), np.std(all_test_acc)])

# plot the above data.
all_neuron = np.array(full_data_for_analysis['all_neuron_acc'])
spatial_class1 = np.array(full_data_for_analysis['spatail class 1'])
spatial_class2 = np.array(full_data_for_analysis['spatail class 2'])
spatial_class3 = np.array(full_data_for_analysis['spatail class 3']) 
spatial_class4 = np.array(full_data_for_analysis['spatail class 4'])
spatial_class5 = np.array(full_data_for_analysis['spatail class 5'])

fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)
axs.plot(np.arange(all_neuron.shape[0]), all_neuron[:,0], marker='o', label=f'All Neuron')
axs.fill_between(np.arange(all_neuron.shape[0]), all_neuron[:,0] - all_neuron[:,1], all_neuron[:,0] + all_neuron[:,1], alpha=0.2)
axs.plot(np.arange(spatial_class1.shape[0]), spatial_class1[:,0], marker='o',label=f'Spatial class 1')
axs.fill_between(np.arange(spatial_class1.shape[0]), spatial_class1[:,0] - spatial_class1[:,1], spatial_class1[:,0] + spatial_class1[:,1], alpha=0.2)
axs.plot(np.arange(spatial_class2.shape[0]), spatial_class2[:,0], marker='o',label=f'Spatial class 2')
axs.fill_between(np.arange(spatial_class2.shape[0]), spatial_class2[:,0] - spatial_class2[:,1], spatial_class2[:,0] + spatial_class2[:,1], alpha=0.2)
axs.plot(np.arange(spatial_class3.shape[0]), spatial_class3[:,0], marker='o',label=f'Spatial class 3')
axs.fill_between(np.arange(spatial_class3.shape[0]), spatial_class3[:,0] - spatial_class3[:,1], spatial_class3[:,0] + spatial_class3[:,1], alpha=0.2)
axs.plot(np.arange(spatial_class4.shape[0]), spatial_class4[:,0], marker='o',label=f'Spatial class 4')
axs.fill_between(np.arange(spatial_class4.shape[0]), spatial_class4[:,0] - spatial_class4[:,1], spatial_class4[:,0] + spatial_class4[:,1], alpha=0.2)
axs.plot(np.arange(spatial_class5.shape[0]), spatial_class5[:,0], marker='o',label=f'Spatial class 5')
axs.fill_between(np.arange(spatial_class5.shape[0]), spatial_class5[:,0] - spatial_class5[:,1], spatial_class5[:,0] + spatial_class5[:,1], alpha=0.2)
axs.set_xlabel('Avg num', fontname='Arial', fontsize=18, fontweight='bold')
axs.set_ylabel('Accuracy', fontname='Arial', fontsize=18, fontweight='bold')

plt.tight_layout()
plt.yticks(fontproperties = 'Arial', size = 12, fontweight='bold')
plt.xticks(fontproperties = 'Arial', size = 12, fontweight='bold')
axs.set_xticks(np.arange(len(num_neuron_list)))
axs.set_xticklabels(num_neuron_list)
threshold1 = 1.0
axs.axhline(threshold1, color='grey', lw=1, alpha=0.7, linestyle ='--')
plt.gca().set_yticklabels([f'{x :.1%}' for x in plt.gca().get_yticks()]) 

axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.legend(loc='upper left', bbox_to_anchor=(0.775, 0.985), prop = {'size':10})  
os.makedirs('./submodules_test_results', exist_ok=True)
plt.savefig(f"./submodules_test_results/{num_submodule}_acc.png", dpi=400, bbox_inches='tight')
axs.cla()
