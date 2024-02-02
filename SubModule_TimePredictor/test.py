import matplotlib
matplotlib.use('Agg')
import pickle, os
import numpy as np
import scipy.io as scio
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import ticker

import torch
import torch.utils.data as Data

import scipy.io as scio
from src.model import CNN
from src.dataset import TestDataset

# we show the example of general time predictor test on the submodule in the full testset of 'Dataset1_SCNProject'
filename = '../SCNData/Dataset1_SCNProject.mat'
trial_per_seed=1000
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
time_len = data.shape[-1]

#! Note: for sub-module test, make sure that you have got the spatial label as '3-class.mat' before this test. 
index_for_spatial_label = (loadmat(f'../SCNData/3-class.mat',variable_names=['index'])['index']).squeeze(-1)

# spatial class 1:
spatial_class = 1
spatial_class_index = list(np.where(index_for_spatial_label==spatial_class)[0])
X = data[spatial_class_index, ...]
y = label[spatial_class_index, ...]
X_train, X_test3, y_train, y_test3 = train_test_split(X, y, test_size=0.3, random_state=0)
train_size = y_train.shape[0]
test_size3 = y_test3.shape[0]

train_size=len(spatial_class_index)

spatial_class_index1 = list(np.where(index_for_spatial_label==2)[0])
X_test1 = data[spatial_class_index1, ...]
y_test1 = label[spatial_class_index1, ...]
test_size1 = len(spatial_class_index1)

spatial_class_index2 = list(np.where(index_for_spatial_label==3)[0])
X_test2 = data[spatial_class_index2, ...]
y_test2 = label[spatial_class_index2, ...]
test_size2 = len(spatial_class_index2)

num_list = [1, 10, 30, 50, 100, 300, 500, 860-1, 1000, 1167-1, 1719-1]

previous_test_1 = 0
previous_test_2 = 0
previous_test_3 = 0
num_final_acc_mean_std = {}
for i, num_neuron in enumerate(num_list):
    model = CNN(time_len, num_neuron, 24).cuda()
    test_acc1_list = [] 
    test_acc2_list = []       
    test_acc3_list = []
    training_num = train_size - 1 if num_neuron >= train_size else num_neuron
    test_num1 = test_size1 - 1 if num_neuron >= test_size1 else num_neuron
    test_num2 = test_size2 - 1 if num_neuron >= test_size2 else num_neuron
    test_num3 = test_size3-1 if num_neuron >= test_size3 else num_neuron
    for seed in range(5):
        torch.manual_seed(seed)
        if num_neuron <= 1719-1  :
            save_filename = f"train_spatial{spatial_class}_log_train_num{training_num}_test1_num{test_num1}_test2_num{test_num2}_test3_num{test_num3}_seed{seed}.pkl"
            with open(os.path.join('./training_log_1for3_100e', save_filename), 'rb') as f:
                res = pickle.load(f)
            
            model.load_state_dict(res['model'])
            model.eval()

            if test_num1 == previous_test_1:
                pass
            else:
                for np_seed in range(trial_per_seed): 
                    np.random.seed(np_seed)
                    test_dataloader = Data.DataLoader(TestDataset(X_test1, y_test1, test_num1), batch_size=1, shuffle=False)
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
                        test_acc /= test_size1 * 24
                        test_acc1_list.append(test_acc)
                        print(f'spatial 1, num:{num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')

            if test_num2 == previous_test_2:
                pass
            else:
                for np_seed in range(trial_per_seed): 
                    np.random.seed(np_seed)
                    test_dataloader = Data.DataLoader(TestDataset(X_test2, y_test2, test_num2), batch_size=1, shuffle=False)
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
                        test_acc /= test_size2 * 24
                        test_acc2_list.append(test_acc)
                        print(f'spatial 2, num:{num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')
            if test_num3 == previous_test_3:
                pass
            else:
                for np_seed in range(trial_per_seed): 
                    np.random.seed(np_seed)
                    test_dataloader = Data.DataLoader(TestDataset(X_test3, y_test3, test_num3), batch_size=1, shuffle=False)
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
                        test_acc /= test_size3 * 24
                        test_acc3_list.append(test_acc)
                        print(f'spatial 3, num:{num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')
        elif num_neuron > 1719-1  :
            pass 

    if test_num1 == previous_test_1:
        pass
    else:
        num_final_acc_mean_std[str(num_neuron)+'_spatial2'] = test_acc1_list
    if test_num2 == previous_test_2:
        pass
    else:
        num_final_acc_mean_std[str(num_neuron)+'_spatial3'] = test_acc2_list
    if test_num3 == previous_test_3:
        pass
    else:
        num_final_acc_mean_std[str(num_neuron)+'_spatial1'] = test_acc3_list
    previous_test_1 = test_num1
    previous_test_2 = test_num2
    previous_test_3 = test_num3
num_final_acc_mean_std['test_spatial2_numlist'] = [1, 10, 30, 50, 100, 300, 500, 1000, 1719-1]
num_final_acc_mean_std['test_spatial3_numlist'] = [1, 10, 30, 50, 100, 300, 500, 1000, 1167-1]
num_final_acc_mean_std['test_spatial1_numlist'] = [1, 10, 30, 50, 100, 300, 500, 860-1]

data_test2 = []
data_test3 = []
data_test1 = []

num_list2 = [1, 10, 30, 50, 100, 300, 500, 1000, 1719-1]
for num in num_list2: 
    key = str(num)+'_spatial2'
    data_test2.append(num_final_acc_mean_std[key])

num_list3 = [1, 10, 30, 50, 100, 300, 500, 1000, 1167-1]
for num in num_list3:
    key = str(num)+'_spatial3'
    data_test3.append(num_final_acc_mean_std[key])

num_list_self = [1, 10, 30, 50, 100, 300, 500, 860-1]
for num in num_list_self:
    key = str(num)+'_spatial1'
    data_test1.append(num_final_acc_mean_std[key])
    
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)  

data_test2=np.array(data_test2)
data_test3=np.array(data_test3)
data_test1=np.array(data_test1)
y1 = np.mean(data_test2, axis=1)
err1 = np.std(data_test2, axis=1)
y2 = np.mean(data_test3, axis=1)
err2 = np.std(data_test3, axis=1)
y3 = np.mean(data_test1, axis=1)
err3 = np.std(data_test1, axis=1)

save_mat1={}
save_mat1['test_spatial2_mean']=y1
save_mat1['test_spatial2_std']=err1
save_mat1['test_spatial2_numlist'] = num_list2
save_mat1['test_spatial3_numlist'] = num_list3
save_mat1['test_spatial3_mean']=y2
save_mat1['test_spatial3_std']=err2
save_mat1['test_spatial1_numlist'] = num_list_self
save_mat1['test_spatial1_mean']=y3
save_mat1['test_spatial1_std']=err3
os.makedirs('./test_results', exist_ok=True)
scio.savemat('./test_results/train_1_test123_acc_5000.mat', save_mat1)

save_mat1 = loadmat('./test_results/train_1_test123_acc_5000.mat')

axs[1].plot(list(range(len(num_list2))), y1, 'o-', color = 'tab:green')
axs[1].fill_between(list(range(len(num_list2))), y1-err1, y1+err1, color='tab:green', alpha=0.2)
axs[1].set_xticks(list(range(len(num_list2))), num_list2)
axs[1].set_xlabel('Average Number', fontname='Arial', fontsize=18,fontweight='bold')
axs[1].set_ylabel('Test Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
axs[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
axs[1].set_title('Test Spatial 2',fontname='Arial', fontsize=18,fontweight='bold')

axs[2].plot(list(range(len(num_list3))), y2, 'o-', color = 'tab:blue')
axs[2].set_xticks(list(range(len(num_list3))), num_list3)
axs[2].fill_between(list(range(len(num_list3))), y2-err2, y2+err2, color = 'tab:blue', alpha=0.2)            
axs[2].set_xlabel('Average Number', fontname='Arial', fontsize=18,fontweight='bold')
axs[2].set_ylabel('Test Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
axs[2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1)) 
axs[2].set_title('Test Spatial 3',fontname='Arial', fontsize=18,fontweight='bold')

axs[0].plot(list(range(len(num_list_self))), y3, 'o-', color = 'tab:red')
axs[0].set_xticks(list(range(len(num_list_self))), num_list_self)
axs[0].fill_between(list(range(len(num_list_self))), y3-err3, y3+err3, color = 'tab:red', alpha=0.2)            
axs[0].set_xlabel('Average Number', fontname='Arial', fontsize=18,fontweight='bold')
axs[0].set_ylabel('Test Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
axs[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1)) 
axs[0].set_title('Test Spatial 1',fontname='Arial', fontsize=18,fontweight='bold')
os.makedirs('./test_results',exist_ok=True)
plt.savefig(f"./test_results/train_spatial1_test123_acc_5000.png", dpi=400, bbox_inches='tight')
plt.cla()




#! spatial class 2:
spatial_class=2
spatial_class_index = list(np.where(index_for_spatial_label==spatial_class)[0])
X = data[spatial_class_index, ...]
y = label[spatial_class_index, ...]
X_train, X_test3, y_train, y_test3 = train_test_split(X, y, test_size=0.3, random_state=0)
train_size = y_train.shape[0]
test_size3 = y_test3.shape[0]

spatial_class_index = list(np.where(index_for_spatial_label==1)[0])
X_test1 = data[spatial_class_index, ...]
y_test1 = label[spatial_class_index, ...]
test_size1 = len(spatial_class_index)

spatial_class_index = list(np.where(index_for_spatial_label==3)[0])
X_test2 = data[spatial_class_index, ...]
y_test2 = label[spatial_class_index, ...]
test_size2 = len(spatial_class_index)

num_list = [1, 10, 30, 50, 100, 300, 500, 606-1, 1000, 1167-1, 1210-1]

previous_test_1 = 0
previous_test_2 = 0
previous_test_3 = 0
num_final_acc_mean_std = {}
for i, num_neuron in enumerate(num_list):
    model = CNN(time_len, num_neuron, label.max()+1).cuda()
    test_acc1_list = [] 
    test_acc2_list = [] 
    test_acc3_list = [] 
    training_num = train_size - 1 if num_neuron >= train_size else num_neuron
    test_num1 = test_size1 - 1 if num_neuron >= test_size1 else num_neuron
    test_num2 = test_size2 - 1 if num_neuron >= test_size2 else num_neuron
    test_num3 = test_size3 - 1 if num_neuron >= test_size3 else num_neuron            
    
    for seed in range(5):
        torch.manual_seed(seed)
        if num_neuron <= 1210-1:
            
            save_filename = f"train_spatial{spatial_class}_log_train_num{training_num}_test1_num{test_num1}_test2_num{test_num2}_test3_num{test_num3}_seed{seed}.pkl"
            with open(os.path.join('./training_log_1for3_100e', save_filename), 'rb') as f:
                res = pickle.load(f)
            model.load_state_dict(res['model'])
            model.eval()
            if test_num1 == previous_test_1:
                pass
            else:
                for np_seed in range(trial_per_seed): 
                    np.random.seed(np_seed)
                    test_dataloader = Data.DataLoader(TestDataset(X_test1, y_test1, test_num1), batch_size=1, shuffle=False)
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
                        test_acc /= test_size1 * 24
                        test_acc1_list.append(test_acc)
                        print(f'spatial 1, num:{num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')

            if test_num2 == previous_test_2:
                pass
            else:
                for np_seed in range(trial_per_seed): 
                    np.random.seed(np_seed)
                    test_dataloader = Data.DataLoader(TestDataset(X_test2, y_test2, test_num2), batch_size=1, shuffle=False)
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
                        test_acc /= test_size2 * 24
                        test_acc2_list.append(test_acc)
                        print(f'spatial 2, num:{num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')
            if test_num3 == previous_test_3:
                pass
            else:
                for np_seed in range(trial_per_seed): 
                    np.random.seed(np_seed)
                    test_dataloader = Data.DataLoader(TestDataset(X_test3, y_test3, test_num3), batch_size=1, shuffle=False)
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
                        test_acc /= test_size3 * 24
                        test_acc3_list.append(test_acc)
                        print(f'spatial 3, num:{num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')

        
        elif num_neuron > 1210-1 :
            pass 
            
    if test_num1 == previous_test_1:
        pass
    else:
        num_final_acc_mean_std[str(num_neuron)+'_spatial1'] = test_acc1_list
    if test_num2 == previous_test_2:
        pass
    else:
        num_final_acc_mean_std[str(num_neuron)+'_spatial3'] = test_acc2_list
    if test_num3 == previous_test_3:
        pass
    else:
        num_final_acc_mean_std[str(num_neuron)+'_spatial2'] = test_acc3_list
    previous_test_1 = test_num1
    previous_test_2 = test_num2
    previous_test_3 = test_num3
num_final_acc_mean_std['test_spatial1_numlist'] = [1, 10, 30, 50, 100, 300, 500, 1000, 1210-1]
num_final_acc_mean_std['test_spatial3_numlist'] = [1, 10, 30, 50, 100, 300, 500, 1000, 1167-1]
num_final_acc_mean_std['test_spatial2_numlist'] = [1, 10, 30, 50, 100, 300, 500, 606-1]
data_test1 = []
data_test3 = []
data_test2 = []
num_list2 = [1, 10, 30, 50, 100, 300, 500, 1000, 1210-1]
for num in num_list2:
    key = str(num)+'_spatial1'
    data_test1.append(num_final_acc_mean_std[key])

num_list_self = [1, 10, 30, 50, 100, 300, 500, 606-1]
for num in num_list_self: 
    key = str(num)+'_spatial2'
    data_test2.append(num_final_acc_mean_std[key])


num_list3 = [1, 10, 30, 50, 100, 300, 500, 1000, 1167-1]
for num in num_list3: 
    key = str(num)+'_spatial3'
    data_test3.append(num_final_acc_mean_std[key])

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)  

data_test1=np.array(data_test1)
data_test3=np.array(data_test3)
y1 = np.mean(data_test1, axis=1)
err1 = np.std(data_test1, axis=1)
y2 = np.mean(data_test3, axis=1)
err2 = np.std(data_test3, axis=1)
y3 = np.mean(data_test2, axis=1)
err3 = np.std(data_test2, axis=1)

save_mat1={}
save_mat1['test_spatial1_mean']=y1
save_mat1['test_spatial1_std']=err1
save_mat1['test_spatial1_numlist'] = num_list2
save_mat1['test_spatial3_numlist'] = num_list3
save_mat1['test_spatial3_mean']=y2
save_mat1['test_spatial3_std']=err2
save_mat1['test_spatial2_numlist'] = num_list_self
save_mat1['test_spatial2_mean']=y3
save_mat1['test_spatial2_std']=err3
os.makedirs('./test_results', exist_ok=True)
scio.savemat('./test_results/train_2_test123_acc_5000.mat', save_mat1)

save_mat1 = loadmat('./test_results/train_2_test123_acc_5000.mat')


axs[0].plot(list(range(data_test1.shape[0])), y1, 'o-', color = 'tab:red')
axs[0].fill_between(list(range(data_test1.shape[0])), y1-err1, y1+err1, color='tab:red', alpha=0.2)
axs[0].set_xticks(list(range(data_test1.shape[0])), num_list2)
axs[0].set_xlabel('Average Number', fontname='Arial', fontsize=18,fontweight='bold')
axs[0].set_ylabel('Test Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
axs[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
axs[0].set_title('Test Spatial 1',fontname='Arial', fontsize=18,fontweight='bold')

axs[2].plot(list(range(y2.shape[0])), y2, 'o-', color = 'tab:blue')
axs[2].set_xticks(list(range(y2.shape[0])), num_list3)
axs[2].fill_between(list(range(y2.shape[0])), y2-err2, y2+err2, color = 'tab:blue', alpha=0.2)            
axs[2].set_xlabel('Average Number', fontname='Arial', fontsize=18,fontweight='bold')
axs[2].set_ylabel('Test Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
axs[2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1)) 
axs[2].set_title('Test Spatial 3',fontname='Arial', fontsize=18,fontweight='bold')

axs[1].plot(list(range(y3.shape[0])), y3, 'o-', color = 'tab:green')
axs[1].set_xticks(list(range(y3.shape[0])), num_list_self)
axs[1].fill_between(list(range(y3.shape[0])), y3-err3, y3+err3, color = 'tab:green', alpha=0.2)            
axs[1].set_xlabel('Average Number', fontname='Arial', fontsize=18,fontweight='bold')
axs[1].set_ylabel('Test Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
axs[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1)) 
axs[1].set_title('Test Spatial 2',fontname='Arial', fontsize=18,fontweight='bold')

os.makedirs('./test_results',exist_ok=True)
plt.savefig(f"./test_results/train_spatial2_test123_acc_5000.png", dpi=400, bbox_inches='tight')
plt.cla()



#! spatial class 3:
spatial_class=3
spatial_class_index = list(np.where(index_for_spatial_label==spatial_class)[0])
X = data[spatial_class_index, ...]
y = label[spatial_class_index, ...]
X_train, X_test3, y_train, y_test3 = train_test_split(X, y, test_size=0.3, random_state=0)
train_size = y_train.shape[0]
test_size3 = y_test3.shape[0]

spatial_class_index = list(np.where(index_for_spatial_label==1)[0])
X_test1 = data[spatial_class_index, ...]
y_test1 = label[spatial_class_index, ...]
test_size1 = len(spatial_class_index)

spatial_class_index = list(np.where(index_for_spatial_label==2)[0])
X_test2 = data[spatial_class_index, ...]
y_test2 = label[spatial_class_index, ...]
test_size2 = len(spatial_class_index)

num_list = [1, 10, 30, 50, 100, 300, 351-1, 500, 700-1]#


previous_test_1 = 0
previous_test_2 = 0
previous_test_3 = 0
num_final_acc_mean_std = {}
for i, num_neuron in enumerate(num_list):
    model = CNN(time_len, num_neuron, label.max()+1).cuda()
    test_acc1_list = [] 
    test_acc2_list = []
    test_acc3_list = []  
    training_num = train_size - 1 if num_neuron >= train_size else num_neuron
    test_num1 = test_size1 - 1 if num_neuron >= test_size1 else num_neuron
    test_num2 = test_size2 - 1 if num_neuron >= test_size2 else num_neuron
    test_num3 = test_size3 - 1 if num_neuron >= test_size3 else num_neuron      
    
    
    for seed in range(5):
        torch.manual_seed(seed)
        if num_neuron <= 700-1  :
            
            save_filename = f"train_spatial{spatial_class}_log_train_num{training_num}_test1_num{test_num1}_test2_num{test_num2}_test3_num{test_num3}_seed{seed}.pkl"
            with open(os.path.join('./training_log_1for3_100e', save_filename), 'rb') as f:
                res = pickle.load(f)
            model.load_state_dict(res['model'])
            model.eval()
            if test_num1 == previous_test_1:
                pass
            else:
                for np_seed in range(trial_per_seed):
                    np.random.seed(np_seed)
                    test_dataloader = Data.DataLoader(TestDataset(X_test1, y_test1, test_num1), batch_size=1, shuffle=False)
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
                        test_acc /= test_size1 * 24
                        test_acc1_list.append(test_acc)
                        print(f'spatial 1, num:{num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')

            if test_num2 == previous_test_2:
                pass
            else:
                for np_seed in range(trial_per_seed): 
                    np.random.seed(np_seed)
                    test_dataloader = Data.DataLoader(TestDataset(X_test2, y_test2, test_num2), batch_size=1, shuffle=False)
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
                        test_acc /= test_size2 * 24
                        test_acc2_list.append(test_acc)
                        print(f'spatial 2, num:{num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')
            
            if test_num3 == previous_test_3:
                pass
            else:
                for np_seed in range(trial_per_seed): 
                    np.random.seed(np_seed)
                    test_dataloader = Data.DataLoader(TestDataset(X_test3, y_test3, test_num3), batch_size=1, shuffle=False)
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
                        test_acc /= test_size3 * 24
                        test_acc3_list.append(test_acc)
                        print(f'spatial 3, num:{num_neuron}, test_acc: {np.round(test_acc*100, 2)}%')

        elif num_neuron > 700-1  : 
            pass
    if test_num1 == previous_test_1:
        pass
    else:
        num_final_acc_mean_std[str(num_neuron)+'_spatial1'] = test_acc1_list
    if test_num2 == previous_test_2:
        pass
    else:
        num_final_acc_mean_std[str(num_neuron)+'_spatial2'] = test_acc2_list
    if test_num3 == previous_test_3:
        pass
    else:
        num_final_acc_mean_std[str(num_neuron)+'_spatial3'] = test_acc3_list
    previous_test_1 = test_num1
    previous_test_2 = test_num2
    previous_test_3 = test_num3
    
num_final_acc_mean_std['test_spatial1_numlist'] = [1, 10, 30, 50, 100, 300, 500, 700-1]
num_final_acc_mean_std['test_spatial2_numlist'] = [1, 10, 30, 50, 100, 300, 500, 700-1]
num_final_acc_mean_std['test_spatial3_numlist'] = [1, 10, 30, 50, 100, 300, 351-1]

data_test1 = []
data_test2 = []
data_test3 = []
num_list2 = [1, 10, 30, 50, 100, 300, 500, 700-1]
for num in num_list2:
    key = str(num)+'_spatial1'
    data_test1.append(num_final_acc_mean_std[key])

num_list3 = [1, 10, 30, 50, 100, 300, 500, 700-1]
for num in num_list3: 
    key = str(num)+'_spatial2'
    data_test2.append(num_final_acc_mean_std[key])

num_list_self = [1, 10, 30, 50, 100, 300, 351-1]
for num in num_list_self: 
    key = str(num)+'_spatial3'
    data_test3.append(num_final_acc_mean_std[key])

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)  

data_test1=np.array(data_test1)
data_test2=np.array(data_test2)
data_test3=np.array(data_test3)
y1 = np.mean(data_test1, axis=1)
err1 = np.std(data_test1, axis=1)
y2 = np.mean(data_test2, axis=1)
err2 = np.std(data_test2, axis=1)
y3 = np.mean(data_test3, axis=1)
err3 = np.std(data_test3, axis=1)

save_mat1={}
save_mat1['test_spatial1_mean']=y1
save_mat1['test_spatial1_std']=err1
save_mat1['test_spatial1_numlist'] = num_list2
save_mat1['test_spatial2_numlist'] = num_list3
save_mat1['test_spatial2_mean']=y2
save_mat1['test_spatial2_std']=err2
save_mat1['test_spatial3_numlist'] = num_list_self
save_mat1['test_spatial3_mean']=y3
save_mat1['test_spatial3_std']=err3
os.makedirs('./test_results', exist_ok=True)
scio.savemat('./test_results/train_3_test123_acc_test5000.mat', save_mat1)

save_mat1 = loadmat('./test_results/train_3_test123_acc_test5000.mat')


axs[0].plot(list(range(data_test1.shape[0])), y1, 'o-', color = 'tab:red')
axs[0].fill_between(list(range(data_test1.shape[0])), y1-err1, y1+err1, color='tab:red', alpha=0.2)
axs[0].set_xticks(list(range(data_test1.shape[0])), num_list2)
axs[0].set_xlabel('Average Number', fontname='Arial', fontsize=18,fontweight='bold')
axs[0].set_ylabel('Test Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
axs[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
axs[0].set_title('Test Spatial 1',fontname='Arial', fontsize=18,fontweight='bold')

axs[1].plot(list(range(data_test2.shape[0])), y2, 'o-', color = 'tab:green')
axs[1].set_xticks(list(range(data_test2.shape[0])), num_list3)
axs[1].fill_between(list(range(data_test2.shape[0])), y2-err2, y2+err2, color = 'tab:green', alpha=0.2)            
axs[1].set_xlabel('Average Number', fontname='Arial', fontsize=18,fontweight='bold')
axs[1].set_ylabel('Test Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
axs[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1)) 
axs[1].set_title('Test Spatial 2',fontname='Arial', fontsize=18,fontweight='bold')

axs[2].plot(list(range(data_test3.shape[0])), y3, 'o-', color = 'tab:blue')
axs[2].set_xticks(list(range(data_test3.shape[0])), num_list_self)
axs[2].fill_between(list(range(data_test3.shape[0])), y3-err3, y3+err3, color = 'tab:blue', alpha=0.2)            
axs[2].set_xlabel('Average Number', fontname='Arial', fontsize=18,fontweight='bold')
axs[2].set_ylabel('Test Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
axs[2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1)) 
axs[2].set_title('Test Spatial 3',fontname='Arial', fontsize=18,fontweight='bold')

os.makedirs('./test_results',exist_ok=True)
plt.savefig(f"./test_results/train_spatial3_test123_acc_test5000.png", dpi=400, bbox_inches='tight')
plt.cla()
