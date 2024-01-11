import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.utils.data as Data
from scipy.io import loadmat
import pickle, os
import matplotlib
import matplotlib.colors as colors
from sklearn.model_selection import train_test_split
from captum.attr import IntegratedGradients

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() 
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5),
            nn.Conv1d(32, 32, kernel_size=3, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        self.fc = nn.Sequential(*[nn.Linear(224, 24)])

    def forward(self, x):
        feature = self.conv(x)
        return self.fc(feature)

class TestDataset(Data.Dataset):
    def __init__(self, x, y, average_num):
        super(TestDataset, self).__init__()
        self.x = x
        self.y = y
        self.average_num = average_num
        

    def __getitem__(self, index):
        label = self.y[index]
        random_idx = np.random.choice(self.x.shape[0], self.average_num, replace=False)
        return np.mean(self.x[random_idx], axis=0), label

    def __len__(self):
        return self.x.shape[0]

os.environ['OPENBLAS_NUM_THREADS'] = '1'
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
filename_list = [
        './SCNData/all_neuron_20210916.mat',
        './SCNData/all_neuron_20210918.mat',
        './SCNData/all_neuron_20210922.mat']

model = CNN().cuda()

for filename in filename_list:
    if '0916' in filename:
        final_average_num=1816
    elif '0918' in filename:
        final_average_num=2335
    elif '0922' in filename:
        final_average_num=2350
    average_list = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 900, 1000, 1500]
    average_list.extend([final_average_num])
    for i, average_num in enumerate(average_list):
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

        batch_size = 1
        train_size = int(data.shape[0] * 0.6)
        val_size = int(data.shape[0] * 0.1)
        test_size = data.shape[0] - train_size - val_size

        # shuffle_idx = np.arange(num_neuron)
        # np.random.shuffle(shuffle_idx)
        # data = data[shuffle_idx]
        # label = label[shuffle_idx]
        X_train_val, X_test_all, y_train_val, y_test_all = train_test_split(data, label, test_size=0.3, random_state=0)
        for class_id in range(24):
            X_test = X_test_all[:,class_id,:][:,np.newaxis,:]
            y_test = y_test_all[:,class_id][:,np.newaxis]
            test_dataloader = Data.DataLoader(TestDataset(X_test, y_test, average_num), batch_size=batch_size, shuffle=False)

            
            weight_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{0}.pkl"
            with open(os.path.join('./training_log', weight_filename), 'rb') as f:
                res = pickle.load(f)
            # set the model to evaluation mode
            model.load_state_dict(res['model'])
            model.eval()

            # get the features before the final FC layer
            
            # for label in range(24):
            # test_iter = iter(test_dataloader) 
            count = 0
            for id, data in enumerate(test_dataloader):                
                x, y = data #next(test_iter)
                # x = x[:, 0, :].unsqueeze(1)      
                # y = y[:,0] 
                                    
                x = x.cuda()
                y = y.cuda()
                # x = x.reshape(-1, 1, time_len)
                y = y.reshape(-1)
                count +=1
                    
            baseline_tensor = torch.zeros_like(x).to(x.device)

            # Define the Integrated Gradients method
            ig = IntegratedGradients(model)
            target_class = int(y.detach().cpu().item())
            attribution_map,_ = ig.attribute(x, baseline_tensor, target=target_class, return_convergence_delta=True) #! means target_class
            attribution_map = attribution_map.sum(dim=1, keepdim=True)
            attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-9)
            # Convert the attribution map to a numpy array
            attribution_map = attribution_map.detach().view(1, 200).cpu().numpy()
            input_tensor = x.detach().view(1, 200).cpu().numpy()
            

            # Create a heatmap of the merged data
            fig, ax = plt.subplots(2, 1, figsize=(9, 4), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)

            # fig.tight_layout()
            x = np.arange(input_tensor.shape[1])
            # plt.plot(x, color='black')
            ax[1].plot(input_tensor[0], color='black')

            # Add title and axis labels to the original data plot
            ax[1].set_title(f'Average {average_num} for Time {target_class+1}', fontproperties = 'Arial', size = 14, fontweight='bold')
            ax[1].set_xlabel('Time Steps', fontproperties = 'Arial', size = 12, fontweight='bold')
            ax[1].set_ylabel('Amplitude', fontproperties = 'Arial', size = 12, fontweight='bold')
            ax[1].set_xticks(np.arange(0, 200, 20))
            plt.xlim(0, 200)        
            
            # Plot the attribution map
            im = ax[0].pcolormesh(attribution_map, cmap='jet', shading='auto')
            ax[0].set_title('Attribution Map', fontproperties = 'Arial', size = 14, fontweight='bold')
            # ax[0].set_ylabel('Importance', fontproperties = 'Arial', size = 12, fontweight='bold')
            ax[0].set_ylabel(ylabel='Importance', fontproperties = 'Arial',size = 12,fontweight='bold')
            for label in ax[0].get_yticklabels():
                label.set_weight('bold')
            # cbar = fig.colorbar(im, ax=ax[1], orientation='vertical')
            # cbar.ax.tick_params(labelsize=1)
            # Add colorbar to the plot
            # ax[0].tick_params(axis='both', which='both', labelsize=8)
            cbar = fig.colorbar(im, ax=ax[0], orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.ax.tick_params(labelsize=8)
            # ax[1].plot(attribution_map, cmap='hot', shading='auto')
            plt.yticks(fontproperties = 'Arial',fontweight='bold')
            plt.xticks(fontproperties = 'Arial',fontweight='bold')
            # Adjust spacing between subplots
            fig.tight_layout()
            plt.subplots_adjust(wspace = 2)
            os.makedirs('./attribution_maps', exist_ok=True)
            if '0916' in filename:
                sub_name = '0916'
            elif '0918' in filename:
                sub_name = '0918'
            elif '0922' in filename:
                sub_name = '0922'
            os.makedirs(f'./attribution_maps/Dataset{sub_name}', exist_ok=True)
            plt.savefig(f'./attribution_maps/Dataset{sub_name}/{sub_name}_attribution_map_avg{average_num}_Time_{target_class+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
            if count ==1:
                break