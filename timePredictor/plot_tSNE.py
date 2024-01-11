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
        return feature, self.fc(feature)

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
filename_list = [
        './SCNData/all_neuron_20210916.mat',
        './SCNData/all_neuron_20210918.mat',
        './SCNData/all_neuron_20210922.mat']

model = CNN().cuda()

for filename in filename_list:
    if '0916' in filename:
        final_average_num=1816
        # average_list = list(range(851,860,1))
    elif '0918' in filename:
        final_average_num=2335
        # average_list = list(range(831,840,1))
    elif '0922' in filename:
        final_average_num=2350
        # average_list = list(range(791,800,1))
    average_list = [1,20,100,300,900]#[1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 900, 1000, 1500]
    # average_list.extend([final_average_num])
    # average_list = list(range(760,800,10))
    # average_list=[final_average_num-1]
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

        seed = 0
        np.random.seed(seed)
        torch.manual_seed(seed)

        batch_size = 128
        train_size = int(data.shape[0] * 0.6)
        val_size = int(data.shape[0] * 0.1)
        test_size = data.shape[0] - train_size - val_size

        shuffle_idx = np.arange(num_neuron)
        np.random.shuffle(shuffle_idx)
        data = data[shuffle_idx]
        label = label[shuffle_idx]

        test_dataloader = Data.DataLoader(TestDataset(data[train_size+val_size:], label[train_size+val_size:], average_num), batch_size=batch_size, shuffle=True)

        
        weight_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{0}.pkl"
        with open(os.path.join('./training_log', weight_filename), 'rb') as f:
            res = pickle.load(f)
        # set the model to evaluation mode
        model.load_state_dict(res['model'])
        model.eval()

        # generate some example data
        x = torch.randn(10, 1, 100)

        # get the features before the final FC layer
        features_list=[]
        test_embeddings = torch.zeros((0, 224), dtype=torch.float32).cuda()
        test_predictions = []
        for x, y in test_dataloader:
            
            x = x.cuda()
            y = y.cuda()
            x = x.reshape(-1, 1, time_len)
            y = y.reshape(-1)
            with torch.no_grad():
                feature, logits = model(x)
                test_embeddings = torch.cat((test_embeddings, feature.detach()),0)
                preds = torch.argmax(logits, dim=1)
                test_predictions.extend(preds.detach().cpu().tolist())
        features = np.array(test_embeddings.cpu().numpy())
        test_predictions = np.array(test_predictions)
        # flatten the features for input to t-SNE
        features_flat = features.reshape(features.shape[0], -1)

        ### plot t-SNE visualization with 2D map
        tsne = TSNE(n_components=2)
        ### plot the t-SNE visualization with 3D map
        # tsne = TSNE(n_components=3)
        # features_tsne = tsne.fit_transform(features_flat)
        features_tsne = tsne.fit_transform(features_flat)
        num_categories = 24
        for color_map in ['jet', 'rainbow', 'turbo', 'gist_rainbow']:
            cmap = matplotlib.colormaps.get_cmap(color_map)
            values = list(range(num_categories)) 
            values = [x/(num_categories-1) for x in values]
            new_cmap = colors.ListedColormap(cmap(values))

            fig, ax = plt.subplots(figsize=(8,8))
            
            ### plot the t-SNE visualization with 2D map
            for lab in range(num_categories):
                indices = test_predictions==lab
                ax.scatter(features_tsne[indices, 0],features_tsne[indices,1], c=np.array(new_cmap(lab)).reshape(1,4), label =f'Time {lab+1}', alpha=1.0)
            
            ### plot the t-SNE visualization with 3D map
            # for lab in range(num_categories):
            #     indices = test_predictions == lab
            #     ax.scatter(features_tsne[indices, 0],
            #             features_tsne[indices, 1],
            #             features_tsne[indices, 2],
            #             c=np.array(new_cmap(lab)).reshape(1, 4),
            #             label= f'Time {lab+1}',
            #             alpha=1.0)

            ax.legend(loc='upper right', fontsize='small', markerscale=2)
            plt.title(f'Average {average_num}', size = 20, fontweight='bold')
            plt.yticks(fontproperties = 'Arial', size = 20, fontweight='bold')
            plt.xticks(fontproperties = 'Arial', size = 20, fontweight='bold')
            save_filename = os.path.basename(filename).split('.')[0].split('_')[-1] 
            if '0916' in filename:
                sub_name = 'Dataset1'
            elif '0918' in filename:
                sub_name = 'Dataset2'
            elif '0922' in filename:
                sub_name = 'Dataset3'
            os.makedirs(f'./t-SNE_v2/{color_map}/{sub_name}', exist_ok = True)
            plt.savefig(f'./t-SNE_v2/{color_map}/{sub_name}/{save_filename}_avg{average_num}_{color_map}.png')
            
            plt.cla()
            plt.close()