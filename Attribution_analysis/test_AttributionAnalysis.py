import torch
import numpy as np
from src import CNN
from src.dataset import NeuronData
from torch.utils.data import DataLoader
import os
from captum.attr import IntegratedGradients

import pickle
import scipy.io as scio 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pandas as pd

dataset_dir = [
    '../SCNData/Dataset1_SCNProject.mat',
    '../SCNData/Dataset2_SCNProject.mat',
    '../SCNData/Dataset3_SCNProject.mat',
    '../SCNData/Dataset4_SCNProject.mat',
    '../SCNData/Dataset5_SCNProject.mat',
    '../SCNData/Dataset6_SCNProject.mat',
    ]



for filename in dataset_dir:
    print(f'In processing: {filename}')
    spatial_data = np.array(scio.loadmat(filename, variable_names=['POI'])['POI'])
    spatial_data = np.array([spatial_data[indx][0][0] for indx in range(spatial_data.shape[0])])
        
    seed_level_all_random_add_normalNeuron=[]
    seed_level_allrandom_add_keyNeuron=[]

    for seed in range(5): # here seed only refer to load weights.
        np.random.seed(0)
        print(f'In processing seed {seed}') 

        if 'Dataset1' in filename:
            num_neurons=6049
            sub_name='Dataset1'
        elif 'Dataset2' in filename:
            num_neurons=7782
            sub_name='Dataset2'
        elif 'Dataset3' in filename:
            num_neurons=7828
            sub_name='Dataset3'
        elif 'Dataset4' in filename:
            num_neurons=6445
            sub_name='Dataset4'
        elif 'Dataset5' in filename:
            num_neurons=8229
            sub_name='Dataset5'
        elif 'Dataset6' in filename:
            num_neurons=8968
            sub_name='Dataset6'
        weight_filename = f'CNN_{sub_name}_seed{seed}'
        with open(os.path.join('training_log', f"{os.path.basename(filename).split('_SCNProject')[0]}_log_num_neuron{num_neurons}_seed{seed}.pkl"), 'rb') as f:
            res = pickle.load(f)
        model = CNN(num_seq=num_neurons, base_channel=16, num_class=24).to(device)
        model.load_state_dict(res['model'], strict=True)
        
        val_dataset = NeuronData(filename, class_num=24)
        test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, sampler=None,
                        num_workers=1, pin_memory=True, drop_last=False)

        test_iter = iter(test_loader) 
        all_key_neuron_pos={}
        for class_index in range(24):
            x, label = next(test_iter)
            x = x.to(device)
            target_class = int(label.item())
            input_tensor = x.to(device, non_blocking=True).view(1, num_neurons)
            
            baseline_tensor = torch.zeros_like(input_tensor).to(device)
            # Define the Integrated Gradients method
            ig = IntegratedGradients(model)
            attribution_map, _ = ig.attribute(input_tensor, baseline_tensor, target=target_class, return_convergence_delta=True) #! means target_class
            
            attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-9)
            # Convert the attribution map to a numpy array
            attribution_map = attribution_map.view(1,num_neurons).detach().cpu().numpy()[0,:]
            all_key_neuron_pos['Time'+str(class_index+1)] = attribution_map 
        
        os.makedirs(f'./neuron_weight/', exist_ok = True)
        os.makedirs(f'./neuron_weight/{sub_name}', exist_ok = True)
        scio.savemat(f'./neuron_weight/{sub_name}/{sub_name}_NeuronWeight_seed{seed}.mat', all_key_neuron_pos)

        for class_id in range(24):
            weight_for_class = all_key_neuron_pos['Time'+str(class_id+1)]

            dict_weight_per_neuron = {}
            for item in np.arange(0, 1.01, 0.01):
                dict_weight_per_neuron[round(item, 2)] = 0
            for weight_perNeuron in weight_for_class:  
                weight_perNeuron = round(weight_perNeuron, 2)
                dict_weight_per_neuron[weight_perNeuron] = dict_weight_per_neuron[weight_perNeuron]+1
            print(dict_weight_per_neuron)
            print(np.sum(list(dict_weight_per_neuron.values())))
            
            os.makedirs(f'./neuron_weight_CNN_perclass/{sub_name}', exist_ok = True)  
            with open(f'./neuron_weight_CNN_perclass/{sub_name}/SCN{sub_name}_time{class_id}_Weight_Dist.csv', 'w') as f:
                [f.write('{0},{1}\n'.format(key, value)) for key, value in dict_weight_per_neuron.items()]
            data = {'weight': dict_weight_per_neuron.keys(), 'neuron_num': dict_weight_per_neuron.values()}
            df = pd.DataFrame(data)
            df.to_excel(f'./neuron_weight_CNN_perclass/{sub_name}/SCN{sub_name}_time{class_id}_Weight_Dist.xlsx', index=False)

                