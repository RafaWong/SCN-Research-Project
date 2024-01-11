import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, os
import numpy as np
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
import scipy.io as scio
filename_list = [
        './SCNData/all_neuron_20210916.mat',
        './SCNData/all_neuron_20210918.mat',
        './SCNData/all_neuron_20210922.mat']
        

#! this part is for visualizing the 'test-set' accuracy curve during training.
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)
for filename in filename_list:
    if '0916' in filename:
        final_average_num=1816
    elif '0918' in filename:
        final_average_num=2335
    elif '0922' in filename:
        final_average_num=2350
    average_list = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 1000, 1500]
    # average_list.extend([final_average_num])
    all_s=[]
    for i, average_num in enumerate(average_list):
        test_loss_list = list()
        test_acc_list = list() 
        for seed in range(5):
            save_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{seed}.pkl"
            with open(os.path.join('./training_log', save_filename), 'rb') as f:
                res = pickle.load(f)
            test_loss = res['test_loss_list'][0]
            test_acc = res['test_acc_list'][0]
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

        test_loss_list = np.array(test_loss_list)
        test_acc_list = np.array(test_acc_list)
        x = np.arange(100)
        
        y = np.mean(test_acc_list, axis=0)
        err = np.std(test_acc_list)
        s1 = axs.plot(x, y, label=f'Average {average_num}')
        # axs.fill_between(x, y-err, y+err)
        axs.set_xlabel('Training epoch', fontname='Arial', fontsize=18,fontweight='bold')
        axs.set_ylabel('Accuracy', fontname='Arial', fontsize=18,fontweight='bold')
        axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
        all_s.append(s1)
    plt.tight_layout()
    plt.yticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.xticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    
    plt.legend(loc='upper left', bbox_to_anchor=(0.775, 0.985), prop = {'size':9})  
    plt.savefig(f"./acc_images/{save_filename.split('_seed')[0]}_res_acc.png", dpi=400, bbox_inches='tight')
    axs.cla()

# #! this part is for visualizing the 'test-set' loss curve during training.
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)
for filename in filename_list:
    if '0916' in filename:
        final_average_num=1816
    elif '0918' in filename:
        final_average_num=2335
    elif '0922' in filename:
        final_average_num=2350
    average_list = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 1000, 1500]
    # average_list.extend([final_average_num])
    all_s=[]
    for i, average_num in enumerate(average_list):
        test_loss_list = list()
        test_acc_list = list() 
        for seed in range(5):
            save_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{seed}.pkl"
            with open(os.path.join('./training_log', save_filename), 'rb') as f:
                res = pickle.load(f)
                
            test_loss = res['test_loss_list'][0]
            test_acc = res['test_acc_list'][0]
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
        test_loss_list = np.array(test_loss_list)
        x = np.arange(100)
        y = np.mean(test_loss_list, axis=0)
        err = np.std(test_loss_list, axis=0)
        s1 = axs.plot(x, y, label=f'Average {average_num}')
        # axs.fill_between(x, y-err, y+err)
        axs.set_xlabel('Training epoch', fontname='Arial', fontsize=18, fontweight='bold')
        axs.set_ylabel('CE loss', fontname='Arial', fontsize=18, fontweight='bold')
        all_s.append(s1)
    plt.tight_layout()
    plt.yticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.xticks(fontproperties = 'Arial', size = 12, fontweight='bold')

    plt.legend(loc='upper left', bbox_to_anchor=(0.775, 0.985), prop = {'size':10})  
    plt.savefig(f"./acc_images/{save_filename.split('_seed')[0]}_res_loss.png", dpi=400, bbox_inches='tight')
    axs.cla()
    
# #! this part is for visualizing the 'test-set' accuracy with different average number.
fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)
for filename in filename_list:
    if '0916' in filename:
        final_average_num=1816
    elif '0918' in filename:
        final_average_num=2335
    elif '0922' in filename:
        final_average_num=2350
    average_list = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 1000, 1500]
    # average_list.extend([final_average_num])
    all_s=[]
    for i, average_num in enumerate(average_list):
        test_loss_list = list()
        test_acc_list = list() 
        for seed in range(5):
            save_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{seed}.pkl"
            with open(os.path.join('./training_log', save_filename), 'rb') as f:
                res = pickle.load(f)
                
            test_loss = res['test_loss_list'][0]
            test_acc = res['test_acc_list'][0]
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

        test_loss_list = np.array(test_loss_list)
        test_acc_list = np.array(test_acc_list)

        y = np.mean(test_acc_list)
        # print(f'average{average_num}, acc:{y}')
        err = np.std(test_acc_list)
        s1 = axs.bar([i], y, label=f'Average {average_num}')
        all_s.append(s1)
        axs.errorbar([i], y, yerr=err, fmt='-', color='k', capsize=4, capthick=1, barsabove=True) #! T error bar
    axs.set_xticks(np.arange(len(average_list)))
    axs.set_xticklabels(average_list)
    axs.set_xlabel('Average traces', fontname='Arial', fontsize=18, fontweight='bold')
    axs.set_ylabel('Accuracy', fontname='Arial', fontsize=18, fontweight='bold')
    threshold1 = 1.0
    axs.axhline(threshold1, color='grey', lw=1, alpha=0.7, linestyle ='--')
    threshold2 = 1/24
    axs.axhline(threshold2, color='grey', lw=1, alpha=0.7, linestyle ='--')
    axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))

    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.985), prop = {'size':9})
    plt.tight_layout()
    plt.yticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.xticks(fontproperties = 'Arial', size = 12, fontweight='bold')

    plt.savefig(f"./acc_images/{save_filename.split('_seed')[0]}_res_avg.png", dpi=400, bbox_inches='tight')
    axs.cla()

fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)
for filename in filename_list:
    if '0916' in filename:
        final_average_num=1816
    elif '0918' in filename:
        final_average_num=2335
    elif '0922' in filename:
        final_average_num=2350
    average_list = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 1000, 1500]
    # average_list.extend(list(range(500,1000,50)))
    # average_list.extend([final_average_num])
    all_s=[]
    all_err=[]
    for i, average_num in enumerate(average_list):
        test_loss_list = list()
        test_acc_list = list() 
        for seed in range(5):
            save_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{seed}.pkl"
            with open(os.path.join('./training_log', save_filename), 'rb') as f:
                res = pickle.load(f)
                
            test_loss = res['test_loss_list'][0]
            test_acc = res['test_acc_list'][0]
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

        test_loss_list = np.array(test_loss_list)
        test_acc_list = np.array(test_acc_list)

        y = np.mean(test_acc_list)
        err = np.std(test_acc_list)
        # s1 = axs.plot([i], y, label=f'Average {average_num}')
        # all_s.append(s1)
        all_s.append(y)
        all_err.append(err)
        # axs.errorbar([i], y, yerr=err, fmt='-', color='k', capsize=4, capthick=1, barsabove=True) #! T error bar
        
    if '0916' in filename:
        color='tab:red'
        label = 'Dataset 1'
    elif '0918' in filename:
        color='tab:green'
        label = 'Dataset 2'
    elif '0922' in filename:
        color='tab:blue'
        label = 'Dataset 3'
    plt.plot(list(np.arange(len(average_list))), all_s, 'o-', color = color, label=label)
    axs.fill_between(list(np.arange(len(average_list))), np.array(all_s) - np.array(all_err), np.array(all_s) + np.array(all_err), alpha=0.2)
    # if '0916' in filename:
    axs.set_xticks(np.arange(len(average_list)))
    axs.set_xticklabels(average_list)
    axs.set_xlabel('Average traces', fontname='Arial', fontsize=18, fontweight='bold')
    axs.set_ylabel('Accuracy', fontname='Arial', fontsize=18, fontweight='bold')
    threshold1 = 1.0
    axs.axhline(threshold1, color='grey', lw=1, alpha=0.7, linestyle ='--')
    threshold2 = 1/24
    axs.axhline(threshold2, color='grey', lw=1, alpha=0.7, linestyle ='--')
    # axs.axvline(len(average_list)-1, color='tab:grey', lw=1, alpha=0.7, linestyle ='--')
    
    axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    
    # if '0916' in filename:
    #     plt.annotate(f'{average_list[-1]}', xy=(list(np.arange(len(average_list)))[-1], 0), xytext=(0, 0), textcoords='offset points', ha='center')
    # if '0918' in filename:
    #     plt.annotate(f'{average_list[-1]}', xy=(list(np.arange(len(average_list)))[-1], 0), xytext=(0, -30), textcoords='offset points', ha='center',fontproperties = 'Arial', size = 12, fontweight='bold')
    # elif '0922' in filename:
    #     plt.annotate(f'{average_list[-1]}', xy=(list(np.arange(len(average_list)))[-1], 0), xytext=(0, -42.5), textcoords='offset points', ha='center',fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.925), prop = {'size':12})
    plt.tight_layout()
    plt.yticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.xticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.text(-0.925, threshold2, '%.1f%%'%(threshold2*100), ha='right', va='center',fontproperties = 'Arial', size = 12, fontweight='bold')
plt.savefig(f"./acc_images/Alldata_res_avg_line_final1500.png", dpi=400, bbox_inches='tight')

for filename in filename_list:
    if '0916' in filename:
        final_average_num=1816
    elif '0918' in filename:
        final_average_num=2335
    elif '0922' in filename:
        final_average_num=2350
    average_list = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 1000, 1500]
    # average_list.extend(list(range(500,1000,50)))
    # average_list.extend([final_average_num])
    all_mean=[]
    all_err=[]
    for i, average_num in enumerate(average_list):
        test_loss_list = list()
        test_acc_list = list() 
        for seed in range(5):
            save_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{seed}.pkl"
            with open(os.path.join('./training_log', save_filename), 'rb') as f:
                res = pickle.load(f)
                
            test_loss = res['test_loss_list'][0]
            test_acc = res['test_acc_list'][0]
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

        test_loss_list = np.array(test_loss_list)
        test_acc_list = np.array(test_acc_list)

        y = np.mean(test_acc_list)
        err = np.std(test_acc_list)
        # s1 = axs.plot([i], y, label=f'Average {average_num}')
        # all_s.append(s1)
        all_mean.append(y)
        all_err.append(err)
    
    mean_std_avg={}
    mean_std_avg['avgList']=average_list
    mean_std_avg['mean']=all_mean
    mean_std_avg['std']=all_err
    print(len(average_list))
    print(len(all_mean))
    print(len(all_err))
    if '20210916' in filename:
        sub_name = '0916'
    elif '20210918' in filename:
        sub_name = '0918'
    elif '20210922' in filename:
        sub_name = '0922'
    # os.makedirs(f'./mean_acc/', exist_ok = True)  
    # scio.savemat(f'./mean_acc/mean_std_{sub_name}.mat', mean_std_avg)
    with open(f'./mean_acc/{sub_name}_acc_mean_std.csv', 'w') as f:
        [f.write('{0},{1},{2}\n'.format(avg, mean, err)) for avg, mean, err in zip(average_list, all_mean, all_err)]


fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)
for filename in filename_list:
    if '0916' in filename:
        final_average_num=1816
    elif '0918' in filename:
        final_average_num=2335
    elif '0922' in filename:
        final_average_num=2350
    average_list = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 1000, 1500]
    # average_list.extend(list(range(500,1000,50)))
    # average_list.extend([final_average_num])
    all_s=[]
    all_err=[]
    for i, average_num in enumerate(average_list):
        test_loss_list = list()
        test_acc_list = list() 
        for seed in range(5):
            save_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{seed}.pkl"
            with open(os.path.join('./training_log', save_filename), 'rb') as f:
                res = pickle.load(f)
                
            test_loss = res['test_loss_list'][0]
            test_acc = res['test_acc_list'][0]
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

        test_loss_list = np.array(test_loss_list)
        test_acc_list = np.array(test_acc_list)

        y = np.mean(test_acc_list)
        err = np.std(test_acc_list)
        # s1 = axs.plot([i], y, label=f'Average {average_num}')
        # all_s.append(s1)
        all_s.append(y)
        all_err.append(err)
        # axs.errorbar([i], y, yerr=err, fmt='-', color='k', capsize=4, capthick=1, barsabove=True) #! T error bar
        
    if '0916' in filename:
        color='tab:red'
        label = 'Dataset 1'
    elif '0918' in filename:
        color='tab:green'
        label = 'Dataset 2'
    elif '0922' in filename:
        color='tab:blue'
        label = 'Dataset 3'
    plt.plot(list(np.arange(len(average_list))), all_s, 'o-', color = color, label=label)
    axs.fill_between(list(np.arange(len(average_list))), np.array(all_s) - np.array(all_err), np.array(all_s) + np.array(all_err), alpha=0.2)
    if '0916' in filename:
        axs.set_xticks(np.arange(len(average_list)))
        axs.set_xticklabels(average_list)
        threshold1 = 1.0
        axs.axhline(threshold1, color='grey', lw=1, alpha=0.7, linestyle ='--')
        threshold2 = 1/24
        axs.axhline(threshold2, color='grey', lw=1, alpha=0.7, linestyle ='--')
    axs.set_xlabel('Average traces', fontname='Arial', fontsize=18, fontweight='bold')
    axs.set_ylabel('Accuracy', fontname='Arial', fontsize=18, fontweight='bold')
    
    axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
    
    # if '0916' in filename:
    #     plt.annotate(f'{average_list[-1]}', xy=(list(np.arange(len(average_list)))[-1], 0), xytext=(0, 0), textcoords='offset points', ha='center')
    if '0918' in filename:
        plt.annotate(f'{average_list[-1]}', xy=(list(np.arange(len(average_list)))[-1], 0), xytext=(0, -30), textcoords='offset points', ha='center',fontproperties = 'Arial', size = 12, fontweight='bold')
    elif '0922' in filename:
        plt.annotate(f'{average_list[-1]}', xy=(list(np.arange(len(average_list)))[-1], 0), xytext=(0, -42.5), textcoords='offset points', ha='center',fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.925), prop = {'size':12})
    plt.tight_layout()
    plt.yticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.xticks(fontproperties = 'Arial', size = 12, fontweight='bold')
    plt.text(-0.975, threshold2, '%.1f%%'%(threshold2*100), ha='right', va='center',fontproperties = 'Arial', size = 12, fontweight='bold')
plt.savefig(f"./acc_images/Alldata_res_avg_line.png", dpi=400, bbox_inches='tight')

fig, axs = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=2.0, bottom=0.1, top=2.0, wspace=0.2)
for filename in filename_list:
    if '0916' in filename:
        final_average_num=1816
    elif '0918' in filename:
        final_average_num=2335
    elif '0922' in filename:
        final_average_num=2350
    average_list = [1, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400, 500, 1000, 1500]
    # average_list.extend(list(range(500,1000,50)))
    # average_list.extend([final_average_num])
    all_mean=[]
    all_err=[]
    for i, average_num in enumerate(average_list):
        test_loss_list = list()
        test_acc_list = list() 
        for seed in range(5):
            save_filename = f"{filename.split('all_neuron_')[-1].split('.mat')[0]}_log_average{average_num}_seed{seed}.pkl"
            with open(os.path.join('./training_log', save_filename), 'rb') as f:
                res = pickle.load(f)
                
            test_loss = res['test_loss_list'][0]
            test_acc = res['test_acc_list'][0]
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

        test_loss_list = np.array(test_loss_list)
        test_acc_list = np.array(test_acc_list)

        y = np.mean(test_acc_list)
        err = np.std(test_acc_list)
        # s1 = axs.plot([i], y, label=f'Average {average_num}')
        # all_s.append(s1)
        all_mean.append(y)
        all_err.append(err)
        # axs.errorbar([i], y, yerr=err, fmt='-', color='k', capsize=4, capthick=1, barsabove=True) #! T error bar
        
    if '0916' in filename:
        color='tab:red'
        sub_name = 'Dataset1'
    elif '0918' in filename:
        color='tab:green'
        sub_name = 'Dataset2'
    elif '0922' in filename:
        color='tab:blue'
        sub_name = 'Dataset3'
    import pandas as pd
    data = {'avg_num': average_list, 'mean': all_mean, 'std':all_err}
    df = pd.DataFrame(data)
    df.to_excel(f'./mean_acc/{sub_name}_acc_mean_std.xlsx', index=False)

    # with open(f'./mean_acc/{sub_name}_acc_mean_std.csv', 'w') as f:
    #     [f.write('{0},{1},{2}\n'.format(avg, mean, err)) for avg, mean, err in zip(average_list, all_mean, all_err)]
    # plt.plot(list(np.arange(len(average_list))), all_s, 'o-', color = color, label=label)
    # axs.fill_between(list(np.arange(len(average_list))), np.array(all_s) - np.array(all_err), np.array(all_s) + np.array(all_err), alpha=0.2)
    # # if '0916' in filename:
    # axs.set_xticks(np.arange(len(average_list)))
    # axs.set_xticklabels(average_list)
    # axs.set_xlabel('Average traces', fontname='Arial', fontsize=18, fontweight='bold')
    # axs.set_ylabel('Accuracy', fontname='Arial', fontsize=18, fontweight='bold')
    # threshold1 = 1.0
    # axs.axhline(threshold1, color='grey', lw=1, alpha=0.7, linestyle ='--')
    # threshold2 = 1/24
    # axs.axhline(threshold2, color='grey', lw=1, alpha=0.7, linestyle ='--')
    # # axs.axvline(len(average_list)-1, color='tab:grey', lw=1, alpha=0.7, linestyle ='--')
    
    # axs.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))