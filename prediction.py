import torch
from utils import *
from model import *
import os
import random
import numpy as np
import scipy.io as scio

def predictbatch(model, device, pred_dataloader):
    model.eval()
    preds = []
    # trues = []
    for i, batch in enumerate(pred_dataloader):
        x = batch[0].squeeze().to(device)
        # label = batch[1].to(device)
        edge = batch[2].squeeze().to(device)
        with torch.no_grad():
            y = model(edge, x)
        # print(y.max(dim = 1)[1])
        pred = y.max(dim = 1)[1]
        preds.append(pred.data.cpu().numpy())
        # trues.append(label.data.cpu().numpy())
    return preds

def main():
    # DEVICE = 'cpu'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    data_path = './data_pred'
    pred_dataloader = pred_dataset(data_path)
    model = MultiLayerGCN(num_classes=6).to(DEVICE)
    print('-----model:-----')
    print(model)
    model.load_state_dict(torch.load(f'./result/best_scn.pt'))
    model.eval()
    preds = predictbatch(model, DEVICE, pred_dataloader)
    out_path = './result_pred'
    scio.savemat(f'{out_path}/'+'predictlist.mat', {'predictlist':preds})
    # scio.savemat(f'{out_path}/'+'truelist.mat', {'truelist':trues})
    
if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    main()