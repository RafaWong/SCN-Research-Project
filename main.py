import torch
from utils import *
from model import *
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score

# config
num_epochs = 160
learning_rate = 0.001
# num_edges = 0

def train(model, optimizer, train_dataloader, valid_dataloader, test_dataloader, device, epoch):
    model.train()
    loss = 0
    optimizer.zero_grad()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_epoch = []
    val_min = 1000
    best_model = model

    for i, batch in enumerate(train_dataloader):
        x = batch[0].squeeze().to(device)
        y = batch[1].to(device)
        # edge = torch.unique(batch[2].squeeze(),dim=1)
        edge = batch[2].squeeze().to(device)
        random_num = np.random.uniform()
        if random_num>0.5 and random_num < 0.75:
            edge = edge[:,:87]
            x = x[:88,:]
        elif random_num> 0.75:
            edge = edge[:,87:]-87
            x = x[87:,:]
        # print(x,y,edge)
        # exit()
        optimizer.zero_grad()
        pred = model(edge, x)
        # print(pred,y)
        # exit()
        loss = criterion(pred, y)
        loss_epoch.append(loss.item())
        loss.backward()
        optimizer.step()
    
    mean_loss = np.mean(loss_epoch)
    # scheduler.step()
    print(f'epoch {epoch + 1} meanloss: {mean_loss}')

def test(model, device, test_loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    test_loss = 0.0 
    acc = 0 
    loss_values = []
    pred_scores = []
    true_scores = []
    for i, batch in enumerate(test_loader):
        x = batch[0].squeeze().to(device)
        label = batch[1].to(device)
        edge = batch[2].squeeze().to(device)
        with torch.no_grad():
            y = model(edge, x)
        loss = criterion(y, label)
        loss_values.append(loss.item())
        # print(y.max(dim = 1)[1])
        pred = y.max(dim = 1)[1]
        pred_scores.append(pred.data.cpu().numpy())
        true_scores.append(label.data.cpu().numpy())

    pred_scores = np.concatenate(pred_scores)
    true_scores = np.concatenate(true_scores)
    mean_loss = np.mean(loss_values)
    # print(true_scores,pred_scores)
    overall_acc = accuracy_score(true_scores, pred_scores)
    avg_class_acc = balanced_accuracy_score(true_scores, pred_scores)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%,Avg_class_accuracy:{:.2f}%'.format(
        # mean_loss, overall_acc*100,avg_class_acc*100))
    return overall_acc,avg_class_acc

def predictbatch(model, device, pred_dataloader):
    model.eval()
    preds = []
    for i, batch in enumerate(pred_dataloader):
        x = batch[0].squeeze().to(device)
        label = batch[1].to(device)
        edge = batch[2].squeeze().to(device)
        with torch.no_grad():
            y = model(edge, x)
        # print(y.max(dim = 1)[1])
        pred = y.max(dim = 1)[1]
        preds.append(pred.data.cpu().numpy())
    return preds

def main():
    # DEVICE = 'cpu'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    data_path = './data'
    train_dataloader, valid_dataloader, test_dataloader = get_dataset(data_path)
    model = MultiLayerGCN(num_classes=6).to(DEVICE)
    print('-----model:-----')
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4, lr=learning_rate)
    best_acc = 0.0
    best_epoch = 0
    best_avg_class_acc = 0.0
    for epoch in range(num_epochs):
        if epoch > 0 and epoch % 20 == 0:
            optimizer.param_groups[0]['lr'] *= 0.75
        train(model, optimizer, train_dataloader, valid_dataloader, test_dataloader, DEVICE, epoch)
        acc,avg_class_acc = test(model, DEVICE, valid_dataloader)
        if best_acc < acc: 
            best_acc = acc 
            best_epoch = epoch
            torch.save(model.state_dict(), f'./result/best_scn.pt')
        if best_avg_class_acc < avg_class_acc:
            best_avg_class_acc = avg_class_acc
        print("acc is: {:.4f}, best acc is {:.4f} in epoch {}, best avg_class_acc is {:.4f}\n".format(acc, best_acc, best_epoch + 1, best_avg_class_acc))
        # if epoch >= 100 and best_epoch * 1.0 < epoch * 0.5:
        #     break
    torch.load(f'./result/best_scn.pt')
    model.load_state_dict(torch.load(f'./result/best_scn.pt'))
    model.eval()
    acc,avg_class_acc = test(model, DEVICE, test_dataloader)
    print("Test:acc is: {:.4f}, avg_class_acc is {:.4f}\n".format(acc, avg_class_acc))
    # print(labels)








if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    main()
