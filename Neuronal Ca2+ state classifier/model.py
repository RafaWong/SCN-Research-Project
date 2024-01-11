from torch_geometric.nn import GCNConv, SGConv
import torch
import torch.nn as nn
import numpy as np
import torch_geometric.nn as pyg_nn
from collections import OrderedDict
import torch.nn.functional as F

class GlobalPooling(torch.nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        x = x.T.unsqueeze(0)
        batch_size = x.size(0)
        x0 = self.max_pool(x).view(batch_size, -1)
        x1 = self.avg_pool(x).view(batch_size, -1)
        x = torch.cat((x0, x1), dim=-1)
        # x = x.squeeze(0)
        return x

class GlobalMaxPooling(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()
        self.max_pool = torch.nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.max_pool(x).view(batch_size, -1)
        return x

class MultiLayerGCN(nn.Module):
    def __init__(self, dropout=0.5, num_classes=40):
        super(MultiLayerGCN, self).__init__()
        self.conv0 = GCNConv(3, 32)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        # self.conv3 = GCNConv(128, 256, bias=False)
        # self.conv4 = GCNConv(512, 1024, bias=False)
        # self.bn0 = nn.BatchNorm1d(32)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.bn4 = nn.BatchNorm1d(1024)
        self.pool = GlobalPooling()
        self.classifier = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(32 * 3 * 2, 16, bias=False)),
            ('relu0', nn.LeakyReLU(negative_slope=0.2)),
            # ('bn0', nn.BatchNorm1d(16)),
            ('drop0', nn.Dropout(p=dropout)),
            # ('fc1', nn.Linear(512, 256, bias=False)),
            # ('relu1', nn.LeakyReLU(negative_slope=0.2)),
            # ('bn1', nn.BatchNorm1d(256)),
            # ('drop1', nn.Dropout(p=dropout)),
            ('fc2', nn.Linear(16, num_classes)),
        ]))

    def forward(self, adj, x):
        x0 = F.leaky_relu(self.conv0(x, adj), negative_slope=0.2)
        x1 = F.leaky_relu(self.conv1(x0, adj), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1, adj), negative_slope=0.2)
        # x3 = F.leaky_relu(self.bn3(self.conv3(adj, x2)), negative_slope=0.2)
        x = torch.cat((x0, x1, x2), dim=1)
        
        # x = F.leaky_relu(self.bn4(self.conv4(adj, x)), negative_slope=0.2)
        x = self.pool(x)
        # print(x.shape)
        x = self.classifier(x)
        return x