import torch
import torch.nn as nn


class resBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3):
        super(resBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, inplanes, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.bn2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out
    
class CNN(nn.Module):
    def __init__(self, time_len, num_seq, num_class):
        super(CNN, self).__init__()
        self.time_len = time_len
        self.num_seq = num_seq
        self.num_class = num_class
        
        self.conv = nn.Sequential(*[
            nn.Conv1d(self.num_seq, 32, kernel_size=7, bias=True),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool1d(5),
            resBlock(32, 32, kernel_size=3),
            nn.MaxPool1d(5),
            nn.Flatten(start_dim=1, end_dim=-1),
        ])
        with torch.no_grad():
            x = torch.zeros(1, self.num_seq, self.time_len)
            x = self.conv(x)
            self.fc_dim = x.reshape(1, -1).shape[-1]
        self.fc = nn.Sequential(*[
            nn.Linear(self.fc_dim, self.num_class)
        ])
    
    def forward(self, x):
        return self.fc(self.conv(x))