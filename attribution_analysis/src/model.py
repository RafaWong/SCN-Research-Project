import torch
import torch.nn as nn

class LSTM(torch.nn.Module):
    def __init__(self, c_in=6049, num_class=24, seq_len=200, hidden_size=32, 
                 rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.2, 
                 bidirectional=False, shuffle=False, fc_dropout=0.1):
        super(LSTM, self).__init__()
                    
        # RNN
        self.rnn = nn.LSTM(c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else nn.Identity()
                
        # Common
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else nn.Identity()
        self.fc = nn.Linear(hidden_size * (1 + bidirectional), num_class)
        

    def forward(self, x):  
        # RNN
        rnn_input = x.permute(0,2,1) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)
        x = self.fc_dropout(last_out)
        x = self.fc(x)
        return x

class CNN(nn.Module):
    def __init__(self, time_len=1, num_seq=6049, base_channel=32, num_class=24):
        super(CNN, self).__init__()
        self.time_len = time_len
        self.num_seq = num_seq
        self.num_class = num_class
        
        self.conv = nn.Sequential(*[
            nn.Linear(self.num_seq, base_channel),
            nn.LayerNorm(base_channel),
            nn.LeakyReLU(inplace=True),
            nn.Linear(base_channel, base_channel),
            nn.LayerNorm(base_channel),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(start_dim=1, end_dim=-1),
        ])
        with torch.no_grad():
            x = torch.zeros(1, self.num_seq)
            x = self.conv(x)
            self.fc_dim = x.reshape(1, -1).shape[-1]
            # print(self.fc_dim)
        self.fc = nn.Sequential(*[
            nn.Linear(self.fc_dim , self.num_class)
        ])
    
    def forward(self, x):
        return self.fc(self.conv(x))

# if __name__ == '__main__':
#     input_data = torch.rand(8, 6049)
#     model = CNN()
#     output = model(input_data)
#     print(output.shape)
