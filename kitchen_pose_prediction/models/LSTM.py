# -*- coding: utf-8 -*-
"""
This needs documentation at some point.
"""

import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, hidden_size=64, input_size=1, output_size=1, layers=4, nonlinearity="relu"):
        super(LSTM, self)
        assert layers >= 2
        self.layer = nn.LSTM(input_size, hidden_size, layers, batch_first=True, dropout=0.1, proj_size=input_size)
        
    def forward(self, y, min_seq_len=1):
        return self.layer(y)[:, min_seq_len:]

# =============================================================================
# class TwoLayerLSTM(nn.Module):
#     def __init__(self, hidden_layers=64, frame_dimension = 1):
#         super(TwoLayerLSTM, self).__init__()
#         self.hidden_layers = hidden_layers
#         self.input_size = frame_dimension
#         # lstm1, lstm2, linear are all layers in the network
#         self.lstm1 = nn.LSTMCell(frame_dimension, self.hidden_layers)
#         self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
#         self.linear = nn.Linear(self.hidden_layers, frame_dimension)
#         
#     def forward(self, y, pre_output_len=1):
#         outputs = []
#         h_t = torch.zeros(1, self.hidden_layers, dtype=torch.float32)
#         c_t = torch.zeros(1, self.hidden_layers, dtype=torch.float32)
#         h_t2 = torch.zeros(1, self.hidden_layers, dtype=torch.float32)
#         c_t2 = torch.zeros(1, self.hidden_layers, dtype=torch.float32)
#         
#         for i, frame in enumerate(y.split(1, dim=0)):
#             h_t, c_t = self.lstm1(frame, (h_t, c_t)) # initial hidden and cell states
#             h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
#             output = self.linear(h_t2) # output from the last FC layer
#             if i >= pre_output_len: outputs.append(output)
# 
#         # transform list to tensor    
#         outputs = torch.stack(outputs)
#         return outputs
# =============================================================================
