# -*- coding: utf-8 -*-
"""
This needs documentation at some point.
"""

import torch
import torch.nn as nn

class LSTMBenchmark(nn.Module):
    def __init__(self, hidden_size=64, input_size=1, output_size=1, layers=4, dropout=0.0):
        super(LSTMBenchmark, self).__init__()
        assert layers >= 2
        self.layer = nn.LSTM(input_size, hidden_size, layers, batch_first=True, dropout=dropout, proj_size=input_size)
        
    def forward(self, y, min_seq_len=1):
        return self.layer(y)[0][:, min_seq_len:]

class LSTMMultistep(LSTMBenchmark):
    def __init__(self, hidden_size=64, input_size=1, output_size=1, layers=4, dropout=0.0):
        super(LSTMMultistep, self).__init__(hidden_size, input_size, output_size, layers, dropout)
        
    def forward_multistep(self, y, min_seq_len=1, steps=2):
        assert steps > 1
        ans = []
        other_inputs = None
        for step in range(len(y.shape[1])):
            out, other_inputs = self.layer(y[:,:step], other_inputs)
            if step >= min_seq_len:
                temp_other_inputs = other_inputs
                for forward_step in range(steps-1):
                    frame, temp_other_inputs = self.layer(out, temp_other_inputs)
                    out = torch.cat((out, frame), 1)
                ans.append(out[:,-1])
        return torch.stack(ans, 1)
