# -*- coding: utf-8 -*-
"""
This needs documentation at some point.
"""

import math
import torch
import torch.nn as nn

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class PositionalEncoding(nn.Module):
    """
    This class is blatantly stolen in its entirety from the PyTorch docs,
    specifically the following URL:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TwoLayerLSTM(nn.Module):
    def __init__(self, hidden_layers=64, frame_dimension = 1):
        super(TwoLayerLSTM, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_size = frame_dimension
        # lstm1, lstm2, linear are all layers in the network
        self.lstm1 = nn.LSTMCell(frame_dimension, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, frame_dimension)
        
    def forward(self, y, pre_output_len=1):
        outputs = []
        h_t = torch.zeros(1, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(1, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(1, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(1, self.hidden_layers, dtype=torch.float32)
        
        for i, frame in enumerate(y.split(1, dim=0)):
            h_t, c_t = self.lstm1(frame, (h_t, c_t)) # initial hidden and cell states
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # new hidden and cell states
            output = self.linear(h_t2) # output from the last FC layer
            if i >= pre_output_len: outputs.append(output)

        # transform list to tensor    
        outputs = torch.stack(outputs)
        return outputs

class TransformerEncoder(nn.Module):
    
    def __init__(self, hidden_size = 64, heads = 4, frame_dimension = 1, layers = 2, positional_embedding_max_len = 2048):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = frame_dimension
        self.encoding = nn.Linear(frame_dimension, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_len = positional_embedding_max_len)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, heads, hidden_size, batch_first=True), layers)
        
    def forward(self, y, pre_output_len=1):
        frames = self.positional_encoding(self.encoding(y))
        mask = generate_square_subsequent_mask(frames.size()[1])
        ans = self.transformer(frames, mask)[:, pre_output_len:]
        ans = torch.matmul(ans-self.encoding.bias, torch.linalg.pinv(self.encoding.weight).t()) # This line effectively runs the self.encoding layer in reverse # TODO It may or may not be making this model significantly worse...

        return ans