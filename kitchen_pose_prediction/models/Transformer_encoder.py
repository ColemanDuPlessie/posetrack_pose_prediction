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
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs) 
        self.pe = self.pe.to(*args, **kwargs) 
        return self

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    
    def __init__(self, hidden_size = 64, heads = 4, frame_dimension = 1, layers = 2, positional_embedding_max_len = 2048, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = frame_dimension
        self.encoding = nn.Linear(frame_dimension, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_len = positional_embedding_max_len, dropout = dropout)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden_size, heads, hidden_size, batch_first=True, dropout = dropout), layers)
        
    def forward(self, y, pre_output_len=1):
        frames = self.positional_encoding(self.encoding(y))
        mask = generate_square_subsequent_mask(frames.size()[1])
        ans = self.transformer(frames, mask)[:, pre_output_len:]
        ans = torch.matmul(ans-self.encoding.bias, torch.linalg.pinv(self.encoding.weight).t()) # This line effectively runs the self.encoding layer in reverse # TODO It may or may not be making this model significantly worse...

        return ans
    
class TransformerEncoderMultistep(TransformerEncoder):
    def __init__(self, hidden_size = 64, heads = 4, frame_dimension = 1, layers = 2, positional_embedding_max_len = 2048, dropout=0.0):
        super(TransformerEncoderMultistep, self).__init__(hidden_size, heads, frame_dimension, layers, positional_embedding_max_len, dropout)
    
    def forward_multistep(self, y, pre_output_len=1, steps = 2):
        ans = self.forward(y, pre_output_len)
        for step in range(1, steps):
            for frame_idx in range(ans.shape[1]):
                print(self.forward(torch.cat((y[:,:pre_output_len+frame_idx], ans[:,frame_idx:frame_idx+1]), dim=1), pre_output_len+frame_idx).shape)
                ans[:, frame_idx] = self.forward(torch.cat((y[:,:pre_output_len+frame_idx], ans[:,frame_idx:frame_idx+1]), dim=1), pre_output_len+frame_idx).squeeze(1)
        return ans      