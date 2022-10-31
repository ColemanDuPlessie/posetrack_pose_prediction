# -*- coding: utf-8 -*-
"""
This file is just a wrapper. You should use it instead of model.py.
"""

import torch
from . import model

class Informer(model.Informer):
    
    def __init__(self, data_size, batch_size, 
                d_model, n_heads, e_layers, d_layers, dropout=0.0, 
                device=torch.device('cuda:0'), factor=5, attn='prob', embed='timeF',
                activation='gelu', output_attention = False, distil=True, mix=True):
        super(Informer, self).__init__(data_size, data_size, data_size,
                                       "These parameters do nothing", "and to prove it, I'm passing in random strings",
                                       batch_size, factor, d_model, n_heads, e_layers,
                                       d_layers, 4*d_model, dropout, attn, embed, data_size,
                                       activation, output_attention, distil, mix, device)
    
    def forward(self, y, pre_output_len = 1):
        enc_timesteps = torch.arange(0.0, 1.0, 1.0/y.shape[1]).unsqueeze(0).unsqueeze(2).expand(y.shape[0], -1, y.shape[2]).clone()
        dec_timesteps = enc_timesteps.clone()
        return super().forward(y, enc_timesteps, y, dec_timesteps)[:, pre_output_len:]
    
    def forward_multistep(self, y, pre_output_len = 1, steps = 2):
        pass # TODO