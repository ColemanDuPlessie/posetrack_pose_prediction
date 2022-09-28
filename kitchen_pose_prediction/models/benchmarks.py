# -*- coding: utf-8 -*-
"""
This needs documentation at some point.
"""

import torch.nn as nn

class SimpleRepeater(nn.Module):
    """
    This "model" just assumes that the next position will be the exact same as
    the previous position. For testing purposes only!
    """
    def __init__(self, input_size = 1):
        super(SimpleRepeater, self).__init__()
        self.input_size = 1
        
    def forward(self, y, pre_output_len=1):
        return y[:, pre_output_len:, :]
    
    def forward_multistep(self, y, pre_output_len=1, steps=1):
        return self.forward(y, pre_output_len)