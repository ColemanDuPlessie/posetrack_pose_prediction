# -*- coding: utf-8 -*-
"""
This needs documentation at some point
"""

import torch

class BatchManager(torch.utils.data.Dataset):
  def __init__(self, dirname, min_file_num=0, max_file_num=438):
      self.dirname = dirname
      self.min_file_num = min_file_num
      self.max_file_num = max_file_num

  def __len__(self):
      return self.max_file_num-self.min_file_num

  def __getitem__(self, index):
      assert index + self.min_file_num <= self.max_file_num
      return torch.load(self.dirname + "/batch" + str(index+self.min_file_num).zfill(4) + '.pt')