# -*- coding: utf-8 -*-
# @Time    : 18-11-14 下午11:53
# @Author  : thuzjf
# @File    : data_provider.py
# @Software: PyCharm

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import mnist as mnist_dset

class MyDataset(Dataset):
  def __init__(self, data_dir, is_training):
    super(MyDataset, self).__init__()
    self.is_training = is_training
    if is_training:
      data_path = os.path.join(data_dir, "train-images.idx3-ubyte")
      label_path = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    else:
      data_path = os.path.join(data_dir, "t10k-images.idx3-ubyte")
      label_path = os.path.join(data_dir, 't10k-labels.idx1-ubyte')
    self.data = mnist_dset.read_image_file(data_path).float()
    self.data = self.data / 255.0
    self.data = self.data.unsqueeze_(dim=1)
    self.labels = mnist_dset.read_label_file(label_path).long()
    

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):

    if self.is_training:
      return self.data[idx], self.labels[idx]
    else:
      return self.data[idx]


