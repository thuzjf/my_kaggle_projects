# -*- coding: utf-8 -*-
# @Time    : 18-11-14 下午11:53
# @Author  : thuzjf
# @File    : data_provider.py
# @Software: PyCharm

import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
  def __init__(self, data_dir, is_training):
    super(MyDataset, self).__init__()
    self.data = []
    self.labels = []
    self.is_training = is_training
    if is_training:
      csv_path = os.path.join(data_dir, "train.csv")
    else:
      csv_path = os.path.join(data_dir, "test.csv")
    data_frame = pd.read_csv(csv_path)
    for idx, row in data_frame.iterrows():
      if is_training:
        label = row[0]
        data = np.array(row[1:]).reshape([28, 28])
        self.labels.append(label)
      else:
        data = np.array(row).reshape([28, 28])
      self.data.append(data)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image_tensor = torch.tensor(self.data[idx])
    image_tensor.unsqueeze_(0)
    if self.is_training:
      return image_tensor.float(), self.labels[idx]
    else:
      return image_tensor.float()


