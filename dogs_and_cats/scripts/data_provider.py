# -*- coding: utf-8 -*-
# @Time    : 18-11-6 上午12:03
# @Author  : thuzjf
# @File    : data_provider.py
# @Software: PyCharm
import cv2
import numpy as np
import os
import torch

from torch.utils.data import Dataset


class MyDataset(Dataset):

  def __init__(self, image_dir, resize_num):
    self.image_dir = image_dir
    self.resize_num = resize_num
    self.image_list = [f for f in os.listdir(self.image_dir) if '.jpg' in f]
    self.data_length = len(self.image_list)
    self.data = []
    for image_file in self.image_list:
      im_path = os.path.join(self.image_dir, image_file)
      self.data.append(im_path)


  def __len__(self):
    return self.data_length


  def __getitem__(self, idx):
    im_path = self.data[idx]
    im = cv2.imread(im_path)
    height, width, _= im.shape
    if height >= width:
      new_width = int(1.0 * self.resize_num * width / height + 0.5)
      if new_width % 2 == 1:
        new_width -= 1
      # 注意cv2.resize 的size是【宽，高，通道】
      im = cv2.resize(im, (new_width, self.resize_num), interpolation=cv2.INTER_LINEAR)
      pad = np.zeros([self.resize_num, (self.resize_num - new_width) / 2, 3])
      im = np.concatenate((pad, im, pad), axis=1)
    else:
      new_height = int(1.0 * self.resize_num * height / width + 0.5)
      if new_height % 2 == 1:
        new_height -= 1
      im = cv2.resize(im, (self.resize_num, new_height), interpolation=cv2.INTER_LINEAR)
      pad = np.zeros([ (self.resize_num - new_height) / 2, self.resize_num, 3])
      im = np.concatenate((pad, im, pad), axis=0)

    flip = np.random.randint(0,1)
    if flip:
      im = np.fliplr(im)

    im_tensor = torch.tensor(im)
    im_tensor.transpose_(0, 2)
    image_file = self.image_list[idx]
    label = 0
    if 'dog' in image_file:
      label = 1

    return im_tensor.float(), label







