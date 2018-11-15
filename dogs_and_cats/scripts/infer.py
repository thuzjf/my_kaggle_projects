# -*- coding: utf-8 -*-
# @Time    : 18-11-13 下午10:54
# @Author  : thuzjf
# @File    : infer.py
# @Software: PyCharm
import cv2
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from my_utils_for_dl.helpers import load_checkpoint
from network import VClassNet


im_dir = "../data/debug_dataset"
filelist = [f for f in os.listdir(im_dir) if '.jpg' in f]

model_dir = '../output/checkpoints'
resize_num = 128
epoch = 49

net = VClassNet(3, 2, resize_num)
net = net.cuda()
model = nn.DataParallel(net, device_ids=[0])
model = load_checkpoint(model, epoch, model_dir)

pred_labels = []
for image_file in filelist:
  im_path = os.path.join(im_dir, image_file)
  im = cv2.imread(im_path)
  height, width, _ = im.shape
  if height >= width:
    new_width = int(1.0 * resize_num * width / height + 0.5)
    if new_width % 2 == 1:
      new_width -= 1
    # 注意cv2.resize 的size是【宽，高，通道】
    im = cv2.resize(im, (new_width, resize_num), interpolation=cv2.INTER_LINEAR)
    pad = np.zeros([resize_num, (resize_num - new_width) / 2, 3])
    im = np.concatenate((pad, im, pad), axis=1)
  else:
    new_height = int(1.0 * resize_num * height / width + 0.5)
    if new_height % 2 == 1:
      new_height -= 1
    im = cv2.resize(im, (resize_num, new_height), interpolation=cv2.INTER_LINEAR)
    pad = np.zeros([(resize_num - new_height) / 2, resize_num, 3])
    im = np.concatenate((pad, im, pad), axis=0)

  flip = np.random.randint(0, 1)
  if flip:
    im = np.fliplr(im)

  im_tensor = torch.tensor(im)
  im_tensor.transpose_(0, 2)
  im_tensor.unsqueeze_(0)
  im_tensor = im_tensor.float().cuda()

  with torch.no_grad():
    pred = model(im_tensor)
    pred_label = pred.argmax().item()
  pred_labels.append(pred_label)

dataframe = pd.DataFrame(columns=['id', 'label'])
dataframe.id = range(1, len(filelist) + 1)
dataframe.label = pred_labels
dataframe.to_csv('../output/submission.csv', index=False)





