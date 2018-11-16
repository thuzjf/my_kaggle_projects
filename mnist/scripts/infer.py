# -*- coding: utf-8 -*-
# @Time    : 18-11-16 上午12:33
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
from network import Mnistnet
from data_provider import MyDataset
from torch.utils.data import DataLoader


im_dir = "../data"
model_dir = '../output/checkpoints'

epoch = 133

net = Mnistnet()
net = net.cuda()
net.eval()
model = nn.DataParallel(net, device_ids=[0])
model = load_checkpoint(model, epoch, model_dir)

testing_set = MyDataset(im_dir, False)
testing_loader = DataLoader(testing_set, batch_size=1000)
testing_iterator = iter(testing_loader)

pred_labels = []
for idx in range(len(testing_iterator)):
  im_tensor = testing_iterator.next()
  im_tensor = im_tensor.cuda()
  with torch.no_grad():
    pred = model(im_tensor)
    pred_label = pred.argmax(dim=1)
    pred_label = pred_label.tolist()
  pred_labels.extend(pred_label)


dataframe = pd.DataFrame(columns=['ImageId', 'Label'])
dataframe.ImageId = range(1, len(pred_labels) + 1)
dataframe.Label = pred_labels
dataframe.to_csv('../output/submission.csv', index=False)


