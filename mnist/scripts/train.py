# -*- coding: utf-8 -*-
# @Time    : 18-11-15 下午10:36
# @Author  : thuzjf
# @File    : train.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from my_utils_for_dl.losses import Focalloss
from data_provider import MyDataset
from my_utils_for_dl.helpers import save_checkpoint, setup_logger, resume_training, Ploter
from network import Mnistnet


if __name__ == "__main__":
  im_dir = '../data'
  out_dir = '../output2'

  html_dir = os.path.join(out_dir, 'htmls')
  if not os.path.isdir(html_dir):
    os.makedirs(html_dir)

  checkpoint_dir = os.path.join(out_dir, 'checkpoints')
  if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  logfile = os.path.join(out_dir, 'log.txt')

  training_set = MyDataset(im_dir, True)
  training_set_loader = DataLoader(training_set,
                                   batch_size=500,
                                   shuffle=True)

  net = Mnistnet()
  net = net.cuda()
  model = nn.DataParallel(net, device_ids=[0])
  model.train()

  criterion = Focalloss(num_class=10, size_average=False)
  optimizer = Adam(model.parameters())

  model, start_epoch = resume_training(model, checkpoint_dir)
  if start_epoch == 0:
    print "no checkpoints found, starting training from scratch!"
    if os.path.isfile(logfile):
      os.remove(logfile)
  logger = setup_logger(logfile)

  sum_error_rate = 0
  current_epoch_error_rate = 0
  for epoch in range(start_epoch, 500):
    epoch_sum_error_rate = 0
    train_iterator = iter(training_set_loader)
    for batch_idx in range(len(train_iterator)):
      image_tensor, label = train_iterator.next()
      image_tensor, label = image_tensor.cuda(), label.cuda()

      optimizer.zero_grad()
      pred = model(image_tensor)
      loss = criterion(pred, label)
      loss.backward()

      pred_label = torch.argmax(pred, dim=1)
      error_mask = pred_label != label
      error_rate = 1.0 * error_mask.sum().item() / label.size(0)
      sum_error_rate += error_rate
      epoch_sum_error_rate += error_rate

      current_batch_num = epoch * len(train_iterator) + batch_idx + 1
      average_error_rate = sum_error_rate / current_batch_num

      msg = "epoch_num:{:0>4d}, batch_num:{:0>8d}, loss:{:.5f}, error_rate:{:.3f}, average_error_rate:{:.3f}, epoch_error_rate:{:.3f},".format(
        epoch, current_batch_num, loss.item(), error_rate, average_error_rate, current_epoch_error_rate)
      logger.info(msg)
      optimizer.step()

    save_checkpoint(model, epoch, checkpoint_dir)
    ploter = Ploter(logfile, html_dir, keywords=['loss', 'average_error_rate', 'epoch_error_rate'])
    ploter.plot([0])
    ploter.plot([1])
    ploter.plot([2])

    current_epoch_error_rate = epoch_sum_error_rate / len(train_iterator)
