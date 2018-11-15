# -*- coding: utf-8 -*-
# @Time    : 18-11-9 上午12:57
# @Author  : thuzjf
# @File    : losses.py
# @Software: PyCharm
import torch
import torch.nn as nn

class Focalloss(nn.Module):
  def __init__(self, num_class, alpha = None, gamma = 2, size_average = True):
    super(Focalloss, self).__init__()
    if alpha is None:
      self.alpha = torch.ones(num_class, 1) / num_class
    else:
      assert len(alpha) == num_class
      self.alpha = torch.FloatTensor(alpha)
      self.alpha = self.alpha.unsqueeze(1)
      self.alpha = self.alpha / self.alpha.sum()
    self.alpha = self.alpha.cuda()
    self.num_class = num_class
    self.gamma = gamma
    self.size_average = size_average
    self.one_hot = torch.eye(self.num_class).cuda()

  def forward(self, pred, target):
    assert pred.dim() == 2, "the shape of pred must be [sample, num_class]"
    target = target.long().view(-1)

    mask = self.one_hot[target]
    alpha = self.alpha[target]

    probs = (pred * mask).sum(1).view(-1,1) + 1e-12
    log_probs = probs.log()

    if self.gamma > 0:
      batch_loss = -alpha * ((1 - probs) ** self.gamma) * log_probs
    else:
      batch_loss = -alpha * log_probs
    if self.size_average:
      loss = batch_loss.mean()
    else:
      loss = batch_loss.sum()

    return loss
