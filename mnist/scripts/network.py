# -*- coding: utf-8 -*-
# @Time    : 18-11-15 下午11:05
# @Author  : thuzjf
# @File    : network.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Time    : 18-11-6 上午12:02
# @Author  : thuzjf
# @File    : network.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.init as init


class Mnistnet(nn.Module):
  def __init__(self):
    super(Mnistnet, self).__init__()
    self.convblock = nn.Sequential(
      nn.Conv2d(1, 32, 3),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 32, 3),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),

      nn.Conv2d(32, 64, 3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, 3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),

      nn.Conv2d(64, 128, 3),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, 2),
      nn.BatchNorm2d(128),
      nn.ReLU())

    self.fc = nn.Sequential(
      nn.Linear(128, 100),
      nn.Linear(100, 10))

    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.convblock(x)
    x = x.view(-1, x.size(1))
    x = self.fc(x)
    x = self.softmax(x)
    return x


if __name__ == "__main__":

  net = Mnistnet()
  print net
  param_num = sum(p.numel() for p in net.parameters())
  print "there are {} Mb parameters.".format(1.0 / 1e6 * param_num)

  a = torch.ones([2, 1, 28, 28])
  a = a.cuda()
  net = net.cuda()
  b = net(a)
  print b.size()
  print b