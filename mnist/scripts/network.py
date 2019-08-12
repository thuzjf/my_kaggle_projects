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
      nn.Conv2d(1, 64, 3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, 3),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      
      nn.Conv2d(64, 128, 3),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, 3),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2, 2),
      
      nn.Conv2d(128, 256, 3),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.Conv2d(256, 256, 2),
      nn.BatchNorm2d(256),
      nn.ReLU())
    
    self.fc = nn.Sequential(
      nn.Linear(256, 128),
      nn.Dropout(0.2),
      nn.ReLU(),
      nn.Linear(128, 10))
    
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x):
    x = self.convblock(x)
    x = x.permute(0, 2, 3, 1)
    x = x.view(-1, x.size(-1))
    x = self.fc(x)
    x = self.softmax(x)
    return x


class Mnistlstm(nn.Module):
  def __init__(self):
    super(Mnistlstm, self).__init__()
    self.lstm = nn.LSTM(28, 128, 2, bias=True, batch_first=True)
    self.fc = nn.Sequential(
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 10))
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, x):
    x = x.squeeze_(1)
    x, _ = self.lstm(x)
    x = self.fc(x[:, -1, :])
    x = self.softmax(x)
    return x


if __name__ == "__main__":
  net = Mnistnet()
  print(net)
  param_num = sum(p.numel() for p in net.parameters())
  print("there are {:.2f} Mb parameters.".format(1.0 / 1e6 * param_num))
  
  a = torch.ones([2, 1, 28, 28])
  b = net(a)
  print(b.size())
  
  print("\n**********************************\n")
  net2 = Mnistlstm()
  print(net2)
  param_num = sum(p.numel() for p in net2.parameters())
  print("there are {:.2f} Mb parameters.".format(1.0 / 1e6 * param_num))
  
  c = net2(a)
  print(c.size())
  
  print(b)
  print(c)