# -*- coding: utf-8 -*-
# @Time    : 18-11-6 上午12:02
# @Author  : thuzjf
# @File    : network.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.init as init


class InBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(InBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.relu(self.bn(self.conv(x)))
    return x


class DownBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DownBlock, self).__init__()
    self.down_conv = nn.Conv2d(in_channels, out_channels, 2, 2, bias=False)
    self.resblock = ResBlock(out_channels, 3)

  def forward(self, x):
    x = self.down_conv(x)
    x = x + self.resblock(x)
    return x


class ResBlock(nn.Module):
  def __init__(self, in_channels, num_convs):
    super(ResBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
    self.bn = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU(inplace=True)
    models = nn.Sequential(self.conv, self.bn, self.relu)
    for i in range(num_convs - 1):
      models.add_module('conv{}'.format(i + 1), self.conv)
      models.add_module('bn{}'.format(i + 1), self.bn)
      models.add_module('relu{}'.format(i + 1), self.relu)

    self.module = nn.Sequential(models)

  def forward(self, x):
    x = x + self.module(x)
    x = self.relu(self.bn(x))
    return x


class UpBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UpBlock, self).__init__()
    self.up_conv = nn.ConvTranspose2d(in_channels, out_channels // 2, 2, 2, bias=False)
    self.resblock = ResBlock(out_channels, 3)

  def forward(self, x, skip_x):
    x = self.up_conv(x)
    x = torch.cat((x, skip_x), 1)
    x = x + self.resblock(x)
    return x


class OutBlock(nn.Module):
  def __init__(self, in_channels, num_classes, resize_num):
    super(OutBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
    self.bn = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv_fc = nn.Conv2d(in_channels, num_classes, resize_num)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.relu(self.bn(self.conv(x)))
    x = self.softmax(self.conv_fc(x))
    return x.squeeze()


class VClassNet(nn.Module):
  def __init__(self, in_channels, num_classes, resize_num):
    super(VClassNet, self).__init__()
    self.inblock = InBlock(in_channels, 4)
    self.down32 = DownBlock(4, 8)
    self.down64 = DownBlock(8, 16)
    self.down128 = DownBlock(16, 32)
    self.up128 = UpBlock(32, 32)
    self.up64 = UpBlock(32, 16)
    self.up32 = UpBlock(16, 8)
    self.outblock = OutBlock(8, num_classes, resize_num)

  def forward(self, x):
    x16 = self.inblock(x)
    x32 = self.down32(x16)
    x64 = self.down64(x32)
    x = self.down128(x64)
    x = self.up128(x, x64)
    x = self.up64(x, x32)
    x = self.up32(x, x16)
    x = self.outblock(x)
    return x

  def max_stride(self):
    return 8


if __name__ == "__main__":
  resize_num = 128
  net = VClassNet(3, 2, resize_num)
  print net
  param_num = sum(p.numel() for p in net.parameters())
  print "there are {} Mb parameters.".format(1.0 / 1e6 * param_num)

  a = torch.ones([2, 3, resize_num, resize_num])
  a = a.cuda()
  net = net.cuda()
  b = net(a)
  print b.size()
  print b