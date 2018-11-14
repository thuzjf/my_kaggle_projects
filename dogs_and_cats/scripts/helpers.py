# -*- coding: utf-8 -*-
# @Time    : 18-11-11 下午11:16
# @Author  : thuzjf
# @File    : helpers.py
# @Software: PyCharm
import logging
import os
import plotly.offline as pyo
import plotly.graph_objs as go
import torch


def save_checkpoint(model, epoch, save_dir):
  fname = "ckpt_{:0>4d}.pth".format(epoch)
  fpath = os.path.join(save_dir, fname)
  state_dict = model.state_dict()
  model_dict = dict()
  model_dict['state_dict'] = state_dict
  model_dict['epoch_num'] = epoch
  torch.save(model_dict, fpath)

def load_checkpoint(model, epoch, save_dir):
  fname = "ckpt_{:0>4d}.pth".format(epoch)
  fpath = os.path.join(save_dir, fname)
  model_dict = torch.load(fpath)
  model.load_state_dict(model_dict['state_dict'])
  return model

def resume_training(model, save_dir):
  checkpoints = [f for f in os.listdir(save_dir) if '.pth' in f]
  if checkpoints:
    checkpoints.sort()
    last_checkpoint = os.path.join(save_dir, checkpoints[-1])
    model_dict = torch.load(last_checkpoint)
    model.load_state_dict(model_dict['state_dict'])
    start_epoch = model_dict['epoch_num'] + 1
  else:
    start_epoch = 0
  return model, start_epoch


def setup_logger(logfile, CMDH=True):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

  file_handle = logging.FileHandler(logfile, mode='a')
  file_handle.setFormatter(formatter)
  logger.addHandler(file_handle)

  if CMDH:
    cmd_handle = logging.StreamHandler()
    cmd_handle.setFormatter(formatter)
    logger.addHandler(cmd_handle)

  return logger


class Ploter(object):
  def __init__(self, logfile, html_dir, keywords, var_x ='batch_num'):
    super(Ploter, self).__init__()
    with open(logfile) as f:
      self._texts = f.readlines()
    self.html_file = html_dir
    self._info_dict = {'x': []}
    self.keywords = keywords
    self._var_x = var_x
    for keyword in self.keywords:
      self._info_dict.update({keyword: []})

    self.keywords.append(var_x)
    for line in self._texts:
      self._extract_info(line)
    self.keywords.remove(self._var_x)


  def _extract_info(self, line):
    for keyword in self.keywords:
      if keyword not in line:
        continue
      begin = line.find(keyword) + len(keyword) + 1
      end = begin + line[begin:].find(',')
      value = float(line[begin:end])
      if keyword == self._var_x:
        keyword = 'x'
      self._info_dict[keyword].append(value)

  def plot(self, keyword_indexes):
    assert isinstance(keyword_indexes, list), "keyword_indexes must be list!"
    traces = []
    filename = ""
    for idx in keyword_indexes:
      keyword = self.keywords[idx]
      filename += keyword + '_'
      trace = go.Scatter(
        name = keyword,
        x=self._info_dict['x'],
        y=self._info_dict[keyword])
      traces.append(trace)

    layout = go.Layout(showlegend=True)
    fig = go.Figure(data=traces, layout=layout)
    filename = os.path.join(self.html_file, filename + '.html')
    pyo.plot(fig, filename=filename, auto_open=False)