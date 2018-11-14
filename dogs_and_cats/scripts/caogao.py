# -*- coding: utf-8 -*-
# @Time    : 18-11-9 上午12:20
# @Author  : thuzjf
# @File    : caogao.py
# @Software: PyCharm

from helpers import Ploter
logfile = '/media/cnzjf/data/git_work/kaggle/dogs_and_cats/output/log.txt'
html = '/media/cnzjf/data/git_work/kaggle/dogs_and_cats/output/loss.html'

ploter = Ploter(logfile, html, ['loss'])
ploter.plot()