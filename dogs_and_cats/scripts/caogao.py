# -*- coding: utf-8 -*-
# @Time    : 18-11-9 上午12:20
# @Author  : thuzjf
# @File    : caogao.py
# @Software: PyCharm

from helpers import Ploter
logfile = '../output/log.txt'
html = '../output/loss.html'

ploter = Ploter(logfile, html, ['loss'])
ploter.plot()