# -*- coding: utf-8 -*-
# @Time    : 18-11-15 下午10:36
# @Author  : thuzjf
# @File    : setup.py
# @Software: PyCharm

from setuptools import setup

setup(
  name='MyUtilsForDL',  # 包名字
  version='1.0',  # 包版本
  description='My base functions and classes for dl projects to import',  # 简单描述
  author='thuzjf',  # 作者
  author_email='chinazjf@vip.qq.com',  # 作者邮箱
  url='https://thuzjf.github.io',  # 包的主页
  packages=['my_utils_for_dl'],  # 需要打包的文件夹
  install_requires=['torch>=0.4.0',])
