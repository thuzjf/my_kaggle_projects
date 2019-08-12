# -*- coding: utf-8 -*-
# @Time    : 2019/4/6 10:24 AM
# @Author  : jf.zhang
# @Email   : chinazjf@vip.qq.com
# @File    : aa.py

def password():
  print "验证密码"

def decorate(func):
  def inner():
    password()
    func()
  return inner

def deposite():
  print "存款中。。"
  
@decorate
def withdraw():
  print "取款中。。"
  
  

  


  
  
if __name__ == "__main__":
  button = input("press the button:")

  
  
  deposite = decorate(deposite)
  
  # withdraw = decorate(withdraw)
  
  if button == 1:
    deposite()
    
  else:
    withdraw()
    