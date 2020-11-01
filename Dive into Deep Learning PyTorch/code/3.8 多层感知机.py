# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 20:11:48 2020

@author: sxw17
"""


# 3.8 多层感知机


# 3.8.1 隐藏层

# 3.8.2 激活函数

# 3.8.2.1 ReLU

ReLu(x) = max(x, 0)

import torch 
import numpy as np 
import matplotlib.pylab as plt 
import sys 
import d2lzh_pytorch as d2l


def xyplot(x_val, y_val, name):
    d2l.set_figsize(figsize=(5,2.5))
    d2l.plt.plot(x_val.detach().numpy(), y_val.detach().numpy())

    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name+"(x)")
    
    
x = torch.arange(-9.0, 9.0, 0.1, requires_grad= True)
y = x.relu()

xyplot(x,y,'relu')

