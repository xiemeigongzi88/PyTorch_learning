# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 19:24:48 2020

@author: sxw17
"""

# 3.7 SOFTMAX 回归的简洁实现

import torch 
from torch import nn 
from torch.nn import init 
import numpy as np 
import d2lzh_pytorch as d2l

# 3.7.1 获取和读取数据


path = "C:\\Users\\sxw17\\Desktop\\ML&DL\\pytorch\\Dive into Deep Learning PyTorch\\data\\FashionMNIST"

mnist_train = torchvision.datasets.FashionMNIST(root = path, train= True, download = True, transform= transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root = path, train= False, download = True,transform= transforms.ToTensor())


# 3.7.1 获取和读取数据

batch_size = 256 

train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size= batch_size, shuffle=True)

test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


# 3.7.2 定义和初始化模型

num_inputs = 784 
num_outputs = 10 

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        
    def forward(self,x):
        y = self.linear(x.view(x.shape[0],-1))
        return y 
    
net = LinearNet(num_inputs, num_outputs)

"""
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self,x):
        return x.view(x.shape[0],-1)
   
from collections import OrderedDict
net = nn.Sequential(
        OrderedDict([
                ('flatten', FlattenLayer()),
                ('linear', nn.Linear(num_inputs, num_outputs)))])) 
"""    
    
init.normal_(net.linear.weight, mean = 0, std=0.01)
init.constant_(net.linear.bias, val = 0)


# 3.7.3 SOFTMAX 和交叉熵损失函数

loss = nn.CrossEntropyLoss()


# 3.7.4 定义优化算法

print(net)
"""
LinearNet(
  (linear): Linear(in_features=784, out_features=10, bias=True)
)
"""
# print(len(net))

print(net.parameters())

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)


# 3.7.5 训练模型
num_epochs = 10 
d2l.train_ch3(net, train_iter, test_iter,loss,num_epochs,
              batch_size, None, None, optimizer)







