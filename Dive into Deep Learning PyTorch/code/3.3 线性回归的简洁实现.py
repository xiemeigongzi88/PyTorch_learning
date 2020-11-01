# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 01:30:13 2020

@author: sxw17
"""

# 3.3 线性回归的简洁实现

import torch
import numpy as np 

num_inputs= 2 
num_examples = 1000 

true_w = [2, -3.4]
true_b = 4.2 

features = torch.tensor(np.random.normal(0,1, (num_examples, num_inputs)), dtype=torch.float32)

labels = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b

labels += torch.tensor(np.random.normal(0,0.01, size = labels.size()), dtype= torch.float32)


# 3.3.2 读取数据
import torch.utils.data as Data 

batch_size = 10 

dataset = Data.TensorDataset(features, labels)

data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for x, y in data_iter:
    print(x,y)
    break
    
    
# 3.3.3 定义模型

import torch.nn as nn 

"""
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature,1)
        
    def forward(self,x):
        y = self.linear(x)
        return y 
    
net = LinearNet(num_inputs)
print(net)
"""

net = nn.Sequential(
        nn.Linear(num_inputs,1))

print(net)
"""
Sequential(
  (0): Linear(in_features=2, out_features=1, bias=True)
)

"""

for param in net.parameters():
    print(param)

"""
Parameter containing:
tensor([[-0.0599, -0.1635]], requires_grad=True)
Parameter containing:
tensor([-0.2958], requires_grad=True)
"""


# 3.3.4 初始化模型参数
from torch.nn import init 

"""
net[0]
Linear(in_features=2, out_features=1, bias=True)

net[0].weight
Parameter containing:
tensor([[-0.0599, -0.1635]], requires_grad=True)

net[0].bias
 Parameter containing:
tensor([-0.2958], requires_grad=True)
"""

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val =0)
# net[0].bias.data.fill_(0)


# 3.3.5 定义损失函数
loss = nn.MSELoss()


# 3.3.6 定义优化算法
import torch.optim as optim 

optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

"""
SGD (
Parameter Group 0
    dampening: 0
    lr: 0.03
    momentum: 0
    nesterov: False
    weight_decay: 0
)

"""

for param_group in optimizer.param_groups:
    param_group['lr'] *=0.1 
    

for param_group in optimizer.param_groups:
    print(param_group)
    print("#############")


net.parameters()



# 3.3.7 训练模型
    
num_epochs = 10 

for epoch in range(1, num_epochs+1):
    for x, y in data_iter:
        output = net(x)
        l = loss(output, y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
    print(epoch, l.item())

# 获得层 
dense= net[0]

print(true_w, dense.weight)
# [2, -3.4]  [ 1.9935, -3.3898]
print(true_b, dense.bias)
# 4.2 [4.1898]



