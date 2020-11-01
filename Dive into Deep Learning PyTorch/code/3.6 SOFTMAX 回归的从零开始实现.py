# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:34:49 2020

@author: sxw17
"""


# 3.6 SOFTMAX 回归的从零开始实现

import torch
import torchvision
import numpy as np 
import d2lzh_pytorch as d2l
import sys 
import torchvision.transforms as transforms 

import os 
os.getcwd()

path = "C:\\Users\\sxw17\\Desktop\\ML&DL\\pytorch\\Dive into Deep Learning PyTorch\\data\\FashionMNIST"

mnist_train = torchvision.datasets.FashionMNIST(root = path, train= True, download = True, transform= transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root = path, train= False, download = True,transform= transforms.ToTensor())


# 3.6.1 获取和读取数据

batch_size = 256 

train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size= batch_size, shuffle=True)

test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


# 3.6.2 初始化模型参数

num_inputs = 784 
num_outputs = 10 

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype= torch.float)

b = torch.zeros(num_outputs, dtype= torch.float)

w.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)



# 3.6.3 实现 SOFTMAX运算

X = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float)
print(X)
"""
tensor([[1, 2, 3],
        [4, 5, 6]])
"""

# 对列求和
print(X.sum(dim=0, keepdim=True))
# tensor([[5, 7, 9]])

# 对行求和
print(X.sum(dim=1, keepdim=True))
""" tensor([[ 6],
        [15]])  """

X_exp = X.exp()
print(X_exp)
"""
tensor([[  2.7183,   7.3891,  20.0855],
        [ 54.5981, 148.4132, 403.4288]])
"""

partition = X_exp.sum(dim=1, keepdim=True)
print(partition)
print(X_exp/partition)
"""
tensor([[ 30.1929],
        [606.4401]])
tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652]])

"""
def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim =1, keepdim = True)
    return x_exp/partition


x = torch.rand((2,5))
print(x)
"""
tensor([[0.4301, 0.7442, 0.1285, 0.3033, 0.5275],
        [0.3525, 0.1297, 0.4871, 0.9717, 0.8283]])
"""

x_prob = softmax(x)
print(x_prob)
"""

tensor([[0.1964, 0.2689, 0.1453, 0.1730, 0.2165],
        [0.1560, 0.1248, 0.1785, 0.2897, 0.2510]])
"""

print(x_prob.sum(dim=1))
# tensor([1.0000, 1.0000])


# 3.6.4 定义模型

# num_inputs = 784 
# num_outputs = 10 

# 784*10 
# w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype= torch.float)


def net(x):
    return softmax(torch.mm(x.view(-1,num_inputs), w)+b)


# 3.6.5 定义损失函数

y_hat = torch.tensor([[0.1,0.3,0.6],
                      [0.3,0.2,0.5]])

y = torch.LongTensor([0,2])
print(y.view(-1,1))
"""
tensor([[0],
        [2]])
"""

# https://www.cnblogs.com/HongjianChen/p/9451526.html

# 输出列的形式 
y_hat.gather(1, y.view(-1,1))
"""
tensor([[0.1000],
        [0.5000]])
"""


def cross_entropy(y_hat,y):
    return -torch.log(y_hat.gather(1, y.view(-1,1)))


# 3.6.6 计算分类准确率
    
print(y_hat)
"""
tensor([[0.1000, 0.3000, 0.6000],
        [0.3000, 0.2000, 0.5000]])
"""

y_hat.argmax(dim=1) # tensor([2, 2])
print(y) # tensor([0, 2])

print(y) # tensor([0, 2])
print(y.shape[0]) # 2 

y_hat.argmax(dim=1) == y 
# tensor([False,  True])

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1)==y).float().mean().item()

print(accuracy(y_hat,y))  # 0.5 


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0 
    
    for x, y in data_iter:
        acc_sum += (net(x).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
        
    return acc_sum/n



# 3.6.7 训练模型

num_epochs, lr = 10, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs, 
              batch_size, params = None, lr=None, optimizer = None):
    
    for epoch in range(num_epochs):
        train_l_sum = 0.0 
        train_acc_sum = 0.0 
        n = 0 
        
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y).sum()
            
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
                    
                    
            l.backward()
            
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()
                
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            n += y.shape[0]
            
        test_acc = evaluate_accuracy(test_iter, net)
        
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))



train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [w, b], lr)



