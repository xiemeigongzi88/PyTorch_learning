# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:51:45 2020

@author: sxw17
"""

import os 
os.getcwd()

path = "C:\\Users\\sxw17\\Desktop\\ML&DL\\pytorch\\Dive into Deep Learning PyTorch"

os.chdir(path)


# 3.1.2 线性回归的表示方法

import torch 

a = torch.ones(3)
a

b =10 

a+b 


# 3.2 线性回归的从零开始实现

from matplotlib import pyplot as plt 

import numpy as np 
import random


# 3.2.1 生成数据集

num_inputs = 2 
num_examples = 1000 

true_w = [2, -3.4]
true_b = 4.2 


features = torch.from_numpy(np.random.normal(0,1, (num_examples, num_inputs)))

labels = true_w[0]*features[:,0] + true_w[1]*features[:,1]+ true_b

labels

print(labels.size()) # torch.Size([1000])


labels += torch.from_numpy(np.random.normal(0, 0.01,size = labels.size()))


features[0]

labels[0]


fig_size = (3.5,2.5)

#plt.rcParams['figure.figuresize'] = fig_size

len(labels)
len(features)

x_1 = features[:,0]
x_2 = features[:,1]

plt.scatter(x_1, labels)
plt.scatter(x_2, labels)
plt.show()


# 3.2.2 读取数据

# yield 用法
def func():
    for i in range(0,3):
        yield i
 
f = func()
next(f)
next(f)
next(f)
next(f)

test = list(range(1000))
random.shuffle(test)

# print(test) 

features.index_select(0,torch.tensor([0,1,2,3]))

"""
index_select(
    dim,
    index)


dim：表示从第几维挑选数据，类型为int值；
index：表示从第一个参数维度中的哪个位置挑选数据，类型为torch.Tensor类的实例；
"""



t = torch.arange(24).reshape(2, 3, 4) # 初始化一个tensor，从0到23，形状为（2,3,4）
print("t--->", t)

"""
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
"""

index = torch.tensor([1, 2]) # 要选取数据的位置
print("index--->", index)

 
data1 = t.index_select(1, index) # 第一个参数:从第1维挑选， 第二个参数:从该维中挑选的位置
print("data1--->", data1)
"""
tensor([[[ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[16, 17, 18, 19],
         [20, 21, 22, 23]]])
"""
 
data2 = t.index_select(2, index) # 第一个参数:从第2维挑选， 第二个参数:从该维中挑选的位置
print("data2--->", data2)

"""
tensor([[[ 1,  2],
         [ 5,  6],
         [ 9, 10]],

        [[13, 14],
         [17, 18],
         [21, 22]]])

"""


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        
        j = torch.LongTensor(indices[i:min(i+batch_size, num_examples)])
        
        yield features.index_select(0,j), labels.index_select(0,j)
        
        
batch_size = 10
for x, y in data_iter(batch_size, features, labels):
    print(x)
    print("######################")
    print(y)
    
    break



######################################
# 3.2.3 初始化模型参数
    
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs,1)), dtype=torch.float64)

w # tensor([[ 0.0073],
        #[-0.0031]], requires_grad=True)

b = torch.zeros(1, dtype=torch.float64)
b # tensor([0.], requires_grad=True)

w.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)



# 3.2.4 定义模型

w.size() # 2 1 
x.size() # 10 2
b.size() # 1 



def linreg(x, w, b):
    return torch.mm(x,w)+b 


# 3.2.5 定义损失函数
test = torch.arange(0,12).reshape([6,2])
test     
"""
tensor([[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7],
        [ 8,  9],
        [10, 11]])
"""

test_a = torch.arange(0,12)
# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
test_a 
print(test.view(test_a.size()))
print(test)


def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size()))


# 3.2.6 定义优化算法
    
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size
        
        
# 3.2.7 训练模型
        
lr = 0.03 
num_epochs = 1000
net = linreg
loss = squared_loss

"""
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    
    random.shuffle(indices)
    
    for i in range(0, num_examples, batch_size):
        
        j = torch.LongTensor(indices[i:min(i+batch_size, num_examples)])
        
        yield features.index_select(0,j), labels.index_select(0,j)
        
        
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr*param.grad/batch_size
        
"""
for epoch in range(num_epochs):
    
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x,w,b), y).sum()
        l.backward()
        sgd([w,b], lr, batch_size)
        
        
        w.grad.data.zero_()
        b.grad.data.zero_()
        
        
    train_l = loss(net(features,w,b), labels)
    print("epoch %d, loss %f" % (epoch+1, train_l.mean().item()))


print(true_w, w)
print(true_b, b)
        
        
