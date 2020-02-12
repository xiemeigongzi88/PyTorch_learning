# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:12:51 2020

@author: sxw17
"""

import os 

path ='C:\\Users\\sxw17\\Desktop\\ML&DL\\pytorch\\深度学习与 PyTorch 入门实战\\code'

os.getcwd()
os.chdir(path)

import torch
from torch.nn import functional as F
from torch import nn 
from torch import optim
from torchvision import datasets, transforms

batch_size = 20 
learning_rate = 0.001 
epochs = 10 


# 训练集
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,  # train=True则得到的是训练集
				   transform=transforms.Compose([  # 进行数据预处理
					   transforms.ToTensor(),  # 这表示转成Tensor类型的数据
					   transforms.Normalize((0.1307,), (0.3081,))  # 这里是进行数据标准化(减去均值除以方差)
				   ])),
	batch_size=batch_size, shuffle=True)  # 按batch_size分出一个batch维度在最前面,shuffle=True打乱顺序
# 测试集
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=False, transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])),
	batch_size=batch_size, shuffle=True)
 
"""建立三个线性层"""
# 这里表示输入是784个分量,输出是200个分量的,注意这个顺序
w1 = torch.randn(200, 784, requires_grad=True)
b1 = torch.randn(200, requires_grad=True)
w2 = torch.randn(200, 200, requires_grad=True)
b2 = torch.randn(200, requires_grad=True)
w3 = torch.randn(10, 200, requires_grad=True)
b3 = torch.randn(10, requires_grad=True)
 
 
def forward(x):
	"""对输入的样本矩阵x的前向计算过程,x的shape是[样本数,784]"""
    #x: ([20, 784]) , w1: 784 200 -> 20 ,200
	x = x @ w1.t() + b1
	x = F.relu(x)  # 非线性激活
    # (20,200) *(200,200)-> (20,200)
	x = x @ w2.t() + b2
	x = F.relu(x)
    # (20,200)*(200,10) -> (20,10)
	x = x @ w3.t() + b3
	x = F.relu(x)
	return x # (20,10)
 
 
# 建立优化器,指明优化目标和学习率
optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=1e-3)
# 计算CEL的函数
criterion = nn.CrossEntropyLoss()


################################################################

for batch_idx, (data, target) in enumerate(train_loader):
		# 摊平成shape=[样本数,784]的形状
		data = data.reshape(-1, 28 * 28)
		# 前向计算出logits
		logits = forward(data)
		# 计算Loss,这里不需要再Softmax一次,PyTorch计算CEL时已经做了Softmax了
		loss = criterion(logits, target)
		# 清空梯度
		optimizer.zero_grad()
		# 反向传播计算各个参数(优化目标)相对于Loss的梯度信息
		loss.backward()
		# 执行优化器
		optimizer.step()
		# 每100个batch输出一次信息
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				'1', batch_idx * len(data), len(train_loader.dataset),
					   100. * batch_idx / len(train_loader), loss.item()))
            break


for batch_idx, (data, target) in enumerate(train_loader):
    print(data.shape, data)
    # torch.Size([20, 1, 28, 28])
    print("###############################")
    print(target, target.shape) #torch.Size([20])
    break


data = data.view(-1, 28*28) # torch.Size([20, 784])
data.shape 

logits = forward(data)
logits, logits.shape #  torch.Size([20, 10]))
loss = criterion(logits, target)
loss.shape #  torch.Size([]) 标量

optimizer.zero_grad()
# 反向传播计算各个参数(优化目标)相对于Loss的梯度信息
loss.backward()
# 执行优化器
optimizer.step()

loss.item() # 3071.82861328125

print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				'1', batch_idx * len(data), len(train_loader.dataset),
					   100. * batch_idx / len(train_loader), loss.item()))

batch_idx * len(data)

len(train_loader.dataset) # 60000
len(train_loader) # 3000  = 60000 /20 
100. * batch_idx / len(train_loader)



################################################
test_loss = 0  # 在测试集上的Loss,反映了模型的表现
correct = 0  # 记录正确分类的样本数
     
	# 对测试集中每个batch的样本,标签
for data, target in test_loader:
		# 摊平成shape=[样本数,784]的形状
	data = data.reshape(-1, 28 * 28)
	logits = forward(data)
	test_loss += criterion(logits, target).item()
		# 得到的预测值输出是一个10个分量的概率,在第2个维度上取max
		# logits.data是一个shape=[batch_size,10]的Tensor
		# 注意Tensor.max(dim=1)是在这个Tensor的1号维度上求最大值
		# 得到一个含有两个元素的元组,这两个元素都是shape=[batch_size]的Tensor
		# 第一个Tensor里面存的都是最大值的值,第二个Tensor里面存的是对应的索引
		# 这里要取索引,所以取了这个tuple的第二个元素
		# print(type(logits.data), logits.data.shape,type(logits.data.max(dim=1)))
	pred = logits.data.max(dim=1)[1]
		# 对应位置相等则对应位置为True,这里用sum()即记录了True的数量
	correct += pred.eq(target.data).sum()
	test_loss /= len(test_loader.dataset)
    break


	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
        break


############################################
test_loss = 0  # 在测试集上的Loss,反映了模型的表现
correct = 0  # 记录正确分类的样本数

for data, target in test_loader:
    print(data.shape, target.shape)
    
    
    break
    
    
data = data.reshape(-1, 28 * 28)
logits = forward(data) # torch.Size([20, 10])

criterion(logits, target) # tensor(2397.5591, grad_fn=<NllLossBackward>)

test_loss += criterion(logits, target).item()
   
pred = logits.data.max(dim=1)[1]

logits.data  # ([20, 10])
logits.data.max(dim=1)

logits.data.max(dim=1)[1]
# tensor([1, 0, 2, 0, 2, 7, 2, 2, 2, 2, 2, 5, 3, 0, 2, 0, 2, 2, 2, 7])

pred.eq(target.data)
pred.eq(target.data).sum()

correct += pred.eq(target.data).sum()
test_loss /= len(test_loader.dataset)

len(test_loader.dataset)

####################################################
####################################################

import torch
from torch.nn import functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms
 
"""可以将超参数都放到一起写在最前面,方便调参"""
batch_size = 200  # 每批的样本数量
learning_rate = 0.01  # 学习率
epochs = 10  # 跑多少次样本集
 
"""
读取MNIST手写数字数据集,初次运行下载到../data/目录下
要注意所有样本进行标准化的参数要保持一致(这里是样本和方差)
这个标准化的参数是数据提供方计算好的
所以就不用自己计算了,在网上查好然后标准化时候写进去就可以了
"""
# 训练集
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,  # train=True则得到的是训练集
				   transform=transforms.Compose([  # 进行数据预处理
					   transforms.ToTensor(),  # 这表示转成Tensor类型的数据
					   transforms.Normalize((0.1307,), (0.3081,))  # 这里是进行数据标准化(减去均值除以方差)
				   ])),
	batch_size=batch_size, shuffle=True)  # 按batch_size分出一个batch维度在最前面,shuffle=True打乱顺序
# 测试集
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=False, transform=transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])),
	batch_size=batch_size, shuffle=True)
 
"""建立三个线性层"""
# 这里表示输入是784个分量,输出是200个分量的,注意这个顺序
w1 = torch.randn(200, 784, requires_grad=True)
b1 = torch.randn(200, requires_grad=True)
w2 = torch.randn(200, 200, requires_grad=True)
b2 = torch.randn(200, requires_grad=True)
w3 = torch.randn(10, 200, requires_grad=True)
b3 = torch.randn(10, requires_grad=True)
 

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)
 
def forward(x):
	"""对输入的样本矩阵x的前向计算过程,x的shape是[样本数,784]"""
	x = x @ w1.t() + b1
	x = F.relu(x)  # 非线性激活
	x = x @ w2.t() + b2
	x = F.relu(x)
	x = x @ w3.t() + b3
	x = F.relu(x)
	return x
 
 
# 建立优化器,指明优化目标和学习率
optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=1e-3)
# 计算CEL的函数
CEL = nn.CrossEntropyLoss()
 
"""训练+测试过程"""
# 每个epoch是遍历一次样本集
for epoch in range(epochs):
	"""训练"""
	# 对训练集中每个batch的样本,标签
	for batch_idx, (data, target) in enumerate(train_loader):
		# 摊平成shape=[样本数,784]的形状
		data = data.reshape(-1, 28 * 28)
		# 前向计算出logits
		logits = forward(data)
		# 计算Loss,这里不需要再Softmax一次,PyTorch计算CEL时已经做了Softmax了
		loss = CEL(logits, target)
		# 清空梯度
		optimizer.zero_grad()
		# 反向传播计算各个参数(优化目标)相对于Loss的梯度信息
		loss.backward()
		# 执行优化器
		optimizer.step()
		# 每100个batch输出一次信息
		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
					   100. * batch_idx / len(train_loader), loss.item()))
	"""测试"""
	test_loss = 0  # 在测试集上的Loss,反映了模型的表现
	correct = 0  # 记录正确分类的样本数
	# 对测试集中每个batch的样本,标签
	for data, target in test_loader:
		# 摊平成shape=[样本数,784]的形状
		data = data.reshape(-1, 28 * 28)
		logits = forward(data)
		test_loss += CEL(logits, target).item()
		# 得到的预测值输出是一个10个分量的概率,在第2个维度上取max
		# logits.data是一个shape=[batch_size,10]的Tensor
		# 注意Tensor.max(dim=1)是在这个Tensor的1号维度上求最大值
		# 得到一个含有两个元素的元组,这两个元素都是shape=[batch_size]的Tensor
		# 第一个Tensor里面存的都是最大值的值,第二个Tensor里面存的是对应的索引
		# 这里要取索引,所以取了这个tuple的第二个元素
		# print(type(logits.data), logits.data.shape,type(logits.data.max(dim=1)))
		pred = logits.data.max(dim=1)[1]
		# 对应位置相等则对应位置为True,这里用sum()即记录了True的数量
		correct += pred.eq(target.data).sum()
	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))







































