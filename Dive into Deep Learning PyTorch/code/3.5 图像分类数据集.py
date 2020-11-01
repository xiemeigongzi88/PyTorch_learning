# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 22:09:48 2020

@author: sxw17
"""

# 3.5 图像分类数据集 

import torch 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 

path = "C:\\Users\\sxw17\\Desktop\\ML&DL\\pytorch\\Dive into Deep Learning PyTorch\\data\\FashionMNIST"

mnist_train = torchvision.datasets.FashionMNIST(root = path, train= True, download = True, transform= transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root = path, train= False, download = True,transform= transforms.ToTensor())


print(type(mnist_train))
# <class 'torchvision.datasets.mnist.FashionMNIST'>



print(len(mnist_train), len(mnist_test))
# 60000 10000

feature, label = mnist_train[0]
print(feature)
print(label) # 9 
print(feature.shape, label)
# torch.Size([1, 28, 28]) 9



def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    
    return [text_labels[int(i)] for i in labels]


# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    #d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()



X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))


batch_size = 256 

train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size= batch_size, shuffle=True)

test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


import time 

start = time.time()

for x, y in train_iter:
    continue 

print(time.time()-start)
# 5.322762489318848

