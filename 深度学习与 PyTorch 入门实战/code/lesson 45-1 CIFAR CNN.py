# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:27:18 2020

@author: sxw17
"""
import os 

os.getcwd()

path = 'C:\\Users\\sxw17\\Desktop\\ML&DL\\pytorch\\深度学习与 PyTorch 入门实战\\code'

os.chdir(path)

import torch
from torch.utils.data import dataloader
from torchvision import datasets
from torchvision import transforms





def main():
    
    batch_size = 32 
    
    # train 
    cifar_train = datasets.CIFAR10('cifar', True, transform= transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
                    ]), download=True)

    cifar_train = dataloader(cifar_train, batch_size= batch_size, shuffle =True)
    
    # validation 
    cifar_test = datasets.CIFAR10('cifar', False, transform= transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
                    ]), download=True)

    cifar_test = dataloader(cifar_test, batch_size= batch_size, shuffle =True)
    
    
    x, label = iter(cifar_train).next()
    print('x: ', x.shape, 'label: ', label.shape)




if __name__=='__main__':
    main()


















































































