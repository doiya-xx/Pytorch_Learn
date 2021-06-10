#!/usr/bin/env python
# coding: utf-8

# # 图像处理实验2
# 陈乐昕 2020-10-27
# ## 图像处理环境基础
# 图像处理库：opencv, PIL(Pillow)
# 
# ## 提取数据集的数据特征
# CIFAR10的数据特征
# 50000 32*32*3 ->50000*1000
# 利用resent50

# In[1]:


import torch
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from torchvision import transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.optim as optim
import os

# In[2]:


# 定义好的网络结构
resnet50 = models.resnet50(pretrained=True)
resnet50.load_state_dict(torch.load('C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\resnet50-19c8e357.pth'))
resnet50.eval()

# In[3]:


train_set = torchvision.datasets.CIFAR10(
    root="C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\cifar-10-python",
    train=True,
    transform=transforms.ToTensor(),
    download=False
)
train_loader = dataloader.DataLoader(
    dataset=train_set,
    batch_size=100,
    shuffle=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# In[ ]:


# 别运行

# for i,data in enumerate(train_loader):
#     # 1
#     (images, labels) = data
#     # torch.Size([100, 3, 32, 32])
#     images = images.reshape(-1, 32*32*3).to(device)
#     # torch.Size([100, 3072])
#
#     # 2
#     (images, labels) = data
#     # torch.Size([100, 3, 32, 32])
#     images = resnet50(images).to(device)
#     # torch.Size([100, 1000])


# output_images = torch.zeros(100,1000)
# tensor([[0., 0., 0.,  ..., 0., 0., 0.],
#       [0., 0., 0.,  ..., 0., 0., 0.],
#       [0., 0., 0.,  ..., 0., 0., 0.],
#       ...,
#       [0., 0., 0.,  ..., 0., 0., 0.],
#       [0., 0., 0.,  ..., 0., 0., 0.],
#       [0., 0., 0.,  ..., 0., 0., 0.]])


# In[ ]:


# 别运行 会爆内存

output_images = torch.zeros(100, 1000)
for i, data in enumerate(train_loader):
    (images, labels) = data
    images = resnet50(images).to(device)
    if i != 1:
        output_images = torch.cat((output_images, images), 0).to(device)
    else:
        output_images = images
    if i % 10 == 0:
        print(i, output_images.shape)
print(i, output_images.shape)
#
#
# # In[ ]:
#
#
# # 别运行
# images_array = torch.zeros(100,100,1000)
# images_array[0,:,:]
# # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
# #       [0., 0., 0.,  ..., 0., 0., 0.],
# #       [0., 0., 0.,  ..., 0., 0., 0.],
# #       ...,
# #       [0., 0., 0.,  ..., 0., 0., 0.],
# #       [0., 0., 0.,  ..., 0., 0., 0.],
# #       [0., 0., 0.,  ..., 0., 0., 0.]])
# images_array[0,:,:].shape
# # torch.Size([100, 1000])


# In[4]:


# # 测试 先将所有batch的数据弄到一个数组里面
# output_images = torch.zeros(100,1000)
# images_array = torch.zeros(500,100,1000)
# for i,data in enumerate(train_loader):
#     (images, labels) = data
#     images_array[i,:,:] = resnet50(images).to(device)
#
#
# # In[ ]:




