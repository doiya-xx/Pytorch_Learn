#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt
import random
from math import *


# In[15]:


# 1. 数据集
X = np.random.random((50,2))*100
plt.scatter(X[:,0],X[:,1])
plt.show()


# In[16]:


# 2. 确定聚类个数
# 聚类中心个数
k = 3


# In[17]:


# 3. 占位符数组
# 数据集数据个数
data_length = len(X)
# 聚类中心 二维数组[3,2]
cluster_center = np.zeros([3,2], dtype = float, order = 'C')
# array([[0., 0.],
#        [0., 0.],
#        [0., 0.]])
# 距离数组 一维数组[1,3] 
distance = np.zeros([3], dtype = float, order = 'C')
# array([0., 0., 0.])
# 聚类数组 一维数组[1,50]
label = np.zeros([data_length],dtype=int,order='C')
# 标签
tag = 0
# 颜色
colors = ['r', 'b', 'g']


# In[18]:


# 4. 初始化聚类中心
# 生成k个不同随机数
rand_data = random.sample(range(0,data_length), k)
# [19, 40, 39]
# 得出k个聚类中心
for i in range(k):
    # 随机选出
    cluster_center[i] = X[rand_data[i]]
    # 不随机选出
    # cluster_center[i] = X[i]
(rand_data, cluster_center)


# In[19]:


# 7. 重复5、6
# 批次
# 一开始为20次，发现在随机聚类中心的情况下，最少3次就可以收敛
# 在不随机聚类中心的情况下，同样也是最少3次就可以收敛，收敛很快，可以发现是否收敛速度与聚类中心是否随机并没有关系
# 聚类中心只会影响聚类的分类
epoches = 5
for epoch in range(epoches):
    # 5. 计算E：数据到每一个聚类中心的欧式距离
    for n in range(data_length):
        for i in range(k):
            distance[i] = sqrt((X[n,0] - cluster_center[i,0])**2 + (X[n,1] - cluster_center[i,1])**2)
        # print(distance)
        # 数据归入最小E的聚类中心中 np.argmin(distance)会返回最小值的索引
        # print(np.min(distance))
        tag = np.argmin(distance)
        # print(tag)
        label[n] = tag
    print(label)

    # 6. 计算M：根据每一个聚类的数据集，重新计算聚类中心
    for i in range(k):
        cluster_center_x_sum = 0.0
        cluster_center_y_sum = 0.0
        count = 0.0
        for n in range(data_length):
            if label[n]==i:
                cluster_center_x_sum = cluster_center_x_sum + X[n,0]
                cluster_center_y_sum = cluster_center_y_sum + X[n,1]
                count = count + 1
        cluster_center[i] = [cluster_center_x_sum/count, cluster_center_y_sum/count]
    print(cluster_center)

    # 8. 聚类结果表示
    for i in range(k):
        for n in range(data_length):
            if label[n]==i:
                plt.scatter(X[n,0],X[n,1],c=colors[i],marker='o')
        plt.scatter(cluster_center[i,0],cluster_center[i,1],c='#8A2BE2',marker='*')
    # 设置标题
    plt.title('K-means Scatter Diagram')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    plt.show()


# ## 反思
# 1. np的用法：
#     - 如何构造随机数组
#     - 构造占位符数组
#     - 构造的数组的使用
# 2. 随机数的生成：
#     - 用不同方式产生随机数
# 3. math库的使用
# 4. matplotlib库的使用：
#     - 如何画出数据点图
#     - 图参数的使用
