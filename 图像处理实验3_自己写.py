# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from math import *

# 1.数据集
X = np.random.random((50, 2))
plt.scatter(X[:, 0], X[:, 1])
plt.show()
# 2. 确定聚类个数
k = 3
# 3. 占位符数组
data_length = len(X)
cluster_center = np.zeros([3, 2], dtype=float, order='C')
distance = np.zeros([3], dtype=float, order='C')
label = np.zeros([data_length], dtype=int, order='C')
tag = 0
colors = ['r', 'b', 'g']
# 4. 初始化聚类中心
cluster_center[0:3] = X[0:3]
# 7. 重复5、6
epoches = 5
for epoch in range(epoches):
    # 5. 计算E：数据到每一个聚类中心的欧式距离
    for n in range(data_length):
        for i in range(k):
            distance[i] = sqrt((X[n, 0] - cluster_center[i, 0]) ** 2 + (X[n, 1] - cluster_center[i, 1]) ** 2)
        # 数据归入最小E的聚类中心中 np.argmin(distance)会返回最小值的索引
        label[n] = np.argmin(distance)
    # print(label)

    # 6. 计算M：根据每一个聚类的数据集，重新计算聚类中心
    for i in range(k):
        cluster_center_x_sum = 0.0
        cluster_center_y_sum = 0.0
        count = 0.0
        for n in range(data_length):
            if label[n] == i:
                cluster_center_x_sum = cluster_center_x_sum + X[n, 0]
                cluster_center_y_sum = cluster_center_y_sum + X[n, 1]
                count = count + 1
        cluster_center[i] = [cluster_center_x_sum / count, cluster_center_y_sum / count]
    print("epoches=", epoch, cluster_center)

    # 8. 聚类结果表示
    for i in range(k):
        for n in range(data_length):
            if label[n] == i:
                plt.scatter(X[n, 0], X[n, 1], c=colors[i], marker='o')
        plt.scatter(cluster_center[i, 0], cluster_center[i, 1], c='#8A2BE2', marker='*')
    # 设置标题
    plt.title('K-means Scatter Diagram')
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    plt.show()
