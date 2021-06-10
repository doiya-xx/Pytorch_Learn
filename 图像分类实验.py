#!/usr/bin/env python
# coding: utf-8

# # 图像分类实验
# 陈乐昕 2020-10-13
# ## 图像处理环境基础
# 图像处理库: opencv, PIL(Pillow)
# ## 建立简单的神经网络
# 有两种方式进行图像的分类实验：
# - 预训练好神经网络
# - 自定义神经网络
# 
# 分别以两种方式来做简单的实验
# 
# ## 预训练好的神经网络
# 这种方法是定义一些神经网络，并利用imagenet大规模图像集进行训练，训练好的网络的权重可以下载。 具体的模型的定义请参考：
# https://github.com/pytorch/vision/tree/master/torchvision/models

# In[1]:


import torch
import torchvision
from torchvision import datasets, models, transforms


# ### 1 定义好的网络结构，并加载训练好的权重

# In[4]:


# 定义好的网络结构
resnet50 = models.resnet50(pretrained=True)
resnet50.load_state_dict(torch.load('C:\\Users\\Doiya\\.cache\\torch\\hub\\checkpoints\\resnet50-19c8e357.pth'))


# In[ ]:


# 显示网络结构，看看具体的结构
print(resnet50)


# In[27]:


from PIL import Image
img = Image.open("./2.jpg")


# In[28]:


from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])


# In[29]:


img_t = transform(img)


# In[30]:


img_t.shape


# In[31]:


b_t=torch.unsqueeze(img_t, 0)


# In[32]:


b_t.shape


# In[33]:


resnet50.eval()
out = resnet50(b_t)
print(out.shape)


# In[34]:


with open('imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]


# In[35]:


classes


# In[36]:


max_out = torch.argmax(out)


# In[37]:


max_out


# In[38]:


_, indices = torch.sort(out, descending=True)
idd = torch.squeeze(indices)


# In[39]:


# 显示最有可能的n类
for i in range(10):
    print(classes[idd[i]])


# In[ ]:





# ## 自定义神经网络
# 该网络有3层
# 
# - 第一层input layer，有784个神经元（MNIST数据集是28*28的单通道图片，故有784个神经元）
# - 第二层为hidden_layer，设置为500个神经元
# - 最后一层是输出层，有10个神经元（10分类任务）
# - 在第二层之后还有个ReLU函数，进行非线性变换

# In[14]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


train_set = torchvision.datasets.MNIST(
    # root="./data",
    root="C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\data",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
train_loader = dataloader.DataLoader(
    dataset=train_set,
    batch_size=100,
    shuffle=False,
)

test_set = torchvision.datasets.MNIST(
    # root="./data",
    root="C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
test_loader = dataloader.DataLoader(
    dataset=test_set,
    batch_size=100,
    shuffle=False,
)

class NeuralNet(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(NeuralNet, self).__init__()
        # 隐层
        self.fc1 = nn.Linear(input_num, hidden_num)
        # 输出层
        self.fc2 = nn.Linear(hidden_num, output_num)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self,x):
        # 前向模型
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y

# 训练批次
epoches = 3
# 学习率learnRate
lr = 0.001
# 输入层维度
input_num = 784
# 隐层维度
hidden_num = 500
# 输出层维度
output_num = 10
# 选择cpu/gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = NeuralNet(input_num, hidden_num, output_num)
model.to(device)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(epoches):
    for i, data in enumerate(train_loader):
        (images, labels) = data
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epoches, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("The accuracy of total {} images: {}%".format(total, 100 * correct/total))


# 总结：
# - 预先加载训练集和测试集
# - 预先定制模型框架，占位符
# - 初始化训练参数，初始化模型
# - 开始加载数据集进行训练
# - 训练结束，调用测试集得出正确率
# 
# 调参：
# - 不变学习率lr，增加epoches，可以提高准确率 (3->4->5->10):(96.41% ->97.31% -> 97.38% ->97.58%)
# - 不变epoches，调低学习率lr，会使得拟合速度变慢 (0.001->0.0001->0.00001):(96.41% ->93.1% ->83.8%) 需要增加epoches
# - 不变epoches，调高学习率lr，会使得拟合速度变快，但却没法达到最优 需要寻找合适的lr

# ### CIFAR10
# 更改训练集为CIFAR10

# In[15]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


train_set = torchvision.datasets.CIFAR10(
    # root="./data",
    root="C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\cifar-10-python",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
train_loader = dataloader.DataLoader(
    dataset=train_set,
    batch_size=100,
    shuffle=False,
)

test_set = torchvision.datasets.CIFAR10(
    # root="./data",
    root="C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\cifar-10-python",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
test_loader = dataloader.DataLoader(
    dataset=test_set,
    batch_size=100,
    shuffle=False,
)

class NeuralNet(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, output_num)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return y


epoches = 3
lr = 0.001
input_num = 3072
hidden_num = 500
output_num = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_num, hidden_num, output_num)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(epoches):
    for i, data in enumerate(train_loader):
        (images, labels) = data
        images = images.reshape(-1, 32*32*3).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epoches, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 32*32*3).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("The accuracy of total {} images: {}%".format(total, 100 * correct/total))


# 更改模型，改为4层，中间两层隐层

# In[19]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


train_set = torchvision.datasets.CIFAR10(
    # root="./data",
    root="C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\cifar-10-python",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
train_loader = dataloader.DataLoader(
    dataset=train_set,
    batch_size=100,
    shuffle=False,
)

test_set = torchvision.datasets.CIFAR10(
    # root="./data",
    root="C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\cifar-10-python",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
test_loader = dataloader.DataLoader(
    dataset=test_set,
    batch_size=100,
    shuffle=False,
)

class NeuralNet(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden1_num)
        self.fc2 = nn.Linear(hidden1_num, hidden2_num)
        self.fc3 = nn.Linear(hidden2_num, output_num)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        y = self.fc3(x)
        return y


epoches = 5
lr = 0.001
input_num = 3072
hidden1_num = 1500
hidden2_num = 500
output_num = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_num, hidden_num, output_num)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(epoches):
    for i, data in enumerate(train_loader):
        (images, labels) = data
        images = images.reshape(-1, 32*32*3).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epoches, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 32*32*3).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("The accuracy of total {} images: {}%".format(total, 100 * correct/total))


# 总结：
# - 修改模型，增加隐层，并没有使得精度发生显著变化
# - 修改参数，调整lr，并没有使得精度发生显著变化，甚至变小
# - 修改参数，增加epoches，可以增加一点点精度
# 
# 调参，提高准确率
# - 增加epoches
# - 调整lr
# - 增加隐层

# 再增加一层隐层，测试是否能够提高精度

# In[20]:


import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


train_set = torchvision.datasets.CIFAR10(
    # root="./data",
    root="C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\cifar-10-python",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
train_loader = dataloader.DataLoader(
    dataset=train_set,
    batch_size=100,
    shuffle=False,
)

test_set = torchvision.datasets.CIFAR10(
    # root="./data",
    root="C:\\Users\\Doiya\\Desktop\\PythonProject\\Pytorch_Lrean\\cifar-10-python",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
test_loader = dataloader.DataLoader(
    dataset=test_set,
    batch_size=100,
    shuffle=False,
)

class NeuralNet(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_num, hidden1_num)
        self.fc2 = nn.Linear(hidden1_num, hidden2_num)
        self.fc3 = nn.Linear(hidden2_num, hidden3_num)
        self.fc4 = nn.Linear(hidden3_num, output_num)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        y = self.fc4(x)
        return y


epoches = 50
lr = 0.001
input_num = 3072
hidden1_num = 2000
hidden2_num = 1000
hidden3_num = 500
output_num = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_num, hidden_num, output_num)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(epoches):
    for i, data in enumerate(train_loader):
        (images, labels) = data
        images = images.reshape(-1, 32*32*3).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epoches, loss.item()))


with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 32*32*3).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("The accuracy of total {} images: {}%".format(total, 100 * correct/total))


# 结论：
# - 即使模型隐层越多，epoches增加到50批次，loss下降到1以下，精度也没有改变，所以问题不是参数的问题，而是模型的问题，激活函数没起到很好的作用。
# - 目前还没找到很好的解决办法，还得继续学习。

# 收获
# - 通过已训练的模型，加载预训练的参数，进行神经网络的初使用，通过使用图片来得到预测的分类，感受到神经网络运作
# - 自己一步一步的搭建神经网络，使用已有的训练集进行模型训练，得到自己的神经网络
# - 调参的体验

# In[ ]:




