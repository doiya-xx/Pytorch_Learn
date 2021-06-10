#!/usr/bin/env python
# coding: utf-8

# # 图像处理实验4
# 陈乐昕 2020-11-10
# 
# ## 实验环境
# 
# 
# ## 遗传算法
# 思路：
# 1. 初始化种群
# 2. 种群第一代G=0
# 3. 计算适应度
# 4. 选择操作
# 5. 交叉操作
# 6. 变异操作
# 7. 产生下一代
# 8. G=G+1 G>GEN
# 9. 若否则回到第3步重新开始
# 10. 若是则输出最佳方案

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from math import *
import prettytable as pt


# In[6]:


# 1. data_set [0,31]
data_set = np.arange(32)
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])


# In[7]:


# 2. fitness_function
def fitness_function(x):
    # 适应度函数 y = x^2
    y = x**2
    return y


# In[8]:


# 3. 占位符
# 进化代数
G = 0
# 种群规模
k = 4

# 种群
s = np.zeros((4),dtype='int32')
# 种群集合
S = np.zeros((1,4),dtype='int32')
# 染色体
chromosome = np.zeros((4),dtype='<U5')
# 种群染色体集合
S_chromosome = np.zeros((1,4),dtype='<U5')
# [['00000','00000','00000','00000']]
# 适应度
S_fitness = np.zeros((1,4),dtype='float64')
# 个体被选择的概率
S_selective_probability = np.zeros((1,4),dtype='float64')
# 累积概率
S_cumulative_probability = np.zeros((1,4),dtype='float64')
# 插入用
S_insert = np.zeros((4), dtype='float64')
# 变异率
mutation_rate = 0.05
# 基因总数 20
# def gene_counts():
    # return len(S_chromosome[0]) * len(S_chromosome[0,0])
gene_counts = 20
# 变异个数
mutation_counts = mutation_rate * gene_counts


table1 = pt.PrettyTable()
table2 = pt.PrettyTable()
table1.field_names = ["种群","染色体","适应度","选择概率","积累概率"]
table2.field_names = ["种群G的染色体",1,2,3,4]


# In[14]:


# 4.
def initial_population_selection():
    # 初始种群选取
    global G
    if G != 0:
        G = 0
    # 生成k个不同随机数
    rand_data = random.sample(range(0,len(data_set)), k)
    for i in range(k):
        # 随机选出
        S[0, i] = data_set[rand_data[i]]
    return 0

def code_chromosome(Si):
    # 种群Si
    # 编码染色体
    for i in range(k):
        chromosome[i] = to_bin1(Si[i])
    return chromosome

def decode_chromosome(Si_chromosome):
    # 种群染色体Si_chromosome
    # 解码染色体
    for i in range(k):
        s[i] = to_int(Si_chromosome[i])
    return s

def calculate_probability():
    # 计算每一代种群中每个个体的选择概率和累积概率
    S_fitness_sum = 0
    for i in range(k):
        # 计算适应度总和
        S_fitness_sum = S_fitness_sum + S_fitness[G,i]
    # 选择概率 = 适应度/适应度总和
    S_selective_probability[G] = S_fitness[G]/S_fitness_sum
    # 累积概率 = 前一个的积累概率 + 自己的选择概率
    S_cumulative_probability[G,0] = S_selective_probability[G,0]
    for i in range(k-1):
        i += 1
        S_cumulative_probability[G,i] = S_cumulative_probability[G,i-1]+S_selective_probability[G,i]
    return 0

def russian_turntable():
    # 俄罗斯转盘
    # 返回选择哪一个个体进行复制
    # 随机选取一个[0,1)的概率
    rand = np.random.random((1))[0]
    choose = 0
    # 判断这个概率落在那个积累概率之间，就选择哪个
    if rand < S_cumulative_probability[G-1, 0]:
        choose = 0
    elif rand < S_cumulative_probability[G-1, 1]:
        choose = 1
    elif rand < S_cumulative_probability[G-1, 2]:
        choose = 2
    else:
        choose = 3
    return choose

def op_select_copy():
    # 选择-复制操作
    for i in range(k):
        # 根据俄罗斯转盘的选择将上一代复制到下一代
        S_chromosome[G,i] = S_chromosome[G-1, russian_turntable()]
    print("选择-复制操作结束")
    table2.clear_rows()
    table2.add_row(["选择-复制操作",S_chromosome[G,0],S_chromosome[G,1],S_chromosome[G,2],S_chromosome[G,3]])
    print(table2)
    # print("种群G", G, "的染色体：", S_chromosome[G])
    return 0

def op_intersect():
    # 交叉操作
    # 交换二进制的最后两位，0和1交换，2和3交换
    S_chromosome[G,0] = S_chromosome[G,0][0:3] + S_chromosome[G,1][3:]
    S_chromosome[G,1] = S_chromosome[G,1][0:3] + S_chromosome[G,0][3:]
    S_chromosome[G,2] = S_chromosome[G,2][0:3] + S_chromosome[G,3][3:]
    S_chromosome[G,3] = S_chromosome[G,3][0:3] + S_chromosome[G,2][3:]
    print("交叉操作结束")
    table2.clear_rows()
    table2.add_row(["交叉操作",S_chromosome[G,0],S_chromosome[G,1],S_chromosome[G,2],S_chromosome[G,3]])
    print(table2)
    # print("种群G", G, "的染色体：", S_chromosome[G])
    return 0

def op_variation():
    # 变异操作
    # [0,4)之间的随机整数 作为这一代的某一个个体
    a = np.random.randint(0,4)
    # [0,5)之间的随机整数 作为某一个个体里面的某一位
    b = np.random.randint(0,5)
    if S_chromosome[G,a][b] == '1':
        S_chromosome[G,a] = S_chromosome[G,a][0:b] + '0' + S_chromosome[G,a][b+1:]
    else:
        S_chromosome[G,a] = S_chromosome[G,a][0:b] + '1' + S_chromosome[G,a][b+1:]
    print("变异操作结束")
    table2.clear_rows()
    table2.add_row(["变异操作",S_chromosome[G,0],S_chromosome[G,1],S_chromosome[G,2],S_chromosome[G,3]])
    print(table2)
    # print("种群G", G, "的染色体：", S_chromosome[G])
    return 0

def next_generation():
    # 用来产生下一代
    global G, S, S_fitness, S_chromosome, S_selective_probability, S_cumulative_probability
    G = G + 1
    if len(S)<G+1:
        # 扩建占位符
        S = np.insert(S, G, s, 0)
        S_fitness = np.insert(S_fitness, G, S_insert, 0)
        S_chromosome = np.insert(S_chromosome, G, chromosome, 0)
        S_selective_probability = np.insert(S_selective_probability, G, S_insert,0)
        S_cumulative_probability = np.insert(S_cumulative_probability, G, S_insert,0)
    
    # 遗传操作
    op_select_copy()
    op_intersect()
    op_variation()
    print("\n产生新的种群")
    # 染色体编码、计算适应度、计算概率
    S[G] = decode_chromosome(S_chromosome[G])
    S_fitness[G] = fitness_function(S[G])
    calculate_probability()
    
    return 0


# In[15]:


def to_bin1(value):
    # 将十进制数转化为任意位数二进制
    # 转换为二进制，并且去掉“0b”
    o_bin = bin(value)[2:]
    # 原字符串右侧对齐， 左侧补零，补齐后总位宽为8 #不带"0b"
    out_bin = o_bin.rjust(5,'0')
    return out_bin
# 不带"0b"
# out_bin = to_bin2(27)
# 11011
# 带"0b"，和python内置函数bin输出格式一致，字符串开头带"0b"
# out_bin = "0b" + out_bin
# 0b11011

def to_bin2(value, num):
    # 十进制数据，二进制位宽
    # 将十进制数转化为任意位数二进制
    bin_chars = ""
    temp = value
    for i in range(num):
        bin_char = bin(temp % 2)[-1]
        temp = temp // 2
        bin_chars = bin_char + bin_chars
    #输出指定位宽的二进制字符串
    return bin_chars.upper()
# 不带"0b"
# out_bin = to_bin1(2,5)
# 00010
# 带"0b"，和python内置函数bin输出格式一致，字符串开头带"0b"
# out_bin = "0b" + out_bin
# 0b00010

def to_int(vable):
    # 将二进制数转化为任意位数十进制
    out_int = int(vable,2)
    return out_int


# In[16]:


def print_S(g):
    print("种群G", g, "：", S[g])
    print("种群G", g, "的染色体：", S_chromosome[g])
    print("种群G", g, "的适应度：", S_fitness[g])
    print("种群G", g, "的选择概率：", S_selective_probability[g])
    print("种群G", g, "的积累概率：", S_cumulative_probability[g])


# In[17]:


def table_view(g):
    table1.clear_rows()
    for i in range(k):
        table1.add_row([S[g,i],
                       S_chromosome[g,i],
                       S_fitness[g,i],
                       S_selective_probability[g,i],
                       S_cumulative_probability[g,i]])
    print(table1.get_string(title="种群G"+str(g)))


# In[22]:


# 开始初始化种群
initial_population_selection()
S_chromosome[0] = code_chromosome(S[0])
# 计算适应度
S_fitness[0] = fitness_function(S[0])
calculate_probability()
# print_S(0)
table_view(0)

bk=0
while(bk==0):
    next_generation()
    # print_S(G)
    table_view(G)
    for i in range(k):
        if S[G,i]==31:
            bk=1

print("在遗传了"+str(G)+"代后，终于产生了适应度最高的后代了")


# In[23]:


table3 = pt.PrettyTable()
table3.field_names = ["种群代数",1,2,3,4]
table3.clear_rows()
for i in range(G+1):
    table3.add_row(["G"+str(i), S[i,0], S[i,1], S[i,2], S[i,3]])
print(table3)


# In[24]:


table4 = pt.PrettyTable()
table4.field_names = ["种群代数",1,2,3,4]
table4.clear_rows()
for i in range(G+1):
    table4.add_row(["G"+str(i), S_chromosome[i,0], S_chromosome[i,1], S_chromosome[i,2], S_chromosome[i,3]])
print(table4)


# In[25]:


import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),
                               cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))])
fig.show()


# # 总结
# ## 遇到的问题以及怎么解决
# 1. 如何定义数据存储的占位符问题
#    <br>解决：
#    1. 种群用n*4的矩阵存储，类型为`"int32"`
#    2. 最难的是染色体的存储，也是用n*4的矩阵，但是没法定义自己想要的任意为数字符长度，也就是任意位的二进制，比如`"000010101"`这种存储格式。<br>最后，发现numpy中有一种类型为`"<U5"`可以解决这个问题
#    3. 其他类似
# 2. 如何将十进制数与任意位数的二进制进行相互转化问题
#    <br>解决：
#    1. 找到一个`bin()`函数来将十进制数转化为二进制数，但是转化后的字符串会带有`"0b"`的前缀，需要进行切片消除
#    2. `int()`将二进制转为十进制就不用切片了
# 3. 用俄罗斯转盘来进行选择个体进行复制的时候，每一次都选取了固定位置问题
#    <br>解决：
#    错误的将选择代数的参数设置为本代，应该将代数选择为上一代
# 4. 在进行交叉操作时遇到的，无法进行交叉的问题
#    <br>解决：
#    1. 发现了在`numpy`中，字符串元素无法进行切片的修改，例如: `S[1,0][2:]=S[1,1][2:]`
#    2. 想要修改字符串元素，需要将要修改的部分切片然后重新组合再赋值。例如: `S[1,0]=S[1,0][:2]+S[1,1][2:]`
# 5. 如何动态的变化`numpy`数组的大小问题
#    <br>解决：
#    1. 一开始只找到了`append()`函数来增加`ndarray`的行数，但是需要增加相同维数的对象，这点难以处理
#    2. 后来找到了`nmupy.insert()`这个比较满意的函数，可以在矩阵中的任意位置插入任意维度的矩阵，也可以对列进行处理
# 6. 最费时间就是如何将过程用表格打印出来的问题,而且想把表格用图片的形式输出
#    <br>解决：
#    1. 一开始只是普通的打印出来，效果还可以，但是还是想用表格
#       ```
#       print("种群G", g, "：", S[g])
#       print("种群G", g, "的染色体：", S_chromosome[g])
#       print("种群G", g, "的适应度：", S_fitness[g])
#       print("种群G", g, "的选择概率：", S_selective_probability[g])
#       print("种群G", g, "的积累概率：", S_cumulative_probability[g])
#       ```
#    2. 然后就想`matplotlib.pyplot`里面会不会有表格绘制，但是找了很久都是统计图里面的表格，而不是单纯的表格。<br>没有我想要的效果。没法绘制，这就难搞了。
#    3. 然后就找到了`pandas`里面有一个`pivot_table()`函数，可以绘制表格。但是，很遗憾pandas只能读取外部文件的表格，然后再显示出来，有点类似数据库里面的sql语句的表达形式。我不懂怎么将自己的数据用这个函数来绘制表格。
#    4. 再然后就找到了一个新的库`prettytable`，这里面的`PrettyTable()`函数可以将数据用表格的形式打印出来，比较美观和容易上手，可惜的是没法生成图片的格式输出出来。但是最后还是用了这个函数进行表述遗传的过程。
#    5. 其实最后，找了一个可以实现目的的库`plotly.graph_objects`
#       效果如下：
#       ```
#       fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),
#                                cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))])
#       fig.show()
#       ```
#       ![image.png](attachment:image.png)
#       效果还是不错的，但是学习成本太高了，下次再学。

# In[ ]:




