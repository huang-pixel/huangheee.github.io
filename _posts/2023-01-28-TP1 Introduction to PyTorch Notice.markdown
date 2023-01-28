---
layout: post
title: Basic tensor manipulation
date: 2023-01-28 14:00
comments: true
external-url:
categories: Learn
---

## Basic tensor manipulation 

### 一、函数介绍 

#### [torch.matmul()函数用法总结](https://wendy.blog.csdn.net/article/details/121158666?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-121158666-blog-105783610.pc_relevant_vip_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-121158666-blog-105783610.pc_relevant_vip_default&utm_relevant_index=2)

pytorch中两个张量的乘法可以分为两种：

-   两个张量对应元素相乘，在PyTorch中可以通过**torch.mul函数**（或*运算符）实现；
-   两个张量矩阵相乘，在PyTorch中可以通过**torch.matmul函数**实现；

**torch.matmul(input, other) → Tensor**  
计算两个张量input和other的矩阵乘积  
【注意】：matmul函数没有强制规定维度和大小，可以用利用广播机制进行不同维度的相乘操作。

<!-- more -->

### 二、常见用法

#### 2.1 两个一维向量的乘积运算

若两个tensor都是一维的，则返回两个向量的点积运算结果：

```python
import torch
x = torch.tensor([1,2]) 
y = torch.tensor([3,4])
print(x,y)
print(torch.matmul(x,y),torch.matmul(x,y).size()) 
#tensor(11) torch.Size([])
```

![[Pasted image 20230128174927.png]]

#### 2.2 两个二维矩阵的乘积运算

若两个tensor都是二维的，则返回两个矩阵的矩阵相乘结果：

```python
import torch
x = torch.tensor([[1,2],[3,4]])
y = torch.tensor([[5,6,7],[8,9,10]])
print(torch.matmul(x,y),torch.matmul(x,y).size())
# tensor([[21, 24, 27],[47, 54, 61]]) torch.Size([2, 3])
```

![[Pasted image 20230128175119.png]]

