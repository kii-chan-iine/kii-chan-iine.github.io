---
author: kii
title: Python学习札记
categories: [CV]
tags: [CV,DL]
date: 2022-08-11 00:00:00
---

<Boxx changeTime="10000"/>

::: tip 前言

记录python学习的心得。

:::

<!-- more -->

# 生成随机数

random

```python
import random
random.seed(10) #如果不显示设置，则默认使用当前系统时间
random.sample(range(1000),10) #list
```

numpy--生成矩阵

```python
import numpy as np
print(np.random.rand(4,5))
print(np.random.random(size=(2,5))))
# 生成一个1到20之内的整数矩阵
print(np.random.randint(1,20,size=(3,4)))
```

torch

```python
import torch
torch.rand(2,3)
```

# File

```python

```


