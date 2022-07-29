---
author: kii
title: 文本增强技术
categories: [Exp]
tags: [NLP]
date: 2022-06-15 00:20:30
---

<Boxx changeTime="10000"/>

::: tip 前言
实验的一些东西。
:::
<!-- more -->

文本增强任务：-宋学林

# 常用文本增强技术：EDA、UDA、CBert、Lambada

低资源和冷启动问题

# 数据

标注图片费时费力；数据标注难免有错误

数据增强任务：Data augmentation

少量数据--(Data augmentation)->大量标注数据

## 图像领域

评议、旋转、剪裁、遮挡、反转、放缩、灰度

gnn

## 语音

音速扰动、音量扰动

频率遮蔽、时间遮蔽

加入噪声

## 文本增强

难度>图像和语音领域

微小的改动可能带来语义的改变

文本增强的核心：改变文本内容&保持标签不变

# 具体方法

## back-translation

标注文本->翻译->在翻译回来。

## EDA-easy data augmentation

四种操作：

1. 同义词替换：随机选n个同义词替换，(非停用词)
2. 随机插入：插入文本中某个非停用词的同义词
3. 随机交换：
4. 随机删除：按概率p随机删除

增强值$\alpha$

没论证是否能够保证标签不变。只是通过一个图显示影响不大

t-SNE 降维度

数据量大，提升效果不好。预训练复杂模型效果可能不行。

数据量少的时候，可能有几个点的提升

## Contextual Augmentation

1. 使用语言模型进行文本的替换
   
   1. 语言模型：用语言模型评价一句话是否合理或是人话
   2. 数学上讲：P（合理句子）>P（不合理句子）
   3. 用文本中前n个字预测下一个字

2. 语言模型结构：双向LSTM

3. 修改训练目标融入标签信息

4. 利用了语境信息

现有的方法：基于word-Net库，不好

把标签也序列化，融入到训练过程中，从而保证替换后不会对原有的标签有损伤。

## Conditional Bert

1. 使用Bert模型结构

bert-预训练

1. mask language model；依照一定的概率，用mask掩盖文本中的某个词，用剩下的预测这个mask的词
2. next sentence prediction：挨着的两句话为正样本，不挨着的为负样本

![image-20210706211307471](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706211307471.png)

  ![image-20210706211444673](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706211444673.png)

![image-20210706211637664](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706211637664.png)

Conditional bert data augmentation

如何保证标签不变？

把label 加入到输入中

训练的时候是使用的bert的结构

## LAMBADA

Do Not Have Enough Data? Deep Learning to the Rescue!  

基于generative pre-training 2（GPT2）

GPT也是深层的transformer模型，更复杂，深度更深

GPT-写东西的能力很强

![image-20210706215121764](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706215121764.png)

![image-20210706215153910](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706215153910.png)

# UDA-半监督学习

半监督学习：如何结合有标注数据，直接利用无标签数据

数据增强是：如花使用有标注数据，构造更多有标签数据

## 平滑假设

1. 如果两个输入样本相似，那么模型输出结果也应当相似

2. 对样本做某种很小的扰动，得到x2

3. 训练目标：调整模型w，使得w1、w2接近

4. 在这个过程中，y1和y2的实际值并不重要
   
   ![image-20210706220236419](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706220236419.png)
