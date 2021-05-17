---
layout: post
title: 深度学习的一些归纳
date: 2019-11-10
Author: Kii. Chan
categories:
tags: [DeepLearning]
comments: true
---



# 一. 卷积

## 1. 二维卷积

![img-1](https://github.com/kii-chan-iine/Kii.Chan/blob/master/_posts/imgs/deep1.png?raw=true)



- 上图图像的维度为14×14，过滤器大小为5×5，卷积后输出的数据维度为10×10（14−5+1=10）。

- 上述内容没有引入channel的概念，也可以说channel的数量为1。如果将二维卷积中输入的channel的数量变为3，即输入的数据维度变为（14×14×3）。由于卷积操作中过滤器的channel数量必须与输入数据的channel数量**相同**，过滤器大小也变为5×5×3。在卷积的过程中，过滤器与数据在channel方向分别卷积，之后将卷积后的数值相加，即执行10×10次3个数值相加的操作，最终输出的数据维度为10×10。

- 以上都是在过滤器数量为1的情况下所进行的讨论。如果将过滤器的数量增加至16，即16个大小为10×10×3的过滤器，最终输出的数据维度就变为10×10×16。可以理解为分别执行每个过滤器的卷积操作，最后将每个卷积的输出在第三个维度（channel 维度）上进行拼接。

- 二维卷积常用于计算机视觉、图像处理领域。

  

#### Padding
Padding：如果你看到上面的动画，那么会注意到在卷积核滑动的过程中，边缘基本会被「裁剪」掉，将 5*5 特征矩阵转换为 3*3 的特征矩阵。使用padding之后，可以允许边缘在卷积之中。

![img-3](https://github.com/kii-chan-iine/Kii.Chan/blob/master/_posts/imgs/deep2.gif?raw=true)


#### Strides

   Striding： 是改变卷积核的移动步长跳过一些像素。 Stride 是 1 表示卷积核滑过每一个相距是 1 的像素，是最基本的单步滑动，作为标准卷积模式。Stride 是 2 表示卷积核的移动步长是 2，跳过相邻像素，图像缩小为原来的 1/2。 

  ![img-4](https://github.com/kii-chan-iine/Kii.Chan/blob/master/_posts/imgs/deep3.gif?raw=true)

## 2. 一维卷积

![img-5](https://github.com/kii-chan-iine/Kii.Chan/blob/master/_posts/imgs/deep4.png?raw=true)

- 图中的输入的数据维度为8，过滤器的维度为5。与二维卷积类似，卷积后输出的数据维度为8−5+1=4。
- 如果过滤器数量仍为1，输入数据的channel数量变为16，即输入数据维度为8×16。这里channel的概念相当于自然语言处理中的embedding，而该输入数据代表8个单词，其中每个单词的词向量维度大小为16。在这种情况下，过滤器的维度由5变为5×16，最终输出的数据维度仍为4。
- 如果过滤器数量为n，那么输出的数据维度就变为4×n。
- 一维卷积常用于序列模型，自然语言处理领域。

## 3. 三维卷积

![img-6](https://github.com/kii-chan-iine/Kii.Chan/blob/master/_posts/imgs/deep5.png?raw=true)


这里采用代数的方式对三维卷积进行介绍，具体思想与一维卷积、二维卷积相同。

- 假设输入数据的大小为a1×a2×a3，channel数为c，过滤器大小为f，即过滤器维度为f×f×f×c（一般不写channel的维度），过滤器数量为nn。
- 基于上述情况，三维卷积最终的输出为(a1−f+1)×(a2−f+1)×(a3−f+1)×n。该公式对于一维卷积、二维卷积仍然有效，只有去掉不相干的输入数据维度就行。
- 三维卷积常用于医学领域（CT影响），视频处理领域（检测动作及人物行为）。