---
author: kii
title: 深度学习基础知识
categories: [深度学习]
tags: [deeplearn]
date: 2021-07-19 20:00:00
---

<Boxx changeTime="10000"/>

::: tip 前言
这里主要讲深度学习的一些基础知识。
:::
<!-- more -->

# 基础知识

## 1 不同层的作用

### BN层

我们假设有一批图像的feature maps传入网络中（如上）。其中，N表示batch_size，9*9表示图像的大小，5表示channel。

BN做了一件什么事呢。

（1）把不同batch_size的同一个channel的feature map进行求均值，得到mean

（2）把不同batch的同一个channel的feature map进行求标准差，得到std

（3）最后对每一个channel的每一个feature map减去对应channel的mean，再除以std，就得到了新的N*9*9*5的feature maps

![img](https://img-blog.csdnimg.cn/20190601170251915.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjA3ODYxOA==,size_16,color_FFFFFF,t_70)

> BN层的的好处：
> 
> 1. Internal Covariate Shift(本层的输出会导致下一层的输入的分布发生变化，因而导致训练效果变差)
> 2. [减轻了梯度消失](https://blog.csdn.net/ygfrancois/article/details/90382459)和梯度爆炸的问题：
> 3. BN可以支持更多的激活函数
> 4. BN层一定程度上增加了 泛化能力

### LN层（LayerNormalization）

#### 理论

好，各位，刚才的任务完成了，我们进行下一项任务，名叫：LN。。。

（1）第一梯队的**所有通道的第一列**，听清楚了，是第一列，给到我你们的均值（mean）

（2）给完以后，给到我你们的标准差（std）

（3）然后：把你们的数值减去mean，再除以std

（4）接着我会给你们一个gamma，把结果乘上去；还有一个beta，加上去

（5）OK，第一梯队的所有通道的第一列，给我最终结果。

（6）接下来，第一梯队的所有通道的其他列，按照第一列的步骤，开始！

（7）其他梯队，按照第一梯队的流程，GO！

![img](https://img-blog.csdnimg.cn/20190601170311687.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjA3ODYxOA==,size_16,color_FFFFFF,t_70)

#### 实际

1）第一梯队的所有通道的第一列的第一行，听清楚了，是**第一列的第一行**，给到我你们的均值（mean）

（2）给完以后，给到我你们的标准差（std）

（3）然后：把你们的数值减去mean，再除以std

（4）接着我会给你们一个gamma，把结果乘上去；还有一个beta，加上去

（5）OK，第一梯队的所有通道的第一列的第一行，给我最终结果。

（6）接下来，第一梯队的所有通道的第一列的其他行，按照第一列第一行的步骤，开始！

（7）接下来，第一梯队的所有通道的其他列，按照第一列步骤，开始！

（8）其他梯队，按照第一梯队的流程，GO！

这就是（才是实战中的）LN

![img](https://img-blog.csdnimg.cn/20190601170337230.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjA3ODYxOA==,size_16,color_FFFFFF,t_70)

### IN层

好，接下来是最后一个任务了，IN

（1）第一梯队，给出你们所有通道所有行所有列的均值（mean）

（2）第一梯队，给出标准差（std）

（3）乘上gamma和beta，在给到我

（4）其他梯队，跟上！

（5）任务完成，开饭！

这就是IN

![img](https://img-blog.csdnimg.cn/20190601170351203.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjA3ODYxOA==,size_16,color_FFFFFF,t_70)

### 总结

BN做的事情是沿着channel方向，把每个channel的特征图做标准化

LN做的事情是沿着batch方向，同时也沿着time_step方向，把每一个单词做标准化

IN则很生猛，直接把单个输入特征图直接整体做标准化

因此呢，BN对CNN效果很好，因为CNN本身的目的就是结合不同batch_size的特征，做特征提取；LN对单个词做的标准化，对时序特征的效果特别好；IN是对整体单个输入做标准化，所以在风格迁移的时候，能对单一风格做到非常好的特征提取。

**BN的缺陷**

**缺陷如下：**

1、BN是在batch size样本上各个维度做标准化的，所以size越大肯定越能得出合理的μ和σ来做标准化，因此BN比较依赖**bath size**的大小。
2、在训练的时候，是分批量进行填入模型的，但是在预测的时候，如果只有一个样本或者很少量的样本来做inference，这个时候用BN显然偏差很大，例如在**线学习场景**。
3、RNN是一个动态的网络，也就是size是变化的，可大可小，造成多样本维度都没法对齐，所以不适合用BN。

**LN带来的优势：**

1、Layer Normalization是每个样本内部做标准化，跟size没关系，不受其影响。
2、RNN中LN也不受影响，内部自己做标准化，所以LN的应用面更广。

### 卷积层

详见各种卷积一文

### 池化层

#### 一.  池化层主要的作用

1. 首要作用，**下采样（downsamping）**
2. 降维、去除冗余信息、**对特征进行压缩、简化网络复杂度**、减少计算量、减少内存消耗等等。各种说辞吧，总的理解就是减少数量。
3. 实现非线性（这个可以想一下，relu函数，是不是有点类似的感觉？）。
4. 可以扩大感知野
5. 可以**实现不变性**，其中不变性包括，平移不变性、旋转不变性和尺度不变性。

#### 二.  池化主要有哪几种：

一般池化（General Pooling）：其中最常见的池化操作有平均池化和最大池化：

1. 平均池化（average pooling): 计算图像区域的平均值作为该区域池化后的值。

2. 最大池化（max pooling）： 选图像区域的最大值作为该区域池化后的值。

### 全连接层（FC层）



（1）全连接层（fully connected layers，FC）**在整个卷积神经网络中起到“分类器”的作用**。

（2）将学到的“分布式特征表示”映射到样本标记空间的作用。

（3）FC可一定程度保留模型复杂度

（4）卷积神经网络中全连接层的设计，属于人们在传统特征提取+分类思维下的一种"迁移学习"思想
函数

## 2 激活函数

所谓激活函数（Activation Function），就是在[人工神经网络](https://baike.baidu.com/item/人工神经网络/382460)的神经元上运行的[函数](https://baike.baidu.com/item/函数/301912)，负责将神经元的输入映射到输出端。

![img](https://bkimg.cdn.bcebos.com/pic/0eb30f2442a7d933739ff390a14bd11373f00119?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2U4MA==,g_7,xp_5,yp_5/format,f_auto)

#### Softmax

$$
S_i=\frac{e^i}{\sum e^j }
$$

映射区间[0,1],主要用于：离散化概率分布。

https://blog.csdn.net/bitcarmanlee/article/details/82320853

softmax函数，又称归一化指数函数。它**是二分类函数sigmoid在多分类上的推广**，目的是将多分类的结果以概率的形式展现出来。下图展示了softmax的计算方法：

softmax就是先把输出用指数表示，保证输出非负，然后在加权平均，获得0-1之间的预测结果概率。具体如下：

1）分子：通过指数函数，将实数输出映射到零到正无穷。

2）分母：将所有结果相加，进行归一化。

![image-20210716155825571](https://i.loli.net/2021/07/16/tvy3bHkmUTRO2QP.png)

之其中，这里的$W_yx$就是某个的输出结果(softmax之前)

#### Sigmoid

- 映射区间(0, 1)
- 也称logistic函数

$$
f(x)=\frac{1}{1+e^{-x}}
$$

![sigmoid函数](https://img-blog.csdnimg.cn/20181130114706469.gif)

- 映射区间(0, 1)
- 也称logistic函数
- 存在三个**问题**:
  1. 饱和的神经元会"杀死"梯度,指离中心点较远的x处的导数接近于0,停止反向传播的学习过程.
  2. sigmoid的输出不是以0为中心,而是0.5,这样在求权重w的梯度时,梯度总是正或负的.
  3. 指数计算耗时

#### Relu

<img src="https://bkimg.cdn.bcebos.com/pic/d788d43f8794a4c25b5e4dd902f41bd5ac6e39c6?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2U5Mg==,g_7,xp_5,yp_5/format,f_auto" alt="img" style="zoom:50%;" />

#### Leaky ReLUs

****

  ReLU是将所有的负值都设为零，相反，Leaky ReLU是给所有负值赋予一个非零斜率。Leaky ReLU激活函数是在声学模型（2013）中首次提出的。以数学的方式我们可以表示为：

  <img src="http://p0.ifengimg.com/pmop/2017/0701/CFC5A1C95A84A6D8CF3FFC1DD30597782AEEAE57_size20_w740_h231.jpeg" alt="img" style="zoom:33%;" />ai是（1，+∞）区间内的固定参数。

![img](http://p0.ifengimg.com/pmop/2017/0701/C56E5C6FCBB36E70BA5EBC90CBD142BA320B3DF6_size19_w740_h217.jpeg)

#### PRelu

负值部分的斜率是根据数据来定的

#### **RReLU**

**随机纠正线性单元（RReLU）**,训练的时候负数部分的斜率是不固定的。a_ji是从一个均匀的分布U(I,u)中随机抽取的数值

#### ELU

ELU函数公式和曲线如下图

![elu函数公式](https://img-blog.csdn.net/20180104121207844?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![elu函数图](https://img-blog.csdn.net/20180104121237935?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

是有一定的输出的，而且这部分输出还具有一定的抗干扰能力。这样可以消除ReLU死掉的问题，不过还是有梯度饱和和指数运算的问题。**ELU对于输入特征只定性不定量**。

#### SELU(就是在ELU前面乘以一个$\lambda$，并告诉你$\lambda，\alpha$是多少)

上面那个ELU，![α](https://math.jianshu.com/math?formula=%CE%B1)要设多少？后来又出现一种新的方法，叫做：SELU。它相对于ELU做了一个新的变化：就是现在把每一个值的前面都乘上一个![λ](https://math.jianshu.com/math?formula=%CE%BB)，然后他告诉你说![λ](https://math.jianshu.com/math?formula=%CE%BB)跟![α](https://math.jianshu.com/math?formula=%CE%B1)应该设多少，![α=1.67326324……](https://math.jianshu.com/math?formula=%CE%B1%3D1.67326324%E2%80%A6%E2%80%A6)，然后![λ=1.050700987……](https://math.jianshu.com/math?formula=%CE%BB%3D1.050700987%E2%80%A6%E2%80%A6)。

![img](https://upload-images.jianshu.io/upload_images/5631876-a163982aad9150ed.png?imageMogr2/auto-orient/strip|imageView2/2/w/404/format/webp)

#### <font color='red'>GELU</font>

GELU（高斯误差线性单元）是一个非初等函数形式的激活函数，是RELU的变种。由16年论文 [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) 提出，随后被GPT-2、BERT、RoBERTa、ALBERT 等NLP模型所采用。

#### 全家福

![ReLU系列对比](https://images2017.cnblogs.com/blog/606386/201711/606386-20171102101447857-1756364198.png)

#### [Swish](https://arxiv.org/abs/1710.05941)

还有一个新的激活函数叫做**Swish**。这个**Swish**激活函数长什么样子，它是一个非常神奇的激活函数，他把**sigmoid**乘上输入x得到她的output。

![image-20210716172123869](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210716172123869.png)

β是个常数或可训练的参数.Swish 具备无上界有下界、平滑、非单调的特性。
Swish 在深层模型上的效果优于 ReLU。

![image-20210722233255365](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210722233255365.png)

---

GELU 与 Swish 激活函数（x · σ(βx)）的函数形式和性质非常相像，一个是固定系数 1.702，另一个是可变系数 β（可以是可训练的参数，也可以是通过搜索来确定的常数），两者的实际应用表现也相差不大。

#### Hard-Swish

虽然这种Swish非线性提高了精度，但是在嵌入式环境中，他的成本是非零的，因为在移动设备上计算[sigmoid](https://so.csdn.net/so/search?q=sigmoid&spm=1001.2101.3001.7020)函数代价要大得多。
MobileNetV3 作者使用hard-Swish和hard-Sigmoid替换了ReLU6和SE-block中的Sigmoid层，但是**只是在网络的后半段才将ReLU6替换为h-Swish**，因为作者发现Swish函数只有在更深的网络层使用才能体现其优势。

<img src="http://latex.codecogs.com/gif.latex?H-Swish[x]=x\frac{ReLU6(x+3)}{6}" />

#### tanh

![image-20210716172707674](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210716172707674.png)

![蓝色sigmoid-红色tanh](https://images2018.cnblogs.com/blog/606386/201807/606386-20180712202915278-1408388561.png)

#### Maxout

类似个分段线型函数

Maxout可以看做是在深度学习网络中加入一层激活函数层,包含一个参数k.这一层相比ReLU,sigmoid等,其特殊之处在于增加了k个神经元,然后输出激活值最大的值.

#### 为什么tanh相比sigmoid收敛更快:

1. 梯度消失问题程度
   $tanh′(x)=1−tanh(x)^2∈(0,1)$
   
   sigmoid: $s′(x)=s(x)×(1−s(x))∈(0,1/4)$
   可以看出tanh(x)的<font color='red'>梯度消失问题比sigmoid要轻</font>.梯度如果过早消失,收敛速度较慢.
   
   <img src="https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210722233148119.png" alt="image-20210722233148119" style="zoom:67%;" />
   
   <img src="https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210722233101194.png" alt="image-20210722233101194" style="zoom: 67%;" />

2. <font color='red'>以零为中心的影响</font>
   如果当前参数(w0,w1)的最佳优化方向是(+d0, -d1),则根据反向传播计算公式,我们希望 x0 和 x1 符号相反。但是如果上一级神经元采用 Sigmoid 函数作为激活函数，sigmoid不以0为中心，输出值恒为正，那么我们无法进行最快的参数更新，而是走 Z 字形逼近最优解。[[4\]](https://www.cnblogs.com/makefile/p/activation-function.html#fn4)

### 激活函数的作用

1. 加入非线性因素

2. 充分组合特征
   
   **为什么ReLU,Maxout等能够提供网络的非线性建模能力？**它们看起来是分段线性函数，然而并不满足完整的线性要求：加法f(x+y)=f(x)+f(y)和乘法f(ax)=a×f(x)或者写作f(αx1+βx2)=αf(x1)+βf(x2)f(αx1+βx2)=αf(x1)+βf(x2)。非线性意味着得到的输出不可能由输入的线性组合重新得到（重现）。**假如网络中不使用非线性激活函数，那么这个网络可以被一个单层感知器代替得到相同的输出，**因为线性层加起来后还是线性的，可以被另一个线性函数替代。
   
   ![img](http://pic.rmb.bdstatic.com/5b6de25d9929e619784e016466aeb9379259.gif)

### <font color='red'>梯度消失与梯度爆炸</font>

​    在反向传播过程中需要对激活函数进行求导，如果导数大于1，那么随着网络层数的增加梯度更新将会朝着指数爆炸的方式增加这就是梯度爆炸。同样如果导数小于1，那么随着网络层数的增加梯度更新信息会朝着指数衰减的方式减少这就是梯度消失。因此，梯度消失、爆炸，其根本原因在于反向传播训练法则，属于先天不足。

**【<font color='red'>梯度消失</font>】**原因有：一是在**深层网络**中，二是采用了**不合适的损失函数**，比如sigmoid。当梯度消失发生时，接近于输出层的隐藏层由于其梯度相对正常，所以权值更新时也就相对正常，但是当越靠近输入层时，由于梯度消失现象，会导致靠近输入层的隐藏层权值更新缓慢或者更新停滞。这就导致在训练时，只等价于后面几层的浅层网络的学习。

**【<font color='red'>梯度爆炸</font>】**一般出现在**深层网络**和**权值初始化值太大**的情况下。在深层神经网络或循环神经网络中，<font color='blue'>**误差的梯度可在更新中累积相乘**</font>。如果网络层之间的**梯度值大于 1.0**，那么**重复相乘会导致梯度呈指数级增长**，梯度变的非常大，然后导致网络权重的大幅更新，并因此使网络变得不稳定。

#### 原因

##### 深层网络

深度网络是多层非线性函数的堆砌，整个深度网络可以视为是一个**复合的非线性多元函数**。（这些非线性多元函数其实就是每层的激活函数），那么对loss function求不同层的权值偏导，相当于应用梯度下降的链式法则，链式法则是一个连乘的形式，所以当层数越深的时候，梯度将以指数传播。

如果接近<font color='cyan'>输出层</font>的激活函数求导后梯度值大于1，那么层数增多的时候，最终求出的梯度很容易指数级增长，就会产生**梯度爆炸**；相反，如果小于1，那么经过链式法则的连乘形式，也会很容易衰减至0，就会产生**梯度消失**。

从深层网络角度来讲，不同的层学习的速度差异很大，表现为网络中靠近输出的层学习的情况很好，靠近输入的层学习的很慢，有时甚至训练了很久，前几层的权值和刚开始随机初始化的值差不多。因此，<font color='cyan'>梯度消失、爆炸，其根本原因在于反向传播训练法则，属于先天不足</font>。

##### **激活函数**

以下图的反向传播为例（假设每一层只有一个神经元且对于每一层![[公式]](https://www.zhihu.com/equation?tex=y_i%3D%5Csigma%5Cleft%28z_i%5Cright%29%3D%5Csigma%5Cleft%28w_ix_i%2Bb_i%5Cright%29)，其中![[公式]](https://www.zhihu.com/equation?tex=%5Csigma)为sigmoid函数）

![img](https://pic3.zhimg.com/80/v2-ea9beb6c28c7d4e89be89dc5f4cbae2e_720w.png)

可以推导出：

<img src="https://pic1.zhimg.com/80/v2-8e6665fb67f086c0864583caa48c8d30_720w.jpg" alt="img" style="zoom:67%;" />

原因看下图，sigmoid导数的图像。

![img](https://pic3.zhimg.com/80/v2-cd452d42a0f5dcad974098dda44c4622_720w.jpg)

如果使用sigmoid作为损失函数，其梯度是不可能超过0.25的，而我们初始化的网络权值![[公式]](https://www.zhihu.com/equation?tex=%7Cw%7C)通常都小于1，因此![[公式]](https://www.zhihu.com/equation?tex=%7C%5Csigma%27%5Cleft%28z%5Cright%29w%7C%5Cleq%5Cfrac%7B1%7D%7B4%7D)，因此对于上面的链式求导，层数越多，求导结果![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+C%7D%7B%5Cpartial+b_1%7D)越小，因而很容易发生梯度消失。

##### **初始化权重的值过大**

<img src="https://pic1.zhimg.com/80/v2-8e6665fb67f086c0864583caa48c8d30_720w.jpg" alt="img" style="zoom: 80%;" />

如上图所示，当![[公式]](https://www.zhihu.com/equation?tex=%7C%5Csigma%27%5Cleft%28z%5Cright%29w%7C%3E1)，也就是![[公式]](https://www.zhihu.com/equation?tex=w)比较大的情况。根据链式相乘(反向传播)可得，则前面的网络层比后面的网络层梯度变化更快，很容易发生梯度爆炸的问题。（再理解下)

#### 解决办法

梯度消失和梯度爆炸本质上是一样的，都是因为网络层数太深而引发的梯度反向传播中的连乘效应。

解决梯度消失、爆炸主要有以下几种方案：

##### 换用Relu、LeakyRelu、Elu等激活函数(**梯度大部分落在常数上**)

ReLu：让激活函数的导数为1

LeakyReLu：包含了ReLu的几乎所有有点，同时解决了ReLu中0区间带来的影响

ELU：和LeakyReLu一样，都是为了解决0区间问题，相对于来，elu计算更耗时一些（为什么）

具体可以看[关于各种激活函数的解析与讨论](#activation)

##### BatchNormalization

BN本质上是解决传播过程中的梯度问题，具体待补充完善，查看[BN](...)

##### ResNet残差结构

![img](https://pic4.zhimg.com/80/v2-68f5136f96c6ecce7ccc7b9e9a569f63_720w.jpg)

##### LSTM结构

**STM**全称是长短期记忆网络（long-short term memory networks），LSTM的结构设计可以改善RNN中的梯度消失的问题。主要原因在于LSTM内部复杂的“门”(gates)，如下图所示。

![img](https://pic1.zhimg.com/80/v2-2b5e5e1f76374c764d24ae5d70e94288_720w.jpg)

LSTM 通过它内部的“门”可以在接下来更新的时候“记住”前几次训练的”残留记忆“。

##### 预训练加finetunning

此方法来自Hinton在06年发表的论文上，其基本思想是每次训练一层隐藏层节点，将上一层隐藏层的输出作为输入，而本层的输出作为下一层的输入，这就是逐层预训练。

训练完成后，再对整个网络进行“微调（fine-tunning）”。

目前应用的不是很多了。

此方法相当于是找全局最优，然后整合起来寻找全局最优，但是现在<font color='red'>基本都是直接拿imagenet的预训练模型直接进行fine-tunning</font>。

##### 梯度剪切、正则

<font color='blue'>梯度剪切</font>，其思想是**设值一个剪切阈值，如果更新梯度时，梯度超过了这个阈值，那么就将其强制限制在这个范围之内**。这样可以防止梯度爆炸。

<font color='blue'>另一种防止梯度爆炸的手段是采用权重正则化</font>，正则化主要是通过**对网络权重做正则**来限制过拟合，但是根据正则项在损失函数中的形式：

可以看出，如果发生梯度爆炸，那么权值的范数就会变的非常大，反过来，通过限制正则化项的大小，也可以在一定程度上限制梯度爆炸的发生。

参考：

https://zhuanlan.zhihu.com/p/72589432

https://www.jianshu.com/p/3f35e555d5ba

https://baike.baidu.com/item/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98/22761355?fr=aladdin

https://zhuanlan.zhihu.com/p/51490163

## 3 优化算法

https://blog.csdn.net/qunnie_yi/article/details/80129952

https://blog.csdn.net/fengchao03/article/details/78208414

---

#### 1.<font color='red'>随机梯度下降（Stochastic Gradient Descent，SGD）</font>

<img src="http://latex.codecogs.com/gif.latex?\theta =\theta - \eta\frac{\partial}{\partial\theta_j}J(\theta)" />

通过每个样本来迭代更新一次，以损失很小的一部分精确度和增加一定数量的迭代次数为代价，换取了总体的优化效率的提升。增加的迭代次数远远小于样本的数量。

**缺点：**

对于参数比较敏感，需要注意参数的初始化
容易陷入局部极小值
当数据较多时，训练时间长
每迭代一步，都要用到训练集所有的数据。

#### 2.批量梯度下降（Batch gradient descent，BGD）

$$
\theta_j =\theta_j - \eta\frac{\partial}{\partial\theta_j}J(\theta)
$$

θ=θ−η⋅∇θJ(θ)
每迭代一步，都要用到训练集的所有数据，每次计算出来的梯度求平均
$$η$$代表学习率LR

每一步总是寻找使$$J$$下降最“陡”的方向,$$J$$是损失函数

#### 3. 小批量梯度下降（Mini Batch Gradient Descent，MBGD）

θ=θ−η⋅∇θJ(θ;x(i:i+n);y(i:i+n))
为了避免SGD和标准梯度下降中存在的问题，对每个批次中的n个训练样本，这种方法只执行一次更新。【每次更新全部梯度的平均值】

#### 4.指数加权平均的概念

![这里写图片描述](https://img-blog.csdn.net/2018070315244794?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![这里写图片描述](https://img-blog.csdn.net/20180703152534479?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
从这里我们就可已看出指数加权平均的名称由来，第100个数据其实是前99个数据加权和，而前面每一个数的权重呈现指数衰减，即越靠前的数据对当前结果的影响较小
![这里写图片描述](https://img-blog.csdn.net/20180703153101645?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**缺点：**存在开始数据的过低问题，可以通过偏差修正，但是在深度学习的优化算法中一般会忽略这个问题
![这里写图片描述](http://latex.codecogs.com/png.latex?%5Cdpi%7B200%7D%20%5Csmall%20V_%7Bt%7D=%28%5Cbeta%20V_%7Bt-1%7D+%5Cleft%20%28%201-%5Cbeta%20%5Cright%20%29%5Ctheta_%7Bt%7D%29/%281-%5Cbeta%20%5E%7Bt%7D%29)
当t不断增大时，分母逐渐接近1，影响就会逐渐减小了

**优点：**【相较于滑动窗口平均】
1.占用内存小，每次覆盖即可
2.运算简单

#### 5.Momentum（动量梯度下降法）

momentum是模拟物理里动量的概念，积累之前的动量来替代真正的梯度。公式如下：
![这里写图片描述](https://img-blog.csdn.net/20180703160108872?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
然而网上更多的是另外一种版本，即去掉（1-β）
![这里写图片描述](http://latex.codecogs.com/gif.latex?V_%7Bdw%7D=%5Cbeta%20V_%7Bdw%7D+dw)
相当于上一版本上本次梯度的影响权值*1/(1-β)
两者效果相当，只不过会影响一些最优学习率的选取
**优点**

- 下降初期时，使用上一次参数更新，下降方向一致，乘上较大的μ能够进行很好的加速
- 下降中后期时，在局部最小值来回震荡的时候，gradient→0，β得更新幅度增大，跳出陷阱
- 在梯度改变方向的时候，μ能够减少更新

即在正确梯度方向上加速，并且抑制波动方向张的波动大小，在后期本次计算出来的梯度会很小，以至于无法跳出局部极值，Momentum方法也可以帮助跳出局部极值
**参数设置**
β的常用值为0.9，即可以一定意义上理解为平均了前10/9次的梯度。
至于LR学习率的设置，后面所有方法一起总结吧

#### 6.Nesterov accelerated gradient (NAG)

![Momentum图解](https://img-blog.csdn.net/20170728165011954?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdHN5Y2NuaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![NAG图解](https://img-blog.csdn.net/20170803165730092?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdHN5Y2NuaA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](https://img-blog.csdn.net/20180703170914415?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**优点：**
这种基于预测的更新方法，使我们避免过快地前进，并提高了算法地响应能力，大大改进了 RNN 在一些任务上的表现【为什么对RNN好呢，不懂啊】
没有对比就没有伤害，NAG方法收敛速度明显加快。波动也小了很多。实际上NAG方法用到了二阶信息，所以才会有这么好的结果。*先按照原来的梯度走一步的时候已经求了一次梯度，后面再修正的时候又求了一次梯度，所以是二阶信息。*
**参数设置：**
同Momentum

其实，momentum项和nesterov项都是为了使梯度更新更加灵活，对不同情况有针对性。但是，人工设置一些学习率总还是有些生硬，接下来介绍几种**自适应学习率**的方法

#### 7.Adagrad

前面的一系列优化算法有一个共同的特点，就是对于每一个参数都用相同的学习率进行更新。但是在实际应用中各个参数的重要性肯定是不一样的，所以我们对于**不同的参数要动态的采取不同的学习率**，让目标函数更快的收敛。
adagrad方法是将每一个参数的每一次迭代的梯度取平方累加再开方，用基础学习率除以这个数，来做学习率的动态更新。【这样每一个参数的学习率就与他们的梯度有关系了，那么每一个参数的学习率就不一样了！也就是所谓的**自适应学习率**】
![这里写图片描述](https://img-blog.csdn.net/20180703171922378?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**优点：**

- 前期Gt较小的时候， regularizer较大，能够放大梯度
- 后期Gt较大的时候，regularizer较小，能够约束梯度
- 适合处理稀疏梯度:相当于为每一维参数设定了不同的学习率：压制常常变化的参数，突出稀缺的更新。能够更有效地利用少量有意义样本

**参数设置：**
只需要设置初始学习率，后面学习率会自我调整，越来越小

**缺点：**
Adagrad的一大优势时可以避免手动调节学习率，比如设置初始的缺省学习率为0.01，然后就不管它，另其在学习的过程中自己变化。当然它也有缺点，就是它计算时要在分母上计算梯度平方的和，由于所有的参数平方【上述公式推导中并没有写出来是梯度的平方，感觉应该是上文的公式推导忘了写】必为正数，这样就造成在训练的过程中，<font color='red'>分母累积的和会越来越大</font>。这样学习到后来的阶段，网络的更新能力会越来越弱，能学到的更多知识的能力也越来越弱，因为学习率会变得极其小【就会提前停止学习】，为了解决这样的问题又提出了Adadelta算法。

#### 8.Adadelta

Adagrad会累加之前所有的梯度平方，而Adadelta只累加固定大小的项【其实就是相当于指数滑动平均，只用了前多少步的梯度平方平均值】，并且也不直接存储这些项，仅仅是近似计算对应的平均值【这也就是指数滑动平均的优点】
![这里写图片描述](https://img-blog.csdn.net/20180703191607972?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**优点：**
不用依赖于全局学习率了
训练初中期，加速效果不错，很快
避免参数更新时两边单位不统一的问题
**缺点：**
训练后期，反复在局部最小值附近抖动

#### 9.<font color='red'>RMSprop（Root Mean Square Prop）</font>

![这里写图片描述](https://img-blog.csdn.net/2018070319350422?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**特点：**

- 其实RMSprop依然依赖于全局学习率

- RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间

- 适合处理非平稳目标（也就是与时间有关的）

- 对于RNN效果很好，因为RMSprop的更新只依赖于上一时刻的更新，所以适合。
  
  ![优化算法走的路线](https://img-blog.csdn.net/20170923134334368?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2lsbGR1YW4x/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### <font color='red'>10. [Adam](https://baijiahao.baidu.com/s?id=1668617930732883837&wfr=spider&for=pc)（可以减少内存的使用）</font>

Adam = Adaptive + Momentum，顾名思义Adam集成了**SGD的一阶动量**和**RMSProp的二阶动量**。
![这里写图片描述](https://img-blog.csdn.net/20180703194414535?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
**特点：**

- 结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点
- 对内存需求较小
- 为不同的参数计算不同的自适应学习率
- 也适用于大多非凸优化
- 适用于大数据集和高维空间

#### 11.Adamax

![这里写图片描述](https://img-blog.csdn.net/20180703194713886?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 12.Nadam

![这里写图片描述](https://img-blog.csdn.net/20180703194821135?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIzMjY5NzYx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 13.总结

提速可以归纳为以下几个方面：
\- 使用momentum来保持前进方向(velocity)；
\- 为每一维参数设定不同的学习率：在梯度连续性强的方向上加速前进；
\- 用历史迭代的平均值归一化学习率：突出稀有的梯度；

#### Keras中的默认参数

```
optimizers.SGD(lr=0.001,momentum=0.9)

optimizers.Adagrad(lr=0.01,epsilon=1e-8)

optimizers.Adadelta(lr=0.01,rho=0.95,epsilon=1e-8)

optimizers.RMSprop(lr=0.001,rho=0.9,epsilon=1e-8)

optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
```

## 4 损失函数（cost fun）/损失函数 （loss func）

### 二次代价函数

C表示代价，x表示样本，y表示实际值，a表示输出值，n表示样本的总数。

<img src="http://latex.codecogs.com/gif.latex?C=\frac{1}{2n}\sum_x||y(x)-a^L(x)||^2" />

二次代价函数的不足：

1. 误差越大，参数调整的幅度可能更小，训练更缓慢。（求导之后是与sigmoid函数的导数相关，（如果激活函数是sigmoid的化），那么误差越大，梯度更新越缓慢）
   
   z表示神经元的输入，![img](https://img-blog.csdn.net/20160402163620081)表示激活函数
   
   ![img](https://img-blog.csdn.net/20160402175137034)
   
   <img src="https://img-blog.csdn.net/20160402165516510" alt="img" style="zoom: 50%;" />

### 交叉熵CrossEntropyloss

注意公式中 ![[公式]](https://www.zhihu.com/equation?tex=x) 表示样本， ![[公式]](https://www.zhihu.com/equation?tex=y) 表示实际的标签， ![[公式]](https://www.zhihu.com/equation?tex=a) 表示预测的输出， ![[公式]](https://www.zhihu.com/equation?tex=n) 表示样本总数量。

<img src="http://latex.codecogs.com/gif.latex?C=-\frac{1}{n}\sum_x[ylna+(1-y)ln(1-a)]" />

![img](https://img-blog.csdn.net/20160402180457695)

![img](https://img-blog.csdn.net/20160402173528448)

所以，当误差越大，梯度就越大，参数w调整得越快，训练速度也就越快。

### MSEloss均方损失函数

$$
loss(x_i,y_i)=(x_i-y_i)^2
$$

### L1loss

### SmoothL1Loss

### BCELoss

### NLloss

### NLLoss2D

## 5 过拟合(详见面试总结)

### 原因

1. 样本问题：样本少，样本集合划分有问题
2. 样本噪声大
3. 假设的模型无法合理存在，或者说是假设成立的条件实际并不成立；
4. 参数太多，模型复杂
5. 学习迭代太多，学习了噪声
6. *对于决策树模型，如果我们对于其生长没有合理的限制，其自由生长有可能使节点只包含单纯的事件数据(event)或非事件数据(no event)，使其虽然可以完美匹配（拟合）训练数据，但是无法适应其他数据集。*

### 解决

1. L1（稀疏解）L2（平滑解）正则
2. 数据增强，扩样本，生成数据，重采样etc.
3. Early stopping
4. dropout：在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。
5. BN 
6. 交叉验证
7. 选择合适的网络
8. 模型组合（Bagging & Boosting)
9. 决策树剪枝（对应传统学习） 

## 6 模型加速

1. dataloader使用多个work（主要解决加载数据过慢，GPU空闲等待数据加载的情况。）

2. **硬件加速**：GPU

3. **超参数**：选取合适的初始值。调整学习率

4. **框架加速**(在同样的模型，同样的配置下，采用Tensorflow ，caffe，mxnet或者CNTK的速度也不一样。具体性能可以参考这篇文章http://mp.weixin.qq.com/s/Im2JWJYGBQbOfzikFrEMsA，相比来说caffe在多机多卡的GPU环境下，加速更明显。)

5. **模型选取**：当下比较经典的深度学习网络包括AlexNet，GoogleNet(Inception)，ResNet等。模型的层数越多，对硬件的要求越高，受限于GPU的缓存，每次mini batch的数量随层次增多而变少，训练时间越久，效果越差。 最近多伦多大学新提出的RevNet解决了这个问题，可以参考[多伦多大学联手Uber推出RevNet，不用存储激活便可实现反向传播](https://mp.weixin.qq.com/s?__biz=MzI3NjY4NDA1Ng==&mid=2247489374&idx=1&sn=911650349612c8fc42cf92d666cd78ee&source=41#wechat_redirect)。 如果是自己搭建模型，在梯度下降的算法当年可以考虑Adam梯度下降。

6. **数据策略**：数据归一化。在梯度下降算法中，数据尺度的不统一，会导致小尺度维度的梯度下降缓慢，延长迭代轮数。为此，可以采用减去平均值，除以方差的方式标准化输入。当样本量足够大时，使用mini-batch代替batch。mini-batch一般选取64-512，1024比较少见。最好是2的n次方，而且要和CPU/GPU相匹配。

## 7 机器学习中正则化项L1和L2的直观理解

https://blog.csdn.net/program_developer/article/details/80867468

### 正则化（Regularization）

机器学习中几乎都可以看到损失函数后面会添加一个额外项，常用的额外项一般有两种，一般英文称作 ***\*ℓ 1 \ell_1ℓ1\**-norm** 和 ***\*ℓ 2 \ell_2ℓ2\**-norm**，中文称作 ***L1正则化*** 和 ***L2正则化***，或者 ***L1范数*** 和 ***L2范数***。

L1正则化和L2正则化可以看做是损失函数的惩罚项。所谓『惩罚』是指对损失函数中的某些参数做一些限制。对于线性回归模型，使用L1正则化的模型建叫做Lasso回归，使用L2正则化的模型叫做Ridge回归（岭回归）。下图是Python中Lasso回归的损失函数，式中加号后面一项α ∣ ∣ w ∣ ∣ 1 \alpha||w||_1*α*∣∣*w*∣∣1即为L1正则化项。

![lasso regression](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTYwOTA0MTg0MjI4MTU4?x-oss-process=image/format,png#pic_center)

下图是Python中Ridge回归的损失函数，式中加号后面一项α ∣ ∣ w ∣ ∣ 2 2 \alpha||w||_2^2*α*∣∣*w*∣∣22即为L2正则化项。

![ridge regression](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTYwOTA0MTg0MzE0MzMz?x-oss-process=image/format,png#pic_center)

一般回归分析中w w*w*表示特征的系数，从上式可以看到正则化项是对系数做了处理（限制）。**L1正则化和L2正则化的说明如下：**

- L1正则化是指权值向量w w*w*中各个元素的**绝对值之和**，通常表示为∣ ∣ w ∣ ∣ 1 ||w||_1∣∣*w*∣∣1
- L2正则化是指权值向量w w*w*中各个元素的**平方和然后再求平方根**（可以看到Ridge回归的L2正则化项有平方符号），通常表示为∣ ∣ w ∣ ∣ 2 ||w||_2∣∣*w*∣∣2

一般都会在正则化项之前添加一个系数，Python的机器学习包`sklearn`中用α \alpha*α*表示，一些文章也用λ \lambda*λ*表示。这个系数需要用户指定。

那添加L1和L2正则化有什么用？**下面是L1正则化和L2正则化的作用**，这些表述可以在很多文章中找到。

- L1正则化可以产生稀疏权值矩阵，即产生一个稀疏模型，可以用于特征选择
- L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合

#### 稀疏模型与特征选择的关系

上面提到L1正则化有助于生成一个稀疏权值矩阵，进而可以用于特征选择。为什么要生成一个稀疏矩阵？

稀疏矩阵指的是很多元素为0，只有少数元素是非零值的矩阵，即得到的线性回归模型的大部分系数都是0. 通常机器学习中特征数量很多，例如文本处理时，如果将一个词组（term）作为一个特征，那么特征数量会达到上万个（bigram）。在预测或分类时，那么多特征显然难以选择，但是如果代入这些特征得到的模型是一个稀疏模型，表示**只有少数特征对这个模型有贡献，绝大部分特征是没有贡献的，或者贡献微小**（因为它们前面的系数是0或者是很小的值，即使去掉对模型也没有什么影响），此时我们就可以只关注系数是非零值的特征。这就是稀疏模型与特征选择的关系。

### L1和L2正则化的直观理解

这部分内容将解释**为什么L1正则化可以产生稀疏模型（L1是怎么让系数等于零的）**，以及**为什么L2正则化可以防止过拟合**。

#### 正则化和特征选择的关系

假设有如下带L1正则化的损失函数：
J = J 0 + α ∑ w ∣ w ∣ (1) J = J_0 + \alpha \sum_w{|w|} \tag{1}*J*=*J*0​+*α**w*∑​∣*w*∣(1)
其中J 0 J_0*J*0​是原始的损失函数，加号后面的一项是L1正则化项，α \alpha*α*是正则化系数。注意到L1正则化是权值的**绝对值之和**，J J*J*是带有绝对值符号的函数，因此J J*J*是不完全可微的。机器学习的任务就是要通过一些方法（比如梯度下降）求出损失函数的最小值。当我们在原始损失函数J 0 J_0*J*0​后添加L1正则化项时，相当于对J 0 J_0*J*0​做了一个约束。令L = α ∑ w ∣ w ∣ L = \alpha \sum_w{|w|}*L*=*α*∑*w*​∣*w*∣，则J = J 0 + L J = J_0 + L*J*=*J*0​+*L*，此时我们的任务变成**在\**L L\*L\*\**约束下求出\**J 0 J_0\*J\*0​\**取最小值的解**。**考虑二维的情况**，即只有两个权值w 1 w^1*w*1和w 2 w^2*w*2，此时L = ∣ w 1 ∣ + ∣ w 2 ∣ L = |w^1|+|w^2|*L*=∣*w*1∣+∣*w*2∣。对于梯度下降法，求解J 0 J_0*J*0​的过程可以画出等值线，同时L1正则化的函数L L*L*也可以在w 1 w 2 w^1w^2*w*1*w*2的二维平面上画出来。如下图：

![@图1 L1正则化](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTYwOTA0MTg0NDI4NDU5?x-oss-process=image/format,png#pic_center)
图1 L1正则化

图中等值线是J 0 J_0*J*0的等值线，黑色方形是L L*L*函数的图形。L = ∣ w 1 ∣ + ∣ w 2 ∣ L = |w^1|+|w^2|*L*=∣*w*1∣+∣*w*2∣，这个函数画出来就是一个方框（可以自己动手画一下）。

在图中，当J 0 J_0*J*0等值线与L L*L*图形首次相交的地方就是最优解。上图中J 0 J_0*J*0与L L*L*在L L*L*的一个顶点处相交，这个顶点就是最优解。注意到这个顶点的值是( w 1 , w 2 ) = ( 0 , w ) (w^1, w^2) = (0, w)(*w*1,*w*2)=(0,*w*)。可以直观想象，因为L L*L*函数有很多『突出的角』（二维情况下四个，多维情况下更多），J 0 J_0*J*0与这些角接触的机率会远大于与L L*L*其它部位接触的机率（这是很直觉的想象，突出的角比直线的边离等值线更近写），而在这些角上，会有很多权值等于0（因为角就在坐标轴上），这就是为什么L1正则化可以产生稀疏模型，进而可以用于特征选择。

而正则化前面的系数α \alpha*α*，可以控制L L*L*图形的大小。α \alpha*α*越小，L L*L*的图形越大（上图中的黑色方框）；α \alpha*α*越大，L L*L*的图形就越小，可以小到黑色方框只超出原点范围一点点，这是最优点的值( w 1 , w 2 ) = ( 0 , w ) (w1,w2)=(0,w)(*w*1,*w*2)=(0,*w*)中的w w*w*可以取到很小的值。

类似地，假设有如下带L2正则化的损失函数：

J = J 0 + α ∑ w w 2 (2) J = J_0 + \alpha \sum_w{w^2} \tag{2}*J*=*J*0+*α**w*∑*w*2(2)

同样可以画出他们在二维平面上的图形，如下：

![@图2 L2正则化](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTYwOTA0MTg0NjQ2OTYz?x-oss-process=image/format,png#pic_center)
图2 L2正则化

二维平面下L2正则化的函数图形是个圆（绝对值的平方和，是个圆），与方形相比，被磨去了棱角。因此J 0 J_0*J*0与L L*L*相交时使得w 1 w^1*w*1或w 2 w^2*w*2等于零的机率小了许多（这个也是一个很直观的想象），这就是为什么L2正则化不具有稀疏性的原因，因为不太可能出现多数w w*w*都为0的情况。

##### 为什么梯度下降的等值线与正则化函数第一次交点是最优解？

评论中有人问到过这个问题，这是带约束的最优化问题。这应该是在大一的高等数学就学到知识点，因为这里要用到拉格朗日乘子。如果有这样的问题，就需要复习一下高等数学了。这里有一个比较详细的数学讲解，可以参考：[带约束的最优化问题](https://blog.csdn.net/NewThinker_wei/article/details/52857397)。

#### L2正则化和过拟合的关系

拟合过程中通常都倾向于让权值尽可能小，最后构造一个所有参数都比较小的模型。因为一般认为参数值小的模型比较简单，能适应不同的数据集，也在一定程度上避免了过拟合现象。可以设想一下对于一个线性回归方程，若参数很大，那么只要数据偏移一点点，就会对结果造成很大的影响；但如果参数足够小，数据偏移得多一点也不会对结果造成什么影响，专业一点的说法是『抗扰动能力强』。

**那为什么L2正则化可以获得值很小的参数？**

以线性回归中的梯度下降法为例，使用Andrew Ng机器学习的参数表示方法。假设要求解的参数为θ \theta*θ*，h θ ( x ) h_\theta(x)*h**θ*(*x*)是我们的假设函数。线性回归一般使用平方差损失函数。单个样本的平方差是( h θ ( x ) − y ) 2 (h_\theta(x) - y)^2(*h**θ*(*x*)−*y*)2，如果考虑所有样本，损失函数是对每个样本的平方差求和，假设有m m*m*个样本，线性回归的代价函数如下，为了后续处理方便，乘以一个常数1 2 m \frac{1}{2m}2*m*1：

J ( θ ) = 1 2 m ∑ i = 1 m ( h θ ( x ( i ) ) − y ( i ) ) 2 (3) J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})^2 \tag{3}*J*(*θ*)=2*m*1*i*=1∑*m*(*h**θ*(*x*(*i*))−*y*(*i*))2(3)

在梯度下降算法中，需要先对参数求导，得到梯度。梯度本身是上升最快的方向，为了让损失尽可能小，沿梯度的负方向更新参数即可。

对于单个样本，先对某个参数θ j \theta_j*θ**j*求导：

∂ ∂ θ j J ( θ ) = 1 m ( h θ ( x ) − y ) ∂ ∂ θ j h θ ( x ) (3.1) \frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} (h_\theta(x) - y) \frac{\partial}{\partial \theta_j} h_\theta(x) \tag{3.1}∂*θ**j*∂*J*(*θ*)=*m*1(*h**θ*(*x*)−*y*)∂*θ**j*∂*h**θ*(*x*)(3.1)

注意到h θ ( x ) h_\theta(x)*h**θ*(*x*)的表达式是h θ ( x ) = θ 0 x 0 + θ 1 x 1 + ⋯ + θ n x n h_\theta(x)=\theta_0 x_0 + \theta_1 x_1 + \dots + \theta_n x_n*h**θ*(*x*)=*θ*0*x*0+*θ*1*x*1+⋯+*θ**n**x**n*. 单个样本对某个参数θ j \theta_j*θ**j*求导，∂ ∂ θ j h θ ( x ) = x j \frac{\partial}{\partial \theta_j} h_\theta(x) = x_j∂*θ**j*∂*h**θ*(*x*)=*x**j*. 最终(3.1)式结果如下：

∂ ∂ θ j J ( θ ) = 1 m ( h θ ( x ) − y ) x j (3.2) \frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} (h_\theta(x) - y) x_j \tag{3.2}∂*θ**j*∂*J*(*θ*)=*m*1(*h**θ*(*x*)−*y*)*x**j*(3.2)

在考虑所有样本的情况，将每个样本对θ j \theta_j*θ**j*的导数求和即可，得到下式：

∂ ∂ θ j J ( θ ) = 1 m ∑ i = 1 m ( h θ ( x ( i ) ) − y ( i ) ) x j ( i ) (3.3) \frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \tag{3.3}∂*θ**j*∂*J*(*θ*)=*m*1*i*=1∑*m*(*h**θ*(*x*(*i*))−*y*(*i*))*x**j*(*i*)(3.3)

梯度下降算法中，为了尽快收敛，会沿梯度的负方向更新参数，因此在(3.3)式前添加一个负号，并乘以一个系数α \alpha*α*（即学习率），得到最终用于迭代计算参数θ j \theta_j*θ**j*的形式：

θ j : = θ j − α 1 m ∑ i = 1 m ( h θ ( x ( i ) ) − y ( i ) ) x j ( i ) (4) \theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \tag{4}*θ**j*:=*θ**j*−*α**m*1*i*=1∑*m*(*h**θ*(*x*(*i*))−*y*(*i*))*x**j*(*i*)(4)

其中α \alpha*α*是学习率（learning rate）。 上式是没有添加L2正则化项的迭代公式，如果在原始代价函数之后添加L2正则化，则迭代公式会变成下面的样子：
θ j : = θ j ( 1 − α λ m ) − α 1 m ∑ i = 1 m ( h θ ( x ( i ) ) − y ( i ) ) x j ( i ) (5) \theta_j := \theta_j(1-\alpha \frac{\lambda}{m}) - \alpha \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \tag{5}*θ**j*:=*θ**j*​(1−*α**m**λ*​)−*α**m*1​*i*=1∑*m*​(*h**θ*​(*x*(*i*))−*y*(*i*))*x**j*(*i*)​(5)

其中λ \lambda*λ***就是正则化参数**。从上式可以看到，与未添加L2正则化的迭代公式相比，**每一次迭代，\**θ j \theta_j\*θ\*\*j\*\**都要先乘以一个小于1的因子**（即( 1 − α λ m ) (1-\alpha \frac{\lambda}{m})(1−*α**m**λ*)），从而使得θ j \theta_j*θ**j*不断减小，因此总的来看，θ \theta*θ*是不断减小的。

最开始也提到L1正则化一定程度上也可以防止过拟合。之前做了解释，当L1的正则化系数很小时，得到的最优解会很小，可以达到和L2正则化类似的效果。

### 正则化参数的选择

#### L1正则化参数

通常越大的λ \lambda*λ*可以让代价函数在参数为0时取到最小值。因为正则化系数越大，正则化的函数图形（上文图中的方形或圆形）会向坐标轴原点收缩得越厉害，这个现象称为**shrinkage**，过程可以称为**shrink to zero**. 下面是一个简单的例子，这个例子来自[Quora上的问答](https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization/answer/Kenneth-Tran?srid=CZEe)。为了方便叙述，一些符号跟这篇帖子的符号保持一致。

假设有如下带L1正则化项的代价函数：

F ( x ) = f ( x ) + λ ∣ ∣ x ∣ ∣ 1 F(x) = f(x) + \lambda ||x||_1*F*(*x*)=*f*(*x*)+*λ*∣∣*x*∣∣1

其中x x*x*是要估计的参数，相当于上文中提到的w w*w*以及θ \theta*θ*. 这个例子中的正则化函数L L*L*就是L = λ ∣ x ∣ L=\lambda |x|*L*=*λ*∣*x*∣。注意到L1正则化在某些位置是不可导的，当λ \lambda*λ*足够大时可以使得F ( x ) F(x)*F*(*x*)在x = 0 x = 0*x*=0时取到最小值。如下图：

![@图3 L1正则化参数的选择](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTYwOTA0MTg0ODAyNjY4?x-oss-process=image/format,png#pic_center)
图3 L1正则化参数的选择

作为一个直观的例子，这个图的示例中，取了f ( x ) = ( x − 1 ) 2 f(x) = (x-1)^2*f*(*x*)=(*x*−1)2作为损失函数，其实可以取更复杂的，但不好画图，不过原理是一样的，因为损失函数都是凸函数，很多性质是一样的。

正则化分别取λ = 0.5 \lambda = 0.5*λ*=0.5和λ = 2 \lambda = 2*λ*=2，可以看到越大的λ \lambda*λ*越容易使F ( x ) F(x)*F*(*x*)在x = 0 x=0*x*=0时取到最小值。

此外也可以自己计算一下，当损失函数f ( x ) f(x)*f*(*x*)和正则化函数L = ∣ x ∣ L=|x|*L*=∣*x*∣在定义域内第一次相交的地方，就是整个代价函数F ( x ) F(x)*F*(*x*)的最优解。

#### L2正则化参数

从公式5可以看到，λ \lambda*λ*越大，θ j \theta_j*θ**j*衰减得越快。另一个理解可以参考图2，λ \lambda*λ*越大，L2圆的半径越小，最后求得代价函数最值时各参数也会变得很小，同样是一个shrink to zero的过程，原理与L1正则化类似。

### Reference

过拟合的解释：
https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s5ss2.html

正则化的解释：
https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s5ss1.html

正则化的解释：
http://blog.csdn.net/u012162613/article/details/44261657

正则化的数学解释（一些图来源于这里）：
http://blog.csdn.net/zouxy09/article/details/24971995
