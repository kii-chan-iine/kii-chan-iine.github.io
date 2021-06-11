---
author: kii
title: Something for Deeplearn
categories: [深度学习]
tags: [deeplearn]
date: 2021-06-03 20:00:00
---

<Boxx changeTime="10000"/>

::: tip 前言
这里主要讲深度学习的一些基础知识。
:::
<!-- more -->



# 深度学习的思考

在VGG中，卷积网络达到了19层，在GoogLeNet中，网络史无前例的达到了22层。那么，网络的精度会随着网络的层数增多而增多吗？在深度学习中，网络层数增多一般会伴着下面几个问题

1. 计算资源的消耗
2. 模型容易过拟合
3. 梯度消失/梯度爆炸问题的产生

问题1可以通过GPU集群来解决，对于一个企业资源并不是很大的问题；问题2的过拟合通过采集海量数据，并配合Dropout正则化等方法也可以有效避免；问题3通过Batch Normalization也可以避免。貌似我们只要无脑的增加网络的层数，我们就能从此获益，但实验数据给了我们当头一棒。

# 收敛性

1. 数据库太小一般不会带来不收敛的问题，只要你一直在train总会收敛（rp问题跑飞了不算）。**反而不收敛一般是由于样本的信息量太大导致网络不足以fit住整个样本空间**。**样本少只可能带来过拟合的问题**，你看下你的training set上的loss收敛了吗？如果只是validate set上不收敛那就说明overfitting了，这时候就要考虑各种anti-overfit的trick了，比如dropout，SGD，增大minibatch的数量，减少fc层的节点数量，momentum，finetune等。
2. .learning rate设大了会带来跑飞（loss突然一直很大）的问题，这个是新手最常见的情况——为啥网络跑着跑着看着要收敛了结果突然飞了呢？**可能性最大的原因是你用了relu作为激活函数的同时使用了softmax或者带有exp的函数做分类层的loss函数**。当某一次训练传到最后一层的时候，某一节点激活过度（比如100），那么exp(100)=Inf，发生溢出，bp后所有的weight会变成NAN，然后从此之后weight就会一直保持NAN，于是loss就飞起来啦。在做GNN实验的时候，经常遇到准确率突然下降的情况，自己也发现不了原因，因为准确率一直不错，索性就一直保留着这个为题，如图，可以看到期间一共跑飞过两次，因为学习率设的并不是非常大所以又拉了回来。如果lr设的过大会出现跑飞再也回不来的情况。这时候你停一下随便挑一个层的weights看一看，很有可能都是NAN了。对于这种情况建议用二分法尝试。0.1~0.0001.不同模型不同任务最优的lr都不一样。

    ![https://img-blog.csdnimg.cn/20190603225532802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNTQ3OTEwOA==,size_16,color_FFFFFF,t_70](https://img-blog.csdnimg.cn/20190603225532802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNTQ3OTEwOA==,size_16,color_FFFFFF,t_70)

3. 尽量收集更多的数据。有个方法是爬flickr，找名人标签，然后稍微人工剔除一下就能收集一套不错的样本。其实收集样本不在于多而在于hard，比如你收集了40张基本姿态表情相同的同一个人的图片不如收集他的10张不同表情的图片。之前做过试验，50张variance大的图per person和300多张类似的图per person训练出来的模型后者就比前者高半个点。
4. 尽量用小模型。如果**数据太少尽量缩小模型复杂度**。考虑减少层数或者减少**kernel numbe**r。

# BN层

为什么提出BN？

深度网络在采用Mini-Batch SGD训练的过程中，隐藏层激活函数的输入分布变化大，导致模型收敛慢。

[https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D%3D%5Cgamma%5Cfrac%7Bx-%5Cmu%7D%7B%5Csqrt%7B%5Csigma%5E%7B2%7D-%5Cvarepsilon%7D%7D%2B%5Cbeta%5C%5C](https://www.zhihu.com/equation?tex=%5Ctilde%7Bx%7D%3D%5Cgamma%5Cfrac%7Bx-%5Cmu%7D%7B%5Csqrt%7B%5Csigma%5E%7B2%7D-%5Cvarepsilon%7D%7D%2B%5Cbeta%5C%5C)

**Batch Normalization参数的形状?**

对feature map的channel方向求均值和方差, 假设feature map.shape=(b,c,w,h)，那么均值和方差的形状为( 1 , c , 1 , 1)，

和

的形状分别也是( 1 , c , 1 , 1)，因此于一层BN层可学习的参数数量为2c。

**Batch Normalization的好处**

- **解决了Internal Covariate Shift的问题**：前人采用**很小的学习率/非常小心的权重初始化**来解决Internal Covariate Shift的问题，BN解决了Internal Covariate Shift问题之后，就可以采用较大的学习率，能更快收敛
- BN减轻了梯度消失，梯度爆炸问题：[详见](https://link.zhihu.com/?target=https%3A//blog.csdn.net/ygfrancois/article/details/90382459)
- BN可支持更多的激活函数
- BN一定程度上增加了泛化能力，dropout等技术可以去掉。

# Resnet

一般情况下，模型退化主要有以下几种原因：

- 过拟合，层数越多，参数越复杂，泛化能力弱
- 梯度消失/梯度爆炸，层数过多，梯度反向传播时由于链式求导连乘使得梯度过大或者过小，使得梯度出现消失/爆炸，对于这种情况，可以通过BN(batch normalization)可以解决
- 由深度网络带来的退化问题，一般情况下，网络层数越深越容易学到一些复杂特征，理论上模型效果越好，但是由于深层网络中含有大量非线性变化，每次变化相当于丢失了特征的一些原始信息，从而导致层数越深退化现象越严重。

![%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%80%83%20b7cca7604a46493bb8be335c2b5b92d0/Untitled.png](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%80%83%20b7cca7604a46493bb8be335c2b5b92d0/Untitled.png)

残差块的计算方式为：$F ( x ) = W 2 ⋅ r e l u ( W 1 x )$ 
残差块的输出为:    $r e l u ( H ( x ) ) = r e l u ( F ( x ) + x )$

**残差块误差优化：** 残差网络通过加入 shortcut connections（或称为 skip connections），变得更加容易被优化。在不用skip连接之前，假设输入是x ，最优输出是x，此时的优化目标是预测输出H ( x ) = x ，加入skip连接后，优化输出H ( x ) 与输入x 的差别，即为残差F ( x ) = H ( x ) − x，此时的优化目标是F(x)的输出值为0。后者会比前者更容易优化。
**用残差更容易优化**：引入残差后的映射对输出的变化更敏感。设$H_{1}(x)$是加入skip连接前的网络映射$H_{2}(x)$是加入skip连接的网络映射。对于输入x = 5，设此时$H_{1}(5)=5.1,H_2(x)=5.1$,那么$H_{2}(5)=F(5)+5,F(5)=0.1$。当输出变为5.2时，F(x)由0.1变为0.2，明显后者输出变化对权重的调整作用更大，所以效果更好。残差的思想都是去掉相同的主体部分，从而突出微小的变化。
简单的加法不会给网络增加额外的参数和计算量，同时可以大大增加模型的训练速度，提高训练效果。并且当模型的层数加深时，能够有效地解决退化问题。
**残差网络为什么是有效的**：对于大型的网络，无论把残差块添加到神经网络的中间还是末端，都不会影响网络的表现。因为可以给残差快中的weight设置很大的L2正则化水平，使得$F(x)=0$，这样使得加入残差块至少不会使得网络变差，此时的残块等价于恒等映射。若此时残差块中的weight学到了有用的信息，那就会比恒等映射更好，对网络的性能有帮助。
总结： ResNet有很多旁路支线可以将输入直接连到后面的层，使得后面的层可以直接学习残差，简化了学习难度。传统的卷积层和全连接层在信息传递时，或多或少会存在信息丢失，损耗等问题。**ResNet将输入信息绕道传到输出，保护了信息的完整性.**

# nn.Sequential

```python
# hyper parameters
in_dim=1
n_hidden_1=1
n_hidden_2=1
out_dim=1

class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()

      	self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), 
            nn.ReLU(True)，
            nn.Linear(n_hidden_1, n_hidden_2)，
            nn.ReLU(True)，
            # 最后一层不需要添加激活函数
            nn.Linear(n_hidden_2, out_dim)
             )

  	def forward(self, x):
      	x = self.layer(x)
      	return x
#其实这个Sequential就是相当于把里面的东西打包了，将网络层和激活函数结合起来。
```

# 激活函数

激活函数（relu，prelu，elu，+BN）对比on cifar10

可参考上一篇：

[激活函数 ReLU、LReLU、PReLU、CReLU、ELU、SELU  的定义和区别](https://www.cnblogs.com/jins-note/p/9646602.html)

一．理论基础

1.1激活函数

![https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150404043-1209381965.png](https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150404043-1209381965.png)

1.2 elu论文（FAST AND ACCURATE DEEP NETWORK LEARNING BY

EXPONENTIAL LINEAR UNITS (ELUS)）

1.2.1 摘要

论文中提到，elu函数可以加速训练并且可以提高分类的准确率。它有以下特征：

1）elu由于其正值特性，可以像relu,lrelu,prelu一样缓解梯度消失的问题。

2）相比relu，elu存在负值，可以将激活单元的输出均值往0推近，达到

batchnormlization的效果且减少了计算量。（输出均值接近0可以减少偏移效应进而使梯

度接近于自然梯度。）

3）Lrelu和prelu虽然有负值存在，但是不能确保是一个噪声稳定的去激活状态。

4）Elu在负值时是一个指数函数，对于输入特征只定性不定量。

1.2.2.bias shift correction speeds up learning

为了减少不必要的偏移移位效应，做出如下改变：（i）输入单元的激活可以

以零为中心，或（ii）可以使用具有负值的激活函数。 我们介绍一个新的

激活函数具有负值，同时保持正参数的特性，即elus。

1.2.4实验

作者把elu函数用于无监督学习中的autoencoder和有监督学习中的卷积神经网络；

elu与relu，lrelu，SReLU做对比实验；数据集选择mnist，cifar10，cifar100.

2ALL-CNN for cifar-10

2.1结构设计

![https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150412758-258836552.png](https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150412758-258836552.png)

ALL-CNN结构来自论文（STRIVING FOR SIMPLICITY:

THE ALL CONVOLUTIONAL NET）主要工作是把pool层用stride=2的卷积来代替，提出了一些全卷积网络架构，kernel=3时效果最好，最合适之类的，比较好懂，同时效果也不错，比原始的cnn效果好又没有用到一些比较大的网络结构如resnet等。

附上：

```python
Lrelu实现：
def lrelu(x, leak=0.2, name="lrelu"):
return tf.maximum(x, leak * x)

Prelu实现：
def parametric_relu(_x):
alphas = tf.get_variable('alpha', _x.get_shape()[-1],
initializer=tf.constant_initializer(0.25),
dtype = tf.float32
)
pos = tf.nn.relu(_x)
neg = alphas * (_x - abs(_x)) * 0.5
print(alphas)
return pos + neg

BN实现：
def batch_norm(x, n_out,scope='bn'):
  """
  Batch normalization on convolutional maps.
  Args:
    x: Tensor, 4D BHWD input maps
    n_out: integer, depth of input maps
    phase_train: boolean tf.Variable, true indicates training phase
    scope: string, variable scope

  Return:
    normed: batch-normalized maps
  """
  with tf.variable_scope(scope):
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
      name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
      name='gamma', trainable=True)
    tf.add_to_collection('biases', beta)
    tf.add_to_collection('weights', gamma)

    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
       return tf.identity(batch_mean), tf.identity(batch_var)
    #mean, var = control_flow_ops.cond(phase_train,
    # mean, var = control_flow_ops.cond(phase_train,
    #   mean_var_with_update,
    #   lambda: (ema.average(batch_mean), ema.average(batch_var)))
    mean, var = mean_var_with_update()
    normed = tf.nn.batch_normalization(x, mean, var,
      beta, gamma, 1e-3)
  return normed
```

在cifar10 上测试结果如下：

![https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150502254-1055081325.png](https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150502254-1055081325.png)

以loss所有结果如下：relu+bn>elu>prelu>elubn>relu

![https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150510382-1875945396.png](https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150510382-1875945396.png)

所有的测试准确率如下

![https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150517165-1710089123.png](https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150517165-1710089123.png)

![https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150525985-582258259.png](https://img2018.cnblogs.com/blog/1470684/201809/1470684-20180914150525985-582258259.png)

relu+bn组合准确率最高，relu+bn>elu>prelu>elubn>relu

可见elu在激活函数里表现最好，但是它不必加BN，这样减少了BN的计算量。

3.ALL-CNN for cifar-100

cifar100数据集

CIFAR-100 python version,下载完之后解压，在cifar-100-python下会出现：meta,test和train

三个文件，他们都是python用cPickle封装的pickled对象

```
解压：tar -zxvf xxx.tar.gz
cifar-100-python/
cifar-100-python/file.txt~
cifar-100-python/train
cifar-100-python/test
cifar-100-python/meta
def unpickle(file):
import cPickle
fo = open(file, ‘rb’)
dict = cPickle.load(fo)
fo.close()
return dict
```

通过以上代码可以将其转换成一个dict对象，test和train的dict中包含以下元素：

data——一个nx3072的numpy数组,每一行都是(32,32,3)的RGB图像,n代表图像个数

coarse_labels——一个范围在0-19的包含n个元素的列表,对应图像的大类别

fine_labels——一个范围在0-99的包含n个元素的列表,对应图像的小类别

而meta的dict中只包含fine_label_names,第i个元素对应其真正的类别。

二进制版本（我用的）：

<1 x coarse label><1 x fine label><3072 x pixel>

…

<1 x coarse label><1 x fine label><3072 x pixel>

网络结构直接在cifar10的基础上输出100类即可，只对cifar100的精细标签100个进行分类任务，因此代码里取输入数据集第二个值做为标签。（tensorflow的cifar10代码）

`label_bytes =2 # 2 for CIFAR-100
#取第二个标签100维
result.label = tf.cast(
tf.strided_slice(record_bytes, [1], [label_bytes]), tf.int32)`

在all CNN 9层上，大约50k步，relu+bn组合测试的cifar100 test error为0.36

PS:

Activation Function Cheetsheet

![https://img2018.cnblogs.com/blog/1470684/201811/1470684-20181107220712431-1920470308.png](https://img2018.cnblogs.com/blog/1470684/201811/1470684-20181107220712431-1920470308.png)

# 层

1. Linear:线性层，最原始的称谓，单层即无隐层。熟悉torch的同学都清楚torch.nn.Linear就是提供了一个in_dim * out_dim的tensor layer而已。
2. Dense：密集层，可以指单层linear也可以指多层堆叠，可无隐层也可有但一般多指后者。熟悉keras的同学也知道dense层其实就是多层线性层的堆叠。(pytorch中的是不是没有，而是Linear？)
3. MLP：多层感知器（Multi-layer perceptron neural networks），指多层linear的堆叠，有隐层。
4. FC：全连接层(fully connected layer)，单层多层均可以表示，是对Linear Classifier最笼统的一种称谓。

# 评价指标

写文章时候可以选用一下几个
1、均方误差：MSE（Mean Squared Error）
2、均方根误差：RMSE（Root Mean Squard Error）RMSE=sqrt（MSE）。
3、平均绝对误差：MAE（Mean Absolute Error）
4、决定系数：R2（R-Square）
一般来说，R-Squared 越大，表示模型拟合效果越好。R-Squared 反映的是大概有多准，因为，随着样本数量的增加，R-Square必然增加，无法真正定量说明准确程度，只能大概定量。

```python
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

mse = mean_squared_error(testY,testPredict)
rmse = np.sqrt(mean_squared_error(testY,testPredict))
mae = mean_absolute_error(testY,testPredict)
r2 = r2_score(testY,testPredict)
```

MAPE需要自己编写

```python
def mape(y_true, y_pred):
return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
print(mape(testPredict,testY))
```

# 损失函数和优化器

## 损失函数（General）

损失函数，又叫目标函数，是编译一个神经网络模型必须的两个参数之一。另一个必不可少的参数是优化器。

损失函数是指用于计算标签值和预测值之间差异的函数，在机器学习过程中，有多种损失函数可供选择，典型的有距离向量，绝对值向量等。

![%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%80%83%20b7cca7604a46493bb8be335c2b5b92d0/Untitled%201.png](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%80%83%20b7cca7604a46493bb8be335c2b5b92d0/Untitled%201.png)

上图是一个用来模拟线性方程自动学习的示意图。粗线是真实的线性方程，虚线是迭代过程的示意，w1 是第一次迭代的权重，w2 是第二次迭代的权重，w3 是第三次迭代的权重。随着迭代次数的增加，我们的目标是使得 wn 无限接近真实值。

那么怎么让 w 无限接近真实值呢？其实这就是损失函数和优化器的作用了。图中 1/2/3 这三个标签分别是 3 次迭代过程中预测 Y 值和真实 Y 值之间的差值（这里差值就是损失函数的意思了，当然了，实际应用中存在多种差值计算的公式），这里的差值示意图上是用绝对差来表示的，那么在多维空间时还有平方差，均方差等多种不同的距离计算公式，也就是损失函数了，这么一说是不是容易理解了呢？

这里示意的是一维度方程的情况，那么发挥一下想象力，扩展到多维度，是不是就是深度学习的本质了？

下面介绍几种常见的损失函数的计算方法，pytorch 中定义了很多类型的预定义损失函数，需要用到的时候再学习其公式也不迟。

我们先定义两个二维数组，然后用不同的损失函数计算其损失值。

```python
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
sample = Variable(torch.ones(2,2))
a=torch.Tensor(2,2)
a[0,0]=0
a[0,1]=1
a[1,0]=2
a[1,1]=3
target = Variable (a)
```

sample 的值为：[[1,1],[1,1]]。

target 的值为：[[0,1],[2,3]]。

### nn.L1Loss

![%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%80%83%20b7cca7604a46493bb8be335c2b5b92d0/Untitled%202.png](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%80%83%20b7cca7604a46493bb8be335c2b5b92d0/Untitled%202.png)

```python
criterion = nn.L1Loss()
loss = criterion(sample, target)
print(loss)
```

最后结果是：1。

它的计算逻辑是这样的：

- 先计算绝对差总和：|0-1|+|1-1|+|2-1|+|3-1|=4；

### nn.SmoothL1Loss

![%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%80%83%20b7cca7604a46493bb8be335c2b5b92d0/Untitled%203.png](%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E6%80%9D%E8%80%83%20b7cca7604a46493bb8be335c2b5b92d0/Untitled%203.png)

SmoothL1Loss 也叫作 Huber Loss，误差在 (-1,1) 上是平方损失，其他情况是 L1 损失。

```python
criterion = nn.SmoothL1Loss()
loss = criterion(sample, target)
print(loss)

最后结果是：0.625。
```

### nn.MSELoss

平方损失函数。其计算公式是预测值和真实值之间的平方和的平均数。

```python
criterion = nn.MSELoss()
loss = criterion(sample, target)
print(loss)
最后结果是：1.5。
```

### nn.BCELoss

二分类用的交叉熵，其计算公式较复杂，这里主要是有个概念即可，一般情况下不会用到。

```python
criterion = nn.BCELoss()
loss = criterion(sample, target)
print(loss)
最后结果是：-13.8155。
```

### nn.CrossEntropyLoss

交叉熵损失函数

该公式用的也较多，比如在图像分类神经网络模型中就常常用到该公式。

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(sample, target)
print(loss)
最后结果是：报错，看来不能直接这么用！
```

看文档我们知道 nn.CrossEntropyLoss 损失函数是用于图像识别验证的，对输入参数有各式要求，这里有这个概念就可以了，在图像识别一文中会有正确的使用方法。

### nn.NLLLoss

负对数似然损失函数（Negative Log Likelihood）

在前面接上一个 LogSoftMax 层就等价于交叉熵损失了。注意这里的 xlabel 和上个交叉熵损失里的不一样，这里是经过 log 运算后的数值。这个损失函数一般也是用在图像识别模型上。

```python
criterion = F.nll_loss()
loss = criterion(sample, target)
print(loss)
loss=F.nll_loss(sample,target)
最后结果会报错！
```

Nn.NLLLoss 和 nn.CrossEntropyLoss 的功能是非常相似的！通常都是用在多分类模型中，实际应用中我们一般用 NLLLoss 比较多。

### nn.NLLLoss2d

和上面类似，但是多了几个维度，一般用在图片上。

```python
input, (N, C, H, W)
target, (N, H, W)
```

比如用全卷积网络做分类时，最后图片的每个点都会预测一个类别标签。

```python
criterion = nn.NLLLoss2d()
loss = criterion(sample, target)
print(loss)
同样结果报错！
```

## 损失函数（Regression）

机器学习中的所有算法都依赖于最小化或最大化函数，我们将其称为“目标函数”。最小化的函数组称为“损失函数”。损失函数是衡量预测模型在能够预测预期结果方面的表现有多好的指标。寻找最小值的最常用方法是“梯度下降”。想想这个函数的作用，如起伏的山脉和梯度下降就像滑下山到达最低点。

没有一种损失函数适用于所有类型的数据。它取决于许多因素，包括异常值的存在，机器学习算法的选择，梯度下降的时间效率，易于找到衍生物和预测的置信度。

损失函数可大致分为两类：**分类和回归损失**。在这篇文章中，专注于讨论回归损失函数。

![https://img-blog.csdnimg.cn/20190521011639824.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI4MzY4Mzc3,size_16,color_FFFFFF,t_70](https://img-blog.csdnimg.cn/20190521011639824.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI4MzY4Mzc3,size_16,color_FFFFFF,t_70)

### **回归损失**

### **1.均方误差**

[均方误差（MSE）](https://medium.freecodecamp.org/machine-learning-mean-squared-error-regression-line-c7dde9a26b93)(又称二次损失，L2损失)是最常用的回归损失函数。MSE是目标变量和预测值之间的平方距离之和。M S E = ∑ i = 1 n ( y i − y i p ) 2 n M S E=\frac{\sum_{i=1}^{n}\left(y_{i}-y_{i}^{p}\right)^{2}}{n}*MSE*=*n*∑*i*=1*n*(*yi*−*yip*)2下面是MSE函数的图，其中真实目标值为100，预测值范围在-10,000到10,000之间。MSE损失（Y轴）在预测（X轴）= 100时达到其最小值。其范围是0到∞。

![https://cdn-images-1.medium.com/max/1600/1*EqTaoCB1NmJnsRYEezSACA.png](https://cdn-images-1.medium.com/max/1600/1*EqTaoCB1NmJnsRYEezSACA.png)

MSE损失（Y轴）与预测（X轴）的关系图

### **2. 平均绝对误差**

[平均绝对误差](https://medium.com/@ewuramaminka/mean-absolute-error-mae-sample-calculation-6eed6743838a)

（MAE）(又称L1损失）是用于回归模型的另一种损失函数。MAE是我们的目标和预测变量之间的绝对差异的总和。因此，它在不考虑方向的情况下测量一组预测中的平均误差大小。（如果我们也考虑方向，那将被称为平均偏差误差（MBE），它是残差/误差的总和）。其范围也是0到∞。

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5ef77bc-96f3-441f-948a-90b521eb51c6/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f5ef77bc-96f3-441f-948a-90b521eb51c6/Untitled.png)

![https://cdn-images-1.medium.com/max/1600/1*8BQhdKu1nk-tAAbOR17qGg.png](https://cdn-images-1.medium.com/max/1600/1*8BQhdKu1nk-tAAbOR17qGg.png)

MAE损失（Y轴）与预测（X轴）的关系图

**MSE与MAE（L2损失与L1损失）**

**简而言之，** **使用平方误差更容易解决，但使用绝对误差对异常值更为稳健。但是让我们明白为什么！**

每当我们训练机器学习模型时，我们的目标是找到最小化损失函数的点。当然，当预测完全等于真实值时，两个函数都达到最小值。

这里是两个python代码的快速回顾。

```
import numpy as np

#true：真实目标变量的数组#prep：预测数组def mse（true，pred）：
    return np.sum（（true  -  pred）** 2）

def mae（true，pred）：
	return np.sum（np.abs（true  -  pred））
12345678910
```

让我们看看MAE和均方根误差的值（RMSE，它只是MSE的平方根，使其与MAE的比例相同）。在第一种情况下，预测接近真实值，并且误差在观察值之间具有小的差异。在第二个，有一个异常值观察，误差很高。

![https://cdn-images-1.medium.com/max/1600/1*KibGRET1M6Bu0-8XmjviMA.png](https://cdn-images-1.medium.com/max/1600/1*KibGRET1M6Bu0-8XmjviMA.png)

**我们从中观察到了什么，它如何帮助我们选择使用哪种损失函数？**

由于MSE平方误差（y-y_predicted = e），如果e> 1，则误差（e）的值会增加很多。如果我们的数据中有异常值，则e的值将为高，e²将为>> | E |。这将使具有MSE损失的模型比具有MAE损失的模型对异常值更敏感。在上面的第二种情况中，将调整RMSE作为损失的模型，以便以牺牲其他常见示例为代价来最小化单个异常情况，这将降低其整体性能。

如果训练数据被异常值破坏（即我们在训练环境中存在错误较大的正值或负值，而不是我们的测试环境），则**MAE损失很有用**。

直观地说，我们可以考虑一下这样的：如果我们只给一个预测为所有尽量减少MSE的意见，那么预测应该是所有目标值的均值。但是，如果我们试图最小化MAE，那么预测将是所有观测的**中位数**。我们知道，中值[对异常值的影响](https://heartbeat.fritz.ai/how-to-make-your-machine-learning-models-robust-to-outliers-44d404067d07)比均值[更强](https://heartbeat.fritz.ai/how-to-make-your-machine-learning-models-robust-to-outliers-44d404067d07)，因此使用MAE对异常值处理效果要比MSE更好。

**使用MAE损失**（尤其是神经网络）的**一个大问题**是它的梯度始终是相同的，这意味着即使对于小的损耗值，梯度也会很大。这对学习不利。为了解决这个问题，我们可以使用随着我们接近最小值而降低的动态学习率。在这种情况下，MSE表现良好，即使具有固定的学习速率也会收敛。MSE损失的梯度对于较大的损失值是高的，并且随着损失接近0而降低，使其在训练结束时更精确（见下图）。

![https://cdn-images-1.medium.com/max/1600/1*JTC4ReFwSeAt3kvTLq1YoA.png](https://cdn-images-1.medium.com/max/1600/1*JTC4ReFwSeAt3kvTLq1YoA.png)

**决定使用哪种损失函数**如果异常值表示对业务很重要且应该检测到的异常，那么我们应该使用MSE。另一方面，如果我们认为异常值只表示损坏的数据，那么我们应该选择MAE作为损失。

我建议阅读这篇文章，并进行一项很好的研究，[比较在](http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/)有异常值存在和不存在的情况下[使用L1损失和L2损失的回归模型的性能](http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/)。请记住，L1和L2损失只是MAE和MSE的另一个名称。

> L1损失对异常值更为稳健，但其衍生物不连续，使得找到解决方案效率低下。L2损失对异常值敏感，但提供更稳定和封闭的形式解决方案（通过将其导数设置为0）。

- *两者都有问题：**可能存在损失函数都没有给出理想预测的情况。例如，如果我们数据中90％的观察值具有150的真实目标值，则剩余的10％具有0-30之间的目标值。然后，MAE作为损失的模型可能预测所有观察值为150，忽略10％的离群值情况，因为它将试图达到中值。在相同的情况下，使用MSE的模型会给出0到30范围内的许多预测，因为它会偏向异常值。
- *在这种情况下该怎么办？**一个简单的解决方法是转换目标变量。另一种方法是尝试不同的损失功能。这是我们的第三次亏损功能背后的动机，Huber损失。

### **3. Huber损失**

[Huber损失](https://en.wikipedia.org/wiki/Huber_loss)(又称平滑平均绝对误差)对数据中的异常值的敏感性低于平方误差损失。它在0处也是可微分的。它基本上是绝对误差，当误差很小时变为二次曲线。该误差必须多小才能使其成为二次方取决于可以调整的超参数δ（delta）。**当δ0时，\**Huber损失接近\**MAE，当δ∞（大数）时，** Huber损耗接近**MSE。**

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c3db79d8-667e-472f-a93a-700e164bdaed/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c3db79d8-667e-472f-a93a-700e164bdaed/Untitled.png)

![https://cdn-images-1.medium.com/max/1600/1*jxidxadWSMLvwLDZz2mycg.png](https://cdn-images-1.medium.com/max/1600/1*jxidxadWSMLvwLDZz2mycg.png)

Hoss损失（Y轴）与预测（X轴）的关系图。

δ的选择至关重要，因为它决定了你愿意考虑的异常值。大于δ的残差最小化为L1（对大异常值不敏感），而小于δ的残差最小化为“适当”L2。

- *为什么要使用Huber Loss？**使用MAE训练神经网络的一个大问题是其持续的大梯度，这可能导致在训练结束时使用梯度下降丢失最小值。对于MSE，随着损失接近其最小值，梯度减小，使其更精确。

在这种情况下，胡贝尔损失确实很有用，因为它在最小值附近弯曲，从而降低了梯度。而且它比MSE更强大。因此，它结合了MSE和MAE的良好特性。然而，**Huber损失**的**问题**是我们可能需要训练超参数δ，这是一个迭代过程。

### **4. Log-Cosh损失**

Log-cosh是回归任务中使用的另一个函数，比L2更平滑。Log-cosh是预测误差的双曲余弦的对数。

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ab8c86e0-4951-464c-946d-45084fa2f34e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ab8c86e0-4951-464c-946d-45084fa2f34e/Untitled.png)

![https://cdn-images-1.medium.com/max/1600/1*BAbgW_JdwyAWLZR2dE1Ujg.png](https://cdn-images-1.medium.com/max/1600/1*BAbgW_JdwyAWLZR2dE1Ujg.png)

Log-cosh Loss（Y轴）与预测（X轴）的关系图。

- *优点：**当x值较小 时，`log(cosh(x))`约等于`(x ** 2) / 2`；当x值较大时，约等于`abs(x) - log(2)`。这意味着’logcosh’的作用大部分类似于均方误差，但不会受到偶然误差预测的强烈影响。它具有Huber损失的所有优点，并且它在各处都是可区分的，与Huber损失不同。
- *为什么我们需要二阶导数？**像[XGBoost](https://heartbeat.fritz.ai/boosting-your-machine-learning-models-using-xgboost-d2cabb3e948f)这样的许多ML模型实现使用牛顿方法来找到最优，这就是为什么需要二阶导数（Hessian）。对于像XGBoost这样的ML框架，两个可区分的函数更有利。

![https://cdn-images-1.medium.com/max/1600/1*FNxOsZLqXVZNFOxGoG9A1Q.png](https://cdn-images-1.medium.com/max/1600/1*FNxOsZLqXVZNFOxGoG9A1Q.png)

XgBoost中使用的目标函数。注意对1阶和2阶导数的依赖性

但Log-cosh损失并不完美。对于非常大的脱靶预测是恒定的，它仍然存在梯度和粗麻布的问题，因此导致没有XGBoost的分裂。

Huber和Log-cosh损失函数的Python代码：

```
# huber lossdef huber(true, pred, delta):
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)

# log cosh lossdef logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)
123456789
```

### **5.分位数损失**

在大多数现实世界预测问题中，我们通常有兴趣了解预测中的不确定性。了解预测范围而不仅仅是点估计可以显着改善许多业务问题的决策过程。

当我们对预测区间而不是仅预测点预测感兴趣时，[分位数损失函数](https://towardsdatascience.com/deep-quantile-regression-c85481548b5a)变得有用。来自最小二乘回归的预测区间基于以下假设：残差在独立变量的值上具有恒定的方差。我们认为违反这一假设的线性回归模型。我们也不能仅仅通过使用非线性函数或基于树的模型来更好地建模。这种情况来将不能利用线性回归模型来拟合。这是分位数损失和分位数回归将派上用场了，因为基于分位数损失的回归甚至对于具有非恒定方差或非正态分布的残差提供了合理的预测区间。

让我们看一个工作示例，以更好地理解为什么基于分位数损失的回归在异方差数据中表现良好。

**分位数回归与普通最小二乘回归**

![https://cdn-images-1.medium.com/max/800/1*A61Xn0hlPcoMKDns5KFD-A.png](https://cdn-images-1.medium.com/max/800/1*A61Xn0hlPcoMKDns5KFD-A.png)

左：具有恒定的残差方差。右：Y的方差随X2增加。

![https://cdn-images-1.medium.com/max/800/1*h_iOn3gSUa2bk6o0foudDA.png](https://cdn-images-1.medium.com/max/800/1*h_iOn3gSUa2bk6o0foudDA.png)

橙色线表示两种情况的OLS估计值

![https://cdn-images-1.medium.com/max/800/1*hdqrLhTXity54wmfXAtBGw.png](https://cdn-images-1.medium.com/max/800/1*hdqrLhTXity54wmfXAtBGw.png)

分位数回归。虚线表示基于回归的0.05和0.95分位数损失函数

**了解分位数损失函数**

基于分位数的回归旨在估计给定某些预测变量值的响应变量的条件“分位数”。分位数损失实际上只是MAE的延伸（当分位数为50％时，它是MAE）。

我们的想法是根据我们是否希望为正误差或负误差提供更多价值来选择分位数值。损失函数试图基于所选分位数（γ \gamma*γ*）的值给予过高估计和低估的不同惩罚。例如，γ \gamma*γ*= 0.25的分位数损失函数给予过高估计更多的惩罚，并试图将预测值保持在中位数以下

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/907d4fd0-cb2c-4123-8700-627a98c9356a/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/907d4fd0-cb2c-4123-8700-627a98c9356a/Untitled.png)

*γ*是所需的分位数，其值介于0和1之间。

![https://cdn-images-1.medium.com/max/800/1*_Msrko0NVv1d43MaVfsZkA.png](https://cdn-images-1.medium.com/max/800/1*_Msrko0NVv1d43MaVfsZkA.png)

分位数损失（Y轴）与预测（X轴）的关系图。Y = 0的真值

我们还可以使用此损失函数来计算神经网络或基于树的模型中的预测间隔。下面是梯度提升树回归器的Sklearn实现示例。

![https://cdn-images-1.medium.com/max/800/0*DQ0t4YXq-xLFsWi1.png](https://cdn-images-1.medium.com/max/800/0*DQ0t4YXq-xLFsWi1.png)

使用分位数损失预测区间（梯度增强回归）http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html*

上图显示了使用sklearn库的GradientBoostingRegression中可用的分位数损失函数计算的90％预测区间。上限构建为γ \gamma*γ*= 0.95，下限使用γ \gamma*γ*= 0.05。

### **比较研究：**

“ [Gradient boosting machines，a tutorial](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3885826/) ”中提供了一个很好的比较模拟。为了证明上述所有损失函数的性质，他们模拟了一个采用[sinc（*x*）](https://en.wikipedia.org/wiki/Sinc_function)函数采样的数据集，该函数具有两个人工模拟噪声源：高斯噪声分量ε~ *N*（0，σ2）和脉冲噪声分量ξ~Bern（*p*）。添加脉冲噪声项以说明鲁棒性效应。以下是使用不同损失函数拟合GBM回归量的结果。

![https://cdn-images-1.medium.com/max/800/1*46WnlaWhfPZaVWzSZPviIg.png](https://cdn-images-1.medium.com/max/800/1*46WnlaWhfPZaVWzSZPviIg.png)

连续损失函数：（A）MSE损失函数; （B）MAE损失函数; （C）Huber损失函数; （D）分位数损失函数演示将平滑的GBM拟合到有噪声的sinc(x)数据（E）原始sinc（*x*）函数; （F）MSE和MAE损失的光滑GBM; （Huber损失的光滑GBM，δ= {4,2,1}; （H）分位数损失的光滑GBM，α= {0.5,0.1,0.9}。

**模拟的一些观察结果：**

- 具有MAE损失函数的模型的预测受脉冲噪声的影响较小，而具有MSE损失函数的预测由于引起的偏差而略微偏差。
- 对于具有huber损失函数的模型，预测对于选择的超参数值非常敏感。
- 分位数损失函数可以很好地估计相应的置信水平。

**单个图中的所有损失函数。**

![https://cdn-images-1.medium.com/max/800/1*BploIBOUrhbgdoB1BK_sOg.png](https://cdn-images-1.medium.com/max/800/1*BploIBOUrhbgdoB1BK_sOg.png)

## 优化器Optim



## 优化器Optim

所有的优化函数都位于torch.optim包下，常用的优化器有：SGD,Adam,Adadelta,Adagrad,Adamax等，下面就各优化器分析。

### 使用

```python
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr = 0.0001)
```

 lr：学习率，大于0的浮点数
momentum:动量参数，大于0的浮点数
parameters：Variable参数，要优化的对象

### 基类 Optimizer

```python
torch.optim.Optimizer(params, defaults)
```

params (iterable) —— Variable 或者 dict的iterable。指定了什么参数应当被优化。
defaults —— (dict)：包含了优化选项默认值的字典（一个参数组没有指定的参数选项将会使用默认值）。

### 方法：

- load_state_dict(state_dict)：加载optimizer状态。
- state_dict()：以dict返回optimizer的状态。包含两项：state - 一个保存了当前优化状态的dict，param_groups - 一个包含了全部参数组的dict。
- add_param_group(param_group)：给 optimizer 管理的参数组中增加一组参数，可为该组参数定制 lr,momentum, weight_decay 等，在 finetune 中常用。
- step(closure) ：进行单次优化 (参数更新)。
- zero_grad() ：清空所有被优化过的Variable的梯度。

## 优化算法

### 随机梯度下降算法 SGD算法

SGD就是每一次迭代计算mini-batch的梯度，然后对参数进行更新，是最常见的优化方法了。即：

```python
torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

params (iterable) ：待优化参数的iterable或者是定义了参数组的dict
lr (float) ：学习率
momentum (float, 可选) ：动量因子（默认：0）
weight_decay (float, 可选) ：权重衰减（L2惩罚）（默认：0）
dampening (float, 可选) :动量的抑制因子（默认：0）
nesterov (bool, 可选) :使用Nesterov动量（默认：False）
可实现 SGD 优化算法，带动量 SGD 优化算法，带 NAG(Nesterov accelerated gradient)动量 SGD 优化算法,并且均可拥有 weight_decay 项。

对于训练数据集，我们首先将其分成n个batch，每个batch包含m个样本。我们每次更新都利用一个batch的数据，而非整个数据集。这样做使得训练数据太大时，利用整个数据集更新往往时间上不现实。batch的方法可以减少机器的压力，并且可以快速收敛。
当训练集有冗余时，batch方法收敛更快。
优缺点：
SGD完全依赖于当前batch的梯度，所以η可理解为允许当前batch的梯度多大程度影响参数更新。对所有的参数更新使用同样的learning rate，选择合适的learning rate比较困难，容易收敛到局部最优。

### *平均随机梯度下降算法 ASGD算法*

ASGD 就是用空间换时间的一种 SGD。

```python
torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
```

params (iterable) ：待优化参数的iterable或者是定义了参数组的dict
lr (float, 可选) ： 学习率（默认：1e-2）
lambd (float, 可选) ：衰减项（默认：1e-4）
alpha (float, 可选) ：eta更新的指数（默认：0.75）
t0 (float, 可选) ：指明在哪一次开始平均化（默认：1e6）
weight_decay (float, 可选) ：权重衰减（L2惩罚）（默认: 0）

### *Adagrad算法*

AdaGrad算法就是将每一个参数的每一次迭代的梯度取平方累加后在开方，用全局学习率除以这个数，作为学习率的动态更新。

其中，r为梯度累积变量，r的初始值为0。ε为全局学习率，需要自己设置。δ为小常数，为了数值稳定大约设置为10^-7 。

```python
torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0)
```

params (iterable) ：待优化参数的iterable或者是定义了参数组的dict
lr (float, 可选) ：学习率（默认: 1e-2）
lr_decay (float, 可选) ：学习率衰减（默认: 0）
weight_decay (float, 可选) ： 权重衰减（L2惩罚）（默认: 0）
优缺点：
Adagrad 是一种自适应优化方法，是自适应的为各个参数分配不同的学习率。这个学习率的变化，会受到梯度的大小和迭代次数的影响。梯度越大，学习率越小；梯度越小，学习率越大。缺点是训练后期，学习率过小，因为 Adagrad 累加之前所有的梯度平方作为分母。随着算法不断迭代，r会越来越大，整体的学习率会越来越小。所以，一般来说AdaGrad算法一开始是激励收敛，到了后面就慢慢变成惩罚收敛，速度越来越慢。在深度学习算法中，深度过深会造成训练提早结束。

### *自适应学习率调整 Adadelta算法*

Adadelta是对Adagrad的扩展，主要针对三个问题：

学习率后期非常小的问题；
手工设置初始学习率；
更新xt时，两边单位不统一
针对以上的三个问题，Adadelta提出新的Adag解决方法。Adagrad会累加之前所有的梯度平方，而Adadelta只累加固定大小的项，并且也不直接存储这些项，仅仅是近似计算对应的平均值。

```python
torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
```

params (iterable) ：待优化参数的iterable或者是定义了参数组的dict
rho (float, 可选) ： 用于计算平方梯度的运行平均值的系数（默认：0.9）
eps (float, 可选)： 为了增加数值计算的稳定性而加到分母里的项（默认：1e-6）
lr (float, 可选)： 在delta被应用到参数更新之前对它缩放的系数（默认：1.0）
weight_decay (float, 可选) ：权重衰减（L2惩罚）（默认: 0）
优缺点：
Adadelta已经不依赖于全局学习率。训练初中期，加速效果不错，很快，训练后期，反复在局部最小值附近抖动。

### *RMSprop算法*

RMSprop 和 Adadelta 一样，也是对 Adagrad 的一种改进。 RMSprop 采用均方根作为分
母，可缓解 Adagrad 学习率下降较快的问题， 并且引入均方根，可以减少摆动。

```python
torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

params (iterable) ：待优化参数的iterable或者是定义了参数组的dict
lr (float, 可选) ：学习率（默认：1e-2）
momentum (float, 可选) : 动量因子（默认：0）
alpha (float, 可选) : 平滑常数（默认：0.99）
eps (float, 可选) : 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
centered (bool, 可选):如果为True，计算中心化的RMSProp，并且用它的方差预测值对梯度进行归一化
weight_decay (float, 可选)：权重衰减（L2惩罚）（默认: 0）

### *自适应矩估计 Adam算法*

```python
torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
```

params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
lr (float, 可选) – 学习率（默认：1e-3）
betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
优缺点：
Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
Adam结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点。

- 计算效率高
- 很少的内存需求
- 梯度的对角线重缩放不变（这意味着亚当将梯度乘以仅带正因子的对角矩阵是不变的，以便更好地理解此堆栈交换）
- 非常适合数据和/或参数较大的问题
- 适用于非固定目标
- 适用于非常嘈杂和/或稀疏梯度的问题
- 超参数具有直观的解释，通常需要很少的调整（我们将在配置部分中对此进行详细介绍）

### *Adamax算法（Adamd的无穷范数变种）*

Adamax 是对 Adam 增加了一个学习率上限的概念，所以也称之为 Adamax。

```python
torch.optim.Adamax(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

```

params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
lr (float, 可选) – 学习率（默认：2e-3）
betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数
eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
优缺点：

Adamax是Adam的一种变体，此方法对学习率的上限提供了一个更简单的范围。
Adamax学习率的边界范围更简单。

### *SparseAdam算法*

针对稀疏张量的一种“阉割版”Adam 优化方法。

```python
torch.optim.SparseAdam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08)

```

params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
lr (float, 可选) – 学习率（默认：2e-3）
betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数
eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）

### *L-BFGS算法*

L-BFGS 属于拟牛顿算法。 L-BFGS 是对 BFGS 的改进，特点就是节省内存。

```python
torch.optim.LBFGS(params, lr=1, max_iter=20, max_eval=None, 
tolerance_grad=1e-05, tolerance_change=1e-09, 
history_size=100, line_search_fn=None)
```

lr (float) – 学习率（默认：1）
max_iter (int) – 每一步优化的最大迭代次数（默认：20）)
max_eval (int) – 每一步优化的最大函数评价次数（默认：max * 1.25）
tolerance_grad (float) – 一阶最优的终止容忍度（默认：1e-5）
tolerance_change (float) – 在函数值/参数变化量上的终止容忍度（默认：1e-9）
history_size (int) – 更新历史的大小（默认：100）

### *弹性反向传播算法 Rprop算法*

该优化方法适用于 full-batch，不适用于 mini-batch。不推荐。

```python
torch.optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
```

params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
lr (float, 可选) – 学习率（默认：1e-2）
etas (Tuple[float, float], 可选) – 一对（etaminus，etaplis）, 它们分别是乘法的增加和减小的因子（默认：0.5，1.2）
step_sizes (Tuple[float, float], 可选) – 允许的一对最小和最大的步长（默认：1e-6，50）
优缺点：
该优化方法适用于 full-batch，不适用于 mini-batch。

## 测试集优于训练集的原因

（1）**数据集太小的话，如果数据集切分的不均匀，或者说训练集和测试集的分布不均匀**，如果模型能够正确捕捉到数据内部的分布模式话，这可能造成训练集的内部方差大于验证集，会造成训练集的误差更大。这时你要重新切分数据集或者扩充数据集，使其分布一样

（2）**由Dropout造成，它能基本上确保您的测试准确性最好，优于您的训练准确性。Dropout迫使你的神经网络成为一个非常大的弱分类器集合，这就意味着，一个单独的分类器没有太高的分类准确性，只有当你把他们串在一起的时候他们才会变得更强大。**

因为在训练期间，Dropout将这些分类器的随机集合切掉，因此，训练准确率将受到影响。在测试期间，Dropout将自动关闭，并允许使用神经网络中的所有弱分类器，因此，测试精度提高。

# 进度条

**一、普通进度条**

示例代码

![https://pic2.zhimg.com/80/v2-355c0c9a1b87146041838ec7887a7cad_720w.jpg](https://pic2.zhimg.com/80/v2-355c0c9a1b87146041838ec7887a7cad_720w.jpg)

展现形式

![https://pic2.zhimg.com/v2-f64b41842668306c5fb1da62cfad94ed_b.jpg](https://pic2.zhimg.com/v2-f64b41842668306c5fb1da62cfad94ed_b.jpg)

**二、带时间的进度条**

导入time模块来计算代码运行的时间，加上代码迭代进度使用格式化字符串来输出代码运行进度

示例代码

![https://pic1.zhimg.com/80/v2-b2372d59e028d8a89ba738954a222fc8_720w.jpg](https://pic1.zhimg.com/80/v2-b2372d59e028d8a89ba738954a222fc8_720w.jpg)

展现形式

![https://pic1.zhimg.com/v2-00dd65d19beadddad65a0d3711a07218_b.jpg](https://pic1.zhimg.com/v2-00dd65d19beadddad65a0d3711a07218_b.jpg)

**三、TPDM 进度条**

这是一个专门生成进度条的工具包，可以使用pip在终端进行下载，当然还能切换进度条风格

示例代码

![https://pic3.zhimg.com/80/v2-b38e3414d5253a0dca08e60ed069d356_720w.jpg](https://pic3.zhimg.com/80/v2-b38e3414d5253a0dca08e60ed069d356_720w.jpg)

展现形式

![https://pic3.zhimg.com/v2-42037d2e020ed31268abaa5b10fd0256_b.jpg](https://pic3.zhimg.com/v2-42037d2e020ed31268abaa5b10fd0256_b.jpg)

**四、progress 进度条**

只需要定义迭代的次数、进度条类型并在每次迭代时告知进度条即可

相关文档：[https://pypi.org/project/progress/1.5/](https://link.zhihu.com/?target=https%3A//pypi.org/project/progress/1.5/)

示例代码

![https://pic2.zhimg.com/80/v2-9b08de855fbc6f1e0b5f7a066a3712c1_720w.jpg](https://pic2.zhimg.com/80/v2-9b08de855fbc6f1e0b5f7a066a3712c1_720w.jpg)

展现形式

![https://pic2.zhimg.com/v2-12eebc070634d4f13e6e4febd208efd9_b.jpg](https://pic2.zhimg.com/v2-12eebc070634d4f13e6e4febd208efd9_b.jpg)

**五、alive_progress 进度条**

顾名思义，这个库可以使得进度条变得生动起来，它比原来我们见过的进度条多了一些动画效果，需要使用pip进行下载

相关文档：[https://github.com/rsalmei/alive-progress](https://link.zhihu.com/?target=https%3A//github.com/rsalmei/alive-progress)

示例代码

![https://pic4.zhimg.com/80/v2-9fde8dbdaaca7120aa07d1deaa4c8483_720w.jpg](https://pic4.zhimg.com/80/v2-9fde8dbdaaca7120aa07d1deaa4c8483_720w.jpg)

展现形式

![https://pic2.zhimg.com/v2-ad7829884b8f61051be639d54dc00a01_b.jpg](https://pic2.zhimg.com/v2-ad7829884b8f61051be639d54dc00a01_b.jpg)

**六、可视化进度条**

用 PySimpleGUI 得到图形化进度条，我们可以加一行简单的代码，在命令行脚本中得到图形化进度条，也是使用pip进行下载

示例代码

![https://pic3.zhimg.com/80/v2-c0fe7244d948af8ad052137da57e645a_720w.jpg](https://pic3.zhimg.com/80/v2-c0fe7244d948af8ad052137da57e645a_720w.jpg)

展现形式

![https://pic3.zhimg.com/v2-2ead8fba626f2d25a58ecd46953950b2_b.jpg](https://pic3.zhimg.com/v2-2ead8fba626f2d25a58ecd46953950b2_b.jpg)

# 小样本

机器学习里，模型越复杂、越具有强表达能力越容易牺牲对未来数据的解释能力，而专注于解释训练数据。这种现象会导致训练数据效果非常好，但遇到测试数据效果会大打折扣。这一现象叫**过拟合（overfitting）**。

深层神经网络因为其结构，所以具有相较传统模型有很强的表达能力，从而也就需要更多的数据来避免过拟合的发生，以保证训练的模型在新的数据上也能有可以接受的表现。

------

对于classification model，有这样一个结论: 
![这里写图片描述](https://img-blog.csdn.net/20180503084154538?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0pIX1poYWk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 
上式中N是训练样本数量，η大于等于0小于等于1，h是classification model的VC dimension。具体见wiki：VC dimension。

其中的这项：

![这里写图片描述](https://img-blog.csdn.net/20180503084238864?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0pIX1poYWk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

也叫model complexity penalty。可以看到，test error小于training error加上model complexity penalty的概率是1-η。如果现在训练模型的算法能使得training error很小，而model complexity penalty又很小，就能保证test error也很小的概率是 1-η。所以要使得模型的generalization比较好，要保证training error和model complexity penalty都能比较小。观察model complexity penalty项，可以看到，h越大，model complexity penalty就会越大。N越大，model complexity penalty则会越小。大致上讲，越复杂的模型有着越大的h（VC dimension），所以为了使得模型有着好的generalization，需要有较大的N来压低model complexity penalty。 这就是为什么深度学习的模型需要大量的数据来训练，否则模型的generalization会比较差，也就是过拟合。

