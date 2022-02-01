---
author: kii
title: Reading List
categories: [深度学习]
tags: [CV,NLP,deeplearn]
date: 2022-01-24 16:33:00
---

<Boxx changeTime="10000"/>

::: tip 前言
这里主要是一些经典网络的学习笔记。
:::
<!-- more -->

# 语言模型

## SWA(Stochastic Weight Averaging)-2018

### 无痛涨点

主要是因为资源受限→需要设计轻量模型。无痛涨点技术就是在前向推理时间一定的前提下，能够提升模型效果的技术。

常见的**无痛涨点**技术有：**数据增广，损失函数的优化，训练手段的改进**等等

### SWA原理

一般情况：最后一个epoch的模型或者在验证集上效果最好的一个模型作为最终模型

SWA: 在最后采用合适的固定学习速率或者周期式学习率额外训练一段时间，取多个权重模型的平均值作为最终模型，

就是对训练过程中的多个权重模型进行平均

$$
\overline{\omega}=1/(n-m+1)\sum_{i=m}^n \omega_i
$$


解释：SGD训练的模型往往收敛到最优解的边界区域，如下图a所示，模型w_{1}, w_{2} 和 w_{3} 都落在了边缘位置，但是平均它们可以接近最优解。

训练误差和测试误差往往不对齐，也就是所说的模型泛化能力，平均处理提升模型泛化性的。（可以这样理解：通过平均可以收敛到一个wide minima，这个wide minima有更好的泛化性）

### 方法

SWA的具体做法如下图(a)所示，前75%的时间使用标准的衰减学习率策略训练，然后剩余25%设置一个合理的固定学习率进行训练，最后平均第二阶段每个epoch的权重



## GPT-3：Language Models are Few-Shot Learners

### 现有的学习方法问题

1. **对领域内有标签数据的过分依赖**：虽然有了预训练+精调的两段式框架，但还是少不了一定量的领域标注数据，否则很难取得不错的效果，而标注数据的成本又是很高的。
2. **对于领域数据分布的过拟合**：在精调阶段，因为领域数据有限，模型只能拟合训练数据分布，如果数据较少的话就可能造成过拟合，致使模型的泛华能力下降，更加无法应用到其他领域。
3. 人识别的时候不需要大量的有监督数据

因此GPT-3的主要目标是**用更少的领域数据、且不经过精调步骤去解决问题**。

### Task-Agnostic Meta-Learning 任务无偏的元学习

基于梯度下降的训练算法，它有两个在传统机器学习框架下不可学习的超参数1）初始的模型参数；2）每步的更新步长。

元学习的一个重要用途，就是通过学习的方法去学习一个对多个任务来说合适的初始参数，使得对这些训练任务和其代表的更多未来任务来说，从这个初始参数开始，对模型进行更新，都可以更快和更好地得到新的模型。这里更快的意思就是**只需要少量的训练样本和少数的几次梯度下降**，我们就可以期望得到合适的新任务的模型 (即few shot learning)。

为了解决这个问题，作者提出一种任务无关(task agnostic)的无偏元学习方法。作者通过对**初始模型加上一个正则化条件，使得它对不同的任务能“一视同仁”**。具体的，

1. 对一个分类任务，可以直接**最大化初始模型在不同类别上的熵**（Entropy Maximization）来实现对任务的无偏性。
2. 另一方面，对一般任务，比如回归或增强学习任务，往往可以通过定义一个损失函数(loss function)或者奖励函数（reward function）来定义和优化这些任务。如果把负损失或者奖励看着是给每个任务的收入（income），我们就可以基于经济学中的度量收入不平等（inequality）的方法来刻画meta-learner 在不同任务的bias。比如，我们可以用广泛应用的基尼系数来度量元学习在不同任务的偏差，除此之外还有GE指数、Theil指数等。这些不平等度量具有不同的特性，可以聚焦考虑在特定的损失或奖励（收入）区间上任务。同时，这些度量还满足若干性质，使得它们非常适合作为不平等度量。比如对称性、伸缩不变性、非负性、传递原则等等。通过最小化不平等度量，我们可以得到对不同任务无偏的meta-learner

#### GPT3结构

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%201.png)

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%202.png)

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%203.png)

High-level steps:

1. Convert the word to [a vector (list of numbers) representing the word](https://link.zhihu.com/?target=https%3A//jalammar.github.io/illustrated-word2vec/)
2. Compute prediction
3. Convert resulting vector to word

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%204.png)

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%205.png)

See all these layers? This is the “depth” in “deep learning”.

Each of these layers has its own 1.8B parameter to make its calculations. That is where the “magic” happens. This is a high-level view of that process:

## GPT-2

大规模无监督 NLP 模型

GPT-2 并没有特别新颖的架构，它和只带有解码器的 transformer 模型很像。

#### 语言模型

根据已有句子的一部分，来预测下一个单词会是什么。

#### Transformer语言建模

原始的 transformer 模型由编码器（encoder）和解码器（decoder）组成，二者都是由被称为「transformer 模块」的部分堆叠而成。

Transformer 的许多后续工作尝试去掉编码器或解码器，也就是只使用一套堆叠得尽可能多的 transformer 模块，然后使用海量文本、耗费大量的算力进行训练。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%206.png)

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%207.png)

#### 与bert的区别

GPT-2 是使用「transformer 解码器模块」构建的，而 BERT 则是通过「transformer 编码器」模块构建的。将在下一节中详述二者的区别，但这里需要指出的是，二者一个很关键的不同之处在于：GPT-2 就像传统的语言模型一样，一次只输出一个单词（token）。下面是引导训练好的模型「背诵」机器人第一法则的例子：

![https://img2020.cnblogs.com/blog/1862108/202011/1862108-20201118102602073-938855114.gif](https://imagerk.oss-cn-beijing.aliyuncs.com/img/1862108-20201118102602073-938855114.gif)

这种模型之所以效果好是因为在每个新单词产生后，该单词就被添加在之前生成的单词序列后面，这个序列会成为模型下一步的新输入。这种机制叫做自回归（auto-regression），这也是令 RNN 模型效果拔群的重要思想。

![https://img2020.cnblogs.com/blog/1862108/202011/1862108-20201118102650458-341662911.gif](https://imagerk.oss-cn-beijing.aliyuncs.com/img/1862108-20201118102650458-341662911.gif)

<font color='red'>GPT-2，以及一些诸如 TransformerXL 和 XLNet 等后续出现的模型，本质上都是自回归模型</font>，而 BERT 则不然。这就是一个权衡的问题了，虽然没有使用自回归机制，但 BERT 获得了结合单词前后的上下文信息的能力，从而取得了更好的效果。**XLNet 使用了自回归，并且引入了一种能够同时兼顾前后的上下文信息的方法。**

#### **编码器**

原始 transformer 论文中的编码器模块可以接受长度不超过最大序列长度（如 512 个单词）的输入。如果序列长度于该限制，就在其后填入预先定义的空白单词。padding mask

#### 解码器

解码器在自注意力（self-attention）层上还有一个关键的差异：它将后面的单词掩盖掉了。但并不像 BERT 一样将它们替换成特殊定义的单词，而是在自注意力计算的时候屏蔽了来自当前计算位置右边所有单词的信息。**sequence mask**+padding mask

举个例子，如果重点关注 4 号位置单词及其前续路径，可以模型只允许注意当前计算的单词以及之前的单词：

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%208.png)

#### 只含解码器模块

在 transformer 原始论文发表之后，一篇名为「Generating Wikipedia by Summarizing Long Sequences」的论文提出用另一种 transformer 模块的排列方式来进行语言建模------它直接扔掉了所有的 transformer 编码器模块......姑且就管它叫做「Transformer-Decoder」模型吧。这个早期的基于 transformer 的模型由 6 个 transformer 解码器模块堆叠而成：(请注意，该模型在某个片段中可以支持最长 4000 个单词的序列，相较于 transformer 原始论文中最长 512 单词的限制有了很大的提升。)

去除了原Transformer decoder的第二个自注意力层

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%209.png)

#### GPT-2内部机制

GPT-2 可以处理最长 1024 个单词的序列。每个单词都会和它的前续路径一起「流过」所有的解码器模块。

想要运行一个训练好的 GPT-2 模型，最简单的方法就是让它自己随机工作（从技术上说，叫做生成无条件样本）。换句话说，也可以给它一点提示，让它说一些关于特定主题的话（即生成交互式条件样本）。在随机情况下，只简单地提供一个预先定义好的起始单词（训练好的模型使用(endoftext作为它的起始单词，不妨将其称为s），然后让它自己生成文字。（**如下图给了个<s>tocken，不给也行**）

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2010.png)

此时，模型的输入只有一个单词，所以只有这个单词的路径是活跃的。单词经过层层处理，最终得到一个向量。向量可以对于词汇表的每个单词计算一个概率（词汇表是模型能「说出」的所有单词，GPT-2 的词汇表中有 50000 个单词）。在本例中，选择概率最高的单词「The」作为下一个单词。

但有时这样会出问题------就像如果持续点击输入法推荐单词的第一个，它可能会陷入推荐同一个词的循环中，只有你点击第二或第三个推荐词，才能跳出这种循环。同样的，GPT-2 也有一个叫做「top-k」的参数，模型会从概率前 k 大的单词中抽样选取下一个单词。显然，在之前的情况下，top-k = 1。

请注意，第二个单词的路径是当前唯一活跃的路径了。GPT-2 的每一层都保留了它们对第一个单词的解释，并且将运用这些信息处理第二个单词（具体将在下面一节对自注意力机制的讲解中详述），GPT-2 不会根据第二个单词重新解释第一个单词。

### 深入了解内部原理

#### 输入编码

从嵌入矩阵中查找单词对应的嵌入向量，该矩阵也是模型训练结果的一部分。

每一行都是一个词嵌入向量：**一个能够表征某个单词，并捕获其意义的数字列表。**嵌入向量的长度和 GPT-2 模型的大小有关，最小的模型使用了长为 768 （**embedding size**）的嵌入向量来表征一个单词（开始的时候，这个可以自己定，但是bert是人家训好的模型）。

所以在一开始，需要在嵌入矩阵中查找起始单词s对应的嵌入向量。但在将其输入给模型之前，还需要引入位置编码------一些向 transformer 模块指出序列中的单词顺序的信号。1024 个输入序列位置中的每一个都对应一个位置编码，这些编码组成的矩阵也是训练模型的一部分。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2011.png)

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2012.png)

至此，输入单词在进入模型第一个 transformer 模块之前所有的处理步骤就结束了。如上文所述，训练后的 GPT-2 模型包含两个权值矩阵：**嵌入矩阵**和**位置编码矩阵**。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2013.png)

将单词输入第一个 transformer 模块之前需要查到它对应的嵌入向量，再加上 1 号位置位置对应的位置向量。

#### 堆叠

第一个 transformer 模块处理单词的步骤如下：首先通过自注意力层处理，接着将其传递给神经网络层。第一个 transformer 模块处理完但此后，会将结果向量被传入堆栈中的下一个 transformer 模块，继续进行计算。每一个 transformer 模块的处理方式都是一样的，但每个模块都会维护自己的自注意力层和神经网络层中的权重。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2014.png)

#### Self-Attention

自注意力机制所做的工作，它在处理每个单词（将其传入神经网络）之前，融入了模型对于用来解释某个单词的上下文的相关单词的理解。

自注意力机制沿着序列中每一个单词的路径进行处理，主要由 3 个向量组成：

- 查询向量（Query 向量）：当前单词的查询向量被用来和其它单词的键向量相乘，从而得到其它词相对于当前词的注意力得分。只关心目前正在处理的单词的查询向量。
- 键向量（Key 向量）：键向量就像是序列中每个单词的标签，它使搜索相关单词时用来匹配的对象。
- 值向量（Value 向量）：值向量是单词真正的表征，当算出注意力得分后，使用值向量进行加权求和得到能代表当前位置上下文的向量。

这样将值向量加权混合得到的结果是一个向量，它将其 50% 的「注意力」放在了单词「robot」上，30% 的注意力放在了「a」上，还有 19% 的注意力放在「it」上。

#### 模型的输出

当最后一个 transformer 模块产生输出之后（即经过了它自注意力层和神经网络层的处理），模型会将输出的向量乘上嵌入矩阵。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2015.png)

嵌入矩阵的每一行都对应模型的词汇表中一个单词的嵌入向量。所以这个乘法操作得到的结果就是词汇表中每个单词对应的注意力得分。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2016.png)

简单地选取得分最高的单词作为输出结果（即 top-k = 1）。但其实如果模型考虑其他候选单词的话，效果通常会更好。所以，一个更好的策略是对于词汇表中得分较高的一部分单词，将它们的得分作为概率从整个单词列表中进行抽样（得分越高的单词越容易被选中）。通常一个折中的方法是，将 top-k 设为 40，这样模型会考虑注意力得分排名前 40 位的单词。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2017.png)

这样，模型就完成了一轮迭代，输出了一个单词。模型会接着不断迭代，直到生成一个完整的序列------序列达到 1024 的长度上限或序列中产生了一个终止符。



### 语言模型应用

#### 机器翻译

进行翻译时，模型不需要编码器。同样的任务可以通过一个只有解码器的 transformer 来解决

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2018.png)

#### 自动摘要生成

#### 迁移学习

在论文「Sample Efficient Text Summarization Using a Single Pre-Trained Transformer」（[https://arxiv.org/abs/1905.08836](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1905.08836)）中，首先使用只包含解码器的 transformer 在语言建模任务中进行预训练，然后通过调优来完成摘要生成任务。结果表明，在数据有限的情况下，该方案比预训练好的编码器-解码器 transformer 得到了更好的效果。

GPT2 的论文也展示了对语言建模模型进行预训练后取得的摘要生成效果。

#### 音乐生成

音乐 transformer（[https://magenta.tensorflow.org/music-transformer](https://link.zhihu.com/?target=https%3A//magenta.tensorflow.org/music-transformer)）采用了只包含解码器的 transformer 来生成具有丰富节奏和动感的音乐。和语言建模相似，「音乐建模」就是让模型以一种无监督的方式学习音乐，然后让它输出样本（我们此前称之为「随机工作」）。

你可能会好奇，在这种情境下是如何表征音乐的？请记住，语言建模可以通过对字符、单词（word）、或单词（word）某个部分的词（token）的向量表征来实现。面对一段音乐演奏（暂时以钢琴为例），我们不仅要表征这些音符，还要表征速度------衡量钢琴按键力度的指标。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2019.png)

一段演奏可以被表征为一系列的 one-hot 向量。一个 MIDI 文件可以被转换成这样的格式。论文中展示了如下所示的输入序列的示例：

![https://img2020.cnblogs.com/blog/1862108/202011/1862108-20201118101233900-1139623680.png](https://imagerk.oss-cn-beijing.aliyuncs.com/img/1862108-20201118101233900-1139623680.png)

这个输入序列的 one-hot 向量表征如下：

![https://img2020.cnblogs.com/blog/1862108/202011/1862108-20201118101242316-233755271.png](https://imagerk.oss-cn-beijing.aliyuncs.com/img/1862108-20201118101242316-233755271.png)

我喜欢论文中用来展示音乐 transformer 中自注意力机制的可视化图表。我在这里加了一些注释：

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2020.png)

这段作品中出现了反复出现的三角轮廓。当前的查询向量位于后面一个「高峰」，它关注前面所有高峰上的高音，一直到乐曲的开头。图中显示了一个查询向量（所有的注意力线来源）和正要处理的以前的记忆（突出了有更高 softmax 概率的音符）。注意力线的颜色对应于不同的注意力头，而宽度对应于 softmax 概率的权重。

## Bert

[https://www.cnblogs.com/jiangxinyang/p/11715678.html](https://www.cnblogs.com/jiangxinyang/p/11715678.html)

BERTBASE (L=12, H=768, A=12, Total Parameters=110M)

- **`[BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)`**：12层(Transformer blocks)，768 hidden size，12头，110M参数
- **`[BERT-Large, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)`**：24层，1024 hidden size，16个头，340M参数
- Size of the encoder layers and the pooler layer.
- feed-forward/filter size to be 4H

#### 几个要思考的点

1. 输入是什么形式
2. padding mask vs. sequence mask是怎么样的？
3. bert的encoder中 feed-forward:4H
4. bert base的有12个Transformer blocks, 每个transformer blocks具有？个encodder layer and pooler layer? 大小是768
5. 输出是一个词向量，然后从词库里找最相近的

## XLNet

**XLNet 使用了自回归，并且引入了一种能够同时兼顾前后的上下文信息的方法。**



## TransformerXL 

---

# 图像分类

---

## 图像

Relu，dropout

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2021.png)

网络总共的层数为8层，5层卷积，3层全连接层。

## VGG

### 优点

1. 结构简单，卷积核大小都一样3*3，最大池化层。

## ResNet（分类）

模型退化原因：

1. 过拟合，层数越多，参数越复杂，泛化能力弱

2. 梯度消失/梯度爆炸，层数过多，**梯度反向传播时由于链式求导连乘**使得梯度过大或者过小，使得梯度出现消失/爆炸，对于这种情况，可以通过BN(batch normalization)可以解决

3. 由深度网络带来的退化问题，一般情况下，网络层数越深越容易学到一些复杂特征，理论上模型效果越好，但是由于深层网络中含有大量非线性变化，每次变化相当于丢失了特征的一些原始信息，从而导致层数越深退化现象越严重。

   ![img](https://pic4.zhimg.com/80/v2-252e6d9979a2a91c2d3033b9b73eb69f_720w.jpg)

![img](https://pic4.zhimg.com/80/v2-ea924e733676e0da534f677a97c98653_720w.jpg)

<img src="https://pic1.zhimg.com/80/v2-1dfd4022d4be28392ff44c49d6b4ed94_720w.jpg" alt="img" style="zoom:150%;" />

## Inception v4-

《Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning》 2016 Google

## ResNeXt（分类）

ResNeXt是ResNet[2]和Inception[3]的结合体

利用了分组卷积的思想

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2022.png)

ResNeXt和ResNet一样，也是通过一系列的block来构建，每个block聚合了一组有相同拓扑的转换。这种设计产生了一个同构的、多分支的架构。另外，除了除了深度和宽度之外的一个关键因子，这个策略揭示了一个新的维度，"**基数-cardinality** "。

作者通过实验表明，在控制复杂度的受限情况下，增加基数可以提升分类精度。而且当模型复杂度增加时，提升基数比更深或更宽更加有效。


![img](https://pic1.zhimg.com/80/v2-0c5ac698702de2970016afc2834816d5_720w.jpg?source=1940ef5c)

首先这是两者block的区别，左侧为ResNet的block结构，右侧的是ResNeXt的block结构。每一个方框代表一层，其中的三个数据代表的意义分别为（输入数据的channel，filter大小，输出数据的channel）。其好处就是在保证了FLOPs和参数量的前提下，通过更宽或者深的网络，来提高精度。同时<font color='red'>**每一个Block中每一条path在文章中被称为一个cardinality**</font> (the size of the set of transformations) 。它是一个除了宽度和深度的维度外具体的，可测量的维度，这也是至关重要的。实验表明，增加cardinality是获得精度的一种更有效的方法，而不是加深或更加宽网络，尤其是当深度和宽度开始使现有模型的收益递减时。

**也正是这个cardinality，带给ResNeXt更好的精度与效果。**

不难发现一方面ResNeXt还是继承采用了ResNet的重复层策略（strategy of repeating layers）。但是不同的是，增加了路径数量，以简单，可扩展的方式利用拆分转换合并策略。ResNeXt网络中的一个模块执行一组转换，每个转换都在低维嵌入中进行，其输出通过求和进行汇总，并且每一个路径都**具有相同的拓扑结构**。这种设计使ResNeXt无需特殊设计即可扩展到任何数量的转换。

除此之外，ResNeXt还存在者另外两种等价形式。

![img](https://pic1.zhimg.com/80/v2-46f05b9dd9d121340117b9483c337227_720w.jpg?source=1940ef5c)

（b）中的重组与Inception-ResNet模块相似，因为它连接了多个路径。但是ResNeXt的模块与所有现有的Inception模块不同，因为其所有的路径都共享相同的拓扑结构，因此可以轻松地隔离路径数量作为要调查的因素。（c）等价于(a)(b)，实现了分组卷积。

![img](https://pic3.zhimg.com/80/v2-aa850a77dde8260331d90bd1de54f8d9_720w.jpg?source=1940ef5c)

根据上图可知，ResNeXt与ResNet的拓扑极为相似，并且受VGG / ResNets启发遵循两个简单规则：

1. 如果生成相同大小的空间图，则这些块共享相同的超参数（宽度和过滤器大小）
2. 每次以2为系数对空间图进行下采样时，块的宽度乘以系数2。

第二条规则确保就FLOP（浮点运算）对于所有块而言大致相同。

使用这两个规则，我们只需要设计一个模板模块，就可以相应地确定网络中的所有模块。 因此，这两个规则极大地缩小了设计空间，使我们可以专注于一些关键因素。 这些规则构成的网络在上图中表示。

> 这里纠正一下问题下某位答主的关于Group卷积“使得同等的FLOPs下，ResNeXt的参数变多了”的观点。根据上图的数据显示是不正确的。作者为了对比两个网络的效果，在参数上也是严格进行限制了的。

**所以ResNeXt的一个优点就是简化了网络的设计难度。**

与此同时，作者也在论文中将论文最大创新点cardinality与网络中**宽度、深度**这两个超参进行了比较。

实验部分，作者在**ImageNet-1K、ImageNet-1K、CIFAR、COCO object detection**上进行了实验，以下是部分实验截图。

![img](https://pic3.zhimg.com/80/v2-b2bde5b6e2bd92790d06a697fa318ca4_720w.jpg?source=1940ef5c)

![img](https://pic1.zhimg.com/80/v2-903faedb33aef63be9f4f1abff7aea68_720w.jpg?source=1940ef5c)

---

ResNeXt和Inception v4是非常像的。不同之处有两点：

1. ResNeXt的分支的拓扑结构是相同的，Inception V4需要人工设计；
2. ResNeXt是先进行 [公式] 卷积然后执行单位加，Inception V4是先拼接再执行1x1卷积

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2023.png)

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2024.png)

## Densenet（分类）

先放一个dense block的结构图。在传统的卷积神经网络中，如果你有L层，那么就会有L个连接，但是在DenseNet中，会有L(L+1)/2个连接。**简单讲，就是每一层的输入来自前面所有层的输出。**如下图：x0是input，H1的输入是x0（input），H2的输入是x0和x1（x1是H1的输出）……

DenseNet的一个优点是网络更窄，参数更少

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2025.png)

## Shufflenet v2（分类）

## MobileNet v3（分类）

**论文名称:Searching for MobileNetV3**

**作者：Googler**

**[论文链接**：https://arxiv.org/abs/1905.02244](https://arxiv.org/abs/1905.02244)

### 轻量化网络

下面总结了目前常用的一些减少网络计算量的方法：

- **基于轻量化网络设计**：比如mobilenet系列，shufflenet系列， Xception等，使用Group卷积、1x1卷积等技术减少网络计算量的同时，尽可能的保证网络的精度。[各种卷积](https://www.notion.so/205f3011ff394ad58f1e5f8c4336eef0)
- **模型剪枝**： 大网络往往存在一定的冗余，通过剪去冗余部分，减少网络计算量。
- **量化**：利用TensorRT量化，一般在GPU上可以提速几倍。

[TensorRT](https://www.notion.so/TensorRT-49e4c797ae6b4c10a56a8c8aa36c0c7e)

- **知识蒸馏**：利用大模型（teacher model）来帮助小模型（student model）学习，提高student model的精度。

#### V1

**创新点**：分组卷积，其实这里用的可以说是**深度可分卷积**：网络的分组数与网络的channel数量相等，使用的point-wise conv,即使用1x1的卷积进行channel之间的融合

<font color='red'>深度可分离卷积block</font>可以归纳为以下**直筒层级结构**:

![2020032613241636](https://imagerk.oss-cn-beijing.aliyuncs.com/img/2020032613241636.png)

整体架构：

![image-20220125150254977](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220125150254977.png)

#### v2

[MobileNet v2中 Inverted Residual 和 Linear Bottleneck 是怎么回事_flyfish-CSDN博客](https://blog.csdn.net/flyfish1986/article/details/97017017)

**创新点**：

- 引入了bottleneck结构。

- 将bottleneck结构变成了纺锤型（如下图），即resnet是先缩小为原来的1/4，再放大，他是放大到原来的6倍，再缩小。起名子叫：倒残差（Inverted Residual）

- 并且去掉了Residual Block最后的ReLU。

<font color='red'>Depthwise convolution之前添加一层Pointwise convolution</font>,以下为MobilenetV2的残差块。

  ![20200326132436263](https://imagerk.oss-cn-beijing.aliyuncs.com/img/20200326132436263.png)

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2027.png)

这一层的Pointwise 卷积就可以叫expansion卷积，这一层就可以叫expansion layer，因为它用来升维了。升了多少倍或者说通道扩大了多少倍就有了expansion factor。这里144/24=6

那么6就是expansion factor。

```python
(conv): Sequential(
        (0): ConvBNReLU(
          (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): ConvBNReLU(
          (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
          (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
```

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2028.png)

第一行表示升维, 第二行卷积, 第三行降维

1. 第一行的升维叫 Expansion Layer

2. 第二行的卷积叫 Depthwise Layer

3. 第三行的降维叫 Projection Layer


**为什么叫线性瓶颈（Linear Bottleneck）？**

先说瓶颈（Bottleneck），1×1卷积小像个瓶口所以叫瓶颈，该词的来源是resnet的经典网络使用的词汇，当前的MobileNet v2依旧使用了该结构的block。再说linear，看图，这个图是从下往上看的，看最后的pointwise卷积，之前的pointwise卷积是升维的，轮到最后的pointwise卷积就是降维，设计该网的作者说<font color='red'>高维加个非线性挺好，低维要是也加非线性就把特征破坏了，不如线性的好，所以1*1后 不加ReLU6 ，改换线性</font>。

![https://img-blog.csdnimg.cn/20190723181015933.png](https://imagerk.oss-cn-beijing.aliyuncs.com/img/20190723181015933.png)

- stride=1的时候:（<font color='magenta'>**Q:为啥等于1的时候是相加？下面一点给了解释**</font>）

  1. point-wise升维

  2. depth-wise提取特征

  3. 通过Linear的point-wise降维，得到out。

  4. input与out相加（残差结构）

     ```python
     out=out+self.shortcut(x)
     ```

     

- stride为2时，**因为input与output的大小不同，所以没有添加shortcut结构**


- ReLU6就是普通的ReLU但是限制最大输出值为6（对输出值做clip），这是为了在移动端设备float16的低精度的时候，也能有很好的数值分辨率，如果对ReLU的激活范围不加限制，输出范围为0到正无穷，如果激活值非常大，分布在一个很大的范围内，则低精度的float16无法很好地精确描述如此大范围的数值，带来精度损失。


#### V3

- 1. 引入SE结构:
  
  2. https://blog.csdn.net/u014380165/article/details/78006626
  
  3. Hard-Swish(深层网络体现优势)
  
      <img src="http://latex.codecogs.com/gif.latex?H-Swish[x]=x\frac{ReLU6(x+3)}{6}" />

SE模块

Squeeze-and-Excitation Networks，SE模块是**一种轻量级的通道注意力模块**，能够让网络模型对特征进行校准的机制，使得有效的权重大，无效或效果小的权重小的效果。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2029.png)

```python
# SE-Block
import torch
import torch.nn as nn
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self,in_chnls,ratio):#在执行实例化过程Student()时传入一些参数，以方便且正确地初始化/设置一些属性值，那么如何定义这种初始化行为呢？
        super(SE,self).__init__()
        self.squeeze=nn.AdaptiveAvgPool2d((1,1))
        self.compress=nn.Conv2d(in_chnls,in_chnls//ratio,1,1,0)
        self.excitation=nn.Conv2d(in_chnls//ratio,in_chnls,1,1,0)

    def forward(self,x):
        out=self.squeeze(x)
        out=self.compress(out)
        out=F.relu(out)
        out=self.excitation(out)
        return torch.sigmoid(out)
```

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2030.png)

在bottleneck结构中加入了SE结构，并且放在了depthwise filter之后，如下图。因为SE结构会消耗一定的时间，所以作者在含有SE的结构中，将expansion layer的channel变为原来的1/4,这样作者发现，即提高了精度，同时还没有增加时间消耗。并且SE结构放在了depthwise之后。

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2031.png)

![img](https://img-blog.csdnimg.cn/20200822113315506.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQyNjE3NDU1,size_16,color_FFFFFF,t_70)

- 2.修改尾部结构：

在mobilenetv2中，在avg pooling之前，存在一个1x1的卷积层，目的是提高特征图的维度，更有利于结构的预测，但是这其实带来了一定的计算量了，所以这里作者修改了，将其放在avg pooling的后面，首先利用avg pooling将特征图大小由7x7降到了1x1，降到1x1后，然后再利用1x1提高维度，这样就减少了7x7=49倍的计算量。并且为了进一步的降低计算量，作者直接去掉了前面纺锤型卷积的3x3以及1x1卷积，进一步减少了计算量，就变成了如下图第二行所示的结构，作者将其中的3x3以及1x1去掉后，精度并没有得到损失。这里降低了大约15ms的速度。

![https://img-blog.csdnimg.cn/20190612203455246.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NodW5mZW5neWFueXVsb3Zl,size_16,color_FFFFFF,t_70](https://imagerk.oss-cn-beijing.aliyuncs.com/img/20190612203455246.png)

- 3.修改channel数量

修改头部卷积核channel数量，mobilenet v2中使用的是32 x 3 x 3，作者发现，其实32可以再降低一点，所以这里作者改成了16，在保证了精度的前提下，降低了3ms的速度。，这里给出了mobilenet v2以及mobilenet v3的结构对比：

![https://img-blog.csdnimg.cn/20190612205949629.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NodW5mZW5neWFueXVsb3Zl,size_16,color_FFFFFF,t_70](https://img-blog.csdnimg.cn/20190612205949629.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NodW5mZW5neWFueXVsb3Zl,size_16,color_FFFFFF,t_70)

![https://img-blog.csdnimg.cn/20190612205905304.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NodW5mZW5neWFueXVsb3Zl,size_16,color_FFFFFF,t_70](https://img-blog.csdnimg.cn/20190612205905304.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NodW5mZW5neWFueXVsb3Zl,size_16,color_FFFFFF,t_70)

- 4.非线性变换的改变

    使用h-swish替换swish，swish是谷歌自家的研究成果，颇有点自卖自夸的意思，这次在其基础上，为速度进行了优化。swish与h-swish公式如下所示，由于sigmoid的计算耗时较长，特别是在移动端，这些耗时就会比较明显，所以作者使用ReLU6(x+3)/6来近似替代sigmoid，观察下图可以发现，其实相差不大的。**利用ReLU有几点好处，1.可以在任何软硬件平台进行计算，2.量化的时候，它消除了潜在的精度损失，使用h-swish替换swith，在量化模式下回提高大约15%的效率，另外，h-swish在深层网络中更加明显。**如下两张图展示的是使用h-swish对于时间以及精度的影响，可以发现，使用h-swish@16可以提高大约0.2%的精度，但是持剑延长了大约20%。

    <img src="http://latex.codecogs.com/gif.latex?H-Swish[x]=x\frac{ReLU6(x+3)}{6}" />

    ![https://img-blog.csdnimg.cn/20190613144850897.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NodW5mZW5neWFueXVsb3Zl,size_16,color_FFFFFF,t_70](https://img-blog.csdnimg.cn/20190613144850897.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NodW5mZW5neWFueXVsb3Zl,size_16,color_FFFFFF,t_70)

    ![https://img-blog.csdnimg.cn/20190613145326536.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NodW5mZW5neWFueXVsb3Zl,size_16,color_FFFFFF,t_70](https://img-blog.csdnimg.cn/20190613145326536.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NodW5mZW5neWFueXVsb3Zl,size_16,color_FFFFFF,t_70)

**虽然mobilenet v3结构你已经知道了，但是，本文的重点，如何设计出的这个网络，即如何进行网络结构的搜索，还是有必要提一下的，这里我也没有细细研究，想深入了解的同学可以自行对应阅读论文**

**总体过程很简单，先通用NAS算法，优化每一个block，得到大体的网络结构，然后使用NetAdapt 算法来确定每个filter的channel的数量**

**这里由于small model的精度以及耗时影响相对较大，mobilenet v3 large和mobilenet v3 small是分别使用NAS设计的。**

**NAS之后，可以使用NetAdapt算法设计每个layer，过程如下：**

- 先用NAS找到一个可用的结构A。
- 在A的基础上生成一系类的候选结构，并且这些候选结构消耗在一点点减少，其实就是穷举子结构。对于每个候选结构，使用前一个模型进行初始化，（前一个模型没有的参数随机初始化就行），finetune T个epoch，得到一个大致的精度。在这些候选结构中，找到最好的。
- 反复迭代，知道目标时间到达，找到最合适的结果。

**候选是怎么选取的呢?**

- 降低expansion layer的size.
- 减少botleneck

---

> 总结：
>
> SE_bottleneck结构

![image-20220127221254807](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220127221254807.png)

## ACNet

https://zhuanlan.zhihu.com/p/131282789

## MicroNets（分类）

结果说是biMobileNetV3好

关于卷积，作者提出了一种微分解卷积(MF-Conv)来将Pointwise卷积分解成两组卷积层。其中，组数G适应于通道C的数量为：

[https://mmbiz.qpic.cn/mmbiz_png/BJbRvwibeSTsq8W59nOl0AaVAcnibgPjcU0wsP3Z3w3G9m4EeW4NkRUeEHw53tXq0IWSFabVua6hdr0S5VPaZzHQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1](https://mmbiz.qpic.cn/mmbiz_png/BJbRvwibeSTsq8W59nOl0AaVAcnibgPjcU0wsP3Z3w3G9m4EeW4NkRUeEHw53tXq0IWSFabVua6hdr0S5VPaZzHQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

R是通道减少率。在给定的计算成本下，这个等式在通道数和节点连接性之间实现了很好的权衡。数学上，Pointwise卷积矩阵由一个块矩阵(G×G块)近似，其块的rank为1。这保证了**最小的路径冗余和最大的输入覆盖率** ，从而使网络能够为给定的计算预算实现更多的通道。

关于非线性，作者提出了一种新的激活函数（Dynamic Shift-Max ，DY-Shift-Max），用于非线性地将通道与动态系数融合。新的激活函数迫使网络学习融合输入特征图的不同圆形通道移动（使用适应输入的系数），并在这些融合中选择最好的，结果表明，这可以在计算成本较小的情况下增强了group分解的表示能力。

## Residual Attention--多标签
[Residual Attention: A Simple but Effective Method for Multi-Label Recognition](https://arxiv.org/abs/2108.02456)

### 多标签图像识别

#### 问题

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled.png)

**（1）传统单标签分类** ：

city（person）

**（2）多标签分类** ：

city , river, person, European style

**（3）人的认知** ：

两个人在河道边走路

欧洲式建筑，可猜测他们在旅游

天很蓝，应该是晴天但不是很晒

相比较而言，**单标签** 分类需要得到的信息量**最少** ，**人的认知** 得到的信息量**最多** ，**多标签** 分类在它们**两者之间** 。

#### Motivation

近年来对多标签识别的研究主要集中在**标签间的语义关系 、对象proposal 和注意力机制** 三个方面。

基于标签间的语义关系方法，**计算成本较高**而且存在**手工定义邻接矩阵**的问题；

基于对象 proposal的方法，在**处理对象proposal上花费太多时间**；

尽管注意力模型是一种端到端的，相对比较简单的方法，但对于多标签分类，这些模型往往**过于复杂**，导致难以优化、实现或解释。

基于以上的问题，作者出了一个简单而容易的类特定残差注意力(<font color='red'>class-specific residual attention，CSRA</font>)模块，通过充分利用每个对象类别单独的空间Attention，取得了较高的准确性。

#### **Overview**

<img src="C:\Users\kaich\AppData\Roaming\Typora\typora-user-images\image-20210822155206960.png" alt="image-20210822155206960" style="zoom:67%;" />

#### **Code**

```python
def forward(self, x):
        b, c, h, w = x.shape
        y_raw = self.fc(x).flatten(2)  # b,num_class,hxw
        y_avg = torch.mean(y_raw, dim=2)  # b,num_class
        y_max = torch.max(y_raw, dim=2)[0]  # b,num_class
        score = y_avg + self.la * y_max
        return score
```

#### **3.1. 为什么max pooling会有用？**

文章结果显示对于不同的模型和数据集，CSRA都能提升性能（其中是一个超参数）。对于**多标签**任务，作者使用**mAP**作为评价指标，而ImageNet(单标签任务)使用**Accuracy**。

这些结果表明，简单地增加个max-pooling可以提高多标签识别的精度，特别是当baseline模型的mAP不高时。从上面的代码中可以看出，CSRA就是多了一行max-pooling，那么，为什么这个max-pooling是有用的呢？

1）首先，**y_max**获取了每个类别的**所有空间位置中的最大值**。因此，它可以被看作是一种class-specific的注意力机制。

2）另外，作者推测CSRA能够让模型关注**不同物体类别在不同位置的分类得分**，因此相比于传统的分类网络，CSRA更加适用于多标签分类的任务。

#### **Residual attention**

对于一张图片$I$，首先通过一个CNN网络来提取特征$x$，其中$x$是一个$d\times h \times \omega$的特征矩阵：

$$
x=\Phi(I;\theta)
$$
在实验中，特征的维度通常是2048x7x7，因此在空间维度打平之后就可以表示成$x_1, x_2, x_3,...,x_{49}$。然后通过一个FC分类器得到分类的结果，其中$m_i \in R^{2048}$为第i类分类器的参数。

然后就可以定义第`i`个类第`j`个位置上的class-specific attention scores（在空间维度进行softmax，使得每个类所有空间上概率之和为1，以此来得到每个类别的空间attention map）：
$$
s_j^i=\frac{exp(Tx_j^Tm_i)}{\sum_{k=1}^{49}exp(Tx_k^Tm_i)}\\
\sum_{j=1}^{49}s_j^i=1
$$


**其中`T>0`是用来控制score的 temperature**，是一个超参数。$s_j^i$代表了第`i`类在第`j`个位置上出现的概率。

得到第`i`类在第`j`个位置上出现的概率之后，我们就可以把这个概率和特征进行相乘求和，得到class-specific的特征向量了：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIh0eMOicgoOkAEjqA2J5EpDxkfxv4fN4azK7yvA8va16jw8tmKxxQNqw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

全局的class-agnostic(`不可知论者`)特征可以通过将所有位置的特征进行求平均得到：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFI1qmJ48UlAjVDhFtGGibwHsS8NlqWibfpEhibBd7VY0M8EWzN9Fh6PyHkA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由于在多分类任务中能够达到比较好的效果，所以作者依旧将作为主特征。（**这里个人其实有一点疑问，既然作者想证明class-specific的特征是非常有用的，但是为什么作者又用了一个比较小的权重来减少class-specific特征的影响？** ）

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIiaTgAAClBva3uRA4Z95I4rPVHTnB13Ap926rew2mBM775DVsID4w2icA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

如上图所示，最终的特征$f^i$有class-specific特征和class-agnostic特征相加得到：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIbV0jKhKacOnstw12ALCucVw2RROlKqlkkwGfdrC66LuKSPeFMO4iaHw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

最终分类的结果就是用特征$f$用分类器进行分类的结果：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIgrtoEicm2wLxERXD8iczCx6HgFnSPcVw7lwYAnhRD9R0BmU6tIaatYUQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中为数据集中类别的数量。

#### **3.3. CSRA 的解释**

在本节中，作者将会证明代码中的max pooling的实现方式就是CSRA的一种特殊情况。首先将分类的结果重新展开可以得到：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIGvptRVMt6WlNqkhIjVJ6sP4oQohCPCErGMCQU9Sdpb7NLSMhb4Cviaw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

公式第一行的第一项的是第`i`个类的base logit，也可以用下面的公式表示：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIenbMXR57jmUMUqiaUY7cWfjIicBz6Lbg528GfrlNEnyr3QUm2VWLxaaA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

第二项的$x_km_i^T$是第`k`个位置的分类分数，然后再用$s^i_k$来进行加权。

当`T`趋向于正无穷时，softmax的输出结果就变成了一个Dirac delta函数：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIFODqTo5TibDQSJclVu2Y1YPcm7QepraD0vyYHSS7cQdsYhE7g2BrriaA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

因此就可以用最大项来代替，这也就得到代码中max pooling的由来：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIibkmib3OSXNll38d2h2wDLViaUryUj96ukBpibTB1VJ0PZy0AmZNk0UIJw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

另外，`f`可以被表示成：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIs5jqPiar5Ysxhbrg2tBgHm6me4x8yibMaG4PfKpWGc7RP55pqGMiahjUw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在这个式子中常数可以直接被忽略，因此CSRA的特征就是特征被加权之后的结果。

#### **3.4. Multi-head attention**

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFIcGwR1OAYQPPfVgsWrViaZ5y0jL8YAnIHl7KJeFgECklTXOdlZANqU7w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在上面的代码和式子中，我们可以看到有一个 temperature超参数`T`需要去调，不同的类可能需要不同的超参数。为了避免这个调参的过程，作者引入了一个`mul-head attention`的方式来避免这个调参的过程。

如上图所示，作者将CSRA改成了一个多分支的结构（每一个分支就代表是一个head），每个分支使用相同的$\lambda$，但是使用不同的`T`。当只有一个head时，固定为1；随着head的增加，不同head的`T`也不断增大，如下所示：

![Untitled](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gYUsOT36vfr3yqaFonpYicricSrpiaOoOFItXG37a2VHzNDcU2GUwGsdiaO90yEHc8HUX9NEzpnwgfasBYOu4WkprQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

通过将不同`T`的结果进行融合，就能够避免了调参的过程。

## Swin Transformer

1. 受限于图像的矩阵性质，一个能表达信息的图片往往至少需要几百个像素点，而建模这种几百个长序列的数据恰恰是Transformer的天生缺陷；
2. 目前的基于Transformer框架更多的是用来进行图像分类，理论上来讲解决检测问题应该也比较容易，但是对**实例分割**这种密集预测的场景Transformer并不擅长解决。

[vit](https://zhuanlan.zhihu.com/p/354140152)

1. 引入CNN中常用的<font color='red'>层次化构建方式</font>构建层次化Transformer 

2. 引入locality思想，对无重合的window区域内进行self-attention计算。解决：不重合的window之间缺乏信息交流

相比于ViT，Swin Transfomer计算复杂度大幅度降低，**具有输入图像大小线性计算复杂度**。Swin Transformer随着深度加深，逐渐合并图像块来构建层次化Transformer，可以作为通用的视觉骨干网络，应用于图像分类、目标检测和语义分割等任务。

![preview](https://pic4.zhimg.com/v2-73d7cf21aa863d6f52fb67181c64f177_r.jpg)

整个Swin Transformer架构，和CNN架构非常相似，构建了4个stage，每个stage中都是类似的重复单元。和ViT类似，通过patch partition将输入图片HxWx3划分为不重合的patch集合，其中每个patch（<font color='red'>可以理解为块</font>）尺寸为4x4，那么每个patch的特征维度为4x4x3=48，patch块的数量为H/4 x W/4；stage1部分，先通过一个linear embedding将输入划分后的patch特征维度变成C，然后送入Swin Transformer Block；stage2-stage4操作相同，先通过一个patch merging，将输入按照2x2的相邻patches合并，这样子patch块的数量就变成了H/8 x W/8，特征维度就变成了4C，这个地方文章写的不清楚，猜测是跟stage1一样使用linear embedding将4C压缩成2C，然后送入Swin Transformer Block



另外有一个细节，Swin Transformer和ViT划分patch的方式类似，<font color='red'>Swin Transformer也是先确定每个patch的大小，然后计算确定patch数量</font>。不同的是，随着网络深度加深ViT的patch数量不会变化，而Swin Transformer随着网络深度的加深数量会逐渐减少并且每个patch的感知范围会扩大，这个设计是为了方便Swin Transformer的层级构建，并且能够适应视觉任务的多尺度。



![img](https://pic1.zhimg.com/80/v2-c46495b418921e47588bd6d01118c0d8_720w.jpg)

另外W-MSA（multi-head self-attention）虽然降低了计算复杂度，但是不重合的window之间缺乏信息交流，于是作者进一步引入shifted window partition来解决不同window的信息交流问题，在两个连续的Swin Transformer Block中交替使用W-MSA和SW-MSA。以上图为例，将前一层Swin Transformer Block的8x8尺寸feature map划分成2x2个patch，每个patch尺寸为4x4，然后将下一层Swin Transformer Block的window位置进行移动，得到3x3个不重合的patch。移动window的划分方式使上一层相邻的不重合window之间引入连接，大大的增加了感受野。



![img](https://pic2.zhimg.com/80/v2-8fe0f1e4c0f76d93ec6986ba21b5979d_720w.jpg)

![image-20210723111743694](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210723111743694.png)

但是shifted window划分方式还引入了另外一个问题，就是会产生更多的windows，并且其中一部分window小于普通的window，比如2x2个patch -> 3x3个patch，windows数量增加了一倍多。于是作者提出了通过沿着左上方向cyclic shift的方式来解决这个问题，移动后，一个batched window由几个特征不相邻的sub-windows组成，因此使用masking mechanism来限制self-attention在sub-window内进行计算。cyclic shift之后，batched window和regular window数量保持一致，极大提高了Swin Transformer的计算效率。这一部分比较抽象复杂，不好理解，等代码开源了再补上。

![img](https://pic2.zhimg.com/80/v2-0fc6b9b6753a5416f6cf3d7361c967d1_720w.jpg)

## CSwin transformer (cross-shape window)

CSWin Transformer的网络结构如图1所示。它的输入是一个3通道彩色图像，尺寸为 ![[公式]](https://www.zhihu.com/equation?tex=H%5Ctimes+W%5Ctimes3) ，图像首先经过一组个步长为4的 ![[公式]](https://www.zhihu.com/equation?tex=7%5Ctimes7) 卷积，得到的Feature Map的尺寸为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7BH%7D%7B4%7D+%5Ctimes+%5Cfrac%7BH%7D%7B4%7D+%5Ctimes+C) 。这点比之前的直接无重叠的拆分是要有所提升的。之后CSWin Transformer分成4个Stage，Stage之间通过步长为2的 ![[公式]](https://www.zhihu.com/equation?tex=3%5Ctimes3) 卷积来进行降采样，这一点就和VGG等CNN网络结构很像了。

每个Stage是 ![[公式]](https://www.zhihu.com/equation?tex=N_i) 个CSwin Transformer Block组成，如图4所示。它的结构和传统的Transformer类似不同点有两个：

1. 将self-attention替换为提出的十字形窗口的self-attention;
2. 添加作者提出的LePE（Local-Enhanced Positional Encoding）位置编码。

![img](https://pic1.zhimg.com/80/v2-8c60da31dbe5cb5eefe3be12f6a6761c_720w.jpg)

## DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification  
> DynamicViT: Effificient Vision Transformers with Dynamic Token Sparsifification
> 论文地址：https://arxiv.org/abs/2106.02034
> 代码地址：https://github.com/raoyongming/DynamicViT
>
> 在Vision Transformer中，Attention其实是比较稀疏的。作者观察到，在视觉Transformer中的最终预测仅基于信息最丰富的token的一个子集，这足以进行准确的图像识别。基于这一观察结果，作者提出了一个动态token稀疏化框架，基于输入逐步、动态地删除冗余的token。具体来说，作者设计了一个轻量级的预测模块，来估计给定特征的每个token的重要性分数。该模块被添加到不同的层中，分层地删除冗余token。为了以端到端的方式优化预测模块，作者提出了一种attention masking策略，通过阻止一个token与其他token的交互来可区分地修剪一个token。得益于 self-attention的机制，非结构化的稀疏token对硬件是友好的，这使得本文的框架很容易实现实际的加速。通过分层修剪66%的输入token，本文的方法大大减少了31%∼37%FLOPs，提高了40%以上的吞吐量，而各种视觉Transformer的精度下降在0.5%以内。

# 目标检测

## YOLO v5(目标检测)

如果有空的话，可以用在微藻上试一下





## SSD(目标检测)

## [XNeT](https://www.cnblogs.com/zhongzhaoxie/p/14299143.html#1-%E8%83%8C%E6%99%AF%E4%BB%8B%E7%BB%8D)(姿态)-Max Planck Institute for informatics

论文地址：[https://gvv.mpi-inf.mpg.de/projects/XNect/](https://gvv.mpi-inf.mpg.de/projects/XNect/)

### 背景

光学人体动作捕捉技术

存在极具挑战性问题，比如深度模糊，遮挡，和外观和场景的多样性。

最近的一些方法侧重于自我中心设置。由外向内设置的单人跟踪（非自我中心）是已经受到严格的约束；因此多人跟踪变得异常困难，由于多重遮挡，挑战身体部分人员分配，计算上要求更高。这为许多应用程序（如游戏）带来了实际障碍以及社交VR/AR。

### **相关工作**

接下来重点讨论相关的2D和3D人体姿态估计，数据集和神经网络架构。

**多人2D姿态估计**：多人二维姿态估计方法可分为自顶向下（top-down）和自底向上（bottom-up）两种。top-down的方法先进行目标检测（人），再对每个检测到的人进行单人的关键点检测（单人姿态估计）；bottom-up方法先检测所有人的关键点，然后再对关键点进行分组关联。一般来说，top-down方法精度更高，而bottom-up方法速度更快。

**单人3D姿态估计：** 单人3D姿态估计以前是通过使用物理先验的生成方法进行的，或者半自动综合分析拟合参数体模型。最近的一些方法在网络中集成了3D身体模型，并使用2D和3D姿势的混合训练从单个图像中预测三维姿态和形状。其他方法优化人体模型或模板来适应2D姿势和轮廓。

**多人3D姿态估计：** 本文的方法是自底向上的，不会对每个人进行多次检测。自底向上的方法使用固定数量的特征映射来预测场景中所有个体的2D和3D姿势，并且为每个个体进行编码。3D编码将每个肢体和躯干视为不同的物体，并对特征中的每个“物体”的3D姿势进行编码映射到与“对象”的2D关节对应的像素位置。因此，该编码可以处理部分个人之间的遮挡通过不同的身体部位。

**3D姿态数据集：** MarCOnI、Panopticon、MuCo-3DHP等3D姿态数据集。

**卷积网络设计：** ResNet、Inception 和ResNext 等等变种；本文提出的CNN架构背后的关键是使用选择性的**远距离和近距离**合并连接，而不是DenseNet的密集连接模式。这使得网络速度明显快ResNet-50 同时保持相同的精度水平，避免精度和网络复杂度之间的平衡，消除内存瓶颈。

### 整体方法

![Untitled](https://imagerk.oss-cn-beijing.aliyuncs.com/img/Untitled%2032.png)

整体方法的大概的计算分为三个阶段，前两个阶段分别为每帧局部（每个身体关节）和全局（所有身体关节）推理，第三个阶段执行时序推理：第一个阶段为使用新的SelecSLS网络架构可见的身体关节推断2D姿态和中间3D姿态编码。每个关节的三维位姿编码仅考虑运动序列中的局部上下文。第二个阶段是一个紧凑的全连接网络为每个被检测到的人（并行运行），并通过利用全局上下文重建完整的3D姿势，包括被遮挡的关节。第三阶段提供了时序的稳定性，相对于摄像机的定位，以及通过运动学骨架拟合的关节角度参数。

### 姿态估计

#### Step 1

#### Step 2

#### Step 3

# 活体检测

## Searching Central Difference Convolutional Networks for Face Anti-Spoofing

### CDC-中心差分卷积

### CDCN++

主要是在DepthNet的技术上修改的，引入了NAS（神经架构搜索）和MAFM多级注意力融合模块

# 图像文本检索

## Dynamic Modality Interaction Modeling for Image-Text Retrieval（待看）

本文分享一篇 SIGIR 2021 最佳学生论文『Dynamic Modality Interaction Modeling for Image-Text Retrieval』，图像文本检索的动态模态交互建模。

https://mp.weixin.qq.com/s/xL5zr6QdkFwVH1g6tEjtSA

- 论文链接：https://dl.acm.org/doi/abs/10.1145/3404835.3462829
- 项目链接：未开源



# 视频段落等

## Towards Diverse Paragraph Captioning for Untrimmed Videos

论文地址：https://arxiv.org/abs/2105.14477
代码地址：https://github.com/syuqings/video-paragraph

视频段落字幕（Video paragraph captioning）的目的是在未修剪的视频中描述多个事件。现有的方法主要通过事件检测和事件字幕两个步骤来解决问题。这种二阶段的方式使生成的段落的质量高度依赖于事件建议（event proposal）检测的准确性，然而事件建议检测也是一项具有挑战性的任务。在本文中，作者提出了一个一阶段的段落字幕模型，避免了事件检测阶段，直接为未修剪的视频生成段落描述。为了描述连贯和多样化的事件，作者提出使用动态视频记忆来增强时间维度的Attention。通过逐步暴露新的视频特征，抑制过度访问的视频内容，来控制模型的视觉焦点。
此外，作者还提出了多样性驱动的训练策略，以提高语言角度的多样性。考虑到未修剪的视频通常包含大量冗余的帧，作者进一步用视频编码器提取关键帧，以提高效率。在ActivityNet和Charades数据集上的实验结果表明，作者提出的模型在不使用任何事件边界注释的情况下，在准确性和多样性度量上都显著优于目前SOTA模型的性能。



# 小样本

## Mining Latent Classes for Few-shot Segmentation





# 图像复原

## SwinIR: Image Restoration Using Swin Transformer

