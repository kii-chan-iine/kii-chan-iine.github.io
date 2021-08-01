---
author: kii
title: Transformer & Bert
categories: [深度学习]
tags: [NLP,deeplearn]
date: 2021-07-19 23:00:00
---

<Boxx changeTime="10000"/>

::: tip 前言
学习札记。
:::
<!-- more -->

# Transformer(是一个模型，RNN，LSTM)

可用于机器翻译，转写等

> - 输入：我爱中国
> - 输出： I Love China
>
> Input Embedding ：我，爱，中国 这三个词的词向量(word2vector等词嵌入方法获得)
>
> Positional Encoding: 利用词的位置和长度对位置进行编码，代码实现可在Google开源的算法中`get_timing_signal_1d()`函数找到。
>
> 词向量与位置编码相加，就能给每个词赋上位置信息。
>
> **模型执行**
>
> **Encoder步骤：**
>
> 每次的输入都是我，爱，中国 这三个词的词向量和其位置编码的和。
>
> **Decoder步骤(**每一步都预测一个词**):**
>
> 
> **Step 1**
>
> - - Outputs： 起始符</s> + Positional Encoding（位置编码）
>   - 输出最大概率的词：“I”
>
> **Step 2**
>
> - - 初始输入：起始符</s> + “I”+ Positonal Encoding
>   - 输出最大概率的词：“Love”
>
> **Step 3**
>
> - - 初始输入：起始符</s> + “I”+ “Love”+ Positonal Encoding
>   - 最终输出：产生预测“China”

## 整体框架

和Attention一样，Transformer模型中也采用了 encoer-decoder 架构。但其结构相比于Attention更加复杂，论文中encoder层由6个encoder堆叠在一起，decoder层也一样。

<img src="https://img-blog.csdnimg.cn/20190407193306430.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ppYW93b3Nob3V6aQ==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom: 50%;" />

每一个encoder和decoder的内部简版结构如下图

<img src="https://img-blog.csdnimg.cn/2019040719332630.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ppYW93b3Nob3V6aQ==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:80%;" />

对于encoder，包含两层，一个self-attention层和一个前馈神经网络，self-attention能帮助当前节点不仅仅只关注当前的词，从而能获取到上下文的语义。**decoder也包含encoder提到的两层网络，但是在这两层中间还有一层<font color='red'>Encoder-Decoder attention层</font>，帮助当前节点获取到当前需要关注的重点内容。**

现在我们知道了模型的主要组件，接下来我们看下模型的内部细节。首先，<font color='green'>模型需要对输入的数据进行一个**Positional Embedding**操作，（也可以理解为类似w2v的操作，考虑词在句子中的位置顺序关系）</font>，enmbedding结束之后，输入到encoder层，self-attention处理完数据后把数据送给前馈神经网络，前馈神经网络的计算可以并行，得到的输出会输入到下一个encoder。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190407193828541.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ppYW93b3Nob3V6aQ==,size_16,color_FFFFFF,t_70)

再宏观缩放一下

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190407194033648.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ppYW93b3Nob3V6aQ==,size_16,color_FFFFFF,t_70)

换个方式，Multi-head attention(一次初始化多个KVQ)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190407194054634.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ppYW93b3Nob3V6aQ==,size_16,color_FFFFFF,t_70)

## **Positional Embedding** 

类似加入位置信息

![image-20210719223938607](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210719223938607.png)

### 具体

到目前为止，transformer模型中还缺少一种**解释输入序列中单词顺序**的方法。为了处理这个问题，transformer给encoder层和decoder层的输入添加了一个额外的向量Positional Encoding，维度和embedding的维度一样，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离。这个位置向量的具体计算方法有很多种，论文中的计算方法如下
$$
PE=(pos,2i)=sin(pos/10000^{2i}/d_model)\\
PE=(pos,2i+1)=cos(pos/10000^{2i}/d_model)
$$


其中pos是指当前词在句子中的位置，i是指向量中每个值的index，可以看出，在偶数位置，使用正弦编码，在奇数位置，使用余弦编码。最后把这个Positional Encoding与embedding的值相加，作为输入送到下一层。

## Self-Attention

例子分为以下步骤：

1. 准备输入
2. 初始化权重
3. 导出key, query and value的表示
4. 计算输入1 的注意力得分(attention scores)
5. 计算softmax
6. 将attention scores乘以value
7. 对加权后的value求和以得到输出1
8. 对输入2重复步骤4–7

**Note:**


实际上，数学运算是向量化的，即所有输入都一起进行数学运算。我们稍后会在“代码”部分中看到此信息。

### **1 准备输入**

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzNG82WjZkdmxEVmtaa0M0ZEZLOWRBVGc5OVRLSk5aWHNtNGJrWGcyV0xzOW1uR1M3WTk3YmFRLzY0MA?x-oss-process=image/format,png)

Fig. 1.1: Prepare inputs

在本教程中，我们从3个输入开始，每个输入的尺寸为4。

```css
    Input 1: [1, 0, 1, 0]     Input 2: [0, 2, 0, 2]    Input 3: [1, 1, 1, 1]
```

### **2 初始化权重**

每个输入必须具有三个表示形式（请参见下图）。这些表示称为key（橙色），`query（红色）和value（紫色）。在此示例中，假设我们希望这些表示的尺寸为3。由于每个输入的尺寸均为4，这意味着每组权重的形状都必须为4×3。

**Note:**


 稍后我们将看到value的维度也就是输出的维度。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzWW04N3VueHRiVVh6cHpndnN3UnBUSFFLeXNxY2liSWt6N0ZCZzM1ZXI3QVh2VTBFckJSclIwUS82NDA?x-oss-process=image/format,png)

Fig. 1.2: Deriving key, query and value representations from each input

为了获得这些表示，将每个输入（绿色）乘以一组用于key的权重，另一组用于query的权重和一组value的权重。在我们的示例中，我们如下初始化三组权重。

**key的权重**

```json
    [[0, 0, 1],     [1, 1, 0],     [0, 1, 0],     [1, 1, 0]]
```

**query的权重**

```json
    [[1, 0, 1],     [1, 0, 0],     [0, 0, 1],     [0, 1, 1]]
```

**value的权重**

```json
    [[0, 2, 0],     [0, 3, 0],     [1, 0, 3],     [1, 1, 0]]
```

**Note:**

在神经网络的设置中，这些权重通常是很小的数，使用适当的随机分布（如高斯，Xavie 和 Kaiming 分布）随机初始化。初始化在训练之前完成一次。

### **3 从每个输入中导出key, query and value的表示**

现在我们有了三组值的权重，让我们实际查看每个输入的键，查询和值表示形式。

**输入 1 的key的表示形式**

```cs
                   [0, 0, 1]



    [1, 0, 1, 0] x [1, 1, 0] = [0, 1, 1]



                   [0, 1, 0]



                   [1, 1, 0]
```

使用相同的权重集获得输入 2 的key的表示形式：

```cs
                   [0, 0, 1]



    [0, 2, 0, 2] x [1, 1, 0] = [4, 4, 0]



                   [0, 1, 0]



                   [1, 1, 0]
```

使用相同的权重集获得输入 3 的key的表示形式：

```cs
                   [0, 0, 1]



    [1, 1, 1, 1] x [1, 1, 0] = [2, 3, 1]



                   [0, 1, 0]



                   [1, 1, 0]
```

一种更快的方法是对上述操作进行矩阵运算：

```cs
                   [0, 0, 1]



    [1, 0, 1, 0]   [1, 1, 0]   [0, 1, 1]



    [0, 2, 0, 2] x [0, 1, 0] = [4, 4, 0]



    [1, 1, 1, 1]   [1, 1, 0]   [2, 3, 1]
```

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzbDJwYVUydndPUEJpY1hscHdhdXdBdXRsM21rNkJ5SktUS3l3eGgzWXB6RHRvaDRYVnV5aWE5ZncvNjQw?x-oss-process=image/format,png)

Fig. 1.3a: Derive key representations from each input

让我们做同样的事情以获得每个输入的value表示形式：

```cs
                   [0, 2, 0]



    [1, 0, 1, 0]   [0, 3, 0]   [1, 2, 3] 



    [0, 2, 0, 2] x [1, 0, 3] = [2, 8, 0]



    [1, 1, 1, 1]   [1, 1, 0]   [2, 6, 3]
```

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzY0pCUWliNWljVE5rTkpBUHhuWjFkdmNsZEpGUmp2TUVyNXE0QXFpYnhkZVNZR1o3aWM3Rkh3dTEzZy82NDA?x-oss-process=image/format,png)

Fig. 1.3b: Derive value representations from each input

以及query的表示形式:

```cs
                   [1, 0, 1]



    [1, 0, 1, 0]   [1, 0, 0]   [1, 0, 2]



    [0, 2, 0, 2] x [0, 0, 1] = [2, 2, 2]



    [1, 1, 1, 1]   [0, 1, 1]   [2, 1, 3]
```

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzcjN6QU5nRmRpYkNpYWxkM1Z4cnFTajBlak1SQm92U3dWT0lSeEY1WTRiTnBpYmZiTmRuUEhMZkJRLzY0MA?x-oss-process=image/format,png)

Fig. 1.3c: Derive query representations from each input

**Notes:**

实际上，可以将偏差向量b添加到矩阵乘法的乘积中。

译者注:y=w·x+b

### **4 计算输入的注意力得分(attention scores)**

为了获得注意力分数，我们首先在输入1的query（红色）与所有key（橙色）（包括其自身）之间取点积。由于有3个key表示（因为我们有3个输入），因此我们获得3个注意力得分（蓝色）。

```cs
                [0, 4, 2]



    [1, 0, 2] x [1, 4, 3] = [2, 4, 4]



                [1, 0, 1]
```

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzY0JQOVVNVTE0eWljUTVrMVRUT3F1WXdzdmVGeEt0ak00ZEN6YjlsS3VXZEpDalExdEM4OFZDQS82NDA?x-oss-process=image/format,png)

Fig. 1.4: Calculating attention scores (blue) from query 1

请注意，在这里我们仅使用输入1的query。稍后，我们将对其他查询重复相同的步骤。

**Note:**


上面的操作被称为"点积注意力"，是几种sorce之一。其他评分功能包括缩放的点积和拼接。

更多：

sorce:https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3

### **5 计算softmax**

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzZTFnU0JlQmFjcVZXeFdvaWExS3A5clUyUjNLaWM0YmQ2WFRVNTh6WWliZWlheER3UjVEQnA4NGpaQS82NDA?x-oss-process=image/format,png)

Fig. 1.5: Softmax the attention scores (blue)

将attention scores通过 softmax 函数(蓝色)得到概率

```apache
    softmax([2, 4, 4]) = [0.0, 0.5, 0.5]
```

### **6 将attention scores乘以value**

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzb2lheGI2RUM3QjlPVDBsSFY2dGw3S1FyRWdlaWFXZlZHMmxSMUhjYmJSVGVBTktBUXIxaE5zNUEvNjQw?x-oss-process=image/format,png)

Fig. 1.6: Derive weighted value representation (yellow) from multiply value(purple) and score (blue)

每个输入的softmax注意力得分（蓝色）乘以其相应的value（紫色）。这将得到3个对齐的向量（黄色）。在本教程中，我们将它们称为"加权值"。

```http
    1: 0.0 * [1, 2, 3] = [0.0, 0.0, 0.0]



    2: 0.5 * [2, 8, 0] = [1.0, 4.0, 0.0]



    3: 0.5 * [2, 6, 3] = [1.0, 3.0, 1.5]
```

### **7 对加权后的value求和以得到输出1**

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzb2lheGI2RUM3QjlPVDBsSFY2dGw3S1FyRWdlaWFXZlZHMmxSMUhjYmJSVGVBTktBUXIxaE5zNUEvNjQw?x-oss-process=image/format,png)

Fig. 1.7: Sum all weighted values (yellow) to get Output 1 (dark green)

对所有加权值(黄色)按元素求和：

```cs
      [0.0, 0.0, 0.0]



    + [1.0, 4.0, 0.0]



    + [1.0, 3.0, 1.5]



    -----------------



    = [2.0, 7.0, 1.5]
```

得到的向量[2.0, 7.0, 1.5] (深绿)是输出 1 , 它是基于“输入1”的“query表示的形式” 与所有其他key(包括其自身）进行的交互。

### **8 对输入2重复步骤4–7**

现在我们已经完成了输出1，我们将对输出2和输出3重复步骤4至7。我相信我可以让您自己进行操作????????。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X2dpZi9pYVRhOHV0NkhpYXdEMVU3TkF4eGVCdzhWRURyT0paaDYzQ2dhdmN2Y3JNcVFLWnZCOGdBSnFBdGxsZmNHTnBMaWNLQWtsaWM1NmhReGZyWXZsb1hqSHpqMlEvNjQw?x-oss-process=image/format,png)

Fig. 1.8: Repeat previous steps for Input 2 & Input 3

Notes:

因为点积得分函数 query和key的维度必须始终相同.但是value的维数可能与query和key的维数不同。因此输出结果将遵循value的维度。

### 9 来个总结

![image-20210719225018921](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210719225018921.png)

### **代码**



```python
# %% 准备输入
from torch.nn.functional import softmax
import torch
x = [
    [1, 0, 1, 0],  # Input 1
    [0, 2, 0, 2],  # Input 2
    [1, 1, 1, 1]  # Input 3
]
x = torch.tensor(x, dtype=torch.float32)

# %% 初始化权重

w_key = [
    [0, 0, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 0]
]
w_query = [
    [1, 0, 1],
    [1, 0, 0],
    [0, 0, 1],
    [0, 1, 1]
]
w_value = [
    [0, 2, 0],
    [0, 3, 0],
    [1, 0, 3],
    [1, 1, 0]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)


# %% 导出KQV的表示
keys = x @ w_key
querys = x @ w_query
values = x @ w_value
print(keys)

print(querys)

print(values)
# %%  计算输入的注意力得分(attention scores)
attn_scores = querys @ keys.T
# tensor([[ 2.,  4.,  4.],  # attention scores from Query 1
#         [ 4., 16., 12.],  # attention scores from Query 2
#         [ 4., 12., 10.]]) # attention scores from Query 3
# %%  计算softmax
attn_scores_softmax = softmax(attn_scores, dim=-1)
print(attn_scores_softmax)
# tensor([[6.3379e-02, 4.6831e-01, 4.6831e-01],
#         [6.0337e-06, 9.8201e-01, 1.7986e-02],
#         [2.9539e-04, 8.8054e-01, 1.1917e-01]])

# For readability, approximate the above as follows
#以下为近似值
attn_scores_softmax = [
    [0.0, 0.5, 0.5],
    [0.0, 1.0, 0.0],
    [0.0, 0.9, 0.1]
]
attn_scores_softmax = torch.tensor(attn_scores_softmax)
# %% 将attention scores乘以value
weighted_values = values[:, None] * attn_scores_softmax.T[:, :, None]

print(weighted_values.size())
# %%
outputs = weighted_values.sum(dim=0)
print(outputs.size())
print(outputs)

#%%


```

**Note：**

PyTorch has provided an API for this called* *nn.MultiheadAttention*. However, this API requires that you feed in key, query and value PyTorch tensors. Moreover, the outputs of this module undergo a linear transformation.

## Encoder-Decoder

### Mask

mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果。Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。

<font color='red'>**其中，padding mask 在所有的 scaled dot-product attention 里面都需要用到，而 sequence mask 只有在 decoder 的 self-attention 里面用到。**</font>

#### Padding Mask

什么是 padding mask 呢？因为**每个批次输入序列长度是不一样的**也就是说，我们要对输入序列进行对齐。具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。

具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！

而我们的 padding mask 实际上是一个张量，每个值都是一个Boolean，值为 false 的地方就是我们要进行处理的地方。

#### Sequence mask

文章前面也提到，sequence mask 是为了使得 decoder 不能看见未来的信息。也就是对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。

那么具体怎么做呢？也很简单：产生一个上三角矩阵，上三角的值全为0。把这个矩阵作用在每一个序列上，就可以达到我们的目的。

对于 decoder 的 self-attention，里面使用到的 scaled dot-product attention，同时需要padding mask 和 sequence mask 作为 attn_mask，具体实现就是两个mask相加作为attn_mask。
其他情况，attn_mask 一律等于 padding mask。

### 具体事项

***注意encoder的输出并没直接作为decoder的直接输入。\***

训练的时候，

1. 初始decoder的time step为1时(也就是第一次接收输入)，其输入为一个特殊的token，可能是目标序列开始的token(如<BOS>)，也可能是源序列结尾的token(如<EOS>)，也可能是其它视任务而定的输入等等，不同源码中可能有微小的差异，其目标则是**预测翻译后的第1个单词(token)**是什么；

2. 然后<BOS>和预测出来的第1个单词一起，再次作为decoder的输入，得到第2个预测单词；3后续依此类推；

具体的例子如下：

样本：“我/爱/机器/学习”和 "i/ love /machine/ learning"

**训练：**

1. 把“我/爱/机器/学习”embedding后输入到encoder里去，最后一层的encoder最终输出的outputs [10, 512]（假设我们采用的embedding长度为512，而且batch size = 1),此outputs 乘以新的参数矩阵，可以作为decoder里每一层用到的K和V；

2. 将<bos>作为decoder的初始输入，将decoder的最大概率输出词 A1和‘i’做cross entropy计算error。

3. 将<bos>，"i" 作为decoder的输入，将decoder的最大概率输出词 A2 和‘love’做cross entropy计算error。

4. 将<bos>，"i"，"love" 作为decoder的输入，将decoder的最大概率输出词A3和'machine' 做cross entropy计算error。

5. 将<bos>，"i"，"love "，"machine" 作为decoder的输入，将decoder最大概率输出词A4和‘learning’做cross entropy计算error。

6. 将<bos>，"i"，"love "，"machine"，"learning" 作为decoder的输入，将decoder最大概率输出词A5和终止符</s>做cross entropy计算error。

**Sequence Mask**

上述训练过程是**挨个单词串行进行的**，那么能不能并行进行呢，当然可以。可以看到上述单个句子训练时候，输入到 decoder的分别是

<bos>

<bos>，"i"

<bos>，"i"，"love"

<bos>，"i"，"love "，"machine"

<bos>，"i"，"love "，"machine"，"learning"

那么为何不将这些输入组成矩阵，进行输入呢？这些输入组成矩阵形式如下：

【<bos>

<bos>，"i"

<bos>，"i"，"love"

<bos>，"i"，"love "，"machine"

<bos>，"i"，"love "，"machine"，"learning" 】

怎么操作得到这个矩阵呢？

将decoder在上述2-6步次的输入补全为一个完整的句子

【<bos>，"i"，"love "，"machine"，"learning"
<bos>，"i"，"love "，"machine"，"learning"
<bos>，"i"，"love "，"machine"，"learning"
<bos>，"i"，"love "，"machine"，"learning"
<bos>，"i"，"love "，"machine"，"learning"】

然后将上述矩阵矩阵乘以一个 mask矩阵

【1 0 0 0 0

1 1 0 0 0

1 1 1 0 0

1 1 1 1 0

1 1 1 1 1 】

这样是不是就得到了

【<bos>

<bos>，"i"

<bos>，"i"，"love"

<bos>，"i"，"love "，"machine"

<bos>，"i"，"love "，"machine"，"learning" 】

这样的矩阵了 。着就是我们需要输入矩阵。这个mask矩阵就是 sequence mask，其实它和encoder中的padding mask 异曲同工。

这样将这个矩阵输入到decoder（其实你可以想一下，此时这个矩阵是不是类似于批处理，矩阵的每行是一个样本，只是每行的样本长度不一样，每行输入后最终得到一个输出概率分布，作为矩阵输入的话一下可以得到5个输出概率分布）。

这样我们就可以进行并行计算进行训练了。

**测试**

训练好模型， 测试的时候，比如用 '机器学习很有趣'当作测试样本，得到其英语翻译。

这一句经过encoder后得到输出tensor，送入到decoder(并不是当作decoder的直接输入)：

1. 然后用起始符<bos>当作decoder的 输入，得到输出 machine

2. 用<bos> + machine 当作输入得到输出 learning

3. 用 <bos> + machine + learning 当作输入得到is

4. 用<bos> + machine + learning + is 当作输入得到interesting

5. 用<bos> + machine + learning + is + interesting 当作输入得到 结束符号<eos>

我们就得到了完整的翻译 'machine learning is interesting'

可以看到，在测试过程中，只能一个单词一个单词的进行输出，是串行进行的。

# Bert（预训练模型，先放放）

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

https://zhuanlan.zhihu.com/p/364966458

BERT提供了简单和复杂两个模型，对应的超参数分别如下：

- ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BBERT%7D_%7B%5Cmathbf%7BBASE%7D%7D) : L=12，H=768，A=12，参数总量110M；
- ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BBERT%7D_%7B%5Cmathbf%7BLARGE%7D%7D) : L=24，H=1024，A=16，参数总量340M；

在上面的超参数中，L表示网络的层数（即**Transformer blocks的数量**），A表示Multi-Head Attention中self-Attention的数量，filter的尺寸是4H。隐含层大小H。



Bert包括两个阶段:<font color='red'>预训练阶段(Pre-train)、微调阶段(Fine Tuning)</font>

> 预训练：一个寻网络权值初值的过程，将pre-train的结果作为BP算法的权值的初值，能够解决深度网络在非凸目标函数上陷入局部最优的问题（没有目标，不考虑结果，无监督的学习：如，nlp，学习到单词与单词之间的关系，单词的表示）
>
> Fine Turning：用**训练好的参数**（可以从已训练好的模型中获得）初始化自己的网络，然后用自己的数据接着训练（专业目的性学习，有监督学习）
>
> ```mermaid
> graph LR
> aa[Bert]-->A[Pre-Train]
> aa[Bert]-->bb[Fine Tune]
> A[Pre-Train]-->B[语言模型-Masked LM]
> A[Pre-Train]-->C[下一个句子的预测-Next Seq Predict]
> ```
>
> **语言模型：**首先修改原文章中的句子。
>
> 1. 80%的概率真的用[MASK]取代被选中的词：--解决完形填空
>    my dog is hairy-> my dog is [MASK]
>
> 2. 10%的概率用一个随机词取代它：--解决纠错问题
>    my dog is hairy-> my dog is apple
>
> 3. 10%的概率保持不变：
>    my dog is hairy-> my dog is hairy
>
> **Next Seq Predict**：
>
> 问答，出了上句，给出下句；一问一答。如果两句相关，训练给出yes；否则为no。
>
> **微调**：<font color='red'>前面的预训练就得到了向量化的表示,用向量化表示进行文本的初始化，然后做一个有监督的学习，就可以完成Fine tune了</font>
>
> •句子对的分类任务
> •单个句子的分类任务
> •问答任务
> •名命名实体识别

BERT模型具有以下两个特点：

第一，是这个模型非常的深，12层transformer，并不宽(wide），中间层只有1024，而之前的Transformer模型中间层有2048。这似乎又印证了计算机图像处理的一个观点——深而窄 比 浅而宽 的模型更好。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190407194131428.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ppYW93b3Nob3V6aQ==,size_16,color_FFFFFF,t_70)

**图1中的左侧部分是一个Transformer Block，对应到图2中的一个“Trm”。**

第二，MLM（Masked Language Model），同时利用左侧和右侧的词语，这个在ELMo上已经出现了，绝对不是原创。其次，对于Mask（遮挡）在语言模型上的应用，已经被Ziang Xie提出了（我很有幸的也参与到了这篇论文中）：[1703.02573] Data Noising as Smoothing in Neural Network Language Models。这也是篇巨星云集的论文：Sida Wang，Jiwei Li（香侬科技的创始人兼CEO兼史上发文最多的NLP学者），Andrew Ng，Dan Jurafsky都是Coauthor。但很可惜的是他们没有关注到这篇论文。用这篇论文的方法去做Masking，相信BRET的能力说不定还会有提升。

## **Embedding**

这里的Embedding由三种Embedding求和而成：

![img](https://imagerk.oss-cn-beijing.aliyuncs.com/img/v2-11505b394299037e999d12997e9d1789_720w.jpg)

其中：

- Token Embeddings是词向量，第一个单词是CLS标志，可以用于之后的分类任务
- Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务
- <font color='red'>Position Embeddings和之前文章中的Transformer不一样，不是三角函数而是学习出来的</font>

## **Pre-training Task 1#: Masked LM**

第一步预训练的目标就是做语言模型，从上文模型结构中看到了这个模型的不同，即bidirectional。**关于为什么要如此的bidirectional**，作者在[reddit](https://link.zhihu.com/?target=http%3A//www.reddit.com/r/MachineLearning/comments/9nfqxz/r_bert_pretraining_of_deep_bidirectional/)上做了解释，意思就是如果使用预训练模型处理其他任务，那人们想要的肯定不止某个词左边的信息，而是左右两边的信息。而考虑到这点的模型ELMo只是将left-to-right和right-to-left分别训练拼接起来。直觉上来讲我们其实想要一个deeply bidirectional的模型，但是普通的LM又无法做到，因为在训练时可能会“穿越”（**关于这点我不是很认同，之后会发文章讲一下如何做bidirectional LM**）。所以作者用了一个加mask的trick。

在训练过程中作者随机mask 15%的token，而不是把像cbow一样把每个词都预测一遍。**最终的损失函数只计算被mask掉那个token。**

Mask如何做也是有技巧的，如果一直用标记[MASK]代替（在实际预测时是碰不到这个标记的）会影响模型，所以随机mask的时候10%的单词会被替代成其他单词，10%的单词不替换，剩下80%才被替换为[MASK]。具体为什么这么分配，作者没有说。。。要注意的是Masked LM预训练阶段模型是不知道真正被mask的是哪个词，所以模型每个词都要关注。

因为序列长度太大（512）会影响训练速度，所以90%的steps都用seq_len=128训练，余下的10%步数训练512长度的输入。

## **Pre-training Task 2#: Next Sentence Prediction**

因为涉及到QA和NLI之类的任务，增加了第二个预训练任务，目的是让模型理解两个句子之间的联系。训练的输入是句子A和B，B有一半的几率是A的下一句，输入这两个句子，模型预测B是不是A的下一句。预训练的时候可以达到97-98%的准确度。

**注意：作者特意说了语料的选取很关键，要选用document-level的而不是sentence-level的，这样可以具备抽象连续长序列特征的能力。**

## **Fine-tunning**

分类：对于sequence-level的分类任务，BERT直接取第一个[CLS]token的final hidden state ![[公式]](https://www.zhihu.com/equation?tex=C%5Cin%5CRe%5EH) ，加一层权重 ![[公式]](https://www.zhihu.com/equation?tex=W%5Cin%5CRe%5E%7BK%5Ctimes+H%7D) 后softmax预测label proba： ![[公式]](https://www.zhihu.com/equation?tex=P%3Dsoftmax%28CW%5ET%29+%5C%5C)

其他预测任务需要进行一些调整，如图：

![img](https://pic2.zhimg.com/80/v2-b054e303cdafa0ce41ad761d5d0314e1_720w.jpg)

可以调整的参数和取值范围有：

- Batch size: 16, 32
- Learning rate (Adam): 5e-5, 3e-5, 2e-5
- Number of epochs: 3, 4

因为大部分参数都和预训练时一样，精调会快一些，所以作者推荐多试一些参数。
