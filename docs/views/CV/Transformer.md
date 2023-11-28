---
author: kii
title: Transformer
categories: [CV]
tags: [CV,DL]
date: 2023-08-13 10:44:30
---

<Boxx changeTime="10000"/>

::: tip 前言

Transformer的numpy实现

:::

<!-- more -->

下面的代码自下而上的实现Transformer的相关模块功能。这份文档只实现了主要代码。由于时间关系，我无法实现所有函数。对于没有实现的函数，默认用全大写函数名指出，如SOFTMAX

由于时间限制，以下文档只是实现了Transformer前向传播的过程。

## 输入层

输入层包括Word Embedding和Positional Encoding。Word Embedding可以认为是预训练的词向量，Positional Encoding用于捕获词语的相对位置信息。

$\begin{aligned} PE(pos, 2i) &= sin(pos / 10000^{\frac{2i}{d}}) \\ PE(pos, 2i+1) &= cos(pos / 10000^{\frac{2i}{d}}) \end{aligned}$

```python
import numpy as np

# Word embedding matrix。通常从文件读入，这里随机初始化
# word_embedding = np.arange(10)
# word_embedding.reshape(vocabulary_size, word_embedding_size)

max_seq_len = 200 # 假定的最大序列长度
position_size = 512 # Position Embedding的维度

# position_encoding是一个类似于word embeding的二维矩阵
# 其中pos是序列中词语的位置，j是维度
position_encoding = np.array([
          [pos / np.power(10000, 2.0 * (j // 2) / position_size) for j in range(position_size)]
          for pos in range(max_seq_len)])
print("Shape of position encoding: {}".format(position_encoding.shape))

# 每个position encoding的偶数列使用sin，奇数列使用cos处理
position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

# 为了对文本长度对齐，加上Padding行
padding = np.zeros(position_size)
position_encoding = np.vstack((padding, position_encoding))
print("Shape of position encoding after adding padding: {}".format(position_encoding.shape))


def position_encoding(sentence_lens):
    """
    给定一个batch的句子，输出这些句子的Position Embedding
    """
    # 模拟输入，batch_size=4
    sentence_lens = np.array([3,4,5,6])
    print("Shape of input: {}".format(input_len.shape))

    # 生成输入的位置索引，shape[batch_size, max_seq_len]
    # 避开0的索引，不够长度的部分采用0填充
    pos_index = np.array([list(range(1, len+1)) + [0] * (max_seq_len - len) for len in sentence_lens])

    # 利用pos_index在position_encoding中进行Lookup
    position_embedding = LOOKUP(pos_index, position_encoding)

    # 返回维度[batch_size, max_seq_len, position_size]
    return position_embedding

def word_embdding(sentence_words):
    """
    给定一个batch句子，输出这些句子的Word Embedding
    """
    # 将word转换为index，通常输入前就做完了
    word_index = WORD2INDEX(sentences_words)
    word_embedding = LOOKUP(word_index, word_embedding)

    # 返回维度[batch_size, max_seq_len, word_embedding_size]
    return word_embedding
```

输出

```
Shape of position encoding: (200, 512)
Shape of position encoding after adding padding: (201, 512)
```

得到positional encoding和word embedding之后，将两部分拼接，得到输入向量

## 层标准化

层标准化将数据标准化为均值为0，标准差为1.以下是实现代码

$BN(x_i)=\alpha \times \frac{x_i - \mu}{\sqrt{\delta^2 + \epsilon}}+\beta$

```python
def base_layer_norm(x):
    """
    标准化张量x,假设x是三维张量，即
    x.shape = (B, L, D)
    通常第2维是我们要标准化的维度
    """
    # 求均值
    mean = np.mean(x, axis=2)
    # 求标准差
    std = np.std(x, axis=2)

    return (x - mean) / std

def layer_norm(x):
    """
    引入可学习参数gamma、beta, epsilon用来防止发生数值计算错误
    """
    # 求均值
    mean = np.mean(x, axis=2， keepdims=True)
    # 求标准差
    std = np.std(x, axis=2, keepdims=True)

    return gamma * (x - mean) / ((std + epsilon) + beta)
```

## 缩放点积

因为缩放点积(Scaled dot-product Attention)是Self-Attention的基础，因此这里先实现它。该模块输入是K,Q,V三个张量，输出Context上下文张量和Attention张量

$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$

```python
def scaled_dot_product_attention(query, key, value):
    """
    Args:
        query: [batch_size, query_len, query_size]
        key: [batch_size, key_len, key_size]
        value: [batch_size, value_len, value_size]
    """
    scale = 1 / np.sqrt(key_size) # 缩放比例
    att = np.matmul(query, key.swapaxes(1,2)) / scale
    # 利用softmax将att转换为一个概率分布
    att = SOFTMAX(att)
    # 得到上下文张量
    context = np.matmul(att, value)

    return contenx, att
```

## Multi-head Attention

论文中使用了8个head，也就是把上述的K，Q，V三个张量按照维度分为8份，每份都经过仿射变换后送入到缩放点积中。

主要流程为：将K，Q，V进行仿射变换，得到对应的query，key和value；然后将它们根据head数目进行维度划分，送入到对应的缩放点积模块进行训练，得到Context张量和Attention张量；多个head的Context张量拼接后经过线性变换就得到了全局的Context张量；最后为了使模型能够更深，收敛更快，对输出加上了dropout，残差连接和层标准化。

下面是代码实现：

$MultiHead(Q,K,V)=Concat(head_1, head_2,\cdots,head_h)W_c + b_c$

```python
def multihead_attention(query, key, value, num_heads=8, input_dim=512):
    """
    Args:
        query, key, value和缩放点积部分一致
        num_heads: multi-head attention 个数
        input_dim: 输入维度
    """
    # 恒等映射的残差，先保存下来
    residual = query

    # 每个head分到的维度大小
    per_head = input_dim // num_heads

    # 对query，key，value进行仿射运算
    # W_q,W_k,W_v是三个可学习二维矩阵，shape=[input_dim, (input_dim // num_heads)*num_heads]
    query= np.matmul(query, W_q) + b_q
    key = np.matmul(key, W_k) + b_k
    value = np.matmul(key, W_v) + b_v

    # 根据每个head分到的维度对query，key，value重新切分
    qeury = query.reshape(batch_size * num_heads, -1, per_head)
    key = key.reshape(batch_size * num_heads, -1, per_head)
    value = value.reshape(batch_size * num_heads, -1, per_head)

    # 对切分的query，key，value进行缩放点积
    context, att = scaled_dot_product_attention(query, key, value)

    # 将各个head的上下文向量拼接得到最终的context向量
    context = context.reshape(batch_size, -1, per_head * num_heads)

    # context还需要经过一个线性变换,其中W_c是可学习二维矩阵，shape=[input_dim, input_dim]
    context = np.matmul(context, W_c) + b_c

    # dropout层
    context = DROPOUT(context)

    # 输出前进行残差连接和层标准化
    output = layer_norm(residual + context)

    # 输出
    return output, att
```

## Mask

Transformer中有Padding Mask和Sequence Mask。Padding Mask在计算Attention时用来消除某些位置的Attention值，使其在上下文张量中不起作用。Sequence Mask用于Decoder部分，主要是Mask掉当前输出词之后的序列，因为解码过程中是不知道后续词信息的。

为简单起见，上面的Attention都没有考虑Padding Mask。

## Feed Forward层

该全连接网络首先将输入x做了一次仿射变换，然后经过ReLU激活函数，再做一次仿射变化，得到最终的输出。

$FFN(x)=ReLU(xW_1+b_1)W_2 + b_2$

```python
def feed_forward(x):
    # 进行一次仿射变换，其中W_1和b_1分别为矩阵和偏置
    out = np.matmul(x, W_1) + b_1
    # 施加激活函数
    out = ReLU(out)
    # 再进行仿射运算，其中W_2和b_2分别为矩阵和偏置
    out = np.matmul(out, W_2) + b_2

    # Dropout
    out = DROPOUT(out)

    # 添加残差连接和层标准化
    return layer_norm(x + out)
```

## Encoder

整个的Encoder有流程，每一层都是Multi-head Attention和Feed Forward模块组成。代码如下：

```python
class EncoderLayer(object):
    """
    Encoder部分一层的结构表示
    每层中有Multi-head Attention和Feed Forward前向网络
    """

    def __init__(self):
        """
        一些参数设置，如head大小，输入维度等
        """
        pass

    def encode(self, inputs):
        # Multi-head Attention

        # 先从inputs中获得对应的query，key，value
        query = inputs.GET_QUERY()
        key = inputs.GET_KEY()
        value = inputs.GET_VALUE()
        context, attention = multihead_attention(query, key, value, num_heads, input_dim)

        # Feed forward层
        output = feed_forward(context)

        return output, attention


class Encoder(object):
    """
    完整Encoder的表示
    """

    def __init__(self):
        # 定义Encoder所有的层
        self.encoder_layers = [layer1, layer2, ... ,layer6]

    def forward(self, inputs, input_lens):
        # 获得嵌入表示
        word_embedding = word_embedding(inputs)
        position_embedding = position_encoding(inputs_lens)
        final_embedding = word_embedding + position_embedding

        # 一层层进行编码
        final_attention = []
        for layer in self.encoder_layers:
            output, attention = layer.encode(final_embedding)
            final_attention.append(attention)

        # output只返回最后一层，attention全部返回
        return output, attention
```

## Decoder

Decoder的除了和Encoder一样，有Multi-head Attention和Feed Forward外，还有一层Masked Multi-head Attention在最下面。代码如下：

```python
class DecoderLayer(object):
    """
    Decoder部分一层的结构表示
    每层中有两个Multi-head Attention和一个Feed Forward前向网络模块
    """

    def decode(self, encoder_output, decoder_inputs):
        """
        与Encoder不同，Decoder不仅关注自己的输入，还要考虑Encoder的输出
        """
        # 下层Multi-head Attention
        # 先从decoder_inputs中获得对应的query，key，value
        query = decoder_inputs.GET_QUERY()
        key = decoder_inputs.GET_KEY()
        value = decoer_inputs.GET_VALUE()
        output, attention1 = multihead_attention(query, key, value, num_heads, input_dim)

        # 上层Multi-head Attention
        # 再从encoder_outputs中获取key和value，decoder的output中获取query
        query = output.GET_QUERY()
        key = encoder_outputs.GET_KEY()
        value = encoder_output.GET_VALUE()
        output, attention2 = multihead_attention(query, key, value, num_heads, input_dim)

        # Feed forward层
        output = feed_forward(output)

        return output, attention1, attention2

class Decoder(object):
    """
    完整的Decoder表示
    """
    def __init__(self):
        # 定义Dncoder所有的层
        self.decoder_layers = [layer1, layer2, ... ,layer6]

    def forward(self, inputs, input_lens, encoder_outputs):
        # 获得嵌入表示
        word_embedding = word_embedding(inputs)
        position_embedding = position_encoding(inputs_lens)
        final_embedding = word_embedding + position_embedding

        # Sequence Mask。解码过程中要做Sequence Mask
        seq_mask = SEQUENCE_MASK(inputs)

        # 一层层进行解码
        self_attentions = []
        context_attentions = []
        for layer in self.decoder_layers:
            output, self_attention, context_attention = layer.decode(encoder_outputs, final_embedding)
            self_attentions.append(self_attention)
            context_attentions.append(context_attention)

        # output只返回最后一层，attention全部返回
        return output, self_attentions, context_attentions
```

## Transformer整体

```python
class Transformer(object):
    """
    Transformer整体代码
    """
    def __init__(self):
        """
        参数设置：参数主要有
        Args:
            src_vocab_size: 源语言词汇表大小
            src_max_len: 源语言语句最大长度
            tgt_vocab_size: 目标语言词汇表大小
            tgt_max_len: 目标语言语句最大长度
            num_layers=6: 默认Encoder和Decoder为6层
            inputs_dim=512: 输入维度默认为512
            num_heads=8: 默认Multi-head Attention个数为8
            feed_forward_dim=2048：前馈网络维度
            drop_out=0.2: Dropout概率
        """
        self.encoder = Encoder() # 构造编码器
        self.decoder = Decoder() # 构造解码器

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        """
        编解码一个batch的过程
        Args:
            src_seq: 源语言序列
            src_len: 源语言序列长度
            tgt_seq: 目标语言序列
            tgt_len: 目标语言序列长度
        """
        # 编码过程
        output, encoder_attention = self.encoder.forward(src_seq, src_len)

        # 解码过程
        output, self_attention, context_attention = self.decoder.forward(tgt_seq, tgt_len, output)

        # 最终要输出概率，所以最终结果还要经过线性层和softmax层
        output = np.matmul(output, W_T) + b_T # 其中，W_T和b_T是线性层的二维矩阵和偏置
        # 输出概率
        output = SOFTMAX(output)

        return output, encoder_attention, self_attention, context_attention
```