---
author: kii
title: 变分自编码器VAE
categories: [CV]
tags: [CV,DL]
date: 2022-01-31 18:44:30
---

<Boxx changeTime="10000"/>

::: tip 前言

变分自编码器（Variational Auto-Encoder，VAE）

原论文 Auto-Encoding Variational Bayes。

:::
<!-- more -->

# VAE
## 思路
1. 假设对于每个样本$X_k$存在一个对应的后验概率$ p(Z|X)$是正态分布(为什么可以这样呢？EC：可以这样认为$X_k$是一个图片，图片中的像素点是一个服从正态的样本)

其实，**在整个 VAE 模型中，我们并没有去使用 p(Z)（先验分布）是正态分布的假设（*原因如下*），我们用的是假设 p(Z|X)（后验分布）是正态分布**。具体来说，给定一个真实样本 $X_k$，我们假设存在一个专属于$X_k$ 的分布<font color='red'> $p(Z|X_k)$</font>（学名叫后验分布）是（独立的、多元的）<font color='red'>正态分布</font>。

> 如果假设 p(Z) 是正态分布，然后从 p(Z) 中采样一个 Z，那么我们怎么知道这个 Z 对应于哪个真实的 X 呢？**现在 p(Z|Xk) 专属于 Xk，我们有理由说从这个分布采样出来的 Z 应该要还原到 Xk 中去**。Auto-Encoding Variational Bayes 中公式9强调了该点：
>
> <img src="http://latex.codecogs.com/gif.latex?log q_{\Phi}(z|x^{(i)})=log \mathbb{N}(z;\mu^{i},\sigma^{2(i)}I)" />

![image-20220128204047452](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220128204047452.png)

2. 接下来**那我怎么找出专属于 $X_k$ 的正态分布 $p(Z|X_k)$ 的均值和方差呢？**

   好像并没有什么直接的思路,那好吧，**我就用神经网络来拟合出来**。<font color='red'>构建两个神经网络$\mu_k=f_1(X_k)$，$log\sigma^2=f_2(X_k)$ 来算它们</font>

   > 我们选择拟合 <img src="https://latex.codecogs.com/svg.image?log\sigma^2"/> 而不是直接拟合<img src="https://latex.codecogs.com/svg.image?\sigma^2"/>，是因为<img src="https://latex.codecogs.com/svg.image?\sigma^2"/> 总是非负的，需要加激活函数处理，而拟合 <img src="https://latex.codecogs.com/svg.image?log\sigma^2"/>不需要加激活函数，因为它可正可负。

3. 从这个专属分布<font color='red'> $p(Z|X_k)$</font>中采样一个$ Z_k$ 出来，然后经过一个生成器得到$\hat{X}_k=g(Z_k)$。

4. 最小化 $D(\hat{X}_k,X_k)^2$。

   > 因为 $Z_k$ 是从专属 $X_k$ 的分布中采样出来的，这个生成器应该要把开始的$X_k$还原回来

我们希望重构 X，也就是最小化 $D(\hat{X}_k,X_k)^2$，但是这个重构过程受到噪声的影响，因为 $Z_k$ 是通过重新采样过的，不是直接由 encoder 算出来的。噪声会增加重构的难度，而这个噪声强度（也就是方差$D(\hat{X}_k,X_k)^2$​）是通过一个神经网络算出来的，所以最终模型为了重构得更好，肯定会想尽办法让方差为0。而方差为 0 的话，也就没有随机性了，所以不管怎么采样其实都只是得到确定的结果（也就是均值），只拟合一个当然比拟合多个要容易，而均值是通过另外一个神经网络算出来的。**模型会慢慢退化成普通的 AutoEncoder，噪声不再起作用**。

**其实 VAE 还让所有的 p(Z|X) 都向标准正态分布看齐**，这样就防止了噪声为零，同时保证了模型具有生成能力。原因如下：

如果所有的 p(Z|X) 都很接近标准正态分布 N(0,I)，那么根据定义：

![img](http://5b0988e595225.cdn.sohucs.com/images/20180323/0f0ef0e95f5b40fda562b26225dc48c3.png)

p(Z) 也是标准正态分布，这样，我们就可以从P(Z)中进行采样了。

## KL Loss（均值和方法两个）

如何将每个$p(Z_k)$都向标准正态看齐呢？<font color='red'>构建loss</font>

在重构误差的基础上中加入额外的 loss：

<img src="https://latex.codecogs.com/svg.image?L_\mu =\left\| f_1(X_k)\right\|^2 \quad \& \quad L_{\sigma^2}=\left\|f_2(X_k) \right\|^2" />

因为它们分别代表了均值<img src="https://latex.codecogs.com/svg.image?\mu_k"/> 和方差的对数 <img src="https://latex.codecogs.com/svg.image?log\sigma^2"/>的loss，达到<img src="https://latex.codecogs.com/svg.image?N(0,I)"/> 就是希望二者尽量接近于 0 了。不过，这又会面临着**这两个损失的比例**要怎么选取的问题，选取得不好，生成的图像会比较模糊。




### 两个loss的比例分析

> KL 散度
>
> 设![img](https://imagerk.oss-cn-beijing.aliyuncs.com/img/2391b5c44d8e3659ca18641aa2c88c9a.svg) 是随机变量![img](https://bkimg.cdn.bcebos.com/formula/9f7d1d2e6f98698b18cf0939756901ac.svg)上的两个概率分布，则在离散和连续随机变量的情形下，相对熵的定义分别为
>
> 
>
> ![img](https://imagerk.oss-cn-beijing.aliyuncs.com/img/a15d7aa79192a3043c86ac654b88fb5f.svg)
>
>  

原论文直接算了一般（各分量独立的）正态分布与标准正态分布的 KL 散度$KL(N(μ,σ^2)‖N(0,I))$作为这个额外的 loss，计算结果为：

<img src="https://latex.codecogs.com/svg.image?L_{\mu.\rho^2}=\frac{1}{2}\sum_{i=1}^{d}(\mu^2_{(i)}&plus;\sigma^2_{(i)}-log\sigma^2_{(i)}-1)" title="L_{\mu.\rho^2}=\frac{1}{2}\sum_{i=1}^{d}(\mu^2_{(i)}+\sigma^2_{(i)}-log\sigma^2_{(i)}-1)" />

这里的 $d$ 是隐变量 Z 的维度，而 <img src="https://latex.codecogs.com/svg.image?\mu_{(i)}"/> 和  <img src="https://latex.codecogs.com/svg.image?\sigma_{(i)}^2"/> 分别代表一般正态分布的均值向量和方差向量的第 i 个分量。直接用这个式子做补充 loss，就不用考虑均值损失和方差损失的相对比例问题了。

显然，这个 loss 也可以分两部分理解：

![image-20220130134808726](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220130134808726.png)

### KL散度的计算分析

由于我们考虑的是各分量独立的多元正态分布，因此只需要推导一元正态分布的情形即可，根据定义我们可以写出：

<img src="https://latex.codecogs.com/svg.image?KL\left( N\left( \mu ,\sigma ^2 \right) ||N\left( 0,1 \right) \right) =\int{\frac{1}{\sqrt{2\pi \sigma ^2}}e^{-\frac{\left( x-\mu \right) ^2}{2\sigma ^2}}}\log \frac{\frac{1}{\sqrt{2\pi \sigma ^2}}e^{-\frac{\left( x-\mu \right) ^2}{2\sigma ^2}}}{\frac{e^{-\frac{x^2}{2}}}{\sqrt{2\pi}}}dx" />



![img](https://imagerk.oss-cn-beijing.aliyuncs.com/img/449d5b8c1ad54a548d916b5260cd98da.jpeg)

整个结果分为三项积分，第一项实际上就是 <img src="https://latex.codecogs.com/svg.image?−log\sigma^2" />乘以概率密度的积分（也就是 1），所以结果是 <img src="https://latex.codecogs.com/svg.image?−log\sigma^2" />；第二项实际是正态分布的二阶矩，熟悉正态分布的朋友应该都清楚正态分布的二阶矩为 <img src="https://latex.codecogs.com/svg.image?\mu^2+\sigma^2" />；而根据定义，第三项实际上就是“方差除以方差=1”。所以总结果就是：

<img src="https://latex.codecogs.com/svg.image?KL\left( N\left( \mu ,\sigma ^2 \right) ||N\left( 0,1 \right) \right) =\frac{1}{2}(-log\sigma^2+\mu^2+\sigma^2-1)" />

### 损失函数的实现

```python
def loss_function(recon_x,x,mu,logvar):
    BCE_loss=nn.BCELoss(reduction='sum')
    reconstruction_loss=BCE_loss(recon_x,x)
    KL_divergence=-0.5*torch.sum(1+logvar-torch.exp(logvar)-mu**2)
    print(reconstruction_loss, KL_divergence)
    return reconstruction_loss + KL_divergence
```

## 重参数技巧

最后是实现模型的一个技巧，英文名是 Reparameterization Trick，我这里叫它做重参数吧。

<img src="http://5b0988e595225.cdn.sohucs.com/images/20180323/11a4f3c7717e40b2b2939f57a49bc791.png" alt="img" style="zoom: 80%;" />

**▲** 重参数技巧

其实很简单，就是我们要从 p(Z|Xk) 中采样一个 Z<sub>k</sub> 出来，尽管我们知道了 p(Z|Xk) 是正态分布，但是均值方差都是靠模型算出来的，我们要靠这个过程反过来优化均值方差的模型，但是“采样”这个操作是不可导的，而采样的结果是可导的，于是我们利用了一个事实：

![img](http://5b0988e595225.cdn.sohucs.com/images/20180323/be4121f0a35f4eba981568ff185049b7.png)

所以，我们将从 N(μ,σ^2) 采样变成了从 N(0,1) 中采样，然后通过参数变换得到从 N(μ,σ^2) 中采样的结果。**这样一来，“采样”这个操作就不用参与梯度下降了，改为采样的结果Z参与，使得整个模型可训练了。**





---



VAE 的示意图：

<img src="http://5b0988e595225.cdn.sohucs.com/images/20180323/ae18d3f17d614ba1b3eb2ed10570c5f4.jpeg" alt="img" style="zoom:67%;" />

以上就是VAE的训练目的：

我们希望重构$X$，也就是最小化 $D(X_k,\hat{X}_k)^2$，但是这个重构过程受到噪声的影响，因为 $Z_k$ 是通过重新采样过的，不是直接由 encoder 算出来的。

 Auto-Encoding Variational Bayes 中公式9

<img src="http://latex.codecogs.com/gif.latex?log q_{\Phi}(z|x^{(i)})=log \mathbb{N}(z;\mu^{i},\sigma^{2(i)}I)" />

___

## 重构误差

用于评估生成效果的，整个网络的loss为KL loss+Reconstruction Loss

## 后记

<font color='red'>VAE的 Encoder 有两个，一个用来计算均值，一个用来计算方差</font>：Encoder 不是用来 Encode 的，是用来算均值和方差的。

事实上，我觉得 **VAE 从让普通人望而生畏的变分和贝叶斯理论出发，最后落地到一个具体的模型中**，虽然走了比较长的一段路，但最终的模型其实是很接地气的。

1. **它本质上就是在我们常规的自编码器的基础上，对 encoder 的结果（在VAE中对应着计算均值的网络）加上了“高斯噪声”，使得结果 decoder 能够对噪声有鲁棒性**；而那个额外的 KL loss（目的是让均值为 0，方差为 1），事实上就是相当于对 encoder 的一个正则项，希望 encoder 出来的东西均有零均值。
2. 计算方差的encoder（对应着**计算方差**的网络）：用来**动态调节噪声的强度**的。

训练时，loss的变化如下:

1. **当 decoder 还没有训练好时（重构误差远大于 KL loss），就会适当降低噪声（KL loss 增加），使得拟合起来容易一些（重构误差开始下降）**。

2. **如果 decoder 训练得还不错时（重构误差< KL loss），这时候噪声就会增加（KL loss 减少），使得拟合更加困难了（重构误差又开始增加），这时候 decoder 就要想办法提高它的生成能力了**。

<img src="http://5b0988e595225.cdn.sohucs.com/images/20180323/b65176b6c8314465b3639222933dd6e4.jpeg" alt="img" style="zoom:80%;" />

**▲** VAE的本质结构

说白了，<font color='red'>**重构的过程是希望没噪声的，而 KL loss 则希望有高斯噪声的，两者是对立的。所以，VAE 跟 GAN 一样，内部其实是包含了一个对抗的过程，只不过它们两者是混合起来，共同进化的**。</font>

VAE：造假者和鉴别者共同进化。

GAN：真正高明的地方是：**它连度量都直接训练出来了**，而且这个度量往往比我们人工想的要好（然而 GAN 本身也有各种问题，这就不展开了）。

**正态分布？**

对于 p(Z|X) 的分布，读者可能会有疑惑：是不是必须选择正态分布？可以选择均匀分布吗？

首先，这个本身是一个实验问题，两种分布都试一下就知道了。但是从直觉上来讲，正态分布要比均匀分布更加合理，因为正态分布有两组独立的参数：均值和方差，而均匀分布只有一组。

前面我们说，**在 VAE 中，重构跟噪声是相互对抗的，重构误差跟噪声强度是两个相互对抗的指标，而在改变噪声强度时原则上需要有保持均值不变的能力，不然我们很难确定重构误差增大了，究竟是均值变化了（encoder的锅）还是方差变大了（噪声的锅）**。

而均匀分布不能做到保持均值不变的情况下改变方差，所以正态分布应该更加合理。

![image-20220201154403879](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220201154403879.png)

## 变分在哪里

还有一个有意思（但不大重要）的问题是：VAE 叫做“变分自编码器”，它跟变分法有什么联系？在VAE 的论文和相关解读中，好像也没看到变分法的存在？

其实如果读者已经承认了 KL 散度的话，那 VAE 好像真的跟变分没多大关系了，因为 KL 散度的定义是：

![img](http://5b0988e595225.cdn.sohucs.com/images/20180323/9662d78e3e6340a2b9f802ba115fb823.png)

如果是离散概率分布就要写成求和，我们要证明：**已概率分布 p(x)（或固定q(x)）的情况下，对于任意的概率分布 q(x)（或 p(x)），都有 KLp(x)‖q(x))≥0，而且只有当p(x)=q(x)时才等于零**。

因为 KL(p(x)‖q(x))实际上是一个泛函，要对泛函求极值就要用到变分法，当然，这里的变分法只是普通微积分的平行推广，还没涉及到真正复杂的变分法。而 VAE 的变分下界，是直接基于 KL 散度就得到的。所以直接承认了 KL 散度的话，就没有变分的什么事了。

一句话，VAE 的名字中“变分”，是因为它的推导过程用到了 KL 散度及其性质。

## **条件VAE**

最后，因为目前的 VAE 是无监督训练的，因此很自然想到：如果有标签数据，那么能不能把标签信息加进去辅助生成样本呢？

这个问题的意图，往往是希望能够实现控制某个变量来实现生成某一类图像。当然，这是肯定可以的，我们把这种情况叫做 **Conditional VAE**，或者叫 CVAE（相应地，在 GAN 中我们也有个 CGAN）。

但是，CVAE 不是一个特定的模型，而是一类模型，总之就是把标签信息融入到 VAE 中的方式有很多，目的也不一样。这里基于前面的讨论，给出一种非常简单的 VAE。

![img](http://5b0988e595225.cdn.sohucs.com/images/20180323/52dfbb7fb9a24e89be4d7f9c4c4885c5.jpeg)

**▲** 一个简单的CVAE结构

在前面的讨论中，我们希望 X 经过编码后，Z 的分布都具有零均值和单位方差，这个“希望”是通过加入了 KL loss 来实现的。

如果现在多了类别信息 Y，**我们可以希望同一个类的样本都有一个专属的均值 μ^Y（方差不变，还是单位方差），这个 μ^Y 让模型自己训练出来**。

这样的话，有多少个类就有多少个正态分布，而在生成的时候，我们就可以**通过控制均值来控制生成图像的类别**。

事实上，这样可能也是在 VAE 的基础上加入最少的代码来实现 CVAE 的方案了，因为这个“新希望”也只需通过修改 KL loss 实现：

![img](http://5b0988e595225.cdn.sohucs.com/images/20180323/772c9d72b0a54548b17fde7e98b8fb4e.png)

下图显示这个简单的 CVAE 是有一定的效果的，不过因为 encoder 和 decoder 都比较简单（纯 MLP），所以控制生成的效果不尽完美。

![img](http://5b0988e595225.cdn.sohucs.com/images/20180323/e7b870ea4ba54b93a1c2899016058c5d.jpeg)

用这个 CVAE 控制生成数字 9，可以发现生成了多种样式的 9，并且慢慢向 7 过渡，所以初步观察这种 CVAE 是有效的。

更完备的 CVAE 请读者自行学习了，最近还出来了 CVAE 与 GAN 结合的工作 CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training，模型套路千变万化。

代码

我把 Keras 官方的 VAE 代码复制了一份，然后微调并根据前文内容添加了中文注释，也把最后说到的简单的 CVAE 实现了一下，供读者参考。

代码：https://github.com/bojone/vae

终点站

磕磕碰碰，又到了文章的终点了。不知道讲清楚了没，希望大家多提点意见。

总的来说，VAE 的思路还是很漂亮的。倒不是说它提供了一个多么好的生成模型（因为事实上它生成的图像并不算好，偏模糊），而是它提供了一个将概率图跟深度学习结合起来的一个非常棒的案例，这个案例有诸多值得思考回味的地方。

