---
author: kii
title: Yolo
categories: [CV]
tags: [CV,DL]
date: 2022-02-02 10:44:30
---

<Boxx changeTime="10000"/>

::: tip 前言

YOLO V1-V5

原论文 Auto-Encoding Variational Bayes。

:::
<!-- more -->



https://blog.csdn.net/weixin_38842821/article/details/108544609



# VAE
## 思路
1. 假设对于每个样本$X_k$存在一个对应的后验概率$ p(Z|X)$是正态分布(为什么可以这样呢？EC：可以这样认为$X_k$是一个图片，图片中的像素点是一个服从正态的样本)

