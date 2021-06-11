---
author: kii
title: 实验札记
categories: [Exp]
tags: [Exp]
date: 2021-06-04 12:29:30
---

<Boxx changeTime="10000"/>

::: tip 前言
实验的一些东西。
:::
<!-- more -->

# NO实验



## NO的制备

### Preparation of NO in PBS（Three-dimensional cell-adhesive matrix of silk cocoon derived carbon fiber assembled with iron-porphyrin for monitoring cell released signal molecules）

[NO](https://www.sciencedirect.com/topics/engineering/nitric-oxide) was prepared according to the procedures reported with slight modification. In details, 2.0 M H2SO4 was released drop by drop from a funnel into a glass flask containing saturated NaNO2 solution under continuous stirring to ensure a homogenous reaction. The produced NO was passed sequentially through saturated, 10.0 wt % and 2.5 wt % KOH solution to remove any other nitrogen oxide impurities. Finally, NO saturated solution in 0.01 M PBS was obtained with a concentration of 1.8 mM, which was stored in a nitrogen-protected environment at 4 °C for further use.

2M浓硫酸，一滴一滴的滴入盛有饱和$NaNO_2$的烧瓶中，产出的NO通过10.0 wt % 和 2.5 wt % KOH溶液取出杂质，之后通入0.01M的PBS中，获得NO饱和溶液（<font color='orange'>1.8mM</font>）



## 测量方案

### 方案1

​	沉积前， 电探头在 PBS（放在细胞浴皿中）中用循环伏安法（CV 法）在 0-0.9V以扫速 50mV/s 扫描 2 圈， 作为沉积前数据对比。

​	电探头尖端在含有 1mM 的 CTAB 溶液（ 细胞浴皿中） 中用循环伏安法在-1.8~1.8V 以扫速 50mV/s 扫描 10 圈进行电沉积， 沉积结束尖端再蘸一下 0.5%的Nafion 溶液， 自然干燥以供使用。  

​	沉积后， 电探头在 PBS 中用循环伏安法（CV 法） 在 0-0.9V 以扫速 50mV/s扫描 2 圈， 作为沉积后数据与沉积前数据对比， 如果有明显电容增加， 证明沉积较好。  

​	选择电学检测软件中的电流曲线法（Amperomentric i-t curve） 进行测试， 设置初始电位为 0.85V， 运行时间 2000S，刺激细胞生成 NO 进行检测。  


### 方案2

​	-1.6V~1.6V，100mV/s，10个循环，2mMCTAB<font color='red'>（+0.04% nafion）</font>

​	测量：0.75V初始电压。

