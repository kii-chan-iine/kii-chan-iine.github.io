---
author: 星火燎原@vxhly
title: Ubuntu 下连接 WIFI
categories: [networking]
tags: [Liunx, Network]
date: 2016-08-01 12:24:19
---

<Boxx changeTime="10000"/>

::: tip 前言
基于已经安装双系统, 或是只有 Ubuntu 系统, 在虚拟机下不存在此问题。
:::
<!-- more -->

## 查看无线网卡信息

``` bash
lspci | grep -i network
```

## 下载无线网卡驱动

本人的无线网卡为 `Realtek-RTL8723BE`
下载地址 [rtl8723be.tar.gz](http://oss-blog.test.upcdn.net/rtl8723be.tar.gz?attname=)

### 驱动安装(解压进入目录)

``` bash
make
sudo make install
sudo modprobe rtl8723be (rtl8723be 需替换成自己的网卡)
```

### 设置为开机启动项

``` bash
sudo modprobe -r rtl8723be (rtl8723be 需替换成自己的网卡)
sudo modprobe rtl8723be (rtl8723be 需替换成自己的网卡)
```

### 重启网卡

``` bash
ifconfig wlan0 down
ifconfig wlan0 up
```

### 重启网络服务

``` bash
service network restart
```
