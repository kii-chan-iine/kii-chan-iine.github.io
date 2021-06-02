---
author: 星火燎原@vxhly
title: Liunx 服务配置笔记（六）DNS 服务安全配置
categories: [liunx]
tags: [Liunx, Service, Security, Protocol]
date: 2016-09-27 13:25:48
---

<Boxx changeTime="10000"/>

::: tip 前言
这篇是 Liunx 的服务配置笔记系列第五篇, 这一篇将具体介绍 Red Hat Liunx 下 DNS 服务的安全配置。所谓 DNS 就是一个域名解析服务器, 其中 DNS 还包含正向解析和反向解析。
:::
<!-- more -->

## 服务的安装

### 创建挂载目录

 `For Example:`

``` bash
mkdir -p /mnt/cdrom
```

### 挂在光盘

 `For Example:`

``` bash
mount /dev/cdrom /mnt/cdrom
```

### 安装

 `For Example:`

``` bash
cd /mnt/cdrom/Server
rpm -ivh bind-9.3.6-4.P1.el5_4.2.i386.rpm
rpm -ivh caching-namedserver-9.3.6-4.P1.el5_4.2.i386.rpm
```

![DNS](https://vxhly.github.io/assets/liunx-dns-1.png)

### 复制模板配置文件

 `For Example:`

``` bash
cd /etc/
cp -p named.caching-nameserver.conf named.conf
```

![DNS](https://vxhly.github.io/assets/liunx-dns-2.png)

复制的时候, 建议带上 `-p` 选项, 否则不能使用该文件, 如果忘记加上 `-p` 选项, 不要紧, 使用以下命令还可以挽救。

``` bash
chown named:named named.conf
chown named:named named.zones
```

## 安全配置

### 限制递归查询功能

默认没有这个功能, 需要手动添加。递归查询指 一般客户机和服务器之间属 递归查询 , 即当客户机向 DNS 服务器发 出请求后, 若 DNS 服务器本身不能解析, 则会向另外的 DNS 服务 器发出查询请求, 得到结果后转交给客户机。<br>
这个选项, 即可填 IP 地址, 也可以填网络段, 当然也可以填 any（所有）。

修改主配置文件

``` bash
allow-recursion { 192.168.199.0/24;};  # 限制网段查询
allow-recursion { 192.168.199.2;};     # 限制 IP 查询
allow-recursion { any;};               # 所有的都可以查询
```

![DNS](https://vxhly.github.io/assets/liunx-dns-3.png)

### 限制查询

 `Default Configuration:`

``` bash
allow-query { localhost;};
```

allow-query 配置项用来设置允许 DNS 查询的客户端地址, 默认的客户端地址为本机。这里可以设置为某个网段、所有、具体的某台主机三种情况。

修改主配置文件

``` bash
allow-query { 192.168.199.0/24;};  # 限制网段查询
allow-query { 192.168.199.2;};     # 限制 IP 查询
allow-query { any;};               # 所有的都可以查询
```

![DNS](https://vxhly.github.io/assets/liunx-dns-4.png)

### 限制区传送

默认没有这个功能, 需要手动添加。如果没有限制区传送, 那么 DNS 服务器允许对任何人都进行区域传输, 因此网络架构中的主机名、主机 IP 列表、路由器名和路由 IP 列表, 甚至包括各主机所在的位置和硬件配置等情况都很容易被入侵者得到在 DNS 配置文件中通过设置来限制允许区传送的主机, 从一定程度上能减轻信息泄漏。

修改主配置文件

``` bash
allow-transfer{ 192.168.199.0/24;};   # 限制网段查询
allow-transfer { 192.168.199.2;};     # 限制 IP 查询
allow-transfer { any;};               # 所有的都可以查询
```

![DNS](https://vxhly.github.io/assets/liunx-dns-5.png)
