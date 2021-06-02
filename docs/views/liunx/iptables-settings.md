---
author: 星火燎原@vxhly
title: RedHat 系统安全加固（五）防火墙 (iptables) 的设置
categories: [liunx]
tags: [Liunx, Security]
date: 2016-10-27 10:00:55
---

<Boxx changeTime="10000"/>

::: tip 前言
本系列笔记又是 Liunx 学习系列教程的一大步, 本系列学习笔记记录 RedHat 系统的安全加固。本篇笔记记录 RedHat 下的防火请的开启和关闭, 还有允许哪些端口通过防火墙。
:::
<!-- more -->

## 防火墙的开启和关闭

### 永久性生效

使用 `chkconfig` , 重启后不会复原

``` bash
chkconfig iptables on         # 开启
chkconfig iptables off        # 关闭
```

### 即时生效

使用 `service` , 重启后复原

``` bash
service iptables start        # 开启
service iptables stop         # 关闭
```

## iptables 命令解释

### 选项解释

* **-t\<表>**: 指定要操纵的表；
* **-A**: 向规则链中添加条目；
* **-D**: 从规则链中删除条目；
* **-I**: 向规则链中插入条目；
* **-R**: 替换规则链中的条目；
* **-L**: 显示规则链中已有的条目；
* **-F**: 清楚规则链中已有的条目；
* **-Z**: 清空规则链中的数据包计算器和字节计数器；
* **-N**: 创建新的用户自定义规则链；
* **-P**: 定义规则链中的默认目标；
* **-h**: 显示帮助信息；
* **-p**: 指定要匹配的数据包协议类型；
* **-s**: 指定要匹配的数据包源 ip 地址；
* **-j\<目标>**: 指定要跳转的目标；
* **-i\<网络接口>**: 指定数据包进入本机的网络接口；
* **-o\<网络接口>**: 指定数据包要离开本机所使用的网络接口。

### 命令输入顺序

``` bash
iptables -t 表名 <-A/I/D/R> 规则链名 [规则号] <-i/o 网卡名> -p 协议名 <-s 源IP/源子网> --sport 源端口 <-d 目标IP/目标子网> --dport 目标端口 -j 动作
```

* **表名包括**

-- **raw**: 高级功能, 如: 网址过滤。
-- **mangle**: 数据包修改（QOS）, 用于实现服务质量。
-- **net**: 地址转换, 用于网关路由器。
-- **filter**: 包过滤, 用于防火墙规则。

* **规则链名包括**

-- **INPUT 链**: 处理输入数据包。
-- **OUTPUT 链**: 处理输出数据包。
-- **PORWARD 链**: 处理转发数据包。
-- **PREROUTING 链**: 用于目标地址转换（DNAT）。
-- **POSTOUTING 链**: 用于源地址转换（SNAT）。

* **动作包括**

-- **ACCEPT**: 接收数据包。
-- **DROP**: 丢弃数据包。
-- **REDIRECT**: 重定向、映射、透明代理。
-- **SNAT**: 源地址转换。
-- **DNAT**: 目标地址转换。
-- **MASQUERADE**: IP 伪装（NAT）, 用于 ADSL。
-- **LOG**: 日志记录。

## iptables 的设置与查看

### 查看 iptables 的设置

``` bash
iptables -nL
service iptables status
```

![iptables 的设置与查看](https://vxhly.github.io/assets/iptables-1.png)

### 清除已有 iptables 规则

``` bash
iptables -F
iptables -X
iptables -Z
```

![iptables 的设置与查看](https://vxhly.github.io/assets/iptables-2.png)

### 开放指定的端口

``` bash
iptables -A INPUT -s 127.0.0.1 -d 127.0.0.1 -j ACCEPT
# 允许本地回环接口(即运行本机访问本机)

iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
# 允许已建立的或相关连的通行

iptables -A OUTPUT -j ACCEPT
# 允许所有本机向外的访问

iptables -A INPUT -p tcp --dport 22 -j ACCEPT
# 允许访问 22 端口

iptables -A INPUT -p tcp --dport 80 -j ACCEPT
# 允许访问 80 端口

iptables -A INPUT -p tcp --dport 21 -j ACCEPT
# 允许 FTP  服务的 21 端口

iptables -A INPUT -p tcp --dport 20 -j ACCEPT
# 允许 FTP 服务的 20 端口

iptables -A INPUT -j reject
# 禁止其他未允许的规则访问

iptables -A FORWARD -j REJECT
# 禁止其他未允许的规则访问
```

::: tip 提示
`-A` 是在 `iptables` 的最后添加一条规则, `-I` 是在 `iptables` 的前面添加一条规则
:::

![iptables 的设置与查看](https://vxhly.github.io/assets/iptables-3.png)

### 屏蔽 IP

``` bash
iptables -I INPUT -s 123.45.6.7 -j DROP
# 屏蔽单个 IP 的命令

iptables -I INPUT -s 123.0.0.0/8 -j DROP
# 封整个段即从 123.0.0.1 到 123.255.255.254 的命令

iptables -I INPUT -s 124.45.0.0/16 -j DROP
# 封 IP 段即从 123.45.0.1 到 123.45.255.254 的命令

iptables -I INPUT -s 123.45.6.0/24 -j DROP
# 封 IP 段即从 123.45.6.1 到 123.45.6.254 的命令
```

### 保存当前设置的规则

``` bash
/etc/rc.d/init.d/iptables save
```

![iptables 的设置与查看](https://vxhly.github.io/assets/iptables-4.png)

<br>

![iptables 的设置与查看](https://vxhly.github.io/assets/iptables-5.png)
