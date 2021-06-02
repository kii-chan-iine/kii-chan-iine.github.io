---
author: 星火燎原@vxhly
title: 常见的端口号以及其对应的应用程序或服务
categories: [networking]
tags: [Network, Security , Windows, Liunx, Protocol, Pentest]
date: 2016-08-16 16:04:29
---

<Boxx changeTime="10000"/>

::: tip 前言
列出 TCP/UDP 协议常见的端口号占用情况, 以助于分析该服务器开放了哪些端口号, 或者程序占用了哪些端口号。
:::
<!-- more -->

## 如何查看

### Windows 系统

通过 `netstat` 列出所有端口的使用情况, `For Example:`

``` bash
netstat -ano
```

通过 `findstr` 查询包含关键字的端口使用情况, `For Example:`

``` bash
netsata -ano | findstr ":80"
```

通过 `tasklist` 查询哪个程序或应用占用该端口对应的 `PID` , `For Example:`

``` bash
tasklist | findstr "2180"
```

### Liunx 系统

通过 `netstat` 列出监听的端口, `For Example:`

``` bash
netstat -tlun
```

通过 `netstat` 列出所有的网络连接, `For Example:`

``` bash
netstat -an
```

通过 `ps` 查看进程占用状态, `For Example:`

``` bash
ps -ef | grep sshd
```

## 常见端口号

### TCP 协议相关

* `20` => FTP（文件传输协议） 传输数据端口, 用于纯属数据
* `21` => FTP（文件传输协议） 连接端口, 用于登陆认证
* `22` => SSH（安全外壳协议） 加密的远程连接, SFTP（安全的文件传输协议） 传输安全的数据
* `23` => Telnet（远程终端协议） 相对不安全的远程连接
* `25` => SMTP（简单邮件传输协议） 使用的端口
* `53` => DNS 服务器开放的端口, 入侵者可能是试图进行区域传递（TCP）, 所以防火墙常常过滤或记录该端口
* `80` / `8080` / `3128` / `8081` / `9080` => HTTP（超文本传输协议） 使用的端口
* `109` => POP2（邮件协议 2） 服务器开放的端口
* `110` => POP3（邮件协议 3） 服务器开放的端口, 用于接收邮件, 客户端访问服务器的邮件服务
* `139` => 文件共享使用的端口
* `443` => HTTPS（安全套接字层的超文本传输协议） 默认端口, 传输数据更加安全
* `445` => SMB（文件共享服务）开放的端口
* `1433` => SQL Server 默认的端口, 用于供 SQL Server 对外提供服务
* `1521` => Oracle 数据库默认端口
* `3306` => MySQL 数据库默认端口
* `3389` => 远程桌面使用的端口
* `8080` => Tomcat 服务器的默认端口

### UDP 协议相关

* `53` => DNS 服务器开放的端口, 入侵者可能是试图进行欺骗 DNS（UDP）或隐藏其他的通信, 所以防火墙常常过滤或记录该端口
* `69` => TFTP（简单文件传输协议） 默认的端口号
* `139` => 文件共享使用的端口
* `161` => SNMP（简单网络管理协议） 入侵者常探测的端口
* `162` => SNMP（简单网络管理协议） 默认的端口号
* `445` => SMB（文件共享服务）开放的端口
* `1434` => SQL Server 默认的端口, 于向请求者返回 SQL Server 使用了哪个 TCP/IP 端口
