---
author: 星火燎原@vxhly
title: RedHat 系统安全加固（三）系统日志审计
categories: [liunx]
tags: [Liunx, Security, Pentest]
date: 2016-10-16 13:39:09
---

<Boxx changeTime="10000"/>

::: tip 前言
本系列笔记又是 Liunx 学习系列教程的一大步, 本系列学习笔记记录 RedHat 系统的安全加固。本篇笔记是记录系统日志审计的配置, 加强系统的防御能力。
:::
<!-- more -->

## syslog 登录事件记录

查看参数 `authpriv` 值, `For Exmaple:`

``` bash
more /etc/syslog.conf | grep authpriv
```

表名登录事件记录在 `/var/log/secure` 文件中, 使用以下命令检查非法记录

``` bash
more /var/log/secure | grep refused
```

![RedHat 安全加固](https://vxhly.github.io/assets/redhat-reinforce-16.png)

## 审计功能的进程

启动审计进程, `For Exmaple:`

``` bash
service auditd start
service psacct start
```

启动 `psacct` 服务

``` bash
service psacct start
chkconfig psacct on
```

## 查看记录审计的日志

* `auditd` 审计进程日志记录在 `/var/log/audit/audit.log`
* `psacct` 日志记录在 `/var/account/pacct`
::: warning 注意
`pacct` 日志是二进制文件, 使用以下命令进行查看
:::

 `For Exmaple:`

``` bash
lastcomm -f /var/account/pacct
```

## 登录日志查看

`lastlog` 查看用户最后一次登录时间, `For Exmaple:`

``` bash
lastlog
```

![RedHat 安全加固](https://vxhly.github.io/assets/redhat-reinforce-17.png)
