---
author: 星火燎原@vxhly
title: Liunx 学习笔记（七）文件的特殊权限
categories: [liunx]
tags: [Liunx, Security]
date: 2016-10-14 19:25:27
---

<Boxx changeTime="10000"/>

::: tip 前言
本篇是 Liunx 的学习笔记系列第七篇, 在 Liunx 中文件除了 rwx 权限, 还存在 SetUID 、 SetGID 、 Sicky BIT 特殊权限, 本篇就是来介绍这些特殊权限。
:::
<!-- more -->

## SetUID

### 针对文件的作用

* 只有可以执行的二进制程序才能设定 SUID 权限
* 命令执行者要对该程序用于 x（执行）权限
* 命令执行者在执行该程序是获得该程序文件属主的身份（在执行程序的过程中灵魂附体成文件的属主）
* SetUID 权限只在该程序执行过程中有效, 也就是说身份改变只在程序执行过程中有效

### 概念图

![SetUID](https://vxhly.github.io/assets/setuid-1.png)

### 设置 SUID

只能针对二进制可执行文件

#### 4 代表 SUID

 `Command Format:`

``` bash
chmod 4777 [fileName]     # 设置 SUID 权限
chmod 644 [fileName]      # 取消 SUID 权限
```

当然还可以用以下命令设置, `Command Format:`

``` bash
chmod u+s [fileName]      # 设置 SUID 权限
chmod u-s [fileName]      # 取消 SUID 权限
```

#### S 与 s 的区别

::: tip 区别
大写 `S` 其实是报错, 因为我们用来测试的文件并非二进制文件, 或者说是可执行程序, 只需添加 x 权限即可变为小写 `s` , 实际上生效的是小写 `s`
:::

`For Examlpe:` <br>

![SetUID](https://vxhly.github.io/assets/setuid-2.png)

### 危险的 SetUID

* 关键目录应严格控制写权限。比如 `“/”` 、 `”/usr“` 等
* 用户的密码设置要严格遵守密码三原则
* 对系统中默认应该具有 SetUID 权限的文件作一列表, 定时检查没有这个之外的文件是否被设置了 SetUID 权限

## SetGID

既可以针对二进制可执行文件, 也可以针对目录

### 针对文件的作用

* 只有可以执行的二进制程序才能设定 SGID 权限
* 命令执行者要对该程序用于 x（执行）权限
* 命令执行者在执行该程序的时候, 组身份升级为该程序的属组
* SetUID 权限同样只在该程序执行过程中有效, 也就是说组身份改变只在程序执行过成中有效

### 针对目录的作用

* 普通用户必须对此目录拥有 r 和 x 权限, 才能进入此目录
* 普通用户在此目录中的有效组会变成此目录的属组
* 若普通用户对此目录拥有 w 权限时, 新建的文件的默认属组是这个目录的属组

### 设置 SGID

#### 2 代表 SGID

 `Command Format:`

``` bash
chmod 2777 [fileName]      # 设置 SGID 权限
chmod 644 [fileName]       # 取消 SGID 权限
```

当然还可以用以下命令设置, `Command Format:`

``` bash
chmod g+s [fileName]       # 设置 SGID 权限
chmod g-s [fileName]      # 取消 SGID 权限
```

#### S 与 s 的区别

::: tip 区别
大写 S 其实是报错, 因为我们用来测试的文件并非二进制文件, 或者说是可执行程序, 只需添加 x 权限即可变为小写 s
:::

`For Examlpe:` <br>

![SetUID](https://vxhly.github.io/assets/setgid.png)

## Sicky BIT

只能针对目录

### 针对目录的作用

### 设置 SBIT

* 粘着位目前对目录有效
* 普通用户对该目录拥有 w 和 x 权限, 即普通用户可以在此目录拥有写入权限
* 如果没有粘着位, 因为普通用户拥有 w 权限, 所以可以删除此目录下所有文件, 包括其他用户建立的文件。一但赋予粘着位, 除了 root 可以删除所有文件, 普通用户就算拥有 w 权限, 也只能删除自己建立的文件, 但是不能删除其他用户建立的文件

#### 1 代表 SBIT

 `Command Format:`

``` bash
chmod 1777 [folderName]       # 设置 SBIT 权限
chmod 644 [folderName]        # 取消 SBIT 权限
```

当然还可以用以下命令设置, `Command Format:`

``` bash
chmod o+s [folderName]       # 设置 SBIT 权限
chmod o-t [folderName]       # 取消 SBIT 权限
```

#### 举例

`For Examlpe:` <br>

![SetUID](https://vxhly.github.io/assets/sticky-bit.png)
