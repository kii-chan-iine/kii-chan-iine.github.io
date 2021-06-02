---
author: 星火燎原@vxhly
title: Liunx 学习笔记（六）用户的管理
categories: [liunx]
tags: [Liunx, Security]
date: 2016-10-07 11:31:32
---

<Boxx changeTime="10000"/>

::: tip 前言
本篇是 Liunx 的学习笔记系列第五篇, 这一篇将介绍 Liunx 下用户的管理。用户管理看似很简单, 但是里面还隐藏着大学问呢。
:::
<!-- more -->

## 添加用户, 并创建用户宿主目录

 `For Example:`

``` bash
useradd -m [userName]
```

## 删除用户, 并删除用户宿主目录

 `For Example:`

``` bash
userdel -r [userName]
```

## 添加 uid 为 0 的用户

 `For Example:`

``` bash
useradd -u 0 -o [userName]
```

## 查看用户的 uid 或 gid 等信息

 `For Example:`

``` bash
id [userName]
```

## 为用户添加密码

 `For Example:`

``` bash
passwd [userName]                                   # 普通方法
echo "passwd string" | passwd --stdin [userName]    # 输出文本内容作为密码
```

普通用户修改密码命令:  passwd

## 查看用户的密码状态

 `For Example:`

``` bash
passwd -S [userName]                      # 普通方法
cat /etc/passwd | grep [userName]         # 查看配置文件
```

## 禁用和解锁用户

### passwd 方法

 `For Example:`

``` bash
passwd -l [userName]    # 禁用
passwd -u [userName]    # 解锁
```

## 修改 /etc/shadow 文件（root 用户）

 `For Example:`

``` bash
vim /etc/shadow
```

禁用则在密码位前手工添加 `!` （启用删除 `!` 即可）

``` bash
test:!$1$afiVSYfG$utdc2qWLNrqNcy63Ca8301:16998:0:99999:7:::     # 只加一个 ！
test:!!$1$afiVSYfG$utdc2qWLNrqNcy63Ca8301:16998:0:99999:7:::    # 加两个,效果一致
```

使用 `:wq!` 强制保存, 因为其权限为 `000`

### usermod 方法

 `For Example:`

``` bash
usermod -L [userName]    # 禁用
usermod -U [userName]    # 解锁
```

## 设置用户一登录就要修改密码

 `For Example:`

``` bash
chage -d 0 [userName]
# 这个命令其实是把密码修改日期归 0 了 （/etc/shadow 文件第 3 个字段）
```

## 查看当前登录的用户是谁

 `For Example:`

``` bash
whoami
```

## 查看当前登录的用户的环境变量

 `For Example:`

``` bash
env
```

## 切换用户

 `For Example:`

``` bash
su - [userName]
```
