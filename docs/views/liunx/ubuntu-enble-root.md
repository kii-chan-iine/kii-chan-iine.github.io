---
author: 星火燎原@vxhly
title: Ubuntu 开启 root 用户
categories: [liunx]
tags: [Liunx]
date: 2016-08-02 15:29:46
---

<Boxx changeTime="10000"/>

::: tip 前言
Ubuntu 默认不开启 root 账户的, 所以有需开启 root 账户的按照以下做法即可, 切换 root 账户后 chromium 浏览器不能使用。
:::
<!-- more -->

## 打开终端开启 root 账户

``` bash
sudo passwd -u root
```

## 为 root 账户设置密码

输入两次

``` bash
sudo passwd root
```

### 测试 root 账户

输入 root 密码, 进入 root 终端

``` bash
su -
```

## 设置登录面板, 使其实现 root 登录

使用 `vim`

``` bash
sudo vim /usr/share/lightdm/lightdm.conf.d/50-unity-greeter.conf
```

使用 `gedit`

``` bash
sudo gedit /usr/share/lightdm/lightdm.conf.d/50-unity-greeter.conf
```

添加如下代码, 保存退出

``` bash
user-session=ubuntu
greeter-show-manual-login=true
all-guest=false
```

## 重启 ubuntu, 需切换 root 账户

``` bash
su -
reboot
```

## 删除原先账户

如果你不喜欢原先的账户, 大可以彻底删除用户

### 在 root 用户下

``` bash
userdel -r userName
```

### 在普通用户下

``` bash
sudo userdel -r userName
```

::: warning 注意
因为需要彻底删除用户, 所以加上-r 的选项, 在删除用户的同时一起把这个用户的宿主目录和邮件目录删除；userName 需替换成你的原先账户名
:::
