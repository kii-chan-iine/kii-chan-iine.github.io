---
author: 星火燎原@vxhly
title: Liunx 命令练习
categories: [liunx]
tags: [Liunx, Security, Command]
date: 2017-09-03 20:46:08
---

<Boxx changeTime="10000"/>

::: tip 前言
简单的一些 Liunx 命令练习。
:::
<!-- more -->

## 【文件的压缩和解压缩】

::: tip 需求
使用 tar 命令将 /etc 目录打包至 /tmp 目录下（仅打包, 不压缩）, 将打包后的文件, 以 gzip 方式压缩【共两条命令】
:::

``` bash
tar -cvf /tmp/etc.tar /etc
gzip /tmp/etc.tar
```

## 【创建软链接和硬链接】

::: tip 需求
给系统文件 /etc/profile 创建一个软链接 lnsprofile 和一个硬链接 lnhprofile【共两条命令】
:::

``` bash
ln -s /etc/profile ./lnsprofile
ln /etc/profile ./lnhprofile
```

## 【为文件设置隐藏属性】

::: tip 需求
为文件 test 设置隐藏属性, 要求其只能无法写入数据, 也无法删除它【共一条命令】
:::

``` bash
chattr +a test
# or
chattr +i test
```

## 【创建目录和删除目录】

::: tip 需求
创建目录 ./test/2017/6/25, 并切换目录至 ./test/2017/6, 删除空目录 ./25（不能使用 rm -rf）【共三条命令】
:::

``` bash
mkdir -p ./test/2017/6/25
cd ./test/2017/6
rmdir ./25
```

::: tip 需求
查看当前所在的目录【共一条命令】
:::

``` bash
pwd
```

::: tip 需求
查看当前目录下有哪些文件或目录（包括隐藏文件）【共一条命令】
:::

``` bash
ls -la
```

## 【普通用户提权】

::: tip 需求
普通用户切换成 root 超级用户【举例两个】
:::

``` bash
sudo su
# or
su -
# or
sudo -s
```

## 【创建用户和用户组】

::: tip 需求
创建一个用户 stuxxx, 其家目录为 /home/stu, 备注信息为 "this is student, ID is xxxxxx", 登录 shell 为 /sbin/nologin, 并设置登陆密码【共三条命令】
:::

``` bash
useradd -d /home/stu -m -c "this is student,ID is xxxxxx" -s /sbin/nologin -p "" stuxxx
```

::: tip 需求
创建一个 uid 为 0 的用户 student【共一条命令】
:::

``` bash
useradd  -o  -u  0  student
```

::: tip 需求
创建用户组 test【共一条命令】
:::

``` bash
groupadd test
```

## 【用户管理】

::: tip 需求
锁定用户 stuxxx, 使其不能登陆系统【共一条命令】
:::

``` bash
usermod -s /sbin/nologin stu2xxx
# or
passwd -l stu2xxx
```

## 【新建文件】

::: tip 需求
新建一个文件 file, 添加内容 "this is text, ID is xxxxxx", 设置其权限为 000【共三条命令】
:::

``` bash
touch file
echo "this is text,ID is xxxxxx" > file
chmod 000 file
```

## 【文件的快速搜索】

::: tip 需求
在 /etc 目录下搜索 apache、ssh、ftp 的配置文件【共三条命令】
:::

``` bash
find /etc -type f -iname '*apache2.*'
find /etc -type f -iname '*ssh.*'
find /etc -type f -iname '*ftp.*'
```

## 【文件的查看和快速定位】

::: tip 需求
查看 /etc/passwd 文件和 /etc/shadow 文件, 并使用 grep 过滤出 student 用户的信息【共两条命令】
:::

``` bash
cat /etc/passwd | grep 'student'
cat /etc/shadow | grep 'student'
```

## 【更改目录或者文件的属性】

::: tip 需求
将目录 ./test/* 的所有者和所有组修改成 student【共一条命令】
:::

``` bash
chown -R student:student ./test/
```

## 【系统环境变量的查看与配置】

::: tip 需求
查看当前的系统环境变量配置【共两条命令】
:::

``` bash
env
# or
echo $PATH
```

::: tip 需求
将 /tmp 目录添加至系统 $PATH 变量中【共一条命令】
:::

``` bash
PATH=$PATH:/tmp
```

## 【iptables 规则】

::: tip 需求
设置 iptables 规则, 只允许 80 端口和 22 端口通过防火墙【共两条命令】
:::

``` bash
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
```
