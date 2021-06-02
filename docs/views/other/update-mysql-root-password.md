---
author: 星火燎原@vxhly
title: 更新 MySQL root 的密码
categories: [other]
tags: [Security, Liunx, Protocol, Pentest]
date: 2018-09-06 19:19:17
---

<Boxx changeTime="10000"/>

::: tip 前言
最近在 ubuntu 系统上安装了 MySQL, 但是安装时没有提示输入 root 用户密码, 网上胡乱找了一通, 这才知道 MySQL 5.7 以上版本的 root 密码都是随机生成的, 所以这里提供解决 MySQL 5.7 版本的 root 用户重置密码问题。
:::
<!-- more -->

## 安装 MySQL

``` bash
apt-get install mysql-cilent mysql-server 解决MySQL 5.7版本的root用户重置密码问题
```

安装之后可以使用下面的命令来检查是否安装成功

``` bash
sudo netstat -tap | grep mysql
netstat -tap | grep mysql
```

## 登录 MySQL

``` bash
mysql -u root -p
```

输入密码, 如果可以进入则下面的不用看了；如果提示不能登入, 则我们尝试启用安全模式登入MySQL, 这样可以绕过密码登入, 登入后再修改密码。

## 修改密码

### 修改配置文件

`my.cnf` => 不同系统的配置文件名以及存放路径不一致, 但一定能在 `/etc/mysql` 里找到

在 [mysqld] 小节下添加一行, 这一行配置让 mysqld 启动时不对密码进行验证

``` 
[mysqld]
skip - grant - tables = 1
```

修改完成之后保存重启 MySQL 服务

``` bash
/etc/init.d/mysql restart
```

### 无密码登录 MySQL 修改密码

``` bash
mysql -u root -p
```

执行以下操作

``` bash
mysql> use mysql;
 mysql> update user set authentication_string=PASSWORD("这里输入你要改的密码") where User='root';#更改密码
 mysql> update user set plugin="mysql_native_password";#如果没这一行可能也会报一个错误,因此需要运行这一行
 mysql> flush privileges;#更新所有操作权限
 mysql> quit
```

### 完成配置重启

删掉或者注释 `skip-grant-tables=1` 该行配置

修改完成之后保存重启 MySQL 服务

``` bash
/etc/init.d/mysql restart
```

之后使用设置的密码登录即可

``` bash
mysql -u root -p
```

## 附上完全卸载 MySQL 命令

``` bash
apt-get autoremove mysql-*
 dpkg -l |grep ^rc|awk '{print $2}' |sudo xargs dpkg -P
```
