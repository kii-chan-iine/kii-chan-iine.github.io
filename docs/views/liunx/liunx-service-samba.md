---
author: 星火燎原@vxhly
title: Liunx 服务配置笔记（五）SAMBA 服务安全配置
categories: [liunx]
tags: [Liunx, Service, Security, Protocol]
date: 2016-09-19 19:33:21
---

<Boxx changeTime="10000"/>

::: tip 前言
本篇是 Liunx 的服务配置笔记系列第五篇, 这一篇将具体介绍 Red Hat Liunx 下 SAMBA 服务的安全配置。所谓 SAMBA 就是建立 Liunx 和 Windows 之间的共享, 使 Windows 能够访问 Liunx 下共享出来的文件。
:::
<!-- more -->

## 配置文件的位置

主配置文件: **/etc/samba/smb.conf**

## 服务的启动与停止

::: tip 提示
SAMBA 的服务名并不叫做 `samba` , 而是 `smb` 。<br>
修改配置文件过后一定要重启 `smb` 服务。
:::

``` bash
service smb status        # 状态
service smb start         # 启动
service smb stop          # 暂停
service smb restart       # 重启
```

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
rpm -ivh perl-Convert-ASN1-0.20-1.1.noarch.rpm
rpm -ivh samba-3.0.33-3.28.el5.i386.rpm
```

![SAMBA 服务安全配置](https://vxhly.github.io/assets/liunx-samba-1.png)

## 安全配置

### 配置工作组和 NetBIOS 名字

 `Default Configuration:`

``` bash
workgroup = MYGROUP
;netbios name = MYSERVER
```

::: tip 提示
该文件默认配置的工作组是 MYGROUP, 而 Windows 下的工作组默认为 WORKGROUP, Liunx 和 Windows 的工作组一定要一样, 才能访问共享的文件。<br>
该文件默认将 NetBIOS 的名字注释掉了, NetBIOS 名字就是会在 Windows 下的网上邻居能够看到的名字, 注释掉就是会看不到, 建议去掉注释。
:::

![SAMBA 服务安全配置](https://vxhly.github.io/assets/liunx-samba-2.png)

### 配置安全级别

 `Default Configuration:`

``` bash
security = user
```

默认的配置是 `user` 模式, 在该模式下, Windows 下访问 SAMBA 需要用户名和密码, 还可以设置的选项还有 `share/user/server/domain/ads` 。可以根据需求修改此选项。

* **share**: 访问 Samba 服务器共享资源时不需要输入用户名和密码, 属于匿名访问。
* **user**: 访问 Samba 服务器共享资源时需要输入用户名和密码, 认证用户来源为 Samba 服务器本机。
* **server**: 访问 Samba 服务器共享资源时需要输入用户名和密码, 认证用户来源为另一台 Samba 服务器或 Windows 服务器。
* **domain**: Samba 服务器在一个基于 Windows NT 平台的 Windows 域中, 访问共享资源需要输入用户名和密码, 认证用户来源为 Windows 域。
* **ads**: Samba 服务器在一个基于 Windows 200X 平台的 Windows 活动目录中, 访问共享资源需要输入用户名和密码, 认证用户来源为 Windows 活动目录。

### 配置认证密码在传输过程中加密

在默认的配置文件中没有此选项, 可以手动添加, 正常情况下是需要开启的。

 `For Example:`

``` bash
encrypt passwords = yes
```

![SAMBA 服务安全配置](https://vxhly.github.io/assets/liunx-samba-3.png)

### 配置用户密码的存储方式

 `Default Configuration:`

``` bash
passdb backend = tdbsam
```

::: tip 提示
默认的配置为 `tdbsam` , 配置这个选项就是需要 `smbpasswd -a [userName]` 来建立 Samba 用户。还可以配置项为 `tdbsam/smbpasswd/ldapsam/mysql` 。可以根据需求进行修改。
:::

* **tdbsam**: 该方式使用一个数据库文件来建立用户数据库, 数据库文件名为 passdb.tdb 。可以使用 smbpasswd -a [用户名] 来建立 Samba 用户。也可以使用 pdbedit 命令来建立用户。
* **smbpasswd**: 该方式使用 Samba 提供的工具 smbpasswd 来给系统用户设置一个用于访问 Samba 服务的密码, 客户端就用这个密码访问 Samba 共享资源。此方式还要使用一个 smb passwd file = /usr/local/samba/etc/smbpasswd（或 /etc/samba/smbpasswd） 参数来指定保存用户名和密码的文件, 该文件需要手动建立。不推荐使用此方法。
* **ldapsam**: 该方式基于 LADP 的账户管理方式来验证用户, 先要建立 LDAP 服务。
* **mysql**: 该方式是将 Samba 服务器的用户名和密码存储到 MySQL 数据库中。

![SAMBA 服务安全配置](https://vxhly.github.io/assets/liunx-samba-4.png)

### 设置白名单

 `Default Configuration:`

``` bash
;hosts allow = 127\. 192.168.12\. 192.168.13.
```

::: tip 提示
默认的配置是将白名单进行注释掉的, 这里需要注意的是, 可以配置一个具体的 IP 地址, 也可以配置一个 IP 地址段, 在配置 IP 地址段的时候要注意最后一个 `.` 不要忘了写。如默认配置中的 `127.` 就是规定 `127.0.0.1 ～ 127.255.255.254` 这个 IP 地址段可以进行访问, 具体的 IP 地址不需要最后的 `.` .
:::

`For Example:` <br>

![SAMBA 服务安全配置](https://vxhly.github.io/assets/liunx-samba-5.png)

### 单目录限定用户访问, 禁止写入文件

 `For Example:`

``` bash
[test]                    # 共享目录的名字
  comment = test          # 描述信息
  path = /home/test       # 共享目录路径
  browseable = yes        # 是否允许浏览
  writable = no           # 是否允许用户写入或者创建文件
  guest ok = no           # 是否允许 guest 用户访问
  valid users = sx
  # 白名单,指定能够进入此资源目录的特定用户,使用逗号隔开,如果是组的话,使用 @ 指明
```

另外还可以配置以下参数

* **read only** => 设置是否只读, `yes/no`
* **public** => 设置是否让所有可以登录的用户看到这个项目
* **create mode** => 指明新建立的文件的属性, 一般是 `0755` 。
* **directory mode** => 指明新建立的目录的属性, 一般是 `0755` 。

![SAMBA 服务安全配置](https://vxhly.github.io/assets/liunx-samba-6.png)

### 创建共享目录, 设置权限

 `For Example:`

``` bash
mkdir /hoem/test
chmod 777 /home/test
```

::: warning 注意
共享目录一定要有执行权限, 一般 `755` 就够了, 这里设置 `777` 就是权限大一点就是了。
:::

![SAMBA 服务安全配置](https://vxhly.github.io/assets/liunx-samba-7.png)

### 添加 SAMBA 用户

 `For Example:`

``` bash
useradd zc
useradd sx
smbpasswd -a zc
smbpasswd -a sx
```

::: warning 注意
这里需要注意的是, 要使用 `smbpasswd -a` 创建用户, 因为之前设置了用户密码的存储方式为 `tdbsam` 。
:::

![SAMBA 服务安全配置](https://vxhly.github.io/assets/liunx-samba-8.png)

### 使用 windows 访问

`For Example:` <br>

![SAMBA 服务安全配置](https://vxhly.github.io/assets/liunx-samba-9.png)

输入用户名和密码, 就可以愉快的进行测试了, 这里就不多说了。

### 使用 smbclient 访问

#### 列出共享文件夹

列出某个 IP 地址所提供的共享文件夹

 `Connect Format:`

``` bash
smbclient -L [ip] -U [userName]%[password]
```

#### 像 FTP 客户端一样使用 smbclient

::: tip 提示
执行 smbclient 命令成功后, 进入 smbclient 环境, 出现提示符: `smb:\>` 这里有许多命令和 ftp 命令相似, 如 `cd` 、 `lcd` 、 `get` 、 `megt` 、 `put` 、 `mput` 等。通过这些命令, 我们可以访问远程主机的共享资源。
:::

 `Connect Format:`

``` bash
smbclient http://[ip]/[shareFolder] -U [username]%[password]
```
