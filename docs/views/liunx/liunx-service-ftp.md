---
author: 星火燎原@vxhly
title: Liunx 服务配置笔记（三）FTP 服务安全配置
categories: [liunx]
tags: [Liunx, Service, Security, Protocol]
date: 2016-09-16 20:56:49
---

<Boxx changeTime="10000"/>

::: tip 前言
本篇是 Liunx 的服务配置笔记系列第三篇, 这一篇将具体介绍 Red Hat Liunx 下 FTP 服务的安全配置。确切的说, 应该叫它 vsFTP, 就是文件的上传和下载服务, 主要的功能是在服务器与客户端之间进行文件的传输。
:::
<!-- more -->

## 配置文件的位置

主配置文件: **/etc/vsftpd/vsftpd.conf**

## 服务的启动与停止

::: tip 提示
FTP 的服务名并不叫做 `vsftp` , 而是 `vsftpd` 。<br>
修改配置文件过后一定要重启 `vsftpd` 服务。
:::

``` bash
service vsftpd status        # 状态
service vsftpd start         # 启动
service vsftpd stop          # 暂停
service vsftpd restart       # 重启
```

## 服务的安装

### 创建挂载目录

 `For Example:`

``` bash
mkdir -p /mnt/cdrom
```

### 挂载光盘

 `For Example:`

``` bash
mount /dev/cdrom /mnt/cdrom
```

### 安装

 `For Example:`

``` bash
cd /mnt/cdrom/Server
rpm -ivh vsftpd-2.0.5-16.e15i386.rpm
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-1.png)

## 安全配置

### 禁止匿名用户登录

 `Default Configuration:`

``` bash
anonymous_enable=YES
```

默认的值是允许的, 这样匿名用户就可以不需要密码就可以登录 FTP 了, 这样很不安全, 把它修改成 NO 吧。

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-7.png)

#### 使用客户端进行验证

验证之前, 请重启服务。我们就直接使用 FTP 用户登录吧。

 `Connection Format:`

``` bash
ftp [host]
```

若没有成功登录, 则会显示

``` bash
530 Login incorrect.
```

使用 `bye` , 可以退出 FTP, 成功则会显示

``` bash
221 Googbye
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-8.png)

### 使用户不能登录系统, 但可以登录 FTP

首先要创建一个普通的用户, 使他不能登录系统, 只需设置登录的 shell 就好了。

创建一个新用户, 别忘了设置密码, `For Example:`

``` bash
useradd -s /sbin/nologin test
passwd test
```

`For Example:` <br>

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-9.png)

#### 使用客户端进行验证

验证之前, 请重启服务。这时候, 我们就要使用之前创建的用户进行验证了。

若登录成功, 则会显示

``` bash
230 Login successful
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-11.png)

### 禁止用户写入或者上传文件

`Default Configuration:` <br>

``` bash
write_enable=YES
```

默认是允许用户进行上传或者写入文件, 我们可以根据需求进行修改。

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-12.png)

#### 使用客户端进行验证

验证之前, 请重启服务。这时候, 我们仍然使用之前创建的用户进行验证。

无法创建目录, 则会显示

``` bash
550 Permission denied
```

尝试切换目录, 成功则会显示

``` bash
250 Directory successfully changed.
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-12.png)

### 配置空闲会话中断时间

 `Default Configuration:`

``` bash
#idle_session_timeout=600
```

默认的配置时间是十分钟, （默认的单位为秒）, 我们可以根据需求进行修改。

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-2.png)

#### 使用客户端进行验证

验证之前, 请重启服务。成功登录之后, 我们等待一段时间, 在进行操作。

若超时, 则会显示

``` bash
421 Timeout
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-3.png)

### 配置数据连接超时时间

 `Default Configuration:`

``` bash
#data_connection_timeout=120
```

默认的配置是两分钟, （默认的单位为秒）, 我们可以根据需求进行修改。

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-4.png)

#### 使用客户端进行验证

此选项需要在传输数据的时候可用, 这里不再进行演示。

### 限制普通用户只能在家目录中

 `Default Configuration:`

``` bash
#chroot_list_enable=YES
#chroot_list_file=/etc/vsftpd/chroot_list
```

默认是允许用户跳出自己的家目录的, 我们可以限制他只能在自己的家目录里。注意, 要将用户限制在家目录中, 还需要设置 'chroot_local_user=YES'。

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-5.png)

别忘了将禁止的用户添加至 `/etc/vsftpd/chroot_list`

#### 使用客户端进行验证

验证之前, 请重启服务。这时候, 我们仍然使用之前创建的用户进行验证。

不能切换目录, 则会显示

``` bash
550 Failed to changed directory.
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-6.png)

### 设置黑名单

 `Default Configuration:`

``` bash
userlist_enable=YES
```

默认情况下就是开启的, 若无开启, 我们可以根据需求进行修改。

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-13.png)

别忘了, 将禁止的用户填入 `/etc/vsftpd/ftpusers`

#### 使用客户端进行验证

验证之前, 请重启服务。这时候, 我们新建一个用户, 然后将其加入黑名单, 再进行验证。

新建用户, `For Example:` <br>

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-14.png)

添加至黑名单, `For Example:` <br>

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-15.png)

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-16.png)

## 配置虚拟用户登录

### 先安装 db4

 `For Example:`

``` bash
rpm -ivh db4-utils-4.3.29-10.el5.i386.rpm
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-17.png)

### 生成虚拟用户口令库文件

 `For Example:`

``` bash
vim /etc/vsftpd/login.txt
```

需要注意的是, 该文件的格式如下, 单数行为用户名, 偶数行为口令。

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-18.png)

### 生成口令库文件, 并修改其权限

 `For Example:`

``` bash
db_load -T -t hash -f /etc/vsftpd/login.txt /etc/vsftpd/login.db
chmod 600 /etc/vsftpd/login.db
```

注意要设置权限, 只有 root 用户拥有读写权限。

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-19.png)

### 修改虚拟用户的 PAM 文件

 `For Example:`

``` bash
vim /etc/pam.d/vsftpd
```

加入如下内容, 其他的全部注释掉

``` bash
auth required pam_userdb.so db=/etc/vsftpd/login
account required pam_userdb.so db=/etc/vsftpd/login
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-20.png)

数据文件的地址, 记得要写对哦。

### 建立虚拟用户, 并设置权限

 `For Example:`

``` bash
useradd -d /home/ftp -s /sbin/nologin virtual
chmod 777 /home/ftp
passwd virtual
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-21.png)

#### 修改主配置文件

添加如下内容, `For Example:`

``` bash
guest_enable=YES
guest_username=virtual
user_config_dir=/etc/vsftpd_user_conf
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-24.png)

#### 创建配置文件存放目录

此目录一定要与配置文件中的一致, `For Example:`

``` bash
mkdir -p /etc/vsftpd_user_conf
```

#### 为虚拟用户创建配置文件

为虚拟口令库中的每个用户分别创建一个配置文件, 配置文件中的配置于匿名用户的配置相差无几。

先创建配置文件, `For Example:`

``` bash
vim /etc/vsftpd_user_conf/ftp_test
```

在配置文件中配置如下内容, 

``` bash
anon_world_readable_only=NO               # 是否允许浏览 FTP 目录和下载
anon_upload_enable=YES                    # 是否允许上传
write_enable=YES                          # 是否有写权限
download_enable=YES                       # 是否允许下载文件
anon_mkdir_write_enable=YES               # 是否允许建立和删除目录
anon_other_write_enable=YES               # 是否允许改名和删除文件
local_root=/var/ftp/share/123
# 指定虚拟用户在系统用户下面的路径,限制虚拟用户的家目录,虚拟用户登录后的主目录。
```

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-25.png)

最后只需创建一下虚拟用户登录后的主目录, `For Example:`

``` bash
mkdir -p /var/ftp/share/123
```

### 登录测试

这时候就要使用虚拟用户口令库文件里的用户名和密码来登录 FTP 服务器

![FTP 安全配置](https://vxhly.github.io/assets/liunx-ftp-23.png)
