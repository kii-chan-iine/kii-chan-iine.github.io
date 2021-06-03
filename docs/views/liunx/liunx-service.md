---
author: 星火燎原@vxhly
title: Liunx 服务配置笔记（一）概述
categories: [liunx]
tags: [Liunx, Service, Security]
date: 2016-09-15 18:59:15
---

<Boxx changeTime="10000"/>

::: tip 前言
Liunx 的学习笔记已经推出有一段时间了, 现继续推出在 Liunx 中 SAMBA 服务、SSH 服务、FTP 服务、APACHE 服务、DNS 服务的安全配置。本文就先介绍这几个服务的相关内容。
:::
<!-- more -->

## SAMBA 服务

即 Windows 下的共享, SAMBA 服务只不过是将 Liunx 下的文件共享给 Windows。

### 使用到的端口号

* NetBIOS 名字服务（默认为 `137` 端口(UDP)）
* NetBIOS 数据报服务（默认为 `138` 端口(UDP)）
* 文件和打印共享（默认为 `139` 端口(TCP)）
* NetBIOS 服务在 windos 2000 及以后版本使用此端口（默认为 `445` 端口(TCP)）

### 所需软件

* **samba**

-- 这个软件主要包含了 SAMBA 的主要 daemon 文件（smbd 及 nmbd）、SAMBA 的文件（document）与 SAMBA 相关的 logrotate 设置文件及开机默认选项文件等。

* **samba-common**

-- 这个软件主要提供了 SAMBA 的主要配置文件（smb.conf）、smb.conf 语法检验的测试程序（testparm）等

* **samba-client**

-- 这个软件提供了当 Liunx 作为 SAMBA 客户端时, 所需的工具指令, 例如挂载 SAMBA 文件格式的执行文件 smbmount 等

### 主配置文件

* **/etc/samba/smb.conf**

-- 是 SAMBA 的主要配置文件, 基本上, SAMBA 就仅有这个配置文件而已, 且这个配置文件本身就是很详细的说明文件, 这个文件主要在设置工作组、NetBIOS 名称以及共享目录等。

### 查看主配置文件的帮助

``` bash
man 5 smb.conf
```

### 重要选项说明

#### 全局选项

* **workgroup = [MYGROUP]**

-- 定义 samba 服务器所在的工作组或者域（如果设置 `security = domain` 的话）。

* **server string = [Samba server]**

-- 设定 samba 服务器的描述, 通过网络邻居访问时可在备注中查看到该描述信息。

* **netbios name = [MyServer]**

-- 主机 NetBIOS 名称, 每台机子皆不同

* **security = share/user/server/domain/ads**

-- **share**: 访问 Samba 服务器共享资源时不需要输入用户名和密码, 属于匿名访问。
-- **user**: 访问 Samba 服务器共享资源时需要输入用户名和密码, 认证用户来源为 Samba 服务器本机。
-- **server**: 访问 Samba 服务器共享资源时需要输入用户名和密码, 认证用户来源为另一台 Samba 服务器或 Windows 服务器。
-- **domain**: Samba 服务器在一个基于 Windows NT 平台的 Windows 域中, 访问共享资源需要输入用户名和密码, 认证用户来源为 Windows 域。
-- **ads**: Samba 服务器在一个基于 Windows 200X 平台的 Windows 活动目录中, 访问共享资源需要输入用户名和密码, 认证用户来源为 Windows 活动目录。

* **encrypt passwords = yes | no**

-- 设定是否对密码进行加密。如果不对密码进行加密的话, 在认证会话期间, 客户端与服务器传递的是明文密码。

* **passdb backend = tdbsam/smbpasswd/ldapsam/mysql**

-- **tdbsam**: 该方式使用一个数据库文件来建立用户数据库, 数据库文件名为 passdb.tdb 。可以使用 `smbpasswd -a [userName]` 来建立 Samba 用户。也可以使用 `pdbedit` 命令来建立用户。
-- **smbpasswd**: 该方式使用 Samba 提供的工具 smbpasswd 来给系统用户设置一个用于访问 Samba 服务的密码, 客户端就用这个密码访问 Samba 共享资源。此方式还要使用一个 `smb passwd file = /usr/local/samba/etc/smbpasswd` （或 `/etc/samba/smbpasswd` ） 参数来指定保存用户名和密码的文件, 该文件需要手动建立。不推荐使用此方法。
-- **ldapsam**: 该方式基于 LADP 的账户管理方式来验证用户, 先要建立 LDAP 服务。
-- **mysql**: 该方式是将 Samba 服务器的用户名和密码存储到 MySQL 数据库中。

#### 局部设置

* **comment = [comment]**

-- 设定共享服务的描述信息。

* **path = [path]**

-- 设定共享服务的路径。

* **read only = yes | no**

-- 设定该共享服务是否为只读, 该选项有一个同义选项 `writeable` 。

* **user = [userName](@[groupName])**

-- 设定所有可能使用该共享服务的用户, 可以使用 `@group` 来设置群组 group 中的所有用户账号。该选项的值为列表, 不同的项目之间使用空格或逗号隔开。在设置 `security = share` 时, 客户端要访问某共享服务时提供的密码会与该选项指定的所有用户进行一一配对认证, 若某用户认证通过, 则以该用户权限进行共享服务访问, 否则拒绝客户端的访问（设置 `security = share` 不是允许游客访问, 只有 `guest ok = yes` 才是允许游客访问, 切记！！！）。

* **valid users = [userName](@[groupName])**

-- 设定能够使用该共享服务的用户和组, 其值的格式与 user 选项一样。

* **invalid users = [userName](@[groupName])**

-- 设定不能够使用该共享服务的用户和组, 其值的格式与 user 选项一样。

* **invalid users = [userName](@[groupName])**

-- 设定不能够使用该共享服务的用户和组, 其值的格式与 user 选项一样。

* **read list = [userName](@[groupName])**

-- 设定对该共享服务只有读取权限的用户和组, 其值的格式与 user 选项一样。

* **write list = [userName](@[groupName])**

-- 设定对该共享服务拥有读写权限的用户和组, 其值的格式与 user 选项一样。

* **admin list = [userName](@[groupName])**

-- 设定对该共享服务拥有管理权限的用户和组, 其值的格式与 user 选项一样。

* **public = yes | no**

-- 设定该共享服务是否能够被游客访问, 其同义选项有 guest ok。

* **create mode = mode**

-- mode 为八进制值, 如 0755, 其默认值为 0744。该选项指定的值用于过滤新建文件的访问权限, 新建文件的默认权限将与 create mode 指定的值进行按位与操作, 将结果再与 force create mode 指定的值进行按位或操作, 得到的结果即为新建文件的访问权限。

### 服务的启动与停止

SAMBA 的服务名并不叫做 `samba` , 而是 `smb` 。

``` bash
service smb status        # 状态
service smb start         # 启动
service smb stop          # 暂停
service smb restart       # 重启
```

### 设置开机自启

``` bash
chkconfig --list | grep smb   # 查看自启状态
chkconfig --level 35 smb on   # 设置开机自启
```

## SSH 服务

更安全的远程连接, 它通过对联机数据包加密的技术来进行数据传输, 因此, 数据就更安全了。

### 使用到的端口号

* 安全的数据传输（默认为 **`22`** 端口）

### 所需软件

一般情况下, RedHat 已经预装了 openssh, 所以这里不做过多的阐述了

### 主配置文件

* **/etc/ssh/sshd_config**

-- **sshd_config** 是 SSH 的主要配置文件, 基本上, SSH 相关的配置都放在这里, 这个文件的默认配置项已经很完美了, 事实上不需要过多的改动。

### 查看主配置文件的帮助

``` bash
man 5 sshd_config
```

### 重要选项说明

#### 监听相关配置

* **Port 22**

-- 设置连接到远程主机的端口, 可以进行修改

* **ListenAddress [192.168.1.1]**

-- 设置 sshd 服务器绑定的 IP 地址

#### 密码、登录相关配置

* **MaxAuthTries 3**

-- 最大密码输入错误次数

* **PermitRootLogin no**

-- 设置 root 能不能用 ssh 登录。这个选项一定不要设成 `“yes”`
* **LoginGraceTime 600**

-- 当用户连接上 SSH 时, 在输入密码界面, 设置在多长时间内没有成功连接 SSH , 就断线（若无单位默认以秒为单位）

* **PasswordAuthentication yes**

-- 设置是否允许口令验证

* **PermitEmptyPasswords no**

-- 设置是否允许用口令为空的帐号登录

#### 用户限制配置

* **AllowUsers [userName]**

-- 设置白名单, 可以跟着任意的数量的用户名的匹配串（patterns）或 user@host 这样的匹配串, 这些字符串用空格隔开。

* **AllowGroups [groupName]**

-- 设置白名单, 可以跟着任意的数量的组名的匹配串（patterns）或 user@host 这样的匹配串, 这些字符串用空格隔开。

* **DenyUsers [userName]**

-- 设置黑名单, 与 `AllowUsers` 写法一致。

* **DenyGroups [groupName]**

-- 设置黑名单, 与 `AllowGroups` 写法一致。

### 服务的启动与停止

::: tip 提示
SSh 的服务名并不叫做 `ssh` , 而是 `sshd` 。
:::

``` bash
service sshd status        # 状态
service sshd start         # 启动
service sshd stop          # 暂停
service sshd restart       # 重启
```

#### 设置开机自启

``` bash
chkconfig --list | grep sshd   # 查看自启状态
chkconfig --level 35 sshd on   # 设置开机自启
```

## FTP 服务

::: tip 提示
确切的说, 应该叫它 vsFTP。vsFTP 是一个基于 GPL 发布的类 Unix 系统上使用的 FTP 服务器软件, 它的全称是 `Very Secure FTP`
:::

### 使用到的端口号

* 命令通道的 FTP（默认为 **`21`** 端口）
* 数据传输的 FTP-data（默认为 **`20`** 端口）

### 所需软件

* **vsftpd**

-- 这个软件主要包含了 vsFTP 的主要 daemon 文件。

### 主要的配置文件

* **/etc/vsftp/vsftpd.conf**

-- 严格来说, 整个 vsFTP 的配置文件就只有这一个。这个配置文件的设置是以 bash 的变量相同的设置方式来处理的, 也就是 `“参数=设置值”` 来设置的, 注意, 等号两边不能有空格。

* **/etc/vsftpd/chroot_list**

-- 用来限制用户只能在家目录的文件, 需要在主配置文件中进行配置, 默认没有这个文件, 需要手动创建, 如果文件名不同, 主配置文件也要对其相应的修改。

* **/etc/vsftpd/ftpusers**

-- 黑名单列表, 就是用来规定哪些用户不能登录 FTP。

### 查看主配置文件的帮助

``` bash
man 5 vsftpd.conf
```

### 重要选项说明

#### 全局设置

* **write_enable=YES/NO**

-- 是否允许登陆用户有写权限。属于全局设置, 默认值为 YES。

* **data_connection_timeout=120**

-- 设置建立 FTP 数据连接的超时时间, 单位为秒。默认值为 120。

* **idle_session_timeout=300**

-- 设置多长时间不对 FTP 服务器进行任何操作, 则断开该 FTP 连接, 单位为秒。默认值为 300 。

    

#### 与本地用户相关的设置值

* **local_enable=YES/NO**

-- 控制是否允许本地用户登入, YES 为允许本地用户登入, NO 为不允许。默认值为 YES。

* **local_umask=022**

-- 本地用户新增档案时的 umask 值。默认值为 `077` 。

* **file_open_mode=0755**

-- 本地用户上传档案后的档案权限, 与 chmod 所使用的数值相同。默认值为 `0666` 。

* **userlist_enable=YES/NO**

-- 设置是否阻止 user_list 文件中的用户登录 FTP 服务器, 默认为 YES

#### 与虚拟用户相关的设置值

* **guest_enable=YES/NO**

-- 启用虚拟用户。默认值为 NO。启用时, 非匿名用户登录的帐号, 会被当成访客, 访客在 vsFTP 中默认会取得 FTP 这个用户的权限。

* **guest_username=FTP**

-- 在 `guest_enable=YES` 下才生效, 指定游客的身份而已。默认为 FTP。

* **user_config_dir=**

-- 指定游客用户的配置文件存放的目录, 无默认值, 需自己定义。

#### 与匿名用户相关的设置值

* **anonymous_enable=YES/NO**

-- 控制是否允许匿名用户登入, YES 为允许匿名登入, NO 为不允许。

* **anon_upload_enable=YES/NO**

-- 如果设为 YES, 则允许匿名登入者有上传文件（非目录）的权限, 只有在 `write_enable=YES` 时, 此项才有效。当然, 匿名用户必须要有对上层目录的写入权。默认值为 NO。

* **anon_world_readable_only=YES/NO**

-- 如果设为 YES, 则允许匿名登入者下载可阅读的档案（可以下载到本机阅读, 不能直接在 FTP 服务器中打开阅读）。默认值为 YES。

* **anon_mkdir_write_enable=YES/NO**

-- 如果设为 YES, 则允许匿名登入者有新增目录的权限, 只有在 `write_enable=YES` 时, 此项才有效。当然, 匿名用户必须要有对上层目录的写入权。默认值为 NO。

* **anon_other_write_enable=YES/NO**

-- 如果设为 YES, 则允许匿名登入者更多于上传或者建立目录之外的权限, 譬如删除或者重命名。（如果 `anon_upload_enable=NO` , 则匿名用户不能上传文件, 但可以删除或者重命名已经存在的文件；如果 `anon_mkdir_write_enable=NO` , 则匿名用户不能上传或者新建文件夹, 但可以删除或者重命名已经存在的文件夹。）默认值为 NO。

#### 与用户限制相关的设置值

* **chroot_list_enable=YES/NO**

-- 设置是否启用 `chroot_list_file` 配置项指定的用户列表文件。默认值为 NO。

* **chroot_list_file=/etc/vsftpd.chroot_list**

-- 用于指定用户列表文件, 该文件用于控制哪些用户可以切换到用户家目录的上级目录。

* **chroot_local_user=YES/NO**

-- 用于指定用户列表文件中的用户是否允许切换到上级目录。默认值为 NO。

### 服务的启动与停止

::: tip 提示
FTP 的服务名并不叫做 `vsftp` , 而是 `vsftpd` 。
:::

``` bash
service vsftpd status        # 状态
service vsftpd start         # 启动
service vsftpd stop          # 暂停
service vsftpd restart       # 重启
```

#### 设置开机自启

``` bash
chkconfig --list | grep vsftpd   # 查看自启状态
chkconfig --level 35 vsftpd on   # 设置开机自启
```

## APACHE 服务

APACHE 是一款 WEB 服务器软件, 它可以运行在几乎所有广泛使用的计算机平台上, 由于其跨平台和安全性被广泛使用, 是最流行的 WEB 服务器端软件之一。

### 使用到的端口号

* 超文本传输协议（默认为 `80` 端口）
* 更安全的超文本传输协议（默认为 `443` 端口）

### 所需软件

APACHE 服务需要较多的软件进行支持, RedHat 已经预装了此软件, 这里就不一一阐述了。

### 主要的配置文件

* **/etc/httpd/conf/httpd.conf**

-- 最主要的配置文件, 里买内包含了 APACHE 服务器主配置文件. 包括服务器指令的目录设置。

### 查看配置文件爱呢的帮助

``` bash
man 5 httpd.conf
```

### 重要选项说明

* **Listen 80**

-- 设置监听的端口号

* **DocumentRoot "/var/www/html"**

-- 定义网站的根目录, 默认的目录在 `/var/www/html` , 修改此选项时, 要同时修改下面一个选项。

* \<**Directory "/var/www/html"**>

-- 定义网站的一个目录, 修改此选项时, 要同时修改上面一个选项。

* **Alias /icons/ "/var/www/icons"**

-- 目录的别名设置, 就是浏览器访问 `http://[host]/icons` 时, 实际上访问的是 `/var/www/icons` 目录, 配置此选项时, 要同时配置下面的目录权限。

* \<**Directory "/var/www/icons"**>

* **Options Index FollowSymLinks**

-- `Index` 设置是否开启目录浏览, 默认启用, 禁止在前面加一个 `-`
-- `FollowSymLinks` 设置是否禁用符号连接追踪, 默认启用, 禁止在前面加一个 `-`
* **Allow from all**

-- 配置白名单, 默认允许所有的 IP 段访问。

* **ServerSignature On**

-- 配置是否隐藏 APACHE 的版本信息, 默认是开启, 还可以设置为 `Off` 和 `EMail` 。其中 `Off` 是隐藏信息, `EMail` 是会显示管理员的 email 信息的。

* **NameVirtualHost *:443**

-- 设置开放的端口, 一般在配置 `https` 时候修改它, 修改这个选项, 要同时修改下面一个选项。

* \<**VirtualHost *:443**>

-- 设置开放的端口。

### 服务的启动与停止

::: tip 提示
APACHE 的服务名并不叫做 `http` , 而是 `httpd` 。
:::

``` bash
service httpd status        # 状态
service httpd start         # 启动
service httpd stop          # 暂停
service httpd restart       # 重启
```

#### 设置开机自启

``` bash
chkconfig --list | grep httpd   # 查看自启状态
chkconfig --level 35 httpd on   # 设置开机自启
```

## DNS 服务

::: tip 提示
DNS（Domain Name Server, 域名服务器）是进行域名(domain name)和与之相对应的 IP 地址 (IP address) 转换的服务器。
:::

### 使用到的端口号

* UDP 解析（默认为 `53` 端口）

### 所需软件

* **bind**

-- DNS 主要的包, 包含 named 守护进程（用来回答查询结果的）, 还有一些重要命令: `nslookup` , `dig` 和 `host` , 等等。

* **caching-nameserver**

-- 包含 DNS 的主配置文件模板, 区域配置文件模板等。

### 主配置文件

* **/etc/named-caching-nameserver.conf**

-- DNS 主配置文件, 实际中需要把此文件复制成 `named.conf` , 方能使用

* **/etc/named-rfc1912.zones**

-- DNS 区域配置文件, 实际中需要把此文件复制成 `named.zones` , 方能使用

* **/var/named/named.zero**

-- DNS 正向解析文件, 实际中需要把此文件复制成 `named.zones` 中配置的文件名

* **/var/named/named.local**

-- DNS 反向解析文件, 实际中需要把此文件复制成 `named.zones` 中配置的文件名

### 查看主配置文件的帮助

``` bash
man 5 named-caching-nameserver.conf
```

### 重要选项说明

* **allow-query { any; }; **

-- 表示所有人能查询这台服务器, 可以修改成 IP 地址或网段, 限制其查询

* **allow-recursion { any; }; **

-- 只针对收到的递归查询请求, 对于非递归请求不关心, 可以修改成 IP 地址或网段, 限制其递归查询

* **allow-transfer { any; }; **

-- 限制哪些主机或者网络可以传递信息至区域文件

### 服务的启动与停止

::: tip 提示
DNS 的服务名并不叫做 `name` , 而是 `named` 。
:::

``` bash
service named status        # 状态
service named start         # 启动
service named stop          # 暂停
service named restart       # 重启
```

#### 设置开机自启

``` bash
chkconfig --list | grep named   # 查看自启状态
chkconfig --level 35 named on   # 设置开机自启
```
