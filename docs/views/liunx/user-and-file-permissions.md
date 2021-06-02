---
author: 星火燎原@vxhly
title: RedHat 系统安全加固（二）用户和文件权限
categories: [liunx]
tags: [Liunx, Security, Pentest]
date: 2016-10-14 22:08:33
---

<Boxx changeTime="10000"/>

::: tip 前言
本系列笔记又是 Liunx 学习系列教程的一大步, 本系列学习笔记记录 RedHat 系统的安全加固。本篇笔记是记录用户和文件的权限（主要是特殊权限）, 加强系统的防御能力。
:::
<!-- more -->

## 主要的介绍

Linux 文件系统的安全主要是通过设置文件的权限来实现的。每一个 Linux 的文件或目录, 都有 3 组属性, 分别定义文件或目录的所有者, 用户组和其他人的使用权限（ `只读` 、 `可写` 、 `可执行` 、 `允许 SUID` 、 `允许 SGID` 等）。特别注意, 权限为 `SUID` 和 `SGID` 的可执行文件, 在程序运行过程中, 会给进程赋予所有者的权限, 如果被黑客发现并利用就会给系统造成危害。

## 查找同时存在 SUID 和 SGID 的文件

使用 `find` 命令

``` bash
find / -xdev -perm  -6000
```

举例, 普通用户无法直接 vi 修改 `/etc/passwd` 文件, 但却可以通过 `passwd` 命令来修改自身的密码。这时就利用了 `/bin/passwd` 的 SUID 的权限。

如果一个程序被设置成了 `S` 位标志, 那么普通用户就可以以 root 身份来运行这个程序。对系统安全性有较大不利影响, 所以应禁止所有不必要的 `SUID/SGID` 程序。

修改部分系统文件的 SUID 和 SGID 的权限: 

``` bash
chmod a-s /usr/bin/chage
chmod a-s /usr/bin/gpasswd
chmod a-s /usr/bin/wall
chmod a-s /usr/bin/chfn
chmod a-s /usr/bin/chsh
chmod a-s /usr/bin/newgrp
chmod a-s /usr/bin/write
chmod a-s /usr/sbin/usernetctl
chmod a-s /bin/mount
chmod a-s /bin/umount
chmod a-s /sbin/netreport
```

### find 命令

选项解释

* **`-perm <权限数值>`** => 查找符合指定的权限数值的文件或目录；
* **`-xdev`** => 将范围局限在先行的文件系统中；
* `find -perm mode` => 表示严格匹配, 也就是你的文件权限位转换成对应的十进制数字与 mode 一模一样, 那么匹配成功, 需要注意的是如果 mode 给的数字不足 3 位, 那么前面自动添 0 (严格的说是不足 4 位, 原因就是前面所说的 `SetUID` , `SetGID` ）
* `find -perm -mode` => 表示模糊查询, 也就是说你的文件权限位转换成对应的十进制数字被 mode 包含, 就会被搜索出来
* `find -perm +mode` => 与 `-mode` 的区别是 `+mode` 只需其中的任意一个 1 的部分被匹配, `-mode` 是所有 1 的部分都必须被匹配, 同样 `+mode` 也不管 0 位

## 文件的默认权限

使用 `umask` 指定当前用户创建文件夹和文件的默认权限属性

* 若使用者建立为 `文件` , 则预设 `没有可执行 ( x ) 项目` , 亦即只有 `rw` 这两个项目, 也就是最大为 `666` , 即权限为 `-rw-rw-rw-`
* 若使用者建立为 `目录` , 则由于 `x` 与是否可以进入此目录有关, 因此预设为所有权限均开放, 亦即为 `777` , 即权限为 `drwxrwxrwx`

### 修改默认权限

主要修改配置文件 `/etc/profile` 、 `/etc/bashrc` 、 `/etc/csh.login` 、 `/etc/csh.cshrc`
* 生成目录的权限为 `777-002=775` , 即 `drwxrwxr-x`
* 生成文件的权限为 `666-002=664` , 即 `-rw-rw-r--`
 `For Exmaple:`

``` bash
umask 002
```

* 生成目录的权限为 `777-022=755` , 即 `drwxrwxr-x`
* 生成文件的权限为 `666-022=644` , 即 - `rw-r--r--`
 `For Exmaple:`

``` bash
umask 022
```

::: warning 注意
不要单纯的进行数字加减<br>
比如, `umask 003` , 如果以数字加减, 则文件权限为 `666-003=663` , 相当于 `-rw-rw--wx` , 显然不正确, 应该先转换成权限属性再相减 `-rw-rw-rw-` 减去 `--------wx` , 即应该为 `-rw-rw-r--` <br>
影响: 移动规范要求 `umask 027` , 这样生成的文件默认权限为 `-rw-r-----` , 只有本用户可读、写、执行。如果业务需要, 其他用户也需要操作该 用户的文件, 就会有问题。可以使用默认的 `022`
:::

## 环境变量的安全性

### root 用户环境变量的安全性

主要检查 `PATH` 变量, `For Exmaple:`

``` bash
echo $PATH | egrep '(^|:)(\.|:|$)'
```

![RedHat 安全加固](https://vxhly.github.io/assets/redhat-reinforce-12.png)

检查是否包含组目录权限为 `o+w` 的目录或者 `g+w` 的目录

``` bash
find `echo $PATH | tr ':' ' '` -type d \( -perm -002 -o -perm -020 \) -ls
```

* `-002` 表示任何目录或文件, 只要带 `o+w` 就匹配
* `+002` 表示目录或文件只有 `o+w` 权限, 即 `-------w-`
可能会出现报错提示 `/root/bin` 目录不存在, `PATH` 里有该路径, 而实际上没有, 可以先在 `/root` 下建立一个 `bin` 目录

### 检查 root 用户环境变量 path 规范

检查系统 root 用户环境变量 path 中是否包含 `“.”` , `For Exmaple:`

``` bash
env | grep PATH
```

修改文件 `/etc/profile` 、 `/etc/bashrc` 或 `/root/.bash_profile` 在 `“PATH＝……”` 这句中删除

* `-002` 表示任何目录或文件, 只要带 `o+w` 就匹配
* `+002` 表示目录或文件只有 `o+w` 权限, 即 `-------w-`
可能会出现报错提示 `/root/bin` 目录不存在, `PATH` 里有该路径, 而实际上没有, 可以先在 `/root` 下建立一个 `bin` 目录

## 远程连接的安全性配置

主要查找系统中是否含有 `.netrc` 和 `.rhosts` 文件, 如无必要, 删除这两个文件, `For Exmaple:`

``` bash
find / -name .netrc
find / -name .rhosts
```

## 重要目录和文件的权限设置

执行以下命令检查目录和文件的权限设置情况

``` bash
ls -l /etc/
ls -l /etc/rc.d/init.d/
ls -l /tmp
ls -l /etc/inetd.conf
ls -l /etc/passwd
ls -l /etc/shadow
ls -l /etc/group
ls -l /etc/security
ls -l /etc/services
ls -l /etc/rc*.d
```

对于重要目录, 建议执行如下类似操作, `For Exmaple:`

``` bash
chmod -R 750 /etc/rc.d/init.d/*
```

这样只有 root 可以读、写和执行, 编写脚本一次性执行, `For Exmaple:`

``` bash
chmod 644 /etc/passwd
chmod 600 /etc/shadow
chmod 644 /etc/group
chmod 600 /etc/gshadow
chmod 700 /bin/rpm
chmod 644 /usr/bin/xhost
chmod 664 /etc/hosts
chmod 644 /etc/exports
chmod 644 /etc/issue
chmod 664 /var/log/wtmp
chmod 644 /etc/services
chmod 600  /etc/xinetd.conf
chmod 600 /etc/login.defs
chmod 600 /etc/hosts.allow
chmod 600 /etc/hosts.deny
chmod 600 /etc/securetty
chmod 600 /etc/ssh/ssh_host_key
chmod 600 /etc/ssh/sshd_config
chmod 600 /var/log/lastlog
chmod 600 /var/log/messages
chmod 600 /etc/security
```

::: warning 注意
影响:  有的移动安全加固标准要求/etc 目录是 750 权限, 但如果/etc 目录是权限是 750（drwxr-x---）的话, 由于其他组没有 x 权限, 那么普通用户就无权访问/etc 目录以及以下所有文件, 用户获取不到基本的环境变量/etc/profile 和/etc/passwd、group 等文件, 一些基本命令都会出现问题, 本用户的文件只有 uid、gid, 无法对应名用户名称和组名称。
:::

## 检查任何人都有写权限的目录

 `For Exmaple:`

``` bash
find / -type d -perm -0222 -exec ls -ldg {} \;
```

::: tip 提示
这些目录是存放每个用户临时文件的, 目录本身是带 `Sticky BIT` 粘滞位的, 即使里面的文件有 `o+w` 权限也没问题, 系统本身有保护性措施。
::: 

## 查找任何人都有写权限的文件

 `For Exmaple:`

``` bash
find / -type f -perm -0222 -exec ls -lg {} \;
```

![RedHat 安全加固](https://vxhly.github.io/assets/redhat-reinforce-14.png)

## 检查没有属主的文件

 `For Exmaple:`

``` bash
find / -nouser -o -nogroup -print
```

发现没有属主的文件往往就意味着有黑客入侵你的系统了。不能允许没有主 人的文件存在。如果在系统中发现了没有属主的文件或目录, 先查看它的完 整性, 如果一切正常, 给它一个属主。有时候卸载程序可能会出现一些没有 主人的文件或目录, 在这种情况下可以把这些文件和目录删除掉。

在安全加固脚本中, 把没有属主或属组的文件直接赋予了 `nobody` 用户和组, `For Exmaple:`

``` bash
find / -path "/proc" -prune -nouser -o -nogroup -exec chown nobody:nobody {} \;
```

::: tip 提示
命令中排除了 /proc 目录
:::

## 检查异常隐含文件

 `For Exmaple:`

``` bash
find / -xdev -name "..*" -print
find / -xdev -name "...*" -print
find / -xdev -name "..^G" -print
```

同时也要注意象 `“.xx”` 和 `“.mail”` 这样的文件名的。（这些文件名看起来都很象正常的文件名）在系统的每个地方都要查看一下有没有异常隐含文件（点号是起始字符的, 用 `“ls”` 命令看不到的文件）, 因为这些文件可能是隐藏的黑客工具或者其它一些信息（口令破解程序、其它系统的口令文件, 等等）。在 `UNIX` 下, 一个常用的技术就是用一些特殊的名, 如: `“...”` 、 `“.. ”（点点空格` ）或 `“..^G”（点点 ctrl+v 和 ctrl+G）` , 来隐含文件或目录。

基本上系统默认只有一个文件 `/usr/share/man/man1/..1.gz` , 在加固脚本中的处理方式是存在就直接把它删除掉。<br>

![RedHat 安全加固](https://vxhly.github.io/assets/redhat-reinforce-15.png)

## 加锁重要口令文件和组文件

可以用下面的命令给口令文件和组文件设置不可改变位加锁, `For Exmaple:`

``` bash
chattr +i /etc/passwd
chattr +i /etc/shadow
chattr +i /etc/group
chattr +i /etc/gshadow
chattr +i /etc/xinetd.conf
chattr +i /etc/services
```

加锁后, 就不能被改变文件内容。同时会影响 `useradd` 、 `userdel` 等跟帐户相关的操作

查看命令, `For Exmaple:`

``` bash
lsattr /etc/passwd
```

返回结果为: `----i-------- /etc/passwd`
如果要修改文件, 必须先对该文件进行解锁, `For Exmaple:`

``` bash
chattr -i /etc/passwd
chattr -i /etc/shadow
chattr -i /etc/group
chattr -i /etc/gshadow
chattr -i /etc/xinetd.conf
chattr -i /etc/services
```
