---
author: 星火燎原@vxhly
title: Liunx 学习笔记（五）基础操作
categories: [liunx] 
tags: [Liunx, Command]
date: 2016-09-15 16:09:27
---

<Boxx changeTime="10000"/>

::: tip 前言
本篇是 Liunx 的学习笔记系列第五篇, 这一篇将介绍 Liunx 的基础操作。很多人觉得 Liunx 非常难学, 但不其然, 只要掌握一些基础的命令操作, Liunx 还是非常好上手的。
:::
<!-- more -->

## Liunx 终端

### 终端和控制台的区别

终端 (Terminal) 和控制台 (Console) 是有区别的

终端本质上是对应着 Liunx 上的 `/dev/tty` 设备, Liunx 的多用户登陆就是通过不同的 `/dev/tty` 设备完成的, Liunx 默认提供了 6 个纯命令行界面的 `"terminal"` （准确的说这里应该是 6 个 `virtual consoles` ）来让用户登录, 你可以通过使用 `[Ctrl]+[Alt]+[F1]～[F6]` 进行切换, 不过在在线实验环境中可能无法切换, 因为特殊功能按键会被你主机系统劫持。当你切换到其中一个终端后想要切换回图形界面, 你可以按下 `[Ctrl]+[Alt]+[F7]` 来完成

### Shell

Shell 是指"提供给使用者使用界面"的软件（命令解析器）, 类似于 DOS 下的 command (命令行)和后来的 cmd.exe。之所以被称作 shell 是因为它隐藏了操作系统底层的细节。就像是个 Shell (壳), 有壳就有核, 这里的核就是指的 UNIX/Liunx 内核。

### 重要的快捷键（只列出常用的）

| 按键              | 作用                                                   |
|-------------------|--------------------------------------------------------|
| **Tab**           | 进行命令补全                                           |
| **Ctrl+c**        | 强行终止当前程序（最常用, 你可以放心它并不会使终端退出) |
| **Ctrl+d**        | 这个才是强制退出终端的(不用敲 exit 了)                 |
| **Ctrl+s**        | 暂定当前程序, 暂停后按下任意键恢复运行(网络交互有用)    |
| **Ctrl+z**        | 将当前程序放到后台运行, 恢复到前台为命令 fg             |
| **Ctrl+u**        | 删除光标所在的整行                                     |
| **Ctrl+l**        | 清屏, 最常用                                            |
| **Ctrl+k**        | 删除从光标所在位置到行末                               |
| **Alt+Backspace** | 向前删除一个单词                                       |
| **Shift+PgUp**    | 将终端显示向上滚动                                     |
| **Shift+PgDn**    | 将终端显示向下滚动                                     |

### 使用通配符

 `举例:`

``` bash
touch love_{1..10}_Liunx.txt
```

![使用通配符](https://vxhly.github.io/assets/liunx-basic-1.png)

### shell 常用通配符

| 通配符                    | 作用                                          |
|---------------------------|-----------------------------------------------|
| *****                     | 匹配 0 或多个字符                             |
| **?**                     | 匹配任意一个字符                              |
| **[list]**                | 匹配 list 中的任意单一字符                    |
| **[!list]**               | 匹配 除 list 中的任意单一字符以外的字符       |
| **[c1-c2]**               | 匹配 c1-c2 中的任意单一字符 如: `[0-9] [a-z]` |
| **{string1, string2, ...}** | 匹配 sring1 或 string2 (或更多)其一字符串     |
| **{c1..c2}**              | 匹配 c1-c2 中全部字符 如 `{1..10}` |

## 命令手册速查

Liunx 下命令成千上百, 有个速查手册也是不错的, [Liunx 命令速查手册](http://man.linuxde.net/)

## Liunx 用户管理

### 获取当前登录的用户

``` bash
who # 获取当前登录的所有用户信息（包括远程登录）
whoami # 获取当前登录的用户名
who am i # 获取当前登录用户的信息
```

![获取当前登录的用户](https://vxhly.github.io/assets/liunx-basic-2.png)

::: tip 提示
第一列表示打开当前伪终端的用户的用户名<br>
第二列的 pts/0 中 pts 表示伪终端, 所谓伪是相对于 /dev/tty 设备而言的, 使用 `[Ctrl]+[Alt]+[F1]～[F7]` 进行切换的 /dev/tty 设备, 这是"真终端", 伪终端就是当你在图形用户界面使用 /dev/tty7 时每打开一个终端就会产生一个伪终端, pts/0 后面那个数字就表示打开的伪终端序号<br>
第三列则表示当前伪终端的启动时间
:::

### 创建用户、切换用户

``` bash
adduser [userName]            # 通过询问的方式进行新增用户
useradd -m [userName]         # 通过自行添加参数方式新增用户
useradd -u 0 -o [userName]    # 添加 uid 为 0 的用户,即 root 用户的别名,拥有 root 的权限
userdel -r [userName]         # 永久删除用户
id                          # 查看用户的 uid 或 gid 等信息
su -                        # 切换到 root 用户
su - [userName]             # 切换某一个用户
passwd                      # 为当前用户设置密码
```

::: tip 提示
用户信息将被保存至 /etc/passwd, 用户密码将被保存至 /etc/shadow<br>
使用 root 用户可以修改其内容, 只需使用 wq!, 强制保存就好
:::

### 创建组、添加组成员

``` bash
groupadd [groupName]                # 添加用户组
groupdel [groupName]                # 删除用户组
usermod -G [groupName] [userName]     # 将用户 添加至用户组,-G 附属组,-g 主属组
```

::: tip 提示
组信息将被保存至 /etc/group, 组密码将保存至 /etc/gshadow<br>
使用 root 用户可以修改其内容, 只需使用 wq!, 强制保存就好
:::

### 禁用和解锁用户

#### passwd 方法

``` bash
passwd -l [userName]    #禁用
passwd -u [userName]    #解锁
```

#### 修改 `/etc/shadow` 文件（root 用户）

``` bash
vim /etc/shadow
```

禁用则在密码位前手工添加 ! （启用删除 ! 即可）, 如: 

``` bash
test:!$1$afiVSYfG$utdc2qWLNrqNcy63Ca8301:16998:0:99999:7:::
```

或

``` bash
test:!!$1$afiVSYfG$utdc2qWLNrqNcy63Ca8301:16998:0:99999:7:::
```

使用 `:wq!` 强制保存

#### usermod 方法

``` bash
usermod -L [userName]    #禁用
usermod -U [userName]    #解锁
```

### 设置用户一登录就要修改密码

``` bash
chage -d 0 [userName]
# 这个命令其实是把密码修改日期归 0 了 （/etc/shadow 文件第 3 个字段）
```

### 查看当前登录的用户的环境变量

``` bash
env
```

## Liunx 文件权限

### 查看文件权限

``` bash
ls -l       # 可以简写成 ll,注意有些版本的 Liunx 不支持
```

#### 权限介绍

![权限介绍](https://vxhly.github.io/assets/liunx-basic-3.png)

<br>

![权限介绍](https://vxhly.github.io/assets/liunx-basic-4.png)

::: tip 提示
通常我们使用数字表示权限, 4 为读取权限, 2 为写入权限, 1 为执行权限。所以权限为 777 的为最高权限
:::

### 权限的设置

``` bash
chmod 777 text.txt    # 设置最高权限
chmod o+w text.txt    # 为其他用户添加写权限
chmod g-r text.txt    # 为组中的所有用户去除读权限
```

### 修改所属者和所属组

``` bash
chown root text.txt         # 修改所属者
chgrp text text.txt         # 修该所属组
chown root:text text.txt    # 修改所属者和所属组
```

## Liunx 文件的基本操作（很重要且很常用）

### 文件操作

``` bash
touch test                            # 新建一个空文件
echo "this is a text" > test.txt      # 输出内容重定向到文件,> 覆盖操作
echo "this is a text" >> test.txt     # 输出内容重定向到文件,>> 追加操作
rm -f test.txt                        # 删除文件
```

### 目录操作

``` bash
mkdir [folderName]                        # 新建目录
mkdir -p [path]/[folderName]              # 若父级目录不存在,则一起创建
rm -rf [path]/[folderName]                # 删除目录
```

### 复制操作

``` bash
cp test [path]/[folderName]                       # 复制文件爱内
cp -r [path]/[folderName] [path]/[folderName]     # 复制目录
```

### 移动、重命名操作

``` bash
mv [fileName] [path]/[folderName]           # 移动文件
mv [fileName1] [fileName2]                  # 重命名

# 先用通配符批量创建 5 个文件
touch file{1..5}.txt
# 批量将这 5 个后缀为 .txt 的文本文件重命名为以 .c 为后缀的文件
rename 's/\.txt/\.c/' *.txt
# 批量将这 5 个文件,文件名改为大写
rename 'y/a-z/A-Z/' *.c
```

### 查看文件

``` bash
cat /etc/passwd
tail -n 5 /etc/passwd       # 只输出文件的末尾 5 行
file /bin/ls                # 可以查看文件的类型
```

适用于快速查看文件, 都是用来打印文件内容到标准输出（终端), 相同的命令有: cat (正序)、tac (倒序)、tail(按要求输出)和 nl 命令

## 搜索文件

### whereis

直接从数据库中查询简单快速

``` bash
whereis ls      # 直接从数据库中查询简单快速
```

### locate

快而全

通过 `"/var/lib/mlocate/mlocate.db"` 数据库查找, 不过这个数据库也不是实时更新的, 系统会使用定时任务每天自动执行 updatedb 命令更新一次, 所以有时候你刚添加的文件, 它可能会找不到, 你就得自己执行一次 updatedb 命令（在我们的环境中必须先执行一次该命令）。它可以用来查找指定目录下的不同文件类型, 

如: 查找 `/etc` 下所有以 sh 开头的文件

``` bash
locate /etc/sh
```

注意, 它不只是在 etc 目录下查找并会自动递归子目录进行查找 查找 `/usr/share/` 下所有 jpg 文件

``` bash
locate /usr/share/*.jpg
```

注意要添加 ***** 号前面的反斜杠转义, 否则会无法找到 只统计数目可以加上 -c 参数, -i 参数忽略大小写进行查找, whereis 的 -b, -m, -s 同样适用

### which

避免了进入程序进行测试, 很有用

which 本身是 shell 内建的一个命令, 我们通常使用 which 来确定是否安装了某个指定的软件, 因为它只从 PATH 环境变量指定的路径中去搜索命令

``` bash
which man
```

### find

太精和太细, 知道前面的命令就已经足够了

find 应该是这几个命令中最强大的了, 它不但可以通过文件类型、文件名进行查找而且可以根据文件的属性（如文件的时间戳, 文件的权限等）进行搜索

举个栗子: 在指定目录下搜索指定文件名的文件

``` bash
find /etc/ -name interfaces
```

注意 find 命令的路径是作为第一个参数的, 基本命令格式为 `find [path] [option] [action]`

## 简单文件系统操作

### 查看磁盘和目录的容量

#### 使用 df 命令查看磁盘的容量

``` bash
df            # 以 blocks 的大小展示
df -h         # 人性化输出,以更易读的方式展示
```

#### 使用 du 命令查看目录的容量

``` bash
# 默认同样以blocks的大小展示
du
# 加上-h参数,以更易读的方式展示
du -h

# 只查看1级目录的信息
du -h -d 0 ~
# 查看2级$ du -h -d 1 ~
```

### 挂载光盘和卸载光盘

``` bash
mount /dev/cdrom /mnt/cdrom     # 挂载光盘
umount /dev/cdrom               # 卸载光盘
```

### wc 命令, 简单小巧的计数工具

``` bash
wc /etc/passwd
```

分别只输出行数、单词数、字节数、字符数和输入文本中最长一行的字节数

``` bash
# 行数
wc -l /etc/passwd

# 单词数
wc -w /etc/passwd

# 字节数
wc -c /etc/passwd

# 字符数
wc -m /etc/passwd

# 最长行字节数
wc -L /etc/passwd
```

::: warning 注意
对于西文字符来说, 一个字符就是一个字节, 但对于中文字符一个汉子是大于 2 个字节的, 具体数目是由字符编码决定的
:::

## 常用的文本处理命令

### tr 命令

tr 命令可以用来删除一段文本信息中的某些文字。或者将其进行转换 使用方式: 

``` bash
tr [option]...SET1 [SET2]
```

常用选项: 

| 选项   | 说明                                                        |
|--------|-------------------------------------------------------------|
| **-d** | 删除和 set1 匹配的字符, 注意不是全词匹配也不是按字符顺序匹配 |
| **-s** | 去除 set1 指定的在输入文本中连续并重复的字符                |

 `操作举例: `

``` bash
# 删除 "hello shiyanlou" 中所有的'o','l','h'
echo 'hello shiyanlou' | tr -d 'olh'

# 将"hello" 中的ll,去重为一个l
echo 'hello' | tr -s 'l'

# 将输入文本,全部转换为大写或小写输出
cat /etc/passwd | tr '[:lower:]' '[:upper:]'
# 上面的'[:lower:]' '[:upper:]'你也可以简单的写作'[a-z]' '[A-Z]',当然反过来将大写变小写也是可以的
```

### col 命令

col 命令可以将 Tab 换成对等数量的空格建, 或反转这个操作. 用来格式化代码超爽 使用方式: 

``` bash
col [option]
```

常用的选项有: 

| 选项   | 说明                         |
|--------|------------------------------|
| **-x** | 将 Tab 转换为空格            |
| **-h** | 将空格转换为 Tab（默认选项） |

 `操作举例: `

``` bash
# 查看/etc/protocols中的不可见字符,可以看到很多^I,这其实就是Tab键转义成可见字符的符号
cat -A /etc/protocols

# 使用col -x将/etc/protocols中的Tab转换为空格,然后再使用cat查看,你发现^I不见了
cat /etc/protocols | col -x | cat -A
```

## 福利-使用 wget 克隆网站

``` bash
wget -c -m -k -np -p -E -U Mozilla –no-check-certificate [URL]
```
