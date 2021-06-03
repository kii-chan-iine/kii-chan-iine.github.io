---
author: 星火燎原@vxhly
title: CMD 常用命令
categories: [windows]
tags: [Security, Command]
date: 2016-08-02 18:59:24
---

<Boxx changeTime="10000"/>

::: tip 前言
记录 windows 下常用的 CMD 命令。
:::
<!-- more -->

## 改变 CMD 显示颜色

``` bash
color 0
```

### 支持的颜色

* 0 = 黑色
* 1 = 蓝色
* 2 = 绿色
* 3 = 浅绿色
* 4 = 红色
* 5 = 紫色
* 6 = 黄色
* 7 = 白色
* 8 = 灰色
* 9 = 淡蓝色
* A = 淡绿色
* B = 淡浅绿色
* C = 淡红色
* D = 淡紫色
* E = 淡黄色
* F = 亮白色

## 用户和用户组相关的命令

### 添加或删除用户

`For Example:` 

``` bash
# 添加一个永不过期的用户, 并且设置登录口令
net user [userName] [password] /add /expires:never

# 删除用户
net user [userName] /delete
将用户添加到管理员组或从管理员组删除

# 将用户添加到管理员组
net localgroup Administrators [userName] /add

# 将用户从管理员组删除
net localgroup Administrators [userName] /delete

# 修改当前用户的密码
net user [userName] [password]
```

### 激活或禁用用户

`For Example:` 

``` bash
net user [userName] /active:yes    # 激活用户
net user [userName] /active:no    # 禁用用户
```

### 新建或删除用户组

`For Example:` 

``` bash
net localgroup [groupName] /add      # 新建用户组
net localgroup [groupName] /delete    # 删除用户组
```

## 文件和目录相关命令

### 列出目录结构

`For Example:` 

``` bash
dir [path]:/[folderName]
dir /a:h /s    # 显示当前目录及其子目录下的隐藏文件
dir /a:s /s    # 显示当前目录及其子目录下的系统文件
dir /a:r /s    # 显示当前目录及其子目录下的只读文件
dir /a:a /s    # 显示当前目录及其子目录下的存档文件
```

### 创建文件并向文件输出信息

`For Example:` 

``` bash
echo "test text" > 1.txt    # 覆盖
echo "tesr text" >> 1.txt   # 追加
conpy con [path]:/[fileName]    # ctrl+Z 回车结束输入
```

### 创建目录

`For Example:` 

``` bash
md [folderName]
```

### 切换目录、 盘符

`For Example:` 

``` bash
cd [path]:/[folderName]
cd /      # 切换到根目录
d:        # 切换到 D 盘
```

### 删除文件或目录

`For Example:` 

``` bash
del [fileName]           # 删除文件
rd [folderName]          # 删除空目录（目录中不能有子目录和文件）
rd [folderName] /S /Q    # 删除目录并同时删除目录中的子目录以及文件
```

### 移动或复制文件

`For Example:` 

``` bash
move [path]:/[fileName] [path]:/[fileName]    # 移动文件（也可重命名）
copy [path]:/[fileName] [path]:/[fileName]    # 复制单文件

# 复制目录及其子目录（包含空目录）
xcopy [path]:/[folderName] [path]:/[folderName] /E

# 复制目录及其子目录（不包含空目录）
xcopy path:/[folderName] [path]:/[folderName] /S

# 复制隐藏文件和系统文件
xcopy [path]:/[fileName] [path]:/[fileName] /H
```

### 查看文件的内容

`For Example:` 

``` bash
type [fileName]     # 在 DOS 下查看
start [fileName]    # 调用默认程序查看
```

### 设置文件或目录属性

`For Example:` 

``` bash
attrib +S +H [path]:/[fileName]    # 添加系统文件和隐藏文件属性

# 处理文件夹及子文件里匹配的文件（包含该文件夹）
attrib +S +H [path]:/[fileName] /S /D
```

### 重命名

`For Example:` 

``` bash
ren [fileName1] [fileName2]
```

### 显示目录树

`For Example:` 

``` bash
tree [path]:/[folderName] /F
```

## 网络、 服务相关

### ping

`For Example:` 

``` bash
ping -t -l 65500 [IP or domain]      # 死亡之 ping
ping [ip]                           # 测试网络的连通性
```

### ipconfig

`For Example:` 

``` bash
ipconfig  -all      # 查看当前所有的 ip 地址信息
ipconfig /release   # 清除当前获取到的 ip 地址
ipconfig /renew     # 重新获取 ip 地址
```

### natstat

`For Example:` 

``` bash
netstat -ano    # 列出所有端口的使用情况
netstat -o      # 显示连接进程的情况, 通常用于查找是否有木马程序
```

### arp

`For Example:` 

``` bash
arp -a      # 显示 ARP 列表
arp -d      # 清除 ARP 列表, 需要管理员权限
arp -s [ip] [macAddress]    # 添加静态项
```

### net

`For Example:` 

``` bash
net view      # 查看当前局域网内的其他连接者
net start     # 查看开启了哪些服务
net start [serviceName]     # 开启某一个服务
net stop [serviceName]      # 停止某一个服务

# 将某一个服务器的 C 盘映射成 K 盘, 攻击者常用命令
net use k: \/[ipAddress]\c$

# 以下命令建议需管理员权限
net share             # 查看本地开启的共享
net share ipc$        # 开启 ipc$ 共享
net share ipc$ /del   # 删除 ipc$ 共享
net share c$ /del     # 删除 C 盘的共享
```

### tracert

`For Example:` 

``` bash
tracert [IP or domain]       # 跟踪路由
```

### mstsc

`For Example:` 

``` bash
mstsc /v: [ip] /console     # 远程连接
```

## 关机和重启

`For Example:` 

``` bash
shutdown -s -t 180 -c "确定关机吗？ "
```

选项解释

* `-s` 关机
* `-r` 重启
* `-t` 指定时间
* `-c` 对话框确认信息
* `-a` 取消上一个任务

## 获取命令的帮助信息

`For Example:` 

``` bash
net help [command]
help [command]
[command] /?
```

## 系统文件修复（需管理员权限）

### 查看当前的系统信息

`For Example:` 

``` bash
systeminfo
```

### 扫描系统文件错误并复原

`For Example:` 

``` bash
sfc /scannow
```

