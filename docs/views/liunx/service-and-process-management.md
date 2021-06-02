---
author: 星火燎原@vxhly
title: RedHat 系统安全加固（四）服务和进程管理
categories: [liunx]
tags: [Liunx, Security, Service, Protocol]
date: 2016-10-19 11:01:04
---

<Boxx changeTime="10000"/>

::: tip 前言
本系列笔记又是 Liunx 学习系列教程的一大步, 本系列学习笔记记录 RedHat 系统的安全加固。本篇笔记是记录系统服务和进程的查看与关闭, 通俗的来说进程是运行起来的程序。唯一标示进程的是进程描述符（PID）, 在 linux 内核中是通过 task_struck 和 task_list 来定义和管理进程的。
:::
<!-- more -->

## ps 查看系统进程

::: tip 命令作用
`ps` 命令用于报告当前系统的进程状态。可以搭配 `kill` 指令随时中断、删除不必要的程序。 `ps` 命令是最基本同时也是非常强大的进程查看命令, 使用该命令可以确定有哪些进程正在运行和运行的状态、进程是否结束、进程有没有僵死、哪些进程占用了过多的资源等等, 总之大部分信息都是可以通过执行该命令得到的。
:::

### 相关选项

* **-a** -------------------- 显示所有终端机下执行的程序, 除了阶段作业领导者之外
* **a** -------------------- 显示现行终端机下的所有程序, 包括其他用户的程序
* **-A** -------------------- 显示所有程序
* **-u\<用户识别码>** -------------------- 此选项的效果和指定 `"-U"` 选项相同
* **u** -------------------- 以用户为主的格式来显示程序状况
* **-U\<用户识别码>** -------------------- 列出属于该用户的程序的状况, 也可使用用户名称来指定
* **U\<用户名称>** -------------------- 列出属于该用户的程序的状况
* **-e** -------------------- 此选项的效果和指定 `"A"` 选项相同
* **e** -------------------- 列出程序时, 显示每个程序所使用的环境变量
* **-f** -------------------- 显示 UID, PPIP, C 与 STIME 栏位
* **f** -------------------- 用 ASCII 字符显示树状结构, 表达程序间的相互关系

### 示例

 `For Example:`

``` bash
ps aux | grep ssh
```

![RedHat 安全加固](https://vxhly.github.io/assets/process-1.png)

## netstat 查看网络情况

netstat 命令用来打印 Linux 中网络系统的状态信息, 可让你得知整个 Linux 系统的网络情况。

### 相关选项

* **-t 或 --tcp** -------------------- 显示 TCP 传输协议的连线状况
* **-u 或 --udp** -------------------- 显示 UDP 传输协议的连线状况
* **-n 或 --numeric** -------------------- 直接使用 ip 地址, 而不通过域名服务器
* **-p 或 --programs** -------------------- 显示正在使用 Socket 的程序识别码和程序名称
* **-l 或 --listening** -------------------- 显示监控中的服务器的 Socket

### 示例

 `For Example:`

``` bash
netstat -tulnp
```

![RedHat 安全加固](https://vxhly.github.io/assets/process-2.png)

## chkconfig 查看系统服务

::: tip 命令作用
chkconfig 命令检查、设置系统的各种服务。这是 Red Hat 公司遵循 GPL 规则所开发的程序, 它可查询操作系统在每一个执行等级中会执行哪些系统服务, 其中包括各类常驻服务。谨记 chkconfig 不是立即自动禁止或激活一个服务, 它只是简单的改变了符号连接。
:::

### 选项解释

* **--add** -------------------- 增加所指定的系统服务, 让 `chkconfig` 指令得以管理它, 并同时在系统启动的叙述文件内增加相关数据；
* **--del** -------------------- 删除所指定的系统服务, 不再由 `chkconfig` 指令管理, 并同时在系统启动的叙述文件内删除相关数据；
* **--level\<等级代号>** -------------------- 指定读系统服务要在哪一个执行等级中开启或关毕。

-- 等级 `0` 表示 -------------------- 表示关机
-- 等级 `1` 表示 -------------------- 单用户模式
-- 等级 `2` 表示 -------------------- 无网络连接的多用户命令行模式
-- 等级 `3` 表示 -------------------- 有网络连接的多用户命令行模式
-- 等级 `4` 表示 -------------------- 不可用
-- 等级 `5` 表示 -------------------- 带图形界面的多用户模式
-- 等级 `6` 表示 -------------------- 重新启动

* **--list** -------------------- 列出系统服务列表

### 示例

 `For Example:`

``` bash
chkconfig --list
```

![RedHat 安全加固](https://vxhly.github.io/assets/process-3.png)

## 关闭进程或服务

### kill 杀死进程

::: tip 命令作用
kill 命令用来删除执行中的程序或工作。kill 可将指定的信息送至程序。预设的信息为 SIGTERM(15), 可将指定程序终止。若仍无法终止该程序, 可使用 SIGKILL(9) 信息尝试强制删除程序。程序或工作的编号可利用 ps 指令或 job 指令查看。
:::

#### 选项解释

* **-a**-------------------- 当处理当前进程时, 不限制命令名和进程号的对应关系；
* **-l \<信息编号>**-------------------- 若不加 `<信息编号>` 选项, 则 `-l` 参数会列出全部的信息名称；
* **p**-------------------- 指定 kill 命令只打印相关进程的进程号, 而不发送任何信号；
* **-s \<信息名称或编号>**-------------------- 指定要送出的信息；
* **-u**-------------------- 指定用户。 参数

只有第 9 种信号( `SIGKILL` )才可以无条件终止进程, 其他信号进程都有权利忽略, 下面是常用的信号--------------------

* **HUP** 1 终端断线
* **INT** 2 中断（同 `Ctrl + C` ）
* **QUIT** 3 退出（同 `Ctrl + \` ）
* **TERM** 15 终止
* **KILL** 9 强制终止
* **CONT** 18 继续（与 `STOP` 相反, `fg/bg` 命令）
* **STOP** 19 暂停（同 `Ctrl + Z` ）

#### 示例

用 ps 查找进程, 然后用 kill 杀掉, `For Examlpe:`

``` bash
ps -ef | grep ssh
kill 4456
```

用 netstat 查找进程, 然后用 kill 杀掉, `For Examlpe:`

``` bash
netstat -tulnp | grep ssh
kill 4456
```

![RedHat 安全加固](https://vxhly.github.io/assets/process-4.png)

### service 关闭服务

::: tip 命令作用
service 命令是 Redhat Linux 兼容的发行版中用来控制系统服务的实用工具, 它以启动、停止、重新启动和关闭系统服务, 还可以显示所有系统服务的当前状态。
:::

#### 选项信息

* **-h**-------------------- 显示帮助信息；
* **--status-all**-------------------- 显示所服务的状态。

#### 示例

 `For Examlpe:`

``` bash
service mysqld status
service mysqld stop
```

![RedHat 安全加固](https://vxhly.github.io/assets/process-5.png)

那么要是系统没有 `service` 命令, 怎么办呢？Liunx 系统下是所有的服务名是存放在 `/etc/init.d/` 下的, 所以也可以使用以下命令来停止服务, `For Examlpe:`

``` bash
/etc/init.d/named status
/etc/init.d/named stop
```

![RedHat 安全加固](https://vxhly.github.io/assets/process-6.png)

### chkconfig 删除服务

#### 示例

 `For Examlpe:`

``` bash
chkconfig --list | grep sendmail
chkconfig --del sendmail
```

![RedHat 安全加固](https://vxhly.github.io/assets/process-7.png)