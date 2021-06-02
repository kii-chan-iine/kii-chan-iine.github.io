---
author: 星火燎原@vxhly
title: 浅谈 SSH 服务
categories: [liunx]
tags: [Liunx, Security, Service]
date: 2017-08-15 21:42:25
---

<Boxx changeTime="10000"/>

::: tip 前言
SSH 是每一台 Linux 系统的标准配置服务, 本文将总结 SSH 的常见用法。
:::
<!-- more -->

## 什么是 SSH

SSH 为 `Secure Shell` 的缩写, 中文名字为安全外壳协议。SSH 是目前较可靠, 专为远程登录会话和其他网络服务提供安全性的协议。利用 SSH 协议可以有效防止远程管理过程中的信息泄露问题。需要指出的是, SSH 只是一种协议, 存在多种实现, 既有商业实现, 也有开源实现。

# 基本用法

SSH 主要用于远程登录, 只要一条简单命令就可以了。

``` bash
ssh [userName]@[hostIP]
```

如果本地用户名与远程用户名一致, 登录时可以省略用户名。

``` bash
ssh [hostIP]
```

SSH 的默认端口是 22, 当然也可以使用 `-p` 修改这个端口。

``` bash
ssh -p [port] [userName]@[hostIP]
```

## 中间人攻击

SSH 之所以能够保证安全, 原因在于它采用了公钥加密。

整个过程是这样的: 

* 远程主机收到用户的登录请求, 把自己的公钥发给用户。
* 用户使用这个公钥, 将登录密码加密后, 发送回来。
* 远程主机用自己的私钥, 解密登录密码, 如果密码正确, 就同意用户登录。

这个过程本身是安全的, 但是实施的时候存在一个风险: 如果有人截获了登录请求, 然后冒充远程主机, 将伪造的公钥发给用户, 那么用户很难辨别真伪。因为不像 https 协议, SSH 协议的公钥是没有证书中心（CA）公证的, 也就是说, 都是自己签发的。

可以设想, 如果攻击者插在用户与远程主机之间（比如在公共的 wifi 区域）, 用伪造的公钥, 获取用户的登录密码。再用这个密码登录远程主机, 那么 SSH 的安全机制就荡然无存了。

这种风险就是著名的["中间人攻击"](http://en.wikipedia.org/wiki/Man-in-the-middle_attack)（Man-in-the-middle attack）。

## 口令登录

如果你是第一次登录对方主机, 系统会出现下面的提示: 

::: tip 提示
The authenticity of host 'host (12.18.429.21)' can't be established.<br>
RSA key fingerprint is 98:2e:d7:e0:de:9f:ac:67:28:c2:42:2d:37:16:58:4d.<br>
Are you sure you want to continue connecting (yes/no)?
:::

这段话的意思是, 无法确认 host 主机的真实性, 只知道它的公钥指纹, 问你还想继续连接吗？

所谓"公钥指纹", 是指公钥长度较长（这里采用 RSA 算法, 长达 1024 位）, 很难比对, 所以对其进行 MD5 计算, 将它变成一个 128 位的指纹。上例中是 `98:2e:d7:e0:de:9f:ac:67:28:c2:42:2d:37:16:58:4d` , 再进行比较, 就容易多了。

很自然的一个问题就是, 用户怎么知道远程主机的公钥指纹应该是多少？回答是没有好办法, 远程主机必须在自己的网站上贴出公钥指纹, 以便用户自行核对。

假定经过风险衡量以后, 用户决定接受这个远程主机的公钥。

::: tip 提示
Are you sure you want to continue connecting (yes/no)? yes
:::

系统会出现一句提示, 表示 host 主机已经得到认可。

::: tip 提示
Warning: Permanently added 'host, 12.18.429.21' (RSA) to the list of known hosts.
::: 

然后, 会要求输入密码。

::: tip 提示
Password: (enter password)
:::

如果密码正确, 就可以登录了。

当远程主机的公钥被接受以后, 它就会被保存在文件 `$HOME/.ssh/known_hosts` 之中。下次再连接这台主机, 系统就会认出它的公钥已经保存在本地了, 从而跳过警告部分, 直接提示输入密码。

每个 SSH 用户都有自己的 `known_hosts` 文件, 此外系统也有一个这样的文件, 通常是 `/etc/ssh/ssh_known_hosts` , 保存一些对所有用户都可信赖的远程主机的公钥。

## 公钥登录

使用密码登录, 每次都必须输入密码, 非常麻烦。好在 SSH 还提供了公钥登录, 可以省去输入密码的步骤。

所谓"公钥登录", 原理很简单, 就是用户将自己的公钥储存在远程主机上。登录的时候, 远程主机会向用户发送一段随机字符串, 用户用自己的私钥加 密后, 再发回来。远程主机用事先储存的公钥进行解密, 如果成功, 就证明用户是可信的, 直接允许登录 shell, 不再要求密码。

这种方法要求用户必须提供自己的公钥。如果没有现成的, 可以直接用 ssh-keygen 生成一个: 

``` bash
ssh-keygen
```

运行上面的命令以后, 系统会出现一系列提示, 可以一路回车。其中有一个问题是, 要不要对私钥设置口令（passphrase）, 如果担心私钥的安全, 这里可以设置一个。

运行结束以后, 在 `$HOME/.ssh/` 目录下, 会新生成两个文件: `id_rsa.pub` 和 `id_rsa` 。前者是你的公钥, 后者是你的私钥。

这时再输入下面的命令, 将公钥传送到远程主机 host 上面: 

``` bash
ssh-copy-id [userName]@[hostIP]
```

好了, 从此你再登录, 就不需要输入密码了。

## authorized_keys 文件

远程主机将用户的公钥, 保存在登录后的用户主目录的 `$HOME/.ssh/authorized_keys` 文件中。公钥就是一段字符串, 只要把它追加在 authorized_keys 文件的末尾就行了。

这里不使用上面的 `ssh-copy-id` 命令, 改用下面的命令, 解释公钥的保存过程: 

``` bash
ssh [userName]@[hostIP] 'mkdir .ssh && cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub
```

这条命令由多个语句组成, 依次分解开来看: 

* **`ssh [userName]@[hostIP]`**, 表示登录远程主机
* 单引号中的 **`mkdir .ssh && cat >> .ssh/authorized_keys`**, 表示登录后在远程 shell 上执行的命令
* **`mkdir .ssh`** 的作用是, 如果用户主目录中的 .ssh 目录不存在, 就创建一个
* **`cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub`** 的作用是, 将本地的公钥文件 `~/.ssh/id_rsa.pub` , 重定向追加到远程文件 authorized_keys 的末尾。

写入 authorized_keys 文件后, 公钥登录的设置就完成了。

## 远程操作

``` bash
ssh [userName]@[hostIP] 'mkdir .ssh && cat >> .ssh/authorized_keys' < ~/.ssh/id_rsa.pub
```

单引号中间的部分, 表示在远程主机上执行的操作；后面的输入重定向, 表示数据通过 SSH 传向远程主机。

这就是说, SSH 可以在用户和远程主机之间, 建立命令和数据的传输通道, 因此很多事情都可以通过 SSH 来完成。

### 【例 1】

将 `$HOME/src/` 目录下面的所有文件, 复制到远程主机的 `$HOME/src/` 目录。

``` bash
cd && tar czv src ssh [userName]@[hostIP] 'tar xz'
```

### 【例 2】

将远程主机 `$HOME/src/` 目录下面的所有文件, 复制到用户的当前目录。

``` bash
ssh [userName]@[hostIP] 'tar cz src' tar xzv
```

### 【例 3】

查看远程主机是否运行进程 httpd。

``` bash
ssh [userName]@[hostIP] 'ps ax grep [h]ttpd'
```

## 绑定本地端口

既然 SSH 可以传送数据, 那么我们可以让那些不加密的网络连接, 全部改走 SSH 连接, 从而提高安全性。

假定我们要让 8080 端口的数据, 都通过 SSH 传向远程主机, 命令就这样写: 

``` bash
ssh -D 8080 [userName]@[hostIP]
```

SSH 会建立一个 socket, 去监听本地的 8080 端口。一旦有数据传向那个端口, 就自动把它转移到 SSH 连接上面, 发往远程主机。可以想象, 如果 8080 端口原来是一个不加密端口, 现在将变成一个加密端口。

## 本地端口转发

有时, 绑定本地端口还不够, 还必须指定数据传送的目标主机, 从而形成点对点的"端口转发"。为了区别后文的"远程端口转发", 我们把这种情况称为"本地端口转发"（Local forwarding）。

假定 host1 是本地主机, host2 是远程主机。由于种种原因, 这两台主机之间无法连通。但是, 另外还有一台 host3, 可以同时连通前面两台主机。因此, 很自然的想法就是, 通过 host3, 将 host1 连上 host2。

我们在 host1 执行下面的命令: 

``` bash
ssh -L 2121:[host2IP]:21 [host3IP]
```

命令中的 L 参数一共接受三个值, 分别是"本地端口: 目标主机: 目标主机端口", 它们之间用冒号分隔。这条命令的意思, 就是指定 SSH 绑定本地端口 2121, 然后指定 host3 将所有的数据, 转发到目标主机 host2 的 21 端口（假定 host2 运行 FTP, 默认端口为 21）。

这样一来, 我们只要连接 host1 的 2121 端口, 就等于连上了 host2 的 21 端口。

``` bash
ftp localhost:2121
```

"本地端口转发"使得 host1 和 host3 之间仿佛形成一个数据传输的秘密隧道, 因此又被称为"SSH 隧道"。

## 远程端口转发

既然"本地端口转发"是指绑定本地端口的转发, 那么"远程端口转发"（remote forwarding）当然是指绑定远程端口的转发。

还是接着看上面那个例子, host1 与 host2 之间无法连通, 必须借助 host3 转发。但是, 特殊情况出现了, host3 是一台内网机器, 它可以连接外网的 host1, 但是反过来就不行, 外网的 host1 连不上内网的 host3。这时, "本地端口转发"就不能用了, 怎么办？

解决办法是, 既然 host3 可以连 host1, 那么就从 host3 上建立与 host1 的 SSH 连接, 然后在 host1 上使用这条连接就可以了。

我们在 host3 执行下面的命令: 

``` bash
ssh -R 2121:[host2IP]:21 [host1IP]
```

R 参数也是接受三个值, 分别是"远程主机端口: 目标主机: 目标主机端口"。这条命令的意思, 就是让 host1 监听它自己的 2121 端口, 然后将所有数据经由 host3, 转发到 host2 的 21 端口。由于对于 host3 来说, host1 是远程主机, 所以这种情况就被称为"远程端口绑定"。

绑定之后, 我们在 host1 就可以连接 host2 了: 

``` bash
ftp localhost:2121
```

这里必须指出, "远程端口转发"的前提条件是, host1 和 host3 两台主机都有 sshD 和 ssh 客户端。

## SHH 的其他参数

SSH 还有一些别的参数, 也值得介绍。

N 参数, 表示只连接远程主机, 不打开远程 shell；T 参数, 表示不为这个连接分配 TTY。这个两个参数可以放在一起用, 代表这个 SSH 连接只用来传数据, 不执行远程操作。

``` bash
ssh -NT -D 8080 [hostIP]
```

f 参数, 表示 SSH 连接成功后, 转入后台运行。这样一来, 你就可以在不中断 SSH 连接的情况下, 在本地 shell 中执行其他操作。

``` bash
ssh -f -D 8080 [hostIP]
```

要关闭这个后台连接, 就只有用 kill 命令去杀掉进程。
