---
author: 星火燎原@vxhly
title: Liunx 服务配置笔记（二）SSH 服务安全配置
categories: [liunx]
tags: [Liunx, Service, Security, Protocol]
date: 2016-09-16 11:59:15
---

<Boxx changeTime="10000"/>

::: tip 前言
本篇是 Liunx 的服务配置笔记系列第二篇, 这一篇将具体介绍 Red Hat Liunx 下 SSH 服务的安全配置。所谓 SSH 就是加密的数据传输, 但是使用其的默认配置并不是安全的, 所以还需要对其进行详细的配置。
:::
<!-- more -->

## 配置文件的位置

主配置文件: **/etc/ssh/sshd_config**

## 服务的启动与停止

::: tip 提示
SSH 的服务名并不叫做 `ssh` , 而是 `sshd` 。<br>
修改配置文件过后一定要重启 `sshd` 服务。
:::

``` bash
service sshd status        # 状态
service sshd start         # 启动
service sshd stop          # 暂停
service sshd restart       # 重启
```

## 安全配置

### 修改端口号

 `Default Configuration:`

``` bash
# Port 22
```

::: tip 提示
配置文件中将其注释掉, 则采用默认的端口 22, 我们可以修改它。去掉 # , 修改成我们想要的端口号。
:::

`Example Configuration:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-1.png)

#### 使用客户端进行验证

::: tip 提示
验证之前, 请重启服务。默认是允许 root 用户登录的, 以下测试例子是使用 root 用户登录的。
:::

使用默认端口连接, `Connection Format:`

``` bash
ssh [userName]@[host | ip]
```

若端口不对则会显示

``` bash
ssh: connect to host [host | ip] [port] 22: Connection refused
```

`Example Connection:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-2.png)

使用配置端口连接, `Connection Format:`

``` bash
ssh [userName]@[host | ip] -p [port]
```

连接成功则会显示

``` bash
Last login: [date] from [host | ip]
```

`Example Connection:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-3.png)

### 禁止 root 用户登录

 `Default Configuration:`

``` bash
# PermitRootLogin yes
```

默认情况下是允许 root 用户登录的, 是十分不安全, 毕竟 root 用户权限太大了。我们可以去除 #, 将 yes 修改成 no, 这就禁止了 root 用户登录。

 `Example Configuration:`

``` bash
PermitRootLogin no
```

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-4.png)

#### 使用客户端进行验证

验证之前, 请重启服务。我们已经禁止 root 用户登录了, 以下测试例子仍然使用 root 用户进行验证。

由于之前配置过端口号, 这时候就要使用配置的端口号了, `Connection Format:`

``` bash
ssh root@[host | ip] -p [port]
```

使用 `exit` 可以退出当前的连接, 退出成功将会显示

``` bash
Connection to [host | ip] closed
```

`Example Connection:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-5.png)

### 禁止空密码登录

 `Default Configuration:`

``` bash
# PermitEmptyPasswords no
```

默认情况下, SSH 已经禁止了使用空密码登录系统, 即使配置文件中将其注释掉。如果被设置成 yes, 则就是不安全的, 我们可以修改成 no。

 `Example Configuration:`

``` bash
PermitEmptyPasswords no
```

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-6.png)

#### 使用客户端进行验证

验证之前, 请重启服务。我们已经禁止 root 用户登录了, 以下测试我们将使用普通用户。注意: 使用普通用户之前, 先确保服务端是有该用户的。

由于之前配置过端口号, 这时候就要使用配置的端口号了, `Connection Format:`

``` bash
ssh [userName]@[host | ip] -p [port]
```

如果是第一次连接的话, 会提示是否保存密钥信息, 输入 `yes` 就好。

`Example Connection:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-7.png)

断开连接, 我们尝试使用空密码连接, 只需回车就好

`Example Connection:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-8.png)

### 设置必需使用密码登录

 `Default Configuration:`

``` bash
# PasswordAuthentication yes
```

默认的配置就是需要使用密码登录的, 即使它是注释的。若遇到配置值为 no 的, 一定要修改成 yes。

 `Example Configuration:`

``` bash
PasswordAuthentication yes
```

#### 使用客户端进行验证

前面已经验证过了, 这里不再重复

### 设置仅允许密钥登录

::: warning 注意
注意密钥登录是与密码登录是不一样的。只允许密钥登录的时候必须要把允许密码登录的选项禁用。
:::

 `For Example:`

``` bash
PasswordAuthentication no
```

接下来就是将下图所示的注释去掉, 就好了<br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-16.png)

#### 免密码登录

主机生成密钥对（包含公钥和私钥）

``` bash
ssh-keygen -t rsa -P ""
```

把本地主机的公钥复制到远程主机的 `authorized_keys` 文件上

``` bash
ssh-copy-id [user]@[server]
```

### 限制登录失败之后重试次数

 `Default Configuration:`

``` bash
# MaxAuthTries 6
```

默认情况下可以重试 6 次, 如果你觉得这个不合理, 当然可以修改它。同理将 # 去除, 将 6 修改成我们想要的次数。

 `Example Configuration:`

``` bash
MaxAuthTries 3
```

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-9.png)

#### 使用客户端进行验证

验证之前, 请重启服务。之前我们在服务器上新建了一个用户, 那么我们将用这个用户进行连接。

当我们输入错误的密码时, 则会显示

``` bash
Permission denied, please try again.
```

当我们最后一次输入错误的密码时, 则会显示

``` bash
Permission denied (publickey,gssapi-with-mic,password).
```

`Example Connection:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-10.png)

### 设置多长时间没有成功连接上, 就断线

 `Default Configuration:`

``` bash
# LoginGraceTime 2m
```

默认的等待时间为 2 分钟, 如果没有单位将以秒作为单位, 可用的单位分别为 h, m、s。我们可以根据需求修改, 当然是越短越安全。

 `Example Configuration:`

``` bash
LoginGraceTime 10s
```

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-11.png)

#### 使用客户端进行验证

验证之前, 请重启服务。之前我们在服务器上新建了一个用户, 那么我们仍然使用用这个用户进行连接。

我们可以输入一个错误的密码, 在等待一段时间, 按下回车, 则会显示

``` bash
Connection closed by [host | ip]
```

`Example Connection:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-12.png)

### 设置白名单和黑名单

默认配置中是没有白名单和黑名单这个选项的, 我们可以自己添加上去。多个用户之间必须使用空格分割。同理, 用户组也一样。

 `白名单`

``` bash
AllowUsers [userName]1 [userName]2 ...
AllowGroups groupName1 groupName2 ...
```

 `黑名单`

``` bash
DenyUsers [userName]1 [userName]2 ...
DenyGroups groupName1 groupName2 ...
```

`Example Configuration:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-13.png)

#### 使用客户端进行验证

验证之前, 请重启服务。由于白名单与黑名单效果是一致的, 这里只提供白名单的验证测试。注意: 服务器上依然需要新建测试用户（包含设置密码, 否则, 不能登录）。

这时候, 我们就需要输入正确的密码。

`Example Connection:` <br>

![SSH 服务安全配置](https://vxhly.github.io/assets/liunx-ssh-14.png)

### 限制 IP 登录

#### 方法一

在 `/etc/hosts.allow` 中添加 SSH 登录的 ip 或者网段, `For Example:`

``` bash
sshd:192.168.1.2:allow        # ip 地址
sshd:192.168.1.0/24:allow     # 网段
```

在 `/etc/hosts.deny` 中添加不允许 SSH 登录的 IP, `For Example:`

``` bash
sshd:ALL                      # ALL 表示除了上面允许的,其他的 ip 都拒绝登录
```

#### 方法二

使用 `iptables` , `For Example:`

``` bash
iptables -A INPUT -p tcp -s 192.168.1.2 --destination-port 22 -j ACCEPT
iptables -A INPUT -p tcp --destination-port 22 -j DROP
```

#### 方法三

修改 sshd 主配置文件 `/etc/ssh/sshd_config` , `For Example:`

``` bash
AllowUsers test@192.168.1.2
```
