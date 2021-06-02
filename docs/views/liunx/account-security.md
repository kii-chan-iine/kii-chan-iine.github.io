---
author: 星火燎原@vxhly
title: RedHat 系统安全加固（一）帐号安全
categories: [liunx]
tags: [Liunx, Security, Pentest]
date: 2016-10-14 11:06:38
---

<Boxx changeTime="10000"/>

::: tip 提示
本系列笔记又是 Liunx 学习系列教程的一大步, 本系列学习笔记记录 RedHat 系统的安全加固。 本篇主要介绍相关于帐号安全的加固策略, 加强系统的防御能力。
:::
<!-- more -->

## 密码规则

### 密码长度和有效期

**/etc/login.defs** 文件是当创建用户时的一些规划, 比如创建用户时, 是否需要家目录, UID 和 GID 的范围; 用户的期限等等, 这个文件是可以通过 `root` 来定义的

 `Default Configuration:`

``` text
PASS_MAX_DAYS 99999 # 两次改变密码之间相距的最大天数, 密码有效最大天数
PASS_MIN_DAYS 0 # 两次改变密码之间相距的最小天数, 为零时代表任何时候都可以更改密码
PASS_MIN_LEN 5 # 密码最小长度
PASS_WARN_AGE 7 # 在密码过期之前警告的天数
```

::: warning 注意
以上只对之后新增的用户有效, 如果要修改已存在的用户密码规则, 需要使用 `chage` 命令
:::

使用以下命令, 可以查看用户的密码规则, `Command Format:`

``` bash
chage -l [userName]
```

`For Examlpe:` <br>

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-1.png)

## chage 命令

设置最大最小使用天数, 以及警告天数, `For Examlpe:`

``` bash
chage -M 90 -m 6 -W 30 [userName]
```

::: warning 注意
不要用该命令给 root 用户加上有效期, 如果密码过期, 再加上后文说的 `/etc/shadow` 文件加锁禁止修改, 会导致 root 提示修改密码, 无法成功修改密码, 从而无法登陆。
::: 

`chage` 选项解释

* **-m** => 密码可更改的最小天数。 为零时代表任何时候都可以更改密码。
* **-M** => 密码保持有效的最大天数。
* **-w** => 用户密码到期前, 提前收到警告信息的天数。
* **-E** => 帐号到期的日期。 过了这天, 此帐号将不可用。
* **-d** => 上一次更改的日期。
* **-i** => 停滞时期。 如果一个密码已过期这些天, 那么此帐号将不可用。
* **-l** => 例出当前的设置。 由非特权用户来确定他们的密码或帐号何时过期。

如果要修改密码过期时间为 `“从不”` , `For Examlpe:`

``` bash
chage -M 99999 [userName]
```

## 空密码检查

**/etc/shadow** 文件中的记录行与 **/etc/passwd** 中的一一对应, 是记录系统中用户的密码的, 此文件只有 `root` 用户可以对其进行修改

使用 `awk` 命令可以检查是否存在空密码, `For Examlpe:`

``` bash
awk -F ':' '($2==""){print $1}' /etc/shadow
```

## 没密码检查

空密码跟没有密码是俩回事, 体现于 **/etc/shadow** 中的密码那一列, 空密码是显示空, 没有密码是显示 `！ ！`
使用 `awk` 命令可以检查是否存在空密码, `For Examlpe:`

``` bash
awk -F ':' '($2=="！ ！ "){print $1}' /etc/shadow
```

## awk 命令

查询 UID 为 0 的用户, `For Examlpe:`

``` bash
awk -F ':' '($3==0){print $1}' /etc/passwd
```

::: tip 提示
UID 为 0 的任何用户都拥有系统的最高特权, 保证只有 root 用户的 UID 为 0
:::

`awk` 选项解释

* **`-F fs`** => fs 指定输入分隔符, fs 可以是字符串或正则表达式, 如 `-F:`
* **`-v var=value`** => 赋值一个用户定义变量, 将外部变量传递给 `awk`
* **`-f scripfile`** => 从脚本文件中读取 awk 命令
* **`-m[fr] val`** => 对 val 值设置内在限制, `-mf` 选项限制分配给 val 的最大块数目; `-mr` 选项限制记录的最大数目。 这两个功能是 Bell 实验室版 `awk` 的扩展功能, 在标准 `awk` 中不适用。

## 密码复杂度

redhat 公司专门开发了 `cracklib` 这个安装包来判断密码的复杂度。

检查是否安装了 `cracklib` 模块

``` bash
rpm -qa | grep cracklib
```

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-3.png)

在 `/etc/pam.d/system-auth` 中修改

``` 
password requisite pam_cracklib.so try_first_pass retry = 3 difok = 3 minlen = 8 dcredit = -1 ucredit = -1 lcredit = -1 ocredit = -1
password sufficient pam_unix.so md5 shadow nullok try_first_pass use_authtok remember = 5
```

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-4.png)

### 参数说明

* **`retry=N`** => 改变输入密码的次数, 默认值是 1, 一般设置为 3。 就是说, 如果用户输入的密码强度不够可以重复输入的次数。
* **`minlen=N`** => 密码最小长度（除了 `credit` 信用度的字符长度以外）
* **`dcredit=N`** => 密码中至少 `(N<0)` 或至多 `(N>=0)` 有几个数字
* **`ucredit=N`** => 密码中至少 `(N<0)` 或至多 `(N>=0)` 有几个大写字母
* **`lcredit=N`** => 密码中至少 `(N<0)` 或至多 `(N>=0)` 有几个小写字母
* **`ocredit=N`** => 密码中至少 `(N<0)` 或至多 `(N>=0)` 有几个特殊字符
* **`difok=M`** => 新密码与前一个旧密码之间至少有 M 个字符不相同

::: warning 注意
`dcredit` 、 `ucredit` 、 `lcredit` 、 `ocredit` 为信用度字符, 如果 `N<0` , 表示至少, 则新密码长度最小长度就等于 minlen.。 而当 `N>0` 时, 表示至多, 比如 `ocredit=1` , 至多有一个特殊字符, 这样的话, 即使没有特殊字符也会允许设置新密码。 还有, 此时新密码长度会大于 minlen。 算法交复杂, 这里不赘述。 因此, 通常情况下推荐使用 `N<0` 来做密码复杂度限制。 另外, 此密码规则对 root 用户无效, 只针对普通用户修改自身密码
:::

## 登录验证规则

### 记录普通用户登录失败次数并锁定用户

在 ** `/etc/pam.d/system-auth**` 中 `"#%PAM-1.0"` 下面一行加入, `For Example:`

``` 
auth required pam_tally2.so deny = 3 unlock_time = 300
```

::: warning 注意
一定要加在 `#%PAM-1.0` 下面一行
:::

参数说明

* **`deny[=n]`** => 用户连续错误登陆的最大次数, 超过则锁定
* **`unlock_time`** => 设定普通用户锁定后, 多少时间后解锁, 单位是秒
* **`even_deny_root`** => 也限制 root 用户
* **`root_unlock_time[=n]`** => 设定 root 用户锁定后, 多少时间后解锁, 单位是秒
* **`quiet`** => 不对已锁定的用户发出提示信息

::: warning 注意
以上参数根据实际需要取舍, 如果使用了 even_deny_root 参数限制 root 用户登录错误次数, 而没有配置 root_unlock_time 的话, 一旦 root 用户被锁, 解锁将很麻烦。
:::

### 查看用户登录失败次数

当登录错误次数达到最大限制后, 用户再次登录时, 会提示

``` text
Your account is locked.Maximum amount of failed attempts was reached.
```

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-6.png)

通过以下命令可以查看用户登录失败次数, `Command Format:`

``` bash
pam_tally2 -u [userName]
```

`For Example:` <br>

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-7.png)

### 解锁已经锁定的用户

* 自动解锁

-- 如果配置了 `unlock_time` 或 `root_unlock_time` (针对 `root` 用户)参数, 则等待超过这个时长后, 帐户会自动解锁

* 手动解锁: 

-- 只能以 `root` 用户通过命令 `pam_tally2 -u [userName] -r` 可以手动解锁因登录错误次数达到最大限制后自动锁定的用户, 同时登录错误次数会被重置为 0

## 登录超时

对于 bash 用户, 修改 `/etc/profile` 或 `/etc/bashrc` , 所有使用 bash 的用户都会继承, `For Example:` <br>

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-8.png)

对于 csh 用户, 修改 `/etc/csh.cshrc` , 所有使用 csh 的用户都会继承, `For Example:` <br>

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-9.png)

对于 bash 用户, 修改 `/etc/profile` 或 `/etc/bashrc` , 所有使用 bash 的用户都会继承, `For Example:` <br>

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-8.png)

对于 csh 用户, 修改 `/etc/csh.cshrc` , 所有使用 csh 的用户都会继承, `For Example:` <br>

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-9.png)

## SSH

### 禁止 root 用户远程 ssh 登录

::: tip 提示
由于之前的笔记已经记录过了, 这里就不再重复了
:::

### ssh 的黑白名单（指定用户或组）

::: tip 提示
由于之前的笔记已经记录过了, 这里就不再重复了
:::

### ssh 的黑白名单（指定 IP）

::: tip 提示
由于之前的笔记已经记录过了, 这里就不再重复了
:::

## Telnet

### 禁止 root 用户远程 telnet 登录

::: tip 提示
系统安装 `telnet-server` 服务后, 默认就是禁止 root 用户直接远程 telnet 登录的。 确认只要存在 `/etc/securetty` 文件, 就可以限制 root 直接远程 telnet 登录。
:::

### telnet 的黑白名单（指定用户或组）

系统不支持限制普通用户 telnet 远程登录

### telnet 的黑白名单（指定 IP）

与 `SSH` 中的相差无几, 同样是在 `/etc/hosts.allow` 里添加白名单, 在 `/etc/hosts.deny` 里添加黑名单, `For Example:`

``` 
in .telnetd: 192.168 .1 .10 192.168 .1 .11# / etc / hosts.allow in .telnetd: ALL# / etc / hosts.deny
```

## 禁止普通用户登录 shell

使用 `usermod` 命令修改用户的登录 `shell` 就好, `Command Format:`

``` bash
usermod -s /sbin/nologin [userName]
```

## 避免普通用户 su 到 root 用户

在 ** `/etc/pam.d/su**` 中 `"#%PAM-1.0"` 下面一行加入

``` 
auth sufficient pam_rootok.so debug
auth required pam_wheel.so group = wheel
```

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-5.png)

当然只去除红框中的 `#` 也是 ok 的<br>

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-10.png)

这样以来只有 wheel 组的用户可以 su 作为 root

## 指定可以 su 到 root 的用户

指定某个用户可以 su 到 root 用户, 需要将该用户加入到 wheel 组中

``` bash
usermod -G wheel [username]
```

::: warning 注意
如果禁止 root 用户直接登录, 也不存在 wheel 组的用户, 就无法进入 root 用户了
:::

![RedHat 系统安全加固](https://vxhly.github.io/assets/redhat-reinforce-11.png)
