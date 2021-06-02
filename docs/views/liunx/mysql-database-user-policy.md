---
author: 星火燎原@vxhly
title: MySQL 数据库安全加固
categories: [liunx]
tags: [MySQL, Security, Liunx, Service]
date: 2016-10-30 16:02:43
---

<Boxx changeTime="10000"/>

::: tip 前言
数据库作为数据管理的平台, 它的安全性首先由系统内部和网络安全两部分来决定的。对于系统管理员来说, 首先要保证系统本身的安全, 在安装 MySQL 数据时, 需要对基础环境进行较好的配置。本篇笔记将记录在 RedHat Liunx 系统下 MySQL 数据库的安全加固方法。
:::
<!-- more -->

## 修改 root 用户口令

缺省安装的 MySQL 的 root 用户是空密码的, 为了安全起见, 必须修改为强密码, 所谓的强密码, 至少 8 位, 由字母、数字和符号组成的不规律密码。使用 MySQL 自带的命令 `mysqladmin` 修改 root 密码, 同时也可以登陆数据库, 修改数据库 mysql 下的 user 表的字段内容

### 使用 mysqladmin 命令

修改密码, `For Example:`

``` bash
mysqladmin -u root password "upassword"
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-1.png)

### 使用 SQL 语句修改

 `Connection Example:`

``` bash
mysql -h 127.0.0.1 -u root -p
```

输入之前设置的密码 `upassword` , 连接数据库

修改密码, `For Example:`

``` sql
mysql> use mysql;        -- 切换当前的数据库
mysql> update mysql.user set password=password('upassword') where user='root';
mysql> flush privileges; -- 强制刷新内存授权表,否则用的还是内存缓存中的口令。
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-2.png)

修改完成之后可以执行 SQL 语句查询, `For Example:`

``` sql
mysql> select * from mysql.user;
```

## 删除默认数据库和数据库用户

一般情况下, MySQL 数据库安装在本地, 并且也只需要本地的 php 脚本对 mysql 进行读取, 所以很多用户不需要, 尤其是默认安装的用户。MySQL 初始化后会自动生成空用户和 test 库, 进行安装的测试, 这会对数据库的安全构成威胁, 有必要全部删除, 最后的状态只保留单个 root 即可, 当然以后根据需要增加用户和数据库。

执行以下 SQL 语句, `For Example:`

``` sql
mysql> show databases;   -- 显示当前的所有数据库
mysql> drop database test;-- 删除数据库 test
mysql> use mysql;
mysql> delete from db;   -- 删除存放在数据库的表信息,因为还没有数据库信息
mysql> delete from mysql.user where not (user='root');
mysql> delete from mysql.user where user='root' and password='';
mysql> flush privileges;
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-3.png)

<br>

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-4.png)

<br>

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-5.png)

## 改变默认 MySQL 管理员账号

系统 MySQL 的管理员名称是 root, 而一般情况下, 数据库管理员都没进行修改, 这一定程度上对系统用户穷举的恶意行为提供了便利, 此时修改为复杂的用户名, 请不要在设定为 admin 或者 administraror 的形式, 因为它们也在易猜的用户字典中。

``` sql
mysql> update mysql.user set user="newroot" where user="root";
mysql> flush privileges;
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-6.png)

## 关于密码的管理

密码是数据库安全管理的一个很重要因素, 不要将纯文本密码保存到数据库中。如果你的计算机有安全危险, 入侵者可以获得所有的密码并使用它们。相反, 应使用 `MD5()` 、 `SHA1()` 或单向哈希函数。也不要从词典中选择密码, 有专门的程序可以破解它们, 请选用至少八位, 由字母、数字和符号组成的强密码。在存取密码时, 使用 mysql 的内置函数 `password（）` 的 sql 语句, 对密码进行加密后存储。例如以下方式在 users 表中加入新用户。

``` sql
mysql> insert into users values (1,password(1234),'test');
```

## 使用独立用户运行 MySQL

绝对不要作为使用 root 用户运行 `MySQL 服务器` 。这样做非常危险, 因为任何具有 FILE 权限的用户能够用 root 创建文件(例如: `~root/.bashrc` )。mysqld 拒绝使用 root 运行, 除非使用 `–user=root` 选项明显指定。应该用普通非特权用户运行 mysqld。

要想用其它 Unix 用户启动 mysqld, , 增加 user 选项指定 `/etc/my.cnf` 选项文件或服务器数据目录的 `my.cnf` 选项文件中的 `[mysqld]` 组的用户名。

### 加固

`/etc/my.cnf` 配置文件

``` bash
[mysqld]
user=mysql
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-7.png)

该命令使服务器用指定的用户来启动, 无论你手动启动或通过 `mysqld_safe` 或 `mysql.server` 启动, 都能确保使用 mysql 的身份。也可以在启动数据库是, 加上 `user` 参数。

``` bash
/usr/bin/mysqld_safe --user=mysql
```

作为其它 linux 用户而不用 root 运行 mysqld, 你不需要更改 user 表中的 root 用户名, 因为 `MySQL 账户` 的用户名与 `linux 账户` 的用户名无关。确保 mysqld 运行时, 只使用对数据库目录具有读或写权限的 linux 用户来运行。

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-8.png)

## 禁止远程连接数据库

在命令行 `netstat -ant` 下看到, 默认的 `3306` 端口是打开的, 此时打开了 `mysqld` 的网络监听, 允许用户远程通过帐号密码连接数本地据库, 默认情况是允许远程连接数据的。为了禁止该功能, 启动 `skip-networking` , 不监听 sql 的任何 `TCP/IP` 的连接, 切断远程访问的权利, 保证安全性。假如需要远程管理数据库, 可通过安装 `PhpMyadmin` 来实现。假如确实需要远程连接数据库, 至少修改默认的监听端口, 同时添加防火墙规则, 只允许可信任的网络的 `mysql` 监听端口的数据通过。

### 加固

编辑 `/etc/my.cnf` , 加入如下语句

``` bash
[mysqld]
skip-networking
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-9.png)

执行重启语句, `For Example:`

``` bash
service mysqld restart
```

## 限制连接用户的数量

数据库的某用户多次远程连接, 会导致性能的下降和影响其他用户的操作, 有必要对其进行限制。可以通过限制单个账户允许的连接数量来实现, 设置 `my.cnf` 文件的 `mysqld` 中的 `max_user_connections` 变量来完成。 `GRANT` 语句也可以支持 资源控制选项来限制服务器对一个账户允许的使用范围。

编辑 `/etc/my.cnf` , 加入如下语句

``` bash
[mysqld]
max_user_connections 2
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-10.png)

## 用户目录权限限制

默认的 MySQL 是安装在 `/usr/share/mysql` , 而对应的数据库文件在 `/var/lib/mysql` 目录下, 因此, 必须保证该目录不能让未经授权的用户访问后把数据库打包拷贝走了, 所以要限制对该目录的访问。确保 `mysqld` 运行时, 只使用对数据库目录具有读或写权限的 linux 用户来运行。

修改权限, `For Example:`

``` bash
chown -R root /usr/share/mysql
chown -R mysql:mysql /var/lib/mysql
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-11.png)

## 命令历史记录保护

数据库相关的 shell 操作命令都会分别记录在 `.bash_history` , 如果这些文件不慎被读取, 会导致数据库密码和数据库结构等信息泄露, 而登陆数据库后的操作将记录在 `/etc/my.cnf` 文件配置的文件中, 如果使用 `update` 表信息来修改数据库用户密码的话, 也会被读取密码, 因此需要删除这两个文件, 同时在进行登陆或备份数据库等与密码相关操作时, 应该使用 `-p` 参数加入提示输入密码后, 隐式输入密码, 建议将以上文件置空。

### 加固

配置文件 `/etc/my.cnf` 中配置数据库操作日志存放路径, `For Example:`

``` bash
[mysqld_safe]
log=~/.mysql_history
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-12.png)

随意执行一些操作, `For Example:` <br>

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-13.png)

查看 `~/.mysql_history` 文件, 可以看到数据库操作纪录（将会纪录所有成功或者失败的操作 SQL 语句）<br>

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-14.png)

查看 `~/.bash_history` 文件, 可以看到有关于数据操作的 shell 命令<br>

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-15.png)

删除操作命令, `For Example:`

``` bash
rm .bash_history .mysql_history
ln -s /dev/null .bash_history
ln -s /dev/null .mysql_history
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-16.png)

## 禁止 MySQL 对本地文件存取

在 mysql 中, 提供对本地文件的读取, 使用的是 `load data local infile` 命令, 默认在 5.0 版本中, 该选项是默认打开的, 该操作令会利用 MySQL 把本地文件读到数据库中, 然后用户就可以非法获取敏感信息了, 假如你不需要读取本地文件, 请务必关闭。

### 测试存取

测试: 首先在测试数据库下建立 `sqlfile.txt` 文件, 用逗号隔开各个字段<br>

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-19.png)

在测试数据库, 先新建一张表, 建表的 SQL 语句如下, `For Example:`

``` sql
create table users (
  userid int(11),
  username varchar(255),
  password int(11)
);
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-17.png)

执行导入语句, `For Example:`

``` sql
mysql> load data local infile 'sqlfile.txt' into table test.users fields terminated by ',';
mysql> select * from test.users;
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-18.png)

成功的将本地数据插入数据中, 此时应该禁止 MySQL 中用 `load data local infile` 命令。网络上流传的一些攻击方法中就有用它 `load data local infile` 的, 同时它也是很多新发现的 `SQL Injection` 攻击利用的手段！黑客还能通过使用 `load data local infile` 装载 `/etc/passwd` 进一个数据库表, 然后能用 `SELECT` 显示它, 这个操作对服务器的安全来说, 是致命的。可以在 `/etc/my.cnf` 中添加 `local-infile=0` , 或者加参数 `local-infile=0` 启动 `mysql` 。

### SQL 攻击

黑客使用 `load data local infile` 获取 `/etc/passwd` 文件内容步骤<br>

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-20.png)

接下来黑客只需执行 `select * from test.passwd` 就可以查询到数据了。

#### 加固

编辑 `/etc/my.cnf` 文件, 加入如下语句, `For Example:`

``` bash
[mysqld]
local-infile=0
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-21.png)

或者直接执行命令, `For Example:`

``` bash
mysql_safe --user=mysql --local-infile=0
```

再次执行 `load data local infile` 命令, 导入 `/etc/passwd` 内容, 发现命令出错

``` sql
mysql> load data local infile '/etc/passwd' into table test.passwd fields terminated by ':';
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-22.png)

## MySQL 服务器权限控制

MySQL 权限系统的主要功能是证实连接到一台给定主机的用户, 并且赋予该用户在数据库上的 `SELECT` 、 `INSERT` 、 `UPDATE` 和 `DELETE` 等权限（详见 user 超级用户表）。它的附加的功能包括有匿名的用户并对于 MySQL 特定的功能例如 `load data local infile` 进行授权及管理操作的能力。

FILE 权限允许用户在 MySQL 服务器具有写权限的目录下创建新文件, 但不能覆盖已有文件在 user 表的 `File_priv` 设置 `Y` 或 `N` 。, 所以当你不需要对服务器文件读取时, 请关闭该权限。

### 加固

::: tip 提示
需要将之前的加固配置项（ `local-infile=0` ）清掉
:::

``` sql
mysql> update mysql.user set file_priv='N' where user='root';
mysql> flush privileges;
```

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-23.png)

<br>

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-24.png)

::: warning 注意
刷新授权表之后, 一定要重新登录再进行测试, 否则还是会执行成功的
:::

当 `file_priv='N'` , 黑客进行以下语句的时候, 就会出错

``` sql
mysql> select * from test.passwd into outfile 'test.txt' fields terminated by ':';
```

当然还原配置之后, 黑客就可以很愉快的获取数据库中内容

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-25.png)

<br>

![MySQL 数据库加固](https://vxhly.github.io/assets/mysql-reinforce-26.png)

## 使用 chroot 来控制 MySQL 的运行目录

`Chroot` 是 linux 中的一种系统高级保护手段, 它的建立会将其与主系统几乎完全隔离, 也就是说, 一旦遭到什么问题, 也不会危及到正在运行的主系统。这是一个非常有效的办法, 特别是在配置网络服务程序的时候。

## 关闭对 Web 访问的支持

如果不打算让 Web 访问使用 MySQL 数据库, 没有提供诸如 PHP 这样的 Web 语言的时候, 重新设置或编译你的 PHP, 取消它们对 MySQL 的默认支持。假如服务器中使用 PHP 等 web 程序, 试试用 Web 形式非法的请求, 如果得到任何形式的 MySQL 错误, 立即分析原因, 及时修改 Web 程序, 堵住漏洞, 防止 MySQL 暴露在 web 面前。 对于 Web 的安全检查, 在 MySQL 官方文档中这么建议, 对于 web 应用, 至少检查以下清单: 

* 试试用 Web 形式输入单引号和双引号( `‘’’` 和 `‘”’` )。如果得到任何形式的 MySQL 错误, 立即分析原因。
* 试试修改动态 URL, 可以在其中添加 `%22(‘”’)` 、 `%23(‘#’)` 和 `%27(‘’’)` 。
* 试试在动态 URL 中修改数据类型, 使用前面示例中的字符, 包括数字和字符类型。你的应用程序应足够安全, 可以防范此类修改和类似攻击。
* 试试输入字符、空格和特殊符号, 不要输入数值字段的数字。你的应用程序应在将它们传递到 MySQL 之前将它们删除或生成错误。将未经过检查的值传递给 MySQL 是很危险的！
* 将数据传给 MySQL 之前先检查其大小。
* 用管理账户之外的用户名将应用程序连接到数据库。不要给应用程序任何不需要的访问权限。

## 数据库备份策略

使用 `mysqldump` 可以把整个数据库装载到一个单独的文本文件中。这个文件包含有所有重建您的数据库所需要的 SQL 命令。

``` bash
mysqldump -u root -p mysql > abc.sql
```

## mysqld 安全相关启动项

下列 `mysqld` 选项影响安全: 

* **`--allow-suspicious-udfs`**

-- 该选项控制是否可以载入主函数只有 xxx 符的用户定义函数。默认情况下, 该选项被关闭, 并且只能载入至少有辅助符的 UDF。这样可以防止从未包含合法 UDF 的共享对象文件载入函数。

* **`--local-infile[={0|1}]`**

-- 如果用 `–local-infile=0` 启动服务器, 则客户端不能使用 LOCAL in LOAD DATA 语句。

* **`--old-passwords`**

-- 强制服务器为新密码生成短( `pre-4.1` )密码哈希。当服务器必须支持旧版本客户端程序时, 为了保证兼容性这很有用。

* **`(OBSOLETE) --safe-show-database`**

-- 在以前版本的 MySQL 中, 该选项使 `SHOW DATABASES` 语句只显示用户具有部分权限的数据库名。在 `MySQL 5.1` 中, 该选项不再作为现在的 默认行为使用, 有一个 `SHOW DATABASES` 权限可以用来控制每个账户对数据库名的访问。

* **`--safe-user-create`**

-- 如果启用, 用户不能用 `GRANT` 语句创建新用户, 除非用户有 `mysql.user` 表的 `INSERT` 权限。如果你想让用户具有授权权限来创建新用户, 你应给用户授予下面的权限: `mysql> GRANT INSERT(user) ON mysql.user TO ‘user_name’@'host_name’;` 这样确保用户不能直接更改权限列, 必须使用 GRANT 语句给其它用户授予该权限。

* **`--secure-auth`**

-- 不允许鉴定有旧( `pre-4.1` )密码的账户。

* **`--skip-grant-tables`**

-- 这个选项导致服务器根本不使用权限系统。这给每个人以完全访问所有的数据库的权力！（通过执行 `mysqladmin flush-privileges` 或 `mysqladmin eload` 命令, 或执行 `FLUSH PRIVILEGES` 语句, 你能告诉一个正在运行的服务器再次开始使用授权表。）

* **`--skip-name-resolve`**

-- 主机名不被解析。所有在授权表的 Host 的列值必须是 IP 号或 localhost。

* **`--skip-networking`**

-- 在网络上不允许 TCP/IP 连接。所有到 mysqld 的连接必须经由 Unix 套接字进行。

* **`--skip-show-database`**

-- 使用该选项, 只允许有 `SHOW DATABASES` 权限的用户执行 `SHOW DATABASES` 语句, 该语句显示所有数据库名。不使用该选项, 允许所有用户执行 `SHOW DATABASES` , 但只显示用户有 `SHOW DATABASES` 权限或部分数据库权限的数据库名。请注意全局权限指数据库的权限。
