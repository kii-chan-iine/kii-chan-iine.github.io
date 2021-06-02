---
author: 星火燎原@vxhly
title: Liunx 服务配置笔记（四）APACHE 服务安全配置
categories: [liunx]
tags: [Liunx, Service, Security, Protocol]
date: 2016-09-19 14:58:33
---

<Boxx changeTime="10000"/>

::: tip 前言
本篇是 Liunx 的服务配置笔记系列第四篇, 这一篇将具体介绍 Red Hat Liunx 下 APACHE 服务的安全配置。所谓 APACHE 就是一个 WEB 服务器, 为 WEB 网页提供服务。
:::
<!-- more -->

## 配置文件的位置

主配置文件: **/etc/httpd/conf/httpd.conf**

## 服务的启动与停止

::: tip 提示
APACHE 的服务名并不叫做 `http` , 而是 `httpd` 。<br>
修改配置文件过后一定要重启 `httpd` 服务。
:::

``` bash
service httpd status        # 状态
service httpd start         # 启动
service httpd stop          # 暂停
service httpd restart       # 重启
```

## 安全配置

### 修改网站的根目录

 `Default Configuration:`

``` bash
DocumentRoot "/var/www/html"
<Directory "/var/www/html">
```

::: tip 提示
APACHE 默认配置的网站根目录是在 `/var/www/html` 下, 我们可以根据需求修改它。修改这个配置项的时候, 还需同时修改 `<Directory "/var/www/html">` 这个配置项。
:::

修改主配置文件

``` bash
DocumentRoot "/mnt/www"
<Directory "/mnt/www/">
```

#### 使用客户端进行验证

验证之前, 请重启服务。使用浏览器打开即可。

验证之前, 还需创建网站存放的目录, 然后在创建一个测试的网页进行测试。

 `For Example:`

``` bash
mkdir /mnt/www/
echo "this is test." >> /mnt/www/index.html
```

接下来, 在浏览器里输入你的服务器的 `IP` 地址就可以访问了。<br>

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-2.png)

### 配置目录的别名

修改主配置文件

目录别名设置, 就是浏览器访问 `http://[host]/test` 时, 实际上访问的是 `/var/www/html/test` 目录。

``` bash
Alias /test "/var/www/html/test"

<Directory>
  Options Indexes
  Order allow,deny
  allow from all
</Directory>
```

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-3.png)

#### 使用客户端进行验证

验证之前, 请重启服务。使用浏览器打开即可。这时候就需要跟上目录名了。

验证之前, 还需要创建测试目录和测试网站文件。

 `For Example:`

``` bash
mkdir /var/www/html/test
echo "dir dir dir dir." >> /var/www/html/index.html
```

接下来, 在浏览器里输入 `IP/test` 就可以访问了。注意大小写敏感。<br>

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-4.png)

### 关闭版本信息的显示

 `Default Configuration:`

``` bash
ServerSignature On
```

如果你访问一个不存在的目录, 默认是会显示 APACHE 的版本信息, 这是很不安全的, 建议关掉, 注意大小写敏感。还可以设置为 `Off` 和 `EMail` 。其中 `Off` 是隐藏信息, `EMail` 是会显示管理员的 email 信息的。

修改主配置文件

``` bash
ServerSignature Off
```

#### 使用客户端进行验证

先看一下, 没有进行修改之前会显示什么吧！随便写一个目录, 访问它吧！

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-5.png)

修改好配置之后, 重新访问, 再次看一下有没有吧！

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-6.png)

### 禁用目录浏览、符号连接追踪

 `Default Configuration:`

``` bash
<Directory "/var/www/html">
  Options Index FollowSymLinks
  AllowOverride None
  Order allow,deny
  Allow from all
</Directory>
```

`Index` 是定义是否开启目录浏览, 默认开启, 要关闭只需添加 `-` ； `FollowSymLinks` 是定义是否符号连接追踪, 默认开启, 要关闭只需添加 `-` 。

修改主配置文件

由于之前修改过网站根目录, 所以在这里直接使用。

``` bash
<Directory "/mnt/html">
  Options -Index -FollowSymLinks
  AllowOverride None
  Order allow,deny
  Allow from all
</Directory>
```

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-7.png)

#### 使用客户端进行验证

* 配置禁用目录浏览, 说白了就是当一个目录下没有类似于 `index.php` 、 `index.html` 、 `default.php` 、 `defaule.html` 等这样子的主页, APACHE 默认会把该目录下的目录结构给列出来, 禁用就是阻止其列出。
* 配置禁用符号连接追踪, 说白了就是防止某个用户偶然创建的一个符号连接指向文件爱呢系统的某一部分。
* 配置白名单, 其实就是限制一些 IP 地址, 让其不能访问 APACHE。
* 这些配置, 在实际使用的过程中才会有体会, 这里就不再进行测试了。

### 开启白名单

 `Default Configuration:`

``` bash
<Directory "/var/www/html">
  Options Index FollowSymLinks
  AllowOverride None
  Order allow,deny
  Allow from all
</Directory>
```

直接修改 `Allow from` 这一项, `Allow from` 是定义白名单, 默认是允许所有 IP 地址访问, 要限制只需把 `all` 替换成允许访问的 IP 地址就好。

修改主配置文件

``` bash
<Directory "/mnt/html">
  Options -Index -FollowSymLinks
  AllowOverride None
  Order allow,deny
  Allow from 192.168.1.0/24
</Directory>
```

### 限制 http 请求包上限

HTTP 协议包包含请求包和应答包。HTTP 请求包(GET、POST 等请求方法)由三个部分构成, 分别是: `方法-URI-协议/版本` , 请求头即请求正文；HTTP 应答包和 HTTP 请求包相似, 由三个部分构成, 分别是: `协议-状态代码-描述` , 应答头即应答正文。

#### LimitRequestLine 指令

设置 HTTP 请求行的字节数限制。

LimitRequestLine 指令允许服务器管理员增加或减少客户端 HTTP 请求行允许大小的限制。因为请求行包括 HTTP 方法、URI、协议版本, 所以 LimitRequestLine 指令会限制请求 URI 的长度。服务器会需要这个值足够大以装载它所有的资源名, 包括可能在 GET 请求中所传递的查询部分的所有信息。

这个指令给了服务器管理员更大的可控性以控制客户端不正常的请求行为。这有助于避免某些形式的拒绝服务攻击。

#### LimitRequestFieldSize 指令

指定 HTTP 请求头允许的字节大小。

LimitRequestFieldSize 指令允许服务器管理员增加或减少 HTTP 请求头域大小的限制。一般来说, 服务器需要此值足够大, 以适应普通客户端的任何请求的头域大小。一个普通头域的大小对于不同的客户端来说是有很大差别的, 一般与用户配置他们的浏览器以支持更多的内容协议密切相关。SPNEGO 的认证头最大可能达到 12392 字节。

这个指令给了服务器管理员更大的可控性以控制客户端不正常的请求行为。这有助于避免某些形式的拒绝服务攻击。

#### 示例配置

默认配置中是没有 LimitRequestLine 指令和 LimitRequestFieldSize 指令, 需要手动添加

``` bash
LimitRequestLine 1024k
LimitRequestFieldSize 1024k
```

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-15.png)

## 配置 https

HTTPS 协议是由 SSL+HTTP 协议构建的可进行加密传输、身份认证的网络协议要比 http 协议安全。

### 生成密钥

``` bash
openssl genrsa 1024 > server.key
```

这是用 128 位 rsa 算法生成密钥, 得到 `server.key` 文件

### 生成证书请求文件

``` bash
openssl req -new -key server.key > server.csr
```

这是用上面的密钥生成证书请求文件 `server.csr`

### 生成证书

``` bash
openssl req -x509 -days 365 -key server.key -in server.csr > server.crt
```

这是用前面生成的密钥和证书请求生成证书 `server.crt` , -days 参数指明证书有效期, 单位为天

### 生成证书和密钥的例子

注意路径, 记住它。

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-9.png)

### 确认是否安装 ssl 模块

只要确定系统中有 `mod_ssl.so` 文件, 就可以确定已经安装好 ssl 模块。记住路径吧！

``` bash
find / -iname "*mod_ssl.so*"    # 使用 find 命令进行搜索
```

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-8.png)

### 配置 httpd-ssl.conf 文件

注意在此文件中配置证书和密钥, 注意路径, 这里就使用上面的路径。

系统中可能没有此文件, 可以先创建它, `For Example:`

``` bash
vim /etc/httpd/conf/httpd-ssl.conf
```

修改主配置文件, 添加如下内容, `For Example:`

``` bash
SSLCertificateFile /home/server.crt
SSLCertificateKeyFile /home/server.key
```

### 配置 httpd.conf 文件

#### 虚拟机设置

修改主配置文件, `For Example:`

``` bash
NameVirtualHost *:443

<VirtualHost *:443>
  ...
</VirtualHost>
```

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-10.png)

#### 修改主配置文件

打开 ssl 模块, 并引入配置文件

修改主配置文件, `For Example:`

``` bash
LoadModule ssl_module /usr/lib/httpd/modules/mod_ssl.so
Include "/etc/httpd/conf/httpd-ssl.conf"
```

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-11.png)

这里提供简洁方法, 在 vi 下使用 `:r !find / -iname "*mod_ssl.so*"` , 在配置文件中也可以很任性的将 `httpd-ssl.conf` 里的内容填入主配置文件。

修改主配置文件, `For Example:`

``` bash
LoadModule ssl_module /usr/lib/httpd/modules/mod_ssl.so
SSLCertificateFile /home/server.crt
SSLCertificateKeyFile /home/server.key
```

以上的配置写在任何地方均生效, 但要注意的是配置中可能没有以下选项, 需要手动添加, `For Example:`

``` bash
SSLEngine on
```

但是最好还是写在 `<VirtualHost *:443></VirtualHost>` 这个地方, `For Example:`

``` bash
<VirtualHost *:443>
  LoadModule ssl_module /usr/lib/httpd/modules/mod_ssl.so
  SSLEngine on
  SSLCertificateFile /home/server.crt
  SSLCertificateKeyFile /home/server.key
</VirtualHost>
```

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-16.png)

#### 将监听 80 端口注释

如果不注释的话, 依然还可以通过 `http` 方式访问, 注释掉, 只能使用 `https` 方式访问。

`For Example:` <br>

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-12.png)

### 访问

配置好了, 重启服务, 接下来就可以愉快的使用 `https` 来访问了。

`For Example:` <br>

![APACHE 服务安全配置](https://vxhly.github.io/assets/liunx-apache-13.png)
