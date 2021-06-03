---
author: 星火燎原@vxhly
title: Ubuntu 下科学上网
categories: [networking]
tags: [Network, Liunx, Node.js]
date: 2016-07-31 21:32:24
---

<Boxx changeTime="10000"/>

::: tip 前言
翻墙, 主要用于 npm 的代理。没有 vpn 服务器不要紧, 肯花点小钱就可以搞定访问 [高梯子](http://www.gaotizi.com/aff.php?aff=133)。
:::
<!-- more -->

## 方法一

### 下载 shadowsocks-libev

下载 shadowsocks-libev 安装包, 解压并重命名为 shadowsocks-libev

[Source code (zip) 下载地址](http://github.com/shadowsocks/shadowsocks-libev/archive/v2.4.6.zip)<br>
[Source code (tar.gz) 下载地址](http://github.com/shadowsocks/shadowsocks-libev/archive/v2.4.6.tar.gz)

### 安装 shadowsocks-libev

``` bash
cd shadowsocks-libev
sudo apt-get install build-essential autoconf libtool libssl-dev \
    gawk debhelper dh-systemd init-system-helpers pkg-config
dpkg-buildpackage -us -uc -i
cd ..
sudo dpkg -i shadowsocks-libev*.deb
```

### 建议安装前先更新一下软件列表

如果显示 "现在没有可用的软件包" 等字样, 建议先执行

``` bash
sudo rm /var/lib/apt/lists/* -vf
sudo apt-get update
sudo apt-get upgrade
```

### 配置 config.json 文件

使用 `vim`

``` bash
sudo vim /etc/shadowsocks-libev/config.json
```

或者使用 `gedit`

``` bash
sudo gedit /etc/shadowsocks-libev/config.json
```

### 修改服务器地址跟密码

``` json
{
  "server": "",
  "server_port": 8388,
  "local_port": 1080,
  "password": "",
  "timeout": 60,
  "method": "rc4-md5"
}
```

### 配置 shadowsocks-libev 文件

使用 `vim`

``` bash
sudo vim /etc/init.d/shadowsocks-libev
```

或者使用 `gedit`

``` bash
sudo gedit /etc/init.d/shadowsocks-libev
```

### 修改文件

``` bash
DAEMON=/usr/bin/ss-server
```

=>

``` bash
DAEMON=/usr/bin/ss-local
```

### 启动 shadowsocks-libev

``` bash
sudo service shadowsocks-libev restart
```

### 安装 Privoxy

``` bash
sudo apt-get install privoxy
```

### 配置 Privoxy

使用 `vim`

``` bash
sudo vim /etc/privoxy/config
```

或者使用 `gedit`

``` bash
sudo gedit /etc/privoxy/config
```

按 `shift+g` 切到最后一行, 添加

``` bash
forward-socks5 / 127.0.0.1:1080 .
```

### 配置终端 http 代理

``` bash
export http_proxy="127.0.0.1:8118"
export https_proxy="127.0.0.1:8118"
```

### 配置终端 http 代理的启动

使用 `vim`

``` bash
sudo vim ~/.bashrc
```

或者使用 `gedit`

``` bash
sudo gedit ~/.bashrc
```

按 `shift+g` 切到最后一行, 添加

``` bash
export http_proxy="127.0.0.1:8118"
export https_proxy="127.0.0.1:8118"
```

如果此举会影响到终端下载, 建议还是不要设置为好。

### 启动 Privoxy

``` bash
sudo service privoxy restart
```

### 测试代理是否成功（需安装 curl）

``` bash
curl https:https://www.google.com.hk
```

返回带 `google` 字样表示受理成功

### 配置 npm 代理（需安装 node、npm）

``` bash
npm config set proxy http://127.0.0.1:8118
npm config set https-proxy http://127.0.0.1:8118
```

### 清除 npm 代理

``` bash
npm config delete proxy
npm config delete https-proxy
```

## 方法二

### 安装 shadowsocks

``` bash
sudo apt-get install python-gevent python-pip python-setuptools m2crypto
sudo pip install shadowsocks
```

### 建议安装前先更新一下软件列表

``` bash
sudo apt-get update
sudo apt-get upgrade
```

### 新建 shadowsocks.json 文件

使用 `vim`

``` bash
sudo vim /etc/shadowsocks.json
```

或使用 `gedit`

``` bash
sudo gedit /etc/shadowsocks.json
```

### 添加内容

``` json
{
  "server": "",
  "server_port": 8388,
  "local_port": 1080,
  "password": "",
  "timeout": 60,
  "method": "rc4-md5"
}
```

### 启动 shadowsocks

``` bash
sslocal -c /etc/shadowsocks.json
```

设置为启动时自动运行。在 `/etc/rc.local` 中添加如下命令, 注意在 `exit 0` 之前。

``` bash
sudo sslocal -c /etc/shadowsocks.json
```

### 安装 Privoxy

``` bash
sudo apt-get install privoxy
```

### 配置 Privoxy

使用 `vim`

``` bash
sudo vim /etc/privoxy/config
```

或者使用 `gedit`

``` bash
sudo gedit /etc/privoxy/config
```

修改监听地址

``` bash
listen-address  127.0.0.1:8118
```

按 `shift+g` 切到最后一行, 添加

``` bash
forward-socks5 / 127.0.0.1:1080 .
```

### 启动 Privoxy

``` bash
sudo /etc/init.d/privoxy restart
```

### 配置终端 http 代理

``` bash
export http_proxy="127.0.0.1:8118"
export https_proxy="127.0.0.1:8118"
```

为了方便还是在 `/etc/rc.local` 中添加如下命令, 注意在 `exit 0` 之前。

``` bash
sudo /etc/init.d/privoxy start
```

### 配置终端 http 代理的启动

使用 `vim`

``` bash
sudo vim ~/.bashrc
```

或者使用 `gedit`

``` bash
sudo gedit ~/.bashrc
```

按 `shift+g` 切到最后一行, 添加

``` bash
export http_proxy="127.0.0.1:8118"
export https_proxy="127.0.0.1:8118"
```

如果此举会影响到终端下载, 建议还是不要设置为好。

### 测试代理是否成功（需安装 curl）

``` bash
curl https://www.google.com.hk
```

返回带 `google` 字样表示受理成功

### 配置 npm 代理（需安装 node、npm）

``` bash
npm config set proxy http://127.0.0.1:8118
npm config set https-proxy http://127.0.0.1:8118
```

### 清除 npm 代理

``` bash
npm config delete proxy
npm config delete https-proxy
```

## 附上 Shadowsocks 服务器搭建方法

首先你要有一台专属于你的服务器, [hostker](http://www.hostker.com/) 这里提供较为便宜的服务器

### 更新软件源

安装前最好先更新一下软件源

``` bash
sudo apt-get update
sudo apt-get upgrade
```

### 安装 Shadowsocks 服务端

使用 `pip` 方式安装

``` bash
apt-get install python-pip
pip install shadowsocks
```

pip 安装时候可能会报错, 执行以下命令再重新安装一次

``` bash
wget //bootstrap.pypa.io/ez_setup.py -O - | python
```

当然你也可以更新一下 `pip`

``` bash
pip install --upgrade pip
```

### 启动 Shadowsocks 服务端

``` bash
ssserver -p 8388 -k password -m rc4-md5 --user nobody -d start
```

其中 `8388` 是远程服务器端口, `password` 是连接密码, `rc4-md5` 是加密方式

### Shadowsocks Windows 端

[shadowsocks-windows](http://github.com/shadowsocks/shadowsocks-windows/releases)

## 使用 shadowsocks-libev 一键安装脚本

### 默认配置: 

* 服务器端口: `8989`
* 客户端端口: `1080`
* 密码: 自己设定（如不设定, 默认为 `teddysun.com` ）

### 下载安装

#### CentOS 系统

``` bash
wget --no-check-certificate https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-libev.sh
chmod +x shadowsocks-libev.sh
./shadowsocks-libev.sh 2>&1 | tee shadowsocks-libev.log
```

#### Ubuntu, Debian 系统

``` bash
wget --no-check-certificate https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks-libev-debian.sh
chmod +x shadowsocks-libev-debian.sh
./shadowsocks-libev-debian.sh 2>&1 | tee shadowsocks-libev-debian.lo
```

#### 安装完成

安装完成后, 脚本提示如下: 

``` bash
Congratulations, shadowsocks-libev install completed!
Your Server IP:your_server_ip
Your Server Port:8989
Your Password:your_password
Your Local IP:127.0.0.1
Your Local Port:1080
Your Encryption Method:aes-256-cfb

Welcome to visit://teddysun.com/357.html
Enjoy it!
```

### 卸载方法: 

使用 `root` 用户登录, 运行以下命令: 

``` bash
./shadowsocks-libev.sh uninstall
```

### 查看进程是否存在

安装完成后即已后台启动 shadowsocks , 运行: 

``` bash
ps -ef | grep ss-server | grep -v ps | grep -v grep
```

### 其他命令

* 启动: `/etc/init.d/shadowsocks start`
* 停止: `/etc/init.d/shadowsocks stop`
* 重启: `/etc/init.d/shadowsocks restart`
* 查看状态: `/etc/init.d/shadowsocks status`
