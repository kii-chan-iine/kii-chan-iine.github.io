---
author: 星火燎原@vxhly
title: 在你的 VPS 服务器搭建 Ghost 博客系统
categories: [nodejs]
tags: [Node.js, Liunx]
date: 2017-06-02 14:53:49
---

<Boxx changeTime="10000"/>

::: tip 前言
Ghost 采用 Node.js 编写, 默认使用 SQLite3 数据库, 非常轻量, 因为是文件型数据库, 所以无需安装。Ghost 允许多用户编写博文, 使用 Markdown 来编写博文。
:::
<!-- more -->

## 下载 Ghost

[Ghost 中文官网](http:////www.ghostchina.com/)
[Ghost 英文官网](http:////ghost.org/)

## Ghost 中文集成版(推介)

[最新版本: Ghost v0.7.4 full (zh)](http://dl.ghostchina.com/Ghost-0.7.4-zh-full.zip)

::: tip 提示
上方链接已失效, 米娜桑可以通过这个来下载 [最新版本: Ghost v0.7.4 full (zh)](http://oss-blog.test.upcdn.net/Ghost-0.7.4-zh-full-master.zip)
:::

版本特色: 中文汉化、支持七牛、又拍云、阿里云 OSS 存储；增加 Roon 主题；集成 node_modules , 无需忍受 `npm install --production` 的麻烦。

### Ghost 中文标准版

[最新版本: Ghost v0.7.4 (zh)](http://dl.ghostchina.com/Ghost-0.7.4-zh.zip)

版本特色: 中文汉化、支持七牛、又拍云、阿里云 OSS 存储；增加 Roon 主题

### Ghost 英文原版

[最新版本: Ghost v0.7.4](http://dl.ghostchina.com/Ghost-0.7.4.zip)

## 安装前准备

### VPS 服务器和域名

VPS 个人服务器当然是必不可少的, 解析域名的的时候推介解析为 www

### Node.js

Ghost 对 Node.js 版本号要求比较严格, Ghost 官方推荐使用 `Node 0.10.x` 版本, 同时支持 `Node 0.12.x` 和 `Node 4.2.x` 版本。

这边推介使用稳定版 `Node 4.2.x` 。

安装 Node 推介使用 `nvm` 安装。

``` bash
wget -qO- https://raw.githubusercontent.com/creationix/nvm/v0.33.2/install.sh | bash
nvm install 4.2
nvm alias defaults 4.2
```

### 需要 Nginx 服务

Ghost 默认监听的端口为 2368, 而且只监听本地的 IP 地址进行访问, 所以需要 Nginx 这个服务来转发流量

Ubuntu 下安装直接执行命令

``` bash
apt-get install nginx
```

### Ghost+Web 服务器转发原理介绍

官方文档建议使用 Nginx 作为 Ghost 的前端来转发 80 端口请求, 那么 Ghost 的工作总流程就可以描述为: 

::: tip 提示
Nginx 监听 80 端口, Ghost 一直监听本地 127.0.0.1:2368 端口, 你从外网访问 www.xxx.com 时, Nginx 从 80 端口接收请求, 将之转发到本地 127.0.0.1:2368 端口, 然后 Ghost 就收到访问请求了。
:::

可是为什么不直接让 Ghost 使用 80 端口呢却要依赖 Web 服务器转发呢？这是因为 Ghost 需要运行在 root 权限上, 为了避免风险, 就用 Nginx 转发一下。另外, Nginx 毕竟是 Web 服务器, 配置灵活、并发高, 用来做 Ghost 的前端是非常适合的, 如果你服务器上还有其他网站那就更要用 Nginx 了, 不然你的 80 端口给了 Ghost 就没法负载其他网站了。

### 解压文件

``` bash
unzip -uo [fileName].zip -d [folderName]
```

## 开始配置使用

### 安装依赖

如果你下载的不是 Ghost 中文集成版, 那么你就需要在你的项目目录下执行

``` bash
npm install --production
```

当然, 这需要翻墙, Ghost 所依赖的默认数据库 sqlite 包需要从国外服务器上（亚马逊的云存储服务）下载原生插件, 即使是翻墙了也未必能下载下来, 因为 sqlite 的原生插件没有通过 npm 分发。

### 配置 config.js

解压缩之后会有一个 `config.example.js` 默认配置文件, 你需要把它复制重命名成 `config.js`

``` bash
cp config.example.js config.js
```

#### 配置 URL

``` javascript
// ...
config = {
    production: {
        url: '//www.yourdomian',
        // ...
    }
    // ...
}
// ...
```

注意这边必须写上 `//` 或者 `//` , 这里写的 URL 就是以后注册账号和登录后台时所用的 URL, 必须完全一样, 如果注册时用的 URL 和这里的不一样, 那么就会出现错误提示说无权限访问！

### 配置 Nginx 服务进行转发

打开 `/etc/nginx/nginx.conf` , 在 http 语句块内追加上: 

```text
server {
  listen 80;
  server_name www.yourdomian;

  location / {
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_set_header   Host  $http_host;
    proxy_pass http://127.0.0.1:2368;
  }
}
```

#### 重启 Nginx

``` bash
/etc/init.d/nginx restart
```

## 配置 Apache 服务进行转发

打开 `/etc/httpd/conf/httpd.conf`

``` text
# NameVirtualHost *:80 #这一行取消注释
```

再在文件末尾加上如下语句, 配置请求转发

``` text
<VirtualHost *:80>
    ServerName linuxidc.com
    ProxyRequests Off
    ProxyPreserveHost On
<Proxy *>
    AddDefaultCharset Off
    Order deny,allow
    Allow from all
</Proxy>
    ProxyPass / http://127.0.0.1:2368/
    ProxyPassReverse / http://127.0.0.1:2368/
</VirtualHost>
```

#### 重启 httpd 服务

``` bash
service httpd restart
```

## 启动 Ghost 服务

``` bash
npm start --production
```

现在启动 Ghost 并访问域名已经能进入博客页面了, 但下面还需要实现 Ghost 后台运行, 不然你退了终端的话 Ghost 就停止了。

### 使用 Forever 让 Ghost 后台运行

#### 安装 forever 包

``` bash
npm install forever -g
```

#### 启动 Ghost

``` bash
NODE_ENV=production forever start index.js
```

#### 关闭 Ghost

``` bash
forever stop index.js
```

#### 重启动 Ghost

``` bash
NODE_ENV=production forever restart index.js
```

#### 查看运行状态用这条命令

``` bash
forever list
```

### 访问

前台: <http://www.yourdomian>
后台: <http://www.yourdomian/ghost>

## 拓展配置

### Ghost 邮件系统设置详解

[Ghost 邮件系统设置详解](http://www.ghostchina.com/mail-configuration-on-self-hosted-version-of-ghost/)

### 为博客增加 "标签云"

[为博客增加 "标签云"](http://www.ghostchina.com/output-tag-cloud/)
