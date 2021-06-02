---
author: 星火燎原@vxhly
title: 使用 nvm 安装 Node.js
categories: [nodejs]
tags: [Liunx, Node.js]
date: 2016-07-31 15:11:44
---

<Boxx changeTime="10000"/>

::: tip 前言
安装 Node.js 的最佳方式是使用 nvm。nvm 最开始是在 `Mac OS` 下开发的, 所以 `Mac OS` 下同样使用 nvm 安装 Node.js。
:::
<!-- more -->

## 下载 nvm

### 通过 curl 下载（需要安装 curl）

``` bash
apt-get install curl
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.11/install.sh | bash
```

### 通过 wget 下载（liunx 下载命令）

``` bash
wget -qO- https://raw.githubusercontent.com/creationix/nvm/v0.33.2/install.sh | bash
```

## 重启终端, 安装 Node.js

``` bash
nvm install <version>
```

### 设定 Node.js 的默认版本

``` bash
nvm alias defaults <version>
```

### 查看 Node.js 和 npm 版本号

``` bash
node -v
npm -v
```

## npm 淘宝换源

::: tip 提示
npm 加载的模块, 多数是国外源的, 所以很容易被墙, 解决办法, 除了翻墙, 还有就是 npm 换源, 使用国内源, 但是本人还是推介使用国外源, 因为国外源较国内源稳定、全面。
:::

``` bash
npm install -g cnpm --registry=//registry.npm.taobao.org
```

使用 cnpm 安装模块

``` bash
cnpm install [name]
```
