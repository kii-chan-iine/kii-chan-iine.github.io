---
author: 星火燎原@vxhly
title: Windows 下安装 MongoDB
categories: [design]
tags: [MongoDB, Node.js, Windows, Liunx]
date: 2017-05-29 17:11:30
---

<Boxx changeTime="10000"/>

::: tip 前言
MongoDB 是目前在 IT 行业非常流行的一种非关系型数据库 (NoSql), 其灵活的数据存储方式备受当前 IT 从业人员的青睐。这里主要讲如何在 Windows 平台下安装 MongoDB。
:::
<!-- more -->

## 下载

[MongoDB 官网下载链接](http://www.mongodb.com/download-center?jmp=nav#community)

## 手动简单的配置

### 配置环境变量

安装成功之后, 将 mongodb 目录下的 bin 目录添加至 path 变量中, 如: 

![Windows 下安装 MongoDB](https://vxhly.github.io/assets/install-mongodb-1.png)

### 创建数据库文件存放目录

创建数据库文件的存放位置, 比如 `D:\mongodb\data\db` 。启动 mongodb 服务之前需要必须创建数据库文件的存放文件夹, 否则命令不会自动创建, 而且不能启动成功。

![Windows 下安装 MongoDB](https://vxhly.github.io/assets/install-mongodb-2.png)

接下来打开命令行窗口, 切换到 `D:\MongoDB\bin` 目录执行如下命令: 

``` bash
mongod --dbpath D:\MongoDB\data\db # 其中 --dbpath 是指定数据库存放目录
```

这是命令行窗口会打印一些启动信息, 最后一行显示为如下信息时表示启动成功了

``` bash
2017-05-29T17:23:16.752+0800 I NETWORK  [thread1] waiting for connections on port 27017
```

### 创建日志文件存放目录

我们需要指定 log 目录, 所以我们创建 log 目录 `D:\MongoDB\data\logs\mongodb.log` .

``` bash
mongod --dbpath D:\MongoDB\data\db --logpath=D:\MongoDB\data\logs\mongodb.log --logappend
```

![Windows 下安装 MongoDB](https://vxhly.github.io/assets/install-mongodb-3.png)

这是命令行窗口会打印一些启动信息, 最后一行显示为如下信息时表示启动成功了

``` bash
2017-05-29T17:23:16.752+0800 I NETWORK  [thread1] waiting for connections on port 27017
```

### 安装为 Windows 服务

``` bash
sc create MongoDB binPath= "D:\MongoDB\bin\mongod.exe --service --dbpath D:\MongoDB\data\db --logpath=D:\MongoDB\data\logs\mongodb.log --logappend"
```

启动服务时需要管理员权限

``` bash
net start MongoDB
```

![Windows 下安装 MongoDB](https://vxhly.github.io/assets/install-mongodb-4.png)

## 使用配置文件

很遗憾, 官方下载的安装包里面没有默认的配置文件, 若想使用配置文件只能自己建一个了, 另外个人也推荐使用配置文件来管理 MongoDB 的配置, 使用配置文件配置数据库文件、日志文件以及其它的一些配置都一目了然

1. 解压安装包到 `D:\MongoDB`
2. 建立数据库目录 `D:\MongoDB\data`
3. 建立日志目录 `D:\MongoDB\logs`
4. 建立配置文件目录 `D:\MongoDB\etc`
5. 建立配置文件 `D:\MongoDB\etc\mongodb.conf`

### 配置文件

``` shell
dbpath=D:\MongoDB\data # 数据库路径
logpath=D:\MongoDB\logs\mongodb.log # 日志输出文件路径
logappend=true # 错误日志采用追加模式,配置这个选项后 mongodb 的日志会追加到现有的日志文件,而不是从新创建一个新文件
journal=true # 启用日志文件,默认启用
quiet=true # 这个选项可以过滤掉一些无用的日志信息,若需要调试使用请设置为 false
port=27017 # 端口号 默认为 27017
```

这里仅指定了几个常用项, 更多详细配置请参考官方文档 <https://docs.mongodb.org/manual/reference/configuration-options/>

### 普通启动

``` bash
mongod --config D:\MongoDB\etc\mongodb.conf
```

### 安装为 Windows 服务

``` bash
mongod --config D:\MongoDB\etc\mongodb.conf --install
```

### 使用 SC 安装为 Windows 服务

``` bash
sc create MongoDB binPath= "D:\MongoDB\bin\mongod.exe --service --config=D:\MongoDB\etc\mongodb.conf"
```

启动服务时需要管理员权限

``` bash
net start MongoDB
```

### 浏览器测试

打开浏览器, 访问 <http://127.0.0.1:27017/> , 显示以下内容表示安装成功<br>

![Windows 下安装 MongoDB](https://vxhly.github.io/assets/install-mongodb-5.png)

## 附上 Ubuntu 下安装

Ubuntu 下安装非常简单, 执行一条命令就行, 无需任何配置, 执行完后直接使用。

``` bash
sudo apt-get -y install mongodb
```

## 图形化管理工具

::: tip 工具推荐
推荐图形化管理工具 [adminMongo](http://adminmongo.markmoffat.com/)
:::

### 环境需求

* Node.js(使用 nvm 进行安装)
* pm2( `npm -g pm2` )
* git

### 下载

``` bash
git clone https://github.com/mrvautin/adminMongo.git
cd adminMongo && npm install
```

### pm2 启动

``` bash
cd adminMongo
pm2 start app.js
```

### 访问测试

打开浏览器, 访问 <http://127.0.0.1:1234/> , 显示以下内容表示安装成功<br>

![Windows 下安装 MongoDB](https://vxhly.github.io/assets/install-mongodb-6.png)

### 配置使用

Connection name => 连接名随意 Connection string => 连接方法: `mongodb://<user>:<password>@127.0.0.1:<port>/<db>` Connection options => 链接选项 [docs](http://mongodb.github.io/node-mongodb-native/2.1/reference/connecting/connection-settings/)

连接成功后创建一个数据库进行测试<br>

![Windows 下安装 MongoDB](https://vxhly.github.io/assets/install-mongodb-7.png)

随意添加一些数据, 注意数据为 JSON 格式<br>

![Windows 下安装 MongoDB](https://vxhly.github.io/assets/install-mongodb-8.png)
