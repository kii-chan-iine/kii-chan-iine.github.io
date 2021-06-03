---
author: 星火燎原@vxhly
title: docker 使用指南
categories: [manual]
tags: [Liunx, docker]
date: 2019-03-26 14:41:23
---

<Boxx changeTime="10000"/>

::: tip 前言
Docker 是一个开源的应用容器引擎, 基于 Go 语言 并遵从Apache2.0协议开源。 Docker 可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中, 然后发布到任何流行的 Linux 机器上, 也可以实现虚拟化。
:::
<!-- more -->

## 为什么要用Docker？

在没有docker之前, 如果我们要在windows上安装mysql、mongodb需要先下载这些软件, 然后还要一步步点设置, 并且注册到windows服务。并且根本不知道安装到什么地方（某些版本mysql不能指定安装位置, 默认到C盘）。同时也会在控制面板内注册许多相关的插件。以后要卸载的时候可能还会有各种残留。作为一个强迫症患者这是万万不能忍受的。 现在有了docker, 只需要一行代码就能拉取所需软件的镜像, 然后配置一个容器就启动了。需要开发的时候就打开docker, 服务便能够运行。占用的空间也小, 性能和原生差异不大。增删查改都只需要一行代码就能完成。卸载起来也毫无残留。更重要的是, 我可以同时有一个镜像的几个容器。即使我搞坏了一个, 也可以用其他的容器。还能随时监测这些容器的磁盘占用等状态

## docker 文档

::: tip 提示
这里只找到了英文的文档, 所以英文不好的慎阅 [官方 docker 文档](https://docs.docker.com)  [非官方 docker 整理文档](https://docker_practice.gitee.io/)
:::

## 常用的 docker 命令

``` bash
docker pull [image_name]:[version]  # 下载镜像
docker run -it [docker_id] bash     # 运行一个容器实例

docker ps             # 查看当前运行的 docker 容器的进程信息
docker-compose ps     # 查看当前的使用 docker-compose up -d 开启的容器进程信息 
docker-compose up -d  # 使用本地的 docker-compose.yml 开启相关的容器
docker-compose down   # 终止当前的使用 docker-compose up -d 开启的容器
docker images         # 查看本地的所有 docker 镜像
docker image rm [docker_image_id]   # 删除本地的 docker 镜像
docker rmi -f [docker_image_id]     # 删除本地的 docker 镜像
```

## 在你的机子上安装服务

至于一些 docker 镜像下载加速或者其他的知识自行查阅文档, 这里只收集 docker 安装各种服务的方法

### Ubuntu

``` bash
$ docker pull ubuntu
$ docker run -it ubuntu bash
```

### MySQL

#### 通过镜像

``` bash
# 搜索mysql的镜像信息
$ docker search mysql   
# 从仓库拉取mysql,mysql:tag 后面可以跟标签得到指定版本,不加默认为lastest
$ docker pull mysql    

$ docker run --name test-mysql -p 3306:3306 -e MYSQL_ROOT_PASSWORD=root -d mysql:5.6
$ docker run --name test-phpmyadmin -p 8080:80 --link test-mysql:db -d phpmyadmin/phpmyadmin:latest
```

#### 通过 docker-compose.yml

``` yaml
version: "2"
services:
  mysql:
    image: mysql:5.6
    container_name: test-mysql
    restart: always
    ports:

      - "3306:3306"

    environment:
      MYSQL_USER: "root"
      MYSQL_PASSWORD: "root"
      MYSQL_ROOT_PASSWORD: "root"
    networks:

      - net-mysql

  phpmyadmin:
    image: phpmyadmin/phpmyadmin:latest
    container_name: test-phpmyadmin
    ports:

      - "8080:80"

    environment:
      MYSQL_USER: "root"
      MYSQL_PASSWORD: "root"
      MYSQL_ROOT_PASSWORD: "root"
      PMA_HOST: test-mysql
    networks:

      - net-mysql

networks:
  net-mysql:
```

docker-compose.yml 同级目录下执行: `docker-compose up -d` 之后在浏览器访问 `http://127.0.0.1:7777` 即可

如果无法正常登录的话, 使用 `docker exec -it [dockerId] bash` 进入 MySQL 容器并登陆  MySQL 执行以下命令

``` bash
mysql> grant all privileges on *.*  to 'root'@'%';
mysql> flush privileges;
```

### Node + Vue

在项目文件夹下创建 `dockerfile`

``` dockerfile
FROM node: latest

# 设置时区
RUN sh - c "echo 'Asia/Shanghai' > /etc/timezone" && \
    dpkg - reconfigure - f noninteractive tzdata
    
# 使用阿里云的Ubuntu镜像
RUN echo '\n\
deb http://mirrors.aliyun.com/ubuntu/ trusty main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ trusty-security main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ trusty-updates main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ trusty-proposed main restricted universe multiverse\n\
deb http://mirrors.aliyun.com/ubuntu/ trusty-backports main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ trusty main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-security main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-updates main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-proposed main restricted universe multiverse\n\
deb-src http://mirrors.aliyun.com/ubuntu/ trusty-backports main restricted universe multiverse\n'\ >
    /etc/apt / sources.list
    
RUN apt - get update\ &&
    apt - get install - y\# node - sass 等编译依赖
make gcc g++build - essential python2 .7\# 命令行工具
curl wget vim git git - flow yarn xsel

# 切换 node 的软件版本源 => 切源工具 cgr 和 nrm
# cgr 是基于nrm的改进版本, 进行了一些优化, 能同时管理 npm、 yarn 源# cgr use taobao n => 只是切换 npm 源# cgr use taobao y => 只是切换 yarn 源
RUN true\
    &&
    npm install - g cgr nrm\ &&
    cgr use taobao\ &&
    npm i - g serve @vue / cli @vue / cli - service - global\
commitizen conventional - changelog - cli eslint - plugin - vue\
npm - check - updates npx

# 创建工作目录
RUN mkdir / workspace

WORKDIR / workspace

# 添加权限, 755 或者 a + x
RUN chmod - R 755 / workspace

VOLUME / workspace

EXPOSE 8080
EXPOSE 5000
```

在项目文件夹下创建 `docker-compose.yml`

``` yaml
version: '3'
services:
    dev:
    image: vueimage: 1.0
ports:

    - "8080:8080" # 映射端口, 将本机的 8080 端口映射到 docker 的 8080 端口, npm run serve

restart: always # 每次开启 docker 的时候重启服务, 适用于 MySQL、 nginx 类似于这种的服务
volumes:

    - ./:/workspace # 将执行命令的目录映射到 docker 容器的 /workspace目录

stdin_open: true # 命令行输入映射, 不进行设置会导致无法正常进入 docker
tty: true # 使用TTY模式（ pseudo - TTY）。 若要使用bash, 则必须设置该选项。 若不设置该选项, 则可以输入命令, 但不显示shell
web:
    image: vueimage:1.0
ports:

    - "5000:5000" # 映射端口, 将本机的 5000 端口映射到 docker 的 5000 端口, npm run build && serve dist\

restart: always # 每次开启 docker 的时候重启服务, 适用于 MySQL、 nginx 类似于这种的服务
volumes:

    - ./:/workspace # 将执行命令的目录映射到 docker 容器的 /workspace目录

stdin_open: true # 命令行输入映射, 不进行设置会导致无法正常进入 docker
tty: true # 使用TTY模式（ pseudo - TTY）。 若要使用bash, 则必须设置该选项。 若不设置该选项, 则可以输入命令, 但不显示shell
```

在项目文件夹下执行(编译一个镜像容器出来)

``` bash
docker build -t vueimage:1.0 .
```

编译完成之后执行, 用以启动容器

``` bash
docker-compose up -d
```

使用 `docker ps` 查看 容器的 id, 之后使用以下命令进入容器

``` bash
docker exec -it [docker_id] zsh
```

确认当前目录已经挂载了当前的项目目录, 使用 `pwd` 确认当前的目录为 `/workspace` , 执行安装依赖命令并且启动服务

``` bash
yarn install
yarn serve
```

服务启动之后就在浏览器中访问 `http://127.0.0.1:8080`
经过上面的配置, 我们就可以在主机上使用喜欢的编辑器编辑代码, 通过docker执行。但是使用 `yarn serve` 命令进行开发时, 会发现一个问题: 在编辑器中编辑文件后, docker无法监听到文件的修改, 从而重新编译、刷新页面；我们可以采用 webpack 的轮询功能解决: 

关于 webpack 的轮询功能 => [webpack](https://www.webpackjs.com/configuration/watch/)

 `vue.config.js`

``` javascript
module.exports = {
    devServer: {
        watchOptions: {
            aggregateTimeout: 300, // 当第一个文件更改,会在重新构建前增加延迟。这个选项允许 webpack 将这段时间内进行的任何其他更改都聚合到一次重新构建里。以毫秒为单位: 
            poll: 1000 // 通过传递 true 开启 polling,或者指定毫秒为单位进行轮询。
        }
    }
}
```

到此已经全部配置完毕, 重新 `yarn serve` 然后就可以愉快的写代码了

### gitlab

#### 英文官方原版

在项目文件夹下创建 `docker-compose.yml`

``` yaml
version: '3'

services:
  gitlab:
    container_name: gitlab
    image: gitlab/gitlab-ce
    restart: always
    hostname: 'gitlab-en'
    environment:
      GITLAB_OMNIBUS_CONFIG: |
        # external_url 'http://gitlab.example.com:9090'
    ports:

      - '80:80'
      - '443:443'
      - '2224:22'

    volumes:

      - ./en/config:/etc/gitlab
      - ./en/logs:/var/log/gitlab
      - ./en/data:/var/opt/gitlab

 
```

#### 中文社区版

在项目文件夹下创建 `docker-compose.yml`

``` yaml
version: '3'

services:
  gitlab-ce:
    container_name: gitlab-ce
    image: twang2218/gitlab-ce-zh
    restart: always
    hostname: 'gitlab'
    environment:
      TZ: 'Asia/Shanghai'
      GITLAB_OMNIBUS_CONFIG: |
        # pages_external_url = "http://gitlab.demo.io/"
        gitlab_rails['time_zone'] = 'Asia/Shanghai'
        gitlab_pages['enable'] = true
    ports:

      - '80:80'
      - '443:443'
      - '2224:22'

    volumes:

      - ./cn/config:/etc/gitlab
      - ./cn/logs:/var/log/gitlab
      - ./cn/data:/var/opt/gitlab

```

### 拓展

#### 重置普通用户的密码

进入容器之后执行以下命令

``` bash
$ gitlab-rails console production # 进入gitlab管理控制台
irb(main):001:0> user=User.where(name: "[user_name]").first # 将 [user_name] 替换成用户名
irb(main):001:0> user.password=12345678
irb(main):001:0> user.password_confirmation=12345678 
irb(main):001:0> user.save!
irb(main):001:0> quit
```

#### 重设管理员密码

进入容器之后执行以下命令

``` bash
$ gitlab-rails console production # 进入gitlab管理控制台
irb(main):001:0> user=User.where(id: 1).first # 将 [user_name] 替换成用户名
irb(main):001:0> user.password=12345678
irb(main):001:0> user.password_confirmation=12345678 
irb(main):001:0> user.save!
irb(main):001:0> quit
```

## 私有 npm 代理注册表

在项目文件夹下创建 `docker-compose.yml`

``` yaml
version: '3'

services:
  npm-private:
    container_name: npm-private
    image: verdaccio/verdaccio
    restart: always
    ports:
      - '4873:4873'
    
```

### 拓展

::: tip 提示
windows 环境下不想使用 docker 的解决方案, 使用 `forever` + `verdaccio`
:::

安装依赖

``` bash
npm i -g forever verdaccio
```

修改 `verdaccio` 的配置文件, 参考文件路径为 `C:/Users/Administrator/AppData/Roaming/verdaccio/config.yaml` （必须修改, 否则只能本机访问）, 在该文件内添加

``` txt
listen: 0.0.0.0:4873
```

编写 `bat` windows 脚本

开始脚本 `forever-npm-start.bat`

```bat
@echo off
C:
cd C:\Users\Administrator\AppData\Roaming\npm\node_modules\verdaccio\build\lib 
forever start cli.js
exit
```

结束脚本 `forever-npm-stop.bat`

```bat
@echo off
C:
cd C:\Users\Administrator\AppData\Roaming\npm\node_modules\verdaccio\build\lib
forever stop cli.js
exit
```

### CSS预处理SASS的默认实现将迁移到 Dart Sass

Web开发中流行的CSS预处理 - Sass 14, 于2016年11月1日在博客正式对外公布了alpha版本的Dart Sass 32项目, 即他们使用Dart对Sass进行了重写。对于这次迁移/重写, Sass项目组做了以下说明: 

关于 [github:dart-sass](https://github.com/sass/dart-sass) [dart-sass](https://sass-lang.com/dart-sass)

因为使用 `node-sass` 无论是在 Windows 环境下还是在 docker 的 liunx 环境下都需要依赖 `gcc` 和 `python2.7` , 有的时候会出现编译错误（指的是在执行 `npm install` 安装 `node-sass` 的时候由于部分库不齐备或者是编译原因导致 `node-sass` 无法正常安装）, 所以使用 `dart-sass` 来代替 `node-sass` 是个不错的选择, 至少目前在自测中是没有出现编译问题的, 至于兼容性问题, GitHub 文档中也给出承诺 [Browser Compatibility](https://github.com/sass/dart-sass#browser-compatibility)

### 为什么重写Sass？

Sass 的主要实现有 Ruby Sass 7 和 LibSass 10（node-sass 底层使用的是 LibSass）, 它们都有各自的优缺点。Ruby Sass 的实现语言是高级语言 Ruby, 更容易迭代, 但存在运行速度慢, 不易安装的缺点。LibSass 虽然速度快, 但它的开发语言是 C/C++, 迭代速度慢, 无法快速地添加各种功能。

### 为什么使用Dart？

Dart 的运行速度是真的快, 对于大型样式文件, Dart Sass 的处理速度是 Ruby Sass 的5~10 倍, 且只比 LibSass 慢 1.5 倍左右。同时, Dart 是一门具备静态类型的动态语言, 对比 C/C++ 甚至是 Ruby, Dart 相对更容易上手且代码也更易于编写和维护。此外, Dart 具备编译输出为 JavaScript 的能力, 使得 Dart Sass 可以兼容 NodeJS 平台。

### 各种实现的后续规划？

LibSass 作为目前性能最好的 Sass 实现, 后续将继续维护, 只是它不再需要背负快速添加各种新功能的压力。Ruby Sass 目前也会同步维护, 但在无人接手的情况下, 它将逐渐边缘化。Sass项目组后续主要精力将转移到 Dart Sass, 其稳定版本预计在 2017 年初放出。

### git commit

现在由于我们在 docker 内部安装项目依赖, 也就是说我们当前的环境变成了 liunx, 相关的依赖包也就是 liunx 的依赖包, 所以在 windows 环境下执行 `git commit` 的时候会报依赖找不到, 那是因为安装了 liunx 的依赖包, 两个平台的安装包是不一样的

那么在 docker 下执行 `git commit` 的时候又会失败, 因为 commit 的时候 git 需要知道当前的 commit 作者是谁, 这时候不必全局配置 commit 的 config 信息, 只需在项目目录下执行

``` bash
git config   user.name 'XXX'
git config   user.email 'XXX' 
# or
git config  --local  user.name 'XXX' 
git config  --local user.email 'XXX'
```

设置完用 `git config --list` 进行检验, 之后的 commit 操作就在 docker 容器下执行就行了
