---
author: 星火燎原@vxhly
title: 浅谈 Node.js 中的 npm
categories: [nodejs]
tags: [Node.js, Liunx]
date: 2016-08-26 15:30:10
---

<Boxx changeTime="10000"/>

::: tip 前言
主要介绍 npm 中的各个选项的作用以及其区别。npm 依赖于 Node.js, 童鞋们要注意你的系统是否有 Node.js 环境。
:::
<!-- more -->

## npm 是什么

::: tip 简介
NPM 的全称是 Node Package Manager, 是随同 Node.js 一起安装的包管理和分发工具, 它很方便让 JavaScript 开发者下载、安装、上传以及管理已经安装的包。
:::

## npm install 安装模块

::: tip 提示
安装包, 默认会安装最新的版本, 最后所有的依赖会存放在 `node_modules` 目录下的, 如: 
::: 

``` bash
$ npm install gulp
```

### 全局安装

::: tip 提示
只需加上 `-g` 或者 `--global` 参数, 就可以实现全局安装, 如: 
:::

``` bash
$ npm install gulp -g
```

### 安装指定版本

``` bash
$ npm install gulp@3.9.1
```

## -S, --save

### 添加模块命令

``` bash
$ npm install [module-name] --save
$ npm install [module-name] -S
```

### 作用

::: tip 作用
自动把模块和版本号添加至 `package.json` 的 `dependencies` （生产阶段的依赖）部分, 避免了手动添加。
::: 

`package.json` 文件的 `dependencies` 字段: 

``` json
"dependencies": {
  "[module-name]": "^[version]"
}
```

## -D, --save-dev

### 添加模块命令

``` bash
$ npm install [module-name] --save-dev
$ npm install [module-name] -D
```

### 作用

::: tip 作用
自动把模块和版本号添加至 `package.json` 的 `devDependencies` （开发阶段的依赖）部分, 避免了手动添加。
::: 

`package.json` 文件的 `devDependencies` 字段: 

``` json
"devDependencies": {
  "[module-name]": "^[version]"
}
```

## -O, --save-optional

### 添加模块命令

``` bash
$ npm install [module-name] --save-optional
$ npm install [module-name] -O
```

### 作用

::: tip 作用
自动把模块和版本号添加至 `package.json` 的 `optionalDependencies` （可选阶段的依赖）部分, 避免了手动添加。
::: 

`package.json` 文件的 `optionalDependencies` 字段: 

``` json
"optionalDependencies": {
  "[module-name]": "^[version]"
},
```

## -E, --save-exact

### 添加模块命令

``` bash
$ npm install [module-name] --save-exact
$ npm install [module-name] -E
```

### 作用

::: tip 作用
自动把模块和版本号添加至 `package.json` 的 `dependencies` （精确安装指定模块版本）部分, 避免了手动添加。需要注意的是: `package.json` 文件的 `dependencies` 字段, 版本号中的 `^` 消失了
:::

`package.json` 文件的 `dependencies` 字段

``` json
"dependencies": {
  "[module-name]": "[version]"
}
```

## npm 其他命令

### 查看依赖包的所有的版本号

``` bash
$ npm info [package-name] versions
```

### 创建 package.json 文件

::: tip 提示
安装包的信息可保持到项目的 `package.json` 文件中, 以便后续的其它的项目开发或者他人合作使用, 也说 `package.json` 在项目中是必不可少的。
:::

``` bash
$ npm init
```

### 安装模块

::: tip 提示
模块的依赖都被写入了 `package.json` 文件后, 他人打开项目的根目录（项目开源、内部团队合作）, 时, 只需一条命令就可以安装 `package.json` 中的依赖。
:::

``` bash
$ npm install
```

::: tip 提示
加上 `--save` 会安装 `package.json` 中 `dependencies` 里的所有模块。
:::

``` bash
$ npm install --save
```

### 卸载模块

想要卸载模块也很简单。

``` bash
$ npm uninstall [module-name]
```

::: tip 提示
以上写法并不会改变 `package.json` 文件, 使用 `-S | --save | -D | --save-dev | -O | --save-optional` 选项即可删除 `package.json` 文件中的依赖。<br>
如使用 `--save` 选项, 将会删除 `package.json` 文件 `dependencies` 中的该模块名和版本号。
::: 

``` bash
$ npm uninstall [module-name] --save
```

### 更新模块

::: tip 提示
从服务器下载最新版本的模块。需要注意的是, 最新版本的模块并不是最稳定的, 模块还是选择相对稳定的版本较好。支持 `-g` 选项。
:::

``` bash
$ npm update [-g] [module-name]
```

### 检查模块是否已经过时

::: tip 提示
此命令会列出所有已经过时的包, 可以及时进行包的更新.
:::

``` bash
$ npm outdated
```

### 启动模块

``` bash
$ npm start [-- <args>]
```

::: tip 提示
该命令写在 `package.json` 文件 `scripts` 的 `start` 字段中, 可以自定义命令来配置一个服务器环境和安装一系列的必要程序, 如:
:::

``` json
"scripts": {
  "start": "gulp -s"
}
```

::: tip 提示
如果 `package.json` 文件没有设置 `start` , 则将直接启动 `node server.js`
:::

### 停止模块

``` bash
$ npm stop [-- <args>]
```

### 重新启动模块

``` bash
$ npm restart [-- <args>]
```

### 更新模块

可以安装全局插件

``` bash
$ npm install -g npm-check-updates
```

查看最新的版本

``` bash
$ npm-check-updates
$ ncu # 为上一条指令的缩写
```

升级至最新版本

``` bash
$ ncu -a
```

在使用下载最新的安装包

``` bash
$ npm update
```

当然, 如果所有包都是最新版本, 执行 ncu 后会提示: 

``` bash
All dependencies match the latest package versions :)
```

### 查看包的安装路径

::: tip 提示
输出 node_modules 的路径, 支持 `-g` 选项。
:::

``` bash
$ npm root [-g]
```

### 管理 npm 的配置路径

::: tip 提示
对于 config 这块用得最多应该是设置代理, 解决 npm 安装一些模块失败的问题
:::

``` bash
$ npm config set <key> <value> [-g|--global]
$ npm config delete <key>
$ npm config list
```

### 查看命令的帮助

``` bash
$ npm help <command>
```

## 区别

1. `--save-dev` 是你开发时候依赖的东西； `--save` 是你发布之后还依赖的东西。
2. `--save-optional` 是可选依赖, 非必下； `--save-exact` 可以指定精确的版本号, 也就是说当发布了新版本, 系统不会系在最新版本的模块, 只会下载该版本的模块。
3. `npm install` 下载依赖时会安装 `package.json` 中 `dependencies` 和 `devDependencies` 里的所有模块。
4. `npm install --save` 下载依赖时只会安装 `package.json` 中 `dependencies` 里的所有模块。
5. 比如, 你写 `ES6` 代码, 如果你想编译成 `ES5` 发布那么 `babel` 就是 `devDependencies` 。
6. 如果你用了 `jQuery` , 由于发布之后还是依赖 `jQuery` , 所以是 `dependencies` 。
7. 最常用的还是 `npm` 中的 `--save` 和 `--save-dev` 选项。 `--save` 和 `--save-dev` 选项。
