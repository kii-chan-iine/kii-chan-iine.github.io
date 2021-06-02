---
author: 星火燎原@vxhly
title: lerna 的使用指南
categories: [nodejs]
tags: [Node.js]
date: 2020-10-25 14:49:11
---

<Boxx changeTime="10000"/>

::: tip 前言
从 GitHub 上找到了比较多的仓库开始采用 lerna 来管理仓库的多包依赖, 也逐渐发现 lerna 的管理模式确实比较好, so 现贴上我的填坑笔记。
:::
<!-- more -->

## 参考资料

先贴上我的参考资料来源

1. lerna 官网: [lerna.js.org/](https://lerna.js.org/)
2. lerna 仓库：[github.com/lerna/lerna/](https://github.com/lerna/lerna/)

当然也得感谢前人栽的树

1. [lerna的基础使用](https://www.jianshu.com/p/8b7e6025354b)
2. [lerna入门篇](https://www.jianshu.com/p/63ec67445b0f)

## 什么是 lernaJs

lernaJs 是由 Babel 团队出的一个多包管理工具, 是 GitHub 上面开源的一款 js 代码库管理软件, 用来对一系列相互耦合比较大、又相互独立的 js git 库进行管理。因为 Babel 包含很多子包, 以前都是放在多个仓库里的, 管理比较困难, 特别是有调用系统内包的时候, 发布比较麻烦。所以为了能更好更快的跨包管理, babel 推出了 lernaJs, 使用了 monorepo 的概念, 现在 React, Babel, Angular, Jest 都在使用这个工具来管理包。

对于一些功能比较全的库, 我们往往会把各个小功能拆分成独立的npm库, 他们直接有比较强的依赖关系。可以解决各个库之间修改混乱、难以跟踪的问题。lerna可以优化这种情形下的工作流。

简单的说, lernaJs 是用于管理具有多个包的 JavaScript 的项目工具。(官网原话为：`Lerna is a tool that optimizes the workflow around managing multi-package repositories with git and npm.`)

引入 lerna 后, 我们可以解决以下这些问题

1. 自动解决 packages 之间的依赖关系
2. 通过 `git` 检测文件改动, 自动发布
3. 根据 `git` 提交记录, 自动生成 `CHANGELOG`

### 代码库结构

lerna 管理的库文件结构类似如下这样

```text
my-lerna-repo
  |
  |- packages
  |    |
  |    |- package-1
  |    |    |
  |    |    └─ package.json
  |    |
  |    └─ package-2
  |         |
  |         └─ package.json
  |
  |- package.json
  |
  └─ lerna.json
```

### lerna 主要做了什么

1. 通过 lerna 的命令 `lerna bootstrap` 将会把代码库进行 link。
2. 通过 `lerna publish` 发布最新改动的库

## lerna 如何工作的

默认 lerna 有两种管理模式, 固定模式和独立模式

### 固定模式

固定模式, 通过 lerna.json 的版本进行版本管理。当你执行 `lerna publish` 命令时, 如果距离上次发布只修改了一个模块, 将会更新对应模块的版本到新的版本号, 然后你可以只发布修改的库。

这种模式也是 Babel 使用的方式。如果你希望所有的版本一起变更, 可以更新 minor 版本号, 这样会导致所有的模块都更新版本。

### 独立模式

独立模式, init 的时候需要设置选项 `--independent` 。独立模式允许管理者对每个库单独改变版本号, 每次发布的时候, 你需要为每个改动的库指定版本号。这种情况下, `lerna.json` 的版本号不会变化了, 默认为 `independent` 。

## lerna.json 解析

``` json
{
  "version": "1.1.3",
  "npmClient": "npm",
  "command": {
    "publish": {
      "ignoreChanges": [
        "ignored-file",
        "*.md"
      ]
    },
    "bootstrap": {
      "ignore": "component-*",
      "npmClientArgs": ["--no-package-lock"]
    }
  },
  "packages": ["packages/*"],
  "useWorkspaces": false
}
```

* `version` , 当前库的版本
* `npmClient` , 允许指定命令使用的client, 默认是 npm, 可以设置成 yarn
* `command.publish.ignoreChanges` , 可以指定那些目录或者文件的变更不会被 publish
* `command.bootstrap.ignore` , 指定不受 bootstrap 命令影响的包
* `command.bootstrap.npmClientArgs` , 指定默认传给 `lerna bootstrap` 命令的参数
* `command.bootstrap.scope` , 指定那些包会受 `lerna bootstrap` 命令影响
* `packages` , 指定包所在的目录
* `useWorkspaces` , 支持使用 yarn workspaces

## 相关命令

### 安装和初始化 lerna 项目

安装 lernajs

``` bash
npm install lerna -g
```

初始化 lernaJs 项目

``` bash
lerna init
```

复制代码执行后, 会自动生成一个由 lerna 管理的 monorepo。

其文件结构如下：

``` text
lerna-repo/
  packages/          // 子包都放在这个目录中
  package.json
  lerna.json         // lerna js的配置文件
```

### 添加子包

环境初始好以后, 首先需要添加一个子包, 添加子包的命令如下

``` bash
lerna create <pkgName>
```

执行这个命令后, 会问一些包名啊, 版本等问题和执行 npm init 后问的差不多, 填完这些问题后包就会自动创建一个子包。子包默认的目录结构如下

### 为子包单独安装依赖

``` bash
lerna add <packageName2> --scope <packageName1>
```

如果你的仓库里的子包需要互相依赖也可用以上命令来安装

例如：子包1 为 '@demo/packageName1', 子包2 为 '@demo/packageName2', 要为子包1 安装子包2 的依赖, 你可以这样做, 注意要在项目的根目录执行以下命令

``` bash
lerna add @demo/packageName2 --scope @demo/packageName1
```

### 为仓库安装依赖

``` bash
lerna bootstrap
```

以上命令可以为仓库安装所有子包的依赖

### 发布和更新 package

``` bash
lerna publish
```

发布当前项目
他创建一个新的 release, 生成新的版本, 执行 `git commit/tag` 并发布到npm

发布项目里的每个模块
执行 `lerna updated` 确定是否需要发布
假如需要发布 给 lerna.json 版本号做自增
更新 package.json 里的版本号至最新
为新版本更新 dependencies
为新版本创建一个 git commit 和 tag
发布更新项目到 npm
一次发布所有 packages, 删除 `lerna-temp`  `tags` 和增加 tags 到 latest
通过在 package.json 中标记 `"private": true` , 将使该包不发布

::: warning 注意
以@开头包的发布问题
:::

发布 package 的名字如果是以@开头的, 例如 @feu/tools, npm默认以为是私人发布, 需要使用 `npm publish --access public` 发布。但是 `lerna publish` 不支持该参数, 解决方法参考: [issues](https://github.com/lerna/lerna/issues/914)

方案参考:

``` json
// package.json
{
 "name": "@feu/tools",
 "publishConfig": {
   "access": "public" 	// 如果该模块需要发布, 对于scope模块, 需要设置为publish, 否则需要权限验证
  }
}
```

#### options

##### canary

::: tip
可以用来独立发布每个commit, 不打tag
:::

``` bash
lerna publish --canary
# 1.0.0 => 1.0.1-alpha.0+${SHA} of packages changed since the previous commit
# a subsequent canary publish will yield 1.0.1-alpha.1+${SHA}, etc

lerna publish --canary --preid beta
# 1.0.0 => 1.0.1-beta.0+${SHA}

# The following are equivalent:
lerna publish --canary minor
lerna publish --canary preminor
# 1.0.0 => 1.1.0-alpha.0+${SHA}
```

### 删除子包中的依赖项

``` bash
lerna clean
```

从所有软件包中删除 `node_modules` 目录。
`lerna clean` 尊重 `--ignore` 和 `--scope` 标志（请参阅标志）。
