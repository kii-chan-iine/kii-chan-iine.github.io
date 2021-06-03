---
author: 星火燎原@vxhly
title: 基于 lerna + yarn workspaces 的 monorepo 项目实践
categories: [nodejs]
tags: [Node.js]
date: 2020-10-26 12:13:34
---

<Boxx changeTime="10000"/>

::: tip 前言
观察到 GitHub 上面有很多的开源项目使用到了 lerna 和 yarn workspaces 来管理项目中的多包依赖, 好奇起它的实现过程, 于是便写下本篇博文来记录我的填坑过程。
:::
<!-- more -->

## 参考资料

先贴上我的参考资料来源

1. lerna 官网: [lerna.js.org/](https://lerna.js.org/)
2. lerna 仓库：[github.com/lerna/lerna/](https://github.com/lerna/lerna/)
3. yarn workspaces：[classic.yarnpkg.com/zh-Hans/docs/workspaces/](https://classic.yarnpkg.com/zh-Hans/docs/workspaces)

当然也得感谢前人栽的树

1. [Lerna多包管理的搭建指南](https://juejin.im/post/6844903954329894920)
2. [lerna的基础使用](https://www.jianshu.com/p/8b7e6025354b)

## 背景

团队内多项目开发经常会遇到以下几个问题

1. 新建项目需要重新配置各种工具, 繁琐, 虽然也可以用脚手架生成, 但后期涉及到依赖的升级也麻烦
2. 项目内的依赖互相引用虽然有 yarn/npm link, 但还是影响开发体验
3. babel 等配置希望可以多项目共享

### 前端开发 package 面临的问题

在最初开开发 package 的时候, 还属于一种刀耕火种的阶段。没有什么自动化的工具。发布 package 的时候, 都是手动修改版本号(PS: 当然也可以通过脚本工具来升级版本号)。如果 packages 数量不多还可以接受。但是当数量逐渐增多的时候, 且这些 packages 之间还有依赖关系的时候, 在版本号上的管理无法形成统一, 这就对开发人员来说就很痛苦了。工作不仅繁琐, 而且需要用掉不少时间。

举个例子, 如果你要维护两个 package。分别为 module-1, module-2。

下面是这两个 package 的依赖关系。

``` json
// module-1 package.json
{
  "name": "module-1",
  "version": "1.0.0",
  "dependencies": {
    "module-2": "^1.0.0"
  }
}
```

``` json
// module-2 package.json
{
  "name": "module-2",
  "version": "1.0.0"
}
```

在这样的环境下, module-1 是依赖 module-2 的。如果 module-2 有修改, 需要发布。那么你的工作有这些。

修改 module-2 版本号, 发布。
修改 module-1 的依赖关系, 修改 module-1 的版本号, 发布。

这还仅仅只有两个 package, 如果依赖关系更复杂, 大家可以想想发布的工作量有多大。

## 什么是 lernaJs

lernaJs 是由 Babel 团队出的一个多包管理工具, 是 GitHub 上面开源的一款 js 代码库管理软件, 用来对一系列相互耦合比较大、又相互独立的 js git 库进行管理。因为 Babel 包含很多子包, 以前都是放在多个仓库里的, 管理比较困难, 特别是有调用系统内包的时候, 发布比较麻烦。所以为了能更好更快的跨包管理, babel 推出了 lernaJs, 使用了 monorepo 的概念, 现在 React,Babel,Angular,Jest 都在使用这个工具来管理包。

对于一些功能比较全的库, 我们往往会把各个小功能拆分成独立的 npm 库, 他们直接有比较强的依赖关系。可以解决各个库之间修改混乱、难以跟踪的问题。lerna 可以优化这种情形下的工作流。

## 什么是 yarn workspaces

Workspace 能更好的统一管理有多个项目的仓库, 既可在每个项目下使用独立的 package.json 管理依赖, 又可便利的享受一条 yarn 命令安装或者升级所有依赖等。更重要的是可以使多个项目共享同一个 node_modules 目录, 提升开发效率和降低磁盘空间占用。
一句话总结就是可以大大简化对多个项目的统一管理。
很多知名的开源项目也使用了 Yarn Workspace, 如 vue、react、jest 等
使用 lerna 结合 yarn workspaces
各个 package 理论上都是独立的, 所以每个 package 都维护着自己的 dependencies, 而很大的可能性, package 之间有不少相同的依赖, 而这就可能使 install 时出现重复安装, 使本来就很大的  node_modues 继续膨胀。简单的说如果不使用 yarn workspaces 的话, 会导致在安装依赖时, 子包重复安装依赖项。使用 yarn workspaces 时, 我们就可以将这些依赖提至到工作根区。

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

### lerna 的更多知识

::: tip 提示
这边不多阐述 lerna 的相关命令以及其的相关知识点, 有兴趣的贴上地址 [lerna 的使用指南](https://vxhly.github.io/views/nodejs/usage-lerna.html)
:::

## 什么是 monorepo

monorepo 是比较于 multi-package 的。multi-package 就是建立多个仓库, 每个包建立一个仓库。而 monorepo 是建立一个仓库, 多个包都在这个仓库中管理, 这样可以有两个好处：

1. 各个包之间的沟通更加的方便, 如果 multi-package 的话, 系统内其中一个包修改, 需要单独发版, 而且引用这个包的其他包都需要发版。使用 lerna 的话可以自动管理这些包的发版, 很方便。
2. 一些可以共用的配置, 比如 eslint,babel,rollup 等, 可以统一的管理这些开发配置。

## 代码库的结构

::: tip 提示
这边节选 [vue-next](https://github.com/vuejs/vue-next) 的项目结构来讲解实现过程
贴上我已经实现好的 [my-lerna-repo](https://github.com/vxhly/my-lerna-repo)
:::

创建目录结构如下

```text
my-lerna-repo
  |
  |- packages
  |
  |- package.json
  |
  └─ lerna.json
```

当然你也可以通过 `lerna` 的命令来创建

``` bash
lerna init my-lerna-repo
```

修改你的 `lerna.json` 来支持 yarn workspaces

``` json
{
  "npmClient": "yarn",
  "packages": [
    "packages/*"
  ],
  "version": "1.0.0",
  "useWorkspaces": true
}
```

还需要在 `package.json` 中添加以下配置来开启 yarn workspaces

``` json
{
  "private": true,
  "workspaces": [
    "packages/*"
  ]
}
```

接着就是安装依赖项, 你可以通过以下命令来安装

``` bash
lerna bootstrap
```
