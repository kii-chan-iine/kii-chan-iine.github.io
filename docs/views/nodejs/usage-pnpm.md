---
author: 星火燎原@vxhly
title: 使用 pnpm 释放你的磁盘空间
categories: [nodejs]
tags: [Node.js]
date: 2020-08-19 14:14:13
---

<Boxx changeTime="10000"/>

::: tip 前言
主要介绍 pnpm 的使用, 以及如何使用 pnpm 解决 node 依赖包庞大的问题。pnpm 依赖于 Node.js, 童鞋们要注意你的系统是否有 Node.js 环境。
:::
<!-- more -->

## pnpm 是什么

先上官方的文档 [pnpm 的官方文档](https://pnpm.js.org/en/)

::: tip 简介
官方文档中有这么一句：

pnpm uses a content-addressable filesystem to store all files from all module directories on a disk.

中文翻译过来的大概意思是：

pnpm 使用内容可寻址文件系统将磁盘上所有模块目录中的所有文件存储在磁盘上。
:::

简单的说

pnpm 是一个更为高效的 node.js 的包管理器, 类似于 npm 和 yarn。pnpm 结合了 npm 和 yarn 的优点, 它会在本地进行缓存依赖项, 加快了下载速度

pnpm 的原理是会为当前项目下的依赖项在当前磁盘下创建一份 store, 这个store 存放的信息为依赖项的版本号信息, 用于 pnpm 二次下载时进行陈旧版本检测使用的

pnpm 还会为当前项目的 node_modules 下创建一个 .pnpm 的一个依赖项存储位置, 在其他依赖项中如果引用了相同的依赖项它会采用硬链接的形式链接到这个位置

这里寻找到 pnpm 的缺点是：

* pnpm 在每次 install 的时候, 都会去检测陈旧依赖项的版本号信息, 如果项目的依赖项版本号并未进行更新的话, 这一过程无非是比较耗时的
* 不得与 npm 和 yarn 进行混用, 例如使用了 pnpm 安装了一个依赖项, 在使用 npm 或 yarn 进行安装依赖项时并不会将依赖项安装到指定的目录下, 这就违背了 pnpm 的设计初衷
* 虽然不得与 npm 和 yarn 进行混用, 但是仍然可以使用 npm 或者 yarn 执行项目配置的命令, 这个并不会受到影响

以上的这些缺点, 我们可以通过对 pnpm 的简单的配置规避掉上方的问题

## npm 与 yarn 与 pnpm 的比较

### npm

1）同步下载：安装多个包时会使用从事件队列中读取事件的同步下载模式进行安装依赖；

2）本地缓存：npm 会在安装依赖后产生本地缓存, 加快再次重新安装时的下载速度；

3）层级关系复杂：npm 在安装多个依赖包时, 会先遍历各个依赖包, 生成这些包关系树, 这也是npm安装时时间慢的一大主要原因；

4）不稳定, 抛出错误不明显：当某个安装包版本变更后, 在终端编译时会造成抛出错误, 这也是有时候同一份代码你能用别人用不了的原因, 更致命的是, 安装依赖出错时, 抛出错误后并不会停止执行后面的事件, 因此, 如果后面的安装包依赖于出错的安装包, 会报出一堆错误, 你甚至无法得知究竟是哪个包出了问题, 因此大部分时间都是整个删除整个重装 node_modules; 

5) 在 npm5 中, 官方解决了因为版本不一致造成崩溃的问题, 但是需要用户手动执行 npm shrinkwrap 命令, 从而生成一个 package-lock.json 文件, 该文件包含了包的具体信息和版本等内容, 在安装时会首先进行检测, 安装对应版本。

6）在线安装：需要用到互联网

### yarn 

1) 异步下载：多个安装包同时下载, 解决下载慢的问题；

2）本地缓存：同 npm 一样, 会进行本地缓存, 再次下载先从本地获取

3）简化命令：对比 npm, 命令更加简短, 语义更加明确

4）包依赖关系扁平化：npm 会按依赖层级来读取底层包名称, 当层级够深时, windows 下会造成调用数量超限的报错, 而 yarn 则直接取到底层包名称。

5）指定版本安装, 更加稳定：npm 的稳定性一直是一个问题, yarn 解决了这个问题：在使用 yarn 安装时, 会自动生成一个叫做 yarn.lock 的文件, 该文件包含了当前依赖包的版本信息和对应的校验规则, 在安装新的包之后, 会自动更新, 解决了版本库不对应造成的编译错误。

6）提供离线模式：即使没有互联网依旧可以使用

### pnpm 

1) 运行速度超过 npm 和 yarn

2）采用硬链接和符号链接避免复制本地缓存源文件, 解决了 yarn 的一大弱点

3) 提供离线模式：即使没有互联网依旧可以使用

4) 异步下载：多个安装包同时下载, 解决下载慢的问题；

## pnpm 的简单使用

首先我们先全局安装 pnpm

``` bash
npm i -g pnpm
```

pnpm 的一些命令与 npm 的都大同小异, 下方例举几个常用的命令, 具体的不做过多的阐述

### install

安装项目的依赖项

``` bash
$ pnpm install
```

### update

更新项目的依赖项

``` bash
$ pnpm update
```

### add

安装项目的某一个依赖项

``` bash
$ pnpm add lodash
```

### store

向本地的存储中安装一个依赖包

``` bash
$ pnpm store add lodash
```

## .npmrc 文件解析

使用过 node 的同学都知道, npm 会在全局创建一个 .npmrc 文件, 这个文件通常用于存放 npm 的相关配置信息, 例如 npm 的下载软件源

```text
registry=https://registry.npm.taobao.org/
```

在 pnpm 中提供了几个配置项, 而这几个配置项正是 node 工程下实现多个项目共用 node_modules 的核心重点（pnpm 的 .npmrc 的具体官方说明：pnpm 中.npmrc 的配置项）
下面例举可能会使用到的配置项

### virtual-store-dir

这个配置项的作用是用于存放工程依赖项的实际链接位置, 也是依赖包的缓存目录, 默认值为 node_modules/.pnpm, 即为当前工程下的 node_modules/.pnpm

上原文的解释

::: tip 原文
Added in: v4.1.0

* Default: *node_modules/.pnpm*
* Types: *path*

The directory with links to the store. All direct and indirect dependencies of the project are linked into this directory.

This is a useful setting that can solve issues with long paths on Windows. If you have some dependencies with very long paths, you can select a virtual store in the root of your drive (for instance `C:\my-project-store` ).

Or you can set the virtual store to `.pnpm` and add it to `.gitignore` . This will make the stacktraces nicer as paths to dependencies will have one directory less.

*NOTE*: the virtual store cannot be shared between several projects. Every project should have its own virtual store.
:::

### store-dir

这个配置项的作用是存放所有依赖项的的所有版本号信息, 用于pnpm进行陈旧性检测, 默认值为 ~/.pnpm-store（在 linux 下指向的是用户目录 ~, 在 Windows 下指向的是当前的磁盘根目录）

上原文的解释

::: tip 原文
Added in: v4.2.0 (renamed from store)

* Default: *~/.pnpm-store*
* Type: *path*

The location where all the packages are saved on the disk.

The store should be always on the same disk on which installation is happening. So there will be one store per disk. If there is a home directory on the current disk, then the store is created in `<home dir>/.pnpm-store` . If there is no homedir on the disk, then the store is created in the root. For example, if installation is happening on disk D then the store will be created in `D:\.pnpm-store` .

It is possible to set a store from a different disk but in that case pnpm will copy, not link, packages from the store. Hard links are possible only inside a filesystem.
:::

### package-import-method

这个配置项的作用是用于指定依赖包导入的方式, 默认值为 auto, 会自动判断是否需要从网上的源进行下载操作

上原文的解释

::: tip 原文
Added in: v1.25.0

* Default: *auto*
* Type: *auto*, *hardlink*, *copy*, *clone*

Controls the way packages are imported from the store.

*auto* - try to clone packages from the store. If cloning is not supported then hardlink packages from the store. If neither cloning nor linking is possible, falls back to copying
*hardlink* - hardlink packages from the store
*copy* - copy packages from the store
*clone* - clone (aka copy-on-write or reflink) packages from the store
::: 

### shamefully-hoist

这个配置项允许我们在安装依赖项时, 默认值为 false, node_modules 目录像 npm 的一样, 因为 pnpm 默认生成的 node_modules 是属于半严格的树形结构

上原文的解释

::: tip 原文
Added in: v1.34.0 (Renamed from `shamefully-flatten` in v4.0.0)

* Default: *false*
* Type: *Boolean*

By default, pnpm creates a semistrict `node_modules` . It means that your dependencies have access to undeclared dependencies but your code does not. With this layout, most of the packages in the ecosystem work with no issues. However, if some tooling only works when the hoisted dependencies are in the root of `node_modules` , you can set this config to `true` .
::: 

### link-workspace-packages

这个配置项允许我们在安装依赖项的时候, 将所有的子项目依赖项安装到项目的根目录下, 默认值为 true

上原文的解释

::: tip 原文
Added in: v2.14.0

* Default: *true*
* Type: *true*, *false*, *deep*

When `true` , locally available packages are linked to `node_modules` instead of being downloaded from the registry. This is very convenient in a monorepo. If you need local packages also be linked to subdependencies, you can use the `deep` setting (since v5).

When `false` , packages are downloaded and installed from the registry. However, workspace packages can still be linked by using the `workspace:` range protocol. e.g. `pnpm add batman@workspace:*`
*Usage*

Create a `.npmrc` file in the root of your monorepo with the following content:

``` text
link-workspace-packages = true
```

Create a `pnpm-workspace.yaml` file with the following content:

``` text
packages:
  - '**'
```

Run `pnpm recursive install` .
::: 

### registry

定义了 npm 的下载软件源, 需要注意的是：如果采用私有源的话, 私有源的软件包一定要全, 要和官方源一致（否则在 install 的时候会出现某些依赖项安装失败而出现的问题）

上原文的解释

::: tip 原文
- Default: <https://registry.npmjs.org/>
- Type: *url*

The base URL of the npm package registry.
:::

## 为什么要优化 node_modules

最初的 npm 包管理, 是全局共享的, 类似 java 的 maven 仓库, 但是由于前端的 npm 包是经常修改和重新发布的, 这就涉及到一个包版本问题, 不同的项目有可能依赖不同版本的包, 所以 npm 后续版本把仓库独立到各自工程去了, 这样处理最明显的坏处是：

* 每次都需要安装依赖, 费流量, 网速慢时很费时间
* 浪费磁盘空间, 每个 node_modules 中包含的工具很多, 动辄 200M

而好处是：

* 使用 package.json 安装好之后, node_modules 文件夹中没有版本信息, 删掉 package.json 也不会有任何影响。
* 移动/复制/打包项目比较简单, 对于开发、部署都有好处
* 对于设计 npm 的人来说, 这是最省事的包依赖方法。这就好比 maven 安装依赖之后自动将jar 包安装到项目的lib里面。
* 随意改代码。安装在 node_modules 里面的东西, 你可以随便改, 无需担心对其它项目的影响。在 Java 中使用 maven 管理项目时, 如果想要定制某个库, 就需要更改这个库的源代码, 
* 这时就需要把这个库的源代码复制到项目中, 跟 node_modules 是一个道理。npm 的设计者大概认为：前端都是经常修改库的源代码的。

## 解决思路和方案

我们采用 pnpm 的最终就是为了能够解决多个工程下共用一个 node_modules, 并且能够提升我们的 install  速度, 这里采用到的是 pnpm 中  pnpm workspace 这一特性, 这一特性允许我们将子工程的依赖项全部安装到同一个目录下, 而且会保留子工程中的所依赖的不同版本

准备一个项目工程, 工程目录结构如下

``` text
--- test-project 
  |
  |- common
  |
  |- project1
  |
  |- project2
  |
  |- project3
```

依据 pnpm 中 [pnpm workspace](https://pnpm.js.org/en/workspaces) 这一特性, 需要我们指定哪一个目录是 pnpm 工作区的根目录, 还必须在这个根目录下提供 pnpm-workspace.yaml, 注意.npmrc文件不是必须的, 如果没有这个文件 pnpm 会按照它默认的配置为我们安装依赖项, 我们需要指定pnpm 的 stor 目录以及virtual-store目录以及其他的一些配置, 所以才创建了这个文件

这边找到的不采用工作区的缺点, 假设我们的子工程下的同一个依赖项有着不同的版本号, 于是在子工程中执行 pnpm install 的时候, pnpm 会将依赖项安装到指定目录下, 但是两个工程的 virtual-store目录 是一致的话, 并不能成功的安装多个版本的依赖项（pnpm 为采用最新的版本依赖项, 而将旧版本的依赖项进行删除, 最后导致旧版本的依赖项链接失败）, 而采用了工作区正好可以解决这一个问题, 它会在工作区的根下创建一个锁文件, 这个锁文件包含了工程下的所有依赖项信息, 每一个工程的依赖项信息都是独立的

我们指定 common 文件夹即是 pnpm  工作区的跟路径, 在这个文件夹里面创建 pnpm-workspace.yaml 和 .npmrc 文件

``` yaml
packages:
  - '../**/' # 允许安装上一级目录下的所有工程的依赖
  - '!../**/node_modules/**' # 不允许安装 上一级目录下 node_modules 下的依赖项
```

然后我们还需要对 `.npmrc` 文件进行一定的配置, 配置如下

``` text
link-workspace-packages = true  # 告诉 pnpm 如何去下载工作区的依赖项
shamefully-hoist = true # 告诉 pnpm 不采用半严格的 node_modules 目录结构
store-dir= .pnpm-store # 所有软件包保存在磁盘上的位置
virtual-store-dir= node_modules/.pnpm #  项目的所有直接和间接依赖关系都链接到此目录
package-import-method= auto # 依赖包从 store 中导入的方式
registry = https://registry.npm.taobao.org
```

然后我们在 common 文件目录下执行 pnpm install, 之后我们观察到子工程中的 node_modules 目录的大小实际不到 100kb, common 目录下的 node_modules也在 300M到500M, 其余i的依赖项大多数都是链接到 common 文件夹下的 node_modules/.pnpm  这个目录

但是, 在运行 npm run serve 的时候, 继而发现了一个严峻的问题

比如在 `babel.config.js` 中配置了 `@babel/preset-env` 这一个配置项, 然而在 `package.json` 中并没有此项依赖, 在该工程的 node_modules 下也没有找到它的链接

接着尝试在子工程中也创建 `.npmrc` 这个文件, 文件内容如下

``` text
store-dir= ../common/.pnpm-store # 所有软件包保存在磁盘上的位置, 需要指向 common 目录
virtual-store-dir= ../common/node_modules/.pnpm #  项目的所有直接和间接依赖关系都链接到此目录, 需要指向 common 目录
package-import-method= auto # 依赖包从 store 中导入的方式
registry = https://registry.npm.taobao.org 
shamefully-hoist = true # 告诉 pnpm 不采用半严格的 node_modules 目录结构
```

然后再次在子工程中执行 pnpm install, 之后我们观察到子工程中的 node_modules 目录的大小会适当的加大了几百kb, 因为我们又为子工程安装了一些链接不到的依赖项, 这一操作是为了安装子工程中没有链接到 common 目录下的依赖包, 不过不用担心, 在第一个子工程中执行的时候会比较耗时, 大概在 1~2 分钟, 然后在其他的子工程中再次执行的时候, 几乎不占时间的为子工程创建链接（除非这个依赖项在 common 中找不到）

这里需要注意的是 common 下面配置的 registry 和子工程下配置的 registry 是需要一致的, 否则 pnpm 为我们安装不同软件源的依赖项, 这一过程是多余的
