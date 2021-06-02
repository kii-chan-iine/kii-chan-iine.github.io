---
author: 星火燎原@vxhly
title: React Native 填坑指南
categories: [app]
tags: [ReactNative, HybridApp, App]
date: 2020-07-02 20:58:47
---

<Boxx changeTime="10000"/>

::: tip 前言
开始学习 React Native, 期间也遇上了许多坑, 本篇将在官方文档基础上补充一些填坑的指南。
:::
<!-- more -->

## 官方文档

[React Native 官方文档](https://reactnative.cn/)

## 搭建开发环境

::: tip 提示
这里并不推荐大家使用沙箱环境。沙盒环境大量依赖于国外网络环境, 也不能直接安装第三方原生组件。不建议国内用户使用。
:::

这一步及其重要, 多数人都会卡在这一步, 包括博主也是在这一步也是卡了很久, 开发环境搭建好了一定会事半功倍, 接下的内容将会带领你少走弯路。

### 搭建前请换源

使用过 Node.js 开发项目的都知道我们的 Node.js 项目在安装项目依赖时受限于国内的网络环境, 并不能快速的下载项目依赖, 接下几个步骤教你换源

首先是 `yarn` 和 `npm` 的软件源, 使用到的是 `cgr` 这个命令来快速的切源

``` bash
npm i -g cgr
cgr use taobao # 此命令可以同时切换 yarn 和 npm 的软件源
```

当然也可以使用以下命令来快速的切源

``` bash
npm config set registry https://registry.npm.taobao.org --global
npm config set disturl https://npm.taobao.org/dist --global
```

在开发 IOS 的应用时还需切换 CocoaPods 镜像, 可以参照 [CocoaPods 镜像使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/CocoaPods/) 进行切源

``` bash
cd ~/.cocoapods/repos 
pod repo remove master
git clone https://mirrors.tuna.tsinghua.edu.cn/git/CocoaPods/Specs.git master
```

MacOS 下建议还需更换 brew 的软件源, 可以参照 [阿里云的镜像源](https://developer.aliyun.com/mirror/homebrew) 进行切源

Bash 终端配置

``` bash
# 替换brew.git:
cd "$(brew --repo)"
git remote set-url origin https://mirrors.aliyun.com/homebrew/brew.git
# 替换homebrew-core.git:
cd "$(brew --repo)/Library/Taps/homebrew/homebrew-core"
git remote set-url origin https://mirrors.aliyun.com/homebrew/homebrew-core.git
# 应用生效
brew update
# 替换homebrew-bottles:
echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.aliyun.com/homebrew/homebrew-bottles' >> ~/.bash_profile
source ~/.bash_profile
```

Zsh 终端配置

``` bash
# 替换brew.git:
cd "$(brew --repo)"
git remote set-url origin https://mirrors.aliyun.com/homebrew/brew.git
# 替换homebrew-core.git:
cd "$(brew --repo)/Library/Taps/homebrew/homebrew-core"
git remote set-url origin https://mirrors.aliyun.com/homebrew/homebrew-core.git
# 应用生效
brew update
# 替换homebrew-bottles:
echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.aliyun.com/homebrew/homebrew-bottles' >> ~/.zshrc
source ~/.zshrc
```

::: danger 注意
windows 环境下不使用 cnpm 就好, cnpm 安装的模块路径比较奇怪, packager 不能正常识别！
:::

### 安装依赖和运行

::: tip 提示
安装依赖这边不做重复声明, 官方文档已经非常详细了
:::

::: warning 坑
React Native 运行时将监听 8081 端口, 所以 8081 端口绝对不能被占用, 否则会导致编译失败等问题

当然如果是因为这个引起的编译失败也不要慌, 将你的 8081 端口占用的进程关掉即可

``` bash
lsof -i tcp:8081
kill [pid]
```

:::

## 开发填坑

### 导入 IconFont 自定义图标库

参考 [react-native-iconfont-cli](https://github.com/iconfont-cli/react-native-iconfont-cli)

#### Step 1

安装插件

``` bash
# Yarn
yarn add react-native-svg
yarn add react-native-iconfont-cli --dev

# Npm
npm install react-native-svg
npm install react-native-iconfont-cli --save-dev
```

#### Step 2

静态链接。请注意您使用的 React-Native 版本号

``` bash
# RN < 0.60
react-native link react-native-svg

# RN >= 0.60
cd ios && pod install
```

#### Step 3

生成配置文件

``` bash
npx iconfont-init
```

#### Step 4

开始生成React-Native标准组件

``` bash
npx iconfont-rn
```
