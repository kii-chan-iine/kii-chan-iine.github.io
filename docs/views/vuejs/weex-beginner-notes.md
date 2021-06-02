---
author: 星火燎原@vxhly
title: weex 初学笔记
categories: [vuejs]
tags: [Vue.js, Node.js, Weex, HybridApp]
date: 2020-05-12 09:58:29
---

<Boxx changeTime="10000"/>

::: tip 前言
Weex 是使用流行的 Web 开发体验来开发高性能原生应用的框架。关于 Weex 的一些基础概念不多阐述, 这篇文章记录的是在开发过程中遇到一些坑以及其解决方案。
:::
<!-- more -->

## 前期环境搭建

[Weex 的官方文档](https://weex.apache.org/zh/)

::: warning 大坑
前期开发环境搭建这一步很重要, 有可能会失败也有可能会因为网速等问题导致失败, 但是一定要耐心多多尝试, 多尝试之后就会体验到飞起来的感觉
:::

::: tip 开发系统环境
本篇文章使用到的系统环境 `macOS Catalina 10.15.4` , 注意这边不推荐 `win 7` 用户学习 Weex, 可使用 `win 8` 、 `win 10`
:::

### OSX 环境

``` bash
sudo chmod -R 777 /usr/local/lib/node_modules/
npm i -g weex-toolkit # 安装不要使用sudo执行
weex -v # 查看当前weex工具版本
```

### Windows 环境

``` bash
npm i -g weex-toolkit
weex -v # 查看当前weex工具版本
```

## 构建 IOS APP

要想构建 IOS APP, 首先需要 `MacOS` 的一个系统环境, 构建之前需要进行以下步骤

``` bash
weex platform add ios # 添加 IOS 平台的支持
```

接下来这个很关键, 换源（换的是 cocoapods 的安装软件源）, 除非你非常自信自己的网络是可以访问至国外的服务器, 或是已经有了科学上网的本领

``` bash
gem --version # 检测机子上是否正确安装了 ruby 环境
sudo gem install cocoapods # 需要使用 sudo 的权限
```

换源 [清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/help/CocoaPods/)

``` bash
cd ~/.cocoapods/repos 
pod repo remove master
git clone https://mirrors.tuna.tsinghua.edu.cn/git/CocoaPods/Specs.git master # 这一步可能会下载失败啥的, 一定要多试几次并确保下载完整
```

最后进入自己的工程, 在自己工程的podFile第一行加上：

``` bash
source 'https://mirrors.tuna.tsinghua.edu.cn/git/CocoaPods/Specs.git' # 最好是将原第一行内容直接替换成这个
```

## Debug

如果执行 `weex debug src` 发生类似于以下错误的可以使用 `weex doctor` 来进行修复你的一些缺失依赖

::: danger Error
./src/index.vue
Module build failed: Error: 

Vue packages version mismatch:

* vue@2.5.21 (/Users/codeman/.wx/modules/node_modules/_vue@2.5.21@vue/dist/vue.runtime.common.js)
* vue-template-compiler@2.6.11 (/Users/codeman/.wx/modules/node_modules/_vue-template-compiler@2.6.11@vue-template-compiler/package.json)

This may cause things to work incorrectly. Make sure to use the same version for both.
If you are using vue-loader@>=10.0, simply update vue-template-compiler.
If you are using vue-loader@<10.0 or vueify, re-installing vue-loader/vueify should bump vue-template-compiler to the latest.
:::
