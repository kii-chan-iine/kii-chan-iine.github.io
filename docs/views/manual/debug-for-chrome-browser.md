---
author: 星火燎原@vxhly
title: 关于 Chrome 浏览器的调试方法
categories: [manual]
tags: [JavaScript, HTML5, CSS]
date: 2016-08-05 18:24:29
---

<Boxx changeTime="10000"/>

::: tip 前言
作为前端开发人员, 最佳的调试环境便是浏览器自带的开发者工具, 此文则介绍 Chrome 浏览器的开发者工具的使用方法。
:::
<!-- more -->

## Chrome 浏览器调试窗口中常用的标签页

* **Elements**: 用于查看页面 `html` 代码
* **NetWork**: 用于查看页面的网络请求
* **Sources**: 用于查看页面所加载的源文件
* **Timeline**: 用于查看页面加载 `JavaScript` 、页面元素渲染等的时间
* **Profiles**: 用于做性能优化的, 包括查看 CPU 执行时间与内存占用
* **Resource**s: 用于查看保存再本地的一些信息, 比如 `cookie` 等
* **Audits**: 一般用于优化前端页面、加速网页加载速度
* **Console**: `JavaScript` 控制台, 这个面板可以查看错误信息、打印调试信息、写一些测试脚本, 还可以当作 `JavaScript API` 查看用。

Chrome 浏览器不仅可以调试页面、 JavaScript、请求、资源、 cookie, 还可以模拟手机进行调试

## 打开调试方法

### 方法一

直接在页面上点击右键, 然后选择审查元素

![打开调试方法](https://vxhly.github.io/assets/open-browser-debug-1.png)

### 方法二

在 Chrome 的工具中找到开发者工具

![打开调试方法](https://vxhly.github.io/assets/open-broswer-debug-2.png)

#### 方法三

使用快捷键 Ctrl+Shift+I (或者 Ctrl+Shift+J 直接打开控制台), 或者直接按 F12

![打开调试方法](https://vxhly.github.io/assets/open-browser-debug-3.png)

### Elements 标签页

Elements 标签页的左侧就是对页面 HTML 结构的查看与编辑, 你可以直接在某个元素上双击修改元素的属性。

![Elements 标签页](https://vxhly.github.io/assets/broswer-debug-elements-1.png)

* **Add attribute**: 向该元素添加其他的属性
* **Edit as HTML**: 直接对元素的 `HTML` 进行编辑, 或者删除某个元素, 所有的修改都会即时在页面上得到呈现。
* **Copy**: 可以将 `HTML` 代码直接复制下来, 在拷贝别人网站上面的 `HTML` 代码的时候灰常方便, 你懂的~~
* **Hide element**: 隐藏该元素
* **Delete element**: 删掉该元素
* **Scroll into view**: 视图滚动到选择的位置
* **Break on**: 可以对某个元素进行监听, 在 `JavaScript` 中对元素的属性或者 `HTML` 进行修改的时候, 直接触发断点, 跳转到对改元素进行修改的 `JavaScript` 代码处

Elements 标签页的右侧可以对元素的 CSS 进行查看与编辑修改

![Elements 标签页](https://vxhly.github.io/assets/broswer-debug-elements-2.png)

* **Styles**: 看 `HTML` 元素的样式
* **Computed**: 可以看元素的盒子模型
* **Event Listeners**: 对元素的监听方法将会显示在这里
* **DOM Breakpoints**: `DOM` 树的断点调试, （通过右键某一个元素, 依次选择 `Break on` , 选择 `Attributes modifications` ）, 设置后刷新页面, 当元素的属性发生改变时, 暂停脚本并定位到该位置
* **Properties**: 可以查看到元素具有的方法与属性, 比查 `API` 手册要方便得多

## Network 标签页

Network 标签页对于分析网站请求的网络情况、查看某一请求的请求头和响应头还有响应内容很有用。 注意是在你打开 Chrome 开发者工具后发起的请求, 才会在这里显示的哦。

![Network 标签页](https://vxhly.github.io/assets/browser-debug-network-1.png)

点击左侧某一个具体请求 URL, 可以看到该请求的详细 HTTP 请求情况

![Network 标签页](https://vxhly.github.io/assets/browser-debug-network-2.png)

我们可以在这里看到 `HTTP 请求头` 、 `HTTP 响应头` 、 `HTTP 返回的内容` 等信息。

* **Headers**: 请求头信息和响应头信息
* **Preview**: 预览结果, 如果是文件可以查看这个文件；如果是图片可以预览这个图片；如果是从服务器返回来的 `JSON` 数据, 可以查看格式话后的 `JSON`
* **Response**: 从服务器返回的响应结果
* **Cookies**: 请求和响应的 `Cookie`
* **Timing**: 具体的响应时间

![Network 标签页](https://vxhly.github.io/assets/browser-debug-network-4.png)

<br>

![Network 标签页](https://vxhly.github.io/assets/browser-debug-network-5.png)

<br>

![Network 标签页](https://vxhly.github.io/assets/browser-debug-network-6.png)

## Sources 标签页

Sources 标签页可以查看到请求的资源情况, 包括 CSS、 JavaScript、图片等的内容。也可以设置各种断点。对存储的内容进行编辑然后保存也会实时的反应到页面上。

![Sources 标签页](https://vxhly.github.io/assets/browser-debug-sources-1.png)

## Audits 标签页

这个对于优化前端页面、加速网页加载速度很有用哦。（相当与 Yslow）

![Audits 标签页](https://vxhly.github.io/assets/browser-debug-audits-3.png)

## Console 标签页

JavaScript 控制台, 前端开发人员最经常使用它来调试 js 的错误。

![Console 标签页](https://vxhly.github.io/assets/browser-debug-console-1.png)

<br>

![Console 标签页](https://vxhly.github.io/assets/browser-debug-console-2.png)

## 移动端开发调试

可以模拟出不同分辨率的移动端显示的效果, 此效果仅供参考, 因为还有可能受浏览器的不同而显示效果不同。

![移动端开发调试](https://vxhly.github.io/assets/browser-debug-moblie-1.png)

<br>

![移动端开发调试](https://vxhly.github.io/assets/browser-debug-moblie-2.png)
