---
author: 星火燎原@vxhly
title: WEB 前端冷门知识
categories: [web]
tags: [WEB, HTML5, JavaScript, CSS, jQuery]
date: 2019-02-26 20:22:38
---

<Boxx changeTime="10000"/>

::: tip 前言
最近在[牛客网](https://www.nowcoder.com/)上刷前端相关的题目, 发现自己真的不堪一击, 菜的要死, 毕竟都是大厂出的题目, 在刷题的过程中同时也学到了不少的冷门知识（主要是自己做错的题目里的知识点）, 现在分享出来, 真的有用, 注意这里面内容纯理论。
:::
<!-- more -->

## HTML

* **置换元素**: 浏览器根据元素的标签和属性来决定元素的具体显示内容
* **非置换元素**: 浏览器的大多数元素都是不可置换元素, 即其内容直接展示给浏览器
* input 元素设置 readonly:true => 脚本无法修改 input 的值 （F）
* input 元素设置 disabled:true => 脚本无法修改 input 的值（F）
* input 元素设置 readonly:true => input 的值不会随表单提交（F）
* input 元素设置 disabled:true => input 的值不会随表单提交（T）
* DHTML 的动态样式作用: 使网页作者改变内容的外部特征而不强制用户再次下载全部内容
* HTML5 获取某用户当前位置信息 => getCurrentPosition()
* reflow(回流) => 对 DOM 树进行渲染, 修改 DOM 或修个元素的形状大小, 就会触发 reflow
* enctype => 规定在发送发送服务器之前对表单数据进行编码的方式

-- `application/x-www-form-urlencoded` 在发送前边吗所有的字符
-- `multipart/form-data` 部队字符编码的使用包含文件上传控件表单, 必须包含的方式
-- `text/plan` 空格转换为 '+' 加号, 但不对特殊字符进行编码

* htmlDoctype 告诉浏览器文档使用哪一种 html 或者 xhtml
* 不同文档模式（htmlDoctype）影响 css 内容呈现, 尤其是对盒模型
* 文档模式使得不同浏览器在混杂模式下差异大

## CSS

* 当 `margin-top` 、 `padding-top` 的值是百分比时, 分别计算相对最近父级元素的 width, 相对最近父级元素的 width
* `position: static` => 无定位
* `position: absolute` => 绝对定位
* `position: fixed` => 相对窗口的固定定位
* `position: relative` => 相对定位
* mozilla(gecko) 内核 => 浏览器: firefox、flock 等 => css 兼容前缀: `-moz`
* webkit 内核 => 浏览器: safari、chrome 等 => css 兼容前缀: `-webkit`
* opera 内核 => 浏览器: opera 等 => css 兼容前缀: `-o`
* trident 内核 => 浏览器: ie 等 => css 兼容前缀: `-ms`
* 选择器优先级: ID 选择器>class选择器=伪类选择器>标签名陈选择器
* 在不涉及样式情况下, 页面元素的优先显示与结构摆放顺序无关（F）

## JavaScript

* Promise 的三种状态

-- 异步操作（未完成）=> `pending`
-- 异步操作（已完成）=> `resolved` (又称 `fulfilled` )
-- 异步操作（失 败） => `rejected`
-- 状态改变只能是 `pending` => `resolved` 或者 `pending` => `rejected`
-- 状态一旦改变就不能再次进行改变

* JavaScript 基本数据类型: String、Boolean、Null、Number、undefind
* JavaScript 复杂数据类型: Object、Array、Function、RegExp、Date、Error
* JavaScript 全局数据类型: Math
* JavaScript 全局函数: escape()、eval()、isFinte()、isNaN()、parseFloat()、parseInt()、unescape()
* JavaScript 异步编程

-- `回调函数` : 这是异步编程最基本的方法
-- `事件监听` : 另一种思路是采用事件驱动模式, 任务的执行不取决于代码的顺序, 而取决于某事件是否发生
-- `发布/订阅` : 上一节的"事件", 完全可以理解成"信号"
-- `Promise 对象` : Promise 对象是 CommanJS 工作组提出的一种规范, 目的是为异步编程提供统一接口

* RegExp 对象三个方法 => `test()` 、 `exec()` 、 `compile()`
* JavaScript 内置对象: Arguments（函数参数集合）、Array（数组）、Boolean（布尔）、Date（日期）、Error（异常）、Function（函数）、Math（数学）、Number（数值）、Object（基础）、String（字符串）、RegExp（正则表达式）
* `history.length` 返回浏览器你是列表的 URL 的数量
* `history.back()` 加载 history 列表的前一个 URL
* `history.forward()` 加载 history 列表的下一个 URL
* `history.go()` 加载 history 列表中的某个具体页面
* call 与 apply 都属于 Function.prototype 的一个方法, 所以每一个 Function 实例都有 call、apply 属性
* call 第一个参数是 this, 传递给函数的参数必须逐个列举出来；apply 传递函数的参数是数组, 就是多个参数组合成一个数组传入
* call 传入的是直接参数列表, 可见一个函数的对象上下文从初始的上下文改变为由 thisObj 指定的新对象
* 跨域 => 浏览器不会将存在跨域限制的服务端口响应正常交给客户端代码去处理
* `hasOwnproperty()` => 判断一个对象是否包含有属性或者对象
* `isPrototypeOf()` => 判断检查其原型链对象是否存在于指定对象中
* 把鼠标一定到按钮上点击, 发生 => hover -> focus -> active
* Ajax 的优势在于开放性、易用性及易于开发
* Flash 的优势在于多媒体处理, 可以更容易地调用浏览器之外的外部资源
* Ajax 最主要的批评就是他可能坏浏览器地后退功能
* Flash 文件经常会很大, 用户第一次使用的是否需要让来的时间较长

## 构建工具

### gulp 

gulp 强调的是前端开发的工作流程, 我们可以通过配置一系列的 task, 定义 task 处理的事务（例如文件压缩合并、雪碧图、启动 server、版本控制等）, 然后定义执行顺序, 来让 gulp 执行这些task, 从而构建项目的整个前端开发流程。

PS: 简单说就一个Task Runner

### webpack

webpack 是一个前端模块化方案, 更侧重模块打包, 我们可以把开发中的所有资源（图片、js 文件、css 文件等）都看成模块, 通过 loader（加载器）和 plugins（插件）对资源进行处理, 打包成符合生产环境部署的前端资源。

PS: webpack is a module bundle

### 两者区别

虽然都是前端自动化构建工具, 但看他们的定位就知道不是对等的。gulp与webpack上是互补的, 还是可替换的, 取决于你项目的需求
gulp 严格上讲, 模块化不是他强调的东西, 他旨在规范前端开发流程。
webpack 更是明显强调模块化开发, 而那些文件压缩合并、预处理等功能, 不过是他附带的功能。

## 版本控制工具

### GIT

Git 是一个分布式版本控制系统, 操作命令包括: clone, pull, push, branch , merge , push, rebase, Git擅长的是程序代码的版本化管理。

### SVN

SVN 是一个集中式版本控制系统, SVN只能有一个指定中央版本库。当这个中央版本库有问题时, 所有工作成员都一起瘫痪直到版本库维修完毕或者新的版本库设立完成

### 两者区别

Git 与 SVN 区别点: 

* GIT 是分布式的, SVN 不是: 这是 GIT 和其它非分布式的版本控制系统, 例如 SVN, CVS 等, 最核心的区别。
* GIT 把内容按元数据方式存储, 而 SVN 是按文件: 所有的资源控制系统都是把文件的元信息隐藏在一个类似.svn, .cvs等的文件夹里。
* GIT 分支和 SVN 的分支不同: 分支在 SVN 中一点不特别, 就是版本库中的另外的一个目录。
* GIT 没有一个全局的版本号, 而 SVN 有: 目前为止这是跟 SVN 相比 GIT 缺少的最大的一个特征。
* GIT 的内容完整性要优于 SVN: GIT 的内容存储使用的是 SHA-1 哈希算法。这能确保代码内容的完整性, 确保在遇到磁盘故障和网络问题时降低对版本库的破坏。
* 是通过.git和.svn文件, .git只是在本地的版本库的目录下存在, 而.svn存在于每一个文件夹, 当我们不需要版本控制的时候, 要删除.svn很费时。

## 网络基础

* HTTP 协议所使用的运输层协议是 `TCP/IP`
* HTTPS 的默认端口号是 `443`
* TCP 注重数据可靠性, UDP 注重数据传输快
* 传输层提供端对端的可靠文传递和错误恢复
* DNS 协议大多数运行在 `UDP` 协议上
* DNS 的默认端口号是 `53`
* HTTP 工作于应用层, HTTPS 工作于传输层

## 进程和线程

* 进程是资源分配的最小单位
* 线程是资源调度的最小单位
* 进程是动态的、多个进程可以包含相同的程序和多个进程可以进行并发运行

## 算法

* 排序算法: 插入排序、希尔排序、冒泡排序、堆排序、归并排序、快速排序、选择排序
* 排序算法不稳定的有: 希尔排序、堆排序、快速排序、选择排序
