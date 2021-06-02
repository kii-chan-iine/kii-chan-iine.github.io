---
author: 星火燎原@vxhly
title: Gulp 使用指南
categories: [manual]
tags: [Node.js, JavaScript]
date: 2016-10-28 19:06:15
---

<Boxx changeTime="10000"/>

::: tip 前言
本文主要介绍如何使用 Gulp.js, 来帮助你构建一个自动化的前端开发流程。总结并归类自身使用中所遇到的方法、问题, 便于新人上手和理解, 同时也作为一份 Gulp 参考文档来使用, 让你对 Gulp 有个更深入的了解。
:::
<!-- more -->

## Gulp 是什么

Gulp 是基于 `Node.js` 的一个构建工具（自动任务运行器）, 开发者可以使用它构建自动化工作流程（前端集成开发环境）。一些常见、重复的任务, 例如: 网页自动刷新、CSS 预处理、代码检测、压缩图片、等等, 只需用简单的命令就能全部完成。使用它, 可以简化工作, 让你把重点放在功能开发上；同时减少人为失误, 提高开发效率和项目质量, 让专注更为专注。如果你之前接触过 Grunt, 那上手 Gulp 就会觉得非常容易理解。

## 为什么使用 Gulp 而不使用 Grunt

构建前端自动化的工具有很多, 如: `Grunt` , `Brunch` , `Broccoli` 等, 而目前过内最流行的属于 Grunt, 之前我也在用使用, 那为什么选择 Gulp 来代替 Grunt？

简单的总结一下, 有三点: 

* 简洁: Gulp 侧重 `“代码优于配置”(code over configuration)` 。最直观的感受, 更为简单和清晰, 不需要像 Grunt 一样写一堆庞大的配置文件。
* 高效: Gulp 基于 `Node Streams（流` ）来构建任务, 避免磁盘反复 `I/O（读取/写入）` 。每个任务都是单独执行（坚持做一件事并且做好它）, 这使得它速度更快、更为纯粹。
* 易学: Gulp 核心 API 只有 4 个。简洁的 API 易于上手, 学习过程比较平滑。

## Gulp 核心 API

Gulp 的核心 API 只有 4 个: `src` 、 `dest` 、 `task` 、 `watch`
* `gulp.src(globs[, options])` => 指明源文件路径。**globs**: 路径模式匹配; **options**: 可选参数；
* `gulp.dest(path[, options])` => 指明处理后的文件输出路径。**path**: 路径（一个任务可以有多个输出路径）；**options**: 可选参数；
* `gulp.task(name[, deps], fn)` => 注册任务。**name**: 任务名称（通过 gulp name 来执行这个任务）; **deps**: 可选的数组, 在本任务运行中所需要所依赖的其他任务（当前任务在依赖任务执行完毕后才会执行）; **fn**: 任务函数（function 方法）；
* `gulp.watch(glob [, opts], tasks)` => 监视文件的变化并运行相应的任务。**glob**: 路径模式匹配; **opts**: 可以选配置对象; **taks**: 执行的任务；

[API 帮助文档](http://github.com/gulpjs/gulp/blob/master/docs/API.md)

## 安装 Gulp

安装 Gulp 之前要确定系统中安装了 Node.js, 有了之后只需创建一个工作目录, 以下操作均在工作目录下执行。

### 全局安装

深入设置任务之前, 需先安装 gulp: 

``` bash
npm install gulp -g
```

这会将 gulp 安装到全域环境下, 让你可以存取 gulp 的 CLI。

### 本地安装

接着, 需要在本地的工在目录进行安装。切换至你的工作目录, 执行以下命令（当然执行之前要先确定该目录下有 `package.json` 文件, 如果没有可以通过 `npm init` 进行创建）

``` bash
npm install gulp --save-dev
```

上述指令将 gulp 安装到本地的工作目录, 并纪录于 `package.json` 中的 `devDependencies` 选项内。

## 安装 Gulp 插件

### 推荐插件

* 静态服务器 ([gulp-webserver](http://github.com/schickling/gulp-webserver))
* 网页自动刷新 ([gulp-livereload][<https://github.com/vohof/gulp-livereload>])
* 编译 **Sass** ([gulp-ruby-sass](http://github.com/sindresorhus/gulp-ruby-sass))
* **Autoprefixer** ([gulp-autoprefixer](http://github.com/Metrime/gulp-autoprefixer))
* 缩小化(minify) **CSS** ([gulp-minify-css](http://github.com/jonathanepollack/gulp-minify-css))
* **JSHint** ([gulp-jshint](http://github.com/wearefractal/gulp-jshint))
* 拼接 ([gulp-concat](http://github.com/wearefractal/gulp-concat))
* 丑化(Uglify) ([gulp-uglify](http://github.com/terinjokes/gulp-uglify))
* 图片压缩 ([gulp-imagemin](http://github.com/sindresorhus/gulp-imagemin))
* 即时重整(LiveReload) ([gulp-livereload](http://github.com/vohof/gulp-livereload))
* 清理档案 ([gulp-clean](http://github.com/peter-vilja/gulp-clean))
* 图片快取, 只有更改过得图片会进行压缩 ([gulp-cache](http://github.com/jgable/gulp-cache/))
* 更动通知 ([gulp-notify](http://github.com/mikaelbr/gulp-notify))

### 通过 npm 进行安装

 `Install Format:`

``` bash
npm install [Module_Name] --save-dev
```

## Gulp 任务配置

### 创建 gulpfile.js 文件

在项目根目录中, 创建 `gulpfile.js` 文件, 用来配置和定义任务（ `task` ）。

``` bash
touch gulpfile.js
```

### 我的安装插件

``` bash
npm install gulp --save-dev               # 基础库
npm install gulp-livereload --save-dev    # 网页自动刷新
npm install gulp-webserver --save-dev     # 本地服务器
npm install gulp-notify --save-dev        # 更新
```

### 我的 gulpfile.js 配置

打开 `gulpfile.js` 文件, 填写相关配置。

``` javascript
// 引入 gulp
var gulp = require("gulp"); // 基础库

// 引入 gulp 插件
var livereload = require("gulp-livereload"); // 网页自动刷新（服务器控制客户端同步刷新）
var webserver = require("gulp-webserver"); // 本地服务器

// 注册任务
gulp.task("webserver", function() {
    gulp
        .src("./") // 服务器目录（./ 代表根目录）
        .pipe(
            webserver({
                // 运行 gulp-webserver
                livereload: true, // 启用 LiveReload
                open: true // 服务器启动时自动打开网页
            })
        );
});

// 监听任务
gulp.task("watch", function() {
    gulp.watch("*.html", ["html"]); // 监听根目录下所有 .html 文件
});

// 默认任务
gulp.task("default", ["webserver", "watch"]);
```
