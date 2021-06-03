---
author: 星火燎原@vxhly
title: CSS 代码书写顺序以及代码规范
categories: [web]
tags: [CSS, Criterion]
date: 2016-08-24 20:40:06
---

<Boxx changeTime="10000"/>

::: tip 前言
一个优秀的前端开发人员, 在书写 CSS 时必定要注意的重点, 而不是随意的书写 CSS, 这样会给自己还有他人带来困扰。
:::
<!-- more -->

## CSS 文件的编码

表明 CSS 文件的页面编码为 UTF-8。 在 CSS 文件的最前面添加如下代码

 `For Example:`

``` html
@charset "UTF-8";
```

## CSS 书写顺序

* 位置属性 { `position` , `top` , `right` , `z-index` , `display` , `float` 等}
* 大小 { `width` , `height` , `padding` , `margin` 等}
* 文字系列 { `font` , `line-height` , `letter-spacing` , `color` , `text-align` 等}
* 背景 { `background` , `border` 等}
* 其他 { `animation` , `transition` 等}

 `For Example:`

``` css
/* Low */
.example {
    color: red;
    z-index: -1;
    background-color: #9e0;
    display: inline-block;
    font-size: 1.5em;
}

/* Good */
.example {
    z-index: -1;
    display: inline-block;
    font-size: 1.5em;
    color: red;
    background-color: #9e0;
}
```

### 利用插件实现自动排序

使用过 Sublime text 的童鞋应该知道有一款插件 [CSScomb](http://csscomb.com/) 是用来自动排序 CSS 属性的, 这是一个不错的偷懒方式, 也是加快开发速度的技巧

## CSS 书写规范

### 使用 CSS 缩写属性

CSS 有些属性是可以缩写的, 比如 padding, margin, font 等等, 这样精简代码同时又能提高用户的阅读体验。

 `For Example:`

``` css
/* Low */
.box {
    border-top-style: none;
    font-family: serif;
    font-size: 100%;
    line-height: 1.6;
    padding-bottom: 2em;
    padding-left: 1em;
    padding-right: 1em;
    padding-top: 0;
}

/* Good */
.box {
    border-top: 0;
    font: 100%/1.6 serif;
    padding: 0 1em 2em;
}
```

### 去掉小数点前的 "0"

若属性的值小于 1, 建议去掉小数点前的 "0"

 `For Example:`

``` css
/* Low */
.box {
    font-size: 0.8em;
}

/* Good */
.box {
    font-size: 0.8em;
}
```

### 值为 0 时, 去掉单位

若属性的值等于 0 时, 建议去掉值后面跟着的单位

 `For Example:`

``` css
/* Low */

* {

    margin: 0px;
    padding: 0px;
}

/* Good */

* {

    margin: 0;
    padding: 0;
}
```

### 简写命名

很多用户都喜欢简写类名, 但前提是要让人看懂你的命名才能简写哦

 `For Example:`

``` css
/* Low */
.navigation {
    margin: 0 0 1em 2em;
}

.atr {
    color: #93c;
}

/* Good */
.nav {
    margin: 0 0 1em 2em;
}

.author {
    color: #93c;
}
```

### 16 进制颜色代码可以缩写

有些颜色代码是可以缩写的, 我们就尽量缩写吧, 提高用户体验为主, 简写规则:  同时满足 1、 2 相同, 3、 4 相同, 5、 6 相同

 `For Example:`

``` css
/* Low */
.box {
    color: #eebbcc;
}

.font {
    color: #333333;
}

/* Good */
.box {
    color: #ebc;
}

.font {
    color: #333;
}
```

## 选择器命名规范

* 长名称或词组可以使用中横线来为选择器命名
* 一律小写避免使用大写, 区分于 `JS` 的函数名
* 强制建议使用英文, 不使用拼音
* 不建议使用 `_` 下划线来命名 `CSS` 选择器, 为什么呢？

-- 输入的时候少按一个 `shift` 键；
-- 浏览器兼容问题 （比如使用 `_tips` 的选择器命名, 在 `IE6` 是无效的）
-- 能良好区分 `JavaScript` 变量命名（ `JS` 变量命名是用 `_` ）

 `For Example:`

``` css
/* Low */
.maintitle {
    color: #e3c;
}

.main_title {
    color: #e3c;
}

/* Good */
.main-title {
    color: #e3c;
}
```

## CSS 选择器最好不要随意使用 id

* `id` 在 `JS` 是唯一的, 不能多次使用, 而使用 `class` 类选择器却可以重复使用
* `id` 的优先级优先与 `class` , 所以 `id` 应该按需使用, 而不能滥用
* `id` 选择器多用于 `JS` 方法的调用

 `For Example:`

``` css
/* Low */
#info-title {
    font-size: 3em;
}

/* Good */
.info-title {
    font-size: 3em;
}
```

## 为选择器添加前缀

* 有时候可以给选择器添加一个表示状态的前缀, 让语义更明了
* 当你在开发属于自己的 `CSS` 代码时也可以为你的 `CSS` 代码添加前缀

 `For Example:`

``` css
/* Low */
.whithdrawal {
    background-color: #ccc;
}

.divconent {
    width: 60rem;
    font-family: serif;
    color: #999;
}

.sapninline {
    display: inline-block;
}

/* Good */
.is-whithdrawal {
    background-color: #ccc;
}

.my-div-conent {
    width: 60rem;
    font-family: serif;
    color: #999;
}

.my-sapn-inline {
    display: inline-block;
}
```

## 命名规范举例

* 所有命名一律使用小写
* 所有的命名一律使用英文, 不使用中文或者拼音
* 用 `-` 代替 `_` , 文件命名时尽量少用 `-`
* 文件命名时尽量不使用缩写, 除非是一眼就可以看出来该文件是干什么用
* 文件命名时, 如有该文件的版本号, 尽量要跟上, 如:  jquery-2.1.4.min.js
* 引用外部文件时, 尽量选择压缩版的文件, 这样可以加快网页的加载速度

### 常用的 CSS / ID 命名

* **header / footer** => 头部 / 尾部
* **content / container** => 内容
* **nav / subnav** => 导航 / 子导航
* **sidebar** => 侧边栏
* **row / column** => 行 / 列
* **top / left / bottom / right** => 上 / 左 / 下 / 右
* **logo** => 标志
* **banner** => 广告
* **main** => 页面主体
* **hot** => 热点
* **news** => 新闻
* **download** => 下载
* **menu / submenu** => 菜单 / 子菜单
* **search** => 搜索
* **friendlink** => 友情链接
* **copyright** => 版权
* **tags** => 标签
* **list** => 列表
* **loginbar** => 登录条
* **msg** => 提示信息
* **status** => 状态

### 常用 CSS / JS 文件命名

* **master.css** => 主要的
* **modeule.css** => 模块
* **base.css** => 基本共用
* **base.min.css** => 基本共用样式表压缩版
* **layout.css** => **布局、 版面
* **themes.css** => 主题
* **font.css** => 文字
* **from.css** => 表单
* **jquery-ui-1.9.0.js** => jQuery UI 的脚本, 版本为 1.9.0
* **jquery.min.js** => jQuery 压缩版
* **base.js** => 基本共用
* **from-ajax.js** => 通过 Ajax 技术提交的表单
