---
author: 星火燎原@vxhly
title: HTML 和 CSS 代码规范
categories: [web]
tags: [Criterion, HTML5, CSS]
date: 2016-10-03 21:45:11
---

<Boxx changeTime="10000"/>

::: tip 前言
介绍 HTML 和 CSS 的格式和代码规范, 旨在提高代码质量和协作效率。
:::
<!-- more -->

## 通用样式规范

### 协议

省略图片、样式、脚本以及其他媒体文件 URL 的协议部分（ `http:` , `https:` ）, 除非文件在两种协议下都不可用。这种方案称为 `protocol-relative URL` , 好处是无论你是使用 `HTTPS` 还是 `HTTP` 访问页面, 浏览器都会以相同的协议请求页面中的资源, 同时可以节省一部分字节。

 `For Example:`

``` html
<!-- Low-->
<script src="https://www.google.com/js/gweb/analytics/autotrack.js"></script>

<!-- Good -->
<script src="//www.google.com/js/gweb/analytics/autotrack.js"></script>
```

``` css
/* Low */
.example {
    background: url("https://www.google.com/images/exampl"e);
}

/* Good */
.example {
    background: url("//www.google.com/images/example");
}
```

## 通用格式规范

### 缩进

一次缩进 2 个空格, 不要使用 tab 或者混合 tab 和空格的缩进。

 `For Example:`

``` html
<ul>
    <li>Fantastic
    <li>Great
</ul>
```

``` css
.example {
    color: blue;
}
```

### 大小写

以下都应该用小写: HTML 元素名称, 属性, 属性值（除非 text/CDATA）, CSS 选择器, 属性, 属性值。

 `For Example:`

``` html
<!-- Low -->
<A HREF="/">Home</A>

<!-- Good -->
<img src="google.png" alt="Google">
```

``` css
/* Low */
.example {
    color: #e5e5e5;
}

/* Good */
.example {
    color: #e5e5e5;
}
```

### 结尾空格

结尾空格不仅多余, 而且在比较代码时会更麻烦。

 `For Example:`

``` html
<!-- Low -->
<p>What?
<p>

    <!-- Good -->
<p>Yes please.</p>
```

## 通用元规范

### 编码

在 HTML 中通过 `<meta charset="utf-8">` 指定编码方式, CSS 中不需要指定, 因为默认是 UTF-8。

### 注释

使用注释来解释代码: 包含的模块, 功能以及优点。

### 任务项

用 TODO 来标记待办事项, 而不是用一些其他的标记, 像 @@。

 `For Example:`

``` html
<!-- TODO: remove optional tags -->
<ul>
    <li>Apples</li>
    <li>Oranges</li>
</ul>
```

## HTML 风格规范

### 文档类型

HTML 文档应使用 HTML5 的文档类型: `<!DOCTYPE html>` 。

孤立标签无需封闭自身, 如: `<br>` 不要写成 `<br />` 。

### HTML 正确性

尽可能使用正确的 HTML。

 `For Example:`

``` html
<!-- Low -->
<title>Test</title>
<article>This is only a test.

    <!-- Good -->
    <!DOCTYPE html>
    <meta charset="utf-8">
    <title>Test</title>
    <article>This is only a test.</article>
```

### 语义化

根据使用场景选择正确的 HTML 元素（有时被错误的称为"标签"）。例如, 使用 h1 元素创建标题, p 元素创建段落, a 元素创建链接等等。正确的使用 HTML 元素对于可访问性、可重用性以及编码效率都很重要。

 `For Example:`

``` html
<!-- Low -->
<div onclick="goToRecommendations();">All recommendations</div>

<!-- Good -->
<a href="recommendations/">All recommendations</a>
```

### 多媒体元素降级

对于像图片、视频、canvas 动画等多媒体元素, 确保提供其他可访问的内容。图片可以使用替代文本（alt）, 视频和音频可以使用文字版本。

 `For Example:`

``` html
<!-- Low -->
<img src="spreadsheet.png">

<!-- Good -->
<img src="spreadsheet.png" alt="Spreadsheet screenshot.">
```

### 关注分离

标记、样式和脚本分离, 确保相互耦合最小化。

### 实体引用

如果团队中文件和编辑器使用同样的编码方式, 就没必要使用实体引用, 除了一些在 HTML 中有特殊含义的字符（如 `<` 和 `&` ）以及不可见的字符（如空格）。

 `For Example:`

``` html
<!-- Low -->
The currency symbol for the Euro is “&eur;”.

<!-- Good -->
The currency symbol for the Euro is “€”.
```

### type 属性

在引用样式表和脚本时, 不要指定 type 属性, 除非不是 CSS 或 JavaScript。

因为 HTML5 中已经默认指定样式变的 type 是 `text/css` , 脚本的 type 是 `text/javascript` 。

 `For Example:`

``` html
<!-- Low -->
<link rel="stylesheet" href="https://www.google.com/css/maia.css" type="text/css">

<!-- Good -->
<link rel="stylesheet" href="//www.google.com/css/maia.css">

<!-- Low -->
<script src="https://www.google.com/js/gweb/analytics/autotrack.js" type="text/javascript"></script>

<!-- Good -->
<script src="//www.google.com/js/gweb/analytics/autotrack.js"></script>
```

## HTML 格式规范

### HTML 引号

属性值用双引号。 `For Example:`

``` html
<!-- Low -->
<a class='maia-button maia-button-secondary'>Sign in</a>

<!-- Good -->
<a class="maia-button maia-button-secondary">Sign in</a>
```

## CSS 风格规范

### ID 和 Class 命名

使用有含义的 id 和 class 名称。 `For Example:`

``` css
/* Low: 毫无意义 */
#yee-1901 {}

/* Low: 表述直觉 */
.button-green {}

.clear {}

/* Good: 具体的 */
#gallery {}

#login {}

.video {}

/* Good: 通用的 */
.aux {}

.alt {}
```

### ID 和 Class 命名风格

id 和 class 应该尽量简短, 同时要容易理解。 `For Example:`

``` css
/* Low */
#navigation {}

.atr {}

/* Good */
#nav {}

.author {}
```

### 选择器

除非需要, 否则不要在 id 或 class 前加元素名。 `For Example:`

``` css
/* Low */
ul#example {}

div.error {}

/* Good */
#example {}

.error {}
```

### 属性简写

尽量使用 CSS 中可以简写的属性 (如 font), 可以提高编码效率以及代码可读性。 `For Example:`

``` css
/* Low */
.example {
    border-top-style: none;
    font-family: palatino, georgia, serif;
    font-size: 100%;
    line-height: 1.6;
    padding-bottom: 2em;
    padding-left: 1em;
    padding-right: 1em;
    padding-top: 0;
}

/* Good */
.example {
    border-top: 0;
    font: 100%/1.6 palatino, georgia, serif;
    padding: 0 1em 2em;
}
```

### 0 和单位

值为 0 时不用添加单位。 `For Example:`

``` css
.example {
    margin: 0;
    padding: 0;
}
```

### 开头的 0

值在 -1 和 1 之间时, 不需要加 0。 `For Example:`

``` css
.example {
    font-size: 0.8em;
}
```

### 16 进制表示法

 `For Example:`

``` css
/* Low */
.example {
    color: #eebbcc;
}

/* Good */
.example {
    color: #ebc;
}
```

### 前缀

使用带前缀的命名空间可以防止命名冲突, 同时提高代码可维护性。 `For Example:`

``` css
.adw-help {}

/* AdWords */
#maia-note {}

/* Maia */
```

### 选择器中使用连字符可以提高可读性。

 `For Example:`

``` css
/* Low: does not separate the words “demo” and “image” */
.demoimage {}

/* Low: uses underscore instead of hyphen */
.error_status {}

/* Good */
#video-id {}

.ads-sample {}
```

## CSS 格式规范

### CSS 书写顺序

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

### 块级内容缩进

为了反映层级关系和提高可读性, 块级内容都应缩进。 `For Example:`

``` css
@media screen,
projection {
    html {
        background: #fff;
        color: #444;
    }
}
```

### 声明结束

每行 CSS 都应以分号结尾。 `For Example:`

``` css
/* Low */
.test {
    display: block;
    height: 100px;
}

/* Good */
.test {
    display: block;
    height: 100px;
}
```

### 属性名结尾

属性名和值之间都应有一个空格。 `For Example:`

``` css
/* Low */
h3 {
    font-weight: bold;
}

/* Good */
h3 {
    font-weight: bold;
}
```

### 声明样式块的分隔

在选择器和 {} 之间用空格隔开。 `For Example:`

``` css
/* Low: missing space */
#video {
    margin-top: 1em;
}

/* Low: unnecessary line break */
#video {
    margin-top: 1em;
}

/* Good */
#video {
    margin-top: 1em;
}
```

### 选择器分隔

每个选择器都另起一行。 `For Example:`

``` css
/* Low */
a:focus,
a:active {
    position: relative;
    top: 1px;
}

/* Good */
h1,
h2,
h3 {
    font-weight: normal;
    line-height: 1.2;
}
```

### 规则分隔

规则之间都用空行隔开。 `For Example:`

``` css
html {
    background: #fff;
}

body {
    margin: auto;
    width: 50%;
}
```

### CSS 引号

属性选择器和属性值用单引号, URI 的值不需要引号。 `For Example:`

``` css
/* Low */
@import url("https://www.google.com/css/maia.css");

html {
    font-family: "open sans", arial, sans-serif;
}

/* Good */
@import url("//www.google.com/css/maia.css");

html {
    font-family: "open sans", arial, sans-serif;
}
```

## CSS 元规则

### 分段注释

用注释把 CSS 分成各个部分。 `For Example:`

``` css
/* Header */

#adw-header {}

/* Footer */

#adw-footer {}

/* Gallery */

.adw-gallery {}
```
