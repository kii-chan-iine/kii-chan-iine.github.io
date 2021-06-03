---
author: 星火燎原@vxhly
title: Emmet 用法举例
categories: [manual]
tags: [HTML5, CSS]
date: 2016-08-04 13:20:34
---

<Boxx changeTime="10000"/>

::: tip 前言
Emmet 的前身是大名鼎鼎的 Zen coding。如果你从事 Web 前端开发的话, 对该插件一定不会陌生。它使用仿 CSS 选择器的语法来代码, 大大提高了 HTML/CSS 代码编写的速度。
:::
<!-- more -->

## 子元素: >

### 例子

 `nav>ul>li`

``` html
<nav>
    <ul>
        <li></li>
    </ul>
</nav>
```

## 兄弟元素: +

### 例子

 `div+p+bq`

``` html
<div></div>
<p></p>
<blockquote></blockquote>
```

## 在某个元素之上: ^

### 例子一

 `div+div>p>span+em^bq`

``` html
<div></div>
<div>
    <p><span></span><em></em></p>
    <blockquote></blockquote>
</div>
```

### 例子二

 `div+div>p>span+em^^bq`

``` html
<div></div>
<div>
    <p><span></span><em></em></p>
</div>
<blockquote></blockquote>
```

## 分为一组: ()

### 例子一

 `div>(header>ul>li*2>a)+footer>p`

``` html
<div>
    <header>
        <ul>
            <li><a href=""></a></li>
            <li><a href=""></a></li>
        </ul>
    </header>
    <footer>
        <p></p>
    </footer>
</div>
```

### 例子二

 `(div>dl>(dt+dd)*3)+footer>p`

``` html
<div>
    <dl>
        <dt></dt>
        <dd></dd>
        <dt></dt>
        <dd></dd>
        <dt></dt>
        <dd></dd>
    </dl>
</div>
<footer>
    <p></p>
</footer>
```

## 倍增元素: *

### 例子

 `ul>li*5`

``` html
<ul>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
    <li></li>
</ul>
```

## 项目数字递增: $

### 例子一

 `ul>li.item$*5`

``` html
<ul>
    <li class="item1"></li>
    <li class="item2"></li>
    <li class="item3"></li>
    <li class="item4"></li>
    <li class="item5"></li>
</ul>
```

### 例子二

 `h$[title=item$]{Header $}*3`

``` html
<h1 title="item1">Header 1</h1>
<h2 title="item2">Header 2</h2>
<h3 title="item3">Header 3</h3>
```

### 例子三

 `ul>li.item$*5`

``` html
<ul>
    <li class="item001"></li>
    <li class="item002"></li>
    <li class="item003"></li>
    <li class="item004"></li>
    <li class="item005"></li>
</ul>
```

### 例子四

 `ul>li.item$@-*5`

``` html
<ul>
    <li class="item5"></li>
    <li class="item4"></li>
    <li class="item3"></li>
    <li class="item2"></li>
    <li class="item1"></li>
</ul>
```

### 例子五

 `ul>li.item$@3*5`

``` html
<ul>
    <li class="item3"></li>
    <li class="item4"></li>
    <li class="item5"></li>
    <li class="item6"></li>
    <li class="item7"></li>
</ul>
```

## ID 和 CLASS 属性

### 例子一

``` html
<div id="header"></div>
```

### 例子二

 `.title`

``` html
<div class=“title”></div>
```

### 例子三

 `footer#search.wide`

``` html
<footer id="search" class="wide"></footer>
```

### 例子四

 `p.class1.class2.class3`

``` html
<p class="class1 class2 class3"></p>
```

## 自定义属性

### 例子一

 `p[title="Hello world"]`

``` html
<p title="Hello world"></p>
```

### 例子二

 `td[rowspan=2 colspan=3 title]`

``` html
<td rowspan="2" colspan="3" title=""></td>
```

### 例子三

 `[a="value1" b="value2"]`

``` html
<div a="value1" b="value2"></div>
```

## 文本内容: {}

### 例子一

 `a{Click me}`

``` html
<a href="">Click me</a>
```

### 例子二

 `p>{Click }+a{here}+{ to continue}`

``` html
<p>Click <a href="">here</a> to continue</p>
```

## 缺省标签名

### 例子一

 `.class1`

``` html
<div class="class1"></div>
```

### 例子二

 `em>.class1`

``` html
<em><span class="class1"></span></em>
```

### 例子三

 `ul>.class1`

``` html
<ul>
    <li class="class1"></li>
</ul>
```

### 例子四

 `table>.row>.col`

``` html
<table>
    <tr class="row">
        <td class="col"></td>
    </tr>
</table>
```

# HTML

## HTML 基本模板

### doc

 `扩展写法`
 `html>(header>meta[charset=UTF-8]+title{${1:Document}})+body`

``` html
<html>

<head>
    <meta charset="UTF-8">
    <title>Document</title>
</head>

<body>

</body>

</html>
```

### ! 或者 html:5

 `扩展写法`
 `!!!+doc[lang=${lang}]`

``` html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <title>Document</title>
</head>

<body>

</body>

</html>
```

### doc4

 `扩展写法`
 `html>(head>meta[http-equiv="Content-Type" content="text/html;charset=${charset}"]+title{${1:Document}})+body`

``` html
<html>

<head>
    <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
    <title>Document</title>
</head>

<body>

</body>

</html>
```

### html:4t

 `扩展写法`
 `!!!4t+doc4[lang=${lang}]`

``` html
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "https://www.w3.org/TR/html4/loose.dtd">
<html lang="en">

<head>
    <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
    <title>Document</title>
</head>

<body>

</body>

</html>
```

### html:4s

 `扩展写法`
 `!!!4s+doc4[lang=${lang}]`

``` html
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "https://www.w3.org/TR/html4/strict.dtd">
<html lang="en">

<head>
    <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
    <title>Document</title>
</head>

<body>

</body>

</html>
```

## HTML 基本元素

### A

#### a:link

``` html
<a href="//"></a>
```

#### a:mail

``` html
<a href="mailto:"></a>
```

#### area

``` html
<area shape="" coords="" href="" alt="">
```

#### area:d

``` html
<area shape="default" coords="" href="" alt="">
```

#### area:c

``` html
<area shape="circle" coords="" href="" alt="">
```

#### area:r

``` html
<area shape="rect" coords="" href="" alt="">
```

#### area:p

``` html
<area shape="poly" coords="" href="" alt="">
```

### B

#### btn:r

``` html
<button type="reset"></button>
```

#### btn:s

``` html
<button type="submit"></button>
```

### F

#### form:get

``` html
<form action="get"></form>
```

#### form:post

``` html
<form action="post"></form>
```

### L

#### link

``` html
<link rel="stylesheet" href="">
```

#### link:css

``` html
<link rel="stylesheet" href="style.css">
```

#### link:print

``` html
<link rel="stylesheet" href="print.css" media="print">
```

#### link:favicon

``` html
<link rel="shortcut icon" type="image/x-icon" href="favicon.ico">
```

#### link:touch

``` html
<link rel="apple-touch-icon" href="favicon.png">
```

### I

#### img

``` html
<img src="" alt="">
```

#### input

``` html
<input type="text">
```

#### inp

这种写法也适用: `input:text input:t`

``` html
<input type="text" name="" id="">
```

#### input:hideen

这种写法也适用: `input:h`

``` html
<input type="hidden" name="">
```

#### input:search

``` html
<input type="search" name="" id="">
```

#### input:email

``` html
<input type="email" name="" id="">
```

#### input:url

``` html
<input type="url" name="" id="">
```

#### input:password

这种写法也适用: `input:p`

``` html
<input type="password" name="" id="">
```

#### input:datetime

``` html
<input type="datetime" name="" id="">
```

#### input:date

``` html
<input type="date" name="" id="">
```

#### input:datetime-local

``` html
<input type="datetime-local" name="" id="">
```

#### input:month

``` html
<input type="month" name="" id="">
```

#### input:week

``` html
<input type="week" name="" id="">
```

#### input:time

``` html
<input type="time" name="" id="">
```

#### input:number

``` html
<input type="number" name="" id="">
```

#### input:color

``` html
<input type="color" name="" id="">
```

#### input:checkbox

这种写法也适用: `input:c`

``` html
<input type="checkbox" name="" id="">
```

#### input:radio

这种写法也适用: `input:r`

``` html
<input type="radio" name="" id="">
```

#### input:range

``` html
<input type="range" name="" id="">
```

#### input:file

这种写法也适用: `input:f`

``` html
<input type="file" name="" id="">
```

#### input:submit

这种写法也适用: `input:s`

``` html
<input type="submit" value="">
```

#### input:image

这种写法也适用: `input:i`

``` html
<input type="image" src="" alt="">
```

#### input:button

这种写法也适用: `input:b`

``` html
<input type="button" value="">
```

#### input:reset

``` html
<input type="reset" value="">
```

### M

#### meta:utf

``` html
<meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
```

#### meta:win

``` html
<meta http-equiv="Content-Type" content="text/html;charset=windows-1251">
```

#### meta:vp

``` html
<meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minmum-scale=1.0">
```

#### meta:compat

``` html
<meta http-equiv="X-UA-Compatible" content="IE=7">
```

#### menu:context

这种写法也适用: `menu:c`

``` html
<menu type="context"></menu>
```

#### menu:toolbar

这种写法也适用: `menu:t`

``` html
<menu type="toolbar"></menu>
```

# CSS

## 视觉格式

### position

* **pos** => `position: relative;`
* **pos:s** => `position: static;`
* **pos:a** => `position: relative;`
* **pos:r** => `position: relative;`
* **pos:f** => `position: fixed;`
* **t** => `top: ;`
* **t:a** => `top: auto;`
* **l** => `left: ;`
* **l:a** => `left: auto;`
* **b** => `bottom: ;`
* **b:a** => `bottom: auto;`
* **r** => `right: ;`
* **r:a** => `right: auto;`
* **z** => `z-indet: ;`
* **z:a** => `z-indet: auto;`

### float

* **fl** => `float: left;`
* **fl:n** => `float: none;`
* **fl:l** => `float: left;`
* **fl:r** => `float: right;`
* **cl** => `clear: both;`
* **cl:n** => `clear: none;`
* **cl:l** => `clear: left;`
* **cl:r** => `clear: right;`
* **cl:b** => `clear: both;`

### dispaly

* **d** => `display: block;`
* **d:n** => `display: none;`
* **d:b** => `display: block;`
* **d:i** => `display: inline;`
* **d:ib** => `display: inline-block;`
* **d:f** => `display: flex;`
* **d:if** => `display: inline-flex;`
* **d:li** => `display: list-item;`
* **d:ri** => `display: run-in;`
* **d:cp** => `display: compat;`
* **d:tb** => `display: table;`
* **d:itb** => `display: inline-table;`
* **d:tbcp** => `display: table-caption;`
* **d:tbcl** => `display: table-column;`
* **d:tbclg** => `display: table-column-group;`
* **d:tbhg** => `display: table-header-group;`
* **d:tbfg** => `display: table-footer-group;`
* **d:tbr** => `display: table-row;`
* **d:tbrg** => `display: table-row-group;`
* **d:tbc** => `display: table-cell;`

### visibility

* **v** => `visibility: hidden;`
* **v:v** => `visibility: visible;`
* **v:h** => `visibility: hidden;`
* **v:c** => `visibility: collapse;`

### overflow

* **ov** => `overflow: hidden;`
* **ov:v** => `overflow: visible;`
* **ov:h** => `overflow: hidden;`
* **ov:s** => `overflow: scroll;`
* **ov:a** => `overflow: auto;`
* **ovx** => `overflow-x: hidden;`
* **ovx:v** => `overflow-x: visible;`
* **ovx:h** => `overflow-x: hidden;`
* **ovx:s** => `overflow-x: scroll;`
* **ovx:a** => `overflow-x: auto;`
* **ovy** => `overflow-x: hidden;`
* **ovy:v** => `overflow-y: visible;`
* **ovy:h** => `overflow-y: hidden;`
* **ovy:s** => `overflow-y: scroll;`
* **ovy:a** => `overflow-y: auto;`
* **ovs** => `overflow-style: scrollbar;`
* **ovs:a** => `overflow-style: auto;`
* **ovs:s** => `overflow-style: scrollbar;`
* **ovs:p** => `overflow-style: panner;`
* **ovs:m** => `overflow-style: move;`
* **ovs:mq** => `overflow-style: marquee;`

### clip

* **cp** => `clip: ;`
* **cp:a** => `clip: auto;`
* **cp:r** => `clip: rect(top right bottom left);`

### resize

* **rsz** => `resize: ;`
* **rsz:n** => `resize: none;`
* **rsz:b** => `resize: both;`
* **rsz:h** => `resize: horizontal;`
* **rsz:v** => `resize: vertical;`

### cursor

* **cur** => `cursor: pointer;`
* **cur:a** => `cursor: auto;`
* **cur:d** => `cursor: default;`
* **cur:c** => `cursor: crosshair;`
* **cur:ha** => `cursor: hand;`
* **cur:he** => `cursor: help;`
* **cur:m** => `cursor: move;`
* **cur:p** => `cursor: pointer;`
* **cur:t** => `cursor: text;`

## 外边距和内边距

### margin && padding

* **m** => `margin: ;`
* **m:a** => `margin: auto;`
* **mt** => `margin-top: ;`
* **mt:a** => `margin-top: auto;`
* **mr** => `margin-right: ;`
* **mr:a** => `margin-right: auto;`
* **mb** => `margin-bottom: ;`
* **mb:a** => `margin-bottom: auto;`
* **ml** => `margin-left: ;`
* **ml-a** => `margin-left: auto;`
* **p** => `padding: ;`
* **pt** => `padding-top: ;`
* **pr** => `padding-right: ;`
* **pb** => `paddng-bottom: ;`
* **pl** => `padding-left: ;`

## 盒子模型

### box

* **bxz** => `box-sizing: border-box;`
* **bxz:cb** => `box-sizing: content-box;`
* **bxz:bb** => `box-sizing: border-box;`
* **bxsh** => `box-shadow: inset hoff voff blur color;`
* **bxsh:r** => `box-shadow: inset hoff voff blur spread rgb(0, 0, 0);`
* **bxsh:ra** => `box-shadow: inset h v blur spread rgba(0, 0, 0, .5);`
* **bxsh:n** => `box-shadow: none;`

### width && hight

* **w** => `width: ;`
* **w:a** => `width: auto;`
* **h** => `hight: ;`
* **h:a** => `hight: auto;`
* **maw** => `max-width: ;`
* **maw:n** => `max-width: none;`
* **mah** => `max-hight: ;`
* **mah:n** => `max-hight: none;`
* **miw** => `min-width: ;`
* **mih** => `min-hight: ;`

## 字体和文本

### font

* **f** => `font: ;`
* **f+** => `font: 1em Arial,sans-serif;`
* **fw** => `font-weight: ;`
* **fw:n** => `font-weight: none;`
* **fw:b** => `font-weight: bold;`
* **fw:br** => `font-weight: bolder;`
* **fw:lr** => `font-weight: lighter;`
* **fs** => `font-style: italic;`
* **fs:n** => `font-style: normal;`
* **fs:i** => `font-style: italic;`
* **fs:o** => `font-style: oblique;`
* **fv** => `font-variant: ;`
* **fv:n** => `font-variant: normal;`
* **fv:sc** => `font-variant: small-caps;`
* **fz** => `font-size: ;`
* **fza** => `font-size-adjust: ;`
* **fza:n** => `font-size-adjust: none;`
* **ff** => `font-family: ;`
* **ff:s** => `font-family: serif;`
* **ff:ss** => `font-family: sans-serif;`
* **ff:c** => `font-family: cursive;`
* **ff:f** => `font-family: fantasy;`
* **ff:m** => `font-family: monospace;`
* **ff:a** => `font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;`
* **fef** => `font-effect: ;`
* **fef:n** => `font-effect: none;`
* **fef:eg** => `font-effect: engrave;`
* **fef:eb** => `font-effect: emboss;`
* **fef:o** => `font-effect: outline;`
* **@i** => `@import url();`
* **@f** =>

  

``` css
  @font-face {
      font-family: ;
      src: url();
  }
```

* **@f+** =>

  

``` css
  @font-face {
      font-family: "FontName";
      src: url("FileName.eot");
      src: url("FileName.eot?#iefix") format("embedded-opentype"), url("FileName.woff") format("woff"), url("FileName.ttf") format("truetype"), url("FileName.svg#FontName") format("svg");
      font-style: normal;
      font-weight: normal;
  }
```

### vertical

* **va** => `vertical-align: top;`
* **va:sup** => `vertical-align: super;`
* **va:t** => `vertical-align: top;`
* **va:tt** => `vertical-align: text-top;`
* **va:m** => `vertical-align: middle;`
* **va:bl** => `vertical-align: baseline;`
* **va:b** => `vertical-align: bottom;`
* **va:tb** => `vertical-align: text-bottom;`
* **va:sub** => `vertical-align: sub;`

### text

* **ta** => `text-align: left;`
* **ta:l** => `text-align: left;`
* **ta:c** => `text-align: center;`
* **ta:r** => `text-align: right;`
* **ta:j** => `text-align: justify;`
* **ta-lst** => `text-align-last: ;`
* **tal:a** => `text-align-last: auto;`
* **tal:l** => `text-align-last: left;`
* **tal:c** => `text-align-last: center;`
* **tal:r** => `text-align-last: right;`
* **tov** => `text-overflow: ellipsis;`
* **tov:c** => `text-overflow: clip;`
* **tov:e** => `text-overflow: ellipsis;`
* **td** => `text-decoration: none;`
* **td:n** => `text-decoration: none;`
* **td:u** => `text-decoration: underline;`
* **td:o** => `text-decoration: overline;`
* **td:l** => `text-decoration: line;`

## CSS 3 动画

### keyframes

* **@kf** =>

  

``` css
  @-webkit-keyframes identifier {
      from {}

      to {}
  }

  @-o-keyframes identifier {
      from {}

      to {}
  }

  @-moz-keyframes identifier {
      from {}

      to {}
  }

  @keyframes identifier {
      from {}

      to {}
  }
```

### animation

* **anim-** => `animation: name duration timing-function delay iteration-count direction fill-mode;`
* **animdel** => `animation-delay: time;`
* **animdir** => `animation-direction: normal;`
* **animdir:a** => `animation-direction: alternate;`
* **animdir:ar** => `animation-direction: alternate;`
* **animdir:n** => `animation-direction: normal;`
* **animdir:r** => `animation-direction: reverse;`

### transform

* **trf** => `transform: ;`
* **trf:r** => `transform: rotate(angle);`
* **trf:sc** => `transform: scale(x, y);`
* **trf:scx** => `transform: scale(x);`
* **trf:scy** => `transform: scale(y);`
* **trf:skx** => `transform: skewX(angle);`
* **trf:sky** => `transform: skewY(angle);`
* **trf:t** => `transform: translate(x, y);`
* **trf:tx** => `transform: translateX(x);`
* **trf:ty** => `transform: translateY(y);`
* **trfo** => `transform-origin: ;`
* **trfs** => `transform-style: preserve-3d;`

### transition

* **trs** => `transition: prop time;`
* **trsde** => `transition-delay: time;`
* **trsdu** => `transition-duration: time;`
* **trsp** => `transition-property: prop;`
* **trstf** => `transition-timing-function: tfunc;` # 更多用法实例

在线查看查看 [emmet-usage-examples.jpg](http://oss-blog.test.upcdn.net/emmet-usage-examples.jpg)<br>
请下载查看 [emmet-usage-examples.jpg](http://oss-blog.test.upcdn.net/emmet-usage-examples.jpg?attname=)
