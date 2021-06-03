---
author: 星火燎原@vxhly
title: CSS 选择器介绍
categories: [web]
tags: [CSS]
date: 2017-03-03 19:52:30
---

<Boxx changeTime="10000"/>

::: tip 前言
CSS 选择器是 WEB 开发中一个重要部分, 通过 CSS 选择器, 可以提高开发人员的工作效率。 在本篇文章中将介绍属性选择器和伪类选择器的基本用法。
:::
<!-- more -->

## 属性选择器(字符串匹配)

::: tip 作用
属性选择器其主要作用是为待遇制定属性的 HTML 元素设置样式。 例如, 通过制定 div 元素的 id 属性, 设置相关样式
:::

属性选择器一共分为 4 种匹配模式选择器: 

* **完全匹配属性选择器**
* **包含匹配选择器**
* **首字符匹配选择器**
* **尾字符匹配选择器**

### 完全匹配属性选择器

其含义就是完全匹配字符串。 当 div 元素的 id 属性值为 test 时, 利用完全匹配选择器选择任何 id 值 为 test 的元素都使用该样式。 如下代码通过指定 id 值将属性设定为红色字体: 

``` html
<div id="article">测试完全匹配属性选择器</div>

<style type="text/css">
    [id=article] {
        color: red;
    }
</style>
```

### 包含匹配选择器

包含匹配比完全匹配范围更广。 只要元素中的属性包含有指定的字符串, 元素就使用该样式。

其语法是: `[attribute*=value]` 。 其中 `attribute` 指的是属性名, `value` 指的是属性值, 包含匹配采用 `“*=”` 符号。

例如下三个 div 元素都符合匹配选择器的选择, 并将 div 元素内的字体设置为红色字体: 

``` html
<div id="article">测试包含匹配属性选择器</div>
<div id="subarticle">测试包含匹配属性选择器</div>
<div id="article1">测试包含匹配属性选择器</div>

<style type="text/css">
    [id*=article] {
        color: red;
    }
</style>
```

### 首字符匹配选择器

首字符匹配就是匹配属性值开头字符, 只要开头字符符合匹配, 则元素使用该样式。

其语法是: `[attribute^=value]` 。 其中 `attribute` 指的是属性名, `value` 指的是属性值, 包含匹配采用 `“^=”` 符号。

例如下三个 div 元素使用首字符匹配选择器后, 只有 id 为 `article` 和 `article1` 的元素才被设置为红色字体: 

``` html
<div id="article">测试首字符匹配选择器</div>
<div id="subarticle">测试首字符匹配选择器</div>
<div id="article1">测试首字符匹配选择器</div>

<style type="text/css">
    [id^=article] {
        color: red;
    }
</style>
```

### 尾字符匹配选择器

尾子符匹配跟首字符匹配原理一样。 尾子符只匹配结尾的字符串, 只要结尾字符串符合匹配, 则元素使用该样式。

其语法是: `[attribute$=value]` 。 其中 `attribute` 指的是属性名, `value` 指的是属性值, 包含匹配采用 `“$=”` 符号。

例如下三个 div 元素使用首字符匹配选择器后, 只有 id 为 `article` 和 `subarticle` 的元素才被设置为红色字体: 

``` html
<div id="article">测试首字符匹配选择器</div>
<div id="subarticle">测试首字符匹配选择器</div>
<div id="article1">测试首字符匹配选择器</div>

<style type="text/css">
    [id$=article] {
        color: red;
    }
</style>
```

## 属性选择器（包含匹配）

### [attr]

::: tip 作用
匹配 html 属性, 可自定义属性
:::

``` html
<div flex>测试</div>

<style type="text/css">
    [flex] {
        display: flex;
    }
</style>
```

### [attr=val]

::: tip 作用
属性值一定只能等于某个字符串
:::

``` html
<div flex="inline">测试</div>

<style type="text/css">
    [flex="inline"] {
        display: inline-flex;
    }
</style>
```

### [attr~=val]

::: tip 作用
属性值包含其中一个字符串
:::

``` html
<div flex="dir:left">测试</div>
<div flex="dir:left main:center">测试</div>

<style type="text/css">
    [flex] {
        display: flex;
    }

    [flex~='dir:left'] {
        flex-direction: row;
    }

    [flex~='main:center'] {
        justify-content: center;
    }
</style>
```

### [attr|=val]

::: tip 作用
属性值只能匹配以 `-` 分割的字符串
:::

``` html
<div id="line-1-2">测试</div>
<div id="line-1-4">测试</div>

<style type="text/css">
    [id|='line-1'] {
        display: flex;
    }
</style>
```

## 后代选择器

``` html
<div class="content">
    <p>测试</p>
</div>

<style type="text/css">
    .content p {
        color: red;
    }
</style>
```

## 并列选择器

``` html
<div class="content">
    <p>测试</p>
</div>

<style type="text/css">
    .content,
    p {
        font-size: 16px;
    }
</style>
```

## 子元素选择器

::: tip 作用
范围比后代选择器小, 只选中直系子元素
:::

``` html
<div class="content">
    <div>
        测试
        <div class="test">测试2</div>
    </div>
</div>

<style type="text/css">
    .test {
        color: red;
        line-height: 12px;
        font-size: 12px;
    }

    .content>div {
        font-size: 20px;
    }
</style>
```

## 相邻兄弟选择器

``` html
<div class="content">
    <div>
        测试
        <div class="test">测试2</div>
    </div>
</div>
<div class="content2">
    测试3
</div>

<style type="text/css">
    .content+.content2 {
        font-size: 20px;
    }
</style>
```

## 伪类选择器

在 CSS3 选择器中, 伪类选择器种类非常多。 然后在 CSS2.1 时代, 伪类选择器就已经存在, 例如超链接的四个状态选择器: `a:link` 、 `a:visited` 、 `a:hover` 、 `a:active` 。

* `a:link` => 指普通的、 未被访问的链接
* `a:visited` => 指用户已访问的链接
* `a:hover` => 表示鼠标悬停在链接上方
* `a:active` => 链接被点击时刻

CSS3 增加了非常多的选择器, 其中包括: 

* `first-line` 伪元素选择器
* `first-letter` 伪元素选择器
* `not` 选择器
* `root` 选择器
* `empty` 选择器
* `target` 选择器

### focus

foucus 伪类元素选择器用于选取获得焦点的元素

``` css
input:focus {
    background-color: yellow;
}
```

### first-child

指定元素列表中第一个元素的样式。 语法如下: 

``` css
li:first-child {
    color: red;
}
```

### last-child

和 `first-child` 是同类型的选择器。 `last-child` 指定元素列表中最后一个元素的样式。 语法如下: 

``` css
li:last-child {
    color: red;
}
```

### nth-child 和 nth-last-child

`nth-child` 和 `nth-last-child` 可以指定某个元素的样式或者从后数起某个元素的样式。 例如: 

``` css
/* 指定第 2 个元素 */
li:nth-child(2) {}

/* 指定倒数第 2 个元素 */
li:nth-last-child(2) {}

/* 指定偶数个元素 */
li:nth-child(even) {}

/* 指定奇数个元素 */
li:nth-child(odd) {}
```

## 伪元素选择器

### before 选择器

before 伪类元素选择器主要作用是在选择某个元素之前插入内容, 一般用于清除浮动。

目前, before 选择器得到支持的浏览器包括: `IE8+` 、 `Firefox` 、 `Chrome` 、 `Safari` 、 `Opera` 、 `Android Browser` 、 `iOS Safari` 。

befor 选择器的语法是: 

``` css
元素标签:before {
    content: "插入内容";
}
```

例如, 在 p 元素之前插入 "文字": 

``` css
p:before {
    content: "文字";
}
```

### after

after 伪类元素选择器和 before 伪类元素选择器原理一样, 但 after 是在选择某个元素之后插入内容。

目前, before 选择器得到支持的浏览器包括: `IE8+` 、 `Firefox` 、 `Chrome` 、 `Safari` 、 `Opera` 、 `Android Browser` 、 `iOS Safari` 。

after 选择器的语法是: 

``` css
元素标签:after {
    content: "插入内容";
}
```

例如, 在 p 元素之后插入 "文字": 

``` css
p:after {
    content: "文字";
}
```

### first-letter

first-letter 可以选中首字符

``` html
<p>测试测试</p>

<style type="text/css">
    p:first-letter {
        font-size: 200%;
        color: #dedede;
    }
</style>
```

### first-line

first-line 可以选中首行

``` html
<p>测试测试</p>

<style type="text/css">
    p:first-line {
        background-color: #dedede;
    }
</style>
```
