---
author: 星火燎原@vxhly
title: CSS 的新特性
categories: [web]
tags: [CSS, HTML5]
date: 2017-03-21 20:04:11
---

<Boxx changeTime="10000"/>

::: tip 前言
在众多的 CSS 新特性中, 有 3 个采用的新特性值得我们一起学习, 这里将记录这些新特性的使用方法。
:::
<!-- more -->

## 特性查询

特性查询, 使用 `@supports` 规则, 允许我们将 CSS 包含在一个条件块中, 这个条件块中会检查当前 `user agent` 是否支持一个 CSS 属性键值对, 在支持的情况下其中的内容才会生效。

这里有一个简单的例子, 只在浏览器支持 Flexbox 的时候应用 `display: flex`

``` css
@supports (display: flex) {
    .foo {
        display: flex;
    }
}
```

另外, 使用像 and 和 not 这样的运算符, 我们甚至可以创建更复杂的特性查询。

比如, 我们可以检测浏览器是否仅仅支持老的 Flexbox 语法 -

``` css
@supports (display: flexbox) and (not (display: flex)) {
    .foo {
        display: flexbox;
    }
}
```

### 支持情况

![CSS 的新特性](https://vxhly.github.io/assets/css-new-features-1.png)

## 表格布局

CSS 表格布局模块 所定义的系统用于创建基于表格的布局。 它与 `弹性盒布局` 模块相似, 但它更明确地为页面布局而设计, 它们之间存在许多不同的特性。

### 明确项目定位

网格由表格容器(通过 `display: grid` 创建)和表格项(子组件)组成。 在我们的 CSS 中, 我们可以清楚的组织表格项的位置和顺序, 而这与它们的标记(指 HTML 标记)无关。<br>

![CSS 的新特性](https://vxhly.github.io/assets/css-new-features-2.gif)

像上面这张图的布局, 它的背后的 CSS 代码是非常简单的

``` css
.hg__header {
    grid-area: header;
}

.hg__footer {
    grid-area: footer;
}

.hg__main {
    grid-area: main;
}

.hg__left {
    grid-area: navigation;
}

.hg__right {
    grid-area: ads;
}

.hg {
    display: grid;
    grid-template-areas:
        "header header header"
        "navigation main ads"
        "footer footer footer";
    grid-template-columns: 150px 1fr 150px;
    grid-template-rows:
        100px 1fr 30px;
    min-height: 100vh;
}

@media screen and (max-width: 600px) {
    .hg {
        grid-template-areas:
            "header"
            "navigation"
            "main"
            "ads"
            "footer";
        grid-template-columns: 100%;
        grid-template-rows:
            100px 50px 1fr 50px 30px;
    }
}
```

### 灵活的长度

`CSS Grid` 模块引入了一个新的长度单位, `fr` , 它代表了表格容器中剩余空间的一个等分。

它使得我们在表格容器根据可用空间来为表格项分配高度和宽度成为可能。 例如在圣杯布局( `Holy Grail Layout` )中, 我希望 main 分区占据两个边栏之外的所有空间。

``` css
.hg {
    grid-template-columns: 150px 1fr 150px;
}
```

### 间距

我们可以使用 `grid-row-gap` 、 `grid-column-gap` 和 `grid-gap` 属性来明确定义间距。 这些属性的值是 `<length-percentage(长度百分比)>` 数据类型 , 是内容区域尺寸的百分比。

比如, 需要 %5 的间距, 可以这样写

``` css
.hg {
    display: grid;
    grid-column-gap: 5%;
}
```

### 支持情况

![CSS 的新特性](https://vxhly.github.io/assets/css-new-features-3.png)

## 原生变量

最后要说的是原生 CSS 变量。 这个模块引入了一种由创建作者定义变量的方法, 这个变量可以像 CSS 属性一样赋值。

比如, 假设在我们的样式表中多处用到一个主题颜色, 我们可以将其抽象出来保存到变量中, 再引用变量, 以此代替多次直接写出实际的值。

``` css
:root {
    --theme-colour: cornflowerblue;
}

h1 {
    color: var(--theme-colour);
}

a {
    color: var(--theme-colour);
}

strong {
    color: var(--theme-colour);
}
```

目前在像 SASS 这样 CSS 预处理器中已经可以做到这些事情, 但 CSS 变量在浏览器中具有活性。 也就是说, 它们的值可以实时更新。 例如, 要改变上例中 `--theme-color` 属性的值, 我们只需要这样做

``` javascript
const rootEl = document.documentElement;
rootEl.style.setProperty("--theme-colour", "plum");
```

### 支持情况

![CSS 的新特性](https://vxhly.github.io/assets/css-new-features-4.png)
