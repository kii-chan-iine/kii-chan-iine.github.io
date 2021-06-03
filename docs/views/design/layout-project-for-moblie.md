---
author: 星火燎原@vxhly
title: 移动端网页自适应方案
categories: [design]
tags: [Layout, JavaScript, jQuery, Moblie]
date: 2016-08-13 19:40:34
---

<Boxx changeTime="10000"/>

::: tip 前言
解决移动端网页的自适应, 让开发变得更加简洁。
:::
<!-- more -->

## mate 标签

::: tip 提示
首先要让页面大小铺满屏幕又不能溢出。只需要在 html 的 head 标签内加入 viewport（如下）, 参数分别表示: 页面宽度＝屏幕宽度, 最大和最小伸缩比都是 1, 不允许用户拉缩。
:::

``` html
<meta name="viewport" content="width=device-width,initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, user-scalable=no">
```

* **width**: viewport 的宽度, 可以指定为一个像素值, 如: 640, 或者为特殊的值, 如: device-width （设备的宽度）。
* **initial-scale**: 初始缩放比例, 即当浏览器第一次加载页面时的缩放比例。值为 1.0 即原始尺寸。
* **maximum-scale**: 允许浏览者缩放到的最大比例, 一般设为 1.0, 即原始尺寸。
* **minimum-scale**: 允许浏览者缩放到的最小比例, 一般设为 1.0, 即原始尺寸。
* **user-scalable**: 浏览者是否可以手动缩放, yes 或 no 。

## 使用百分比自适应

::: tip 提示
把长度单位转换为百分比来表示, 这样在不同的宽度下, 元素的长宽也会随之变化。
:::

* 优点: 宽度之间无缝衔接, 操作起来也相对比较方便。
* 缺点: 字体大小需要另外一套自适应方法来调整；当屏幕宽度大于 `700px` 后, 继续按照百分比元素会偏大, 这个时候调整起来会比较麻烦。

## rem, em 自适应

::: tip 提示
用媒体查询的方法, 确定在不同屏幕宽度下, 改变 html 或 body 的 font-size。再用 rem , em 替代 px 作为单位实现自适应。
:::

* 优点: 可以根据不同屏幕宽度来设置, 可以完美解决上面说的屏幕偏大时的比例问题。字体的大小也不存在问题。
* 缺点: 根据宽度区间来设置, 无法实现无缝变换。

## 引入 JavaScript 代码

::: tip 提示
这些兼容方法各有优缺点, 都不算完美, 怎样才能把优点结合在一起, 同时避免缺点呢？ 引入 JavaScript 方法源码, 方法原理: 使页面 html 的 font-size 会根据屏幕的宽度自动调整, 而且屏幕宽度和所设字体大小的商是一定的, 然后按照固定比例缩小后作为 rem 的单位长度实现自适应。
:::

### 方法源码一（需引入 jQuery）

``` javascript
Zepto(function($) {
    // 获取文档的根节点
    var doc = document.documentElement;

    function setFontSize() {
        // 获取当前窗口的宽度
        var winWidth = $(window).width();
        // 640 宽度以上进行限制
        var size = winWidth / 640 * 100;
        doc.style.fontSize = (size < 100 ? size : 100) + "px";
    }

    // 防止在 html 未加载完毕时执行,保证获取正确的页宽
    setTimeout(function() {
        // 初始化
        setFontSize();
    }, 200);
});
```

### 方法源码二（JavaScript 原生代码）

``` javascript
(function(doc, win) {
    // 获取文档的根节点
    var docEl = doc.documentElement;
    // 判断移动端是否发生方向改变
    var resizeEvt =
        "orientationchange" in window ? "orientationchange" : "resize";

    var recalc = function() {
        // 获取对象的宽度
        var clientWidth = docEl.clientWidth;

        if (!clientWidth) return;

        if (clientWidth >= 640) {
            docEl.style.fontSize = "100px";
        } else {
            docEl.style.fontSize = 100 * (clientWidth / 640) + "px";
        }
    };

    if (!doc.addEventListener) return;

    win.addEventListener(resizeEvt, recalc, false);
    doc.addEventListener("DOMContentLoaded", recalc, false);
})(document, window);
```

## 常见问题

::: tip 提示
以上两个方法源码是 rem 布局的核心代码。 如果你的页面不需要引入 jQuery, 建议你使用 JavaScript 原生代码。 代码大意:  如果页面的宽度超过了 640px, 那么页面中 html 的 font-size 恒为 100px, 否则, 页面中 html 的 font-size 的大小为:  100 * (当前页面宽度 / 640)
:::

### 为什么是 640px？

对于手机屏幕来说, `640px` 的页面宽度是一个安全的最大宽度, 保证了移动端页面两边不会留白。注意这里的 `px` 是 `css` 逻辑像素, 与设备的物理像素是有区别的。 如果要切移动端页面, 你可以先把效果图宽度等比例缩放到 `640px` , 很好用。

### 为什么要设置 html 的 font-size？

`rem` 就是根元素（即: `html` ）的字体大小。 `html` 中的所有标签样式凡是涉及到尺寸的（如: `height` , `width` , `padding` , `margin` , `font-size` 。甚至 `left` , `top` 等）你都可以放心大胆的用 `rem` 作单位。 如果你把 `html` 的 `font-size` 设为 `20px` , 前面说过, `rem` 就是 `html` 的字体大小, 那么 `1rem = 20px` 。

### 模拟为什么要使用 iphone 4？

假设你部门的设计师给你的页面标准宽度为 `640px` , 则页面中所有的 `宽高` 全部除以 2 , Chrome 浏览器 模拟手机页面 设置手机型号为 `IPhone 4` 。 这也就是为什么要将所有的宽高除以 2 的原因: `IPhone 4` 标准 宽 是 `320 px` 。 然后正常的用 `px` 为单位 写你的 `css` , 你所需要做的就是你写的页面, 要在 `IPhone 4` 完全正确显示。 如果你说, 那设计那边给的不是 `640px` 标准怎么办？你只要让你的页面按照设计图在 `iphone 4` 上正确显示, 就木问题。

### rem 单位是如何换算的？

页面 `css` 全部写完之后, 将所有的 `px` 转换为以 `rem` 为单位的数字。 例如 我设置 `body` 的 `width: 320px；` 则根据设置的 `font - size` 值, 转换为 `rem` , 则是 `width: 16 rem；` 【320 / 20 (你设置的标准 `font - size` ) = 16】。

### 如何使用插件换算？

如果你说一个页面的 `css` 有几百行, 写完再去改太麻烦了, 那可以用 `px` 转换成 `rem` 的插件。 我这里有一款推荐: `cssrem` （下载使用网址 [cssrem](http://github.com/flashlizi/cssrem)）。 该插件是 `sublime text` 的一款插件。<br>

![cssrem](https://vxhly.github.io/assets/layout-moblie.gif)
