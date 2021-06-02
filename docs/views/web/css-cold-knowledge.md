---
author: 星火燎原@vxhly
title: CSS 的一些冷知识
categories: [web]
tags: [Layout, CSS, Code]
date: 2016-08-17 17:27:08
---

<Boxx changeTime="10000"/>

::: tip 前言
记录常用而不易于记忆的 css 自定义代码。
:::
<!-- more -->

## 常见实用技巧

``` css
/*禁止长按链接与图片弹出菜单*/
a,
img {
    -webkit-touch-callout: none;
}

/*禁止ios和android用户选中文字*/
html,
body {
    -webkit-user-select: none;
    user-select: none;
}

/*改变输入框placeholder的颜色值*/
::-webkit-input-placeholder {
    /* WebKit browsers */
    color: #999;
}

:-moz-placeholder {
    /* Mozilla Firefox 4 to 18 */
    color: #999;
}

::-moz-placeholder {
    /* Mozilla Firefox 19+ */
    color: #999;
}

:-ms-input-placeholder {
    /* Internet Explorer 10+ */
    color: #999;
}

input:focus::-webkit-input-placeholder {
    color: #999;
}

/*android上去掉语音输入按钮*/
input::-webkit-input-speech-button {
    display: none;
}

/* 消除 transition 闪屏 */
.css {
    -webkit-transform-style: preserve-3d;
    -webkit-backface-visibility: hidden;
    -webkit-perspective: 1000;
}
```

## placeholder

在 `input` 标签中设置 `placeholder` 属性时, 有时候因为需求, 要修改占位符的默认颜色或者字体大小, 这是就可以用下面的 `css` , `For Example:`

``` css
/* firefox */
input::-moz-placeholder {
    color: red;
    font-size: 18px;
}

/* IE */
input:-ms-input-placeholder {
    color: red;
    font-size: 18px;
}

/* chrome */
input::-webkit-input-placeholder {
    color: red;
    font-size: 18px;
}
```

需要注意的是不同浏览器写法不同: 

* 都要加上各自浏览器的前缀(如 `-webkit-` )
* `firefox` 的 `placeholder` 的前面没有 `input-`
* `firefox` 与 `chrome` 都是 `::` 两个冒号, 而 IE 则是一个 `:`
* 低版本的浏览器与新版本浏览器可能写法不同

## 下拉框的小三角

`select` 标签会出现小三角, 通常这个小三角都会去掉, 或者用背景图片的方式替换为符合要求的样子。 去掉小三角的 `css` , `For Example:`

``` css
select {
    -webkit-appearance: none;
    /* chrome */
    -moz-appearance: none;
    /* firefox */
    /* 在IE浏览器中目前还没找到可以去掉小三角的方法。 */
}
```

## input[type=number] 右边的 spinners

`nput[type=number]` 通常用在移动端设备上, 浏览器会识别 `number` 输入类型, 然后改变数字键盘来适应它。 但是它会出现 `spinners` , 一般不需要它。 去掉 `spinners` 的 `css` 如下: 

``` css
/* firefox */
input[type="number"] {
    -moz-appearance: textfield;
}

/* chrome */
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}
```

## -webkit-tap-highlight-color

在移动端浏览器上(如微信、 QQ 内置浏览器), 当你点击一个链接或者通过 `JavaScript` 定义的可点击元素的时候, 会出现蓝色边框, 我是很讨厌这个边框的, 所以一般会这样去除, `For Example:`

``` css
a,
button,
input,
textarea {
    -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
    -webkit-user-modify: read-write-plaintext-only; //-webkit-user-modify有个副作用, 就是输入法不再能够输入多个字符
}

/*也可以..., 简单粗暴*/

* {

    -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
```

将高亮色设为透明, 这样就看不到蓝色边框了。

## 滚动条

`webkit` 现在支持拥有 `overflow` 属性的区域, `列表框` , `下拉菜单` , `textarea` 的滚动条自定义样式。 有时候需要把滚动条去掉, 特别是页面中出现几条滚动条的时候, `For Example:`

``` css
::-webkit-scrollbar {
    width: 0;
    /* 这将去除整个页面的滚动条 */
}
```

设置滚动条的宽度为 `0` 就可以去除滚动条了。

## 文字换行

 `For Example:`

``` css
.text-warp {
    /* 强制不换行 */
    white-space: nowrap;

    /* 自动换行 */
    word-wrap: break-word;
    word-break: normal;

    /* 强制英文单词断行 */
    word-break: break-all;
}
```

## 文字两端对齐

 `For Example:`

``` css
.text-justify {
    text-align: justify;
    text-justify: inter-ideogra;
}
```

## 去掉色焦点框

去掉 Webkit(chrome) 浏览器中 input(文本框) 或 textarea 的黄色焦点框, `For Example:`

``` css
input,
button,
select,
textarea {
    outline: none;
}

textarea {
    font-size: 13px;
    resize: none;
}
```

## 去掉黄色背景

chrome 表单自动填充后, input 文本框的背景会变成黄色的, 通过审查元素可以看到这是由于 chrome 会默认给自动填充的 input 表单加上 input:-webkit-autofill 私有属性, 然后对其赋予以下样式: 

 `For Example:`

``` css
input: -webkit-autofill {
    background-color: #faffbd;
    background-image: none;
    color: #000;
}
```

在有些情况下, 这个黄色的背景会影响到我们界面的效果。 解决方法请往下看。

## 让 input 文本框是纯色背景的

可以对 `input:-webkit-autofill` 使用足够大的纯色内阴影来覆盖 `input` 输入框的黄色背景, `For Example:`

``` css
input: -webkit-autofill {
    -webkit-box-shadow: 0 0 0 1000 white inset;
    border: 1px solid #ccc !important;
}
```

如果你有使用圆角等属性, 或者发现输入框的长度高度不太对, 可以对其进行调整, 除了 `chrome` 默认定义的 `background-color` , `background-image` , `color` 不能用 `!important` 提升其优先级以外, 其他的属性均可使用 `!important` 提升其优先级, `For Example:`

``` css
input: -webkit-autofill {
    -webkit-box-shadow: 0 0 0 1000 white inset;
    border: 1px solid #ccc !important;
    height: 27px !important;
    line-height: 27px !important;
    border-radius: 0 4px 4px 0;
}
```

## input 文本框是使用图片作为背景的

这个比较麻烦, 目前还没找到完美的解决方法, 有两种选择: 

* 如果你的图片背景不太复杂, 只有一些简单的内阴影, 那个人觉得完全可以使用上面介绍的方法用足够大的纯色内阴影去覆盖掉黄色背景, 此时只不过是没有了原来的内阴影效果罢了。
* 如果你实在想留住原来的内阴影效果, 那就只能牺牲 chrome 自动填充表单的功能, 使用 `JavaScript` 去实现, `For Example:`

``` javascript
$(function() {
    if (navigator.userAgent.toLowerCase().indexOf("chrome") >= 0) {
        $(window).load(function() {
            $("ul input:not(input[type=submit])").each(function() {
                var outHtml = this.outerHTML;
                $(this).append(outHtml);
            });
        });
    }
});
```

## 关闭自动填充表单功能

如果你既不想使用 `JavaScript` , 也不想用 `css` , 好吧, 在 `form` 标签上直接关闭了表单的自动填充功能: `autocomplete="off"` 。 `For Example:`

``` html
<from action="" autocomplete="off">
    ...
</from>
```

## seperate-table

 `For Example:`

``` css
.tab {
    border-collapse: separate;
    border: 1px solid #e0e0e0;
}

.tab th,
.tab td {
    padding: 3px;
    font-size: 12px;
    background: #f5f9fb;
    border: 1px solid;
    border-color: #fff #deedf6 #deedf6 #fff;
}

.tab th {
    background: #edf4f0;
}

.tab tr.even td {
    background: #fff;
}
```

 `For Example:`

``` html
<table class="tab" width="100%" cellpadding="0" cellspacing="0" border="0">
    <tr>
        <th>111</th>
        <td>222</td>
    </tr>
    <tr>
        <th>111</th>
        <td>222</td>
    </tr>
</table>
```

[codepen 调试地址](http://codepen.io/anon/pen/zBbRkJ?editors=1100#0)

## 清除浮动, 万能的 float 闭合

给需要闭合的 DIV（class 为 clearfix） 加上如下的 CSS 样式即可, `For Example:`

``` css
.clearfix:after {
    content: ".";
    display: block;
    height: 0;
    clear: both;
    visibility: hidden;
}

*html .clearfix {
    height: 1%;
}

*+html .clearfix {

    height: 1%;
}

.clearfix {
    display: inline-block;
}

.clearfix {
    display: block;
}
```

不过貌似这样有点麻烦, 呵呵, 大前端常用的是下面的, `For Example:`

``` css
.clear {
    clear: both;
    height: 0;
    overflow: hidden;
}
```

上诉办法是在需要清除浮动的地方加个 div.clear 或者 br.clear, 我们知道这样能解决基本清浮动问题。 但是这种方法的最大缺陷就是改变了 html 结构, 虽然只是加个 div
