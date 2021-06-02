---
author: 星火燎原@vxhly
title: 简单而实用的 jQuery 小技巧
categories: [web]
tags: [JavaScript, jQuery, Code]
date: 2016-08-09 15:26:35
---

<Boxx changeTime="10000"/>

::: tip 前言
简单的窍门的集合, 用来帮助你 jQuery 代码的进阶。
:::
<!-- more -->

## 判断 jQuery 是否被正确加载

在用 jQuery 来写一些脚本时, jQuery 必须要被正确加载, 接下来写的脚本才有意义。

``` javascript
if (typeof jQuery == "undefined") {
    console.log("jQuery hasn't loaded");
} else {
    console.log("jQuery has loaded");
}
```

## 禁用右键功能

``` javascript
$(function() {
    $(document).bind("contextmenu", function(e) {
        return false;
    });
});
```

## 回到顶部

利用 jQuery 里的 animate 和 scrollTop 方法, 你便不需要使用插件创建简单的滚动到顶部动画。

 `javascript 代码`

``` javascript
// Back to top
$(".top").click(function(e) {
    e.preventDefault(); // 阻止元素的默认行为
    $("html, body").animate({
        scrollTop: 0
    }, 800); // 用多长时间滚动到何处
});
```

 `html 代码`

``` html
<!-- Create an anchor tag -->
<a class="top" href="#">Back to top</a>
```

通过 scrollTop 的值来改变你想要滚动到的位置。其实你就是做了: 在接下来的 800 毫秒中让页面滚动, 直到它滚动到文档的顶部。

## 一句话搞定 checkbox 全选和全不选

在网站建设中；让复选框全部选中和全部取消是比较常用的；下面就介绍如何用一句话搞定全选功能。

 `javascript 代码`

``` javascript
function checkAll(obj) {
    $("#box input[type='checkbox']").prop("checked", $(obj).prop("checked"));
}
```

 `html 代码`

``` html
<div id="box">
    <input type="checkbox" onclick="checkAll(this)">全选<br><br>
    <input type="checkbox"><br>
    <input type="checkbox"><br>
    <input type="checkbox"><br>
    <input type="checkbox"><br>
    <input type="checkbox"><br>
    <input type="checkbox"><br>
    <input type="checkbox"><br>
    <input type="checkbox"><br>
    <input type="checkbox"><br>
</div>
```

## 图片的预加载

如果你的网页使用了很多隐藏图片文件（例如: 鼠标悬停展示的图片）, 那么图片的预加载是有意义的。

``` javascript
// 扫描网站上的图片
$.preloadImages = function() {
    for (var i = 0; i < arguments.length; i++) {
        $("<img>").attr("src", arguments[i]);
    }
};

// 预加载
$.preloadImages("img/hover-on.png", "img/hover-off.png");
```

## 判断图片是否加载完

有时候你可能需要检查图像是否已经加载完成, 以便于可以继续执行相应的 JavaScript 代码。

``` javascript
$("img").load(function() {
    console.log("image load successful");
});
```

你还可以检查一个特定的图片是否加载, 只需引用该 img 标签的 class 或者 id。

## 自动修补破损图像

如果你碰巧发现在你的网站上发现破损的图像链接, 一个个去替代他们是痛苦的。这个简单的代码可以节省很多的麻烦。

``` javascript
$("img").on("error", function() {
    if (!$(this).hasClass("broken-image")) {
        $(this)
            .prop("src", "img/broken.png")
            .addClass("broken-image");
    }
});
```

即使你没有任何断开的链接, 加入这代码也不会有任何影响。

## hover 切换 class 类

比方说, 当用户将鼠标悬停在你页面上的元素时, 你想改变其视觉效果。当用户鼠标悬停在元素上, 你可以在该元素上添加一个 class 类, 当鼠标停止悬停事件时移除此 class 类。

``` javascript
$(".btn").hover(
    function() {
        $(this).addClass("hover");
    },
    function() {
        $(this).removeClass("hover");
    }
);
```

如果你想要一个更简单的方式使用 toggleClass 方法, 则仅仅需要添加必要的 CSS。

``` javascript
$(".btn").hover(function() {
    $(this).toggleClass("hover");
});
```

备注: CSS 在这种情况下使用是一个快速的解决方案, 但要知道这点知识依旧是值得去了解下的。

## 禁用输入

有时你可能需要用表单的提交按钮或者某个输入框直到用户执行了某个动作（比如: 检查"我已阅读条款"复选框）。在你的输入框上设置 disabled 属性, 然后当你需要的时候启用该属性。

``` javascript
$('input[type="submit"]').prop("disabled", true);
```

你需要做的只是需要在输入框上再次运行 prop 方法, 但设置的被禁用值是 false。

``` javascript
$('input[type="submit"]').prop("disabled", false);
```

## 停止正在加载的链接

有时你不想链接到特定的网页或者重新载入页面；你可能想让他们做一些其他事情, 如触发一些其他的脚本。这是防止违约行动的技巧。

``` javascript
$("a.no-link").click(function(e) {
    e.preventDefault();
});
```

## 限制 Text-Area 域中的字符的个数

``` javascript
jQuery.fn.maxLength = function(max) {
    return this.each(function() {
        var type = this.tagName.toLowerCase(); // 获取当前的标签名
        var inputType = this.type ? this.type.toLowerCase() : null; // 获取当前标签的类型
        if ((type == "input" && inputType == "text") || inputType == "password") {
            this.maxLength = max; // 设置最大值
        } else if (type == "textarea") {
            // 利用键盘事件,阻止继续输入字符
            this.onkeypress = function(e) {
                var ob = e || event;
                var keyCode = ob.keyCode;
                var hasSelection = document.selection ?
                    document.selection.createRange().text.length > 0 :
                    this.selectionStart != this.selectionEnd;
                return !(
                    this.value.length >= max &&
                    (keyCode > 50 || keyCode == 32 || keyCode == 0 || keyCode == 13) &&
                    !ob.ctrlKey &&
                    !ob.altKey &&
                    !hasSelection
                );
            };
            this.onkeyup = function() {
                if (this.value.length > max) {
                    this.value = this.value.substring(0, max);
                }
            };
        }
    });
};
//用法
$("#mytextarea").maxLength(500);
```

## toggle fade/slide

滑动和淡入/淡出 是我们在 jQuery 中经常大量使用的动画。你可能仅仅想在用户做某些点击事件的时候显示一个元素, 这时候需要淡入/淡出或者滑动方法。但是如果你需要那个元素在你第一次点击的时候出现, 在第二次点击的时候消失, 代码如下。

``` javascript
// Fade
$(".btn").click(function() {
    $(".element").fadeToggle("slow");
});

// Toggle
$(".btn").click(function() {
    $(".element").slideToggle("slow");
});
```

## 简单的手风琴

这是个简单快速的方法创建一个手风琴。

``` javascript
// Close all panels
$("#accordion")
    .find(".content")
    .hide();

// Accordion
$("#accordion")
    .find(".accordion-header")
    .click(function() {
        var next = $(this).next();
        next.slideToggle("fast");
        $(".content")
            .not(next)
            .slideUp("fast");
        return false;
    });
```

通过添加这个脚本, 你需要做的则是必要的 HTML 操作在你的页面上。

## 使两个 DIV 同等高度

有时你会想要两个 DIV 有相同的高度, 无论他们都有什么内容.

``` javascript
$(".div").css("min-height", $(".main-div").height());
```

这个例子设置了 DIV 的最小高度, 这意味着它的高度只可以比这个设置的高度大而不能小。然而, 一个更灵活的方法是循环的一组元素, 并设置将最高元素的高度作为高度。

``` javascript
var $columns = $(".column");
var height = 0;
$columns.each(function() {
    if ($(this).height() > height) {
        height = $(this).height();
    }
});
$columns.height(height);
```

如果你想要所有的列有相同的高度。

``` javascript
var $rows = $(".same-height-columns");
$rows.each(function() {
    $(this)
        .find(".column")
        .height($(this).height());
});
```

## 在浏览器标签 / 新窗口打开外部链接

在新的浏览器标签或窗口中打开外部链接, 并确保在同一个标签或窗口中打开的是同一个源的链接。

``` javascript
$('a[href^="http"]').attr("target", "_blank");
$('a[href^="//"]').attr("target", "_blank");
$('a[href^="' + window.location.origin + '"]').attr("target", "_self");
```

备注: window.location.origin 在 IE10 不工作。

## 根据文本获取元素

通过 jQuery 中的 contains()选择器, 你能找到一个元素内的文本内容。如果文本不存在, 则这个元素将被隐藏。通常用于站内搜素。

``` javascript
var search = $("#search").val();
$('div:not(:contains("' + search + '"))').hide();
```

## 可见变化的触发

当用户不再聚焦或者重新聚焦一个标签时触发 javascript 脚本。

``` javascript
$(document).on("visibilitychange", function(e) {
    if (e.target.visibilityState === "visible") {
        console.log("Tab is now in view!");
    } else if (e.target.visibilityState === "hidden") {
        console.log("Tab is now hidden!");
    }
});
```

## Ajax 调用错误处理

当一个 Ajax 调用返回一个 404 或 500 的错误时, 将执行该错误处理。如果该处理未定义, 则其他 jQuery 代码便可能不会执行了。定义一个全局 Ajax 错误处理程序。

``` javascript
$(document).ajaxError(function(e, xhr, settings, error) {
    console.log(error);
});
```

## 解决 jQuery, prototype 共存冲突问题

``` html
<script src="prototype.js"></script>
<script src="jquery.js"></script>
<script type="text/javascript">
    jQuery.noConflict();
</script>
```

::: warning 注意
一定要先引入 prototype.js 再引入 jquery.js, 先后顺序不可错
:::

## 链式操作

jQuery 允许通过链式操作来减轻反复查询 DOM 和创建多个 jQuery 对象的过程。比如下面是你的方法调用。

``` javascript
$("#elem").show();
$("#elem").html("bla");
$("#elem").otherStuff();
```

这代码可以通过链式大大的提高。

``` javascript
$("#elem")
    .show()
    .html("bla")
    .otherStuff();
```

另一个方法是在一个可变的元素缓存（$ 作为前置）。

``` javascript
var $elem = $("#elem");
$elem.hide();
$elem.html("bla");
$elem.otherStuff();
```

链式和 jQuery 缓存方法是最好的做法, 导致更短、更快的代码。
