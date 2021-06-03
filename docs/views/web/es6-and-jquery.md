---
author: 星火燎原@vxhly
title: 用 ES6 写 jquery 语法时的 this 绑定问题
categories: [web]
tags: [JavaScript, Node.js, jQuery, ES6]
date: 2018-04-14 12:11:38
---

<Boxx changeTime="10000"/>

::: tip 前言
在 es6 里使用 jquery 时, 习惯用 es5 中直接获取 dom 元素, 如 $('selector'), 绑 定 click 或 hover 等事件; 若你也追求潮流, 在选择器事件中使用箭头函数来定义, 并用到 $(this) 的话, 那么你会发现这完全行不通, 想要知道解决方案的话就继续阅读下去。
:::
<!-- more -->

## 问题重现

``` javascript
$('[data-action="imgQRCodeCtrl"]')
    .hover(
        () => {
            !Base.browser.mobile &&
                $(this)
                .find(".qr-code")
                .css({
                    height: $(this)
                        .find(".img")
                        .height() +
                        2 +
                        "px",
                    opacity: 1,
                    "z-index": 1
                });
        },
        () => {
            $(this)
                .find(".qr-code")
                .css({
                    opacity: 0,
                    "z-index": -1
                });
        }
    )
    .click(() => {
        if (Base.browser.mobile) {
            location.href = $(this).data("href");
        }
    });
```

乍一看, 这段代码没什么问题, 逻辑也很简单, 就是在所有属性有 `data-action="imgQRCodeCtrl"` 的列表元素上绑定 `hover` 及 `click` 事件, 实现 pc 端 hover 时能够使内部的一个块显示或隐藏, 移动端点击时获取当前属性上的 href 并进行跳转；

当然 so easy, 在过去的 es5 上按这种逻辑写完全么有问题；but ES6 里解析后, 会重新定义 this, 

如下是真实浏览器上加载的解析后的代码: 

``` javascript
var $ = __webpack_require__("./node_modules/jquery/dist/jquery.js");
$(function() {
    var _this = this;
    //....
    $('[data-action="imgQRCodeCtrl"]')
        .hover(
            function() {
                !_base2.default.browser.mobile &&
                    $(_this)
                    .find(".qr-code")
                    .css({
                        height: $(_this)
                            .find(".img")
                            .height() +
                            2 +
                            "px",
                        opacity: 1,
                        "z-index": 1
                    });
            },
            function() {
                $(_this)
                    .find(".qr-code")
                    .css({
                        opacity: 0,
                        "z-index": -1
                    });
            }
        )
        .click(function() {
            if (_base2.default.browser.mobile) {
                location.href = $(_this).data("href");
            }
        });
});
```

### 原因

这是 Stack Overflow 上大牛的解读

> This has nothing to do with Traceur and turning something off, this is simply how ES6 works. It's the specific functionality you're asking for by using => instead of function () { }.

> If you want to write ES6, you need to write ES6 all the time, you can't switch in and out of it on certain lines of code, and you definitely cannot suppress or alter the way => works. Even if you could, you would just wind up with some bizarre version of JavaScript that only you understand and which would never work correctly outside of your customized Traceur, which definitely isn't the point of Traceur.

以上大概意思是 "ES6 的工作方式就是这样, 跟编译无关, 与写法有关；若想用 ES6, 最好全用 ES6 的写法, 不能改变它原本的定义..."

原来箭头函数, 不绑定自己的 this, 这一下就说通了；好好看看下面这句吧, 不遇到问题时永远记忆模糊, 哈哈

> 箭头函数表达式的语法比函数表达式更短, 并且不绑定自己的 this, arguments, super 或 new.target。这些函数表达式最适合用于非方法函数, 并且它们不能用作构造函数。

## 问题解决

### 方法一

绑定 `this` 的函数不使用箭头函数不就行了

``` javascript
$('[data-action="imgQRCodeCtrl"]')
    //...
    .click(function() {
        if (Base.browser.mobile) {
            location.href = $(this).data("href");
        }
    });
```

### 方法二

不使用 `this` , 使用 `event.currentTarget`

``` javascript
$('[data-action="imgQRCodeCtrl"]')
    //...
    .click(event => {
        if (Base.browser.mobile) {
            location.href = $(event.currentTarget).data("href");
        }
    });
```

### 方法三

利用 `each` 函数循环绑定到每一个小元素上, 如此就不需要 `this` 了

``` javascript
$('[data-action="imgQRCodeCtrl"]').each((index, element) => {
    let _this = element;
    $(_this).click(() => {
        if (Base.browser.mobile) {
            location.href = $(_this).data("href");
        }
    });
});
```
