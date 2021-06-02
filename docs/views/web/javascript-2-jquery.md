---
author: 星火燎原@vxhly
title: JavaScript 原生实现 jQuery 的方法
categories: [web]
tags: [JavaScript, jQuery, Vue.js]
date: 2019-01-06 12:11:18
---

<Boxx changeTime="10000"/>

::: tip 前言
提供 jQuery 中的 $()、siblings()、addClass()、hasClass()、removeClass()、toggleClass() 等方法的原生 JavaScript 实现方法, 以下方法适用于 Vue.js 项目。
:::
<!-- more -->

## Vue.js 不等于 jQuery

个人强烈推荐, 既然用了 Vue.js 来做项目的话, 不要去引用 jQuery, 因为 jQuery 能实现的功能, 在 Vue.js 的项目一定能找到对应的解决方案。但是有人就说了, 我就是想用 jQuery 中的某些方法, 然鹅 Vue.js 却找不到对应的 API 或者根本不知道它在哪里, 所以下面收集了几个原生 JavaScript 实现方法仅供参考。

## $()

``` html
<script>
    export default {
        methods: {
            $(obj) {
                return document.querySelectorAll(obj)
            }
        }
    }
</script>
```

## siblings()

``` html
<script>
    export default {
        methods: {
            siblings(elem) {
                let a = [];
                let b = elem.parentNode.children;
                for (let i = 0; i < b.length; i++) {
                    if (b[i] !== elem) a.push(b[i]);
                }
                return a;
            }
        }
    }
```

## addClass()

``` html
<script>
    export default {
        methods: {

            addClass(ele, cls) {
                ele.className = `${ele.className} ${cls}` ;
            }

        }
    }
```

## hasClass()

``` html
<script>
    export default {
        methods: {
            hasClass(ele, cls) {
                return ele.className.match(new RegExp("(\\s|^)" + cls + "(\\s|$)"));
            }
        }
    }
</script>
```

## removeClass()

``` html
<script>
    export default {
        methods: {
            removeClass(ele, cls) {
                if (this.hasClass(ele, cls)) {
                    let reg = new RegExp("(\\s|^)" + cls + "(\\s|$)");
                    ele.className = ele.className.replace(reg, " ");
                }
            }
        }
    }
```

## toggleClass()

``` html
<script>
    export default {
        methods: {
            toggleClass(ele, cls) {
                if (this.hasClass(ele, cls)) {
                    this.removeClass(ele, cls);
                } else {
                    this.addClass(ele, cls);
                }
            }
        }
    }
```
