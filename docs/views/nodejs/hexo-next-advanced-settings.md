---
author: 星火燎原@vxhly
title: Hexo+NexT 进阶设置
categories: [nodejs]
tags: [Liunx, Node.js, Hexo]
date: 2017-10-28 20:24:59
---

<Boxx changeTime="10000"/>

::: tip 前言
HEXO 博客最近大更新, 页面与之前大不相同, 这里跟大家分享一下相关的设置方法。
:::
<!-- more -->

## highlight.js

[highlightjs 官方网站](https://highlightjs.org/)

Next 自带的代码高亮并不符合我的口味, 所以就采用了第三方的 `highlight.js` 

## Use

### 编辑配置文件

先禁用默认的高亮方式

`./_config.yml` 

``` yaml
highlight:
    enable: false
line_number: true
autoi_detect: true
tab_replace:
```

### 引入第三方 highlight

#### 编辑主题源文件

`./themes/layout/_layout.swing` 

注意这边需要引入样式, 相关的样式以及支持的语言请查看 [176 languages and 79 styles](https://highlightjs.org/static/demo/)

因为个人比较喜欢 `Atom` 这款编辑器, 改编辑器用默认的样式为 `atom-one-dark` , 所以这里采用该样式

``` html
<link rel="stylesheet" href="//cdn.bootcss.com/highlight.js/9.12.0/styles/atom-one-dark.min.css">
```

除了引入样式, 还需初始化 highlightjs, 这里提供的代码是可以显示行号的

``` html
<script src="//cdn.bootcss.com/highlight.js/9.12.0/highlight.min.js"></script>
<script type="text/javascript">
    $(function() {
        hljs.initHighlighting();
        $('pre code').each(function() {
            var lines = $(this).text().split('\n').length - 1;
            var $numbering = $('<ul/>').addClass('pre-numbering');
            $(this)
                .addClass('has-numbering')
                .parent()
                .append($numbering);
            for (i = 1; i <= lines; i++) {
                $numbering.append($('<li/>'));
            }
        });
    });
</script>
```

可以看到以上的代码是有添加 class 样式的, 所以还需补上相关的样式表, 以下的样式表在小分辨率的情况下行号是不显示出来的, 优化效果请参照本博客代码高亮区块

``` html
<style>
    pre {
        position: relative;
        padding: 0;
    }

    code.has-numbering {
        margin-left: 2.5rem;
    }

    code.has-numbering::-webkit-scrollbar {
        display: none;
    }

    .pre-numbering {
        margin: 0;
        position: absolute;
        top: 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
    }

    .pre-numbering li {
        list-style-type: decimal-leading-zero !important;
    }

    @media (max-width: 760px) {
        .post-body pre {
            padding: 0px;
        }

        .post-body .pre-numbering {
            display: none;
        }

        code.has-numbering {
            margin: 0;
        }
    }
</style>
```

## Copy

``` html
<script src="https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.1/clipboard.min.js"></script>
```

::: tip 提示
这边不就贴关键代码了, 请移步 [clipboard.js](https://github.com/zenorocha/clipboard.js) [_layout.swig](https://github.com/vxhly/vxhly.github.io/blob/blog-source/themes/next/layout/_layout.swig)
:::

## 优化 blockquote

默认的引用 blockquote 样式比较丑, 写了一些样式优化它, 优化效果请参照本博客的引用区块

### 编辑主题源文件

`./themes/layout/_layout.swing` 

``` html
<style>
    blockquote {
        color: #888;
        background-color: #eae6f3;
        border-left: 1rem solid #5236a0;
        padding: 0 1.5rem;
        position: relative;
        font-family: 'Roboto', sans-serif;
    }

    blockquote:before,
    blockquote:after {
        color: #392570;
        font-size: 2.5rem;
        position: absolute;
        line-height: 2.5rem;
    }

    blockquote:before {
        content: "\201C";
        left: -1rem;
        top: 0;
    }

    blockquote:after {
        content: "\201D";
        right: 0;
        bottom: -1rem;
    }
</style>
```

## 网站已运行时间

### 编辑主题源文件

`./themes/layout/_parrials/footer.swing` 

找到以下代码区块

``` html
{% if theme.footer.theme.enable %}
/* ... */
{% endif %}
```

在 `</div>` 之前添加

``` html
<span> | </span>
<span id="showDays"></span>
```

在该文件末尾处添加

``` html
<script>
    var seconds = 1000;
    var minutes = seconds * 60;
    var hours = minutes * 60;
    var days = hours * 24;
    var years = days * 365;
    var birthDay = Date.UTC(2016, 07, 30, 00, 00, 00); // 这里设置建站时间
    setInterval(function() {
        var today = new Date();
        var todayYear = today.getFullYear();
        var todayMonth = today.getMonth() + 1;
        var todayDate = today.getDate();
        var todayHour = today.getHours();
        var todayMinute = today.getMinutes();
        var todaySecond = today.getSeconds();
        var now = Date.UTC(todayYear, todayMonth, todayDate, todayHour, todayMinute, todaySecond);
        var diff = now - birthDay;
        var diffYears = Math.floor(diff / years);
        var diffDays = Math.floor((diff / days) - diffYears * 365);
        var diffHours = Math.floor((diff - (diffYears * 365 + diffDays) * days) / hours);
        var diffMinutes = Math.floor((diff - (diffYears * 365 + diffDays) * days - diffHours * hours) / minutes);
        var diffSeconds = Math.floor((diff - (diffYears * 365 + diffDays) * days - diffHours * hours - diffMinutes * minutes) / seconds);
        document.getElementById('showDays').innerHTML = "本站已运行 " + diffYears + " 年 " + diffDays + " 天 " + diffHours + " 小时 " + diffMinutes + " 分钟 " + diffSeconds + " 秒";
    }, 1000);
</script>
```

## 开启页面动画效果

### 编辑主题配置文件

`./themes/_config.yml` 

``` yaml
#
Canvas - nest
canvas_nest: true

vendors: #Internal version: 1.0 .0# https: //github.com/hustcc/canvas-nest.js
    canvas_nest: //cdn.bootcss.com/canvas-nest.js/1.0.1/canvas-nest.min.js
```

## 新的评论系统

::: tip 提示
自从多说和网易云跟帖都挂了之后, Next 官方引入了 Valine 这一款评论系统
:::

### 关于 Valine 的介绍

::: tip 提示
请移步 [Valine -- 一款极简的评论系统](https://ioliu.cn/2017/add-valine-comments-to-your-blog/), 配置中出现 `leancloud application appid` 和 `leancloud application appkey` 获取方法也请看此链接
:::

相关的配置指南 <https://valine.js.org/>

### 编辑主题配置文件

`./themes/_config.yml` 

``` yaml
#
Valine.#You can get your appid and appkey from https: //leancloud.cn
    #more info please open https: //github.com/xCss/Valine
    valine:
    enable: true
appid: #your leancloud application appid
appkey: #your leancloud application appkey
notify: false# mail notifier, https: //github.com/xCss/Valine/wiki
    verify: false# Verification code
placeholder: ヾﾉ≧∀≦) o 亲爱的小主, 有何指教
```

### 使用最新的 Valine

#### 修改主题源码文件

`./themes/layout/_third-party/comments/valine.swig` 

``` html
<script src="//unpkg.com/valine/dist/Valine.min.js"></script>
<script type="text/javascript">
    new Valine({
        av: AV,
        el: '#comments',
        app_id: '{{ theme.valine.appid }}',
        app_key: '{{ theme.valine.appkey }}',
        placeholder: '{{ theme.valine.placeholder }}', // 默认 null
        notify: {
            {
                theme.valine.notify
            }
        }, // 默认 false
        verify: {
            {
                theme.valine.verify
            }
        }, // 默认 false
        path: {
            {
                theme.valine.path
            }
        }, // 默认 window.location.pathname
        avatar: '{{ theme.valine.avatar }}' // 默认 ''
    });
</script>
```

