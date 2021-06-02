---
author: 星火燎原@vxhly
title: JavaScript 获取 URL 请求参数
categories: [design]
tags: [JavaScript, Code]
date: 2016-08-07 19:21:58
---

<Boxx changeTime="10000"/>

::: tip 前言
通过 JavaScript 原生方法, 获取 URL 所带的请求参数备用。
:::
<!-- more -->

## 方法一(使用正则表达式匹配)

### 方法源码

``` javascript
function GetValue(name) {
    var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)");
    var r = window.location.search.substr(1).match(reg);
    if (r != null) return unescape(r[2]);
    return null;
}
```

### 如何使用？

若地址栏 URL 为: `abc.html?id=1234&&url=https://www.baidu.com` 则

``` javascript
console.log(GetValue("id"));
```

输出 => `123`

``` javascript
console.log(GetValue("url"));
```

输出 => <https://www.baidu.com`>

## 方法二（普通字符串截取）

### 方法源码

``` javascript
function GetRequest() {
    var url = location.search;
    var theRequest = new Object();
    if (url.indexOf("?") != -1) {
        var str = url.substr(1);
        strs = str.split("&");
        for (var i = 0; i < strs.length; i++) {
            theRequest[strs[i].split("=")[0]] = strs[i].split("=")[1];
        }
    }
    return theRequest;
}
```

### 如何使用？

``` html
<script type="text/javascript">
    var Request = new Object();
    Request = GetRequest();
</script>
```

若地址栏 URL 为: `abc.html?id=1234&&url=https://www.baidu.com` 则

``` javascript
console.log(Request["id"]);
```

输出=> `123`

``` javascript
console.log(Request["url"]);
```

输出 => <https://www.baidu.com`>
