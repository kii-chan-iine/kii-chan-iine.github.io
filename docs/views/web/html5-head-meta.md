---
author: 星火燎原@vxhly
title: HTML5 移动端头部标签
categories: [web]
tags: [HTML5, Moblie]
date: 2016-08-22 16:21:30
---

<Boxx changeTime="10000"/>

::: tip 前言
记录 HTML5 通用的移动端和 PC 端头部标签, 即 meta 标签。
:::
<!-- more -->

## 定义文档类型和编码习惯

``` html
<!DOCTYPE html>
<!-- 使用 HTML5 doctype,不区分大小写 -->
<html lang="en">
<!-- 更加标准的 lang 属性写法 //zhi.hu/XyIa -->

<head>
    <!-- 声明文档使用的字符编码 -->
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8">
</head>

<body>

    <body>

</html>
```

## 兼容性

``` html
<!-- 优先使用 IE 最新版本和 Chrome 内核的浏览器-->
<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">

<!-- 为移动设备添加 viewport -->
<meta name="viewport" content="width=device-width,initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, user-scalable=no">

<!-- 启用360浏览器的极速模式(webkit) -->
<meta name="renderer" content="webkit">
```

## 搜索引擎优化

``` html
<!-- 页面描述 -->
<meta name="description" content="不超过150个字符">

<!-- 页面关键词 -->
<meta name="keywords" content="">

<!-- 网页作者 -->
<meta name="author" content="name, email@gmail.com">

<!-- 搜索引擎抓取 -->
<!-- index: 搜索引擎索引此网页;follow: 搜索引擎继续通过此网页的链接索引搜索其它的网页。 -->
<!-- 搜索引擎索引方式: 通常有如下几种取值: none,noindex,nofollow,all,index和follow。-->
<meta name="robots" content="index,follow">
<!--
    all: 文件将被检索,且页面上的链接可以被查询；
    none: 文件将不被检索,且页面上的链接不可以被查询；
    index: 文件将被检索；
    follow: 页面上的链接可以被查询；
    noindex: 文件将不被检索；
    nofollow: 页面上的链接不可以被查询。
 -->
```

## 页面缓存设置

``` html
<!-- 清除缓存 -->
<meta http-equiv="pragma" content="no-cache">
<meta http-equiv="cache-control" content="no-cache">
<meta http-equiv="expires" content="0">
```

## 移动端字体设置

``` css
/**
中英字体名对照表
宋体      SimSun
黑体      SimHei
微信雅黑   Microsoft Yahei
微软正黑体 Microsoft JhengHei
新宋体    NSimSun
新细明体  MingLiU
细明体    MingLiU
标楷体    DFKai-SB
仿宋     FangSong
楷体     KaiTi
仿宋_GB2312  FangSong_GB2312
楷体_GB2312  KaiTi_GB2312
说明: 中文字体多数使用宋雅黑,英文用 Helvetica
*/
body {
    font-family: Microsoft Yahei, SimSun, Helvetica;
}
```

## 电话号码和邮箱的识别

``` html
<!-- 一般只启用电话号码的识别 -->
<meta name="format-detection" content="telphone=yes">

<!-- 忽略页面中的数字识别为电话,忽略email识别 -->
<meta name="format-detection" content="telphone=no, email=no">

<!-- 启用页面中的数字识别为电话,启用email识别 -->
<meta name="format-detection" content="telphone=yes, email=yes">
```

## 响应式

``` html
<!-- IOS 中 Safari 允许全屏浏览 -->
<meta name=”apple-mobile-web-app-capable” content=”yes”>

<!-- IOS 隐藏状态栏 -->
<meta name=”apple-mobile-web-app-status-bar-style” content=black”>

<!-- 添加到主屏后的标题（IOS 6 新增） -->
<meta name="apple-mobile-web-app-title" content="标题">

<!-- uc 强制竖屏 -->
<meta name="screen-orientation" content="portrait">

<!-- QQ 强制竖屏 -->
<meta name="x5-orientation" content="portrait">

<!-- UC 强制全屏 -->
<meta name="full-screen" content="yes">

<!-- QQ 强制全屏 -->
<meta name="x5-fullscreen" content="true">

<!-- UC 应用模式 -->
<meta name="browsermode" content="application">

<!-- QQ 应用模式 -->
<meta name="x5-page-mode" content="app">

<!-- 可隐藏地址栏,仅针对 IOS 的 Safari（注: IOS7.0 版本以后,safari上已看不到效果） -->
<meta name="apple-mobile-web-app-capable" content="yes" />

<!-- 仅针对 IOS 的 Safari 顶端状态条的样式（可选 default/black/black-translucent ） -->
<meta name="apple-mobile-web-app-status-bar-style" content="black" />

<!-- IOS 中禁用将数字识别为电话号码/忽略 Android 平台中对邮箱地址的识别 -->
<meta name="format-detection" content="telephone=no, email=no" />
```

## favicon icon 图标

``` html
<!-- 添加 favicon icon -->
<link rel="shortcut icon" type="image/ico" href="/favicon.icon">

<!-- iOS 图标 begin -->
<!-- iPhone 和 iTouch,默认 57x57 像素,必须有 -->
<link rel="apple-touch-icon-precomposed" href="/apple-touch-icon-57x57-precomposed.png">

<!-- Retina iPhone 和 Retina iTouch,114x114 像素,可以没有,但推荐有 -->
<link rel="apple-touch-icon-precomposed" sizes="114x114" href="/apple-touch-icon-114x114-precomposed.png">

<!-- Retina iPad,144x144 像素,可以没有,但推荐有 -->
<link rel="apple-touch-icon-precomposed" sizes="144x144" href="/apple-touch-icon-144x144-precomposed.png">
<!-- iOS 图标 end -->
```

## 参考模板

``` html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name=”apple-mobile-web-app-capable” content=”yes”>
    <meta name=”apple-mobile-web-app-status-bar-style” content=black”>
    <meta name="format-detection" content="telphone=no, email=no">
    <title>标题</title>
    <link rel="stylesheet" herf="">
</head>

<body>
    ...
</body>

</html>
```
