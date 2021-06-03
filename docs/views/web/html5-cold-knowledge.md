---
author: 星火燎原@vxhly
title: HTML5 的一些冷知识
categories: [web]
tags: [HTML5, Layout, Code]
date: 2016-08-15 18:12:36
---

<Boxx changeTime="10000"/>

::: tip 前言
现已进入了 HTML5 时代, 这些知识在前端开发人员眼里也就变得更加重要。
:::
<!-- more -->

## HTML5 带来的好处

关于新标记和语义的价值的讨论已经很多, 有些人认为这些语义以及可视化表现方式没有任何意义。因为它们带来的是更复杂的代码, 创造了使用这些新 HTML 代码犯新错误的机会, HTML5 新语义的价值在于提升了网站用户和这些技术之间的交互和协作的水平, 这是使互联网前进的原因。

同样的道理, 我们应该明白, 就像 HTML5 给我们的感觉是那样酷一样, 人们在其上付出了很多的努力, 对未来 5 年互联网的发展寄予厚望。 今天, 这些新标记和语义看起来的确是很棒, 我们都很喜欢, 但别忘了, 任何东西都是在不断变化着来适应人们的需求, 新的语言特征必须依赖于人们的使用, 互联网社区的大量使用, 只有这样才能使单纯的新事物变成有用的新事物。

### 安利一个在线神器

快速生成标准的 HTML5 文档。[initializr](http://www.initializr.com/), 国内访问比较慢, 需要翻墙

## 文档类型和编码

### HTML5 文档类型很简单

``` html
<!doctype html>
```

### HTML5 使用 UTF-8 编码

``` html
<meta charset="UTF-8">
```

## HTML5 新增元素

### 新标签

* **article** => 定义一篇文章
* **aside** => 定义页面内容部分的侧边栏
* **audio** => 定义音频内容
* **canvas** => HTML5 画布
* **command** => 定义一个命令按钮
* **datalist** => 定义一个下拉列表
* **details** => 定义一个元素的详细内容
* **dialog** => 定义一个对话框(会话框)
* **embed** => 定义外部的可交互的内容或插件
* **figure** => 定义一组媒体内容以及它们的标题
* **figcaption** => figure 的标题
* **footer** => 定义一个页面或一个区域的底部
* **header** => 定义一个页面或一个区域的头部
* **hgroup** => 定义文件中一个区块的相关信息
* **keygen** => 定义表单里一个生成的键值
* **mark** => 定义有标记的文本
* **meter** => 定义度量衡, 仅用于已知最大和最小值的度量
* **nav** => 定义导航链接
* **output** => 定义一些输出类型
* **progress** => 定义任务的过程
* **rp** => 在 ruby 注释中使用, 以定义不支持 ruby 元素的浏览器所显示的内容, ruby 注释是中文注音或字符
* **rt** => 定义字符（中文注音或字符）的解释或发音, ruby 注释是中文注音或字符
* **ruby** => 定义 ruby 注释（中文注音或字符）
* **section** => 定义一个区域
* **source** => 定义媒体资源
* **time** => 定义一个日期或者时间
* **video** => 定义一个视频

#### 一个简单的实例

``` html
<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8" />
    <title>Title</title>
</head>

<body>
    <header>
        <h1>Header in h1</h1>
        <h2>Subheader in h2</h2>
    </header>

    <nav>
        <ul>
            <li><a href="#">Menu Option 1</a></li>
            <li><a href="#">Menu Option 2</a></li>
            <li><a href="#">Menu Option 3</a></li>
        </ul>
    </nav>

    <section>
        <article>
            <header>
                <h1>Article #1</h1>
            </header>
            <section>
                This is the first article. This is <mark>highlighted</mark>.
            </section>
        </article>

        <article>
            <header>
                <h1>Article #2</h1>
            </header>
            <section>
                This is the second article. These articles could be blog posts, etc.
            </section>
        </article>
    </section>

    <aside>
        <section>
            <h1>Links</h1>
            <ul>
                <li><a href="#">Link 1</a></li>
                <li><a href="#">Link 2</a></li>
                <li><a href="#">Link 3</a></li>
            </ul>
        </section>

        <figure>
            <img width="150" height="auto" src="https://www.baidu.com/img/bd_logo1.png" alt="baidu logo" />
            <figcaption>baidu logo</figcaption>
        </figure>
    </aside>

    <footer>Footer - Copyright <time>2016</time></footer>
</body>

</html>
```

![H5元素](https://vxhly.github.io/assets/html5-1.png)

### 表单新元素、类型

* **datalist** => 数据列表
* **datetime** => 手工输入完整的时间
* **datetime-local** => 含日期（年月日）和时间
* **output** => 输出一个算法或脚本执行的结果
* **keygen** => 为数据生成一对密钥（公钥和私钥）, 当提交表单时, 私钥存储在本地, 公钥发送到服务器。
* **date** => 只含年月日的日期
* **month** => 只含年月的日期
* **week** => 只含年、周的日期
* **time** => 选择时间
* **color** => 颜色选择器
* **number** => 只能输入数字, 移动端会弹出数字键盘
* **range** => 一个固定范围内的数字
* **email** => 只能输入邮箱地址, 移动端会弹出英文键盘
* **url** => 只能输入 URL 网址, 移动端会弹出英文键盘
* **tel** => 只能输入电话号码, 移动端会弹出数字键盘
* **search** => 搜索内容框, 移动端会弹出英文键盘

#### 一个简单的实例

``` html
<!DOCTYPE html>
<html>

<head>
    <meta charset="utf-8" />
    <title>Title</title>
</head>

<body>
    <form action="" method="post">
        <div>
            <input id="myCar" list="cars" type="text">
            <datalist id="cars">
                <option value="BMW">
                <option value="Ford">
                <option value="Volvo">
            </datalist>
        </div>

        <div><input type="datatime" value="2016-08-15T10:50Z"></div>

        <div>
            <div>Username: <input type="text" name="usr_name"></div>
            <div>Encryption: <keygen name="security"></div>
        </div>

        <div><input type="date"></div>

        <div><input type="month"></div>

        <div><input type="week"></div>

        <div><input type="time"></div>

        <div><input type="datetime-local"></div>

        <div><input type="search" value="百度一下,你就知道"></div>

        <div><input type="color"></div>

        <div><input type="email" value="666666@qq.com"></div>

        <div><input type="url" value="https://vxhly.github.io"></div>

        <div><input type="tel" value="800820820"></div>

        <div><button type="submit">submit</button></div>

        <div><button type="reset">reset</button></div>
    </form>

    <form oninput="x.value=parseInt(a.value)+parseInt(b.value)">
        0 <input type="range" id="a" value="50"> 100 +
        <input type="number" id="b" value="50"> =
        <output name="x" for="a b"></output>
    </form>
</body>

</html>
```

![H5表单](https://vxhly.github.io/assets/html5-2.png)

## HTML5 废弃的内容

HTML5 废弃了一些过时的, 不合理的 HTML 标签

* **acronym**
* **center**
* **dir**
* **font**
* **s**
* **strike**
* **tt**
* **u**
* **frame**
* **frameset**
* **noframe**
* **applet**
* **big**
* **center**
* **basefront**

## 打电话发短信写邮件

### 打电话

``` html
<a href="tel:010-88888">打电话给:010-88888</a>
```

### 发短信

``` html
<a href="sms:88888">发短信给: 88888</a>
```

### 写邮件

``` html
<!-- 注: 在添加这些功能时,第一个功能以"?"开头,后面的以"&"开头 -->
<!-- 1.普通邮件 -->
<a href="mailto:haha@wtf.com">快来点我啊,给你发种子。</a>

<!-- 2.收件地址后添加?cc=开头,可添加抄送地址（Android存在兼容问题） -->
<a href="mailto:haha@wtf.com?cc=666@wtf.com">快来点我啊,给你发种子。</a>

<!-- 3.跟着抄送地址后,写上&bcc=,可添加密件抄送地址（Android存在兼容问题） -->
<a href="mailto:haha@wtf.com?cc=666@wtf.com&bcc=384900096@wtf.com">快来点我啊,给你发种子。</a>

<!-- 4.包含多个收件人、抄送、密件抄送人,用分号(;)隔开多个邮件人的地址 -->
<a href="mailto:haha@wtf.com;384900096@wtf.com">快来点我啊,给你发种子。</a>

<!-- 5.包含主题,用?subject= -->
<a href="mailto:haha@wtf.com?subject=邮件主题">快来点我啊,给你发种子。</a>

<!-- 6.包含内容,用?body=;如内容包含文本,使用%0A给文本换行 -->
<a href="mailto:haha@wtf.com?body=邮件主题内容%0A我是第二行内容%0A你没有猜错,这是我是第三行。">快来点我啊,给你发种子。</a>

<!-- 7.内容包含链接,含http(s)://等的文本自动转化为链接 -->
<a href="mailto:haha@wtf.com?body=//www.wtf.com">快来点我啊,给你发种子。</a>

<!-- 8.内容包含图片（PC不支持） -->
<a href="mailto:haha@wtf.com?body=<img src='images/torrent.jpg' />">快来点我啊,给你发种子。</a>
```

## 取消英文首字母的默认大写

``` html
<input autocapitalize="off" autocorrect="off" />
```

## 屏幕旋转的事件和样式

``` javascript
function orientInit() {
    var orientChk =
        document.documentElement.clientWidth > document.documentElement.clientHeight ?
        "landscape" :
        "portrait";
    if (orientChk == "lapdscape") {
        //横屏下需要执行代码
    } else {
        //竖屏下需要执行代码
    }
}

orientInit();
window.addEventListener(
    "onorientationchange" in window ? "orientationchange" : "resize",
    function() {
        setTimeout(orientInit, 100);
    },
    false
);
```

``` css
/*竖屏时样式*/
@media all and (orientation: portrait) {}

/*横屏时样式*/
@media all and (orientation: landscape) {}
```

## HTML5 多媒体标签

### 嵌入音频

HTML5 支持 MP3、Wav 和 Ogg 格式的音频, 下面是在网页中嵌入音频的简单示例。

``` html
<audio controls>
    <source src="jamshed.mp3" type="audio/mpeg">
    Your browser does’nt support audio embedding feature.
</audio>
```

### 嵌入视频

HTML5 支持 MP4、WebM 和 Ogg 格式的视频, 下面是在网页中嵌入视频的简单示例。

``` html
<video width="450" height="340" controls>
    <source src="jamshed.mp4" type="video/mp4">
    Your browser does’nt support video embedding feature.
</video>
```

### 其他标签

HTML5 对于多媒体提供了强有力的支持, 除了 audio 和 video 标签外, 还支持以下标签: 

#### embed 标签

定义嵌入的内容, 比如插件。

``` html
<embed type="video/quicktime" src="Fishing.mov">
```

#### source 标签

对于定义多个数据源很有用, 解决音频/视频在不同浏览器的解析问题。

``` html
<video width="450" height="340" controls>
    <source src="jamshed.mp4" type="video/mp4">
    <source src="jamshed.ogg" type="video/ogg">
</video>
```

### track 标签

为诸如 video 元素之类的媒介规定外部文本轨道。 用于规定字幕文件或其他包含文本的文件, 当媒介播放时, 这些文件是可见的。

``` html
<video width="450" height="340" controls>
    <source src="jamshed.mp4" type="video/mp4">
    <source src="jamshed.ogg" type="video/ogg">
    <track kind="subtitles" label="English" src="jamshed_en.vtt" srclang="en" default>
    </track>
    <track kind="subtitles" label="Arabic" src="jamshed_ar.vtt" srclang="ar">
    </track>
</video>
```

### 播放视频不全屏

``` html
<!--
1.ios7+支持自动播放
2.支持Airplay的设备（如: 音箱、Apple TV)播放
x-webkit-airplay="true"
3.播放视频不全屏
webkit-playsinline="true"
-->
<video x-webkit-airplay="true" webkit-playsinline="true" preload="auto" autoplay src="http://[IP]"></video>
```

## 利用 canvas 绘制图形

Canvas 元素用于在网页上绘制图形, 该元素标签强大之处在于可以直接在 HTML 上进行图形操作。 利用 Canvas 元素绘制图形, 需要 JavaScript 脚本的支持, 所以 JavaScript 编程能力不是很好的童鞋不要轻易尝试编写, 还是去网上复制粘帖吧！

``` html
<canvas id="canvas1" width="300" height="100">
</canvas>
```

## HTML5 存储类型

HTML5 能够本地存储数据, 在之前都是使用 cookies 使用的。 该存储方案需要编写 JavaScript 脚本支持, 所以 JavaScript 编程能力不是很好的童鞋不要轻易尝试编写, 还是去网上复制粘帖吧！ HTML5 提供了下面两种本地存储方案: 

* **localStorage** 用于持久化的本地存储, 数据永远不会过期, 关闭浏览器也不会丢失。
* **sessionStorage** 同一个会话中的页面才能访问并且当会话结束后数据也随之销毁。因此 `sessionStorage` 不是一种持久化的本地存储, 仅仅是会话级别的存储

## HTML5 提供了新的 API

有助于快速开发应用程序。 HTML5 提供的应用程序 API 主要有: 

* **Media API**
* **Text Track API**
* **Application Cache API**
* **User Interaction**
* **Data Transfer API**
* **Command API**
* **Constraint Validation API**
* **History API**

## HTML5 应用程序缓存

应用程序缓存是 HTML5 的重要特性之一, 它提供了离线使用的功能, 让应用程序可以获取本地的网站内容, 例如 HTML、CSS、图片以及 JavaScript。这个特性可以提高网站性能, 它的实现借助于 manifest 文件, 如下: 

``` html
<!doctype html>
<html manifest="example.appcache">
...

</html>
```

与传统浏览器缓存相比, 它不强制用户访问的网站内容被缓存。

## HTML5 标签完美兼容 IE

为了方便兼容 IE8 中使用 HTML5, 可以使用 JS 的方法来使低于 IE9 版本的 IE 浏览器兼容。

``` html
<!--[if lt IE 9]>
  <script>
  (function(){
    var tags =
      ['header','footer','figure','figcaption','details','summary','hgroup','nav','aside','article','section','mark','abbr','meter','output','progress','time','video','audio','canvas','dialog'];
      for(var i=tags.length - 1;i>-1;i--){
        document.createElement(tags[i]);
      }
  })();
  </script>
<![endif]-->
```

使用 Google 的 html5shiv.js（Google 在中国打不开, 谨慎使用）<br>
**html5shiv.js**: 解决 ie9 以下浏览器对 html5 新增标签的不识别, 并导致 CSS 不起作用的问题。

``` html
<!--[if lt IE 9]>
  <script src="//html5shiv.googlecode.com/svn/trunk/html5.js"></script>
<![endif]-->
```

Google 的 html5shiv 包不适用, 这里提供 Baidu 的 html5shiv.js

``` html
<!--[if lt IE 9]>
  <script src="//libs.baidu.com/html5shiv/3.7/html5shiv.min.js"></script>
<![endif]-->
```

这里还提供 bootstarp CDN 的 htmlshiv.js 和 respond.min.js<br>
**respond.min.js**: 让不支持 css3 Media Query 的浏览器包括 IE6-IE8 等其他浏览器支持查询。

``` html
<!--[if lt IE 9]>
　　<script src="//cdn.bootcss.com/respond.js/1.4.2/respond.js"></script>
　　<script src="//cdn.bootcss.com/html5shiv/3.7.2/html5shiv.min.js"></script>
<![endif]-->
```

添加以上任意脚本中的一个, 之后再在你的 css 文件中编写如下内容。

``` css
article,
aside,
dialog,
footer,
header,
section,
footer,
nav,
figure,
menu {
    display: block;
}
```
