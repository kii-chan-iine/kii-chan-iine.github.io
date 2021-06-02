---
author: 星火燎原@vxhly
title: 浏览器对象模型
categories: [web]
tags: [JavaScript, Pentest]
date: 2016-08-15 21:58:30
---

<Boxx changeTime="10000"/>

::: tip 前言
BOM (Browser Object Model) 是指浏览器对象模型, 是用于描述这种对象与对象之间层次关系的模型。
:::
<!-- more -->

## BOM 的定义

* `BOM` 是 `browser object model` 的缩写, 简称浏览器对象模型
* `BOM` 提供了独立于内容而与浏览器窗口进行交互的对象
* 由于 `BOM` 主要用于管理窗口与窗口之间的通讯, 因此其核心对象是 `window`
* `BOM` 由一系列相关的对象构成, 并且每个对象都提供了很多方法与属性
* `BOM` 缺乏标准, `JavaScript` 语法的标准化组织是 `ECMA` , `DOM` 的标准化组织是 `W3C`
* `BOM` 最初是 `Netscape` 浏览器标准的一部分

### BOM 的结构图

在线查看 [browser-object-model.jpg](https://vxhly.github.io/assets/browser-object-model.jpg?attname=)<br>
请下载查看 [browser-object-model.jpg](https://vxhly.github.io/assets/browser-object-model.jpg?attname=)

## BOM 和 DOM 的不同之处

#

* `BOM` 即浏览器对象模型, 主要用了访问一些和网页无关的浏览器功能。 如: `window` 、 `location` 、 `navigator` 、 `screen` 、 `history` 等对象。
* `DOM` 即文档对象模型, 针对 `HTML` （或 `XML` ）文档的 `API` （应用程序编程接口）。 描绘的一个层次化的节点树, 开发人员可以添加、 修改和删除页面的某一部分。

## 细说 BOM 对象

### window 对象

window 对象表示浏览器的一个实例, 同时也是 ECMAScript 规定的 Global 对象。 （Global:  所有在全局作用域中定义的属性和函数, 都是 Global 对象的属性。 在浏览器中 Global 对象实际上就是 window 对象实现的 ） 全局作用域( 所有全局对象都可以通过 window 来访问 )

 `For Example:`

``` javascript
var str = "张三";

function fun() {
    alert(str);
}
// 全局对象都可以通过 window 的属性方式来访问
alert(window.str);
window.fun();
```

#### 弹出窗口

 `For Example:`

``` javascript
window.open("https://www.baidu.com", "NewWin", "height=400, width=400");
```

第三个参数还可能有的值如: <br>

![window 对象](https://vxhly.github.io/assets/window-object.png)

#### 间歇调用（setTimeout）

 `For Example:`

``` javascript
setTimeout(function() {
    alert("Hello world!");
}, 1000); // 会在 1000 毫秒后执行
```

如果在间歇时间之前想要取消执行, 我们可以 `clearTimeout()` , `For Example:`

``` javascript
// 设置超时调用
var timeoutId = setTimeout(function() {
    alert("Hello world!");
}, 1000);
//注意:  把它取消
clearTimeout(timeoutId);
```

同样我们也可以通过间歇调用模拟实现超时调用, `For Example:`

``` javascript
function fun() {
    // 这里可以实现某些逻辑
    setTimeout(fun(), 100);
}
fun();
```

#### 超时调用（setInterval）

 `For Example:`

``` javascript
setInterval(function() {
    alert("Hello world!");
}, 1000); // 每隔 1 秒 执行一遍
```

如果想要取消执行, 我们可以 `clearTimeout()` , `For Example:`

``` javascript
var i = 1;
var timeId = setInterval(function() {
    i++;
    // 这里可以执行某些逻辑
    if (1 >= 100) {
        clearInterval(timeId); // 注意和取消间歇调用不一样（clearTimeout）
    }
}, 1000); // 每隔1秒 执行一遍
```

### location 对象

location 对象说来也是奇怪, 它既是 window 的属性也是 document 属性, 且同时指向了同一个对象。 window 和 document 的关系:  浏览器中的 html 成为了 document 对象, 使我们可以通过 javascript 来访问、 操作 html 的元素。 且 documnet 对象是 window 对象的一部分, 可以通过 window.document 属性来访问。

#### location 的属性

| 属性名   | 例子                      | 说明                                                                             |
|----------|---------------------------|----------------------------------------------------------------------------------|
| hash     | "#contens"                | 返回 URL 中 hash（#号后跟零或多个字符）, 如果 URL 中不包含散列, 则返回空字符串   |
| host     | "www.baidu.com:80"        | 返回服务器名和端口号                                                             |
| hostname | "www.baidu.com"           | 返回不带端口号的服务器名称                                                       |
| href     | "<http://www.baidu.com>"  | 返回当前加载页面的完整 URL。 location 对象的 toString() 方法也返回该值           |
| pathname | "/admin/"                 | 返回 URL 中的目录或文件名                                                        |
| port     | "8080"                    | 返回 URL 中指定的端口号, 如果 URL 中不包含端口号, 则返回空字符串                 |
| protocol | "http\:"                  | 返回页面使用的协议。 通常是 http\: 或者是 https\:                                |
| search   | "?q=javascript&&tags=web" | 返回 URL 中查询的字符串。 查询以 ？ 与前面的字符串分割, 用 && 连接多个类型的查询 |

有了上面这些属性, 我们可以非常方便的修改 `url` , `For Example:`

``` javascript
// 假设初始 URL 为 http://www.wrox.com/WileyCDA/
// 将 URL 修改为 "http://www.wrox.com/WileyCDA/#section1"
location.hash = "#section1";
// 将 URL 修改为 "http://www.wrox.com/WileyCDA/?q=javascript"
location.search = "?q=javascript";
// 将 URL 修改为 "https://www.yahoo.com/WileyCDA/"
location.hostname = "www.yahoo.com";
// 将 URL 修改为 "https://www.yahoo.com/mydir/"
location.pathname = "mydir";
// 将 URL 修改为 "https://www.yahoo.com:8080/WileyCDA/"
location.port = 8080;
```

每次修改 `location` 的属性（ `hash` 除外）, 页面都会以新 URL 重新加载。 也就是说, 浏览器会生成一天后退的历史记录。 如果我们不想能够回退可以使用 `replace（）` 方法, `For Example:`

``` javascript
location.replace("https://www.baidu.com"); // 如此就不会有回退记录了。
```

除了 `replace` 方法外还有个比较重要的方法。 `reload` 方法用来刷新。

``` javascript
location.reload(); // 重新加载（有可能从缓存中加载）
location.reload(true); // 重新加载（从服务器重新加载）也就是强制刷新
```

### navigator 对象、 screen 对象

navigator 对象主要用来识别客户端浏览器, 但是由于各类浏览器对 navigator 对象的实现各有不同, 这里就不细分析了。 screen 对象基本上只用来表明客户端的能力, 其中包括浏览器窗口外部的显示器的信息, 如像素宽度和高度等。 每种浏览器对其的支持程度也是不同的, 这里也不分析了。

### history 对象

history 对象主要保存当前也网页的历史记录。 但出于安全考虑, 程序员不能够知道详细的 url。

 `For Example:`

``` javascript
history.length; // 历史记录的数量
history.back(); // 后退一页
history.forward(); // 前进一页
history.go(-1); // 后退一页
history.go(1); // 前进一页
history.go(2); // 前进两页
history.go("wrox.com"); // 跳转到最近的 wrox.com 页面
```
