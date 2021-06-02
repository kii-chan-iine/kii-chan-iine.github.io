---
author: 星火燎原@vxhly
title: Console 的一些应用小技巧
categories: [web]
tags: [Debug, JavaScript]
date: 2016-08-10 16:57:53
---

<Boxx changeTime="10000"/>

::: tip 前言
一个简单而实用的控制台命令, 用来调试 JavaScript 代码的不合理性。
:::
<!-- more -->

## console.log(object)

一个取代 alert() 和 document.write() 的最佳途径。是前端开发人员使用频率最高的一条语句, 它向控制台输出一条消息。支持 C 语言 printf 式的格式化输出。当然, 也可以不使用格式化输出来达到同样的目的。

### 简单的输出

``` javascript
var animal = "tiger";
var count = 5;
console.log("The %s jumped over %d tall buildings", animal, count);
console.log("The", animal, "jumped over", count, "tall buildings");
```

![console.log](https://vxhly.github.io/assets/debug-console-1.png)

console 对象可以使用 printf 风格的占位符。支持的占位符, 有字符（%s）、整数（%d 或%i）、浮点数（%f）和对象（%o）四种。

### 有意思的 %c

``` javascript
console.log("%c css88.com", "font-size:20pt");
console.log(
    "%c 前端开发 %c css88.com %c github",
    "color:red",
    "",
    "color:orange;font-weight:bold"
);
console.log("%c 阴影文字", "text-shadow: 3px 1px 1px grey");
console.log(
    "%c 彩色文字",
    "background-image:-webkit-gradient( linear, left top, right top, color-stop(0, #f22), color-stop(0.15, #f2f), color-stop(0.3, #22f), color-stop(0.45, #2ff), color-stop(0.6, #2f2),color-stop(0.75, #2f2), color-stop(0.9, #ff2), color-stop(1, #f22) );color:transparent;-webkit-background-clip: text;font-size:5em;"
);
```

![console.log](https://vxhly.github.io/assets/debug-console-2.png)

以%c 开头, 后面的文字就打印的信息, 后面一个参数就是样式属性；可以尝试多个样式, 每碰到一个%c 开头就会应用对应的样式。

## 不同性质的信息用不同的方法

根据信息的不同性质, console 对象还有 4 种显示信息的方法, 分别是一般信息 console.info()、除错信息 console.debug()、警告提示 console.warn()、错误提示 console.error()。

``` javascript
console.info("this is info");
console.debug("this is debug");
console.warn("this is warn");
console.error("this is error");
```

![console.log](https://vxhly.github.io/assets/debug-console-3.png)

### console.debug(object)

``` javascript
console.debug(object[, object, …])
```

向控制台输出一条信息, 它包括一个指向该行代码位置的超链接。

### console.info(object)

``` javascript
console.info(object[, object, …])
```

向控制台输出一条信息, 该信息包含一个表示"信息"的图标, 和指向该行代码位置的超链接。

### console.warn(object)

``` javascript
console.warn(object[, object, …])
```

同 console.info()。区别是图标与样式不同。

### console.error(object)

``` javascript
console.error(object[, object, …])
```

同 console.info()。区别是图标与样式不同。error 实际上和 throw new Error() 产生的效果相同, 使用该语句时会向浏览器抛出一个 JavaScript 异常。

## console.group(object)

如果信息太多, 可以分组显示, 用到的方法是 console.group() 和 console.groupEnd()。

``` javascript
console.group("第一组信息");
console.log("第一组第一条");
console.log("第一组第二条");
console.groupEnd();

console.group("第二组信息");
console.log("第二组第一条");
console.log("第二组第二条");
console.groupEnd();
```

![分组显示信息](https://vxhly.github.io/assets/debug-console-4.png)

console.groupCollapsed() 跟 console.group() 相同, 区别在于嵌套块默认是收起的。

## console.dir(object)

console.dir() 可以显示一个对象所有的属性和方法（输出结果类似于 DOM 面板中的样式）。

``` javascript
var dog = {};
dog.name = "大毛";
dog.color = "黄色";
dog.bark = function() {
    alert("汪汪汪");
};
console.dir(dog);
```

![console.dir](https://vxhly.github.io/assets/debug-console-5.png)

## console.dirxml(node)

console.dirxml() 用来显示网页的某个节点（node）所包含的 html/xml 代码。

``` javascript
var footer = document.getElementById("footer");
console.dirxml(footer);
```

![console.dirxml](https://vxhly.github.io/assets/debug-console-6.png)

## console.assert(expression)

断言, 测试一条表达式是否为真, 不为真时将抛出异常（断言失败）, 为真则不抛出异常（断言成功）。

``` javascript
var result = 0;
console.assert(result);
var year = 2016;
console.assert(year == 2015);
var i = 5,
    j = 6;
console.assert(i < j);
```

![console.assert](https://vxhly.github.io/assets/debug-console-7.png)

# console.trace()

console.trace() 输出 JavaScript 执行时的堆栈追踪。

比如, 有一个加法器函数。

``` javascript
function add(a, b) {
    return a + b;
}
```

我想知道这个函数是如何被调用的, 在其中加入 `console.trace()` 方法就可以了。

``` javascript
function add(a, b) {
    console.trace();
    return a + b;
}
```

假定这个函数的调用代码如下: 

``` javascript
var x = add3(1, 1);

function add3(a, b) {
    return add2(a, b);
}

function add2(a, b) {
    return add1(a, b);
}

function add1(a, b) {
    return add(a, b);
}
```

运行后, 会显示 `add()` 的调用轨迹, 从上到下依次为 `add()` 、 `add1()` 、 `add2()` 、 `add3()` 。

![console.trace](https://vxhly.github.io/assets/debug-console-8.png)

## console.time(name)

console.time() 和 console.timeEnd(), 用来显示代码的运行时间。当调用 console.timeEnd(name); 并传递相同的 name 为参数时, 计时停止, 并输出执行两条语句之间代码所消耗的时间（毫秒）。

``` javascript
console.time("time");

for (var i = 0; i < 1000; i++) {
    for (var j = 0; j < 1000; j++) {}
}

console.timeEnd("time");
```

![console.time](https://vxhly.github.io/assets/debug-console-9.png)

注:  console.time(slugName) – console.timeEnd(slugName) 记录某一个 javascript 程序的执行时间, 可用于优化循环语句

## console.clear()

清空控制台

## 其他命令

控制台除了可以运行常规的 javascript 代码, 还内置了相当数量的命令行可以辅助我们的调试工作, 

### $(id)

返回一个给定 id 的元素。无需引用 jQuery, 不可与 jQuery 的方法混用。

### $$(selector)

返回给定的 css 选择器匹配到的一组元素。

### $x(xpath)

返回给定的 XPath 表达式匹配到的一组元素。

### $0

返回在 HTML 面板中选中的元素。

### $1

返回上一次在 HTML 面板中选中的元素。

### $n(index)

返回访问最近 5 个被选中过的元素, index 的范围:  0 – 4。

### dir(object)

同 console.dir(object)。

### dirxml(node)

同 console.dirxml(node)。

### clear()

同 console.clear()。

### inspect(object[, tabName])()

在合适的（或一个指定的） tab 中检视一个对象。

### keys(object)

返回一个对象的所有属性的键。

### values(object)

返回一个对象的所有属性的值。

### debug(fn)

在函数第一行添加一个断点, 使用 undebug(fn) 移除断点。

### monitor(fn)

开启一个函数的调用日志, 使用 unmonitor(fn) 关闭该功能。非常有用的一个命令, 但是它似乎并没有很好地工作。

### monitorEvents(object[, types])

开启一个元素的某个事件（或所有事件）被触发时的日志记录。用例如下: 

### monitorEvents($0, ['click'])

上面的命令行被执行后, 将开启当前在 HTML 面板中被选中元素的 click 事件监控, 一旦这个元素的 click 事件被触发, 事件对象将会在控制台输出。如果不指定第二个参数, 将对所有事件进行记录。
