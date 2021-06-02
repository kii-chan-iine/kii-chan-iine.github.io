---
author: 星火燎原@vxhly
title: WEB 技术基础知识点
categories: [web]
tags: [WEB, HTML5, JavaScript, CSS, jQuery]
date: 2018-01-07 14:36:27
---

<Boxx changeTime="10000"/>

::: tip 前言
WEB 技术基础知识点（包含: HTML、CSS、JavaScript 三大块知点点集合）。
:::
<!-- more -->

## HTML

常见单标记和双标记的基础用法

### 单标记

#### img

``` html
<img src="" alt="">
```

* `src` => 资源文件地址
* `alt` => 资源文件未找到时, 显示的文本信息

::: tip 提示
常见图片文件格式: GIF、JPEG、PNG、JPG、BMP<br>
大小比较: 通常地, PNG ≈ JPG > GIF；透明性: PNG > GIF > JPG；色彩丰富程度: JPG > PNG >GIF；兼容程度: GIF ≈ JPG > PNG；(JPEG、BMP 使用的比肩少, 这里不做比较
:::

#### hr

``` html
<hr align="" size="" width="">
```

* `align` => center/left/right(对齐方式)
* `size` => 高度, 即粗细
* `width` => 宽度

#### input

``` html
<input name="" value="" type="" placeholder="" required>
```

* `name` => 传与后端指定的参数名
* `value` => 输入框的值
* `type` => button/checkbox/file/hidden/image/password/radio/reset/submit/text(类型)
* `placeholder` => 默认文本提示信息
* `required` => 指定必填项

这里只摘取重点属性, 其他属性用法参照 [HTML input 标签](http://www.w3school.com.cn/tags/tag_input.asp)

#### link

``` html
<link rel="stylesheet" src="">
```

* `src` => 资源文件地址

#### meta

``` html
<meta http-equiv="Content-Type" content="text/html;charset=utf-8">
```

更多详细针对 [HTML5 移动端头部标签](https://vxhly.github.io/2016/08/html5-head-meta/) 的介绍

#### embed

``` html
<embed src="" width="" height="">
```

* `src` => 资源文件地址
* `width` => 嵌入内容的宽度
* `height` => 嵌入内容的高度

#### 其他

* `<br/>`

### 双标记

#### a

``` html
<a href="" title="" target=""></a>
```

* `href` => URL, `http://[hostName]/[fileName]?[param]=[value]` 或者 `https://[hostName]/[fileName]?[param]=[value]` 或者 `//[hostName]/[fileName]?[param]=[value]` (SEO 优化中推荐使用)
* `title` => 提示文本
* `target`
-- `_blank` => 浏览器总在一个新打开、未命名的窗口中载入目标文档。
-- `_self` => 这个目标的值对所有没有指定目标的 `<a>` 标签是默认目标, 它使得目标文档载入并显示在相同的框架或者窗口中作为源文档。这个目标是多余且不必要的, 除非和文档标题 `<base>` 标签中的 target 属性一起使用。
-- `_parent` => 这个目标使得文档载入父窗口或者包含来超链接引用的框架的框架集。如果这个引用是在窗口或者在顶级框架中, 那么它与目标 _self 等效。
-- `_top` => 这个目标使得文档载入包含这个超链接的窗口, 用 _top 目标将会清除所有被包含的框架并将文档载入整个浏览器窗口。

#### ul, ol, dl

`ul` => 无序列表

``` html
<ul type="" start="">
    <li></li>
</ul>
```

`ol` => 无序列表

``` html
<ol type="" start="">
    <li></li>
</ol>
```

`dl` => 定义列表

``` html
<dl type="" start="">
    <dt></dt>
</dl>
```

* `type`
-- `disc` => 默认值。实心圆。
-- `circle` => 空心圆。
-- `square` => 实心方块
-- `1/a/A/i/I` => 自定义, 只针对有序列表

* `start` => 起始

#### table

``` html
<table cellspacing="" cellpadding="" border="" width="" align="">
    <caption></caption>
    <tr rowspan="" colspan="">
        <th></th>
    </tr>
    <tr align="">
        <td rowspan="" colspan=""></td>
    </tr>
</table>
```

* `cellspacing` => 单元格与单元格之间的距离
* `cellpadding` => 单元格与内容之间的距离
* `border` => 表格边框线的粗细
* `width` => 表格宽度
* `aligin` => center/left/right/justify(对齐方式)

 `来个例子`
[课程表](https://codepen.io/vxhly/pen/XVzxyO)

#### form

``` html
<form action="" method="" enctype="multipart/form-data">
    <fieldset>
        <legend>个人信息 </legend>
        <div>
            <label for=""></label>
            <input name="" value="" placeholder="" required id="" type="">
        </div>
        <div>
            <select name="">
                <option value=""></option>
            </select>
        </div>
        <div>
            <textarea name="" rows="" cols></textarea>
        </div>
        <div>
            <button type="submit">提交</button>
            <button type="reset">重置</button>
        </div>
    </fieldset>
</form>
```

* `action` => 提交至后端地址
* `method` => get/post(指定发包方式)

 `来个例子`
[表单](https://codepen.io/vxhly/pen/Rxjqap)

#### 其他

* `<h1></h1>` ~ `<h6></h6>`
* `<span><span>`
* `<div></div>`

### 常见行内标记、块级标记

#### 行内标记

* `span`
* `input`
* `img`
* `a`
* `b`
* `font`
* `label`
* `select`

#### 块级标记

* `div`
* `p`
* `h1` ~ `h6`
* `ul/ol/dt`
* `hr`
* `form`

### 标准 HTML 模板

``` html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>

    <link rel="stylesheet" href="style.css">
</head>

<body>
    <!-- 这里是注释信息 -->
    <script src=""></script>
</body>

</html>
```

## CSS

这里只列出可以合并写法的一些 CSS 属性

### 盒子模型

所有 HTML 元素可以看作盒子, 在 CSS 中, "box model"这一术语是用来设计和布局时使用。CSS 盒模型本质上是一个盒子, 封装周围的 HTML 元素, 它包括: 边距, 边框, 填充, 和实际内容。盒模型允许我们在其它元素和周围元素边框之间的空间放置元素。

![box-model](https://vxhly.github.io/assets/box-model.gif)

* `Margin(外边距)` - 清除边框外的区域, 外边距是透明的。
* `Border(边框)` - 围绕在内边距和内容外的边框。
* `Padding(内边距)` - 清除内容周围的区域, 内边距是透明的。
* `Content(内容)` - 盒子的内容, 显示文本和图像。

### class、id

class => 可以有多个, 使用 `.[className]` 表示 id => 只能写一个, 不能重复, 使用 `#[idName]` 表示

``` html
<div class="div1"></div>
<div id="div2"></div>
```

``` css
.div1 {}

#div2 {}
```

### margin、padding

``` css
.content {
    margin: top right bottom left;
    padding: top right bottom left;
}
```

只有一个值的时候, 表示四个方位都设置为该值；只有两个值的时候, 分别表示上下、左右；只有三个值；分别表示上、左右、下；有四个值的时候, 分别表示上、右、下、左

### border

``` css
.content {
    border: width style color;
}
```

* `border-style`
-- `dotted` => 定义点状边框
-- `dashed` => 定义虚线
-- `solid` => 定义实线
-- `double` => 定义双线

### background

``` css
.content {
    background: image repeat position attachment;
}
```

* `background-repeat`
-- `repeat-x` => 只在 X 轴上重复
-- `repeat-Y` => 只在 Y 轴上重复
-- `repeat` => 重复（默认值）
-- `no-repat` => 不重复

* `background-position` => left/right/botton/center

* `background-attachment` => 是否不跟着文档滚动而滚动

-- `fixed` => 不滚动
-- `scroll` => 滚动

### font

``` css
.content {
    font: style weight size/line-height family;
}
```

* `font-style` => italic/oblique
* `font-weight` => 粗细, bold 或者数字
* `font-family` => 字体风格, 空格分隔（最好英文）

### position

* `absolute` => 生成绝对定位的元素, 相对于 static 定位以外的第一个父元素进行定位。元素的位置通过 "left", "top", "right" 以及 "bottom" 属性进行规定。
* `fixed` => 生成绝对定位的元素, 相对于浏览器窗口进行定位。元素的位置通过 "left", "top", "right" 以及 "bottom" 属性进行规定。
* `relative` => 生成相对定位的元素, 相对于其正常位置进行定位。

### CSS 选择器

这里有详细介绍 [CSS 选择器介绍](https://vxhly.github.io/2017/03/css-selector-introduction/)

### 引入 css

可以分为三种

* 内联样式表

  

``` html
<p style="font-size:'16px'">测试</p>
```

* 内部样式表

  

``` html
<p>测试测试</p>

<style type="text/css">
    p:first-line {
        background-color: #dedede;
    }
</style>
```

* 外部样式表

  

``` html
<link rel="stylesheet" src="">
<!-- 或者 -->
<style>
    @import '';
    /* 这里只能有一个导入语句,不可再添加其他自定义的 css 语句 */
</style>
```

::: tip 提示
其优先级 => 内联样式表 > 内部样式表 > 外部样式表
:::

### 标准 CSS 样式表模板

推荐写 CSS 的时候要将浏览器一些默认样式清掉

``` css
@charset 'UTF-8'

/* 这里是注释信息 */

* {

    margin: 0;
    padding: 0;
}

body,
html {
    font-size: 16px;
    background-color: #fff;
    color: #000;
}
```

### 布局方式

* div+css 布局, [例子](https://github.com/vxhly/web-demo)
* flexbox 布局, [flexbox 布局](https://vxhly.github.io/2016/08/flexbox-layout/)# JavaScript

### 嵌入 JavaScript

``` html
<script src="">
    // 这里不得写自定义代码
</script>
<script>
    alert("Hello !!")
</script>
```

### 定义变量

::: tip 提示
不得以数字开头, 可以是 数字, 字母, _, $ 的组合, 不能是关键字
:::

### 数据类型

* `number`
* `string`
* `boolean`
* `object`
* `function`
可以使用 `typeof` 运算符来判断一个变量的数据类型

``` javascript
typeof 1; // number
typeof "a"; // string
typeof true; // boolean
typeof ["1"]; // object
typeof alert; // function
```

可以使用 `instanceof` 运算符来判断一个变量是否为 Array 或者 Object

``` javascript
{
    "k1": "v1"
}
instanceof Array // false
{
    "k1": "v1"
}
instanceof Object // true
    [1, 2] instanceof Array // true
[1, 2] instanceof Object // false
```

可以使用 `Number` 方法返回变量是否为数字

``` javascript
Number(1); // 1
Number("a"); // NaN
Number("1"); // 1
Number("1a"); // NaN
```

### 算术运算符

* `++` => `i++` 相当于 `i = i + 1` , 先赋值后自加; `++i` , 先自加后赋值
* `--` => 同理上
* `+`
* `-`
* `*`
* `/`
* `%`
* `+=` => `i += 2` 相当于 `i = i + 2`
* `-=` => 同理上
* `*=` => 同理上
* `/=` => 同理上
* `%/` => 同理上

### 逻辑运算符

* `&&` => 与, 运算符两边皆成立, 即返回 true, 否则返回 false
* `||` => 或, 运算符两边成立一边, 即返回 true, 否则返回 false
* `!` => 非

### alert/confirm/prompt

``` javascript
alert("hello");

if (confirm("你确定提交吗？")) {
    alert("点击了确定");
} else {
    alert("点击了取消");
}

var name = prompt("请输入您的名字", "");
if (name) {
    alert("欢迎您: " + name);
}
```

### function

``` javascript
function num(a, b) {
    return a + b;
}
document.write(num(1, 2)); // 3
```

### setTimeout/setInterval

* `setTimeout` 只会调用一次, 使用 `clearTimeout` 清掉定时器
* `setInterval` 无限调用, 使用 `clearInterval` 清掉定时器

``` javascript
setTimeout("alert('5 seconds!')", 5000);
setInterval("alert('5 seconds!')", 5000);
```

来个例子, [时钟](https://codepen.io/vxhly/pen/eyebbE)

### DOM 事件

[JavaScript 中的 DOM 事件](https://vxhly.github.io/2016/10/dom-event-in-javascript/)

### BOM 对象

[浏览器对象模型](https://vxhly.github.io/2016/08/browser-object-model/)

### Array

#### 数组的定义

* `var arr = []`
* `var arr = new Array()`
* `var arr = new Array(10)`
* `var arr = [1, '1', true]`
* `var arr = new Array(1, '1', true)`
::: warning 注意
`typeof new Array(1, '1', true)` 返回的是 object
:::

#### 常用属性和方法

* `length` => 数组的长度
* `concat()` => 拼接数组
* `sort()` => 排序

  

``` javascript
  arr.sort(funtion(a, b) {
      return b - a;
  }) // 降序,从大到小

  arr.sort(funtion(a, b) {
      return a - b;
  }) // 升序,从小到大
```

* `toString()` => 转化为字符串

* `reverse()` => 颠倒顺序

* `push()` => 向数组中添加元素

* `pop()` => 删除数组中最后一个元素

* `shift()` => 删除数组中第一个元素

更多属性方法的使用, 参照 [JavaScript Array 对象](http://www.w3school.com.cn/jsref/jsref_obj_array.asp)

### String

#### 字符串定义

* `var str = "string"`
* `var str = String("String")`
* `var str = new String("String")`
::: warning 注意
`typeof String("String")` 返回的是 string, 而 `typeof new String("String")` 返回的是 object
:::

#### 常用属性和方法

* `length` => 字符串的长度
* `split()` => 字符串根据某个字符串分割成数组
* `concat()` => 拼接字符串
* `indexOf()` => 返回指定字符在字符串的位置, 未找到返回 -1
* `lastIndexOf()` => 从字符串的末尾开始检索字符串, 返回结果同上
* `match()/search()` => 可以写正则表达式匹配检索
* `replace()` => 可以写正则表达式匹配替换
* `substr()` => 从起始索引号提取字符串中指定数目的字符
* `substring()` => 提取字符串中两个指定的索引号之间的字符
* `toLowerCase()` => 把字符串转换为小写
* `toUpperCase()` => 把字符串转换为大写

更多属性方法的使用, 参照 [JavaScript String 对象](http://www.w3school.com.cn/jsref/jsref_obj_string.asp)

### Math

#### 常用属性和方法

* `PI` => 圆周率
* `ceil()` => 向上取整, `Math.ceil(1.34)` => 2
* `floor()` => 向下取整, `Math.flor(1.34)` => 1
* `max()` => 返回最大值
* `min()` => 返回最小值

  

``` javascript
  // 返回数组中的最大值和最小值
  var arr = [1, 2, 3, 5];
  alert(Math.max.apply(null, arr)); //最大值
  alert(Math.min.apply(null, arr)); //最小值
```

> 更多属性方法的使用, 参照 [JavaScript Math 对象](http://www.w3school.com.cn/jsref/jsref_obj_math.asp)

### Date

#### 日期的定义

* `var now = new Date()`
* `var now = new Date(2016, 4, 1)` => 2016 年 5 月 1 日

#### 常用属性和方法

* `getDate()` => 返回一个月中的某一天 (1~31)
* `getDay()` => 返回一周中的某一天 (0~6)
* `getMonth()` => 返回月份 (0~11)
* `getFullYear()` => 完整的年份
* `getYear()` => 非完整年份(只有后两位)
* `getTime()` => 返回时间戳(毫秒, 1970 年 1 月 1 日至今的毫秒数)
* `toString()` => 转化成字符串
* `toLocaleString()` => 根据本地格式转换

更多属性方法的使用, 参照 [JavaScript Date 对象](http://www.w3school.com.cn/jsref/jsref_obj_date.asp)

### try-catch

``` javascript
try {
    document.getElementById("now").innerHTML = "hello";
} catch (e) {
    console.log("未找到对应元素", e);
}
```

### 分支语句语法

#### if-else

``` javascript
if () {
    //
} else if () {
    //
} else {
    //
}
```

#### while

``` javascript
while () {
    //
}
```

#### do-while

``` javascript
do {
    //
} while ()
```

#### for

``` javascript
for (;;) {
    //
}
```

#### switch-case

``` javascript
switch () {
    case '':
        //
        break;
    case '':
    case '':
        //
        break;
    default:
        //
}
```

#### for-in

``` javascript
for ( in ) {
    //
}
```
