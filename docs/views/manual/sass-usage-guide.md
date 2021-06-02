---
author: 星火燎原@vxhly
title: SASS 用法指南
categories: [manual]
tags: [CSS]
date: 2016-10-11 18:57:49
---

<Boxx changeTime="10000"/>

::: tip 前言
SASS 是 CSS 预处理器之一, 它的基本思想是, 用一种专门的编程语言, 进行网页样式设计, 然后再编译成正常的 CSS 文件, 它提供了许多便利的写法, 大大节省了设计者的时间, 使得 CSS 的开发, 变得简单和可维护。
:::
<!-- more -->

## 环境布置

### 安装 Ruby

SASS 是 Ruby 语言写的, 但是两者的语法没有关系。不懂 Ruby, 照样使用。只是必须先[安装 Ruby](http://www.ruby-lang.org/zh_cn/downloads/), 然后再安装 SASS.

在 Ubuntu 下安装 Ruby, `For Example:`

``` bash
sudo apt-get install ruby
```

查看版本号, 确认 Ruby 环境安装完成

``` bash
ruby -v
```

### 安装 SASS

Ruby 安装好了, 之后执行命令

``` bash
gem install sass
```

查看版本号, 确认 SASS 环境安装完成

``` bash
sass -v
```

## 使用

### 编译

SASS 文件就是普通的文本文件, 里面可以直接使用 CSS 语法。文件后缀名是 `.scss` , 意思为 `Sassy CSS` 。

下面的命令, 可以在屏幕上显示 `.scss` 文件转化的 `.css` 代码。（假设文件名为 `test` 。）

``` bash
sass test.scss
```

### 编译风格

如果要将显示结果保存成文件, 后面再跟一个 `.css` 文件名, `For Example:`

``` bash
sass test.scss test.css
```

SASS 提供四个[编译风格](http://sass-lang.com/docs/yardoc/file. SASS_REFERENCE.html#output_style)的选项

* **nested** => 嵌套缩进的 css 代码, 它是默认值。
* **expanded** => 没有缩进的、扩展的 css 代码。
* **compact** => 简洁格式的 css 代码。
* **compressed** => 压缩后的 css 代码。

生产环境当中, 一般使用最后一个选项, `For Example:`

``` bash
sass --style compressed test.sass test.css
```

你也可以让 SASS 监听某个文件或目录, 一旦源文件有变动, 就自动生成编译后的版本

``` bash
sass --watch input.scss:output.css            # watch a file
sass --watch app/sass:public/stylesheets      # watch a directory
```

## 基本用法

### 变量

SASS 允许使用变量, 所有变量以 开头。 `For Example:`

``` scss
$blue: #1875e7;

div {
    color: $blue;
}
```

如果变量需要镶嵌在字符串之中, 就必须需要写在 `#{}` 之中。 `For Example:`

``` scss
$side: left;

.rounded {
    border-#{$side}-radius: 5px;
}
```

### 计算功能

SASS 允许在代码中使用算式。 `For Example:`

``` scss
body {
    margin: (14px/2);
    top: 50px + 100px;
    right: $var * 10%;
}
```

### 嵌套

SASS 允许选择器嵌套。比如, 下面的 CSS 代码: 

``` scss
div h1 {
    color: red;
}
```

可以写成: 

``` scss
div {
    h1 {
        color: red;
    }
}
```

属性也可以嵌套, 比如 `border-color` 属性, 可以写成: 

``` scss
p {
    border: {
        color: red;
    }
}
```

::: warning 注意
`border` 后面必须加上冒号。
:::

在嵌套的代码块内, 可以使用 `&` 引用父元素。比如 `a:hover` 伪类, 可以写成: 

``` scss
a {
    &:hover {
        color: #ffb3ff;
    }
}
```

### 注释

SASS 共有两种注释风格。

标准的 CSS 注释 `/* comment */` , 会保留到编译后的文件。 单行注释 `// comment` , 只保留在 SASS 源文件中, 编译后被省略。

在 `/*` 后面加一个感叹号, 表示这是"重要注释"。即使是压缩模式编译, 也会保留这行注释, 通常可以用于声明版权信息。

``` scss
/*!
 * 重要注释！
 */
```

## 代码的重用

### 继承

SASS 允许一个选择器, 继承另一个选择器。比如, 现有 `class1` : 

``` scss
.class1 {
    border: 1px solid #ddd;
}
```

`class2` 要继承 `class1` , 就要使用 `@extend` 命令: 

``` scss
.class2 {
    @extend .class1;
    font-size: 120%;
}
```

### Mixin

Mixin 有点像 C 语言的宏（ `macro` ）, 是可以重用的代码块。 使用 `@mixin` 命令, 定义一个代码块。

``` scss
@mixin left {
    float: left;
    margin-left: 10px;
}
```

使用 `@include` 命令, 调用这个 `mixin` 。

``` scss
div {
    @include left;
}
```

`mixin` 的强大之处, 在于可以指定参数和缺省值。

``` scss
@mixin left($value: 10px) {
    float: left;
    margin-right: $value;
}
```

使用的时候, 根据需要加入参数: 

``` scss
div {
    @include left(20px);
}
```

下面是一个 `mixin` 的实例, 用来生成浏览器前缀。

``` scss
@mixin rounded($vert, $horz, $radius: 10px) {
    border-#{$vert}-#{$horz}-radius: $radius;
    -moz-border-radius-#{$vert}#{$horz}: $radius;
    -webkit-border-#{$vert}-#{$horz}-radius: $radius;
}
```

使用的时候, 可以像下面这样调用: 

``` scss
#navbar li {
    @include rounded(top, left);
}

#footer {
    @include rounded(top, left, 5px);
}
```

### 颜色函数

SASS 提供了一些内置的颜色函数, 以便生成系列颜色。

``` scss
lighten(#cc3, 10%) // #d6d65c
darken(#cc3, 10%) // #a3a329
grayscale(#cc3) // #808080
complement(#cc3) // #33c
```

### 插入文件

`@import` 命令, 用来插入外部文件。

``` scss
@import "path/filename.scss";
```

如果插入的是 `.css` 文件, 则等同于 css 的 `import` 命令。

``` scss
@import "foo.css";
```

## 高级用法

## 条件语句

`@if` 可以用来判断: 

``` scss
p {
    @if 1+1==2 {
        border: 1px solid;
    }

    @if 5 < 3 {
        border: 2px dotted;
    }
}
```

配套的还有 `@else` 命令: 

``` scss
@if lightness($color)>30% {
    background-color: #000;
}

@else {
    background-color: #fff;
}
```

### 循环语句

SASS 支持 `for` 循环: 

``` scss
@for $i from 1 to 10 {
    .border-#{$i} {
        border: #{$i}px solid blue;
    }
}
```

也支持 `while` 循环: 

``` scss
$i: 6;

@while $i>0 {
    .item-#{$i} {
        width: 2em * $i;
    }

    $i: $i - 2;
}
```

`each` 命令, 作用与 `for` 类似: 

``` scss
@each $member in a,
b,
c,
d {
    .#{$member} {
        background-image: url("/image/#{$member}.jpg");
    }
}
```

### 自定义函数

SASS 允许用户编写自市场上市己的函数。

``` scss
@function double($n) {
    @return $n * 2;
}

#sidebar {
    width: double(5px);
}
```
