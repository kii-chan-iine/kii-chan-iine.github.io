---
author: 星火燎原@vxhly
title: JavaScript 生成随机数
categories: [design]
tags: [JavaScript, Code]
date: 2016-08-02 13:49:41
---

<Boxx changeTime="10000"/>

::: tip 前言
为 JS 中的 Math 对象新增生成不重复随机数方法。
:::
<!-- more -->

## 方法源码一

``` javascript
/**
 * Math 对象的扩展方法,可用于生成随机数
 * @param  {number} startNum  起始数,整型
 * @param  {number} endNum    终止数,整型
 * @param  {number} count     生成的个数,整型
 * @param  {boolean} repeat   是否重复,true or false
 * @return {Array}            返回数组
 */
Math.randNum = function(startNum, endNum, count, repeat) {
    if (repeat) {
        var ret = new Array();
        for (var i = 0; i < count; i++) {
            // 生成区间范围内的随机数
            ret[i] = Math.floor(Math.random() * (endNum - startNum) + startNum);
        }
        return ret;
    } else {
        var tmp = new Array();
        for (var i = 0;; i++) {
            if (tmp.length < count) {
                // 生成区间范围内的随机数
                var rand = Math.floor(Math.random() * (endNum - startNum) + startNum);
                for (var i = 0; i < tmp.length; i++) {
                    if (tmp[i] == rand) {
                        // 将生成的随机数进行比对,如果相同则重新生成
                        return Math.randNum(startNum, endNum, count, repeat);
                    }
                }
                // 向数组添加数据
                tmp.push(rand);
            } else {
                break;
            }
        }
        return tmp;
    }
};
```

### 如何调用

``` javascript
Math.randNum(number, number, number, true | false);
```

### demo1

控制台代码

``` javascript
console.log(Math.randNum(3, 16, 5, true);)
```

输出 => `[4, 15, 15, 15, 17]`

### demo2

控制台代码

``` javascript
console.log(Math.randNum(3, 16, 5, false);)
```

输出 => `[15, 14, 6, 11, 4]`

## 方法源码二

> 一种更简单的写法

``` javascript
var arr = new Array(); //新建一个数组

// 向数组赋值 0~100
for (var i = 0; i <= 100; i++) {
    arr.push(i);
}

// 随机排位
arr.sort(function() {
    return 0.5 - Math.random();
});

arr.length = 50; // 指定数组长度

document.write(arr); // 输出数组
```
