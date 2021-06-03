---
author: 星火燎原@vxhly
title: 简单又实用的 ES6 方法
categories: [web]
tags: [JavaScript, Code, ES6]
date: 2019-11-23 11:48:00
---

<Boxx changeTime="10000"/>

::: tip 前言
收集日常开发中经常用到的一些 ES6 方法（PS：又一次的懒人速养记）。
:::
<!-- more -->

## 返回新数组

::: tip 需求
对数组 [1, 2, 3] 分别加上 2, 返回的是一个新的数组
:::

``` javascript
[1, 2, 3].map(item => item += 2)
// 返回 [3, 4, 5]
```

## 数组去重

::: tip 需求
对数组 [1, 2, 3, 4, 3, 5] 去除重复的元素, 返回的是一个新的数组
:::

``` javascript
[...new Set([1, 2, 3, 4, 3, 5])]
// 返回 [1, 2, 3, 4, 5]
```

## 数组去元素

::: tip 需求
对数组 [1, 2, 3, 4, 3, '', 0, 5, '', 2] 去除空元素, 返回的是一个非零新的数组
:::

``` javascript
[1, 2, 3, 4, 3, '', 0, 5, '', 2].filter(item => item)
// 返回 [1, 2, 3, 4, 3, 5, 2]
```

## 字符串去除重复字符

::: tip 需求
对字符串 `JavaScipt WEB Vue.js` 删除重复的元素, 区分大小写
:::

``` javascript
Array.from(new Set([..."JavaScipt WEB Vue.js"])).join("")
// 返回 "JavScipt WEBVue.js"
```

## 字符串去除空格

::: tip 需求
对字符串 `JavaScipt WEB Vue.js` 删除所有空格
:::

``` javascript
Array.from([..."JavaScipt WEB Vue.js"].filter(item => item !== " ")).join("")
// 返回 JavaSciptWEBVue.js"
```

## 打乱数组

::: tip 需求
对数组 ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 进行随机打乱
:::

``` javascript
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'].sort(() => {
    return Math.random() - Math.random();
})
```

## 升序数组

::: tip 需求
对数组 ["0", "1", "2", "6", "3", "4", "5", "7", "8", "9"] 进行排序, 排序方式为升序
:::

``` javascript
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'].sort((a, b) => {
    return a - b;
})
```

## 降序数组

::: tip 需求
对数组 ["0", "1", "2", "6", "3", "4", "5", "7", "8", "9"] 进行排序, 排序方式为降序
:::

``` javascript
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'].sort((a, b) => {
    return b - a;
})
```

## 反转数组

::: tip 需求
对数组 [1, 2, 3, 4, 3, '', 0, 5, '', 2] 进行反转
:::

``` javascript
[1, 2, 3, 4, 3, '', 0, 5, '', 2].reverse()
// 返回 [2, "", 5, 0, "", 3, 4, 3, 2, 1]
```

## 字符串反转

::: tip 需求
对字符串 `JavaScipt WEB Vue.js` 进行反转
:::

``` javascript
"JavaScipt WEB Vue.js".split('').reverse().join('')
// "sj.euV BEW tpicSavaJ"
```

## 数值格式化添加字符

::: tip 需求
对数值 `1234567890` 每隔 3 位添加一个逗号
:::

``` javascript
String("1234567890")
    .replace(/\d+/, function(s) {
        return s.replace(/(\d)(?=(\d{3})+$)/g, '$1,')
    })
// "1,234,567,890"
```

::: tip 需求
对电话号码 `15012365498` 每隔 4 位添加一个空格
:::

``` javascript
String("15012365498")
    .replace(/\d+/, function(s) {
        return s.replace(/(\d)(?=(\d{4})+$)/g, '$1 ')
    })
// "150 1236 5498"
```

::: tip 需求
对身份证号码 `123456201910011234` 格式化成 `xxxxxx xxxx xxxx xxxx` 形式
:::

``` javascript
String("123456201910011234")
    .replace(/\d+/, function(s) {
        return s.replace(/(\d)(?=(\d{4})+$)/g, '$1 ')
            .replace(/^(?:\S*?)(\s)/, function(s) {
                return s.replace(/ /, '')
            })
    })
// "123456 2019 1001 1234"
```

## 返回数组中的最大值和最小值

::: tip 需求
返回数组 `[1, 2, 3]` 中的最大值和最小值
:::

``` javascript
Math.max(...[1, 2, 3])
// 3
Math.min(...[1, 2, 3])
// 1
Math.max.apply(null, [1, 2, 3])
// 3
Math.min.apply(null, [1, 2, 3])
// 1
```
