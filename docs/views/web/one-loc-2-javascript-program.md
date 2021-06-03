---
author: 星火燎原@vxhly
title: 单行代码就能实现的 JavaScript 实用程序
categories: [web]
tags: [JavaScript, Code, ES6]
date: 2020-10-29 16:29:37
---

<Boxx changeTime="10000"/>

::: tip 前言
【纯属翻译】翻译来源地址：[212 Favorite JavaScript Utilities in single line of code! No more!](https://1loc.dev/)。
:::
<!-- more -->

## Array

### 检查数组是否为空

``` javascript
// `arr` is an array
const isEmpty = arr => !Array.isArray(arr) || arr.length === 0;

// Examples
isEmpty([]); // true
isEmpty([1, 2, 3]); // false
```

### 克隆数组

``` javascript
// `arr` is an array
const clone = arr => arr.slice(0);

// Or
const clone = arr => [...arr];

// Or
const clone = arr => Array.from(arr);

// Or
const clone = arr => arr.map(x => x);

// Or
const clone = arr => JSON.parse(JSON.stringify(arr));

// Or
const clone = arr => arr.concat([]);
```

### 比较两个数组, 而不考虑顺序

``` javascript
// `a` and `b` are arrays
const isEqual = (a, b) => JSON.stringify(a.sort()) === JSON.stringify(b.sort());

// Examples
isEqual([1, 2, 3], [1, 2, 3]); // true
isEqual([1, 2, 3], [1, 3, 2]); // true
isEqual([1, 2, 3], [1, '2', 3]); // false
```

### 比较两个数组

``` javascript
// `a` and `b` are arrays
const isEqual = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// Or
const isEqual = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// Examples
isEqual([1, 2, 3], [1, 2, 3]); // true
isEqual([1, 2, 3], [1, '2', 3]); // false
```

### 将数组转换为单个对象

``` javascript
const toObject = (arr, key) => arr.reduce((a, b) => ({
    ...a,
    [b[key]]: b
}), {});

// Example
toObject(
    [{
            id: '1',
            name: 'Alpha',
            gender: 'Male'
        },
        {
            id: '2',
            name: 'Bravo',
            gender: 'Male'
        },
        {
            id: '3',
            name: 'Charlie',
            gender: 'Female'
        },
    ],
    'id'
);
/* 
{
    '1': { id: '1', name: 'Alpha', gender: 'Male' },
    '2': { id: '2', name: 'Bravo', gender: 'Male' },
    '3': { id: '3', name: 'Charlie', gender: 'Female' },
}
*/
```

### 将字符串数组转换成数字

``` javascript
const toNumbers = arr => arr.map(Number);

// Or
const toNumbers = arr => arr.map(x => +x);

// Example
toNumbers(['2', '3', '4']); // [2, 3, 4]
```

### 按对象数组的属性计数

``` javascript
const countBy = (arr, prop) => arr.reduce((prev, curr) => (prev[curr[prop]] = ++prev[curr[prop]] || 1, prev), {});

// Example
countBy([{
        branch: 'audi',
        model: 'q8',
        year: '2019'
    },
    {
        branch: 'audi',
        model: 'rs7',
        year: '2020'
    },
    {
        branch: 'ford',
        model: 'mustang',
        year: '2019'
    },
    {
        branch: 'ford',
        model: 'explorer',
        year: '2020'
    },
    {
        branch: 'bmw',
        model: 'x7',
        year: '2020'
    },
], 'branch');

// { 'audi': 2, 'ford': 2, 'bmw': 1 }
```

### 计算数组中的某个值的出现

``` javascript
const countOccurrences = (arr, val) => arr.reduce((a, v) => (v === val ? a + 1 : a), 0);

// Examples
countOccurrences([2, 1, 3, 3, 2, 3], 2); // 2
countOccurrences(['a', 'b', 'a', 'c', 'a', 'b'], 'a'); // 3
```

### 计算数组元素的出现次数

``` javascript
const countOccurrences = arr => arr.reduce((prev, curr) => (prev[curr] = ++prev[curr] || 1, prev), {});

// Examples
countOccurrences([2, 1, 3, 3, 2, 3]); // { '1': 1, '2': 2, '3': 3 }
countOccurrences(['a', 'b', 'a', 'c', 'a', 'b']); // { 'a': 3, 'b': 2, 'c': 1 }
```

### 创建一个累积和的数组

``` javascript
const accumulate = arr => arr.map((sum => value => sum += value)(0));

// Or
const accumulate = arr => arr.reduce((a, b, i) => i === 0 ? [b] : [...a, b + a[i - 1]], []);

// Or
const accumulate = arr => arr.reduce((a, b, i) => i === 0 ? [b] : [...a, b + a[i - 1]], 0);

// Example
accumulate([1, 2, 3, 4]); // [1, 3, 6, 10]
// 1             = 1
// 1 + 2         = 3
// 1 + 2 + 3     = 6
// 1 + 2 + 3 + 4 = 10
```

### 给定范围内创建一个数字数组

``` javascript
const range = (min, max) => [...Array(max - min + 1).keys()].map(i => i + min);

// Or
const range = (min, max) => Array(max - min + 1).fill(0).map((_, i) => min + i);

// Or
const range = (min, max) => Array.from({
    length: max - min + 1
}, (_, i) => min + i);

// Example
range(5, 10); // [5, 6, 7, 8, 9, 10]
```

### 空数组

``` javascript
const empty = arr => arr.length = 0;

// Or
arr = [];
```

### 在数组中查找最接近的数字

``` javascript
// Find the number from `arr` which is closest to `n`
const closest = (arr, n) => arr.reduce((prev, curr) => Math.abs(curr - n) < Math.abs(prev - n) ? curr : prev);

// Or
const closest = (arr, n) => arr.sort((a, b) => Math.abs(a - n) - Math.abs(b - n))[0];

// Example
closest([29, 87, 8, 78, 97, 20, 75, 33, 24, 17], 50); // 33
```

### 查找数字中最长字符串的长度

``` javascript
const findLongest = words => Math.max(...(words.map(el => el.length)));

// Example
findLongest(['always', 'look', 'on', 'the', 'bright', 'side', 'of', 'life']); // 6
```

### 从一个数组中找到最大的元素

``` javascript
const max = arr => Math.max(...arr);
```

### 从一个数组中找到最小的元素

``` javascript
const min = arr => Math.min(...arr);
```

### 展平数组

``` javascript
const flat = arr => [].concat.apply([], arr.map(a => Array.isArray(a) ? flat(a) : a));
// Or
const flat = arr => arr.reduce((a, b) => Array.isArray(b) ? [...a, ...flat(b)] : [...a, b], []);

// Or
// See the browser compatibility at https://caniuse.com/#feat=array-flat
const flat = arr => arr.flat();

// Example
flat(['cat', ['lion', 'tiger']]); // ['cat', 'lion', 'tiger']
```

### 获取数组的平均值

``` javascript
const average = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
```

### 获取数组的交集

``` javascript
const getIntersection = (a, ...arr) => [...new Set(a)].filter(v => arr.every(b => b.includes(v)));

// Examples
getIntersection([1, 2, 3], [2, 3, 4, 5]); // [2, 3]
getIntersection([1, 2, 3], [2, 3, 4, 5], [1, 3, 5]); // [3]
```

### 获取数字数组的总和

``` javascript
const sum = arr => arr.reduce((a, b) => a + b, 0);
```

### 获取数组的唯一值

``` javascript
const unique = arr => [...new Set(arr)];

// Or
const unique = arr => arr.filter((el, i, array) => array.indexOf(el) === i);

// Or
const unique = arr => arr.reduce((acc, el) => acc.includes(el) ? acc : [...acc, el], []);
```

### 获取数组的并集

``` javascript
const union = (...arr) => [...new Set(arr.flat())];

// Example
union([1, 2], [2, 3], [3]); // [1, 2, 3]
```

### 通过键将对象数组分组

``` javascript
const groupBy = (arr, key) => arr.reduce((acc, item) => ((acc[item[key]] = [...(acc[item[key]] || []), item]), acc), {});

// Example
groupBy([{
        branch: 'audi',
        model: 'q8',
        year: '2019'
    },
    {
        branch: 'audi',
        model: 'rs7',
        year: '2020'
    },
    {
        branch: 'ford',
        model: 'mustang',
        year: '2019'
    },
    {
        branch: 'ford',
        model: 'explorer',
        year: '2020'
    },
    {
        branch: 'bmw',
        model: 'x7',
        year: '2020'
    },
], 'branch');

/*
{
    audi: [
        { branch: 'audi', model: 'q8', year: '2019' },
        { branch: 'audi', model: 'rs7', year: '2020' }
    ],
    bmw: [
        { branch: 'bmw', model: 'x7', year: '2020' }
    ],
    ford: [
        { branch: 'ford', model: 'mustang', year: '2019' },
        { branch: 'ford', model: 'explorer', year: '2020' }
    ],
}
*/
```

### 合并两个数组

``` javascript
// Merge but don't remove the duplications
const merge = (a, b) => a.concat(b);
// Or
const merge = (a, b) => [...a, ...b];

// Merge and remove the duplications
const merge = [...new Set(a.concat(b))];
// Or
const merge = [...new Set([...a, ...b])];
```

### 根据条件对数组进行分区

``` javascript
const partition = (arr, criteria) => arr.reduce((acc, i) => (acc[criteria(i) ? 0 : 1].push(i), acc), [
    [],
    []
]);

// Example
partition([1, 2, 3, 4, 5], n => n % 2); // [[2, 4], [1, 3, 5]]
```

### 从数组中删除虚假值

``` javascript
const removeFalsy = arr => arr.filter(Boolean);

// Example
removeFalsy([0, 'a string', '', NaN, true, 5, undefined, 'another string', false]); // ['a string', true, 5, 'another string']
```

### 随机排列数组

``` javascript
const shuffle = arr => arr.map(a => ({
    sort: Math.random(),
    value: a
})).sort((a, b) => a.sort - b.sort).map(a => a.value);

// Or
const shuffle = arr => arr.sort(() => .5 - Math.random());

// Example
shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]); // [9, 1, 10, 6, 8, 5, 2, 3, 7, 4]
```

### 排序数字数组

``` javascript
const sort = arr => arr.sort((a, b) => a - b);

// Example
sort([1, 5, 2, 4, 3]); // [1, 2, 3, 4, 5]
```

### 将数组拆分为大块

``` javascript
const chunk = (arr, size) => arr.reduce((acc, e, i) => (i % size ? acc[acc.length - 1].push(e) : acc.push([e]), acc), []);

// Examples
chunk([1, 2, 3, 4, 5, 6, 7, 8], 3); // [[1, 2, 3], [4, 5, 6], [7, 8]]
chunk([1, 2, 3, 4, 5, 6, 7, 8], 4); // [[1, 2, 3, 4], [5, 6, 7, 8]]
```

### 交换矩阵的行和列

``` javascript
const transpose = matrix => matrix[0].map((col, i) => matrix.map(row => row[i]));

// Or
const transpose = matrix => matrix[0].map((col, c) => matrix.map((row, r) => matrix[r][c]));

// Or
const transpose = matrix => matrix.reduce((prev, next) => next.map((item, i) => (prev[i] || []).concat(next[i])), []);

// Example
transpose([ // [
    [1, 2, 3], //      [1, 4, 7],
    [4, 5, 6], //      [2, 5, 8],
    [7, 8, 9], //      [3, 6, 9],
]); //  ]
```

### 解压缩数组中的数组

``` javascript
const unzip = arr => arr.reduce((acc, c) => (c.forEach((v, i) => acc[i].push(v)), acc), Array.from({
    length: Math.max(...arr.map(a => a.length))
}, (_) => []));

// Example
unzip([
    ['a', 1],
    ['b', 2],
    ['c', 3],
    ['d', 4],
    ['e', 5]
]); // [['a', 'b', 'c', 'd', 'e'], [1, 2, 3, 4, 5]]

/*
    a     1
     b   2
      c 3
      d 4
      e 5
*/
```

### 压缩多个数组

``` javascript
const zip = (...arr) => Array.from({
    length: Math.max(...arr.map(a => a.length))
}, (_, i) => arr.map(a => a[i]));

// Example
zip(['a', 'b', 'c', 'd', 'e'], [1, 2, 3, 4, 5]); // [['a', 1], ['b', 2], ['c', 3], ['d', 4], ['e', 5]]

/*
Does it look like a zipper?
        a 1
        b 2
       c   3
      d     4
     e       5
*/
```

## 日期时间

### 计算两个日期之间的差异天数

``` javascript
const diffDays = (date, otherDate) => Math.ceil(Math.abs(date - otherDate) / (1000 * 60 * 60 * 24));

// Example
diffDays(new Date('2014-12-19'), new Date('2020-01-01')); // 1839
```

### 计算两个日期之间的月数

``` javascript
const monthDiff = (startDate, endDate) => Math.max(0, (endDate.getFullYear() - startDate.getFullYear()) * 12 - startDate.getMonth() + endDate.getMonth());

// Example
monthDiff(new Date('2020-01-01'), new Date('2021-01-01')); // 12
```

### 比较两个日期

``` javascript
// `a` and `b` are `Date` instances
const compare = (a, b) => a.getTime() > b.getTime();

// Example
compare(new Date('2020-03-30'), new Date('2020-01-01')); // true
```

### 将日期转换为 yyyy-mm-dd 格式

``` javascript
// `date` is a `Date` object
const formatYmd = date => date.toISOString().slice(0, 10);

// Example
formatYmd(new Date()); // 2020-05-06
```

### 将秒转换为 hh:mm:ss 格式

``` javascript
// `s` is number of seconds
const formatSeconds = s => new Date(s * 1000).toISOString().substr(11, 8);

// Or
const formatSeconds = s => (new Date(s * 1000)).toUTCString().match(/(\d\d:\d\d:\d\d)/)[0];

// Or
const formatSeconds = s => [parseInt(s / 60 / 60), parseInt(s / 60 % 60), parseInt(s % 60)].join(':').replace(/\b(\d)\b/g, '0$1');

// Examples
formatSeconds(200); // 00:03:20
formatSeconds(500); // 00:08:20
```

### 从日期中提取年月日时分秒和毫秒

``` javascript
// `date` is a `Date` object
const extract = date => date.toISOString().split(/[^0-9]/).slice(0, -1);

// `extract` is an array of [year, month, day, hour, minute, second, millisecond]
```

### 格式化给定语言环境的日期

``` javascript
// `date` is a `Date` object
// `locale` is a locale (en-US, pt-BR, for example)
const format = (date, locale) => new Intl.DateTimeFormat(locale).format(date);

// Example
format(new Date(), 'pt-BR'); // 06/05/2020
```

### 以秒为单位获取当前时间戳

``` javascript
const ts = () => Math.floor(new Date().getTime() / 1000);
```

### 从日期获取一年中的某天

``` javascript
// `date` is a Date object
const dayOfYear = date => Math.floor((date - new Date(date.getFullYear(), 0, 0)) / (1000 * 60 * 60 * 24));

// Example
dayOfYear(new Date(2020, 04, 16)); // 137
```

### 获取日期的月份名称

``` javascript
// `date` is a Date object
const getMonthName = date => ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', ' November', 'December'][date.getMonth()];
```

### 获取给定月份的天数

``` javascript
// `month` is zero-based index
const daysInMonth = (month, year) => new Date(year, month, 0).getDate();
```

### 获取明天的日期

``` javascript
const tomorrow = (d => new Date(d.setDate(d.getDate() + 1)))(new Date);

// Or
const tomorrow = new Date((new Date()).valueOf() + 1000 * 60 * 60 * 24);
```

### 获取日期的工作日

``` javascript
// `date` is a Date object
const getWeekday = date => ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][date.getDay()];
```

### 获取昨天的日期

``` javascript
const yesterday = (d => new Date(d.setDate(d.getDate() - 1)))(new Date);

// Or
const yesterday = new Date((new Date()).valueOf() - 1000 * 60 * 60 * 24);
```

### 排序日期数组

``` javascript
// `arr` is an array of `Date` items
const sortDescending = arr => arr.sort((a, b) => a.getTime() > b.getTime());
const sortAscending = arr => arr.sort((a, b) => a.getTime() < b.getTime());
```

### 检查日期是否为工作日

``` javascript
const isWeekday = (date) => date.getDay() % 6 !== 0;
console.log(isWeekday(new Date(2021, 0, 11)));
// Result: true (Monday)
console.log(isWeekday(new Date(2021, 0, 10)));
// Result: false (Sunday)
```

### 从日期中获取时间

``` javascript
const timeFromDate = date => date.toTimeString().slice(0, 8);
console.log(timeFromDate(new Date(2021, 0, 10, 17, 30, 0))); 
// Result: "17:30:00"
console.log(timeFromDate(new Date()));
// Result: will log the current time
```

## DOM

### 检查一个元素是否是另一个元素的后代

``` javascript
const isDescendant = (child, parent) => parent.contains(child);
```

### 检查元素是否获得焦点

``` javascript
const elementIsInFocus = (el) => (el === document.activeElement);
elementIsInFocus(anyElement)
// Result: will return true if in focus, false if not in focus
```

### 检查是否支持触摸事件

``` javascript
const touchSupported = () => {
  ('ontouchstart' in window || window.DocumentTouch && document instanceof window.DocumentTouch);
}
console.log(touchSupported());
// Result: will return true if touch events are supported, false if not
```

### 检测 Internet Explorer 浏览器

``` javascript
const isIE = !!document.documentMode;
```

### 检测 macos 浏览器

``` javascript
const isAppleDevice = /Mac|iPod|iPhone|iPad/.test(navigator.platform);
console.log(isAppleDevice);
// Result: will return true if user is on an Apple device
```

### 获取元素的所有同级

``` javascript
const siblings = ele => [].slice.call(ele.parentNode.children).filter((child) => (child !== ele));
```

### 获取所选文本

``` javascript
const getSelectedText = () => window.getSelection().toString();
```

### 返回上一个页面

``` javascript
history.back();

// Or
history.go(-1);
```

### 隐藏元素

``` javascript
// Pick the method that is suitable for your use case
const hide = ele => ele.style.display = 'none';

// Or
const hide = ele => ele.style.visibility = 'hidden';
```

### 在另一个元素之后插入一个元素

``` javascript
const insertAfter = (ele, anotherEle) => anotherEle.parentNode.insertBefore(ele, anotherEle.nextSibling);

// Or
const insertAfter = (ele, anotherEle) => anotherEle.insertAdjacentElement('afterend', ele);
```

### 在其他元素之前插入元素

``` javascript
const insertBefore = (ele, anotherEle) => anotherEle.parentNode.insertBefore(ele, anotherEle);

// Or
const insertBefore = (ele, anotherEle) => anotherEle.insertAdjacentElement('beforebegin', ele);
```

### 在元素后插入给定的 html

``` javascript
const insertHtmlAfter = (html, ele) => ele.insertAdjacentHTML('afterend', html);
```

### 在元素之前插入给定的 html

``` javascript
const insertHtmlBefore = (html, ele) => ele.insertAdjacentHTML('beforebegin', html);
```

### 重定向到另一个页面

``` javascript
const goTo = url => location.href = url;
```

### 重新加载当前页面

``` javascript
const reload = () => location.reload();

// Or
const reload = () => (location.href = location.href);
```

### 替换元素

``` javascript
const replace = (ele, newEle) => ele.parentNode.replaceChild(newEle, ele);
```

### 滚动到页面顶部

``` javascript
const goToTop = () => window.scrollTo(0, 0);
goToTop();
// Result: will scroll the browser to the top of the page
```

### 显示元素

``` javascript
const show = ele => ele.style.display = '';
```

### 从给定的文本中删除 HTML

``` javascript
const stripHtml = html => (new DOMParser().parseFromString(html, 'text/html')).body.textContent || '';
```

### 切换元素

``` javascript
const toggle = ele => (ele.style.display = (ele.style.display === 'none') ? 'block' : 'none');
```

### 检查当前 Tab 页是否在前台

``` javascript
const isBrowserTabInView = () => document.hidden;
isBrowserTabInView();
// Result: returns true or false depending on if tab is in view / focus
```

## 功能

### 装箱机

``` javascript
const boxHandler = x => ({ next: f => boxHandler(f(x)), done: f => f(x) });

// Example 1
const getPercentNumber = str => boxHandler(str).next(str => str.replace(/\%/, '')).next(str => parseFloat(str)).done(res => res * 0.01);

getPercentNumber('50%');    // 0.5

// Example 2
const getMoney = (price) => Number.parseFloat(price.replace(/\$/, ''));
const getPercent = (percent) => Number.parseFloat(percent.replace(/\%/)) * 0.01;

const getDiscountPrice = (price, discount) => boxHandler(getMoney(price)).done(cents => boxHandler(getPercent(discount)).next(save => cents - (cents * save))).done(res => res);

getDiscountPrice('$6.00', '20%');    // 4.8
```

### 检查值是否为函数

``` javascript
const isFunction = v => ['[object Function]', '[object GeneratorFunction]', '[object AsyncFunction]', '[object Promise]'].includes(Object.prototype.toString.call(v));

// Examples
isFunction(function() {});          // true
isFunction(function*() {});         // true
isFunction(async function() {});    // true
```

### 检查值是否是生成器函数

``` javascript
const isGeneratorFunction = v => Object.prototype.toString.call(v) === '[object GeneratorFunction]';

// Examples
isGeneratorFunction(function() {});     // false
isGeneratorFunction(function*() {});    // true
```

### 检查值是否是异步函数

``` javascript
const isAsyncFunction = v => Object.prototype.toString.call(v) === '[object AsyncFunction]';

// Examples
isAsyncFunction(function() {});         // false
isAsyncFunction(function*() {});        // false
isAsyncFunction(async function() {});   // true
```

### 从左到右编写功能

``` javascript
// Compose functions from left to right
const pipe = (...fns) => x => fns.reduce((y, f) => f(y), x);

// Example
const lowercase = str => str.toLowerCase();
const capitalize = str => `${str.charAt(0).toUpperCase()}${str.slice(1)}`;
const reverse = str => str.split('').reverse().join('');

const fn = pipe(lowercase, capitalize, reverse);

// We will execute `lowercase`, `capitalize` and `reverse` in order
fn('Hello World') === 'dlrow olleH';
```

### 撰写功能

``` javascript
// Compose functions from right to left
const compose = (...fns) => x => fns.reduceRight((y, f) => f(y), x);

// Example
const lowercase = str => str.toLowerCase();
const capitalize = str => `${str.charAt(0).toUpperCase()}${str.slice(1)}`;
const reverse = str => str.split('').reverse().join('');

const fn = compose(reverse, capitalize, lowercase);

// We will execute `lowercase`, `capitalize` and `reverse` in order
fn('Hello World') === 'dlrow olleH';
```

### 创建一个空函数

``` javascript
const noop = () => {};

// Or
const noop = Function();
// calling Function() might be detected as using eval by some security tools
```

### 咖喱功能

``` javascript
const curry = (fn, ...args) => fn.length <= args.length ? fn(...args) : curry.bind(null, fn, ...args);

// Example
const sum = (a, b, c) => a + b + c;
curry(sum)(1)(2)(3);    // 6
curry(sum)(1, 2, 3);    // 6
curry(sum, 1)(2, 3);    // 6
curry(sum, 1)(2)(3);    // 6
curry(sum, 1, 2)(3);    // 6
curry(sum, 1, 2, 3);    // 6
```

### 延迟功能评估

``` javascript
// returns a new version of `fn` that returns values as lazy evaluable
const thunkfy = fn => (...args) => () => fn(...args);

// Example
const heavyComputation = x => doStuff(x);
const unnecessarySlow = manyThings.map(heavyComputation)
    .find(result => result.criteria);
const probablyFaster = manyThings.map(thunkfy(heavyComputation))
    .find(thunk => thunk().criteria);
```

### 一次执行一个功能

``` javascript
const once = fn => ((ran = false) => () => ran ? fn : (ran = !ran, fn = fn()))();

// Example
let n = 0;
const incOnce = once(() => ++n);
incOnce();      // n = 1
incOnce();      // n = 1
incOnce();      // n = 1
```

### 翻转函数的参数

``` javascript
// Reverse the order of arguments
const flip = fn => (...args) => fn(...args.reverse());

// For binary functions
const flip = fn => (b, a) => fn(a, b);

// Or for curried functions
const flip = fn => b => a => fn(a)(b);

// Example
const isParent = (parent, child) => parent.children.includes(child);
const isChild = flip(isParent);
```

### 身份功能

``` javascript
const identity = x => x;
```

### 逻辑异或运算符

``` javascript
// returns `true` if one of the arguments is truthy and the other is falsy

const xor = (a, b) => (a && !b) || (!a && b);

// Or
const xor = (a, b) => !(!a && !b) && !(a && b);

// Or
const xor = (a, b) => Boolean(!a ^ !b);

// Examples
xor(true, true);        // false
xor(false, false);      // false
xor(true, false);       // true
xor(false, true);       // true
```

### 记忆功能

``` javascript
const memoize = fn => ((cache = {}) => arg => cache[arg] || (cache[arg] = fn(arg)))();

// Example
// Calculate Fibonacci numbers
const fibo = memoize(n => n <= 2 ? 1 : fibo(n - 1) + fibo(n - 2));

fibo(1);    // 1
fibo(2);    // 1
fibo(3);    // 2
fibo(4);    // 3
fibo(5);    // 5
fibo(6);    // 8
```

### 部分应用功能

``` javascript
const partial = (fn, ...a) => (...b) => fn(...a, ...b);

// Example
const sum = (x, y) => x + y;
const inc = partial(sum, 1);
inc(9);     // 10
```

### 泛函功能

``` javascript
// `fn` is a curried function
// `n` is the depth of parameters
const uncurry = (fn, n = 1) => (...args) => (acc => args => args.reduce((x, y) => x(y), acc))(fn)(args.slice(0, n));

// Example
const sum = a => b => c => a + b + c;
uncurry(sum, 1)(1)(2)(3);       // 6
uncurry(sum, 2)(1, 2)(3);       // 6
uncurry(sum, 3)(1, 2, 3);       // 6
```

## MISC

### 检查代码是否在节点js中运行

``` javascript
const isNode = typeof process !== 'undefined' && process.versions != null && process.versions.node != null;
```

### 检查代码是否在浏览器中运行

``` javascript
const isBrowser = typeof window === 'object' && typeof document === 'object';
```

### 将摄氏度转换为华氏度

``` javascript
const celsiusToFahrenheit = celsius => celsius * 9/5 + 32;

const fahrenheitToCelsius = fahrenheit => (fahrenheit - 32) * 5/9;

// Examples
celsiusToFahrenheit(15);    // 59
celsiusToFahrenheit(0);     // 32
celsiusToFahrenheit(-20);   // -4

fahrenheitToCelsius(59);    // 15
fahrenheitToCelsius(32);    // 0
```

### 将 cookie 转换为对象

``` javascript
const cookies = document.cookie.split(';').map(item => item.split('=')).reduce((acc, [k, v]) => (acc[k.trim().replace('"', '')] = v) && acc, {});
```

### 将十六进制转换为 RGB

``` javascript
const hexToRgb = hex => hex.replace(/^#?([a-f\d])([a-f\d])([a-f\d])$/i, (_, r, g, b) => `#${r}${r}${g}${g}${b}${b}`).substring(1).match(/.{2}/g).map(x => parseInt(x, 16));

// Examples
hexToRgb('#00ffff');    // [0, 255, 255] 
hexToRgb('#0ff');       // [0, 255, 255]
```

### 将 RGB 颜色转换为十六进制

``` javascript
const rgbToHex = (red, green, blue) => `#${((1 << 24) + (red << 16) + (green << 8) + blue).toString(16).slice(1)}`;

// Or
const rgbToHex = (red, green, blue) => `#${[red, green, blue].map(v => v.toString(16).padStart(2, '0')).join('')}`;

// Example
rgbToHex(0, 255, 255);      // '#00ffff' 
```

### 解码 jwt 令牌

``` javascript
const decode = token => decodeURIComponent(atob(token.split('.')[1].replace('-', '+').replace('_', '/')).split('').map(c => `%${('00' + c.charCodeAt(0).toString(16)).slice(-2)}`).join(''));

// Example
decode(`
    eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.
    eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0I
    joxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
`);

// { "sub": "1234567890", "name": "John Doe", "iat": 1516239022 }
```

### 检测暗模式

``` javascript
const isDarkMode = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
```

### 缓动功能

``` javascript
// Some easing functions
// See https://gist.github.com/gre/1650294 and https://easings.net

const linear = t => t;

const easeInQuad = t => t * t;
const easeOutQuad = t => t * (2-t);
const easeInOutQuad = t => t < .5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

const easeInCubic = t => t * t * t;
const easeOutCubic = t => (--t) * t * t + 1;
const easeInOutCubic = t => t < .5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;

const easeInQuart = t => t * t * t * t;
const easeOutQuart = t => 1 - (--t) * t * t * t;
const easeInOutQuart = t => t < .5 ? 8 * t * t * t * t : 1 - 8 * (--t) * t * t * t;

const easeInQuint = t => t * t * t * t * t;
const easeOutQuint = t => 1 + (--t) * t * t * t * t;
const easeInOutQuint = t => t < .5 ? 16 * t * t * t * t * t : 1 + 16 * (--t) * t * t * t * t;

const easeInSine = t => 1 + Math.sin(Math.PI / 2 * t - Math.PI / 2);
const easeOutSine = t => Math.sin(Math.PI / 2 * t);
const easeInOutSine = t => (1 + Math.sin(Math.PI * t - Math.PI / 2)) / 2;

const easeInElastic = t => (.04 - .04 / t) * Math.sin(25 * t) + 1;
const easeOutElastic = t => .04 * t / (--t) * Math.sin(25 * t);
const easeInOutElastic = t => (t -= .5) < 0 ? (.02 + .01 / t) * Math.sin(50 * t) : (.02 - .01 / t) * Math.sin(50 * t) + 1;
```

### 模拟掷骰子

``` javascript
const throwdice = () => ~~(Math.random() * 6) + 1;

// Examples
throwdice();    // 4 
throwdice();    // 1 
throwdice();    // 6 
```

### 编码网址

``` javascript
// `encodeURIComponent` doesn't encode -_.!~*'()
const encode = url => encodeURIComponent(url).replace(/!/g, '%21').replace(/~/g, '%7E').replace(/\*/g, '%2A').replace(/'/g, '%27').replace(/\(/g, '%28').replace(/\)/g, '%29').replace(/%20/g, '+');
```

### 获取第一个已定义且非 null 的参数

``` javascript
const coalesce = (...args) => args.find(item => item !== undefined && item !== null);

// Or
const coalesce = (...args) => args.find(item => ![undefined, null].includes(item));

// Examples
coalesce(undefined, null, 'helloworld', NaN);     // 'helloworld'
```

### 获取 Cookie 的值

``` javascript
const cookie = name => `; ${document.cookie}`.split(`; ${name}=`).pop().split(';').shift();

// Example
cookie('_ga');      // GA1.2.825309271.1581874719
```

### 从网址获取参数值

``` javascript
const getParam = (url, param) => new URLSearchParams(new URL(url).search).get(param);

// Example
getParam('http://domain.com?message=hello', 'message');     // 'hello'
```

### 将页面重定向到https（如果位于http中）

``` javascript
const redirectHttps = () => (location.protocol === 'https:') ? {} : location.replace(`https://${location.href.split('//')[1]}`);

// Or
const redirectHttps = () => (location.protocol === 'https:') ? {} : (location.protocol = 'https:');
```

### 依次兑现承诺

``` javascript
// `promises` is an array of `Promise`
const run = promises => promises.reduce((p, c) => p.then(rp => c.then(rc => [...rp, rc])), Promise.resolve([]));

// Example
run(promises).then((results) => {
    // `results` is an array of promise results in the same order
});
```

### 交换两个变量

``` javascript
[a, b] = [b, a];

// Or
a = [b, b = a][0];

// Or
a = (x => x)(b, b = a);

// Or
// (only works with numbers)
a = b + (b = a, 0);

a = b * (b = a, 1);
```

### 等待一段时间

``` javascript
const wait = async (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
```

## 数字

### 在数字后添加序数后缀

``` javascript
// `n` is a position number
const addOrdinal = n => `${n}${['st', 'nd', 'rd'][((n + 90) % 100 - 10) % 10 - 1] || 'th'}`;

// Or
const addOrdinal = n => `${n}${[, 'st', 'nd', 'rd'][/1?.$/.exec(n)] || 'th'}`;

// Or
const addOrdinal = n => `${n}${[, 'st', 'nd', 'rd'][n % 100 >> 3^1 && n % 10] || 'th'}`;

// Or
const addOrdinal = n => `${n}${{one: 'st', two: 'nd', few: 'rd', other: 'th'}[new Intl.PluralRules('en', { type: 'ordinal' }).select(n)]}`;

// Examples
addOrdinal(1);      // '1st'
addOrdinal(2);      // '2nd'
addOrdinal(3);      // '3rd'
addOrdinal(11);     // '11th'
addOrdinal(12);     // '13th'
addOrdinal(13);     // '13th'
```

### 计算斐波那契数

``` javascript
const fibo = (n, memo = {}) => memo[n] || (n <= 2 ? 1 : (memo[n] = fibo(n - 1, memo) + fibo(n - 2, memo)));

// Examples
fibo(1);    // 1
fibo(2);    // 1
fibo(3);    // 2
fibo(4);    // 3
fibo(5);    // 5
fibo(6);    // 8
```

### 计算参数的平均值

``` javascript
const average = (...args) => args.reduce((a, b) => a + b) / args.length;

// Example
average(1, 2, 3, 4);    // 2.5
```

### 技术参数除法

``` javascript
const division = (...args) => args.reduce((a, b) => a / b);

// Example
division(1, 2, 3, 4);   // 0.04166666666666666
```

### 计算数字的阶乘

``` javascript
const factorial = n => n <= 1 ? 1 : n * factorial(n - 1);

// Examples
factorial(2);   // 2
factorial(3);   // 6
factorial(4);   // 24
factorial(5);   // 120
factorial(6);   // 720
```

### 计算集合索引的 mod

``` javascript
const mod = (a, b) => ((a % b) + b) % b;

// Examples
mod(-1, 5);     // 4
mod(3, 5);      // 3
mod(6, 5);      // 1
```

### 计算参数除法的余数

``` javascript
const remainder = (...args) => args.reduce((a, b) => a % b);

// Example
remainder(1, 2, 3, 4);      // 1
```

### 计算参数之和

``` javascript
const sum = (...args) => args.reduce((a, b) => a + b);

// Example
sum(1, 2, 3, 4);    // 10
```

### 将数字限制在两个值之间

``` javascript
const clamp = (val, min = 0, max = 1) => Math.max(min, Math.min(max, val));

// Example
clamp(199, 10, 25);     // 25
```

### 计算两个数字之间的最大公约数

``` javascript
const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);

// Example
gcd(10, 15);    // 5
```

### 将数字转换为等效字符

``` javascript
const toChars = n => `${n >= 26 ? toChars(Math.floor(n / 26) - 1) : ''}${'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[n % 26]}`;

// Examples
toChars(0);     // A
toChars(1);     // B
toChars(25);    // Z

toChars(26);     // AA
toChars(27);     // AB
toChars(51);     // AZ

toChars(701);   // ZZ
toChars(702);   // AAA
toChars(703);   // AAB
```

### 将字符串转换为数字

``` javascript
const toNumber = str => +str;

// Example
toNumber('42');     // 42
```

### 将度转换为弧度

``` javascript
const degsToRads = deg => (deg * Math.PI) / 180.0;

const radsToDegs = rad => rad * 180 / Math.PI;
```

### 从数字中获取数字数组

``` javascript
const digits = n => `${n}`.split('').map(v => parseInt(v));

// Example
digits(123);    // [1, 2, 3]
```

### 相乘参数

``` javascript
const mul = (...args) => args.reduce((a, b) => a * b);

// Example
mul(1, 2, 3, 4);    // 24
```

### 在整数前加上零

``` javascript
const prefixWithZeros = (number, length) => (number / Math.pow(10, length)).toFixed(length).substr(2);

// Or
const prefixWithZeros = (number, length) => `${Array(length).join('0')}${number}`.slice(-length);

// Or
const prefixWithZeros = (number, length) => String(number).padStart(length, '0');

// Example
prefixWithZeros(42, 5);     // '00042'
```

### 将数字四舍五入到给定的数字

``` javascript
const round = (n, decimals = 0) => Number(`${Math.round(`${n}e${decimals}`)}e-${decimals}`);

// Examples
round(1.234567, 3);     // 1.235
round(1.234567, 4);     // 1.2346
```

### 减去参数

``` javascript
const subtract = (...args) => args.reduce((a, b) => a - b);

// Example
subtract(1, 2, 3, 4);       // -8
```

### 截断十进制数字

``` javascript
const truncate = n => ~~n;

// Examples
truncate(25.198726354);         // 25
truncate(-25.198726354);        // -25
```

### 将数字截断为给定的小数位数而不四舍五入

``` javascript
const toFixed = (n, fixed) => `${n}`.match(new RegExp(`^-?\\d+(?:\.\\d{0,${fixed}})?`))[0];

// Or
const toFixed = (n, fixed) => ~~(Math.pow(10, fixed) * n) / Math.pow(10, fixed);

// Examples
toFixed(25.198726354, 1);       // 25.1
toFixed(25.198726354, 2);       // 25.19
toFixed(25.198726354, 3);       // 25.198
toFixed(25.198726354, 4);       // 25.1987
toFixed(25.198726354, 5);       // 25.19872
toFixed(25.198726354, 6);       // 25.198726
```

## 目的

### 检查多个对象是否相等

``` javascript
const isEqual = (...objects) => objects.every(obj => JSON.stringify(obj) === JSON.stringify(objects[0]));

// Examples
isEqual({ foo: 'bar' }, { foo: 'bar' });    // true
isEqual({ foo: 'bar' }, { bar: 'foo' });    // false
```

### 创建一个没有属性的空地图

``` javascript
// `map` doesn't have any properties
const map = Object.create(null);

// The following `map` has `__proto__` property
// const map = {};
```

### 从键和值对创建对象

``` javascript
const toObj = arr => Object.fromEntries(arr)

// Or
const toObj = arr => arr.reduce((a, c) => ((a[c[0]] = c[1]), a), {});

// Example
toObj([['a', 1], ['b', 2], ['c', 3]]);      // { a: 1, b: 2, c: 3 }
```

### 从对象数组中提取属性的值

``` javascript
const pluck = (objs, property) => objs.map(obj => obj[property]);

// Example
pluck([
    { name: 'John', age: 20 },
    { name: 'Smith', age: 25 },
    { name: 'Peter', age: 30 },
], 'name');         // ['John', 'Smith', 'Peter']
```

### 获取对象给定路径的值

``` javascript
const getValue = (path, obj) => path.split('.').reduce((acc, c) => acc && acc[c], obj);

// Example
getValue('a.b', { a: { b: 'Hello World' } });   // 'Hello World';
```

### 反转对象的键和值

``` javascript
const invert = obj => Object.keys(obj).reduce((res, k) => Object.assign(res, {[obj[k]]: k}), {});

// Example
invert({ a: '1', b: '2', c: '3' });     // { 1: 'a', 2: 'b', 3: 'c' }
```

### 忽略对象的属性子集

``` javascript
const omit = (obj, keys) => Object.keys(obj).filter(k => !keys.includes(k)).reduce((res, k) => Object.assign(res, {[k]: obj[k]}), {});

// Example
omit({a: '1', b: '2', c: '3'}, ['a', 'b']);     // { c: '3' }
```

### 选择一个对象的属性子集

``` javascript
const pick = (obj, keys) => Object.keys(obj).filter(k => keys.includes(k)).reduce((res, k) => Object.assign(res, {[k]: obj[k]}), {});

// Example
pick({ a: '1', b: '2', c: '3' }, ['a', 'b']);   // { a: '1', b: '2' }
```

### 浅复制对象

``` javascript
const shallowCopy = obj => Object.assign({}, obj);

// or
const shallowCopy = obj => {...obj};
```

## 随机

### 产生一个随机的布尔值

``` javascript
const randomBoolean = () => Math.random() >= 0.5;
```

### 生成给定范围内的随机浮点数

``` javascript
const randomFloat = (min, max) => Math.random() * (max - min) + min;
```

### 生成随机的十六进制颜色

``` javascript
const randomColor = () => `#${Math.random().toString(16).slice(2, 8).padEnd(6, '0')}`;

// Or
const randomColor = () => `#${(~~(Math.random()*(1<<24))).toString(16)}`;
```

### 生成给定范围内的随机整数

``` javascript
const randomInteger = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;
```

### 生成一个随机IP地址

``` javascript
const randomIp = () => Array(4).fill(0).map((_, i) => Math.floor(Math.random() * 255) + (i === 0 ? 1 : 0)).join('.');

// Example
randomIp();     // 175.89.174.131
```

### 从给定字符生成随机字符串

``` javascript
const generateString = (length, chars) => Array(length).fill('').map((v) => chars[Math.floor(Math.random() * chars.length)]).join('');

// Example
generateString(10, '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ');
```

### 使用节点加密模块生成随机字符串

``` javascript
const randomStr = () => require('crypto').randomBytes(32).toString('hex');
```

### 生成给定长度的随机字符串

``` javascript
const generateString = length => Array(length).fill('').map((v) => Math.random().toString(36).charAt(2)).join('');
```

### 生成随机的uuid

``` javascript
const uuid = (a) => a ? (a ^ Math.random() * 16 >> a / 4).toString(16) : ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, uuid);
```

### 生成给定范围内的随机整数数组

``` javascript
const randomArrayInRange = (min, max, n) => Array.from({ length: n }, () => Math.floor(Math.random() * (max - min + 1)) + min);

// Example
randomArrayInRange(1, 100, 10);     // [11, 82, 41, 35, 76, 83, 43, 15, 60, 54]
```

### 从数组中获取随机项

``` javascript
const randomItem = arr => arr[(Math.random() * arr.length) | 0];
```

## 字符串

### 截取最后一个反斜杠后的内容

``` javascript
const interceptAfter = (str) => str.substring(str.lastIndexOf('/') + 1, str.length)
```

### 截取最后一个反斜杠前的内容

``` javascript
const interceptBefore = (str) => str.substring(0, str.lastIndexOf('/') + 1)
```

### 大写字符串

``` javascript
const capitalize = str => `${str.charAt(0).toUpperCase()}${str.slice(1)}`;

// Example
capitalize('hello world');      // 'Hello world'
```

### 检查路径是否相对

``` javascript
const isRelative = path => !/^([a-z]+:)?[\\/]/i.test(path);

// Examples
isRelative('/foo/bar/baz');         // false
isRelative('C:\\foo\\bar\\baz');    // false
isRelative('foo/bar/baz.txt');      // true
isRelative('foo.md');               // true
```

### 检查字符串是否是回文

``` javascript
const isPalindrome = str => str === str.split('').reverse().join('');

// Examples
isPalindrome('abc');        // false
isPalindrom('abcba');       // true
```

### 检查网址是否为绝对网址

``` javascript
const isAbsoluteUrl = url => /^[a-z][a-z0-9+.-]*:/.test(url);

// Example
isAbsoluteUrl('https://1loc.dev');          // true
isAbsoluteUrl('https://1loc.dev/foo/bar');  // true
isAbsoluteUrl('1loc.dev');                  // false
isAbsoluteUrl('//1loc.dev');                // false
```

### 检查两个字符串是否为字谜

``` javascript
const areAnagram = (str1, str2) => str1.toLowerCase().split('').sort().join('') === str2.toLowerCase().split('').sort().join('');

// Examples
areAnagram('listen', 'silent');         // true
areAnagram('they see', 'the eyes');     // true
areAnagram('node', 'deno');             // true
```

### 转换字母以关联表情符号

``` javascript
const letterToEmoji = c => String.fromCodePoint(c.toLowerCase().charCodeAt() + 127365);

// Examples
letterToEmoji('a');     // 🇦
letterToEmoji('b');     // 🇧
```

### 将字符串转换为驼峰式大小写

``` javascript
const toCamelCase = str => str.trim().replace(/[-_\s]+(.)?/g, (_, c) => c ? c.toUpperCase() : '');

// Examples
toCamelCase('background-color');            // backgroundColor
toCamelCase('-webkit-scrollbar-thumb');     // WebkitScrollbarThumb
toCamelCase('_hello_world');                // HelloWorld
toCamelCase('hello_world');                 // helloWorld
```

### 将字符串转换为 Pascal 大小写

``` javascript
const toPascalCase = str => (str.match(/[a-zA-Z0-9]+/g) || []).map(w => `${w.charAt(0).toUpperCase()}${w.slice(1)}`).join('');

// Examples
toPascalCase('hello world');    // 'HelloWorld'
toPascalCase('hello.world');    // 'HelloWorld'
toPascalCase('foo_bar-baz');    // FooBarBaz
```

### 将字符串转换为 url slug

``` javascript
const slugify = string => string.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]+/g, '');

// Example
slugify('Chapter One: Once upon a time...');    // 'chapter-one-once-upon-a-time'
```

### 将 Windows 文件路径转换为 Unix 路径

``` javascript
const toUnixPath = path => path.replace(/[\\/]+/g, '/').replace(/^([a-zA-Z]+:|\.\/)/, '');

// Examples
toUnixPath('./foo/bar/baz');        // foo/bar/baz
toUnixPath('C:\\foo\\bar\\baz');    // /foo/bar/baz
```

### 将驼峰命名法转换为 kebab-case 形式, 反之亦然

``` javascript
const kebabToCamel = str => str.replace(/-./g, m => m.toUpperCase()[1]);

const camelToKebab = str => str.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase();

// Examples
kebabToCamel('background-color');   // 'backgroundColor'
camelToKebab('backgroundColor');    // 'background-color'
```

### 将 snake-case 形式转换成驼峰命名形式

``` javascript
const snakeToCamel = str => str.toLowerCase().replace(/(_\w)/g, m => m.toUpperCase().substr(1));

// Example
snakeToCamel('HELLO_world');    // 'helloWorld'
```

### 将 Excel 列名转换成数字

``` javascript
const getIndex = col => col.split('').reduce((prev, next) => prev * 26 + parseInt(next, 36) - 9, 0);

// Examples
getIndex('A');      // 1
getIndex('B');      // 2
getIndex('C');      // 3
getIndex('Z');      // 26

getIndex('AA');     // 27
getIndex('AB');     // 28
getIndex('AC');     // 29
getIndex('AZ');     // 52

getIndex('AAA');    // 703
getIndex('AAB');    // 704
```

### 转义 HTML 特殊字符

``` javascript
const escape = str => str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/'/g, '&#39;').replace(/"/g, '&quot;');

// Or
const escape = str => str.replace(/[&<>"']/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[m]);
```

### 获取不带任何参数的基本 URL

``` javascript
const baseUrl = url => url.indexOf('?') === -1 ? url : url.slice(0, url.indexOf('?'));

// Or
// Note that `includes` isn't supported in IE 11
const baseUrl = url => url.includes('?') ? url.slice(0, url.indexOf('?')) : url;

// Example
baseUrl('https://domain.com/path/sub/path?foo=bar&hello=world');    // 'https://domain.com/path/sub/path'
```

### 从文件名获取文件扩展名

``` javascript
const ext = fileName => fileName.split('.').pop();
```

### 从网址获取文件名

``` javascript
const fileName = url => url.substring(url.lastIndexOf('/') + 1);

// Example
fileName('http://domain.com/path/to/document.pdf');     // 'document.pdf'
```

### 获取字符串的长度（以字节为单位）

``` javascript
const bytes = str => new Blob([str]).size;

// Examples
bytes('hello world');       // 11
bytes('🎉');                // 4
```

### 获取字符串中的字符数

``` javascript
const characterCount = (str, char) => str.split(char).length - 1;

// Or
const characterCount = (str, char) => str.replace((new RegExp(String.raw`[^${char}]`, 'g')), '').length;

// Examples
characterCount('192.168.1.1', '.');     // 3
characterCount('star wars', 's');       // 2
```

### 使字符串的第一个字符小写

``` javascript
const lowercaseFirst = str => `${str.charAt(0).toLowerCase()}${str.slice(1)}`;

// Example
lowercaseFirst('Hello World');      // 'hello World'
```

### 标准化文件路径斜杠

``` javascript
const normalizePath = path => path.replace(/[\\/]+/g, '/');

// Example
normalizePath('\\foo\\bar\\baz\\');         // /foo/bar/baz/
normalizePath('.//foo//bar///////baz/');    // ./foo/bar/baz/
```

### 从字符串中删除空格

``` javascript
const removeSpaces = str => str.replace(/\s/g, '');

// Example
removeSpaces('hel lo wor ld');      // 'helloworld'
```

### 重复一个字符串

``` javascript
const repeat = (str, numberOfTimes) => str.repeat(numberOfTimes);

// Or
const repeat = (str, numberOfTimes) => Array(numberOfTimes + 1).join(str);
```

### 用 br 元素替换所有换行符

``` javascript
const nl2br = str => str.replace(new RegExp('\r?\n', 'g'), '<br>');

// In React
str.split('\n').map((item, index) => <React.Fragment key={index}>{item}<br /></React.Fragment>)
```

### 用一个空格替换多个空格

``` javascript
// Replace spaces, tabs and new line characters
const replaceSpaces = str => str.replace(/\s\s+/g, ' ');

// Only replace spaces
const replaceOnlySpaces = str => str.replace(/  +/g, ' ');

// Example
replaceSpaces('this\n   is     \ta    \rmessage');  // 'this is a message'
```

### 将字符串的第一个给定字符数替换为另一个字符

``` javascript
const mask = (str, num, mask) => `${str}`.slice(num).padStart(`${str}`.length, mask);

// Example
mask(1234567890, 3, '*');       // ***4567890
```

### 反转字符串

``` javascript
const reverse = str => str.split('').reverse().join('');

// Or
const reverse = str => [...str].reverse().join('');

// Or
const reverse = str => str.split('').reduce((rev, char)=> `${char}${rev}`, '');

// Or
const reverse = str => (str === '') ? '' : `${reverse(str.substr(1))}${str.charAt(0)}`;

// Example
reverse('hello world');     // 'dlrow olleh'
```

### 按字母顺序对字符串的字符进行排序

``` javascript
const sort = str => str.split('').sort((a, b) => a.localeCompare(b)).join('');

// Example
sort('hello world');    // dehllloorw
```

### 从字符串中删除 ANSI 代码

``` javascript
const stripAnsiCodes = str => str.replace(/[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g, '');

// Example
stripAnsiCodes('\u001B[4mcake\u001B[0m');                                                               // 'cake'
stripAnsiCodes('\u001B[0m\u001B[4m\u001B[42m\u001B[31mfoo\u001B[39m\u001B[49m\u001B[24mfoo\u001B[0m');  // 'foofoo'
```

### 在字符串的开头和结尾处使用斜杠

``` javascript
const trimSlashes = str => str.replace(/^\/+|\/+$/g, '');

// Or
const trimSlashes = str => str.split('/').filter(Boolean).join('/');

// Example
trimSlashes('//hello/world///');    // hello/world
```

### 修剪一些角色

``` javascript
const trim = (str, char) => str.split(char).filter(Boolean).join();

// Examples
trim('/hello world//', '/');        // hello world
trim('"hello world"', '"');         // hello world
trim('   hello world ', ' ');       // hello world
```

### 从文件名修剪文件扩展名

``` javascript
const trimExt = fileName => fileName.indexOf('.') === -1 ? fileName : fileName.split('.').slice(0, -1).join('.');

// Examples
trimExt('document');            // document
trimExt('document.pdf');        // document
trimExt('document.2020.pdf');   // document.2020
```

### 用完整的单词截断字符串

``` javascript
const truncate = (str, max, suffix) => str.length < max ? str : `${str.substr(0, str.substr(0, max - suffix.length).lastIndexOf(' '))}${suffix}`;

// Example
truncate('This is a long message', 20, '...');  // 'This is a long...'
```

### Unescape html 特殊字符

``` javascript
const unescape = str => str.replace(/&/g , '&').replace(/&lt;/g  , '<').replace(/&gt;/g  , '>').replace(/&#0*39;/g , "'").replace(/&quot;/g, '"');
```

### 将字符串中每个单词的第一个字符大写

``` javascript
const uppercaseWords = str => str.split(' ').map(w => `${w.charAt(0).toUpperCase()}${w.slice(1)}`).join(' ');

// Or
const uppercaseWords = str => str.replace(/^(.)|\s+(.)/g, c => c.toUpperCase());

// Example
uppercaseWords('hello world');      // 'Hello World'
```

## 验证器

### 检查日期是否在两个日期之间

``` javascript
// `min`, `max` and `date` are `Date` instances
const isBetween = (date, min, max) => (date.getTime() >= min.getTime() && date.getTime() <= max.getTime());
```

### 检查日期是否是今天

``` javascript
// `date` is a Date object
const isToday = (date) => date.toISOString().slice(0, 10) === new Date().toISOString().slice(0, 10);
```

### 检查给定的整数是否是质数

``` javascript
const isPrime = num => (num > 1) && Array(Math.floor(Math.sqrt(num)) - 1).fill(0).map((_, i) => i + 2).every(i => num % i !== 0);
```

### 检查数字是否为 2 的幂

``` javascript
const isPowerOfTwo = number => (number & (number - 1)) === 0;

// Examples
isPowerOfTwo(256);      // true
isPowerOfTwo(129);      // false
```

### 检查数字是否为偶数

``` javascript
const isEven = number => number % 2 === 0;

// Or
const isEven = number => (number & 1) === 0;

// Or
const isEven = number => !(number & 1);

// Or
const isEven = number => Number.isInteger(number / 2);

// Examples
isEven(1);      // false
isEven(2);      // true
```

### 检查数字是否在给定范围内

``` javascript
const inRange = (num, a, b) => (Math.min(a, b) <= num && num < Math.max(a, b));

// Example
inRange(10, 5, 15);         // true
inRange(10, 5, 6);          // false
inRange(10, 15, 5);         // true
inRange(-10, -5, -15);      // true
```

### 检查数字是否为负

``` javascript
const isNegative = number => Math.sign(number) === -1;

// Or
const isNegative = number => number < 0;

// Examples
isNegative(-3);     // true
isNegative(8);      // false
```

### 检查数字是否为奇数

``` javascript
const isOdd = number => number % 2 !== 0;

// Or
const isOdd = number => !!(number & 1);

// Or
const isOdd = number => !Number.isInteger(number / 2);

// Examples
isOdd(1);   // true
isOdd(2);   // false
```

### 检查数字是否为正

``` javascript
const isPositive = number => Math.sign(number) === 1;

// Examples
isPositive(3);      // true
isPositive(-8);     // false
```

### 检查字符串是否包含小写字符

``` javascript
const containsLowerCase = str => str !== str.toUpperCase();

// Examples
containsLowerCase('Hello World');   // true
containsLowerCase('HELLO WORLD');   // false
```

### 检查字符串是否仅包含 ASCII 字符

``` javascript
const isAscii = str => /^[\x00-\x7F]+$/.test(str);
```

### 检查字符串是否仅包含数字

``` javascript
const isNumeric = str => !/[^0-9]/.test(str);

// Examples
isNumeric(2);               // true
isNumeric('23');            // true
isNumeric('00123');         // true

isNumeric('1.23');          // false
isNumeric('-Infinity');     // false
isNumeric('Infinity');      // false
isNumeric('NaN');           // false
```

### 检查字符串是否仅包含字母和数字

``` javascript
const isAlphanumeric = str => /^[0-9A-Z]+$/i.test(str);

// Examples
isAlphanumeric('helloworld');           // true
isAlphanumeric('HelloWorld');           // true
isAlphanumeric('hello world');          // false
isAlphanumeric('hello123');             // true
isAlphanumeric('hello 123');            // false
```

### 检查字符串是否仅包含字母

``` javascript
const isAlpha = str => /^[A-Z]+$/i.test(str);

// Examples
isAlpha('helloworld');          // true
isAlpha('HelloWorld');          // true
isAlpha('hello world');         // false
isAlpha('0123456789');          // false
```

### 检查字符串是否包含大写字符

``` javascript
const containsUpperCase = str => str !== str.toLowerCase();

// Examples
containsUpperCase('Hello World');   // true
containsUpperCase('hello world');   // false
```

### 检查字符串是否包含空格

``` javascript
const containsWhitespace = str => str => /\s/.test(str);

// Example
containsWhitespace('hello world');      // true
```

### 检查字符串是否为十六进制颜色

``` javascript
const isHexColor = color => /^#([0-9A-F]{3}|[0-9A-F]{4}|[0-9A-F]{6}|[0-9A-F]{8})$/i.test(color);

// Examples
isHexColor('#012');         // true
isHexColor('#A1B2C3');      // true
isHexColor('012');          // false
isHexColor('#GHIJKL');      // false
```

### 检查字符串是否为十六进制数字

``` javascript
const isHexadecimal = str => /^[A-F0-9]+$/i.test(str);

// Or
const isHexadecimal = str => str.split('').every(c => '0123456789ABCDEFabcdef'.indexOf(c) !== -1);

// Examples
isHexadecimal('123');       // true
isHexadecimal('A1B2C3');    // true
isHexadecimal('#123');      // false
```

### 检查字符串是否为小写

``` javascript
const isLowerCase = str => str === str.toLowerCase();
```

### 检查字符串是否为大写

``` javascript
const isUpperCase = str => str === str.toUpperCase();
```

### 检查值是否为数字

``` javascript
const isNumber = value => !isNaN(parseFloat(value)) && isFinite(value);
```

### 检查值是否为普通对象

``` javascript
const isPlainObject = v => (!!v && typeof v === 'object' && (v.__proto__ === null || v.__proto__ === Object.prototype));

// Examples
isPlainObject(null);                    // false
isPlainObject('hello world');           // false
isPlainObject([]);                      // false
isPlainObject(Object.create(null));     // false
isPlainObject(function() {});           // false

isPlainObject({});                      // true
isPlainObject({ a: '1', b: '2' });      // true
```

### 检查值是否为正则表达式

``` javascript
const isRegExp = value => Object.prototype.toString.call(value) === '[object RegExp]';
```

### 检查值是否为字符串

``` javascript
const isString = value => Object.prototype.toString.call(value) === '[object String]';

// Examples
isString('hello world');                // true
isString(new String('hello world'));    // true
isString(10);                           // false
```

### 检查值是否为对象

``` javascript
const isObject = v => (v !== null && typeof v === 'object');

// Examples
isObject(null);             // false
isObject('hello world');    // false

isObject({});               // true
isObject([]);               // true 
```

### 检查值是否为零

``` javascript
const isNil = (value) => value == null;
```

### 检查一年是否为 leap 年

``` javascript
const isLeapYear = year => (((year % 4 === 0) && (year % 100 !== 0)) || (year % 400 === 0));

// Or
// Get the number of days in February
const isLeapYear = year => new Date(year, 1, 29).getDate() === 29;
```

### 检查所有数组元素是否等于给定值

``` javascript
const isEqual = (arr, value) => arr.every(item => item === value);

// Or
// Ends earlier for false arrays
const isEqual = (arr, value) => !arr.some(item => item !== value);

// Examples
isEqual(['foo', 'foo'], 'foo');     // true
isEqual(['foo', 'bar'], 'foo');     // false
isEqual(['bar', 'bar'], 'foo');     // false
```

### 检查数组中的所有项目是否相等

``` javascript
const areEqual = arr => arr.length > 0 && arr.every(item => item === arr[0]);

// Or
const areEqual = arr => new Set(arr).size === 1;

// Examples
areEqual([1, 2, 3, 4]);                 // false
areEqual(['hello', 'hello', 'hello']);  // true
```

### 检查数组是否包含与某些条件匹配的值

``` javascript
const contains = (arr, criteria) => arr.some(v => criteria(v));

// Or
const contains = (arr, criteria) => arr.some(criteria);

// Or
const contains = (arr,criteria) => arr.filter(criteria).length > 0;

// Examples
contains([10, 20, 30], v => v > 25 );               // true
contains([10, 20, 30], v => v > 100 || v < 15 );    // true
contains([10, 20, 30], v => v > 100 );              // false
```

### 检查数组是否为空

``` javascript
const isNotEmpty = arr => Array.isArray(arr) && Object.keys(arr).length > 0;

// Examples
isNotEmpty([]);             // false
isNotEmpty([1, 2, 3]);      // true
```

### 检查一个数组是否是其他数组的子集

``` javascript
// Check if `b` is subset of `a`
const isSubset = (a, b) => (new Set(b)).size === (new Set(b.concat(a))).size;

// Or
const isSubset = (a, b) => b.join("|").includes(a.join("|"));

// Examples
isSubset([1, 2], [1, 2, 3, 4]);     // true
isSubset([1, 2, 5], [1, 2, 3, 4]);  // false
isSubset([6], [1, 2, 3, 4]);        // false
```

### 检查一个对象是否是一个 promise 对象

``` javascript
const isPromise = obj => !!obj && (typeof obj === 'object' || typeof obj === 'function') && typeof obj.then === 'function';
```

### 检查对象是否为数组

``` javascript
const isArray = obj => Array.isArray(obj);
```

### 检查对象是否为空

``` javascript
const isEmpty = obj => Reflect.ownKeys(obj).length === 0 && obj.constructor === Object;

// Or for enumerable property names only
const isEmpty = obj => JSON.stringify(obj) === '{}';
```

### 验证公历

``` javascript
// `m`: the month (zero-based index)
// `d`: the day
// `y`: the year
const isValidDate = (m, d, y) => 0 <= m && m <= 11 && 0 < y && y < 32768 && 0 < d && d <= (new Date(y, m, 0)).getDate();
```
