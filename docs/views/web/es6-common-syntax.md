---
author: 星火燎原@vxhly
title: ES6 常用语法推荐
categories: [web]
tags: [Node.js, ES6, JavaScript]
date: 2020-11-02 17:13:15
---

<Boxx changeTime="10000"/>

::: tip 前言
收集在日常开放过程中常用的 ES6 语法.
:::
<!-- more -->

## 变量赋值

### let、const

``` javascript
let a = 123    // 使用 let 定义变量
const b = 456  // 使用 const 定义常量
const c = []   // 使用 const 定义常量

a = 789 // 变量可二次被修改
b = 98 // 常量不可被修改
c.push(1) // 常量为数组的, 可被增删元素
```

### 解构赋值

``` javascript
// 数组
let [x, y] = [1, 2]
// => x = 1, y = 2

// 对象
let { x } = { x: '123', y: '456' }
// => x = 123
```

### 变量重命名

``` javascript
let { foo: baz } = { foo: 'aaa', bar: 'bbb' };
// baz => 'aaa'

// 当无默认值时, 为重命名的变量赋默认值
let obj = {}
let { foo: baz = 'aaa' } = obj
```

## 字符串

### 模板字符串

``` javascript
let a = 'hello'
let b = 'world'
let c = `${a} ${b}`

// c=> hello world
```

### 标签模板

``` javascript
let url = oneLine `
    www.taobao.com/example/index.html
    ?foo=${foo}
    &bar=${bar}
`

// => www.taobao.com/example/index.html?foo=foo&bar=bar
```

### 字符串分割：split()

``` javascript
// 字符串拆分成数组
"1,2,3".split(',')

// => [1,2,3]
```

## 函数

### 箭头函数

``` javascript
let fun = () => {
  // ... ...
}
```

### 函数默认值

``` javascript
let fun = (quantity = 1) => {
  // ... ...
}

let fun = ({a = 1, b = 2} = {}) => {
  // ... ...
}
```

### Rest 参数

``` javascript
// 当一个函数的最后一个参数有...这样的前缀, 它就会变成一个参数的数组。
function test(...args) {
  console.log(args);
}
test(1, 2, 3);
// [1,2,3]

function test2(name, ...args) {
  console.log(args);
}
test2('liangyin', 2, 3);
// [2,3]
```

### async / await

``` javascript
// 使用 async 关键字使函数变成一个异步函数, await 关键字与异步函数一起使用时
/** 注意事项：
		1. await命令后面的Promise对象, 运行结果可能是rejected, 所以最好把await命令放在try...catch代码块中
    2. 多个await命令后面的异步操作, 如果不存在继发关系, 最好让它们同时触发
    3. await命令只能用在async函数之中, 如果用在普通函数, 就会报错
  */ 

async function myFetch() {
  let response = await fetch('coffee.jpg');
  return await response.blob();
}
```

## 数组

### 返回一个新数组：map()

``` javascript
// 数组的遍历, 生成一个新的数组
[1,2,3].map(item => item += 1)

// => [2,3,4]
```

### 数组过滤：filter()

``` javascript
// 数组的过滤,  生成一个新的数组
[1, 2, 3, false, null, ''].filter(item => item)

// => [1,2,3]
```

### 数组转字符串：join()

``` javascript
// 连接所有数组元素组成一个字符串
[1, 2, 3],join(',')

// => 1,2,3
```

### 数组去重：Set()

``` javascript
// 数组的去重, 生成一个新的数组
let arr = [1, 1, 21, 23, 1, 5, 6]
arr = [...new Set(arr)]

// arr => [1, 21, 23, 5, 6]
```

### 数组是否包含某指定的值：includes()

``` javascript
// 判断当前数组是否包含某指定的值, 如果是返回 true, 否则返回 false。
let arr = ['a', 2, false]
arr.includes('a')

// => true
```

### 数组合并

``` javascript
let a = [1, 2, 3]
let b = [4, 5, 6]
let c = [[1, 2, 3], [4, 5, 6], 1, 23]
let d = [...a, ...b, ...c]

// d => [1, 2, 3, 4, 5, 6, [1, 2, 3], [4, 5, 6], 1, 23]
```

### 数组遍历：for...in

``` javascript
let arr = [1, 2, 3]

for(let key in arr) {
	console.log(arr[key])
}

// 1
// 2
// 3
```

## 对象

### 对象的 key：keys()

``` javascript
let obj = { a: '1', b: '2', c: '3' }
let keys = Object.keys(obj)

/// keys => ['a', 'b', 'c']
```

### 对象的 value：values()

``` javascript
let obj = { a: '1', b: '2', c: '3' }
let values = Object.values(obj)

/// keys => ['1', '2', '3']
```

### 对象合并

``` javascript
let obj1 = { a: '1', b: '2', c: '3' }
let obj2 = { a: '5', d: '21', e: '31' }
let obj3 = {...obj1, ...obj2}
let obj4 = Object.assign({}, obj1, obj2)

// obj3 => {a: '5', b: '2', c: '3', d: '21', e: '31'}
// obj4 => {a: '5', b: '2', c: '3', d: '21', e: '31'}
```

### 对象遍历：for...in

``` javascript
let obj = { a: '1', b: '2', c: '3' }

for (let i in obj) {
  console.log(i,":",obj[i])
}

// a : 1
// b : 2
// c : 3
```

## Promise()

### 基本用法

``` javascript
new Promise((resolve, reject) => {
  setTimeout(() => {
    resolve('foo');
  }, 300);
})
  .then(res= > {
    console.log(res
  })
  .catch(error => console.error(error))
	.finally(() => console.log('finished'));
```

### Promise.all()

::: tip 提示
Promise.all 方法用于将多个 Promise 实例, 包装成一个新的 Promise 实例。只有Promise.all 内的所有 promise 状态都变成 fulfilled, 它的状态才会变成 fulfilled, 此时内部 promise 的返回值组成一个数组, 传递给 Promise.all 的回调函数。只要 Promise.all 内部有一个 promise 被 rejected, Promise.all 的状态就变成 rejected, 此时第一个被 reject 的实例的返回值, 会传递给 p 的回调函数。
:::

``` javascript
const promise1 = Promise.resolve(3);
const promise2 = 42;
const promise3 = new Promise((resolve, reject) => {
  setTimeout(resolve, 100, 'foo');
});

Promise.all([promise1, promise2, promise3]).then((values) => {
  console.log(values);
});

// => [3, 42, "foo"]
```

### Promise.race()

::: tip 提示
Promise.race 方法同样是将多个 Promise 实例, 包装成一个新的 Promise 实例。只要Promise.race 中有一个实例率先改变状态, Promise.race 的状态就跟着改变。那个率先改变的 Promise 实例的返回值, 就传递给 Promise.race 的回调函数。
:::

``` javascript
const promise1 = new Promise((resolve, reject) => {
  setTimeout(resolve, 500, 'one');
});

const promise2 = new Promise((resolve, reject) => {
  setTimeout(resolve, 100, 'two');
});

Promise.race([promise1, promise2]).then((value) => {
  console.log(value);
});
// => "two"
```

## 模块

### export

``` javascript
// hello.js
export const PI = 3.14
export function hello() {
  console.log('Hello ES6')
}
export let person = {
   name: 'uzi'
}
```

### import

``` javascript
// main.js
// 使用对象解构赋值加载这3个变量
import {
    PI,
    hello,
    person
} from './hello'

// 也可以将这个模块全部导出
import * as util from './hello'
console.log(util.PI)
// 3.14
```

### export default

``` javascript
const str = "abcdefg"
export default str
```
