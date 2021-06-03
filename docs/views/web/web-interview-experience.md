---
author: 星火燎原@vxhly
title: 前端面经
categories: [web]
tags: [HTML, JavaScript, CSS, Vue.js, Node.js]
date: 2020-07-15 20:35:17
---

<Boxx changeTime="10000"/>

::: tip 前言
收录 WEB 前端常见的面试题以及其答案, 比较偏概念和底层原理（仅供参考, 献给准备换工作的童鞋）。
:::
<!-- more -->

## HTML

::: tip 问: 
你是如何理解 HTML 语义化的?
:::

答: 根据内容的结构化（内容语义化）, 选择合适的标签（代码语义化）来编写 HTML 代码, 这样便于开发者阅读和写出更优雅的代码的同时让浏览器的爬虫和机器很好地解析。在没有 CSS 的情况下, 页面也能呈现出很好地内容结构和代码结构, 有利于 SEO, 和搜索引擎建立良好沟通, 有助于爬虫抓取更多的有效信息, 爬虫依赖于标签来确定上下文和各个关键字的权重；便于团队开发和维护, 语义化更具可读性, 都遵循 W3C 标准的团队, 可以减少差异化。

::: tip 问: 
meta viewport 是做什么用的, 怎么写?
:::

答: `meta viewport` 是专为移动设备下显示所设计的, 只有检测到在移动设备上使用包含 meta 的文档时, meta 标签才会起作用。

``` html
<meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
```

::: tip 问: 
你用过哪些 HTML 5 标签?
:::

答: `<header>` 、 `<footer>` 、 `<address>` 、 `<article>` 、 `<title>` 、 `<canvas>` 、 `<video>` 、 `<time>`
::: tip 问: 
DOCTYPE 有什么作用? HTML4 和 HTML5 有什么区别?
:::

答: DOCTYHPE 是 document type 的简写, 是用来告诉浏览器以什么文档规范来解析 HTML, 它不是 HTML 标签, 位于 HTML 文档第一行。H5 更加语义化能够更加明确地将文档结构表示清楚, H4 的 DOCTYPE 声明需要指定 HTML 使用到的版本号, 但是 H5 并不需要。

::: tip 问: 
HTML 中空（void）元素有哪些?
:::

答: `br` 、 `meta` 、 `img` 、 `hr` 、 `link` 、 `input`

## CSS

::: tip 问: 
CSS 中的盒子模型?
:::

答: 盒子模型分为 IE 盒子模型和 W3C 盒子模型。

* ie的盒子模型包括（margin(外边距), padding(内边距), 边框(border）, 内容(content) (ie的width=content+padding+borde) ）; 
* w3c的盒子模型包括（margin(外边距), padding(内边距), 边框(border）, 内容(content) w3c 的 width=content）

::: tip 问: 
什么是 BFC? 
:::

答: BFC 即 Block Formatting Contexts (块级格式化上下文)。具有 BFC 特性的元素可以看作是隔离了的独立容器, 容器里面的元素不会在布局上影响到外面的元素, 并且 BFC 具有普通容器所没有的一些特性。通俗一点来讲, 可以把 BFC 理解为一个封闭的大箱子, 箱子内部的元素无论如何翻江倒海, 都不会影响到外部。

::: tip 问: 
如何触发 BFC? 
:::

答: 
只要元素满足下面任一条件即可触发 BFC 特性: 

* body 根元素
* 浮动元素: float 除 none 以外的值
* 绝对定位元素: position (absolute、fixed)
* display 为 inline-block、table-cells、flex
* overflow 除了 visible 以外的值 (hidden、auto、scroll)

::: tip 问: 
什么是弹性布局? 
:::

答: 
Flex 是 Flexible Box 的缩写, 意为"弹性布局", 用来为盒状模型提供最大的灵活性。任何一个容器都可以指定为 Flex 布局。Flex 布局, 可以简便、完整、响应式地实现各种页面布局.

## JavaScript

::: tip 问: 
JavaScript 的数据类型?
:::

答: 

* 基本数据类型: Number, String, Boolean, Undefined, Null
* 复杂数据类型: Object, Array, Function, RegExp, Date, Error
* 全局数据类型: Math

::: tip 问: 
`localStorage` 和 `sessionStorage` 区别?
:::

答: `localStorage` 是除了主动去清除它, 否则是会永久保留的； `sessionStorage` 仅在当前会话下有效, 关闭页面或浏览器后被清除, 存放数据大小在 5M 左右。 

::: tip 问: 
JavaScript 中的数组方法有哪些?
:::

答: `map()` 、 `push()` 、 `unshift()` 、 `filter()` 、 `forEach()` 、 `concat()` 、 `join()` 、 `pop()` 、 `sort()` 、 `reverse()` 、 `toString()` 、 `indexOf()` 、 `lastIndexOf()`
::: tip 问: 
JavaScript 的闭包理解?
::: 

答: 在 JavaScript 中, 每当函数被创建, 就会在函数生成时生成闭包。在闭包中, 总体上表现为函数嵌套函数, 它允许我们从内部函数中访问外部函数的作用域, 但是外部函数却访问不到内部函数的作用域, 垃圾回收机制上参数和变量不会被垃圾回收机制回收, 因为内部函数还在引用。闭包有几个好处, 变量可以长期驻扎在内存之中, 不容易引起内存泄漏(引起内存泄漏的基本上算是浏览器处理的问题), 避免全局变量的污染, 有私有成员。

::: tip 问: 
JSONP 和 AJAX 的区别?
:::

答: JSONP 只是支持 GET 请求, 但是不支持其他的请求方式, AJAX 支持所有的请求方式。JSONP 是动态创建一个script标签, 利用 script 标签src 属性访问没有限制来实现跨域, 它允许用户传递一个callback参数给服务端, AJAX 的核心是通过 xmlHttpRequest 获取非本页内容。AJAX 禁止跨域, 但通过服务端代理一样跨域；JSONP 是为了解决 AJAX 禁止跨域的问题, 但 JSONP 并不排斥同域的数据的获取。

::: tip 问: 
什么是 JSONP, 什么是 CORS, 什么是跨域?
:::

答: Jsonp(JSON with Padding) 是 json 的一种"使用模式", 可以让网页从别的域名（网站）那获取资料, 即跨域读取数据。跨域资源共享(CORS) 是一种机制, 它使用额外的 HTTP 头来告诉浏览器, 让运行在一个 origin (domain) 上的Web应用被准许访问来自不同源服务器上的指定的资源。不符合同源策略的皆会引起跨域, 同源（即指在同一个域）就是两个页面具有相同的协议（protocol）, 主机（host）和端口号（port）。

::: tip 问: 
什么是 Promise?
:::

答: Promise 对象用于表示一个异步操作的最终完成 (成功或失败), 及其结果值。是为了解决异步编程造成的回调陷阱等问题。

::: tip 问: 
Promise 有几种状态, 分别是什么? 有什么方法? 
:::

答: 有三种, 分别是等待态、成功态、失败态。 `Promise.all()` 允许发送多个请求, 多个请求中只要有一个请求失败那么最终返回的既是失败; `Promise.race()` 允许发送多个请求, 多个请求中只要有一个请求失败那么最终返回的便是成功; `Promise.reject()` 快速获取一个成功状态的 Promise 对象; `Promise.resolve()` 快速获取一个拒绝状态的 Promise 对象。

::: tip 问: 
Promise 链式调用中的 then、catch、finally 分别代表什么?
:::

答: then 方法。catch 方法是 then 方法的语法糖, 只接受 rejected 态的数据。finally 方法是无论是成功了失败了都会走这里。

::: tip 问: 
原型链是什么?
:::

答: 每个对象都会在其内部初始化一个属性, 就是 prototype(原型), 当我们访问一个对象的属性时, 如果这个对象内部不存在这个属性, 那么他就会去 prototype 里找这个属性。

::: tip 问: 
作用域链是什么?
:::

答: 内部环境可以通过作用域链来访问外部环境的属性和方法, 但是, 外部环境不能访问内部环境的任何属性和方法。

::: tip 问:
原生 JavaScript 实现 AJAX?
:::

答:
AJAX 的核心是 XMLHttpRequest。
一个完整的AJAX请求一般包括以下步骤：

* 实例化 XMLHttpRequest 对象
* 连接服务器
* 发送请求
* 接收响应数据

::: tip 问:
ES6 的新特性?
:::

答:

* 变量声明 const 和 let
* 模版字符串
* 箭头函数
* 函数的参数默认值
* Spread / Rest 操作符（即 `...` ）
* 二进制和八进制字面量
* 对象和数组解构
* for...of 和 for...in
* 对象超类, ES6 允许在对象中使用 super 方法
* ES6 中支持 class 语法, 不过, ES6的class不是新的对象继承模型, 它只是原型链的语法糖表现形式

## 计算机网络知识

::: tip 问: 
HTTP 的请求动词有哪些?
::: 

答: `GET` 、 `POST` 、 `PUT` 、 `PATCH` 、 `DELETE` 、 `OPTIONS` 、 `TRACE` 、 `HEAD` 、 `CONNECT`
::: tip 问: 
常见的 HTTP 状态码哪些? 分别代表的是什么?
::: 

答: 

* `200` (OK): 当您的操作将在响应正文中返回数据时, 出现此结果。
* `204` (No Content): 当您的操作成功, 但不在响应正文中返回数据时, 出现此结果。
* `304` (Not Modified): 当测试实体自上次检索以来是否被修改时, 出现此结果。
* `400` (BadRequest): 当参数无效时, 出现此结果。
* `401` (Unauthorized): 权限验证失败 
* `403` (Forbidden): 客户端错误, 表示资源不可用
* `404` (Not Found): 客户端错误, 当资源不存在时, 出现此结果。
* `413` (Payload Too Large): 当请求长度过长时, 出现此结果。
* `500` (Internal Server Error): 服务器遇到了一个未曾预料的状况, 导致了它无法完成对请求的处理。一般来说, 这个问题都会在服务器端的源代码出现错误时出现。

::: tip 问: 
GET 和 POST 有什么区别?
::: 

答: GET 的传参方式是放在 URL 上面表现为 `xx?xx=xx` , POST 的传参方式是放在请求包体中, 安全性方面 POST 会比 GET 高。GET 的 URL的最大长度是 2083 个字节, path 的部分最长是 2048 个字节（每个浏览器支持的长度都不一致, 一般的不超过 IE 支持的在各个浏览器中都能兼容）, POST 不会受到请求长度的限制。GET 若要支持IE, 则最大长度为 2083byte, 若只支持 Chrome, 则最大长度 8182byte。

::: tip 问: 
浏览器从输入 url 到看到页面的过程发生了什么?
::: 

答: 先是域名解析（DNS）, 把输入的域名解析成其对应的 IP 地址, 然后接着是建立 TCP 三次握手, 三次握手结束之后会发送 HTTP 请求, 服务器处理请求返回 HTTP 报文, 再接着是浏览器下载响应的 HTML 以及相应的资源进行解析渲染工作, 当数据传输结束后需要断开 TCP 连接, 此时会发起 TCP 的四次挥手。

::: tip 问: 
HTTP 缓存有哪几种?
:::

答: 强制缓存、协商缓存、通过文件内容来缓存

::: tip 问: 
前端安全有哪些?
:::

答:

* 浏览器安全（浏览器的同源策略, 限制来自不同源的脚本对当前域的调用和访问。）
* 跨站脚本攻击（XSS, 分为持久型和非持久性）
* CSRF 攻击（跨站请求伪造, 攻击者构造出一个后端请求地址, 诱导用户点击或者通过某些途径自动发起请求）
* 点击劫持（点击劫持是一种视觉欺骗的攻击手段。攻击者将需要攻击的网站通过 iframe 嵌套的方式嵌入自己的网页中, 并将 iframe 设置为透明, 在页面中透出一个按钮诱导用户点击。）

## Vue

::: tip 问:
Vue 的生命周期?
:::

答: beforeCreate（创建前）, created（创建后）, beforeMount(载入前), mounted（载入后）, beforeUpdate（更新前）, updated（更新后）, beforeDestroy（销毁前）, destroyed（销毁后）

::: tip 问:
created（创建后）和 mounted（载入后）有什么区别?
:::

答: created: 在模板渲染成 html 前调用, 即通常初始化某些属性值, 然后再渲染成视图。
mounted: 在模板渲染成 html 后调用, 通常是初始化页面完成后, 再对html的dom节点进行一些需要的操作。

::: tip 问: 
Vue 是如何实现数据响应?
:::

答: 采用数据劫持结合发布者-订阅者模式的方式, 通过 `Object.defineProperty()` 来劫持各个属性的 setter, getter, 在数据变动时发布消息给订阅者, 触发相应监听回调。当把一个普通 Javascript 对象传给 Vue 实例来作为它的 data 选项时, Vue 将遍历它的属性, 用 `Object.defineProperty` 将它们转为 getter/setter。

::: tip 问: 
Proxy 与 Object.defineProperty() 对比?
:::

答: 
Proxy 的优点: 

* 可以直接监听对象而非属性, 并返回一个新对象；
* 可以直接监听数值的变化；
* 可以劫持整个对象, 并返回一个新对象；

Proxy 的缺点: 

* Proxy 是 es6 提供的新特性, 兼容性不好, 所以导致 Vue3.0 一直没有正式发布让开发者使用

Object.defineProperty() 的优点: 

* 兼容性好, 支持 IE9
* IE9 以下的版本不兼容

Object.defineProperty() 的缺点: 

* 无法监控到数组下标的变化, 导致直接通过数组的下标给数组设置值, 不能实时响应；
* 只能劫持对象的属性, 我们需要对每个对象的每个属性进行遍历；

::: tip 问: 
Vue-router 中有几种模式? 有什么区别?
:::

答: hash 和 history。hash 模式下 `#` 后面 hash 值的变化, 并不会导致浏览器向服务器发出请求, 浏览器不发出请求, 也就不会刷新页面。history 模式则会将 URL 修改得就和正常请求后端的URL一样, 如后端没有配置对应 /user/id 的路由处理, 则会返回 404 错误。

::: tip 问: 
v-show 与 v-if 区别?
:::

答: v-show 是 css 切换, v-if 是完整的销毁和重新创建。v-if 切换时开销大, 会在切换过程中对条件块的事件监听器和子组件进行销毁和重建, v-show 的初始化渲染开销大, 在需要频繁切换的或者切换部分内部的 DOM 比较复杂的使用 v-show。

::: tip 问: 
vuex 中的 mutation 和 action 有什么区别?
:::

答: 更改 Vuex 的 store 中的状态的唯一方法是提交 mutation。action 类似于 mutation, 不同在于 action 提交的是 mutation, 而不是直接变更状态, action 亦可包含任意的异步操作。

::: tip 问: 
computed 和 watch 有什么区别?
:::

答: computed 计算属性, 是依赖其他属性的计算值, 并且有缓存, 只有当依赖的值变化时才会更新。
watch 是在监听的属性发生变化时, 在回调中执行一些逻辑。
所以, computed 适合在模板渲染中, 某个值是依赖了其他的响应式对象甚至是计算属性计算而来, 而 watch 适合监听某个值的变化去完成一段复杂的业务逻辑。

::: tip 问: 
为什么 v-for 和 v-if 不建议用在一起?
:::

答: 当 v-for 和 v-if 处于同一个节点时, v-for 的优先级比 v-if 更高, 这意味着 v-if 将分别重复运行于每个 v-for 循环中。如果要遍历的数组很大, 而真正要展示的数据很少时, 这将造成很大的性能浪费。这种场景建议使用 computed, 先对数据进行过滤。

::: tip 问: 
Vue-router 导航守卫有哪些?
:::

答: 

* 全局前置/钩子: beforeEach、beforeResolve、afterEach
* 路由独享的守卫: beforeEnter
* 件内的守卫: beforeRouteEnter、beforeRouteUpdate、beforeRouteLeave

::: tip 问:
Vue 之 EventBus 数据传输优点以及弊端?
:::

答: 

* 优点: 传递数据较为方便, 可以进行兄弟之间, 父子之间的传递
* 缺点: 必须先 on 监听才能进行 emit 触发

::: tip 问:
Vue 子组件如何调用父组件的方法?
:::

答: `$emit` 、 `$parent` 、 `props` 、 `vuex`
::: tip 问:
Vue 组件间的通信方式?
:::

答: `props` 、 `$emit` 、 `EventBus` 、 `vuex` 、 `$attrs` 、 `$listeners` 、 `provide` 、 `inject` 、 `$parent` 、 `$children` 、 `$ref`
