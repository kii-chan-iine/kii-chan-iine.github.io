---
author: 星火燎原@vxhly
title: 谈谈 WEB 前端主流的三大框架
categories: [web]
tags: [Vue.js, React.js, Angular.js, Node.js]
date: 2019-04-08 10:09:47
---

<Boxx changeTime="10000"/>

::: tip 前言
讲到前端的框架, 大家想必都能脱口而出: Angular、React、Vue, 那么这几个框架的优缺点, 以及在项目当中如何抉择框架的使用等等, 本篇就将介绍这三大框架的使用感受。
:::
<!-- more -->

## 双向绑定

开篇就用这三大框架的双向绑定的实现作为一个 demo 引入, 通过这三大框架的双向绑定能够简单的了解三大框架在编码上的一些区别以及其的简易程度, 从下面的代码简洁程度来讲, vue.js 给我们带来的体验还是不错的

### Angular

 `特点`
* 基于 MVVM 开发模式
* Dependency Injection（依赖注入）
* TypeScript
* 模块化

``` html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>

</head>

<body ng-app="MyModule">
    <div ng-controller="MyCtrl">
        <input type="text" ng-model="message">
        <h1>{{ message }}</h1>
    </div>

    <script src="https://cdn.bootcss.com/angular.js/1.7.8/angular.min.js"></script>
    <script>
        // 这边使用到的是推断型依赖注入
        const MyModule = angular.module("MyModule", [])

        const MyCtrl = function($scope) {
            $scope.message = 'Hello world !!!'
        }

        MyModule.controller('MyCtrl', MyCtrl)
    </script>
</body>

</html>
```

### React

 `特点`
* Virtual DOM
* 声明式设计
* 高效、灵活
* JSX
* 组件化
* 单向响应的数据流

``` html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
</head>

<body>
    <div id="root"></div>

    <script src="https://cdn.bootcss.com/react/16.9.0-alpha.0/umd/react.development.js"></script>
    <script src="https://cdn.bootcss.com/react-dom/16.8.6/umd/react-dom.development.js"></script>
    <!-- 注意这边的 babel 只能用 5.x 以下版本的 -->
    <script src="https://cdn.bootcss.com/babel-core/5.8.38/browser.min.js"></script>
    <script type="text/babel">
        // 这边使用到的是 JSX 语法,React 内部并没有实现双向绑定,所以这边使用了 input 的 change  事件来实现双向绑定
    class Input extends React.Component {
      constructor(props) {
        super(props)
        this.state = {
          message: 'Hello world !!!'
        }
        this.handleChange = this.handleChange.bind(this)
      }
      handleChange(event) {
        this.setState({
          message: event.target.value
        });

      }
      render() {
        var message = this.state.message;
        return ( 
          <div>
            <input type = "text" value = { message } onChange = { this.handleChange } /> 
            <h1> { message } </h1> 
          </div>
        );
      }
    }

    ReactDOM.render(<Input/>, document.getElementById('root'));
  </script>
</body>

</html>
```

### Vue

 `特点`
* 基于 MVVM 开发模式
* 渐进式
* 数据驱动友好
* 组件化
* 轻量
* 简洁
* 快速
* 模块化友好

``` html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>

</head>

<body>
    <div id="app">
        <input type="text" v-model="message">
        <h1>{{ message }}</h1>
    </div>

    <script src="https://cdn.bootcss.com/vue/2.6.10/vue.min.js"></script>
    <script>
        new Vue({
            el: '#app',
            data() {
                return {
                    message: 'Hello world !!!'
                }
            }
        })
    </script>
</body>

</html>
```

## Starts

| package                                       | starts | forks | issues | created      | version                                          | downloads                                                  | size                                                                    |
| --------------------------------------------- | ------ | ----- | ------ | ------------ | ------------------------------------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------- |
| [angular](https://github.com/angular/angular) | 47.2k  | 12.6k | 2k+    | Jan 6, 2010  | 

![npm](https://img.shields.io/npm/v/angular.svg) | ![npm dowloads](https://img.shields.io/npm/dm/angular.svg) | ![npm bundle size](https://img.shields.io/bundlephobia/min/angular.svg)

 |
| [react](https://github.com/facebook/react)    | 127k   | 23k   | 400+   | May 25, 2013 | 

![npm](https://img.shields.io/npm/v/react.svg)   | ![npm dowloads](https://img.shields.io/npm/dm/react.svg)   | ![npm bundle size](https://img.shields.io/bundlephobia/min/react.svg)

   |
| [vue](https://github.com/vuejs/vue)           | 136k   | 19.3k | 190+   | Jul 29, 2013 | 

![npm](https://img.shields.io/npm/v/vue.svg)     | ![npm dowloads](https://img.shields.io/npm/dm/vue.svg)     | ![npm bundle size](https://img.shields.io/bundlephobia/min/vue.svg)

     |

## npm 下载趋势

[npm charts](https://npmcharts.com/compare/react,angular,vue?minimal=true&interval=7)

## Vue 与 Angular.js 的区别

### 相同点

* 都支持指令、内置指令和自定义指令
* 都支持过滤器、内置过滤器和自动一过滤器
* 都支持双向绑定
* 都支持双向数据绑定
* 都不支持低端浏览器

### 不同点

* Angular.js 的学习成本比 Vue.js 高, 比如增加了 `Dependency Injection（依赖注入）` 特性, 而 Vue.js 本身提供的 API 都比较简单、直观；而最新版的 Angular.js 直接采用 TypeScript 编写, Vue.js 3.x 版本虽然源码是采用 TypeScript 编写, 但是实际项目中可以不用 TypeScript, 即是一个渐进式框架
* 在性能上, Angular.js 依赖对数据做 `脏检查` , 所以 Watcher 越多越慢。Vue.js 使用基于依赖追踪的观察并且使用异步队列更新, 所以的数据都是独立触发的

## Vue 与 React 的区别

### 相同点

* react 采用特殊的 JSX 语法, Vue.js 在组件开发中也推崇编写 `.vue` 特殊文件格式, 对文件内容都有一定的约定, 两者都需要编译后使用
* 中心思想: 一切都是组件, 组件实例之间可以互相嵌套
* 都提供合理的钩子函数, 可以让开发者定制化的去处理需求
* 都不内置函数 AJAX, Route等功能和到和行包, 而是以插件的方式加载
* 在组件开发中都支持 mixins 的特性

### 不同点

* React 依赖 Virtual DOM, 而 Vue.js 使用的是 DOM 模板。React 采用 Virtual DOM 会对渲染出来的结果做 `脏检查`
* React 不支持双向绑定（需要自己编写相关的 change 函数）
* Vue.js 在模板中提供了指令、过滤器等, 可以非常方便, 快捷的操作 DOM
