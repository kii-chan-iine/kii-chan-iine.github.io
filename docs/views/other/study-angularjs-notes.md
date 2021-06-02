---
author: 星火燎原@vxhly
title: Angularjs 学习笔记
categories: [other]
tags: [Node.js, Angularjs]
date: 2019-04-01 16:44:20
---

<Boxx changeTime="10000"/>

::: tip 前言
Angularjs 1.0 的学习笔记。
:::
<!-- more -->

## 依赖注入

``` html
<!DOCTYPE html>
<html ng-app="MyModule">

<head>
    <title>最基本的依赖注入方式</title>
    <meta charset="utf-8" />
    <script src="../Script/angular.min.js" type="text/javascript"></script>
    <script src="base.js"></script>
    <link rel="stylesheet" href="../bootstrap-3.0.0/css/bootstrap.css">
</head>

<body>
    <div>
        <div ng-controller="MyCtrl" style="text-align: c">
            <span style="font-size: 20px;">{{gameName}}</span>
        </div>
    </div>
</body>

</html>
```

### 内联式注入

``` javascript
//最基本的依赖注入方式
var MyModule = angular.module("MyModule", [])
MyModule.controller('MyCtrl', ['$scope',
    function($scope) { //可以修改参数,如$scope1,但下面必须同时改
        $scope.gameName = "hello"
    }
]);
```

### 推断型注入

``` javascript
//推断型注入方式:函数参数的名称必须要和被注入的对象相同
var MyModule = angular.module("MyModule", [])

var MyCtrl = function($scope) {
    $scope.gameName = "hello"
}

MyModule.controller('MyCtrl', MyCtrl)
```

### 声明式注入

``` javascript
//声明式注入
var MyModule = angular.module("MyModule", [])

var MyCtrl = function(thisISMyName) { //参数名称可以随意变化
    thisISMyName.gameName = "hello"
}

MyCtrl.$inject = ['$scope'] //加载模块

MyModule.controller('MyCtrl', MyCtrl)
```

## 双向绑定

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
