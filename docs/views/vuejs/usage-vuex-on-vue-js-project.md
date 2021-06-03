---
author: 星火燎原@vxhly
title: 在 vue.js 项目中使用 vuex
categories: [vuejs]
tags: [Vue.js, Node.js]
date: 2019-01-26 12:18:11
---

<Boxx changeTime="10000"/>

::: tip 前言
Vuex 是一个专为 Vue.js 应用程序开发的状态管理模式。 它采用集中式存储管理应用的所有组件的状态, 并以相应的规则保证状态以一种可预测的方式发生变化。 这里就简单的介绍 vuex 的使用方法。
:::
<!-- more -->

## 文档

[vuex 官方文档](https://vuex.vuejs.org/zh-cn/)

## 关于 vuex 的严格模式

在 vuex 有个参数是 `strict` 这个是可以配置 vuex 开启严格模式, 这里极力推荐开启严格模式, 当然是在开发模式下, 线上模式可以进行关闭, 因为我们不希望内部逻辑暴露在外面。

至于为什么极力推荐开启 vuex 的严格模式, 有以下这些原因: 

* 官方也是这么推荐的, vuex 中的数据被称作 state, 这些数据通过 getters 将其暴露给组件使用, 那么要想改变其值只能在 vuex 内部实现（即 modules 和 actiions, 具体可以阅读官方文档）
* vuex 被设计出来就是一个第三方的状态管理仓库, 当然这些状态也只能由内部的逻辑实现, 外部也可以对 state 的值进行改变, 但是不推荐
* 我们常说的 cookie 也好 session 也好, 它们有个致命的弱点, 就是比较容易被窃取, 那么 vuex 就规避了这些缺点, 因为所有的数据都在 vuex 的 store 仓库, 只能由 vuex 内部的状态来管理, 就不容易被窃取, 如果由外部的逻辑来改变 state 值, 那么 vuex 的安全性岌岌可危

更多详情请阅读 [vuex 严格模式](https://vuex.vuejs.org/zh/guide/strict.html)

## 关于 vuex

* `state` => 单一状态树, 数据存储库, 即驱动应用的数据源
* `getter` => 派生出 `state` 的值
* `mutation` => 更改状态的唯一途径, 直接修改状态
* `action` => 提交 `mutation` , 不是直接修改状态, 相应在 `view` 上的用户输入导致状态的变化
* `module` => 整合一个模块的 `state` 、 `getter` 、 `mutation` 、 `action`
* `view` => 以声明方式将 `state` 映射到视图

## install

使用命令安装 vuex 依赖

``` bash
npm i -D vuex
# or
yarn add vuex
```

### 目录结构

我们单独创建一个 `store` 目录, 目录结构大致是

``` bash
|-- src
  |-- store
    |-- modules
      |-- index.js
      |-- ... ...
    |-- getters.js      # 根级别的 mutation =>
    |-- index.js        # 我们组装模块并导出 store 的地方
    |-- mutation-types.js        # 根级别的 type => 状态
  |-- main.js
```

### 导入 vuex

 `src/mian.js`

``` javascript
import Vue from 'vue'
import store from '@/store'
Vue.config.productionTip = false
new Vue({
    store,
    render: h => h(App),
}).$mount('#app')
```

 `strore/index.js`

``` javascript
import Vue from 'vue'
import Vuex from 'vuex'
import modules from './modules'
import getters from './getters'
Vue.use(Vuex)
const debug = true // 推荐开启严格模式
export default new Vuex.Store({
    modules,
    getters,
    strict: debug
})
```

 `strore/getters.js`

``` javascript
const getters = {
    [stateName]: state => state. [modulesName]. [stateName]
}
export default getters
```

 `store/modules/index.js`

``` javascript
const files = require.context('.', false, /\.js$/)
const modules = {}
files.keys().forEach(key => {
    if (key === './index.js') return
    modules[key.replace(/(\.\/|\.js)/g, '')] = files(key).default
})
export default modules
```

 `store/mutation-types.js`

``` javascript
// 这边必须使用全大写, 没有为什么
export const TYPES_NAME = 'TYPES_NAME'
```

### vuex 食用例子

做完以上的工作之后我们就可以进行使用 vuex 了, 下方举一个最简单的两个数的 加减乘除 的例子

在线访问 [Edit on CodeSandbox](https://codesandbox.io/s/6j1kojkr9w)

 `store/mutation-types.js`

``` javascript
export const ADDITION = 'ADDITION' // + 加法
export const SUBSTRUCTION = 'SUBSTRUCTION' // - 减法
export const MULTIPLICATION = 'MULTIPLICATION' // * 乘法
export const DIVISION = 'DIVISION' // / 除法
```

 `store/modules/operation.js`

``` javascript
import {
    ADDITION,
    SUBSTRUCTION,
    MULTIPLICATION,
    DIVISION
} from '@/store/mutation-types'
const operation = {
    state: {
        result: 0
    },
    mutations: {
        [ADDITION]: (state, payload) => {
            state.result = Number(payload.a) + Number(payload.b)
        },
        [SUBSTRUCTION]: (state, payload) => {
            state.result = Number(payload.a) - Number(payload.b)
        },
        [MULTIPLICATION]: (state, payload) => {
            state.result = Number(payload.a) * Number(payload.b)
        },
        [DIVISION]: (state, payload) => {
            state.result = Number(payload.a) / Number(payload.b)
        }
    },
    actions: {
        addition({
            commit
        }, payload) {
            commit(ADDITION, payload)
        },
        substrction({
            commit
        }, payload) {
            commit(SUBSTRUCTION, payload)
        },
        multiplication({
            commit
        }, payload) {
            commit(MULTIPLICATION, payload)
        },
        division({
            commit
        }, payload) {
            commit(DIVISION, payload)
        }
    }
}
export default operation
```

 `store/getters.js`

``` javascript
const getters = {
    result: state => state.operation.result
}
export default getters
```

 `views/Operation.vue`

``` html
<template>
    <div>
        <div>
            <input type="number" v-model="payload.a" placeholder="参数 A" style="margin:0 5px;height:18px;width:50px;" />
            {{ operation }}
            <input type="number" v-model="payload.b" placeholder="参数 B" style="margin:0 5px;height:18px;width:50px;" />
            = {{ result }}
        </div>
        <div style="margin-top:10px">
            <button @click="addition" style="margin:0 5px;">+</button>
            <button @click="substrction" style="margin:0 5px;">-</button>
            <button @click="multiplication" style="margin:0 5px;">*</button>
            <button @click="division" style="margin:0 5px;">/</button>
        </div>
    </div>
</template>

<script>
    import {
        mapGetters
    } from 'vuex'

    export default {
        data() {
            return {
                operation: '+',
                payload: {
                    a: 0,
                    b: 0
                }
            }
        },
        computed: {
            ...mapGetters(['result'])
        },
        methods: {
            addition() {
                this.operation = '+'
                this.$store.dispatch('addition', this.payload)
            },
            substrction() {
                this.operation = '-'
                this.$store.dispatch('substrction', this.payload)
            },
            multiplication() {
                this.operation = '*'
                this.$store.dispatch('multiplication', this.payload)
            },
            division() {
                this.operation = '/'
                this.$store.dispatch('division', this.payload)
            }
        }
    }
</script>
```
