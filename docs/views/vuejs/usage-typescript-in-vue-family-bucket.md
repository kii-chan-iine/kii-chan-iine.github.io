---
author: 星火燎原@vxhly
title: vue 全家桶结合 TypeScript
categories: [vuejs]
tags: [Vue.js, Vuex, VueRouter, Node.js]
date: 2018-09-23 15:19:43
---

<Boxx changeTime="10000"/>

::: tip 前言
自从 vue cli 升级到 3.0 之后, vue 项目对 TypeScript 的支持度也不断的提升, 为了更好的学习 TypeScript, 本篇文章记录着 vue 全家桶结合 TypeScript 的食用方式。
:::
<!-- more -->

## vue-cli 3.0

### cli 工具到底升级了什么

* GUI 项目管理界面
* 更加简洁的目录结构
* TypeScript 的支持
* Git Hook 提交 commit 的时候检查代码的规范性
* 依赖和插件管理更加方便(GUI 管理)
* Progressive Web App 的支持
* Vue Router & Vuex
* ESLint / TSLint / Prettier
* 用 Jest 或 Mocha 进行单元测试
* 用 Cypress 或者 Nightwatch 进行 E2E 测试
* 预配置webpack功能, 如模块热替换、代码拆分、 摇树优化（tree-shaking）、高效持久化缓存等
* 通过 Babel 7 + preset-env（Babel插件）对 ES2017进行转换和基于使用情况注入 polyfill
* 支持 PostCSS（默认启用 autoprefixer）和所有主要的 CSS 预处理器

### 安装

``` bash
npm i -g @vue/cli
```

### 创建项目

``` bash
vue create [productName]
# or
vue ui
```

在选方案时, 记得把 TypeScript 的支持勾选

## TypeScript

TypeScript 是一种由微软开发的自由和开源的编程语言。它是 JavaScript 的一个超集, 而且本质上向这个语言添加了可选的静态类型和基于类的面向对象编程. TypeScript 扩展了JavaScript语法, 任何已经存在的JavaScript程序, 可以不加任何改动, 在TypeScript环境下运行。TypeScript只是向JavaScript添加了一些新的遵循ES6规范的语法, 以及基于类的面向对象编程的这种特性。

## 在项目中食用 TypeScript

### vue

#### data

``` html
<script lang="ts">
    import Vue from 'vue'
    import Component from 'vue-class-component'

    @Component
    export default class App extends Vue {
        msg1: number = 123
        msg2: String = 'hello word'
        obj: Object = {
            msg1: 123,
            msg2: 'hello word'
        }
        arr: Array = [
            123,
            'hello word'
        ]
    }
</script>
```

======>

``` html
<script>
    export default {
        name: 'App',
        data() {
            return {
                msg1: 123,
                msg2: 'hello word',
                obj: {
                    msg1: 123,
                    msg2: 'hello word'
                },
                arr: [
                    123,
                    'hello word'
                ]
            }
        }
    }
</script>
```

#### watch

::: tip 提示
@Watch(path: string, options: WatchOptions = {})
:::

``` html
<script lang="ts">
    import {
        Vue,
        Component,
        Watch
    } from 'vue-property-decorator'

    @Component
    export default class YourComponent extends Vue {
        @Watch('child')
        onChildChanged(val: string, oldVal: string) {}

        @Watch('person', {
            immediate: true,
            deep: true
        })
        onPersonChanged(val: Person, oldVal: Person) {}
    }
</script>
```

======>

``` html
<script>
    export default {
        watch: {
            'child': {
                handler: 'onChildChanged',
                immediate: false,
                deep: false
            },
            'person': {
                handler: 'onPersonChanged',
                immediate: true,
                deep: true
            }
        },
        methods: {
            onChildChanged(val, oldVal) {},
            onPersonChanged(val, oldVal) {}
        }
    }
</script>
```

### vuex

### axios

#### 需要安装描述文件

``` bash
yarn add -D @types/axios
```

#### 一个栗子

 `config.ts`

``` javascript
const serverURL: String = 'http://127.0.0.1:8090/api'
export {
    serverURL as url
}
```

 `axios.ts`

``` javascript
import axios from 'axios'
import {
    url
} from '@/config'

const protocol = window.location.protocol
const host = window.location.host
const domain = document.domain
let serverURL: any = ''

if (domain === '127.0.0.1' || domain === 'localhost') {
    serverURL = url
} else {
    serverURL = `${protocol}` + '//' + `${host}:8090/api`
}

// 创建 axios 实例
const instance = axios.create({
    baseURL: serverURL,
    timeout: 0
})

export default instance
```
