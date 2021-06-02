---
author: 星火燎原@vxhly
title: Vue.js 发送全局事件（Bus）
categories: [vuejs]
tags: [Vue.js, Node.js]
date: 2019-01-19 13:27:59
---

<Boxx changeTime="10000"/>

::: tip 前言
vue 1.0 废弃了 vm.$broadcast 和 vm.$dispatch 的 API, 那么在兄弟组件之间进行通信就变的异常为难, 现在就需要一个比较便捷的解决方法。那么, eventBus的作用就体现出来了, 主要是实现途径是在要相互通信的兄弟组件之中, 都引入一个新的 vue 实例, 然后通过分别调用这个实例的事件触发和监听来实现通信和参数传递。
:::
<!-- more -->

## 起因

在 vue1.0 中, 组件之间的通信主要通过 vm.$dispatch 沿着父链向上传播和用 vm.$broadcast 向下广播来实现。然而在 vue2.0 中, 已经废除了这种用法。

vuex加入后, 对组件之间的通信有了更加清晰的操作, 对于中大型的项目来说, 一开始就把vuex的使用计划在内是明智的选择。

然而在一些小型的项目, 或者说像我这样写到一半才发现vue2.0用不了$.broadcast和$dispatch的人来说, 就需要一个比较便捷的解决方法。那么, eventBus的作用就体现出来了。

主要是现实途径是在要相互通信的兄弟组件之中, 都引入一个新的vue实例, 然后通过分别调用这个实例的事件触发和监听来实现通信和参数传递。

## EventBus 的简介

::: tip 提示
`EventBus` 又称为事件总线。在 Vue 中可以使用 `EventBus` 来作为沟通桥梁的概念, 就像是所有组件共用相同的事件中心, 可以向该中心注册发送事件或接收事件, 所以组件都可以上下平行地通知其他组件, 但也就是太方便所以若使用不慎, 就会造成难以维护的灾难, 因此才需要更完善的 Vuex 作为状态管理中心, 将通知的概念上升到共享状态层次。著作权归作者所有。<br>
注意 `eventBus` 是专门处理 `同一个路由下` 的复杂组件之间通信的。, 相对于不同路由之间的通信可以考虑使用 `vuex` 或者 `vue-router`
:::

## 更加详细的 vue 组件通信

[vue 组件的通信](https://www.w3cplus.com/vue/component-communication.html)<br>
[Vue 2.0学习笔记: 事件总线（EventBus）](https://www.w3cplus.com/vue/event-bus.html)

### 总结一下Vue通信的几个方式, 和各自的适用场景

* 父子组件通信: 父子关系
* eventBus通信 : 同一个路由下, 复杂组件的通信。
* Vuex: 全局的、跨越路由、非父子组件的通信都可以用它关系
* 利用$route的params或者query: 跨路由的可以用, 但同一个路由下就不适合用了。
* localStorage / cookie / sessionStorage: 全局可以用, 但是存储到本地
* 甚至利用Vue实例上添加值 （不建议）

## 封装的 bus.js

 `bus.js`

``` javascript
import Vue from 'vue'
const bus = new Vue({})
let _on = bus.$on
let _emit = bus.$emit
let _off = bus.$off
bus.$on = (name, fn) => {
    if (bus._events[name]) {
        bus.$off(name)
    }
    _on.call(bus, name, fn)
}
bus.$off = (name) => {
    if (name && bus._events[name]) {
        _off.call(bus, name)
    }
}
bus.$emit = (name, params) => {
    if (!params || !params.from) {
        console.warn && console.warn('The paramter(from) is null')
    }
    _emit.call(bus, name, params)
}
export default bus
```

 `main.js`

``` javascript
import Vue from 'vue'
import bus from '@/utils/bus'
Vue.prototype.$bus = bus
new Vue({
    el: '#app',
    components: {
        App
    },
    template: '<App/>'
})
```

## 食用栗子

 `a.vue`

``` html
<template>
    <div>
        <button @click="onClick">按钮</button>
    </div>
</template>

<script>
    export default {
        name: 'a',
        methods: {
            onClick() {
                this.$bus.$emit('event', {
                    from: 'a',
                    to: 'b',
                    message: 'hello'
                })
            }
        }
    }
</script>
```

 `b.vue`

``` html
<template>
    <div>
        <span>{{ message }}</button>
    </div>
</template>

<script>
    export default {
        name: 'a',
        data() {
            return {
                message: ''
            }
        },
        mounted() {
            this.$bus.$on('event', res => {
                // console.log(res)    
                // 只有监听一次的时候    
                // this.$bus.$off('event')    
                this.message = res.message
            })
        },
        beforeDestroy() {
            // 事件并不会自动销毁,所以要在逐渐地销毁之前将其关闭    
            this.$bus.$off('event')
        }
    }
</script>
```
