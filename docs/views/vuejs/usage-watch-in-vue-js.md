---
author: 星火燎原@vxhly
title: vue 中 watch 的详细用法
categories: [vuejs]
tags: [Vue.js, Node.js]
date: 2019-01-30 14:03:37
---

<Boxx changeTime="10000"/>

::: tip 前言
一个对象, 键是需要观察的表达式, 值是对应回调函数。值也可以是方法名, 或者包含选项的对象。Vue 实例将会在实例化时调用 $watch(), 遍历 watch 对象的每一个属性。
:::
<!-- more -->

## API

[$watch()](https://cn.vuejs.org/v2/api/#watch)

## 简单的用法

``` html
<script>
    export default {
        data() {
            return {
                cityName: 'shanghai'
            }
        },
        watch: {
            cityName(newName, oldName) {
                // ...
            }
        }
    }
</script>
```

或者

``` html
<script>
    export default {
        data() {
            return {
                cityName: 'shanghai'
            }
        },
        watch: {
            cityName: 'nameChange'
        },
        methods: {
            nameChange(newName, oldName) {
                // ...
            }
        }
    }
</script>
```

## immediate 和 handler

这样使用watch时有一个特点, 就是当值第一次绑定的时候, 不会执行监听函数, 只有值发生改变才会执行。如果我们需要在最初绑定值的时候也执行函数, 则就需要用到immediate属性。

比如当父组件向子组件动态传值时, 子组件props首次获取到父组件传来的默认值时, 也需要执行函数, 此时就需要将immediate设为true。

``` html
<script>
    export default {
        data() {
            return {
                cityName: 'shanghai'
            }
        },
        watch: {
            cityName(newName, oldName) {
                // ...
            },
            immediate: true
        }
    }
</script>
```

监听的数据后面写成对象形式, 包含 handler 方法和 immediate, 之前我们写的函数其实就是在写这个 handler 方法；

immediate 表示在 watch 中首次绑定的时候, 是否执行 handler, 值为 `true` 则表示在 watch 中声明的时候, 就立即执行 handler 方法, 值为 false, 则和一般使用 watch 一样, 在数据发生变化的时候才执行 handler。

## deep

当需要监听一个对象的改变时, 普通的watch方法无法监听到对象内部属性的改变, 只有data中的数据才能够监听到变化, 此时就需要deep属性对对象进行深度监听。

``` html
<script>
    export default {
        data() {
            return {
                cityName: {
                    id: 1,
                    name: 'shanghai'
                }
            }
        },
        watch: {
            cityName: {
                handler(newName, oldName) {
                    // ...
                },
                deep: true,
                immediate: true
            }
        }
    }
</script>
```

设置 deep: true 则可以监听到 cityName.name 的变化, 此时会给 cityName 的所有属性都加上这个监听器, 当对象属性较多时, 每个属性值的变化都会执行 handler。如果只需要监听对象中的一个属性值, 则可以做以下优化: 使用字符串的形式监听对象属性: 

``` html
<script>
    export default {
        data() {
            return {
                cityName: {
                    id: 1,
                    name: 'shanghai'
                }
            }
        },
        watch: {
            'cityName.name': {
                handler(newName, oldName) {
                    // ...
                },
                deep: true,
                immediate: true
            }
        }
    }
</script>
```
