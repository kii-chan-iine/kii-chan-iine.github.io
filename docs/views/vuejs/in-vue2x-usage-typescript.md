---
author: 星火燎原@vxhly
title: 在 vue2.x 下使用 typescript
categories: [vuejs]
tags: [Vue.js, TypeScript]
date: 2021-05-18 11:20:19
---

<Boxx changeTime="10000"/>

::: tip 前言
今天我们来看看如何在vue项目中使用ts,目前使用公司项目用的较多的还是vue2.x,vue是渐进式的框架, 我们学习也要渐进式的, 所以本文也先围绕vue2.x来对Ts进行实战, 为后期切换vue3.0打下基础。
:::
<!-- more -->

## 简介

在使用前我们要先在项目中安装Ts, 安装过程就不细说了, 如果只是学习, 推荐用vue的官方脚手架, 里面就带有安装Ts选项。接着要安装下 `vue-class-component` 和 `vue-property-decorator`

安装之前我们先了解下 `vue-class-component` 和 `vue-property-decorator`

`vue-class-component` 是vue 的官方库, 作用是以class的模式编写组件。这种编写方式使vue组件可以使用继承、混入等高级特性,更重要的是使 Vue 组件更好的跟TS结合使用。

`vue-property-decorator` 是社区出的,
基于 `vue-class-component` 拓展出了很多操作符 @Prop @Emit @Inject 等;可以说是 `vue-class-component` 的一个超集, 使代码更为简洁明了, `options` 里面需要配置 `decorator` 库不支持的属性, 比如 `components`, `filters`, `directives` 等。

这两者都是离不开装饰器的, 装饰器已在ES提案中。`decorator` 是装饰器模式的实践。装饰器模式呢, 它是继承关系的一个替代方案。动态地给对象添加额外的职责。在不改变接口的前提下, 增强类的功能。

## 使用

### Component

装饰器可以接收一个对象作为参数, 可以在对象中声明 `components` , `filters`, `directives` 等未提供装饰器的选项。

``` vue
<template>
 <div class="home">
   {{num | addOne('过滤器第二个参数')}}
   <Test
     ref="helloWorld"
     v-test="'h'"
   />
 </div>
</template>


<script lang='ts'>
  import { Component,Vue} from 'vue-property-decorator'
  import Test from '@/components/Test.vue'

  @Component({
    name: 'Home',
    //组件
    components: {
      Test
    },
    
    //局部指令
    directives: {
      test(el: HTMLElement, binding) {
      console.log('DOW:',el,'局部指令:' ,binding)
      }
    },

    // 局部过滤
    filters: {
      addOne(num: number, towParam: string) {
        console.log(towParam, '局部过滤器')
        return num + 3
      }
    }
    
    //混入
    // mixins: [ResizeMixin]
  })

  export default class extends Vue {
    private num: number = 1 //定义一个变量
  }
</script>
```

::: tip
要使用Ts需要在script标签的lang属性值设为ts
:::

### 命周期

``` vue
<template>
  <div class="home"></div>
</template>
  
<script lang='ts'>
  import { Component, Vue } from 'vue-property-decorator'
  
  @Component({
    name: 'LifeCycle',
  })
  
  export default class extends Vue {
    private num = 1
    private created(): void {
      console.log(this.num)
    }
    private mounted(): void {
      console.log(this.num)
    }
  }
</script>
```

### 方法、属性

``` vue
<template>
  <div class="home">
    <button @click="addAge">加1</button>
    {{ num }}
  </div>
</template>
 
<script lang='ts'>
  import { Component, Vue } from 'vue-property-decorator'
  @Component({
    name: 'AttrMethod',
  })
  export default class extends Vue {
    private num = 1 //属性
    private checked = true

    //方法
    private addAge(): void {
      this.num++
      this.checked = false
    }

    private mounted(): void {
      console.log(this.num)
    }
  }
</script>
```

### computer(计算属性)

``` vue
<template>
  <div class="computer">
    {{ count(this.num, 2) }}
    {{ msg }}
  </div>
</template>
 
<script lang='ts'>
  import { Component, Vue } from 'vue-property-decorator'
  @Component({
    name: 'Computers',
  })
  export default class extends Vue {
    private num = 1
    private mounted(): void {
      console.log(this.num)
    }
  /*计算属性*/

    //传参写法
    private get count() {
      return function (num: number, numbers: number) {
        return num + numbers
      }
    }

    //普通写法
    private get msg() {
      return '普通写法的计算属性'
    }
  }
</script>
```

### watch(监听)

``` vue
<template>
  <div class="watch">
    {{ num }}
  </div>
</template>

<script lang='ts'>
  import { Component, Vue, Watch } from 'vue-property-decorator'
  @Component({
    name: 'Watch',
  })
  export default class extends Vue {
    private num = 1
    private mounted(): void {
      this.timeOut()
    }
    private timeOut() {
      setTimeout(() => {
        this.num++
      }, 1000)
    }
    //监听
    @Watch('num', { immediate: true, deep: true })
    onNumChange(val: string, old: string) {
      console.log(val, old, 'watch')
    }
  }
</script>
```

::: tip
`onNumChange` 方法要紧挨着 `@Watch`,它们中间不能有其他代码,而且这个方法名称可以自定义，没有强制要求。
:::

### ref

``` vue
<template>
  <div class="watch">
    <img alt="Vue logo" src="../assets/logo.png" ref="img" />
    {{ num }}
    <Test ref="test" />
  </div>
</template>

<script lang='ts'>
  import { Component, Vue, Ref } from 'vue-property-decorator'
  import Test from '@/components/Test.vue'
  @Component({
    name: 'Watch',
    components: {
      Test,
    },
  })
  export default class extends Vue {
    private num = 1
    @Ref() readonly test!: Test //引入的组件的ref
    @Ref('img') readonly img!: HTMLButtonElement //普通html标签的ref
    private mounted(): void {
      console.log(
        '普通的ref使用方式:',
        this.$refs.test,
        '定义变量的ref使用方式:',
        this.test,
        '引入组件的ref'
      )
      console.log(this.img, this.$refs.img, '普通img标签的ref')
    }
  }
</script>
```

### 依赖注入

#### Provide

``` vue
<template>
  <div class="home">
    <Inject
      ref="helloWorld"
    />
  </div>
</template>
<script lang='ts'>
  // @ is an alias to /src
  /*eslint-disable */
  import {
    Component,
    Vue,
    Provide,
  } from 'vue-property-decorator'
  import Inject from '@/components/Inject.vue'

  const symbol = Symbol('baz')
  //装饰器注明此类
  @Component({
    name: 'Provide',
    components: {
      Inject
    },
    // mixins: [ResizeMixin]
  })
  export default class extends Vue {
    @Provide() foo = 'foo' //依赖注入
    @Provide() optional = 'optional' //依赖注入
    @Provide('bar') baz = 'bar'
  }
</script>
```

#### Inject

``` vue
<template>
  <div class="hello">
    <h1 @click="returnValue">{{ msg }}</h1>
  </div>
</template>

<script lang="ts">
  import {
    Component,
    Vue,
    Inject
  } from 'vue-property-decorator'
  @Component
  export default class HelloWorld extends Vue {
    @Inject() readonly foo!: string //接收依赖注入的值
    @Inject({ from: 'optional', default: 'default' }) readonly optional!: string //父组件，爷爷组件没传optional时，使用default设置默认值
    @Inject('bar') readonly bar!: string
    
    private moun ted(): void {
      console.log( 22, this.foo, this.optional, this.bar)
    }
  }
</script>
```

### Prop

子组接收父组件传进来的值

#### 父组件

``` vue
<template>
  <div class="home">
    <Props
     :msg="msg"
     prop-c='11'
    />
  </div>
</template>

<script lang='ts'>
  // @ is an alias to /src
  /*eslint-disable */
  import {
  Component,
  Vue,

  } from 'vue-property-decorator'
  import Props from '@/components/Prop.vue'

  const symbol = Symbol('baz')
  //装饰器注明此类
  @Component({
  name: 'Prop',
  components: {
    Props
  },
  
  })
  export default class extends Vue {
  private msg: string = 'hello'
  private name: string = 'sss'
  private checked: boolean = true
  private num: number = 1
  }
</script>
```

#### 子组件

``` vue
<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <span>{{ propB }}</span>
    <span>{{ propC }}</span>
  </div>
</template>

<script lang="ts">
  import {
    Component,
    Prop,
    Vue,
  } from 'vue-property-decorator'
  @Component
  export default class HelloWorld extends Vue {
    @Prop() private msg!: string //!，非null和undefined
    @Prop(Number) readonly propA: number | undefined
    @Prop({ default: 'default value' }) readonly propB!: string
    @Prop([String, Boolean]) readonly propC: string | boolean | undefined
  }
</script>
```

### Emit

向父组件发射个方法

#### 父组件

``` vue
 <template>
  <div class="home">
    <EmitChild
      @return-value="returnValue"
    />
  </div>
</template>
<script lang='ts'>
  // @ is an alias to /src
  /*eslint-disable */
  import {
    Component,
    Vue,
  } from 'vue-property-decorator'
  import EmitChild from '@/components/Emit.vue'

  const symbol = Symbol('baz')
  //装饰器注明此类
  @Component({
    name: 'Emit',
    components: {
      EmitChild
    },
  })

  export default class extends Vue {
    private returnValue(aa:number):void {
      console.log(aa)
    }
  }
</script>
```

#### 子组件

``` vue
<template>
  <div class="hello">
    <p @click="returnValue">
       emit
    </p>
  </div>
</template>

<script lang="ts">
  import {
    Component,
    Vue,
    Emit,
  } from 'vue-property-decorator'
  @Component
  export default class HelloWorld extends Vue {
    @Emit() //把方法发射出去可以让父组件使用
    returnValue() {
      return 10
    }
  }
</script>
```

### PropSync

实现 `sync` 修饰符(`prop` 双向绑定)

#### 父组件

``` vue
<template>
  <div>
    <button @click="exportName">输出name</button>
    <PropSyncChild
      :name.sync="name"
    />
  </div>
</template>
<script lang='ts'>
  // @ is an alias to /src
  /*eslint-disable */
  import {
    Component,
    Vue,
  } from 'vue-property-decorator'
  import PropSyncChild from '@/components/PropSync.vue'

  //装饰器注明此类
  @Component({ 
    name: 'PropSync',
    components: {
      PropSyncChild
    },
  })
  export default class extends Vue {
    private name: string = 'sss'
    exportName():void{
      console.log(this.name)
    }
  }
</script>
```

#### 子组件

``` vue
<template>
  <div class="hello">
    <p @click="setSyncedName">
     我是子组件： 同步、子组件修改父组件
    </p>
  </div>
</template>

<script lang="ts">
  import {
    Component,
    Vue,
    PropSync, 
  } from 'vue-property-decorator'
  @Component
  export default class HelloWorld extends Vue {
    @PropSync('name', { type: String }) syncedName!: string //同步，可让子组件修改父组件的值
    public setSyncedName(): void {
      console.log('prop双向绑定',)
      this.syncedName = '同步、子组件修改父组件'
    }
  }
</script>
```

### Model

实现 `v-model` 双向绑定

#### 父组件

``` vue
<template>
<div >
  <button @click="setChecked"> 修改checked</button>
  <ModelChild
   v-model="checked"
  />
</div>
</template>
<script lang='ts'>
  import {
    Component,
    Vue,
  } from 'vue-property-decorator'
  import ModelChild from '@/components/Model.vue'
  //装饰器注明此类
  @Component({
    name: 'Model',
    components: {
      ModelChild
    },
  })
  export default class extends Vue {
    private checked = false
   setChecked():void{
      this.checked=!this.checked
    }
  }
</script>
```

### 子组件

``` vue
<template>
  <div class="hello">
    我是子组件的checked: {{checked}}
  </div>
</template>

<script lang="ts">
  import {
    Component,
    Vue,
    Model,
  } from 'vue-property-decorator'
  @Component
  export default class HelloWorld extends Vue {
    @Model('change', { type: Boolean }) readonly checked!: boolean //v-model
  }
</script>
```
