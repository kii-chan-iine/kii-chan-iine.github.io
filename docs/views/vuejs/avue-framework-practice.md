---
author: 星火燎原@vxhly
title: Avue 前端框架使用实践
categories: [vuejs]
tags: [Vue.js, Node.js, Component]
date: 2020-08-20 11:39:08
---

<Boxx changeTime="10000"/>

::: tip 前言
Avue 前端是基于 ElementUI 库进行的二次封装, 简化了一些繁琐的操作。本篇博文将带领大家学习这套框架的使用实践, 以及它的一些二次封装的思路。
:::
<!-- more -->

## 关于 Avue

### 参考地址

* Github 仓库：[nmxiaowei/avue](https://github.com/nmxiaowei/avue)
* Gitee 仓库：[smallwei/Avue](https://gitee.com/smallweigit/avue)
* Avue 官方文档：[Avue](https://avuejs.com/)
* ElementUI 官方文档：[ElementUI](https://element.eleme.cn/#/zh-CN/component/installation)
* 基于 Avue 的后台管理系统示例: [Avue 后台管理系统](https://cli.avuejs.com/wel/index)

### 简介

::: tip 简介
Avue.js 是基于现有的 element-ui 库进行的二次封装, 从而简化一些繁琐的操作, 核心理念为数据驱动视图, 主要的组件库针对 table 表格和 form 表单场景, 同时衍生出更多企业常用的组件, 达到高复用, 容易维护和扩展的框架, 同时内置了丰富了数据展示组件, 让开发变得更加容易。

它的核心是数据驱动 UI 的思想, 让我们从繁琐的 crud 开发中解脱出来, 他的写法类似 easyUI, 但是写起来比 easyUI 更加容易, 因为它是集合了 vue 中的数据双向绑定特性以及其他的一些特性。内含大量的组件和插件, 例如 crud、表单、数据展示卡、人物展示卡、打印的插件、Excel 导入插件等。

缺点是生态不好, 出错网上没有解决方案, 对于定制化要求比较高的需求, 使用avue不是很方便。当然除了而二次封装的 element-ui 库, 还有二次封装的 ant-design-vue 、echarts、vant, 维护比较好组件也比较全面当属element-ui 库, 其中二次封装 ant-design-vue 、vant 暂未开放给用户使用。
:::

使用 Avue 好处：

1. 让开发人员更加的关注业务代码, 而无需过多的关心页面的布局
2. 简洁高效, 通过简单的配置项即可组合出我们想要的效果
3. 提供了大量的常用组件库以及丰富的插件库
4. 兼容现在的主流浏览器, 开箱即用的插件引入模式

Avue 的组件组成：

1. 单组件, 在 element-ui 库中的组件进行二次封装, 让使用更加方便 
2. CRUD 组件, 即表格组件, 常用于需要使用表单来展示数据的业务或者是日常 CRUD 的一些操作, 文档地址：[Avue CRUD 组件](https://avuejs.com/doc/crud/crud-doc)
3. 表单组件, 常用于数据收集、数据修改等业务, 文档地址：[Avue FORM 组件](https://avuejs.com/doc/form/form-doc)
4. 数据展示组件, 常用与一些需要数据展示的一些业务, 常见于大数据类的后台管理系统中的首页里面的一些数据统计, 文档地址：[Avue Data 组件](https://avuejs.com/doc/data/data0)
5. 其他组件, 封装的一些常用的组件(比较杂, 例如：打印、评论、拖拽、水印)

### 浏览器兼容性

::: warning Vue.js 的兼容性
Vue 不支持 IE8 及以下版本, 因为 Vue 使用了 IE8 无法模拟的 ECMAScript 5 特性。但它支持所有[兼容 ECMAScript 5 的浏览器](https://caniuse.com/#feat=es5)。
:::

::: warning Avue 的兼容性
支持所有符合 ES5 标准的浏览器(不支持 IE8 及以下版本）.
:::

## 实践

### 下载与引入

#### 下载

使用 npm 或者 yarn 就可以对 Avue 这套 UI 框架进行下载, 并将该依赖写入到 `package.json` 中

``` bash
npm i @smallwei/avue
# or
yarn add @smallwei/avue
```

#### 完整引入

::: tip 提示
Avue 目前不支持按需引入, 只支持完整引入
:::

与其他 UI 框架的一样引入方式, 在项目的 `main.js` 中添加以下代码即可

``` javascript
import Vue from 'vue'
import Avue from '@smallwei/avue'
import '@smallwei/avue/lib/index.css'

Vue.use(Avue)
```

## 使用案例

在使用前, 我们还需要另外的导入 element-ui 库的依赖, 因为 Avue 是基于 element-ui 库进行的二次封装, 并且 Avue 源码中并没有为我们导出  element-ui 库的依赖。所以我们仍然需要在项目的 `main.js` 中添加 element-ui 库的依赖。

``` javascript
import Vue from 'vue'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'

Vue.use(ElementUI)
```

<CodePreview embedUrl="https://codesandbox.io/embed/avue-demo-z4uje?autoresize=1&fontsize=14&hidenavigation=1&theme=dark&view=preview" />

### 表单

``` vue
<template>
  <avue-form
    ref="form"
    v-model="obj"
    :option="option"
    @submit="submit"
  >
    <template
      slot="footer"
    >
      <span style="color:red;text-align:right;display:inline-block;width:100%;">打 * 的为必填项</span>
    </template>
  </avue-form>
</template>

<script>
  export default {
    name: 'MyForm',

    data () {
      return {
        obj: {
          title: '我是表单的表头',
          name: '',
          tag: '',
          sex: ''
        },
        option: {
          card: true,
          column: [
            {
              label: '',
              type: 'title',
              prop: 'title',
              span: 24,
              labelWidth: 0,
              styles: {
                textAlign: 'center'
              }
            },
            {
              label: '名称',
              prop: 'name',
              rules: [
                {
                  required: true,
                  message: '请输入名称'
                }
              ]
            },
            {
              label: '标签',
              prop: 'tag',
              type: 'select',
              dicData: [
                {
                  label: '选项1',
                  value: '1'
                },
                {
                  label: '选项2',
                  value: '2'
                },
                {
                  label: '选项3',
                  value: '3'
                }
              ],
              rules: [
                {
                  required: true,
                  message: '请输入名称'
                }
              ]
            },
            {
              label: '性别',
              prop: 'sex',
              type: 'radio',
              span: 6,
              dicData: [
                {
                  label: '男',
                  value: 0
                },
                {
                  label: '女',
                  value: 1
                }
              ],
              rules: [
                {
                  required: true,
                  message: '请选择性别'
                }
              ]
            },
            {
              label: '兴趣爱好',
              prop: 'hobby',
              type: 'checkbox',
              span: 12,
              dicData: [
                {
                  label: '选项1',
                  value: 0
                },
                {
                  label: '选项2',
                  value: 1
                },
                {
                  label: '选项3',
                  value: 2
                },
                {
                  label: '选项4',
                  value: 3
                },
                {
                  label: '选项5',
                  value: 4
                },
                {
                  label: '选项6',
                  value: 5
                }
              ]
            },
            {
              label: '公开资料',
              prop: 'isPublish',
              span: 6,
              type: 'switch',
              dicData: [
                {
                  label: '选项1',
                  value: 0
                },
                {
                  label: '选项2',
                  value: 1
                }
              ]
            },
            {
              label: '',
              type: 'title',
              prop: 'footer',
              span: 24,
              labelWidth: 0,
              formslot: true
            }
          ]
        }
      }
    },

    methods: {
      emptytChange () {},
      submit () {}
    }
  }
</script>
```

### 表格

``` vue
<template>
  <div>
    <img
      ref="printImg"
      style="display:none;"
      :src="printImg"
    >
    <avue-crud
      ref="crud"
      :data="data"
      :option="option"
      :page.sync="page"
      @row-del="rowDel"
    >
      <template slot="menuLeft">
        <el-button
          icon="el-icon-printer"
          type="primary"
          size="small"
          :loading="printLoading"
          @click="rowPrint"
        >
          打印
        </el-button>
      </template>

      <template
        slot="tags"
        slot-scope="scope"
      >
        <el-tag
          v-for="(item, index) in scope.row.tags"
          :key="index"
        >
          {{ item }}
        </el-tag>
      </template>
    </avue-crud>
  </div>
</template>

<script>
  import html2canvas from 'html2canvas'
  export default {
    name: 'MyCrud',

    data () {
      return {
        printLoading: false,
        printImg: '',
        page: {
          total: 1,
          pageSize: 20,
          pagerCount: 5
        },
        data: [
          {
            title: '1',
            desc: '文件入库, 包含csv、excel等文件入库...',
            tags: ['2', '2'],
            create_user: '1',
            create_update: '1',
            concat: '1'
          }
        ],
        option: {
          // border: true,
          // printBtn: true,
          addBtn: true,
          index: true,
          page: true,
          indexLabel: '序号',
          // align: 'center',
          // menuAlign: 'center',
          column: [
            {
              label: '标题',
              prop: 'title'
            },
            {
              label: '描述',
              prop: 'desc'
            },
            {
              label: '标签',
              prop: 'tags',
              slot: true,
              type: 'select',
              multiple: true,
              filterable: true,
              allowCreate: true,
              defaultFirstOption: true,
              dicData: [
                {
                  label: '数据库',
                  value: '数据库'
                },
                {
                  label: 'HTTP',
                  value: 'HTTP'
                },
                {
                  label: 'API',
                  value: 'API'
                },
                {
                  label: '文件',
                  value: '文件'
                },
                {
                  label: 'HDFS',
                  value: 'HDFS'
                },
                {
                  label: 'SPARK',
                  value: 'SPARK'
                },
                {
                  label: 'HAVE',
                  value: 'HAVE'
                },
                {
                  label: 'FTP',
                  value: 'FTP'
                }
              ]
            },
            {
              label: '上传者',
              prop: 'create_user'
            },
            {
              label: '上传部门',
              prop: 'create_update'
            },
            {
              label: '联系方式',
              prop: 'concat'
            }
          ]
        }
      }
    },

    methods: {
      setClass ({ row, column, rowIndex, columnIndex } = {}) {
        // console.log(column)
        const { label } = column
        if (label === '操作') {
          return 'no-print'
        }
        // if (columnIndex === 8) {
        //   return 'no-print'
        // }
      },
      rowPrint () {
        this.option.menu = false
        const printDOM = this.$refs.crud.$el.querySelector('.el-table')
        this.printLoading = true
        this.toImage(printDOM)
      },
      async toImage (printDOM) {
        setTimeout(async () => {
          const canvas = await html2canvas(printDOM)
          const dataURL = canvas.toDataURL('image/png', 1.0)
          this.printImg = dataURL
          await this.$nextTick()
          if (this.printImg) {
            this.$print(this.$refs.printImg, {
              isImg: true
            })
          }
          this.printLoading = false
          this.option.menu = true
        }, 100)
      },
      rowDel () {
        this.$confirm('此操作将永久删除该数据, 是否继续?', '提示', {
          confirmButtonText: '确定',
          cancelButtonText: '取消',
          type: 'warning'
        })
          .then(() => {
            this.$message({
              type: 'success',
              message: '删除成功!'
            })
          })
          .catch(() => {
            this.$message({
              type: 'info',
              message: '已取消删除'
            })
          })
      }
    }
  }
</script>

<style>
  .el-tag + .el-tag {
    margin-left: 10px;
  }
</style>
```

## 二次封装思路

更多的封装思路阅读 [绝对干货~！学会这些Vue小技巧, 可以早点下班和女神约会了](https://juejin.im/post/6844904191224184840)

### component 动态组件

::: tip 提示
使用到的是 vue 中的动态组件 `component`
:::

* Props
  + `is` : - *string* | *ComponentDefinition* | *ComponentConstructor*
  + `inline-template` : - *boolean*

* 用法

  渲染一个“元组件”为动态组件。依 `is` 的值, 来决定哪个组件被渲染。

``` html
<!-- 动态组件由 vm 实例的 `componentId` property 控制 -->
<component :is="componentId"></component>

<!-- 也能够渲染注册过的组件或 prop 传入的组件 -->
<component :is="$options.components.child"></component>
```

* 参考：[动态组件](https://cn.vuejs.org/v2/guide/components.html#%E5%8A%A8%E6%80%81%E7%BB%84%E4%BB%B6)

示例：

``` vue
<template>
  <component :is="getComponent(type)">
  </component>
</template>

<script>
export default {
  data() {
    return {
      type: 'input'
    }
  },
  methods: {
    getComponent(type) {
      return `el-${type}`
    }
  }
}
</script>
```

### $attrs 与 $listeners

::: tip 提示
`$attrs` : 当组件在调用时传入的属性没有在 `props` 里面定义时, 传入的属性将被绑定到 `$attrs` 属性内（ `class` 与 `style` 除外, 他们会挂载到组件最外层元素上）。并可通过 `v-bind="$attrs"` 传入到内部组件中。

`$listeners` : 当组件被调用时, 外部监听的这个组件的所有事件都可以通过 `$listeners` 获取到。并可通过 `v-on="$listeners"` 传入到内部组件中。
:::

``` vue
<template>
  <component-name
    v-bind="$attrs"
    v-on="$listeners"
  />
</template>
```

对于 `$attrs` , 我们也可以使用 `$props` 来代替, 实现代码如下

``` vue
<template>
  <component-name
    v-bind="$props"
    v-on="$listeners"
  />
</template>
```

### 自定义 v-model

在用 Vue 开发前端时, 不论使用原生还是封装好的UI库, 对于表单组件, 一般都会使用到`v-model`。虽然 `v-model` 是一个语法糖, 但是吃到嘴里挺甜的啊。学会自定义`v-model`, 还是很有必要的。

一个组件上的 `v-model` 默认是通过在组件上面定义一个名为 `value` 的 `props`,同时对外暴露一个名为 `input` 的事件。

方式一：通过 change 监听方式同步数据

``` vue
<template>
  <div class="custom-input">
    <input :value="value" @change="$_handleChange" />
  </div>
</template>
<script>
export default {
  name: 'custom-input',
  props: {
    value: {
      type: String,
      default: ''
    }
  },
  methods: {
    $_handleChange(e) {
      this.$emit('input', e.target.value)
    }
  }
}
</script>
```

方式二：使用 `computed` 中的 `get` 和 `set` 方法进行同步

``` vue
<template>
  <div class="custom-input">
    <input v-model="text" />
  </div>
</template>
<script>
export default {
  name: 'custom-input',
  props: {
    value: {
      type: String,
      default: ''
    }
  },
  computed: {
    text: {
      get () {
        return this.value
      },
      set (value) {
        this.$emit('input', value)
      }
    }
  }
}
</script>
```

使用方式

``` vue
<template>
  <custom-input v-model="text"></custom-input>
</template>
<script>
export default {
  data () {
    return {
      text: ''
    }
  }
}
</script>
```

### 自定义 .sync

在 Vue 中, props 属性是单向数据传输的,父级的prop的更新会向下流动到子组件中, 但是反过来不行。可是有些情况, 我们需要对 `prop` 进行“双向绑定”。上文中, 我们提到了使用 `v-model` 实现双向绑定。但有时候我们希望一个组件可以实现多个数据的“双向绑定”, 而 `v-model` 一个组件只能有一个(Vue3.0 可以有多个), 这时候就需要使用到 `.sync`。

`.sync` 与 `v-model` 的异同

- 相同点：

  - 两者的本质都是语法糖, 目的都是实现组件与外部数据的双向绑定
  - 两个都是通过属性+事件来实现的

- 不同点：

  - 一个组件只能定义一个 `v-model`, 但可以定义多个 `.sync`
  - `v-model` 与 `.sync` 对于的事件名称不同, `v-model` 默认事件为 `input`,可以通过配置 `model` 来修改, `.sync` 事件名称固定为 `update:属性名`

``` vue
<template>
  <div v-show="visible" @click="$_handlerCLick"></div>
</template>

<script>
export default {
  name: 'custom-model',
  props: {
    visible: {
      type: Boolean,
      default: false
    }
  },
  methods: {
    $_handlerCLick() {
      this.$emit('update:visible', false);
    }
  }
}
</script>
```

使用方式

``` vue
<template>
  <custom-model :visible.sync="show"></custom-model>
</template>
<script>
export default {
  data () {
    return {
      show: false
    }
  }
}
</script>
```

### slot

::: tip 提示
二次封装时需要保留原组件中的 slot 插槽, `$slots` 获取到当前组件传入的所有 slot 插槽
:::

``` vue
<template>
  <component-name>
    <template v-for="slot in Object.keys($slots)" :slot="slot">
      <slot :name="slot" />
    </template>
  </component-name>
</template>
```

在开发过程中, 我们还会遇到以下的场景

下面是你封装的一个组件, 可以看到 `a-form-item` 已经有默认的 default 插槽了, 但是我们在使用它时又想自定义 `a-form-item` 中的其他插槽

``` vue
<template>
  <a-form>
    <a-form-item v-for="(item,index) in opt" :key="index">
      <template slot="default">
        <slot :name="item.prop"></slot>
      </template>

      <!-- 以下这种写法不优雅, 比较繁琐, 又有重复的代码 -->
      <template slot="label">
        <slot :name="`${item.prop}Label`"></slot>
      </template>
      <template slot="extra">
        <slot :name="`${item.prop}Extra`"></slot>
      </template>
      <template slot="help">
        <slot :name="`${item.prop}Help`"></slot>
      </template>
      <!-- 之后 a-form-item 如果有添加了其他的 slot,岂不是这里还需要继续添加代码, 这样是不好的 -->
    </a-form-item>
  </a-form>
</template>

<script>
export default {
  props: {
    opt: {
      type: Array,
      default: () => []
    }
  }
}
</script>
```

可以参考以下的方式优化它

``` vue
<template>
  <a-form>
    <a-form-item v-for="(item,index) in opt" :key="index">
      <template slot="default">
        <slot :name="item.prop"></slot>
      </template>

      <template v-for="itemSlot in getItemSlots(Object.keys($slots))" :slot="itemSlot">
        <slot
          :name="getItemSlotName(itemSlot, item.prop)"
          v-if="getItemSlotName(itemSlot,  item.prop)"
        />
      </template>
    </a-form-item>
  </a-form>
</template>

<script>
export default {
  props: {
    opt: {
      type: Object,
      default: () => {}
    }
  },
  methods: {
    getItemSlots (slots) {
      const props = this.getFormPropKey()
      let itemSlots = slots.filter(item => props.indexOf(item) === -1)
      itemSlots = itemSlots.map(item => {
        const tmp = item.split(/(?=[A-Z])/)
        return tmp[tmp.length - 1].toLowerCase()
      })
      return itemSlots
    },
    getItemSlotName(slot) {
      const _slot = slot.replace(/\b\w/g, str => {
        return str.toUpperCase()
      })
      return `${prop}${_slot}`
    },
    getFormPropKey () {
      const { opt } = this
      let props = []
      if (opt.length) {
        props = opt.map(item => item.prop)
      }
      return props
    }
  }
}
</script>
```

### mixins

::: tip 提示
`mixins` 是Vue提供的一种混合机制, 用来更高效的实现组件内容的复用。怎么去理解混入呢, 我觉得和 `Object.assign` , 但实际与 `Object.assign` 又有所不同。
:::

在 `Vue` 中, 一个混入对象可以包含任意组件选项, 但是对于不同的组件选项, 会有不同的合并策略。

1. data 对于 data, 在混入时会进行递归合并, 如果两个属性发生冲突, 则以组件自身为主
2. 生命周期钩子函数

  对于生命周期钩子函数, 混入时会将同名钩子函数加入到一个数组中, 然后在调用时依次执行。混入对象里面的钩子函数会优先于组件的钩子函数执行。如果一个组件混入了多个对象, 对于混入对象里面的同名钩子函数, 将按照数组顺序依次执行, 如下代码:

``` javascript
const mixin1 = {
  created() {
    console.log('我是第一个输出的')
  }
}

const mixin2 = {
  created() {
    console.log('我是第二个输出的')
  }
}

export default {
  mixins: [mixin1, mixin2],
  created() {
    console.log('我是第三个输出的')
  }
}
```

3. 其他选项 对于值为对象的选项, 如 `methods`,`components`,`filter`,`directives`,`props` 等等, 将被合并为同一个对象。两个对象键名冲突时, 取组件对象的键值对。

### class

::: tip 提示
关于统一化 class 命名方式, class 命名时需要添加组织前缀的, 以下的方式比较适用
:::

``` vue
<template>
  <div :class="b('form')"></div>
</template>

<script>
const ELEMENT = '__';
const MODS = '--';

const join = (name, el, symbol) => el ? name + symbol + el : name;

const prefix = (name, mods) => {
  if (typeof mods === 'string') {
    return join(name, mods, MODS);
  }

  if (Array.isArray(mods)) {
    return mods.map(item => prefix(name, item));
  }

  const ret = {};
  Object.keys(mods || {}).forEach(key => {
    ret[name + MODS + key] = mods[key];
  });
  return ret;
};

export default {
  methods: {
    b (el, mods) {
      const { name } = this.$options;

      if (el && typeof el !== 'string') {
        mods = el;
        el = '';
      }
      el = join(name, el, ELEMENT);

      return mods ? [el, prefix(el, mods)] : el;
    }
  }
}
</script>
```

### Provide / Inject

::: tip 提示
这边参考 vue 的官方文档就好： [provide / inject](https://cn.vuejs.org/v2/api/#provide-inject)
:::
