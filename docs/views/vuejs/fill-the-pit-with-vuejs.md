---
author: 星火燎原@vxhly
title: Vue.js 填坑笔记
categories: [vuejs]
tags: [Vue.js, Node.js] 
date: 2017-09-11 16:55:20
---

<Boxx changeTime="10000"/>

::: tip 前言
记录一些使用 vue-cli 脚手架创建项目时, 遇到的一些问题以及解决方案。
:::
<!-- more -->

## vue 1.x 升级到 vue 2.x

首先需要安装 `vue-migration-helper` CLI 工具: 

控制台运行命令: `npm install --global vue-migration-helper` CLI 工具来帮助项目从 Vue 1.x 迁移到 2.x。 它扫描文件以查找特定于 Vue 的代码, 并对需要升级的代码提供详细的警告。 vue-migration-helper 的介绍说明告诉我们它大概能捕获 80% 的升级帮助信息, 而不是全部。所以终端输出的帮助信息并不是完全正确的, 在升级时不要盲目 copy & paste , 还是要根据实际情况去改写。

进入当前的项目: 运行: `vue-migration-helper`

具体升级指南, 请参照[官方文档-从 Vue 1.x 迁移](https://cn.vuejs.org/v2/guide/migration.html)

## 给与 Windows 用户的一条强烈建议

Windows 在执行 `npm install` 期间遇到关于 `node-gyp` 的错误, 那么你很有可能没有在你的系统上安装正确的构建工具。构建工具包括 `Python` 和 `Visual Studio` 等等。

使用 [windows-build-tools](https://github.com/felixrieseberg/windows-build-tools) 来为我们完成大部分烦人的工作。全局安装此工具将依次设置 Visual C++ 软件包、Python 等等

``` bash
npm install --global --production windows-build-tools
```

执行上条命令时注意以管理员权限开启一个 PowerShell 来执行该条命令, 因为自动化工具会自动的添加一些 PATH 变量

如果是 Windows 7 系统, 还需要安装 [. NET Framework 4.5.1](http://www.microsoft.com/en-us/download/details.aspx?id=40773)

还有一些更加恶心的报错信息通常是因为墙的原因, 这里建议安装 `nrm` 来随时切换软件源

``` bash
npm -g nrm
nrm use taobao
```

以下这几种软件源可供选择, 默认使用 npm 官方软件源

``` bash

* npm ---- https://registry.npmjs.org/

  cnpm --- http://r.cnpmjs.org/
  taobao - https://registry.npm.taobao.org/
  nj ----- https://registry.nodejitsu.com/
  rednpm - http://registry.mirror.cqupt.edu.cn/
  npmMirror  https://skimdb.npmjs.com/registry/
  edunpm - http://registry.enpmjs.org/
```

## 设置代理与跨域

### 开发时设置

如果你的后端 API 服务是 Express 提供或者是 Thinkjs 再或者是 koa2 等等, 当你请求数据时就会面临着跨域请求问题

执行 `npm run dev` , 你会发现会报一个错误: `vue-resource.common.js?e289:1071 POST http://localhost:8080/api/use... 404 (Not Found)` 。这是由于直接访问 `8080` 端口, 是访问不到的, 所以这里需要设置一下代理转发映射.

项目根目录下的 config 文件夹中有一个 proxyTable 参数, 用来设置地址映射表, 可以添加到开发时配置（dev）中

``` bash
|-- config
  |-- dev.env.js
  |-- index.js
  |-- prod.env.js
```

#### config/index.js

``` javascript
dev: {
    // ...
    proxyTable: {
        '/api': {
            target: 'http://127.0.0.1:3000/api/',
            changeOrigin: true,
            pathRewrite: {
                '^/api': ''
            }
        }
    },
    // ...
}
```

添加以上代码之后, 请求 `/api` 时就代表 `http://127.0.0.1:3000/api/` (这里要写 ip, 不要写 localhost), changeOrigin 参数接收一个布尔值, 如果为 true, 这样就不会有跨域问题了。

更多接口参数配置, 请参考 <https://github.com/chimurai/http-proxy-middleware#options>

webpack 接口配置文档 <https://webpack.js.org/configuration/dev-server/#devserver-proxy>

### 正式上线时设置

``` bash
|-- src
  |-- axios
    |-- index.js
  |-- config.js
```

正式上线时, 不推荐使用上一个方案, 这里推荐使用 axios 进行转发

#### src/config.js

``` javascript
export default {
    serverUrl: "http://127.0.0.1:3000/"
};
```

#### src/axios/index.js

``` javascript
import axios from "axios";
import config from "@/config";

// 设置全局 axio s默认值
axios.defaults.baseURL = config.serverUrl;
axios.defaults.timeout = 5000; // 5000的超时验证
axios.defaults.headers.post["Content-Type"] = "application/jsoncharset=UTF-8";

// 创建一个 axios 实例
const instance = axios.create();
instance.defaults.headers.post["Content-Type"] =
    "application/jsoncharset=UTF-8";

axios.interceptors.request.use = instance.interceptors.request.use;

export async function postDate(username, password) {
    try {
        const response = await fetch.post("/postDate", {
            username,
            password
        });
        return response.data;
    } catch (err) {
        console.log("message", err);
        if (err.response) {
            throw Error(err.response.data.message);
        }
        throw err;
    }
}
```

## 父子组件

``` bash
|-- src
  |-- components
    |-- HerderBar.vue
    |-- FooterBar.vue
  |-- pages
    |-- Home.vue
```

假如你的 components 目录下有 HerderBar.vue 和 FooterBar.vue 这两个子组件, 而 Home.vue 要引用这两组件, 那么下面这种写法可以完成该需求

### src/pages/Home.vue

``` html
<template>
    <div>
        <header-bar></header-bar>
        <!-- ... ... -->
        <footer-bar></footer-bar>
    </div>
</template>

<script>
    import HeaderBar from '@/components/HeaderBar'
    import FooterBar from '@/components/FooterBar'

    export default {
        name: 'Home',
        components: {
            HeaderBar,
            FooterBar
        }
    }
</script>
```

## 图标库

市面上用的比较广泛的图标库有两个, 一是[阿里巴巴矢量图标库](http://iconfont.cn/), 其有上百万图标共程序员选择, 自定义比较强；二是[Font Awesome](http://fontawesome.io/), 该图标库虽没有上百万图标, 但也受到大部门程序员喜爱。

很多人在写 Vue 项目时, 前端 UI 框架都喜欢使用 Element UI, 但是该 UI 框架默认提供的图标库实现是少之又少, 但是该 UI 框架允许我们引入第三方图标库

### iconfont

这个引入就非常简单了, 在 iconfont 网站上有提供离线版和在线版, 看自己的意愿, 然后在 index.html 里使用 style 标签引入即可。

### fontawesome

参考代码[element-font-awesome](https://github.com/ElementUI/element-font-awesome)

使用 less 时, 别忘了安装 npm 依赖

``` bash
npm install -S less less-loader
```

#### 目录结构

``` bash
|-- src
  |-- font.less
  |-- main.js
```

#### src/main.js

``` javascript
import "./font.less";
```

#### src/font.less

``` javascript
[class ^= "el-icon-fa"], [class *= " el-icon-fa"] {
    display: inline - block;
    font: normal normal normal 14 px / 1 FontAwesome!important;
    font - size: inherit;
    text - rendering: auto; -
    webkit - font - smoothing: antialiased; -
    moz - osx - font - smoothing: grayscale;
};

@import url("../node_modules/font-awesome/less/font-awesome");
@fa - css - prefix: el - icon - fa;
```

## sass

对于 css 的预编译器, 个人比较喜欢 sass 的, 在使用 sass 时仍然需要添加 npm 依赖

``` bash
npm install --save node-sass sass-lodaer
```

### 目录结构

``` bash
|-- src
  |-- assets
    |-- scss
      |-- _public.scss
      |-- index.scss
  |-- App.vue
```

#### src/App.vue

``` html
<style lang="scss">
    @import './assets/scss/index';
</style>

<template>
    <div id="app">
    </div>
</template>

<script>
    export default {
        name: 'app'
    }
</script>
```

#### src/assets/scss/index.scss

``` css
@import "public";
```

#### src/assets/scss/public.scss

``` css
#app {
    font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB",
        "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
```

## vuex

### 目录结构

``` bash
|-- src
  |-- store
    |-- modules
      |-- ... ...
    |-- actions.js      # 根级别的 action =>
    |-- getters.js      # 根级别的 mutation =>
    |-- index.js        # 我们组装模块并导出 store 的地方
    |-- types.js        # 根级别的 type => 状态
  |-- main.js
```

#### src/main.js

``` javascript
import Vue from "vue";
import App from "./App";

import store from "./store";

Vue.config.productionTip = false;

/* eslint-disable no-new */
new Vue({
    el: "#app",
    store,
    template: "<App/>",
    components: {
        App
    }
});
```

## vue-router

### 目录结构

``` bash
|-- router
  |-- axios
    |-- index.js
  |-- main.js
```

#### src/main.js

``` javascript
import Vue from "vue";
import App from "./App";
import router from "./router";

Vue.config.productionTip = false;

/* eslint-disable no-new */
new Vue({
    el: "#app",
    router,
    template: "<App/>",
    components: {
        App
    }
});
```

::: tip
如果遇上重复路由的时候, 可以使用以下方法进行封装
:::

``` javascript
import Vue from 'vue'
import Router from 'vue-router'

// hack router push callback
const originalPush = Router.prototype.push
Router.prototype.push = function push (location, onResolve, onReject) {
  if (onResolve || onReject) return originalPush.call(this, location, onResolve, onReject)
  return originalPush.call(this, location).catch(err => err)
}

const constantRouterMap = [
  {

  }
]

Vue.use(Router)

const router = new Router({
  mode: 'history',
  routes: constantRouterMap
})

// 重写addRoutes方法, 解决重复添加路由问题
router.$addRoutes = (params) => {
  router.matcher = new Router({
    mode: 'history',
    routes: constantRouterMap
  }).matcher
  router.addRoutes(params)
}

export default router

```

### HTML5 History 模式

下面这一代码片段是使用 `vue-cli` 下载的模板写法, 但是这种写法会使你的 URL 变成 `http://localhost:8080/#/`

``` javascript
import Vue from "vue";
import Router from "vue-router";
// ... ...

Vue.use(Router);

export default new Router({
    routes: [{
        path: "/"
        // ... ...
    }]
});
```

对于强迫症的人来说, 这样的 URL 非常丑, 这就需要开启 `HTML5 History` 模式, 更具体的说明请看官方文档 [vue-router HTML5 History 模式](https://router.vuejs.org/zh-cn/essentials/history-mode.html)

#### router/index.js

``` javascript
import Vue from "vue";
import Router from "vue-router";
// ... ...

Vue.use(Router);

const router = new VueRouter({
    mode: "history",
    routes: [{
        path: "/"
        // ... ...
    }]
});

export default router;
```

### 路由拦截

对于进入某些页面需要进行登录验证, 那么就需要设置路由拦截, `vue-router` 官方文档称之为导航钩子, 具体请看官方文档 [vue-router 导航钩子](https://router.vuejs.org/zh-cn/advanced/navigation-guards.html)

实际上在进行路由拦截时需要进行数据验证, 当验证通过时方能允许其通过该路由, 该验证数据通常会存储在 `vuex` 的 `state` 中, 或者会存储在 `Local Storage` , 再或者 `Session Storage` , 无论存储在哪里, `vue-router` 配置文件能够正确访问到即可, 当然验证程序就需要后端服务 API 提供了

#### router/index.js

``` javascript
import Vue from "vue";
import Router from "vue-router";
import store from "@/store";
// ... ...

Vue.use(Router);

const router = new VueRouter({
    mode: "history",
    routes: [{
        path: "/",
        // ... ...
        meta: {
            requireAuth: true // 添加该字段,表示进入这个路由是需要进行验证的
        }
    }]
});

router.beforeEach((to, from, next) => {
    if (to.matched.some(r => r.meta.requireAuth)) {
        // 判断该路由是否需要登录权限
        if (store.state.token) {
            // 通过 vuex state 获取当前的 token 是否存在
            next();
        } else {
            next({
                path: "/login", // 验证失败,将会跳转到该路由
                query: {
                    redirect: to.fullPath
                } // 将跳转的路由 path 作为参数,登录成功后跳转到该路由
            });
        }
    } else {
        next();
    }
});

export default router;
```

## axios

自从 Vue.js 更新至 2.x 版本之后, 官方就不再使用 `vue-resource` , 替而代之的是 [axios](https://github.com/mzabriskie/axios)

### 目录结构

``` bash
|-- src
  |-- axios
    |-- index.js
  |-- pages
    |-- Home.vue
  |-- main.js
```

#### src/main.js

``` javascript
import Vue from "vue";
import App from "./App";

import api from "./axios";
Vue.prototype.$api = api;

Vue.config.productionTip = false;

/* eslint-disable no-new */
new Vue({
    el: "#app",
    template: "<App/>",
    components: {
        App
    }
});
```

### axios 拦截

使用 vue-router 进行路由拦截是不够的, 当然也是需要数据验证的, 更加详细的说明以及例子请移步 [【vue+axios】一个项目学会前端实现登录拦截](https://segmentfault.com/a/1190000008383094?_ea=1639495)

#### axios/index.js

``` javascript
import axios from "axios";
import store from "../store";
import router from "../router";

// 设置全局axios默认值
axios.defaults.timeout = 5000; // 5000的超时验证
axios.defaults.headers.post["Content-Type"] = "application/jsoncharset=UTF-8";

// 创建一个axios实例
const instance = axios.create();
instance.defaults.headers.post["Content-Type"] =
    "application/jsoncharset=UTF-8";

axios.interceptors.request.use = instance.interceptors.request.use;

// http request 拦截器
instance.interceptors.request.use(
    config => {

        if (store.state.token) {
            config.headers.Authorization = `token ${store.state.token}`.replace(
                /(^")|("$)/g,
                ""
            );
        }
        return config;

    },
    error => {

        return Promise.reject(error);

    }
);

// http response 拦截器
instance.interceptors.response.use(
    response => {

        return response;

    },
    error => {

        if (error.response) {
            switch (error.response.status) {
                case 401:
                    store.dispatch("UserLogout");
                    router.replace({
                        path: "login",
                        query: {
                            redirect: router.currentRoute.fullPath
                        }
                    });
            }
        }
        return Promise.reject(error.response);

    }
);

export default {
    // POST
    PostData(data) {

        return instance.post("/api/postData", data);

    },
    // GET
    GetData() {

        return instance.get("/api/user/getData");

    }
};
```

### 在 Vue 组件内调用

#### src/pages/Home.vue

``` html
<template>
    <div>
        <!-- ... ... -->
    </div>
</template>

<script>
    export default {
        name: 'Home',
        data: {
            return {
                fromData: [
                    // ... ...
                ]
            }
        },
        methods: {
            post() {
                const opt = this.fromData
                this.$api.PostData(opt)
                    .then(({
                        data
                    }) => {
                        // ... ...
                    })
                    .catch((err) => {
                        console.log(err)
                    })
            }
        }
        created() {
            this.$api.GetData()
                .then(({
                    data
                }) => {
                    // ... ...
                })
                .catch((err) => {
                    console.log(err)
                })
        }
    }
</script>
```

## 规范性

> 这里只举例 `.vue` 文件 script 模块中, 各个函数的书写顺序, 更多的规范性请查询文档 [风格指南](http://cn.vuejs.org/v2/style-guide/)

``` javascript
export default {
    name: "", // 组件名字
    props: [
        // 继承父组件数据
        ""
    ],

    data() {
        return {
            // 该组件数据
        };
    },
    watch: {
        // 于观察 Vue 实例上的数据变动。对应一个对象,键是观察表达式,值是对应回调。值也可以是方法名,或者是对象,包含选项
    },
    computed: {
        // 相当于属性的一个实时计算,如果实时计算里关联了对象,那么当对象的某个值改变的时候,同事会出发实时计算
    },
    methods: {
        // 该组件的方法
    },
    filter: {
        // 过滤器
    },
    beforeRouteUpdate(to, from, next) {
        // 在当前路由改变,但是该组件被复用时调用
    },
    created() {
        // 在 vue 1.0 中起到初始化数据作用,在 vue 2.0 之后推荐改用 computed
    },
    mounted() {
        // 在这发起后端请求,拿回数据,配合路由钩子做一些事情
    },
    components: {
        // 组件
    }
};
```

## 在 vue 中引入 markdown

### 安装相关依赖

``` bash
npm i -D text-loader
npm i -S marked github-markdown-css
```

## 配置 webpack

 `build/webpack.base.conf.js`

``` javascript
module.exports = {
    module: {
        rules: [{
                test: /.md$/,
                loader: 'text-loader'
            }
        }
    }
}
```

 `vue.config.js`

``` javascript
module.exports = {
    chainWebpack: config => {
        config.module
            .rule('test')
            .test(/\.md$/)
            .use('test-loader')
            .loader('test-loader')
    }
}
```

### 编写 vue 文件

``` html
<template lang="html">
    <div>

        <div class="article markdown-body" v-html="compiledMarkdown">
        </div>

    </div>
</template>

<script>
    import marked from 'marked'
    import versionLog from './versionLog.md'
    import 'github-markdown-css/github-markdown.css'

    export default {
        name: 'VersionLog',
        data() {
            return {
                context: versionLog
            }
        },
        computed: {
            compiledMarkdown() {
                return marked(this.context, {
                    sanitize: true
                })
            }
        }
    }
</script>
```

## 基于 Markdown 的幻灯片

导入 `text-loader` 上一节已经说明过了, 这里就不重复了

> [github vue-mark-display](https://github.com/Jinjiang/vue-mark-display)

``` bash
npm i vue-mark-display
```

### usage

 `demo.md`

``` markdown
# Hello World
----
This is Vue Mark Display
```

 `demo.vue`

``` html
<template>
    <mark-display :markdown="markdown" @title="setTitle" keyboard-ctrl url-hash-ctrl auto-font-size auto-blank-target></mark-display>
</template>

<script>
    import MarkDisplay from "vue-mark-display";
    import markdown from "./demo.md";

    export default {
        components: {
            MarkDisplay
        },
        data() {
            return {
                markdown
            };
        },
        methods: {
            setTitle({
                title
            }) {
                document.title = title;
            }
        }
    };
</script>
```

## 关于父组件通过 v-on 接收子组件多个参数

写组件的时候遇到一个需求, 我需要在子组件向父组件传递信息

``` javascript
this.$emit('myEvent', 信息1, 信息2)
```

在父组件使用 `v-on` 来接收

``` html
<my-component @myEvent="handler" />
```

这样就可以接收到子组件传递的信息1和信息2, easy。

``` javascript
export default {
    methods: {
        handler(param1, param2) {
            console.log(param1, param2) // => 信息1, 信息2
        }
    }
}
```

但我需要在内联语句中传递一个额外参数, 平时子组件只附带一个参数的时候, 可以使用 `$event`

``` html
<my-component @myEvent="handler('extra parameter', $event)" />
```

但是 $event 只接收第一个参数, 也就是这么写只能接收到信息1

``` javascript
export default {
    methods: {
        handler(extra, param1, param2) {
            console.log(extra, param1, param2) // => 'extra parameter', 信息1, undefined
        }
    }
}
```

最后找到了一个解决办法

``` html
<my-component @myEvent="handler('extra parameter', arguments[0], arguments[1])" />
```

可以获取到参数了

``` javascript
export default {
    methods: {
        handler(extra, param1, param2) {
            console.log(extra, param1, param2) // => 'extra parameter', 信息1, 信息2
        }
    }
}
```


因为实际上这里的 `$event` 就是 `arguments[0]`, 下面这一段现在的 vue 文档上好像已经找不到了, 但是 `arguments` 还是可以用的

![fix-vue-1](https://vxhly.github.io/assets/fit-vue-1.png)

多提一句, 当 `v-on` 的参数是 `dom` 事件时, `$event` 代表的是原生的 `event` 事件