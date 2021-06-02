---
author: 星火燎原@vxhly
title: Vue 项目性能优化(首屏优化)
categories: [vuejs]
tags: [Vue.js, Node.js]
date: 2018-10-24 13:51:50
---

<Boxx changeTime="10000"/>

::: tip 前言
随着 Vue 项目越做越大, 依赖的第三方 npm 包越来越多, 构建之后的文件也会越来越大, 尤其是 vendor.js, 甚至会达到 2M 左右, 为了解决这个问题, 需要做一些探索, 在几乎不需要改动业务代码的情况下, 找到了几种有明显效果的优化方案 ---- CDN + Gzip。
:::
<!-- more -->

## 好景不长

先来看看没有任何优化的项目需要优化些什么

### 首先是文件大小

这是一个通过 Vue-cli 3.0 构建的 demo 项目, 先执行一下 `npm run build` , 当然也可以通过 `vue ui` 的 GUI 界面进行编译

![vue-performance](https://vxhly.github.io/assets/vue-performance-1.png)

可以看到这个默认项目的各个 JS 文件还可以, 不算是比较大的

### 加载速度

禁用浏览器缓存, 网速限定为 Fast 3G 下的 Network 图(运行在本地的nginx服务器上)

![vue-performance](https://vxhly.github.io/assets/vue-performance-2.png)

嗯~ , 故意调了个比较慢的网络环境, 可以看到这边的加载速度真的慢到极致了

## CDN

首先是 CDN 优化, 把比较经常使用的依赖, 像 `vue` , `vue-router` , `vuex` , `element-ui` , `axios` , `loadsh` 直接使用 CDN 来分发, 这边选用的 CDN 服务是 [bootcdn](https://www.bootcdn.cn/)

### vue.config.js

``` javascript
const externals = {
    'vue': 'Vue',
    'vue-router': 'VueRouter',
    'vuex': 'Vuex',
    'axios': 'axios',
    'element-ui': 'ELEMENT',
    '_': 'lodash'
}

const cdn = {
    // 开发环境
    dev: {
        css: [
            'https://cdnjs.cloudflare.com/ajax/libs/element-ui/2.4.7/theme-chalk/index.css'
        ],
        js: []
    },
    // 生产环境
    build: {
        css: [
            'https://cdnjs.cloudflare.com/ajax/libs/element-ui/2.4.7/theme-chalk/index.css'
        ],
        js: [
            'https://cdn.bootcss.com/vue/2.5.17/vue.min.js',
            'https://cdn.bootcss.com/vue-router/3.0.1/vue-router.min.js',
            'https://cdn.bootcss.com/vuex/3.0.1/vuex.min.js',
            'https://cdn.bootcss.com/axios/0.18.0/axios.min.js',
            'https://cdn.bootcss.com/lodash.js/4.17.11/lodash.min.js',
            'https://cdnjs.cloudflare.com/ajax/libs/element-ui/2.4.7/index.js'
        ]
    }
}

module.exports = {
    productionSourceMap: false,
    configureWebpack: config => {
        const myConfig = {}
        if (process.env.NODE_ENV === 'production') {
            // 1\. 生产环境 npm 包转 CDN
            myConfig.externals = externals
        }
        if (process.env.NODE_ENV === 'development') {
            /**
             * 关闭host check,方便使用ngrok之类的内网转发工具
             */
            myConfig.devServer = {
                disableHostCheck: true
            }
        }
        return myConfig
    },
    chainWebpack: config => {
        /**
         * 删除懒加载模块的prefetch,降低带宽压力
         * https://cli.vuejs.org/zh/guide/html-and-static-assets.html#prefetch
         * 而且预渲染时生成的prefetch标签是modern版本的,低版本浏览器是不需要的
         */
        config.plugins
            .delete('prefetch')
        /**
         * 添加CDN参数到htmlWebpackPlugin配置中
         */
        config
            .plugin('html')
            .tap(args => {
                if (process.env.NODE_ENV === 'production') {
                    args[0].cdn = cdn.build
                }
                if (process.env.NODE_ENV === 'development') {
                    args[0].cdn = cdn.dev
                }
                return args
            })
    }
}
```

### index.html

``` html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width,initial-scale=1.0">
    <link rel="icon" href="<%= BASE_URL %>favicon.ico">
    <title>demo</title>
    <!-- 使用CDN加速的CSS文件,配置在vue.config.js下 -->
    <% for (var i in htmlWebpackPlugin.options.cdn&&htmlWebpackPlugin.options.cdn.css) { %>
    <link href="<%= htmlWebpackPlugin.options.cdn.css[i] %>" rel="preload" as="style">
    <link href="<%= htmlWebpackPlugin.options.cdn.css[i] %>" rel="stylesheet">
    <% } %>
</head>

<body>
    <noscript>
        <strong>We're sorry but demo doesn't work properly without JavaScript enabled. Please enable it to continue.</strong>
    </noscript>
    <div id="app"></div>
    <!-- 使用CDN加速的JS文件,配置在vue.config.js下 -->
    <% for (var i in htmlWebpackPlugin.options.cdn&&htmlWebpackPlugin.options.cdn.js) { %>
    <script src="<%= htmlWebpackPlugin.options.cdn.js[i] %>"></script>
    <% } %>
    <!-- built files will be auto injected -->
</body>

</html>
```

### 优化对比

再来看看使用了 CDN 之后的编译大小

![vue-performance](https://vxhly.github.io/assets/vue-performance-3.png)

天哪!!! `chunk-vendors` 这个文件小了不是一点点

`chunk-vendors` 该文件打包了 vue, vuex, vue-router 等依赖资源, 这也是这个文件之所以大的原因

因为是优化产品模式下的代码, 这里就构建一个 Nginx 服务器来查看加载的情况

构建的话, 自行百度, 比较懒的推荐 [upupw](https://www.baidu.com/link?url=A3pEi1xHX8pn-3a6xug88gaY_LZPuY9nlZQ51dGXsNW&wd=&eqid=e847c04100012f59000000065bd1d0be) , 当然闲的无聊的话可以写一个 express 当作本地服务器


## Gzip

使用 Gzip 两个明显的好处, 一是可以减少存储空间, 二是通过网络传输文件时, 可以减少传输的时间。

开启 gzip 的方式主要是通过修改服务器配置, 这里贴出 nginx 的配置方式

``` 
http {
    gzip on;
    gzip_static on;
    gzip_min_length 1024;
    gzip_buffers 4 16 k;
    gzip_comp_level 2;
    gzip_types text / plain application / javascript application / x - javascript text / css application / xml text / javascript application / x - httpd - php application / vnd.ms - fontobject font / ttf font / opentype font / x - woff image / svg + xml;
    gzip_vary off;
    gzip_disable "MSIE [1-6]\.";
}
```

当然只有 nginx 这边开启了 gzip 是不够的, webpack 工具提供了 productionGzip 来开启 gzip

在此之前还需安装一个依赖项

``` bash
npm i -D compression-webpack-plugin
```

补全 `vue.config.js`

``` javascript
const CompressionWebpackPlugin = require('compression-webpack-plugin')

const externals = {
    'vue': 'Vue',
    'vue-router': 'VueRouter',
    'vuex': 'Vuex',
    'axios': 'axios',
    'element-ui': 'ELEMENT',
    '_': 'lodash'
}

const cdn = {
    // 开发环境
    dev: {
        css: [
            'https://cdnjs.cloudflare.com/ajax/libs/element-ui/2.4.7/theme-chalk/index.css'
        ],
        js: []
    },
    // 生产环境
    build: {
        css: [
            'https://cdnjs.cloudflare.com/ajax/libs/element-ui/2.4.7/theme-chalk/index.css'
        ],
        js: [
            'https://cdn.bootcss.com/vue/2.5.17/vue.min.js',
            'https://cdn.bootcss.com/vue-router/3.0.1/vue-router.min.js',
            'https://cdn.bootcss.com/vuex/3.0.1/vuex.min.js',
            'https://cdn.bootcss.com/axios/0.18.0/axios.min.js',
            'https://cdn.bootcss.com/lodash.js/4.17.11/lodash.min.js',
            'https://cdnjs.cloudflare.com/ajax/libs/element-ui/2.4.7/index.js'
        ]
    }
}

// 是否使用gzip
const productionGzip = true
// 需要gzip压缩的文件后缀
const productionGzipExtensions = ['js', 'css']

module.exports = {
        productionSourceMap: false,
        configureWebpack: config => {
            const myConfig = {}
            if (process.env.NODE_ENV === 'production') {
                // 1\. 生产环境 npm 包转 CDN
                myConfig.externals = externals
                myConfig.plugins = []

                // 构建时开启gzip,降低服务器压缩对CPU资源的占用,服务器也要相应开启gzip
                productionGzip && myConfig.plugins.push(
                    new CompressionWebpackPlugin({
                        test: new RegExp('\\.(' + productionGzipExtensions.join('|') + ')),
                            threshold: 8192,
                            minRatio: 0.8
                        }))
                }
                if (process.env.NODE_ENV === 'development') {
                    /**
                     * 关闭host check,方便使用ngrok之类的内网转发工具
                     */
                    myConfig.devServer = {
                        disableHostCheck: true
                    }
                }
                return myConfig
            },
            chainWebpack: config => {
                /**
                 * 删除懒加载模块的prefetch,降低带宽压力
                 * https://cli.vuejs.org/zh/guide/html-and-static-assets.html#prefetch
                 * 而且预渲染时生成的prefetch标签是modern版本的,低版本浏览器是不需要的
                 */
                config.plugins
                    .delete('prefetch')
                /**
                 * 添加CDN参数到htmlWebpackPlugin配置中
                 */
                config
                    .plugin('html')
                    .tap(args => {
                        if (process.env.NODE_ENV === 'production') {
                            args[0].cdn = cdn.build
                        }
                        if (process.env.NODE_ENV === 'development') {
                            args[0].cdn = cdn.dev
                        }
                        return args
                    })
            }
        }
```
