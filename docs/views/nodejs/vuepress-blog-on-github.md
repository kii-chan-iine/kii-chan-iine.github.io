---
author: 星火燎原@vxhly
title: 使用 vuepress 更新我的博客
categories: [nodejs]
tags: [Liunx, Node.js, VuePress]
date: 2019-10-28 16:14:34
---

<Boxx changeTime="10000"/>

::: tip 前言
嗯～, 没错, 博客又迎来一次重大的更新, 此次更新不再采用 hexo 这个静态博客解决方案, 将采用 vuepress 这一静态博客解决方案。本博文将会从搭建博客到部署上线进行较详细的介绍。
:::
<!-- more -->

## 开始你的博客

### 前期准备

[VuePress 官方文档](https://vuepress.vuejs.org/zh/plugin/), 不管怎样这份文档多多少少都得阅读一遍吧

[vuepress-theme-reco 博客主题文档](https://vuepress-theme-reco.recoluan.com/), 如果你用的博客主题和我的一样的话, 这份文档也该阅读一遍吧, 如果不是的话或是你找到了更加优秀主题的话请你私信给我

### 安装博客主题 CLI

``` bash
npm i -g vuepress # 如果可以的话, 建议您 vuepress 可以全局安装一下依赖, 当然不装也没啥大问题
npm install @vuepress-reco/theme-cli -g
``` 

### 创建一份基础的 DEMO

命令行会已一问一答的方式指引你创建一份新的博客, 按照提示进行配置即可

``` bash
theme-cli init my-blog
``` 

### 安装依赖

``` bash
cd my-blog
npm install
``` 

### 预览

这一步可分为开发模式预览和线上模式预览。 开发模式预览指未发布前自己个儿边写博文边看效果（PS：可自动刷新浏览器）；线上模式预览指准备发布了,打包之后查看线上的效果

#### 开发模式预览

``` bash
npm dev # 打开浏览器即可浏览, 网址: 127.0.0.1:8080 
```

#### 线上模式预览

``` bash
npm build
server public # 需要提前安装 serve（开启本地的一个服务器, 执行命令 npm i -g serve）
```

## 新的一篇文章

::: tip 提示
该博客主题 CLI 没有像 hexo 的 CLI 一样提供创建文章的功能, 所以在此自己就撸了个 CLI 工具, 有需要的向下看
:::

上 GitHub 地址: [vuepress-article-cli](https://github.com/vxhly/vuepress-article-cli)

### 安装 CLI 工具

``` bash
npm i vuepress-article-cli -g
```

### 使用

新建一篇文章

``` bash
va-cli new
```

![nodejs-cli](https://vxhly.github.io/assets/nodejs-cli-1.png)

批量删除文章

``` bash
va-cli del
```

![nodejs-cli](https://vxhly.github.io/assets/nodejs-cli-2.png)

创建模版文件

``` bash
va-cli tmp
```

![nodejs-cli](https://vxhly.github.io/assets/nodejs-cli-3.png)

## 进阶配置

经过上面的一系列折腾, 就能开始使用 VuePress 这个博客了

::: tip 提示
以下只提供 [vuepress-theme-reco 博客主题文档](https://vuepress-theme-reco.recoluan.com/) 中未提及的配置项
:::

### last-updated

::: warning 注意
无需重新按照该插件, 直接进入配置即可, 这个插件也可不进行配置, 至于为啥要在本博文中写出（PS： 默认显示的时间格式可读性比较差, 所以需要进行格式化一下）
:::

显示最后的更新时间插件, 先上插件文档 [@vuepress/last-updated](https://vuepress.vuejs.org/zh/plugin/official/plugin-last-updated.html)

再上配置, `/.vuepress/config.js` 

``` javascript
module.exports = {
    plugins: [
        [
            '@vuepress/last-updated',
            {
                transformer: (timestamp, lang) => {
                    // 不要忘了安装 moment
                    // moment 时间格式化文档戳这里 http://momentjs.cn/
                    const moment = require('moment')
                    moment.locale(lang)
                    return moment(timestamp).fromNow()
                }
            }
        ]
    ]
}
```

### medium-zoom

::: warning 注意
无需重新按照该插件, 直接进入配置即可
:::

插件大图预览, 先上插件文档 [@vuepress/medium-zoom](https://vuepress.vuejs.org/zh/plugin/official/plugin-pwa.html)

再上配置, `/.vuepress/config.js` 

``` javascript
module.exports = {
    plugins: [
        ['@vuepress/medium-zoom': {
            selector: ".page img",
            options: {
                margin: 16,
                background: "#202124de",
                scrollOffset: 0
            }
        }]
    ]
}
```

### pwa

::: warning 注意
你的 VPS 服务器必须支持 HTTPS, 否则 PWA 将会失效, 当然如果你和我一样使用 GitHub 的 pages 服务的话, 就无须担心这一点<br>
必须重新依赖该插件, 否则尽管配置了也不会加载 pwa
:::

插件 PWA（PS：关于 PWA 的知识戳这里 [PWA](https://lavas.baidu.com/pwa)）, 先上插件文档 [@vuepress/pwa](https://vuepress.vuejs.org/zh/plugin/official/plugin-pwa.html)

按照文档中的步骤来, 第一：在 `.vuepress/public` 提供 `Manifest` 和 `icons` 

一份 `manifest.json` 的例子

``` json
{
  "name": "Coding and Fixing",
  "short_name": "Coding and Fixing",
  "start_url": "./index.html",
  "display": "fullscreen",
  "background_color": "#000000",
  "theme_color": "#22979b",
  "icons": [
    {
      "src": "./icons/android-chrome-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "./icons/android-chrome-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    },
    {
      "src": "./icons/apple-touch-icon-60x60.png",
      "sizes": "60x60",
      "type": "image/png"
    },
    {
      "src": "./icons/apple-touch-icon-76x76.png",
      "sizes": "76x76",
      "type": "image/png"
    },
    {
      "src": "./icons/apple-touch-icon-120x120.png",
      "sizes": "120x120",
      "type": "image/png"
    },
    {
      "src": "./icons/apple-touch-icon-152x152.png",
      "sizes": "152x152",
      "type": "image/png"
    },
    {
      "src": "./icons/apple-touch-icon-180x180.png",
      "sizes": "180x180",
      "type": "image/png"
    },
    {
      "src": "./icons/msapplication-icon-144x144.png",
      "sizes": "144x144",
      "type": "image/png"
    }
  ]
}
```

接下来就是 `icons` 了, 根据你配置的 `manifest.json` 进行裁剪图标即可

第二：在 `.vuepress/config.js` 中添加配置（PS：该步骤略过, 文档中已经提供了）

部署之后, 你会神奇的发现在你浏览器地址栏的右边出现了一个 `+` 号的图标, 小主可点击本博客上的 `+` 号的图标查看效果

### copyright

::: warning 注意
必须重新依赖该插件, 否则尽管配置了也不会加载
:::

指定页面复制行为的插件, 先上插件文档 [vuepress-plugin-copyright](https://vuepress.github.io/zh/plugins/copyright/#vuepress-plugin-copyright)

再上配置, `/.vuepress/config.js` 

``` javascript
module.exports = {
    plugins: [
        ["copyright", {
            // 更多配置项以及配置项说明上 https://github.com/vuepress/vuepress-plugin-copyright 查看
            "noCopy": true,
            "minLength": 150,
            "disabled": false,
            "noSelect": true
        }]
    ]
}
```

### clipboard-copy

::: tip 提示
此插件纯粹是自己为了配合上一个插件自己写的一份一键复制插件, 暂未发布, 下面贴出代码供米娜桑参考（PS：效果米娜桑可以对下面的代码进行一键复制即可看到）
:::

`clipboard-copy.js` 

``` javascript
const {
    resolve
} = require('path')

module.exports = (options, context) => ({
    define: {
        COPY_SELECTOR: options.copy_selector || 'div[class*="language-"] pre',
        COPY_ASIDE: options.copy_selector || 'div[class*="aside-code"] aside'
    },
    clientRootMixin: resolve(__dirname, 'clientRootMixin.js')
})
```

`clientRootMixin.js` 

``` javascript
import './style.css'

import {
    message
} from 'ant-design-vue'
import 'ant-design-vue/lib/message/style/css'
message.config({
    top: `100px` ,
})

export default {
    data() {
        return {
            zoom: null
        }
    },

    mounted() {
        this.updateCopy()
    },

    updated() {
        this.updateCopy()
    },

    methods: {
        updateCopy() {
            setTimeout(() => {
                document.querySelectorAll(COPY_SELECTOR).forEach(this.generateCopyButton)
                document.querySelectorAll(COPY_ASIDE).forEach(this.generateCopyButton)
            }, 1000)
        },
        generateCopyButton: function(parent) {
            if (parent.classList.contains('codecopy-enabled')) return
            const copyElement = document.createElement('span')
            copyElement.className = 'code-copy iconfont reco-api'
            copyElement.title = 'Click to Copy to Clipboard'
            copyElement.addEventListener('click', () => {
                this.copyToClipboard(parent.innerText)
            })
            parent.appendChild(copyElement)
            parent.classList.add('codecopy-enabled')
        },
        copyToClipboard: function(str) {
            const el = document.createElement('textarea')
            el.value = str
            el.setAttribute('readonly', '')
            el.style.position = 'absolute'
            el.style.left = '-9999px'
            document.body.appendChild(el)
            const selected =
                document.getSelection().rangeCount > 0 ?
                document.getSelection().getRangeAt(0) :
                false
            el.select()
            document.execCommand('copy')
            message.success('复制成功,快去粘贴使用吧 !!!')
            document.body.removeChild(el)
            if (selected) {
                document.getSelection().removeAllRanges()
                document.getSelection().addRange(selected)
            }
        }
    }
}
```

`style.css` 

``` css
.code-copy {
    color: #aaa;
    display: inline-block;
    cursor: pointer;
}

div[class*="language-"] pre .code-copy,
div[class*="aside-code"] aside .code-copy {
    position: absolute;
    z-index: 1000;
    top: 5px;
    right: 40px;
    opacity: 0;
    font-size: 16px;
}

div[class*="language-"] pre:hover .code-copy,
div[class*="aside-code"] aside:hover .code-copy {
    opacity: 1;
}

.content pre,
.content pre[class*=language-] {
    overflow-y: hidden;
}

div[class*="language-"] pre,
div[class*="language-"] pre[class*="language-"] {
    position: static !important;
}

div[class~="language-text"]:before {
    content: "text";
}

div[class~="language-yml"]:before {
    content: "yml";
}

div[class*="language-"] pre {
    user-select: text;
}

p code {
    user-select: all;
}
```

### nest

::: tip 提示
自己个写的对接 [canvas-nest.js](https://github.com/hustcc/canvas-nest.js) 的一个插件
:::

页面 canvas-nest 背景插件, 先上插件文档 [vuepress-plugin-nest](https://github.com/vxhly/vuepress-plugin-nest)

再上配置, `/.vuepress/config.js` 

``` javascript
module.exports = {
    plugins: [
        ["nest", {
            // 更多配置项以及配置项说明上 https://github.com/vxhly/vuepress-plugin-nest 查看
            "color": "255,0,255",
            "count": 100,
            "opacity": 0.7
        }],
    ]
}
```

### sitemap

::: warning 注意
如果你的网站希望被搜索引擎搜索到的话, 强烈这个插件需要安装
:::

生成站点地图插件, 先上插件文档 [vuepress-plugin-sitemap](https://github.com/ekoeryanto/vuepress-plugin-sitemap)

再上配置, `/.vuepress/config.js` 

``` javascript
module.exports = {
    plugins: [
        ["sitemap", {
            "hostname": "https://vxhly.github.io", // 替换成你的 GitHub 地址
            "exclude": ['/404.html'],
            "dateFormatter": time => {
                return time
            }
        }]
    ]
}
```

## 部署上线

::: tip 提示
下面提供的是 `GitHub Pages` + `Travis CI` 的部署上线方式, 如果你有更简单的或者更好的方式请你一定私信给我<br>
部分未提及的配置或者其他部署方式, 看这里 [VuePress 部署](https://vuepress.vuejs.org/zh/guide/deploy.html)
:::

### 自动向 GitHub 提交代码脚本

在你的项目目录中创建 `deploy.sh` , 自行修改下方中的注释行代码

::: warning 注意
这边的分支不能提交到 master 分支上, 因为 GitHub Pages 服务中指定了 master 分支为静态页面发布分支, 所以需要在本地创建一个名为 gh-pages 的分支, 我们的博客源码全部托管在这个分支上（当然你也可以叫其他名字）
:::

``` bash 
#!/usr/bin/env sh

# 确保脚本抛出遇到的错误

set -e

date= `date +"%Y-%m-%d %H:%M:%S"` 
commitmsg='Site updated: '$date
git add -A
git commit -m "$commitmsg"

# 如果发布到 https://<USERNAME>.github.io

# 将 <USERNAME> 替换成你的 GitHub 名字

# git push -f git@github.com:<USERNAME>/<USERNAME>.github.io.git gh-pages:gh-pages

``` 

### 自动化脚本

在你的项目目录中创建 `.travis.yml` , 更多的配置项戳这里 [Travis CI Deployment](https://docs.travis-ci.com/user/deployment)

``` yaml
language: node_js
node_js:

  + lts/*

install:

  + npm ci

script:

  + npm run build

deploy:
  provider: pages
  skip-cleanup: true
  local_dir: public
  github-token: $GITHUB_TOKEN # a token generated on github allowing travis to push code on you repository
  target-branch: master
  verbose: true
  keep-history: true
  on:
    branch: gh-pages
branches:
  only:

    - gh-pages

```

### 自动化部署

::: warning 非常重要
需要想你的仓库中正确提交你的  package-lock.json, 否则 Travis CI 在执行 `npm ci` 时可能会发生错误<br>
当然如果你并不想这么做的的话, 可以将 `npm ci` 替换成 `npm install` 
:::

首先进入 [travis-ci 官网](https://travis-ci.org/), 将对应的项目启用 Travis CI

![vuepress-blog](https://vxhly.github.io/assets/vuepress-blog-1.png)

点进去之后可以进行更多的配置

![vuepress-blog](https://vxhly.github.io/assets/vuepress-blog-2.png)

生成你的 GitHub Token, 首先获取这个 token, 进入 Github 设置, 左下的 Developer settings

![vuepress-blog](https://vxhly.github.io/assets/vuepress-blog-3.png)

添加描述, 这里只需勾选 `repo` , 如果需要其他的配置自行勾选

![vuepress-blog](https://vxhly.github.io/assets/vuepress-blog-4.png)

保存之后将会生成的 token, 将生成的 token 放入 Travis 配置中

::: warning 注意
这边配置的变量名需要和 `.travis.yml` 中指定的名字一样, 这边使用的变量名为 GITHUB_TOKEN
:::

![vuepress-blog](https://vxhly.github.io/assets/vuepress-blog-5.png)

至此, 你可以进行发布代码操作了, 执行下面的命令即可将代码提交到你的 GitHub 上了

``` bash
npm run deploy
```

代码提交之后 Travis 也开始对你提交的代码进行编译操作了, 编译成功之后将会自动的把 `public` 目录提交到你的 master 分支了

::: warning 注意
状态颜色一定要是绿色才代表编译并且部署成功了
:::

![vuepress-blog](https://vxhly.github.io/assets/vuepress-blog-6.png)

### 拓展阅读

NPM 5.7.0 引入了一种新的安装依赖的方式：

``` bash
npm ci
```

它是从 `package-lock.json` 文件安装依赖, 具有快以及可靠优点, 关于速度的比较引用官方图

![vuepress-blog](https://vxhly.github.io/assets/vuepress-blog-7.png)

官方有下面补充：

* 项目里面必须存在 `package-lock.json` 或 `npm-shrinkwrap.json` .
* 如果 `package lock` 里面依赖和 `package.json` 不一致, `npm ci` 会报错并且退出, 而不是更新 `package lock` 文件
* `npm ci` 只能一次性安装整个工程的依赖, 不能使用这个命令单独添加依赖
* 如果 `node_modules` 文件夹存在, 它会在安装依赖之前删除这个文件夹
* 它不会改变 `package.json` 或者任何 `package-locks` 
