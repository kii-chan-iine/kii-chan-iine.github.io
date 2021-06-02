---
author: 星火燎原@vxhly
title: 使用 Vue cli 3.0 构建自定义组件库(第三弹)
categories: [vuejs]
tags: [Vue.js, Node.js,  , VuePress]
date: 2020-06-26 10:57:54
---

<Boxx changeTime="10000"/>

::: tip 前言
本文旨在给大家提供一种构建一个完整 UI 库脚手架的思路: 包括如何快速并优雅地构建UI库的主页、如何托管主页、如何编写脚本提升自己的开发效率、如何生成 CHANGELOG 等, 此文乃第二弹版本, 看过第一弹版本和第二弹版本的小伙伴, 想必知道前两弹版本的多多少少存在着一些不足之处, 这一弹的版本为第二弹版本的升级版, 这个版本中我们将提供更多方便于操作的内容。
:::
<!-- more -->

## 前言

[想看第一弹版本的小伙伴, 请戳我](https://vxhly.github.io/views/vuejs/custom-component-library-first.html), [想看第二弹版本的小伙伴, 请戳我](https://vxhly.github.io/views/vuejs/custom-component-library-second.html)

第三弹前期具备知识点

* [vuepress](https://vuepress.vuejs.org/zh/)
* [Vue CLI](https://cli.vuejs.org/zh/)
* [Docker](https://docs.docker.com/docker-hub/)

## 前置工作

以下工作全部基于 Vue CLI 4.x, 所以首先要保证机子上有 `@vue/cli`

``` bash
vue create vtp-ui # vtp-ui 作为教学的库名
```

`dart-sass` , `babel` , `eslint` 这些是该项目使用的依赖项, 小主可以根据自己的需求进行相应的切换

## start

开始造轮子了, 以下示例代码 [请戳这里](https://github.com/vxhly/vtp-ui)

## 工作目录

默认生成的 src 目录以及 public 目录在该项目下是没有作用的, 小主可以删掉了, 这么做的主要目的当然只是保留生成的 `package.json` 文件

调整项目目录如下, 其中 BAT_SCRIPT 文件夹用来存放 windwos 下使用 `forever` 启动 npm 私有代理的脚本, DOCKER 文件夹用来存放有 `docker` 环境下启动 npm 私有代理的脚本, docs 文件夹则是 vuepress 的主体部分了, 其中 .vuepress 是用来存放 vuepress 相关的配置文件, 而组件文档则放置在 docs 的根目录, packages 文件夹是用来存放组件的核心代码, vtp-cli 文件夹则是用来存放自定义的组件库中的组件文件以及相应文档生成的 cli 工具

大致的目录结构如下：

``` text
|-- BAT_SCRIPT
|
|-- DOCKER
|
|-- docs
    |-- .vuepress
        |-- config.js
        |-- enhanceApp.js
    |-- README.md
|
|-- vtp-cli
|
|-- packages 
|   |-- index.js
|
|-- .npmrm
|
|-- .npmignore
```

## 添加编译脚本

::: danger 注意
`core-js` 库需要采用的 2.6.5 这个版本, 高于这个版本的可能会导致开启文档失败
:::

 `package.json`
其中的组件 name 推荐和创建的项目名一致 

``` json
{
  "scripts": {
    "build:lib": "vue-cli-service build --target lib --name vtp-ui --dest lib packages/index.js"
  }
}
```

修改 main 主入口文件

``` json
{
  "main": "lib/vtp-ui.common.js"
}
```

## 一个组件例子

### vuepress 进阶配置

 `.vuepress/config.js`

``` javascript
const fs = require('fs')
const path = require('path')
const baseDir = fs.readdirSync(path.resolve(__dirname, '../base'))
const businessDir = fs.readdirSync(path.resolve(__dirname, '../business'))

const config = {
    dest: 'public',
    plugins: [
        'vue-demo',
        require('./plugins/clipboard-copy')
    ],
    serviceWorker: true,
    themeConfig: {
        sidebar: [
            ['/Guide', '指南'],
            {
                title: '普通组件',
                collapsable: false,
                children: [
                    ...baseDir.map(item => `/base/${item.replace('.md','')}` )
                ]
            },
            {
                title: '业务组件',
                collapsable: false,
                children: [
                    ...businessDir.map(item => `/business/${item.replace('.md','')}` )
                ]
            }
        ]
    },
    markdown: {
        lineNumbers: true,
        extendMarkdown: (md) => {
            // 使用 markdown-it-vuese 自动生成组件
            md.use(require('markdown-it-vuese'), {
                vueseRe: /<\[vuese-h3\]\((.+)\)/i,
                ruleName: 'vuese-h3',
                useRender: (vueseRender) => {
                    const renderRes = vueseRender.render()
                    const genMd = key => `### ${key}\n${renderRes[key]}\n`
                    return Object.keys(renderRes).map(genMd).join('')
                },
            })
        }
    },
    title: 'vtp-ui',
    base: '/vtp-ui',
    chainWebpack: config => {

    }
}

module.exports = config
```

::: danger
在做组件库时, 有可能出现 `windows`、`document` 未能找到问题, 可以参考以下文章进行配置

[VuePress window document is not defined](https://segmentfault.com/a/1190000022727986)
:::



### CLI 工具

::: tip 提示
这才是本弹的重点, 其他内容小主可自主查阅示例代码
:::

 `vtp-cli/bin/vtp-cli.js`

``` javascript
#! /usr/bin/env node

const fs = require('fs')
const {
    version
} = require('../package.json')
const chalk = require('chalk')
const program = require('commander')
const initPackage = require('./vtp-new')
const delPackage = require('./vtp-del')

const VUPRESS_DIR = 'docs/.vuepress'

const isProject = fs.existsSync(VUPRESS_DIR)

if (isProject) {
    program
        .version(version)
        .usage('[options]')
        .description('运行命令来为你的组件库新增或者删除组件。')

    program
        .command('new')
        .description('创建一个组件')
        .action(initPackage)

    program
        .command('del')
        .description('删除一个组件')
        .action(delPackage)
    program.parse(process.argv)
} else {
    console.log(chalk.red('非常抱歉！当前的项目目录非组件库目录。'))
}
```

 `vtp-cli/bin/vtp-new.js`

``` javascript
#! /usr/bin/env node

const path = require('path')
const uppercamelcase = require('uppercamelcase')

const mkdirs = require('./utils/mkdirs')
const copyFile = require('./utils/file')
const newHandleInquirer = require('./inquirer/new-inquier')

const PACKAGES_DIR = `${process.cwd()}/packages`
const CSS_DIR = `${PACKAGES_DIR}/theme-chalk`
const DOCS_DIR = 'docs'

const initPackage = () => {
    newHandleInquirer()
        .then((choices) => {
            const {
                compoenntName,
                compoenntType
            } = choices
            const PACKAGE_DIR = `${PACKAGES_DIR}/${compoenntName}`
            const PACKAGE_DOCS_DIR = `${DOCS_DIR}/${compoenntType}`
            mkdirs(PACKAGE_DIR, () => {
                copyFile(path.join(__dirname, '/tmp'), PACKAGE_DIR, 'entry.js.tmp', 'index.js', choices)
                copyFile(path.join(__dirname, '/tmp'), PACKAGE_DIR, 'entry.vue.tmp', `${uppercamelcase(compoenntName)}.vue` , choices)
                copyFile(path.join(__dirname, '/tmp'), PACKAGE_DOCS_DIR, 'entry.md.tmp', `${uppercamelcase(compoenntName)}.md` , choices)
                copyFile(path.join(__dirname, '/tmp'), CSS_DIR, 'entry.scss.tmp', `${compoenntName}.scss` , choices)
                copyFile(path.join(__dirname, '/tmp'), CSS_DIR, 'entry.css.tmp', `${compoenntName}.css` , choices)
                // copyFile(path.join(__dirname, '/tmp'), CSS_DIR, 'entry.min.css.tmp', `${compoenntName}.min.css` , choices)
                copyFile(path.join(__dirname, '/tmp'), PACKAGES_DIR, 'export.js.tmp', 'index.js', choices)
            })
        })
}

module.exports = initPackage
```

 `vtp-cli/bin/vtp-del.js`

``` javascript
#! /usr/bin/env node

const fs = require('fs')
const path = require('path')
const chalk = require('chalk')
const uppercamelcase = require('uppercamelcase')

const delHandleInquirer = require('./inquirer/del-inquirer.js')
const copyFile = require('./utils/file')

const PACKAGES_DIR = `${process.cwd()}/packages`
const DOCS_DIR = 'docs'
const CSS_DIRECTORY = `${PACKAGES_DIR}/theme-chalk`
const removePromise = (dir) => {
    return new Promise(function(resolve, reject) {
        // 先读文件夹
        fs.stat(dir, function(_err, stat) {
            if (stat.isDirectory()) {
                fs.readdir(dir, function(_err, files) {
                    files = files.map(file => path.join(dir, file)) // a/b  a/m
                    files = files.map(file => removePromise(file)) // 这时候变成了promise
                    Promise.all(files).then(function() {
                        fs.rmdir(dir, resolve)
                    })
                })
            } else {
                fs.unlink(dir, resolve)
            }
        })
    })
}

const delPackage = () => {
    delHandleInquirer()
        .then((choices) => {
            const {
                delPackageName,
                isDel
            } = choices

            if (isDel) {
                const PACKAGE_DIR = `${PACKAGES_DIR}/${delPackageName}`
                const SCSS_NAME = `${CSS_DIRECTORY}/${delPackageName}.scss`
                const CSS_NAME = `${CSS_DIRECTORY}/${delPackageName}.css`
                // const CSS_MIN_NAME = `${CSS_DIRECTORY}/${delPackageName}.min.css`
                const BASE_DOCS = `${DOCS_DIR}/base/${uppercamelcase(delPackageName)}.md`
                const BUSINESS_DOCS = `${DOCS_DIR}/business/${uppercamelcase(delPackageName)}.md`
                removePromise(PACKAGE_DIR)
                console.log(chalk.green( `The package directory ${PACKAGES_DIR}/${delPackageName} was deleted successfully.` ))

                const delFiles = [
                    SCSS_NAME,
                    CSS_NAME,
                    // CSS_MIN_NAME,
                    BASE_DOCS,
                    BUSINESS_DOCS
                ]
                delFiles.map(item => {
                    fs.unlink(item, err => {
                        if (!err) {
                            console.log(chalk.green( `Delete file ${item} successfully.` ))
                        }
                    })
                })
                copyFile(path.join(__dirname, '/tmp'), PACKAGES_DIR, 'export.js.tmp', 'index.js', choices)
            } else {
                console.log(chalk.gray('已经为你取消当前的删除操作.'))
            }
        })
}

module.exports = delPackage
```

## 开启私有代理

### Windows 服务器

* `BAT_SCRIPT` 启动代理或者终止代理的脚本

 `forever-npm-start.bat`

``` bat
@echo off
C:
cd C:\Users\Administrator\AppData\Roaming\npm\node_modules\verdaccio\build\lib
forever start cli.js
exit
```

 `forever-npm-stop.bat`

``` bat
@echo off
C:
cd C:\Users\Administrator\AppData\Roaming\npm\node_modules\verdaccio\build\lib
forever stop cli.js
exit
```

* 安装相关的依赖

``` bash
npm  install  -g  forever  verdaccio
```

* 修改监听的端口号, 必须修改否则外网将访问不到（参考文件目录 C:\Users\Administrator\AppData\Roaming\verdaccio\config.yml）, 添加一行配置 `listen: 0.0.0.0:4873`
* 拷贝项目中的 BAT_SCRIPT 文件夹到 windows 服务器上的任何一个地方

::: danger 注意

* 双击运行文件夹中的 forever-npm-start.bat 即可开启 npm 代理服务
* 双击运行文件夹中的 forever-npm-stop.bat 即可关闭 npm 代理服务
* 需要终止服务的话, 必须运行 forever-npm-stop.bat 否则在不重启机子下永远也关不掉该服务

:::

### 有 Docker 环境的服务器

`DOCKER` 中的 `docker-compose.yml`

``` yaml
version: '3'

services:
  verdaccio:
    image: verdaccio/verdaccio
    restart: always
    container_name: verdaccio
    ports:

      - '4873:4873'

    volumes:

      - ./verdaccio-storage:/verdaccio/storage

volumes:
  verdaccio:
    driver: local
```

* 拷贝项目中的 DOCKER 文件夹到有 Docker 环境的服务器的任何一个地方
* 切换至 DOCKER目录
* 使用 docker-compose 开启服务

``` bash
docker-compose  up  -d
```

* 使用 docker-compose 开启服务

``` bash
docker-compose  stop
```
