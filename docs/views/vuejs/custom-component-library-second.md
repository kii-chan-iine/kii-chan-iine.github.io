---
author: 星火燎原@vxhly
title: 使用 Vue cli 3.0 构建自定义组件库(第二弹)
categories: [vuejs]
tags: [Vue.js, Node.js, Component, VuePress]
date: 2019-08-16 09:30:20
---

<Boxx changeTime="10000"/>

::: tip 前言
本文旨在给大家提供一种构建一个完整 UI 库脚手架的思路: 包括如何快速并优雅地构建UI库的主页、如何托管主页、如何编写脚本提升自己的开发效率、如何生成 CHANGELOG 等, 此文乃第二弹版本, 看过第一弹版本的小伙伴, 想必都知道第一弹版本的组件库文档 UI 是需要自己写的, 最近也是刚研究出来其实 vuepress 也可以作为组件库文档的 UI, 该文档 UI 绝对不输于第一弹版本的。
:::
<!-- more -->

## 前言

[想看第一弹版本的小伙伴, 请戳我](https://vxhly.github.io/views/vuejs/custom-component-library-first.html)

第二弹前期具备知识点

* [vuepress](https://vuepress.vuejs.org/zh/)
* [Vue CLI](https://cli.vuejs.org/zh/)

## 前置工作

以下工作全部基于 Vue CLI 3.x, 所以首先要保证机子上有 `@vue/cli`

``` bash
vue create vtp-component # vtp-component 作为教学的库名
```

`vue-router` , `dart-sass` , `babel` , `eslint` 这些是该项目使用的依赖项, 小主可以根据自己的需求进行相应的切换

## start

开始造轮子了, 以下示例代码 [请戳这里](https://github.com/vxhly/vtp-component)

## 工作目录

默认生成的 src 目录以及 public 目录在该项目下是没有作用的, 小主可以删掉了, 这么做的主要目的当然只是保留生成的 `package.json` 文件

调整项目目录如下, 其中 build 文件夹用来存放编译脚本代码, docs 文件夹则是 vuepress 的主体部分了, 其中 .vuepress 是用来存放 vuepress 相关的配置文件, 而组件文档则放置在 docs 的根目录, packages 文件夹是用来存放组件的核心代码, scripts 文件夹用来存放自定义生成组件和组件的说明文档等脚本

大致的目录结构如下：

``` text
|-- build
|
|-- docs
    |-- .vuepress
        |-- config.js
        |-- enhanceApp.js
    |-- README.md
|
|-- packages 
|   |-- index.js
|
|-- scripts
|
|-- config.js
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
    "build:lib": "vue-cli-service build --target lib --name vtp-component --dest lib packages/index.js"
  }
}
```

修改 main 主入口文件

``` json
{
  "main": "lib/vtp-component.common.js"
}
```

## 一个组件例子

### vuepress 配置

这里提供一些简单的配置, 当然想要高级的配置甚至想要自定义主题皆可从官方文档中获取到帮助

 `config.js`
使用插件 `vuepress-plugin-vue-demo` 将可以像 ElementUI 中写文档演示 demo 一样写文档了

``` javascript
const config = {
    dest: 'public',
    plugins: ['vue-demo'],
    serviceWorker: true,
    themeConfig: {
        sidebar: [
            ['/CHANGELOG', '更新日志'],
            ['/', '指南'],
            {
                title: '组件',
                collapsable: false,
                children: [

                ]
            }
        ]
    },
    markdown: {
        lineNumbers: true
    },
    title: 'vtp-component',
    base: '/'
}

module.exports = config
```

 `enhanceApp.js`

``` javascript
import UI from '../../packages'

export default ({
    Vue, // the version of Vue being used in the VuePress app
    options, // the options for the root Vue instance
    router, // the router instance for the app
    siteData // site metadata
}) => {
    Vue.use(UI)
}
```

### 基础的 packages 配置

``` javascript
import './assets/scss/common.scss'

const version = require('../package.json').version
const components = [

]

const install = Vue => {
    if (install.installed) return
    components.map(component => Vue.component(component.name, component))
}

export {
    install,
    version

}

export default {
    install,
    version
}
```

``` scss
$prefix: 'vtp';

@for $i from 12 through 48 {
    .#{$prefix}-fs#{$i} {
        font-size: #{$i}px !important;
    }

    .#{$prefix}-lh#{$i} {
        line-height: #{$i}px !important;
    }
}

@for $i from 5 through 40 {
    .#{$prefix}-margin#{$i} {
        margin:#{$i}px !important;
    }

    .#{$prefix}-padding#{$i} {
        padding:#{$i}px !important;
    }

    .#{$prefix}-m-l#{$i} {
        margin-left: #{$i}px !important;
    }

    .#{$prefix}-m-r#{$i} {
        margin-right: #{$i}px !important;
    }

    .#{$prefix}-m-lr#{$i} {
        margin-left: #{$i}px !important;
        margin-right: #{$i}px !important;
    }

    .#{$prefix}-m-t#{$i} {
        margin-top: #{$i}px !important;
    }

    .#{$prefix}-m-b#{$i} {
        margin-bottom: #{$i}px !important;
    }

    .#{$prefix}-m-tb#{$i} {
        margin-top: #{$i}px !important;
        margin-bottom: #{$i}px !important;
    }

    .#{$prefix}-p-l#{$i} {
        padding-left: #{$i}px !important;
    }

    .#{$prefix}-p-r#{$i} {
        padding-right: #{$i}px !important;
    }

    .#{$prefix}-p-lf#{$i} {
        padding-left: #{$i}px !important;
        padding-right: #{$i}px !important;
    }

    .#{$prefix}-p-t#{$i} {
        padding-top: #{$i}px !important;
    }

    .#{$prefix}-p-b#{$i} {
        padding-bottom: #{$i}px !important;
    }

    .#{$prefix}-p-tb#{$i} {
        padding-top: #{$i}px !important;
        padding-bottom: #{$i}px !important;
    }
}

.#{$prefix}-tc {
    text-align: center !important;
}

.#{$prefix}-tl {
    text-align: left !important;
}

.#{$prefix}-tr {
    text-align: right !important;
}

.#{$prefix}-fl {
    float: left !important;
}

.#{$prefix}-fr {
    float: riaght !important;
}

.#{$prefix}-cl-both {
    clear: both !important;
}

.#{$prefix}-fw-b {
    font-weight: bold !important;
}

.#{$prefix}-absolute {
    position: absolute !important;
}

.#{$prefix}-relative {
    position: relative !important;
}

.#{$prefix}-fixed {
    position: fixed !important;
}

.#{$prefix}-flex {
    display: flex;
}

.#{$prefix}-f-sc {
    align-self: center;
}
```

### 主入口 README

``` markdown
# 指南

## 介绍

::: tip
开箱即用的 Vue 组件库
:::

## Install

\ `\` \`bash
npm install vtp
# or
yarn add vtp
\ `\` \`

## Used

### 一键全局引入

\ `\` \`javascript
import Vue from 'vue'
import Vtp from 'vtp'

Vue.use(Vtp)
\ `\` \`

### 按需引入

\ `\` \`javascript
import Vue from 'vue'
import {
    VtpButton
} from 'vtp'

Vue.use(VtpButton)
\ `\` \`

## 组件库贡献指南

::: danger
组件库贡献必读, 请严格按照下列贡献指南
::: 

01. 第一次使用该组件库时, 请执行命令 `npm run init`
02. 推荐使用 [git-flow 工作流](https://www.git-tower.com/learn/git/ebook/cn/command-line/advanced-topics/git-flow), 确保一个组件一个分支, 使用命令 `git flow feature start [组件名]` , 开发结束之后需要使用命令 `git flow feature finish [组件名]` 该命令将会自动进行分支合并以及删除当前分支
03. 当需要对一个已存在的组件进行更新维护或者升级时, 请使用命令 `git flow hotfix start [组件名]` , 开发结束之后需要使用命令 `git flow hotfix finish [组件名]` 该命令将会自动进行分支合并以及删除当前分支
04. 当需要对一个已存在的组件进行 bug 修复时, 请使用命令 `git flow bugfix start [组件名]` , 开发结束之后需要使用命令 `git flow bugfix finish [组件名]` 该命令将会自动进行分支合并以及删除当前组件分支
05. 强制使用已配置脚本命令 `yarn new:comp` 来创建一个组件以及组件文档(已包含默认模板, 在此基础上修改即可), 使用命令 `yarn del:comp` 来删除不需要的组件
06. 使用 `git cz` 替换 `git commit -m` , commit 之前请详细阅读 [git commit 规范指南](https://vxhly.github.io/archives/fdd4f330.html)
07. 组件发布前, 请确保当前版本的所有包含组件已经全部开发完毕并且已经合并至 develop 分支

    - 在 develop 分支中使用 `git flow release start [版本号]` 创建一个发布版本分支
    - 修改 `package.json` 中的 `version` 字段为当前的版本号
    - 执行 `npm run docs:build` 编译 vuepress 文档
    - 执行 `npm run build:lib` 编译 组件库
    - 执行 `npm run changelog` 生成当前版本的 CHANGELOHG
    - 以上操作可使用命令 `npm run build` 来代替 `npm run docs:build` 、 `npm run build:lib` 、 `npm run changelog`
    - 进行 commit 操作
    - 使用 `git flow release finish [版本号]` 自动进行分支合并以及删除当前分支, 期间将会询问 tag 的 commit 信息(务必认真填写)
    - 以上操作之后将会把分支自动合并至 master 和 develop 分支

    
    

08. 分支永远只能有两个 master 和 develop 分支, 其他分支应该在开发完毕之后及时的进行合并、删除操作
09. 发布前请确保 tag 已经打包成功, 当前版本的功能已经全部完成, 之后便可以直接运行 `npm publish --registry <私有库地址>` , 如果组件库目录下有配置 `.npmrc` 则直接使用 `npm publish`
10. 下载组件库 `yarn add <组件库名> --registry <私有库地址>` , 如果组件库目录下有配置 `.npmrc` 则直接使用 `yarn add <组件库名> `
```

### 创建组件和组件文档生成脚本

在 `scripts` 中创建以下几个文件, 其中 `create-comp.js` 是用来生成自定义组件目录和自定义组件说明文档脚本, `delete-comp.js` 是用来删除无用的组件目录和自定义组件说明文档脚本, `template.js` 是生成代码的模板文件

``` text
|-- create-comp.js
|
|-- delete-comp.js
|
|-- template.js
```

相关的代码如下, 小主可以根据自己的需求进行相应的简单修改, 下面的代码参考来源 [vue-cli3 项目优化之通过 node 自动生成组件模板 generate View、Component](https://juejin.im/post/5c6772ff518825629c56943c), 当然也是第一弹中的代码改进

全局可配置一些变量

 `config.js`

``` javascript
module.exports = {
    prefix: 'vtp', // 组件库的前缀,使用小写
    docsTitle: 'vtp-component', // 文档的标题
    base: '/', // vuepress 中的 base 配置
    ElementUI: true // 引入 element-ui 表示二次封装 element-ui
}
```

 `create-comp.js`

``` javascript
// 创建自定义组件脚本

const chalk = require('chalk')
const path = require('path')
const fs = require('fs-extra')
const uppercamelize = require('uppercamelcase')
const resolve = (...file) => path.resolve(__dirname, ...file)
const log = message => console.log(chalk.green( `${message}` ))
const successLog = message => console.log(chalk.blue( `${message}` ))
const errorLog = error => console.log(chalk.red( `${error}` ))
const {
    vueTemplate,
    entryTemplate,
    scssTemplate,
    mdDocs
} = require('./template')

const generateFile = (path, data) => {
    if (fs.existsSync(path)) {
        errorLog( `${path}文件已存在` )
        return
    }
    return new Promise((resolve, reject) => {
        fs.writeFile(path, data, 'utf8', err => {
            if (err) {
                errorLog(err.message)
                reject(err)
            } else {
                resolve(true)
            }
        })
    })
}

// 这里生成自定义组件
log('请输入要生成的组件名称, 形如 demo 或者 demo-test')
let componentName = ''
process.stdin.on('data', async chunk => {
    const inputName = String(chunk).trim().toString()
    const upperInputname = uppercamelize(inputName)
    const componentDirectory = resolve('../packages', upperInputname)
    const componentVueName = resolve(componentDirectory, `${upperInputname}.vue` )
    const scssName = resolve(componentDirectory, `${inputName}.scss` )
    const entryComponentName = resolve(componentDirectory, 'index.js')

    const hasComponentDirectory = fs.existsSync(componentDirectory)
    if (upperInputname) {
        // 这里生成组件
        if (hasComponentDirectory) {
            errorLog( `${upperInputname}组件目录已存在, 请重新输入` )
            return
        } else {
            log( `生成 component 目录 ${componentDirectory}` )
            await dotExistDirectoryCreate(componentDirectory)
        }
        try {
            if (upperInputname.includes('/')) {
                const inputArr = upperInputname.split('/')
                componentName = inputArr[inputArr.length - 1]
            } else {
                componentName = upperInputname
            }
            log( `生成 vue 文件 ${componentVueName}` )
            await generateFile(componentVueName, vueTemplate(inputName))
            log( `生成 scss 文件 ${scssName}` )
            await generateFile(scssName, scssTemplate(inputName))
            log( `生成 entry 文件 ${entryComponentName}` )
            await generateFile(entryComponentName, entryTemplate(componentName))
            successLog('生成 component 成功')
        } catch (e) {
            errorLog(e.message)
        }
    } else {
        errorLog( `请重新输入组件名称:` )
        return
    }

    // 这里生成自定义组件说明文档
    const docsDirectory = resolve('../docs/component')
    const docsMdName = resolve(docsDirectory, `${upperInputname}.md` )

    try {
        log( `生成 component 文档 ${docsMdName}` )
        await generateFile(docsMdName, mdDocs( `${inputName}` ))
        successLog('生成 component 文档成功')
    } catch (e) {
        errorLog(e.message)
    }

    process.stdin.emit('end')
})

process.stdin.on('end', () => {
    log('exit')
    process.exit()
})

function dotExistDirectoryCreate(directory) {
    return new Promise((resolve) => {
        mkdirs(directory, function() {
            resolve(true)
        })
    })
}

// 递归创建目录
function mkdirs(directory, callback) {
    var exists = fs.existsSync(directory)
    if (exists) {
        callback()
    } else {
        mkdirs(path.dirname(directory), function() {
            fs.mkdirSync(directory)
            callback()
        })
    }
}
```

 `delete-comp.js`

``` javascript
// 删除自定义组件脚本

const chalk = require('chalk')
const path = require('path')
const fs = require('fs-extra')
const uppercamelize = require('uppercamelcase')
const resolve = (...file) => path.resolve(__dirname, ...file)
const log = message => console.log(chalk.green( `${message}` ))
const successLog = message => console.log(chalk.blue( `${message}` ))
const errorLog = error => console.log(chalk.red( `${error}` ))

log('请输入要删除的组件名称, 形如 demo 或者 demo-test')
process.stdin.on('data', async chunk => {
    let inputName = String(chunk).trim().toString()
    inputName = uppercamelize(inputName)
    const componentDirectory = resolve('../packages', inputName)

    const hasComponentDirectory = fs.existsSync(componentDirectory)

    const docsDirectory = resolve('../docs/component')
    const docsMdName = resolve(docsDirectory, `${inputName}.md` )
    if (inputName) {
        if (hasComponentDirectory) {
            log( `删除 component 目录 ${componentDirectory}` )
            await removePromise(componentDirectory)
            successLog( `已删除 ${inputName} 组件目录` )

            log( `删除 component 文档 ${docsMdName}` )
            fs.unlink(docsMdName)
            successLog( `已删除 ${inputName} 组件说明文档` )
        } else {
            errorLog( `${inputName}组件目录不存在` )
            return
        }
    } else {
        errorLog( `请重新输入组件名称:` )
        return
    }

    process.stdin.emit('end')
})

process.stdin.on('end', () => {
    log('exit')
    process.exit()
})

function removePromise(dir) {
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
```

 `template.js`

``` javascript
const uppercamelize = require('uppercamelcase')
const config = require('../config')

module.exports = {
    vueTemplate: compoenntName => {
        return `<template>
  <div class="${config.prefix}-${compoenntName}">
    ${compoenntName}
  </div>
</template>

<script>
export default {
  name: '${config.prefix}-${compoenntName}',

  data () {
    return {}
  },

  props: {},

  methods: {}
}
</script>

<style lang="scss">
  @import './${compoenntName}';
</style>
`
    },
    entryTemplate: compoenntName => {
        return `import ${compoenntName} from './${compoenntName}'
    
${compoenntName}.install = function (Vue) {
  Vue.component(${compoenntName}.name, ${compoenntName})
}

export default ${compoenntName}

if (typeof window !== 'undefined' && window.Vue) {
  window.Vue.component(${compoenntName}.name, ${compoenntName})
}
`
    },
    scssTemplate: compoenntName => {
        return `.${config.prefix}-${compoenntName} {}`
    },
    mdDocs: compoenntName => {
        const upperCompoenntName = uppercamelize(compoenntName)
        const upperPrefix = uppercamelize(config.prefix)

        return `# ${compoenntName}

::: tip
${compoenntName}
::: 

## Code Demo

::: demo 基础用法

\ `\` \` vue
<${config.prefix}-${compoenntName}></${config.prefix}-${compoenntName}>
\ `\` \`
:::

## Used

### 按需引入

\ `\` \`javascript
import Vue from 'vue'
import {
  ${upperPrefix}${upperCompoenntName}
} from '${config.prefix}'

Vue.use(${upperPrefix}${upperCompoenntName})
\ `\` \`

### 局部引入

\ `\` \` javascript
import {
  ${upperPrefix}${upperCompoenntName}
} from '${config.prefix}'

export default {
  components: {
    ${upperPrefix}${upperCompoenntName}
  }
}
\ `\` \`

## API

### Props

| 参数  | 说明  | 类型  | 可选值 | 默认值 | 版本 |
|-----|-----|-----|-----|-----|-----|
| -   | -   | -   | -   | -   | -   |

### Events

| 事件名  | 说明  | 回调参数  |
|-----|-----|-----|
| -   | -   | -   |

`
    }
}
```

在 `build` 中创建以下几个文件, 其中 `build-entry.js` 脚本是用来生成自定义组件导出 `packages/index.js` , `get-components.js` 脚本是用来获取 `packages` 目录下的所有组件

``` text
|-- build-entry.js
|
|-- get-components.js
```

相关的代码如下, 小主可以根据自己的需求进行相应的简单修改, 下面的代码参考来源 [vue-cards](https://github.com/Eamonnzhang/vue-cards)

 `build-entry.js`

``` javascript
const fs = require('fs-extra')
const path = require('path')
const chalk = require('chalk')
const uppercamelize = require('uppercamelcase')
const Components = require('./get-components')()
const log = message => console.log(chalk.green( `${message}` ))
const config = require('../config')

function buildPackagesEntry() {
    const uninstallComponents = []

    const importList = Components.map(
        name => `import ${uppercamelize(name)} from './${name}'`
    )
    const exportList = Components.map(name => `${uppercamelize(name)}` )
    const intallList = exportList.filter(
        name => !~uninstallComponents.indexOf(uppercamelize(name))
    )
    const content = `import './assets/common.scss'
import './utils/filters'

${importList.join('\n')}

const version = require('../package.json').version
const components = [
  ${intallList.join(',\n  ')}
]

const install = Vue => {
  if (install.installed) return
  components.map(component => Vue.component(component.name, component))
}

export {
  install,
  version,
  ${exportList.join(',\n  ')}
}

export default {
  install,
  version
}
`

    fs.writeFileSync(path.join(__dirname, '../packages/index.js'), content)
    log('packages/index.js 文件已更新依赖')
    log('exit')
}

function setDocsConfig() {
    const docsURL = []
    Components.forEach(item => {
        docsURL.push( `'/component/${item}'` )
    })
    const content = `const config = {
  dest: 'public',
  plugins: ['vue-demo'],
  serviceWorker: true,
  themeConfig: {
    sidebar: [
      ['/CHANGELOG', '更新日志'],
      ['/', '指南'], 
      {
        title: '组件',
        collapsable: false,
        children: [
          ${docsURL.join(',\n          ')}
        ]
      }
    ]
  },
  markdown: {
    lineNumbers: true
  },
  title: '${config.docsTitle}',
  base: '${config.base}'
}

module.exports = config
`

    fs.writeFileSync(path.join(__dirname, '../docs/.vuepress/config.js'), content)
    log('packages/index.js 文件已更新依赖')
    log('exit')
}

buildPackagesEntry()
setDocsConfig()
```

 `get-components.js`

``` javascript
const fs = require('fs')
const path = require('path')

const excludes = [
    'index.js',
    'theme-chalk',
    'mixins',
    'utils',
    '.DS_Store',
    'assets'
]

module.exports = function() {
    const dirs = fs.readdirSync(path.resolve(__dirname, '../packages'))
    return dirs.filter(dirName => excludes.indexOf(dirName) === -1)
}
```

### 生成命令

在 `package.json` 中添加以下内容, 使用命令 `yarn new:comp` 创建组件目录及其文档或者使用命令 `yarn del:comp` 即可删除组件目录及其文档

``` json
{
  "scripts": {
    "new:comp": "node scripts/create-comp.js && node build/build-entry.js",
    "del:comp": "node scripts/delete-comp.js && node build/build-entry.js"
  }
}
```

# changelog

在 `package.json` 中修改 script 字段, 接下来你懂的, 另一篇博客有介绍哦, 小主可以执行搜索

``` json
{
  "scripts": {
    "init": "npm install commitizen -g && commitizen init cz-conventional-changelog --save-dev --save-exact && npm run bootstrap",
    "bootstrap": "npm install",
    "changelog": "conventional-changelog -p angular -i docs/CHANGELOG.md -s -r 0"
  }
}
```

## 关于 pages 服务和发布至 npm

### pages 服务

当然组件库以及文档完成之后, 都希望能够有个地址托管我们的组件文档, 这里比较推荐的是使用 [Gitee](https://gitee.com) 中的项目pages 服务

### 发布至 npm

分情况而言

一是该组件库是公开的, 希望该组件库有较多的社区小伙伴能够一起维护的, 这里就比较推荐使用 [GitHub](https://github.com), 发布命令为 `npm publish` , 相关的 pages 服务可以参考本博客的托管机制(具体的还需要小主自行研究)

二是该组件库是是有的, 希望该组件库只是作为多个项目之间的共用 UI 框架, 这里比较推荐的是使用 [Gitee](https://gitee.com), 发布命令为 `npm publish` , 当然只是这样是不够的, 比较推荐的是使用 npm 私有库来托管我们的组件库(npm 私有库搭建请参考 [docker 使用指南-私有 npm 代理注册表](https://vxhly.github.io/archives/693996ef.html), 具体的还需要小主自行研究)

发布至私有库之前需要详细阅读这两篇文章: [docker 使用指南-私有 npm 代理注册表](https://vxhly.github.io/archives/693996ef.html), [Git 远程推送配置](https://vxhly.github.io/archives/9a1dd005.html)

### 拓展

可以定义无需发布的文将和目录

在项目目录下创建 `.npmignore`

``` text
*.js
.browserslistrc
.editorconfig
.npmrc
build/
docs/
scripts/
test/
public/
```

如果该组件库作为私有库发至私有 npm 代理库中, 建议在组件库项目以及依赖组件库的项目中创建 `.npmrc` , 注意这里的地址千万千万不能写 `127.0.0.1` 或者 `localhost` , 必须写成公网 IP 或者局域网 IP 或者域名, 否则在多人合作模式下则不能正确的发布或者下载

``` text
registry=http://192.168.8.10:4873 
```
