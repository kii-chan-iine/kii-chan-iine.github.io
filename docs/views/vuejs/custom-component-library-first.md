---
author: 星火燎原@vxhly
title: 使用 Vue cli 3.0 构建自定义组件库(第一弹)
categories: [vuejs]
tags: [Vue.js, Node.js, Component]
date: 2019-04-28 17:19:49
---

<Boxx changeTime="10000"/>

::: tip 前言
本文旨在给大家提供一种构建一个完整 UI 库脚手架的思路: 包括如何快速并优雅地构建UI库的主页、如何托管主页、如何编写脚本提升自己的开发效率、如何生成 CHANGELOG 等。
:::
<!-- more -->

## 先言

主流的开源 UI 库代码结构主要分为三大部分: 

* 组件库本身的代码: 这部分代码会发布到 npm 上
* 预览示例和查看文档的网站代码: 类似 Vant、ElementUI 这类网站。
* 配置文件和脚本文件: 用于打包和发布等等

::: tip 提示
编写此博文的灵感 UI 框架库（[vue-cards](https://github.com/Eamonnzhang/vue-cards)）, PS: 此 UI　框架库相对于　Vant、ElementUI　会比较简单点, 可以作为一份自定义　UI　框架库的入坑　demo, 同时这篇博文也是解读这份 UI 框架库的构建到上线的一个过程
:::

## 前置工作

以下工作全部基于 Vue CLI 3.x, 所以首先要保证机子上有 `@vue/cli`

``` bash
vue create vtp-component ## vtp-component 作为教学的库名
```

## start

开始造轮子了

### 工作目录

在根目录下新增四个文件夹, 一个用来存放组件的代码（packages）, 一个用来存放 预览示例的网站 代码（examples）（这里直接把初始化模板的 src 目录更改为 examples 即可, 有需要的话可以将该目录进行清空操作, 这里就不做过多的说明）, 一个用来存放编译脚本代码（build）, 一个用来存放自定义生成组件和组件的说明文档等脚本（scripts）

```text
|-- build
|
|-- examples
|
|-- packages 
|
|-- scripts
```

### 让 webpack 编译 examples

由于我们将 src 目录修改成了 examples, 所以在 `vue.config.js` 中需要进行相应的修改

``` javascript
const path = require('path')

function resolve(dir) {
    return path.join(__dirname, dir)
}

module.exports = {
    productionSourceMap: true,

    // 修改 src 为 examples
    pages: {
        index: {
            entry: 'examples/main.js',
            template: 'public/index.html',
            filename: 'index.html'
        }
    },

    chainWebpack: config => {
        config.resolve.alias
            .set('@', resolve('examples'))
    }
}
```

### 添加编译脚本

 `package.json`
其中的组件 name 推荐和创建的项目名一致 

``` json
{
  "scripts": {
    "lib": "vue-cli-service build --target lib --name vtp-component --dest lib packages/index.js"
  }
}
```

修改 main 主入口文件

``` json
{
  "main": "lib/vtp-component.common.js"
}
```

### 一个组件例子

#### 创建组件和组件文档生成脚本

在 `scripts` 中创建以下几个文件, 其中 `create-comp.js` 是用来生成自定义组件目录和自定义组件说明文档脚本, `delete-comp.js` 是用来删除无用的组件目录和自定义组件说明文档脚本, `template.js` 是生成代码的模板文件

``` text
|-- create-comp.js
|
|-- delete-comp.js
|
|-- template.js
```

相关的代码如下, 小主可以根据自己的需求进行相应的简单修改, 下面的代码参考来源 [vue-cli3 项目优化之通过 node 自动生成组件模板 generate View、Component](https://juejin.im/post/5c6772ff518825629c56943c)

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
    let inputName = String(chunk).trim().toString()
    inputName = uppercamelize(inputName)
    const componentDirectory = resolve('../packages', inputName)
    const componentVueName = resolve(componentDirectory, `${inputName}.vue` )
    const entryComponentName = resolve(componentDirectory, 'index.js')

    const hasComponentDirectory = fs.existsSync(componentDirectory)
    if (inputName) {

        // 这里生成组件
        if (hasComponentDirectory) {
            errorLog( `${inputName}组件目录已存在, 请重新输入` )
            return
        } else {
            log( `生成 component 目录 ${componentDirectory}` )
            await dotExistDirectoryCreate(componentDirectory)
        }
        try {
            if (inputName.includes('/')) {
                const inputArr = inputName.split('/')
                componentName = inputArr[inputArr.length - 1]
            } else {
                componentName = inputName
            }
            log( `生成 vue 文件 ${componentVueName}` )
            await generateFile(componentVueName, vueTemplate(componentName))
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
    const docsDirectory = resolve('../examples/docs')
    const docsMdName = resolve(docsDirectory, `${inputName}.md` )

    try {

        log( `生成 component 文档 ${docsMdName}` )
        await generateFile(docsMdName, mdDocs( `${inputName} 组件` ))
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

    const docsDirectory = resolve('../examples/docs')
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
module.exports = {
    vueTemplate: compoenntName => {

        compoenntName = compoenntName.charAt(0).toLowerCase() + compoenntName.slice(1)
        return `<template>
    
  <div class="vtp-${compoenntName}">

    ${compoenntName}
    
  </div>
</template>

<script>
export default {
  name: 'vtp-${compoenntName}', 

  data () {

    return {
    }
    
  }, 

  props: {

  }, 

  methods: {}
}
</script>

<style lang="scss" scope>
.vtp-${compoenntName}{}
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
    mdDocs: (title) => {

        return `## ${title}

<!-- {.md} -->

---
<!-- {.md} -->

### 如何使用

<!-- {.md} -->

### Attributes

<!-- {.md} -->

| 参数  | 说明  | 类型  | 可选值 | 默认值 |
|-----|-----|-----|-----|-----|
| -   | -   | -   | -   | -   |
    }
}
`
    },
    entryTemplate: compoenntName => {
        return `import ${compoenntName} from './${compoenntName}'
    
$ {compoenntName}.install = function(Vue) {
    Vue.component(${compoenntName}.name, ${compoenntName)
}

export default ${compoenntName}

if (typeof window !== 'undefined' && window.Vue) {
    window.Vue.component(${compoenntName}.name, ${compoenntName})
}
`
    }
}
```

在 `build` 中创建以下几个文件, 其中 `build-entry.js` 脚本是用来生成自定义组件导出 `packages/index.js` , `get-components.js` 脚本是用来获取 `packages` 目录下的所有组件

``` text
|--- build-entry.js
|
|--- get-components.js
```

相关的代码如下, 小主可以根据自己的需求进行相应的简单修改, 下面的代码参考来源 [vue-cards](https://github.com/Eamonnzhang/vue-cards)

 `build-entry.js`

``` javascript
const fs = require('fs-extra')
const path = require('path')
const chalk = require('chalk')
const uppercamelize = require('uppercamelcase')
const Components = require('./get-components')()
const packageJson = require('../package.json')
const log = message => console.log(chalk.green( `${message}` ))

const version = process.env.VERSION || packageJson.version

function buildPackagesEntry() {
    const uninstallComponents = []

    const importList = Components.map(

        name => `import ${uppercamelize(name)} from './${name}'`
    )
    const exportList = Components.map(name => `${uppercamelize(name)}` )
    const intallList = exportList.filter(

        name => !~uninstallComponents.indexOf(uppercamelize(name))

    )
    const content = `import 'normalize.css'
    
${importList.join('\n')}

const version = '${version}'
const components = [
  ${intallList.join(', \n  ')}
]

const install = Vue => {
  if (install.installed) return
  components.map(component => Vue.component(component.name, component))
}

if (typeof window !== 'undefined' && window.Vue) {
  install(window.Vue)
}

export {
  install, 
  version, 
  ${exportList.join(', \n  ')}
}

export default {
  install, 
  version, 
  ...components
}
`

    fs.writeFileSync(path.join(__dirname, '../packages/index.js'), content)
    log('packages/index.js 文件已更新依赖')
    log('exit')
}

buildPackagesEntry()
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
    '.DS_Store'
]

module.exports = function() {
    const dirs = fs.readdirSync(path.resolve(__dirname, '../packages'))
    return dirs.filter(dirName => excludes.indexOf(dirName) === -1)
}
```

#### 让 vue 解析 markdown

::: tip 提示
文档中心的 UI 是如何编码的这里不做阐述, 小主可以自行参照 [vue-cards](https://github.com/Eamonnzhang/vue-cards) 中的实现方式进行改造
:::

需要安装以下的依赖, 让 vue 解析 markdown

``` bash
npm i markdown-it-container -D
npm i markdown-it-decorate -D
npm i markdown-it-task-checkbox -D
npm i vue-markdown-loader -D
```

关于 vue.config.js 的配置在 [vue-cards](https://github.com/Eamonnzhang/vue-cards) 该项目中也有了, 不做阐述

这里将补充高亮 `highlight.js` 以及点击复制代码 `clipboard` 的实现方式

安装依赖

``` bash
npm i clipboard highlight.js
```

改造 `App.vue` , 以下只是列出部分代码, 小主可以根据自己的需求进行添加

``` html
<script>
    import hljs from 'highlight.js'
    import Clipboard from 'clipboard'

    const highlightCode = () => {
        const preEl = document.querySelectorAll('pre')

        preEl.forEach((el, index) => {

            hljs.highlightBlock(el)

            const lang = el.children[0].className.split(' ')[1].split('-')[1]
            const pre = el
            const span = document.createElement('span')
            span.setAttribute('class', 'code-copy')
            span.setAttribute('data-clipboard-snippet', '')
            span.innerHTML = `${lang.toUpperCase()} | COPY`
            pre.appendChild(span)

        })
    }

    export default {
        name: 'App',

        mounted() {

            if ('onhashchange' in window) {
                window.onhashchange = function(ev) {
                    let name = window.location.hash.substring(2)
                    router.push({
                        name
                    })
                }
            }

            highlightCode()

            let clipboard = new Clipboard('.code-copy', {
                text: (trigger) => {
                    return trigger.previousSibling.innerText
                }
            })
            // 复制成功执行的回调
            clipboard.on('success', (e) => {
                e.trigger.innerHTML = `已复制`
            })

        },

        updated() {

            highlightCode()

        }
    }
</script>
```

#### 生成命令

在 `package.json` 中添加以下内容, 使用命令 `yarn new:comp` 创建组件目录及其文档或者使用命令 `yarn del:comp` 即可删除组件目录及其文档

``` json
{
  "scripts": {
    "new:comp": "node scripts/create-comp.js && node build/build-entry.js",
    "del:comp": "node scripts/delete-comp.js && node build/build-entry.js"
  }
}
```

## changelog

在 `package.json` 中修改 script 字段, 接下来你懂的, 另一篇博客有介绍哦, 小主可以执行搜索

``` json
{
  "scripts": {
    "init": "npm install commitizen -g && commitizen init cz-conventional-changelog --save-dev --save-exact && npm run bootstrap",
    "bootstrap": "npm install",
    "changelog": "conventional-changelog -p angular -i CHANGELOG.md -s -r 0"
  }
}
```
