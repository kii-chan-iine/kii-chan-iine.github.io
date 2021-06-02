---
author: 星火燎原@vxhly
title: vscode 配置之 vue.js 篇
categories: [vuejs]
tags: [Vue.js, Node.js]
date: 2020-02-16 13:19:19
---

<Boxx changeTime="10000"/>

::: tip 前言
学习 vue.js 的小伙伴前期一定很苦恼吧, 接下这篇文章带你从配置编辑器入手, 提升你的代码开发效率, 保证代码的整洁度, 让你感受到写代码也能飞起来。主要是推荐几款我本人用起来比较方便的 vscode 插件, 以及这些插件的简单配置。
:::
<!-- more -->

## 先了解 Vue.js 的代码风格

这里比较推荐的是, 官方提供的代码风格, 上地址 [风格指南](https://cn.vuejs.org/v2/style-guide/)

懒癌晚期的同志们看完之后, 一定会觉得这什么呀, 怎么这么多, 只是想愉快的写代码并不想关心这些无谓的代码风格, 那么到底有没有插件帮助这些懒癌晚期又有强迫症的童鞋的 vscode 插件呢？答案当然是有的啦, 接着向下看

## Vue.js 项目代码开启 ESlint

通常使用 `vue create [projectName]` 创建项目时会询问你使用哪种风格的 ESlint, 这时候你就应该毫不犹豫的选择 `ESLint + Standard config` 这种风格, 因为这种风格的代码更符合官方推荐的代码风格

![vscode-config](https://vxhly.github.io/assets/vscode-config-1.png)

当然对于强迫症的童鞋这些是不够的, 能够更加符合官方推荐风格指南你还需要阅读 [eslint-plugin-vue](https://eslint.vuejs.org/rules/)

不用担心你的项目没有安装 `eslint-plugin-vue` 这一插件, 因为官方已经帮助我们集成了, 我们只需进行简单的配置即可

默认生成的项目 eslint 检测规则是这样的, 米娜桑可以查阅项目目录下的 `.eslintrc.js`

``` javascript
module.exports = {
    root: true,
    env: {
        node: true
    },
    extends: [
        'plugin:vue/essential',
        '@vue/standard'
    ],
    parserOptions: {
        parser: 'babel-eslint'
    },
    rules: {
        'no-console': process.env.NODE_ENV === 'production' ? 'error' : 'off',
        'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'off'
    }
}
```

仔细看看, 这个规则只是开启了官方推荐的风格指南中的 A 级风格, 那么想要开启 B 级风格以及 C 级风格要如何做呢？前面让大家阅读的 [eslint-plugin-vue](https://eslint.vuejs.org/rules/) 这个就派上用场了, 查阅这里的配置指南, 配置出符合自己团队的代码风格

下面贴出我个人使用的代码风格配置（强迫症患者福音）

``` javascript
module.exports = {
    root: true,
    env: {
        node: true
    },
    'extends': [
        // 配置参考 https://eslint.vuejs.org/rules/
        'plugin:vue/base',
        'plugin:vue/essential',
        'plugin:vue/strongly-recommended',
        'plugin:vue/recommended',
        '@vue/standard'
    ],
    rules: {
        // 以下风格参照 https://cn.vuejs.org/v2/style-guide/index.html
        'no-console': process.env.NODE_ENV === 'production' ? 'error' : 'off',
        'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'off',
        // 配置参考 https://eslint.vuejs.org/rules/
        'vue/attributes-order': ['error', {
            'order': [
                'DEFINITION', // e.g. 'is'
                'LIST_RENDERING', // e.g. 'v-for item in items'
                'CONDITIONALS', // e.g. 'v-if', 'v-else-if', 'v-else', 'v-show', 'v-cloak'
                'RENDER_MODIFIERS', // e.g. 'v-once', 'v-pre'
                'GLOBAL', // e.g. 'id'
                'UNIQUE', // e.g. 'ref', 'key', 'v-slot', 'slot'
                'TWO_WAY_BINDING', // e.g. 'v-model'
                'OTHER_DIRECTIVES', // e.g. 'v-custom-directive'
                'OTHER_ATTR', //  e.g. 'custom-prop="foo"', 'v-bind:prop="foo"', ':prop="foo"'
                'EVENTS', // e.g. '@click="functionCall"', 'v-on="event"'
                'CONTENT' // e.g. 'v-text', 'v-html'
            ]
        }],
        'vue/order-in-components': ['error', {
            'order': [
                'el',
                'name',
                'parent',
                'functional',
                ['delimiters', 'comments'],
                ['components', 'directives', 'filters'],
                'extends',
                'mixins',
                'inheritAttrs',
                'model',
                ['props', 'propsData'],
                'fetch',
                'asyncData',
                'data',
                'computed',
                'watch',
                'LIFECYCLE_HOOKS',
                'methods',
                'head',
                ['template', 'render'],
                'renderError'
            ]
        }],
        'vue/no-v-html': 'off',
        'vue/array-bracket-spacing': 'error',
        'vue/arrow-spacing': 'error',
        'vue/block-spacing': 'error',
        'vue/camelcase': 'error',
        'vue/comma-dangle': 'error',
        'vue/component-name-in-template-casing': ['error', 'kebab-case', {
            'registeredComponentsOnly': false,
            'ignores': []
        }],
        'vue/eqeqeq': 'error',
        'vue/key-spacing': 'error',
        'vue/match-component-file-name': ['error', {
            'extensions': ['jsx'],
            'shouldMatchCase': false
        }],
        'vue/no-restricted-syntax': 'error',
        'vue/object-curly-spacing': 'error',
        'vue/require-direct-export': 'error',
        'vue/space-infix-ops': 'error',
        'vue/space-unary-ops': 'error',
        'vue/v-on-function-call': ['error', 'never']
    },
    parserOptions: {
        parser: 'babel-eslint'
    }
}
```

::: tip 小结
这一节呢, 主要是介绍项目的 ESlint 配置指南, 但是这些配置项对于懒癌晚期的同志们, 真是让他们一个脑袋两个大, 不要急, 看完下面的插件安利, 你会跪下来喊爸爸的
:::

## vscode 插件推荐

### Vetur

::: tip 插件介绍
这个插件是 vue.js 官方团队开发的 vue.js 语法高亮的一款插件
:::

[Vetur 插件安装地址](https://marketplace.visualstudio.com/items?itemName=octref.vetur)

无需进行过多的配置, 基本上默认的就足够了, 这边推荐的是修改以下这一配置项

![vscode-config](https://vxhly.github.io/assets/vscode-config-2.png)

### ESlint 

::: tip 插件介绍
保证你的代码整洁度的灵魂插件, 配置完成之后即可感受到写代码飞起来
:::

[ESlint 插件安装地址](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint)

配置项需要在 `settings.json` 中添加配置项的, 以下贴出推荐的配置

``` json
{
  "eslint.autoFixOnSave": true,
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    {
      "language": "html",
      "autoFix": true
    },
    {
      "language": "vue",
      "autoFix": true
    },
    "html",
    "vue"
  ]
}
```

这边再贴一段关于自动保存即可格式化你的代码配置

``` json
{
  "eslint.autoFixOnSave": true,
  "files.autoSaveDelay": 500,
  "files.autoSave": "onFocusChange",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  }
}
```

::: tip 小结
基本上配置这些就足够提升写代码的效率了, 不用再去强行记忆那些所谓的代码风格, 使你更加关注于写代码带来的快乐
:::

### Beautify

::: tip 插件介绍
这是配合 ESlint 的一款代码格式化插件, 无需额外的配置
:::

[Beautify 插件安装地址](https://marketplace.visualstudio.com/items?itemName=HookyQR.beautify)

### Add jsdoc comments

::: tip 插件介绍
这是用于编写 javascript 注释文档的辅助插件, 无需额外的配置
::: 

[Add jsdoc comments 插件安装地址](https://marketplace.visualstudio.com/items?itemName=stevencl.addDocComments)

### Auto Close tag

::: tip 插件介绍
自动闭合 HTML 标签的插件, 无需额外的配置
:::

[Auto Close tag 插件安装地址](https://marketplace.visualstudio.com/items?itemName=formulahendry.auto-close-tag)

### Auto Rename Tag

::: tip 插件介绍
自动重命名 HTML 标签的插件, 无需额外的配置
:::

[Auto Rename Tag 插件安装地址](https://marketplace.visualstudio.com/items?itemName=formulahendry.auto-rename-tag)

### Bracket Pair Colorizer

::: tip 插件介绍
高亮显示括号, 无需额外的配置
:::

[Bracket Pair Colorizer 插件安装地址](https://marketplace.visualstudio.com/items?itemName=CoenraadS.bracket-pair-colorizer)

### Color Highlight

::: tip 插件介绍
HTML 的颜色值高亮, 无需额外的配置
:::

[Color Highlight 插件安装地址](https://marketplace.visualstudio.com/items?itemName=naumovs.color-highlight)

### GitLens

::: tip 插件介绍
增强 vscode 中的 git 功能, 无需额外的配置
:::

[GitLens 插件安装地址](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)

### TabNine

::: tip 插件介绍
使用 AI 帮助你提升写代码效率的一款插件
:::

[TabNine 插件安装地址](https://marketplace.visualstudio.com/items?itemName=TabNine.tabnine-vscode)

配置过程也很简单, 新建一个文件输入 `TabNine::config` 会自动开启一个 WEB 配置页面, 按照下图开启配置即可, 至于 `Activation key` 需要通过邮箱验证来获取的

![vscode-config](https://vxhly.github.io/assets/vscode-config-3.png)
