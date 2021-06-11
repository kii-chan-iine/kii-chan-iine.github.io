---
author: kii
title: Gitpages搭建总结
categories: [闲言碎语]
tags: [Git]
date: 2021-06-03 19:11:30
---

<Boxx changeTime="10000"/>

::: tip 前言
这里主要讲如何利用Vue搭建自己的网站。
:::
<!-- more -->

# VuePress

1. 开始

```
# 创建项目目录blog-demo
mkdir blog-demo && cd blog-demo

# 初始化项目
npm init -y

# 本地安装VuePress
npm install -D vuepress

mkdir docs && echo '# Hello VuePress' > docs/README.md
```

2. 在 `package.json` 中添加一些 [scripts(opens new window)](https://classic.yarnpkg.com/zh-Hans/docs/package-json#toc-scripts)

这一步骤是可选的，但我们推荐你完成它。在下文中，我们会默认这些 scripts 已经被添加。

```json
{
  "name": "kii-iine",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "devDependencies": {
    "vuepress": "^1.8.2"
  },//这是添加的部分
  "scripts": {
    "docs:dev": "vuepress dev docs",
    "docs:build": "vuepress build docs"
  }
}
```

3. 在本地启动服务器

```bash
npm run docs:dev
```



```css
npm run docs:build
```

然后看文件变化 多了个node_modules
 docs  多了个 .vuepress文件夹



```go
study
+--docs
+----.vuepress
+------ dist   //打包后的文件夹
+----README.md
+--package.json
+--node_modules
```

我们在.vuepress 创建
 config.js 文件







废话不多说，上解决方案
 Step1：`npm cache clean --force`
 Step2：`rm -rf node_modules`
 Step3：`rm -rf package-lock.json`
 Step4：`npm install`
 `npm install` 成功之后再次启动 `npm run dev`/```npm run build```



---

```
git branch develop main
git checkout develop   #切换
git push --set-upstream origin develop  #第一次的时候要设置这个
```



