---
author: 星火燎原@vxhly
title: 使用 Hexo+NexT 在 Github 上搭建静态博客
categories: [nodejs]
tags: [Liunx, Node.js, Hexo]
date: 2016-08-03 13:11:30
---

<Boxx changeTime="10000"/>

::: tip 前言
Hexo 是一个快速, 简单而高效的静态博客框架, 本文将介绍如何在 GitHub 上使用 Hexo+NexT 搭建静态博客。
:::
<!-- more -->

## 准备

::: tip 提示
你需要在 [Github](http://github.com/) 上创建一个属于自己的账户, 然后新建一个仓库（new repository）, 并命名为 YourSiteName.github.io, 此时 Github 会帮助你初始化一个静态网页, 你可以根据自己的喜好选择一些模版（这都不是重点）, 接着尝试访问下你所建的站点, 成功后就可以开始动工了。
::: 

### Node.js 环境

需安装 nvm, 任何版本皆可

``` bash
nvm 5
nvm alias defaults 5
```

### 安装 Git

如果你的系统没有 git, 则必装

``` bash
sudo apt-get install git
```

## Hexo 初始化博客框架

::: tip 提示
参考链接<br>
[Hexo 官网](http://hexo.io/zh-cn/)<br>
[Hexo 配置文档](http://hexo.io/zh-cn/docs/)<br>
[Hexo 相关指令](http://hexo.io/zh-cn/docs/commands.html)<br>
[Hexo 主题](http://github.com/hexojs/hexo/wiki/Themes)
::: 

### 安装 Hexo

``` bash
npm install -g hexo-cli
```

### 初始化框架

``` bash
hexo init [yourFolder]
cd [folderName]
npm install
```

#### 新建文章

``` bash
hexo new "Hello World"
```

在 `./source/_post` 里添加 `hello-world.md` 文件, 之后新建的文章都将存放在此目录下。

#### 新建页面

``` bash
hexo new page tags
hexo new page categories
```

在 `./source/` 里添加 `tags` 和 `categories` 目录, 即标签页和分类页

#### 取消标签页和分类页的评论功能

标签页 `./source/categories/index.md` 

``` markdown
---
title: null
date:
type: "categories"
comments: false
---
```

分类页 `./source/tags/index.md` 

``` markdown
---
title: null
date:
type: "tags"
comments: false
---
```

#### 建议修改默认的文章模板

``` bash
vim ./scaffolds/post.md
```

修改成

``` markdown
---
title: {{ title }}
date: {{ date }}

categories:
tags:
---
```

### 生成静态网站

``` bash
hexo generate
```

此时会将 `/source` 的 `.md` 文件生成 `html` 文件到 `/public` 中, 形成网站的静态文件。

### 本地预览

``` bash
hexo server -p 3000
```

输入 `//localhost:3000` 即可查看网站, 若无指定端口号, 将采用默认端口 `4000` 。 使用 `Ctrl + C` 快捷键终止该进程

### 部署网站至 github

``` bash
hexo deploy
```

#### 静态化且部署简化命令

``` bash
hexo g -d
```

#### 部署网站前请配置

`./_config.yml` 

``` yaml
deploy:
    type: git
repo: git @github.com: youName / youName.github.io.git
branch: master
```

#### 部署前请执行

``` bash
npm install hexo-deployer-git --save
hexo clean
hexo generate
```

## 使用 NexT 主题让站点更酷炫

::: tip 提示
参考链接<br>
[NexT 参考文档](http://theme-next.iissnan.com/)<br>
[NexT 主题配置](http://theme-next.iissnan.com/theme-settings.html)<br>
[NexT 第三方服务](http://theme-next.iissnan.com/third-party-services.html)<br>
[更多的配置问题](http://theme-next.iissnan.com/faqs.html)
:::

### 下载主题

``` bash
cd <yourFolder>
git clone https://github.com/iissnan/hexo-theme-next themes/next
```

### 修改配置文件

``` bash
vim ./_config.yml
```

修改

``` bash
theme: next
```

### 验证主题效果

``` bash
hexo s --debug
```

访问 `//localhost:4000` , 确保站点正确运行

## 站内搜索

### 安装 hexo-generator-search

``` bash
npm install hexo-generator-search --save
```

### 编辑配置文件

`./_config.yml` 

在任意位置添加以下内容

``` yaml
search:
    path: search.xml
field: post
format: html
limit: 10000
```

### 编辑主题配置文件

`./themes/next/_config.yml` 

``` yaml
#
Local search
local_search:
    enable: true
```

## 开启字数统计和阅读时长

### 安装 hexo-wordcount

``` bash
npm install hexo-wordcount --save
```

### 编辑主题配置文件

`./themes/next/_config.yml` 

``` yaml
post_wordcount:
    item_text: true
wordcount: true
min2read: true
```

# 我的配置（只写出部分配置）

## 站点配置文件

`./_config.yml` 

``` yaml
#
Site
language: zh - Hans

# search
search:
    path: search.xml
field: post

# URL
url: . /

    #Writing
highlight:
    enable: true
line_number: true
auto_detect: true
tab_replace:

    #Pagination
per_page: 5
pagination_dir: page

# Extensions
theme: next

# Deployment
deploy:
    type: git
repo: git @github.com: vxhly / vxhly.github.io.git
branch: master
```

## 主题配置文件文件

`./themes/next/_config.yml` 

``` yaml
# Specify the date when the site was setup
since: 2016

# Footer `powered-by` and `theme-info` copyright
copyright: true

menu:
  home: /
  categories: /categories
  #about: /about
  archives: /archives
  tags: /tags
  #sitemap: /sitemap.xml
  commonweal: /404.html

# Schemes
#scheme: Muse
scheme: Mist
#scheme: Pisces

# Social Links
social:
  #LinkLabel: Link
  GitHub: //github.com/vxhly
  Weibo: //weibo.com/5039312152

# Automatically scroll page to section which is under <!-- more --> mark.
scroll_to_more: true

# Automatically Excerpt. Not recommand.
auto_excerpt:
  enable: true
  length: 150

# Code Highlight theme
highlight_theme: night eighties

# Duoshuo ShortName
duoshuo_shortname: vxhly
  enable: true
  length: 150

# Post wordcount display settings
post_wordcount:

    item_text: true

wordcount: true
min2read: true

# Share
duoshuo_share: true

# Make duoshuo show UA
duoshuo_info:
  ua_enable: true
  admin_enable: true
  user_id: 13846473
  admin_nickname: Author
```

## 腾讯公益 404 页面

`./themes/next/source/404.html` 

``` html
<!DOCTYPE HTML>
<html>

<head>
    <meta http-equiv="content-type" content="text/html;charset=utf-8;" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
    <meta name="robots" content="all" />
    <meta name="robots" content="index,follow" />
</head>

<body>
    <script type="text/javascript" src="//www.qq.com/404/search_children.js" charset="utf-8" homePageUrl="/" homePageName="回到我的主页">
    </script>
</body>

</html>
```

