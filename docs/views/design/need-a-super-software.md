---
author: 星火燎原@vxhly
title: 需要一款称手的编辑器
categories: [design]
tags: [WEB, Code]
date: 2016-08-07 16:49:47
---

<Boxx changeTime="10000"/>

::: tip 前言
以下介绍的编辑器是跨平台, 无论是 Windows, 还是 Liunx, 甚至是 Mac, 都可以成为程序员称手的利器。
:::
<!-- more -->

## Atom

开源。功能强大, 是 Github 团队推出的简洁编辑器, 自带 `Node.js` 环境。

### 下载地址

官网下载地址: [Atom](http://github.com/atom/atom/releases)<br>
官网下载地址下不下来点击以下链接下载: [atom-amd64.deb](http://oss-blog.test.upcdn.net/atom-amd64.deb?attname=) [AtomSetup.exe](http://oss-blog.test.upcdn.net/AtomSetup.exe?attname=)

### 安装插件

使用 `Ctrl+,` 调出设置页面, 在 `install` 中查询插件并安装。自带 `Node.js` 环境。部分插件支持自定义设置。

### 插件推荐

* **emmet** => 前端开发者必备
* **atom-beautify** => 格式化编程语言文件, 支持的语法较广, 如 `PHP` 、 `Python` 、 `Ruby` 、 `Java` 、 `C` 等
* **atom-html-perview** => `html` 文件实时预览
* **atom-minify** => 压缩 `Html` 、 `CSS` 文件
* **autoprefixer** => `CSS3` 私有前缀自动补全插件
* **csscomb** => `CSS` 属性自动排序
* **docblockr** => 可以自动生成 `PHPDoc` 风格的注释
* **atom-ternjs** => `javascript` 代码提示
* **autocomplete-paths** => 引入文件时, 自动补全路径
* **language-markdown** => `markdown` 文件语法高亮
* **language-vue** => `vue` 文件语法高亮
* **logo-file-icons** => 文件图标
* **markdown-scroll-sync** => `markdown` 文件预览自动滚动
* **simplified-chinese-menu** => `Atom` 简体中文语言包
* **vue-autocomplete** => `vue` 文件代码自动提示
* **vue-format** => `vue` 文件自动格式化
* **sync-settings** => atom 插件备份(需要 Githunb Access Token 和 Gist Id)

## Brackets

开源。功能强大, 是一个非常优秀的 `WEB` 设计和前端开发的源代码编辑器。当然也支持 `C` 、 `Java` 、 `Python` 等编程语言。支持实时预览。自带 `Node.js` 环境。

### 下载地址

[Brackets](http://github.com/adobe/brackets/releases)<br>
官网下载地址下不下来点击以下链接下载: [Brackets. Release.1.8.msi](http://oss-blog.test.upcdn.net/Brackets.Release.1.8.msi?attname=)<br>
[Brackets. Release.1.8.64-bit.deb](http://oss-blog.test.upcdn.net/Brackets.Release.1.8.64-bit.deb?attname=)

### 安装插件

部分插件是使用国外源的, 很容易被墙的。所以系统最好要有个代理服务器, 这里推荐 [高梯子](http://www.gaotizi.com/aff.php?aff=133)。

### 如何安装

最快速的安装 `Bracket` s 扩展的方法是使用扩展管理器（ `Extension Manager` ）– 直接在 `Brackets` 的工具栏选择 `File > Extension Manager` 即可。

### 修改界面语言

改成中文。 Debug->Switch Language

### 插件推荐

* **Emmet** => 前端开发者必备
* **Autoprefixer** => `CSS3` 私有前缀自动补全插件
* **CSScomb** => `CSS` 属性自动排序
* **Brackets Icons** => 为你的 `Brackets` 边栏添加文件图标
* **Brackets CSS Class Code hint** => 输入 `class` 或 `id` 时会出现 `css` 里面的 `class／id`
* **JS CSS Minifier** => 压缩 `JS` 和 `CSS` 文件
* **Beautify** => 这个扩展可以让你的 `HTML` 、 `CSS` 、 `JavaScript` 等代码格式化一致, 从而提升可读性
* **Simple To-Do** => 使用 `Simple To-Do` 可以确保你不会忘记项目的任务。它允许你在 `Brackets` 中为每一个项目创建 `TODO` 列表。
* **Markdown Preview** => `Markdown` 文件实时预览
* **Markdown Toolbar** => `Markdown` 文件的快捷工具栏
* **JavaScript Globals** => `javascript` 全局变量高亮
* **Brackets Vue** => `Vue` 文件语法高亮

## Sublime Text 3

非开源。功能强大, 可以编译 `C` 、 `Java` 、 `Python` 等编程语言。 `Liunx` 下启动可以在终端输入 `subl` 。部分插件依赖于 `Node.js` 。

### 下载地址

官网下载地址: [Sublime Text 3](http://www.sublimetext.com/3)<br>
官网下载地址下不下来点击以下链接下载: [Sublime Text Build 3126 Setup.exe](http://oss-blog.test.upcdn.net/Sublime%20Text%20Build%203126%20Setup.exe?attname=)<br>
[Sublime Text Build 3126 x64 Setup.exe](http://oss-blog.test.upcdn.net/Sublime%20Text%20Build%203126%20x64%20Setup.exe?attname=)

### 安装插件

### 直接安装: 

安装 Sublime text 3 插件很方便, 可以直接下载安装包解压缩到 Packages 目录（菜单->preferences->packages）

### 使用 Package Control 安装

按 `Ctrl+` 调出 `console` 粘贴以下代码到底部命令行并回车: 

``` javascript
import urllib.request, os;
pf = 'Package Control.sublime-package';
ipp = sublime.installed_packages_path();
urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler()));
open(os.path.join(ipp, pf), 'wb').write(urllib.request.urlopen('http://sublime.wbond.net/' + pf.replace(' ', '%20')).read())
```

重启 `Sublime Text 3` 。 如果在 `Perferences` -> `package settings` 中看到 `package control` 这一项, 则安装成功。 顺便贴下 Sublime Text2 的代码。

``` javascript
import urllib2, os;
pf = 'Package Control.sublime-package';
ipp = sublime.installed_packages_path();
os.makedirs(ipp) if not os.path.exists(ipp)
else None;
urllib2.install_opener(urllib2.build_opener(urllib2.ProxyHandler()));
open(os.path.join(ipp, pf), 'wb').write(urllib2.urlopen('http://sublime.wbond.net/' + pf.replace(' ', '%20')).read());
print('Please restart Sublime Text to finish installation')
```

### 用组件安装插件的方法: 

1. 按下 `Ctrl+Shift+P` 调出命令面板
2. 输入 `install` 调出 Install Package 选项并回车, 然后在列表中选中要安装的插件。

### 注意

国内使用 `Sublime Text 3` , 经常可能遇到无法安装可用插件问题, 可 `remove` 掉 `Package Control` 重新安装下；如遇到连 `Package Control` 也无法安装, 则可以在别处拷贝一份关于 `Package Control` 的文件( `Package Control.sublime-package` )存放于 `Installed Packages` 目录之下即可

### 插件推荐

* **Emmet** => 前端开发者必备
* **Autoprefixer** => `CSS3` 私有前缀自动补全插件
* **CSScomb** => `CSS` 属性自动排序
* **JsFormat** => 一个 `JS` 代码格式化插件
* **Tag** => 格式化 `Html/Xml` 代码
* **phpFormat** => 格式化 `php` 代码
* **HTML-CSS-JS Prettify** => 一款集成了格式化（美化）html、css、js 三种文件类型的插件, 依赖于 `Node.js`
* **Minify** => 自动压缩 `JS` 、 `CSS` 文件
* **ColorPicker** => 颜色选择器
* **BracketHighlighter** => 类似于代码匹配, 可以匹配括号, 引号等符号内的范围
* **SublimeCodeIntel** => 代码自动提示
* **SublimeLinter** => 用于高亮提示用户编写的代码中存在的不规范和错误的写法
* **DocBlockr** => 可以自动生成 `PHPDoc` 风格的注释
* **ChineseLocalizations** => 汉化 `Sublime Text`
* **MarkDownEditing** => `markdown` 文件语法高亮
* **vue Syntax Highlight** => `vue` 文件语法高亮

## Visual Studio Code

微软开发的一款编辑器, 免费而且跨平台, 可以编写 `C` 、 `Java` 、 `Python` 等代码, 在界面上与 Sublime Text 非常相似

### 下载地址

官网下载地址: [VS code](http://code.visualstudio.com/Download)

### 安装插件

使用 `Ctrl + Shift + X` 调出安装插件界面

### 插件推荐

* **HTML Snippets** => 超级实用且初级的 `H5` 代码片段以及提示
* **HTML CSS Support** => 让 `html` 标签上写 class 智能提示当前项目所支持的样式
* **Debugger for Chrome** => 让 vscode 映射 chrome 的 debug 功能, 静态页面都可以用 vscode 来打断点调试
* **vscode-icons** => 让 vscode 资源树目录加上图标
* **Path Intellisense** => 自动路劲补全, 默认不带这个功能的
* **Document this** => js 的注释模板
* **ESlint** => 代码规范性检查
* **vetur** => `vue` 语法高亮
* **Git Easy** => `git` 管理
