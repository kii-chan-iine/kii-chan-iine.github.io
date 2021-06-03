---
author: 星火燎原@vxhly
title: git commit 规范指南
categories: [manual]
tags: [Liunx, Protocol]
date: 2018-05-14 14:25:43
---

<Boxx changeTime="10000"/>

::: tip 前言
目前, 社区有多种 Commit message 的写法规范。本文介绍 [Angular 规范](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit#heading=h.greljkmo14y0) 是目前使用最广的写法, 比较合理和系统化, 并且有配套的工具。前前端框架 [Angular.js](https://github.com/angular/angular.js/blob/master/CONTRIBUTING.md#toc10) 采用的就是该规范。
:::
<!-- more -->

## Commit message 的作用

### 提供更多的历史信息, 方便快速浏览。

比如, 下面的命令显示上次发布后的变动, 每个 commit 占据一行。你只看行首, 就知道某次 commit 的目的。

``` bash
git log <last tag> HEAD --pretty=format:%s
```

### 可以过滤某些 commit（比如文档改动）, 便于快速查找信息

``` bash
git log <last release> HEAD --grep feature
```

### 可以直接从 commit 生成 Change log

Change Log 是发布新版本时, 用来说明与上一个版本差异的文档, 详见后文

![Change Log](https://vxhly.github.io/assets/git-commit-1.png)

### 其他优点

* 可读性好, 清晰, 不必深入看代码即可了解当前 commit 的作用。
* 为 Code Reviewing 做准备
* 方便跟踪工程历史
* 让其他的开发者在运行 git blame 的时候想跪谢
* 提高项目的整体质量, 提高个人工程素质

## Commit message 的格式

每次提交, Commit message 都包括三个部分: header, body 和 footer。

``` bash
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

其中, header 是必需的, body 和 footer 可以省略。 不管是哪一个部分, 任何一行都不得超过 72 个字符（或 100 个字符）。这是为了避免自动换行影响美观。

### Header

Header 部分只有一行, 包括三个字段: type（必需）、scope（可选）和 subject（必需）。

#### type

用于说明 commit 的类别, 只允许使用下面 7 个标识。

* **feat**: 一项新功能
* **fix**: 修复一个 bug
* **docs**: 仅仅修改文档（documentation）
* **style**:  不影响代码含义的更改（空格, 格式, 缺少分号等）
* **refactor**: 重构（即不是新增功能, 也不是修改 bug 的代码变动）
* **perf**: 改进性能的代码更改
* **test**: 添加缺失测试或更正现有测试
* **build**: 影响构建系统或外部依赖项的更改（示例范围: gulp, broccoli, npm）
* **ci**: 对CI配置文件和脚本的更改（示例范围: Travis, Circle, BrowserStack, SauceLabs）
* **chore**: 其他不修改src或测试文件的更改
* **revert**: 恢复之前的提交（即回滚）

如果 type 为 `feat` 和 `fix` , 则该 commit 将肯定出现在 Change log 之中。其他情况（docs、chore、style、refactor、test）由你决定, 要不要放入 Change log, 建议是不要。

#### scope

scope 用于说明 commit 影响的范围, 比如数据层、控制层、视图层等等, 视项目不同而不同。

例如在 Angular, 可以是 $location, $browser, $compile, $rootScope, ngHref, ngClick, ngView 等。

如果你的修改影响了不止一个 scope, 你可以使用 `*` 代替。

#### subject

subject 是 commit 目的的简短描述, 不超过 50 个字符。

其他注意事项: 

* 以动词开头, 使用第一人称现在时, 比如 change, 而不是 changed 或 changes
* 第一个字母小写
* 结尾不加句号（.）

### Body

Body 部分是对本次 commit 的详细描述, 可以分成多行。下面是一个范例。

``` markdown
More detailed explanatory text, if necessary. Wrap it to
about 72 characters or so.

Further paragraphs come after blank lines.

* Bullet points are okay, too
* Use a hanging indent

```

有两个注意点:

* 使用第一人称现在时, 比如使用 change 而不是 changed 或 changes。
* 永远别忘了第 2 行是空行
* 应该说明代码变动的动机, 以及与以前行为的对比。

### Footer

Footer 部分只用于以下两种情况: 

#### 不兼容变动

如果当前代码与上一个版本不兼容, 则 Footer 部分以 BREAKING CHANGE 开头, 后面是对变动的描述、以及变动理由和迁移方法。

``` markdown
BREAKING CHANGE: isolate scope bindings definition has changed.

    To migrate the code follow the example below:

    Before:

    scope: {
      myAttr: 'attribute',
    }

    After:

    scope: {
      myAttr: '@',
    }

    The removed `inject` wasn't generaly useful for directives so there should be no code using it.

```

#### 关闭 Issue

如果当前 commit 针对某个 issue, 那么可以在 Footer 部分关闭这个 issue 。

``` bash
Closes #234
```

#### Revert

还有一种特殊情况, 如果当前 commit 用于撤销以前的 commit, 则必须以 revert: 开头, 后面跟着被撤销 Commit 的 Header。

``` markdown
revert: feat(pencil): add 'graphiteWidth' option

This reverts commit 667ecc1654a317a13331b17617d973392f415f02.
```

Body 部分的格式是固定的, 必须写成 `This reverts commit <hash>` ., 其中的 hash 是被撤销 commit 的 SHA 标识符。

如果当前 commit 与被撤销的 commit, 在同一个发布（release）里面, 那么它们都不会出现在 Change log 里面。如果两者在不同的发布, 那么当前 commit, 会出现在 Change log 的 Reverts 小标题下面。

## Commitizen

可以使用典型的 git 工作流程或通过使用 CLI 向导 Commitizen 来添加提交消息格式。

安装

``` bash
npm install -g commitizen
```

然后, 在项目目录里, 运行下面的命令, 使其支持 Angular 的 Commit message 格式。

``` bash
commitizen init cz-conventional-changelog --save --save-exact
```

以后, 凡是用到 `git commit` 命令, 一律改为使用 `git cz` 。这时, 就会出现选项, 用来生成符合格式的 Commit message。

## 生成 Change log

如果你的所有 Commit 都符合 Angular 格式, 那么发布新版本时, Change log 就可以用脚本自动生成。生成的文档包括以下三个部分: 

``` markdown
New features

Bug fixes

Breaking changes.
```

每个部分都会罗列相关的 commit , 并且有指向这些 commit 的链接。当然, 生成的文档允许手动修改, 所以发布前, 你还可以添加其他内容。

[conventional-changelog](https://github.com/ajoslin/conventional-changelog) 就是生成 Change log 的工具, 运行下面的命令即可。

``` bash
npm install -g conventional-changelog-cli
cd my-project
conventional-changelog -p angular -i CHANGELOG.md -s -r 0
```

当然你也可以在你的 `package.json` 中添加快捷 script

``` json
{
  "script": {
    "changelog": "conventional-changelog -p angular -i src/CHANGELOG.md -s -r 0"
  }
}
```
