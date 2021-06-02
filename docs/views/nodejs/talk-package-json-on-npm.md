---
author: 星火燎原@vxhly
title: 解读 npm 中的 package.json
categories: [nodejs]
tags: [Node.js]
date: 2016-08-30 17:08:18
---

<Boxx changeTime="10000"/>

::: tip 前言
package.json 是 Node.js 应用中比较重要的部分, 它定义了项目的依赖关系, 这里将介绍该文件各个配置项的作用。
:::
<!-- more -->

## 创建 package.json 文件

::: tip 提示
安装包的信息可保持到项目的 `package.json` 文件中, 以便后续的其它的项目开发或者他人合作使用, 也说 `package.json` 在项目中是必不可少的。
:::

``` bash
npm init
```

## package.json 文件解读

### 默认值

`npm` 会根据包内容设置一些默认值。

* `"scripts": {"start": "node server.js"}`
-- 如果包的根目录有 `server.js` 文件, `npm` 会默认将 `start` 命令设置为 `node server.js` 。

* `"scripts":{"preinstall": "node-waf clean || true;node-waf configure build"}`
-- 如果包的根目录有 `wscript` 文件, `npm` 会默认将 `preinstall` 命令用 `node-waf` 进行编译。

* `"scripts":{"preinstall": "node-gyp rebuild"}`
-- 如果包的根目录有 `binding.gyp` 文件, `npm` 会默认将 `preinstall` 命令用 `node-gyp` 进行编译。

* `"contributors": [...]`
-- 如果包的根目录有 `AUTHORS` 文件, `npm` 会默认逐行按 `Name <email(url)` 格式处理, 邮箱和 `url` 是可选的。 `#号` 和 `空格` 开头的行会被忽略。

### name

::: tip 作用
在 `package.json` 中最重要的就是 `name` 和 `version` 字段。他们都是必须的, 如果没有就无法 `install` 。 `name` 和 `version` 一起组成的标识在假设中是唯一的。改变包应该同时改变 `version` 。<br>
name 是这个东西的名字。注意: 
:::

* 不要把 `node` 或者 `js` 放在名字中。因为你写了 `package.json` 它就被假定成为了 `js` , 不过你可以用 `"engine"字段` 指定一个引擎（见后文）。
* 这个名字会作为在 `URL` 的一部分、命令行的参数或者文件夹的名字。任何 `non-url-safe` 的字符都是不能用的。
* 这个名字可能会作为参数被传入 `require()` , 所以它应该比较短, 但也要意义清晰。
* 在你爱上你的名字之前, 你可能要去 `npm registry` 查看一下这个名字是否已经被使用了。<http://registry.npmjs.org/>

### version

::: tip 作用
在 `package.json` 中最重要的就是 `name` 和 `version` 字段。他们都是必须的, 如果没有就无法 `install` 。 `name` 和 `version` 一起组成的标识在假设中是唯一的。改变包应该同时改变 `version` 。<br>
`version` 必须能被 `node-semver` 解析, 它被包在 `npm` 的依赖中。（要自己用可以执行 `npm install semver` ）
:::

可用的"数字"或者"范围"见 [semver](http://npmjs.org/doc/misc/semver.html)。

### description

放简介, 字符串。方便屌丝们在 `npm search` 中搜索。

### keywords

关键字, 数组、字符串。还是方便屌丝们在 `npm search` 中搜索。

### homepage

项目官网的 `url` 。

注意: 这和 `“url”` 不一样。如果你放一个 `“url”` 字段, `registry` 会以为是一个跳转到你发布在其他地方的地址, 然后喊你滚粗。

嗯, 滚粗, 没开玩笑。

### bugs

你项目的提交问题的 `url` 和（或） `邮件地址` 。这对遇到问题的屌丝很有帮助。 差不多长这样: 

``` json
{
  "url": "https://github.com/owner/project/issues",
  "email": "project@hostname.com"
}
```

你可以指定一个或者指定两个。如果你只想提供一个 `url` , 那就不用对象了, 字符串就行。 如果提供了 `url` , 它会被 `npm bugs` 命令使用。

### license

::: tip 作用
你应该要指定一个许可证, 让人知道使用的权利和限制的。<br>
最简单的方法是, 假如你用一个像 `BSD` 或者 `MIT` 这样通用的许可证, 就只需要指定一个许可证的名字, 像这样: 
:::

``` json
{
  "license": "BSD"
}
```

如果你又更复杂的许可条件, 或者想要提供给更多地细节, 可以这样:

``` json
"licenses" : [
  {
    "type" : "MyLicense" ,
    "url" : "<https://github.com/owner/project/path/to/license>"
  }
]
```

在根目录中提供一个许可证文件也蛮好的。

### people fields: author, contributors

`author` 是一个人。 `contributors` 是一堆人的数组。 `person` 是一个有 `name` 字段, 可选的有 `url` 、 `email` 字段的对象, 像这样: 

``` json
{
  "name": "Barney Rubble",
  "email": "b@rubble.com",
  "url": "https://barnyrubble.tumblr.com/"
}
```

或者可以把所有的东西都放到一个字符串里, `npm ` `会给你解析: 

``` json
{
  "Barney Rubble <b@rubble.com> (https://barnyrubble.tumblr.com/)"
}
```

`email` 和 `url` 在两种形式中都是可选的。

也可以在你的 `npm` 用户信息中设置一个顶级的 `maintainers` 字段。

### files

`files` 是一个包含项目中的文件的数组。如果命名了一个文件夹, 那也会包含文件夹中的文件。（除非被其他条件忽略了）

你也可以提供一个 `.npmignore` 文件, 让即使被包含在 `files` 字段中得文件被留下。其实就像 `.gitignore` 一样。

### main

`main` 字段配置一个文件名指向模块的入口程序。如果你包的名字叫 `foo` , 然后用户 `require("foo")` , `main` 配置的模块的 `exports` 对象会被返回。

这应该是一个相对于根目录的文件路径。

对于大多数模块, 它是非常有意义的, 其他的都没啥。

### bin

很多包都有一个或多个可执行的文件希望被放到 `PATH` 中。 `npm` 让妈妈再也不用担心了（实际上, 就是这个功能让 `npm` 可执行的）。

要用这个功能, 给 `package.json` 中的 `bin` 字段一个命令名到文件位置的 `map` 。初始化的时候 `npm` 会将他链接到 `prefix/bin（全局初始化）` 或者 `./node_modules/.bin/（本地初始化）` 。

比如, `npm` 有: 

``` json
{
  "bin": {
    "npm": "./cli.js"
  }
}
```

所以, 当你初始化 `npm` , 它会创建一个符号链接到 `cli.js` 脚本到 `/usr/local/bin/npm` 。

如果你只有一个可执行文件, 并且名字和包名一样。那么你可以只用一个字符串, 比如: 

``` json
{
  "name": "my-program",
  "version": "1.2.5",
  "bin": "./path/to/program"
}
```

结果和这个一样: 

``` json
{
  "name": "my-program",
  "version": "1.2.5",
  "bin": {
    "my-program": "./path/to/program"
  }
}
```

### man

指定一个单一的文件或者一个文件数组供 `man` 程序使用。如果只提供一个单一的文件, 那么它初始化后就是 `man <pkgname>` 的结果, 而不管实际的文件名是神马, 比如: 

``` json
{
  "name": "foo",
  "version": "1.2.3",
  "description": "A packaged foo fooer for fooing foos",
  "main": "foo.js",
  "man": "./man/doc.1"
}
```

这样 `man foo` 就可以用到 `./man/doc.1` 文件了。

如果文件名不是以包名开头, 那么它会被冠以前缀, 下面的: 

``` json
{
  "name": "foo",
  "version": "1.2.3",
  "description": "A packaged foo fooer for fooing foos",
  "main": "foo.js",
  "man": ["./man/foo.1", "./man/bar.1"]
}
```

会为 `man foo` 和 `man foo-bar` 创建文件。

`man` 文件需要以数字结束, 然后可选地压缩后以 `.gz` 为后缀。

``` json
{
  "name": "foo",
  "version": "1.2.3",
  "description": "A packaged foo fooer for fooing foos",
  "main": "foo.js",
  "man": ["./man/foo.1", "./man/foo.2"]
}
```

会为 `man foo` 和 `man 2 foo` 创建。

### directories

`CommonJS Packages` 规范说明了几种方式让你可以用 `directorieshash` 标示出包得结构。如果看一下 `npm's package.json` , 你会看到有 `directories` 标示出 doc, lib, and man。

在未来, 这个信息可能会被用到。

### directories.lib

告诉屌丝们你的库文件夹在哪里。目前没有什么特别的东西需要用到 `lib` 文件夹, 但确实是重要的元信息。

### directories.bin

如果你指定一个 `"bin"` 目录, 然后在那个文件夹中得所有文件都会被当做 `"bin"` 字段使用。<br>
如果你已经指定了 `"bin"` 字段, 那这个就无效。

### directories.man

一个放满 `man` 页面的文件夹。贴心地创建一个 `"man"` 字段。

### directories.doc

将 `markdown` 文件放在这里。最后, 这些会被很好地展示出来, 

### directories.example

将事例脚本放在这里。某一天, 它可能会以聪明的方式展示出来。

### repository

::: tip 作用
指定你的代码存放的地方。这个对希望贡献的人有帮助。如果 git 仓库在 github 上, 那么 npm docs 命令能找到你。
:::

这样做: 

``` json
"repository" : {
  "type" : "git" ,
  "url" : "<https://github.com/isaacs/npm.git>"
}

"repository" : {
  "type" : "svn" ,
  "url" : "<http://v8.googlecode.com/svn/trunk/>"
}
```

`URL` 应该是公开的（即便是只读的）能直接被未经过修改的版本控制程序处理的 `url` 。不应该是一个 html 的项目页面。因为它是给计算机看的。

### scripts

`"scripts"` 是一个由脚本命令组成的 `hash` 对象, 他们在包不同的生命周期中被执行。 `key` 是生命周期事件, `value` 是要运行的命令。<br>
参见 [npm-scripts](http://npmjs.org/doc/misc/npm-scripts.html)

### config

`"config" hash` 可以用来配置用于包脚本中的跨版本参数。在实例中, 如果一个包有下面的配置: 

``` json
{
  "name": "foo",
  "config": {
    "port": "8080"
  }
}
```

然后有一个 `“start”` 命令引用了 `npm_package_config_port` 环境变量, 用户可以通过 `npm config set foo:port 8001` 来重写他。

参见 [npm-config](http://npmjs.org/doc/misc/npm-config.html) 和 [npm-scripts](http://npmjs.org/doc/misc/npm-scripts.html)。

### dependencies

::: tip 作用
依赖是给一组包名指定版本范围的一个 `hash` 。这个版本范围是一个由一个或多个空格分隔的字符串。依赖还可以用 `tarball` 或者 `git URL` 。<br>
请不要将测试或过渡性的依赖放在 `dependencieshash` 中。见下文的 `devDependencies` 。<br>
详见 [semver](http://npmjs.org/doc/misc/semver.html)
:::

* `version` 必须完全和 `version` 一致
* `>version` 必须比 `version` 大
* `>=version` 同上
* `<version` 同上
* `<=version` 同上
* `~version` 大约一样, 见 [semver](http://npmjs.org/doc/misc/semver.html)
* `1.2.x` , `1.2.0` , `1.2.1` , 等, 但不包括 `1.3.0`
* `//...` 见下文 `'依赖 URL'`
* `*` 所有
* `""` 空, 同 `*`
* `version1 - version2` 同 `>=version1 <=version2`
* `range1 || range2` 二选一。
* `git...` 见下文 `'依赖Git URL'`
* `user/repo` 见下文 `'GitHub URLs'`
比如下面都是合法的: 

``` json
{
  "dependencies": {
    "foo": "1.0.0 - 2.9999.9999",
    "bar": ">=1.0.2 <2.1.2",
    "baz": ">1.0.2 <=2.3.4",
    "boo": "2.0.1",
    "qux": "<1.0.0 || >=2.3.1 <2.4.5 || >=2.5.2 <3.0.0",
    "asd": "https://asdf.com/asdf.tar.gz",
    "til": "~1.2",
    "elf": "~1.2.3",
    "two": "2.x",
    "thr": "3.3.x"
  }
}
```

### 依赖 URL

::: tip 提示
可以指定一个 `tarball URL` , 这个 `tarball` 将在包被初始化的时候下载并初始化。
:::

### 依赖 Git URL

`Git urls` 可以是下面几种形式: 

* git: <https://github.com/user/project.git#commit-ish>
* git+ssh: <https://user@hostname:project.git#commit-ish>
* git+ssh: <https://user@hostname/project.git#commit-ish>
* git+<https://user@hostname/project/blah.git#commit-ish>
* git+<https://user@hostname/project/blah.git#commit-ish>

`commit-ish` 是可以被 `git checkout` 的任何 `tag、sha` 或者 `branch` 。默认为 `master` 。

### GitHub URLs

`1.1.65` 版后, 你可以仅仅用 `“user/foo-project”` 引用 `GitHub urls` , 比如: 

``` json
{
  "name": "foo",
  "version": "0.0.0",
  "dependencies": {
    "express": "visionmedia/express"
  }
}
```

### devDependencies

::: tip 作用
如果有人要使用你的模块, 那么他们可能不需要你开发使用的外部测试或者文档框架。<br>
在这种情况下, 最好将这些附属的项目列在 `devDependencies` 中。
:::

这些东西会在执行 `npm link` 或者 `npm install` 的时候初始化, 并可以像其他 `npm` 配置参数一样管理。详见 [npm-config](http://npmjs.org/doc/misc/npm-config.html)。

对于非特定平台的构建步骤, 比如需要编译 `CoffeeScript` , 可以用 `prepublish` 脚本去实现, 并把它依赖的包放在 `devDependency` 中。（注: `prepublish` 定义了在执行 `npm publish` 的时候先行执行的脚本）

比如: 

``` json
{
  "name": "ethopia-waza",
  "description": "a delightfully fruity coffee varietal",
  "version": "1.2.3",
  "devDependencies": {
    "coffee-script": "~1.6.3"
  },
  "scripts": {
    "prepublish": "coffee -o lib/ -c src/waza.coffee"
  },
  "main": "lib/waza.js"
}
```

`prepublish` 脚本会在 `publishing` 前运行, 这样用户就不用自己去 `require` 来编译就能使用。并且在开发模式中（比如本地运行 `npm install` ）会运行这个脚本以便更好地测试。

### peerDependencies

::: tip 作用
在一些场景中, 如在一个 `host` 中不必须进行 `require` 时候, 你想表现你的 `package` 与一个 `host` 工具或者库的兼容关键。这一般用来引用插件。尤其是你的模块可能要暴露一个特定的接口, 并由 `host` 文档来预期和指定。
:::

比如: 

``` json
{
  "name": "tea-latte",
  "version": "1.3.5",
  "peerDependencies": {
    "tea": "2.x"
  }
}
```

这能保证你的 `package` 可以只和 `tea` 的 `2.x 版本` 一起初始化。 

试图初始化另一个有会冲突的依赖的插件将导致一个错误。因此, 确保你的插件的需求约束越弱越好, 而不要去把它锁定到一个特定的版本。

假设这个 `host` 遵守 `semver` 规范, 只改变这个 `package` 的主版本会打破你的插件。因此, 如果你在 `package` 中用过每个 `1.x 版本` , 就用 `"^1.0"` 或者 " `1.x"` 来表示。如果你依赖于功能介绍 `1.5.2` , 用 `">= 1.5.2 < 2"` 。

### bundledDependencies

::: tip 作用
一组包名, 他们会在发布的时候被打包进去。<br>
拼成 `"bundleDependencies"` （缺 d）也可以: `optionalDependencies` 。
::: 

如果一个依赖可用, 但你希望在它安装错误的时候 `npm` 也能继续初始化, 那么你可以把它放在 `optionalDependencies hash` 中。这是一个包名到版本或者 `url` 的 `map` , 就像 `dependencies hash` 一样。只是它运行错误。

处理缺乏依赖也是你的程序的责任。比如像这样: 

``` javascript
try {
    var foo = require("foo");
    var fooVersion = require("foo/package.json").version;
} catch (er) {
    foo = null;
}
if (notGoodFooVersion(fooVersion)) {
    foo = null;
}

// .. then later in your program ..

if (foo) {
    foo.doFooThings();
}
```

`optionalDependencies` 会覆盖 `dependencies` 中同名的项, 所以通常比只放在一个地方好。

### engines

你可以指定工作的 `node` 的版本: 

``` json
{
  "engines": {
    "node": ">=0.10.3 <0.12"
  }
}
```

::: tip 作用
像 `dependensies` 一样, 如果你不指定版本或者指定 `"*"` 作为版本, 那么所有版本的 `node` 都可以。
:::

如果指定一个 `"engines"` 字段, 那么 `npm` 会需要 `node` 在里面, 如果 `"engines"` 被省略, `npm` 会假定它在 `node` 上工作。

你也可以用 `"engines"` 字段来指定哪一个 `npm` 版本能更好地初始化你的程序, 如: 

``` json
{
  "engines": {
    "npm": "~1.0.20"
  }
}
```

记住, 除非用户设置 `engine-strict` 标记, 这个字段只是建议值。

### engineStrict

如果你确定你的模块一定不会运行在你指定版本之外的 `node` 或者 `npm` 上, 你可以在 `package.json` 文件中设置 `"engineStrict":true` 。它会重写用户的 `engine-strict` 设置。

除非你非常非常确定, 否则不要这样做。如果你的 `engines hash` 过度地限制, 很可能轻易让自己陷入窘境。慎重地考虑这个选择。如果大家滥用它, 它会再以后的 `npm` 版本中被删除。

### os

你可以指定你的模块要运行在哪些操作系统中: 

``` json
{
  "os": ["darwin", "linux"]
}
```

你也可以用黑名单代替白名单, 在名字前面加上"!"就可以了: 

``` json
{
  "os": ["!win32"]
}
```

操作系统用 `process.platform` 来探测。虽然没有很好地理由, 但它是同时支持黑名单和白名单的。

### cpu

如果你的代码只能运行在特定的 `cpu` 架构下, 你可以指定一个: 

``` json
{
  "cpu": ["x64", "ia32"]
}
```

就像 `os` 选项, 你也可以黑一个架构: 

``` json
{
  "cpu": ["!arm", "!mips"]
}
```

`cpu` 架构用 `process.arch` 探测。

### preferGlobal

如果包主要是需要全局安装的命令行程序, 就设置它为 `true` 来提供一个 `warning` 给只在局部安装的人。<br>
它不会真正的防止用户在局部安装, 但如果它没有按预期工作它会帮助防止产生误会。

### private

如果你设置 `"private": true` , `npm` 就不会发布它。<br>
这是一个防止意外发布私有库的方式。如果你要确定给定的包是只发布在特定 `registry` （如内部 `registry` ）的, 用 `publishConfighash` 的描述来重写 `registry` 的 `publish-time` 配置参数。

### publishConfig

::: tip 作用
这是一个在 `publish-time` 使用的配置集合。当你想设置 `tag` 或者 `registry` 的时候它非常有用, 所以你可以确定一个给定的包没有打上 `"lastest"` 的 `tag` 或者被默认发布到全局的公开 `registry` 。
:::

任何配置都可以被重写, 但当然可能只有 `"tag"` 和 `"registry"` 与发布的意图有关。

参见 [npm-config](http://npmjs.org/doc/misc/npm-config.html) 有可以被重写的列表。

### SEE ALSO

* semver
* npm-init
* npm-version
* npm-config
* npm-config
* npm-help
* npm-faq
* npm-install
* npm-publish
* npm-rm
