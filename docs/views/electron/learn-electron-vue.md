---
author: 星火燎原@vxhly
title: 初学 electron-vue
categories: [electron]
tags: [Electron, Vue, Node.js] 
date: 2019-11-05 15:29:47
---

<Boxx changeTime="10000"/>

::: tip 前言
记录学习 electron-vue 时遇到的一些坑, 从报错或者日常需求上出发找寻解决方案。
:::
<!-- more -->

## 坑一

一般的高版本的 node, 大于12的版本时候。初步运行 electron-vue 项目时候会报错！

::: danger 报错特征
ERROR in Template execution failed: ReferenceError: process is not defined

ERROR in   ReferenceError: process is not defined

+ index.ejs:102

  /Users/codeman/github/my-project/src/index.ejs:102:2

+ index.ejs:107 module.exports

  /Users/codeman/github/my-project/src/index.ejs:107:3

+ index.js:284

  [my-project]/[html-webpack-plugin]/index.js:284:18

:::

### 解决方案

修改 `.electron-vue/webpack.renderer.config.js` 和 `.electron-vue/webpack.web.config.js` 如下：

 `.electron-vue/webpack.renderer.config.js`

``` javascript
new HtmlWebpackPlugin({
    filename: 'index.html',
    template: path.resolve(__dirname, '../src/index.ejs'),
    minify: {
        collapseWhitespace: true,
        removeAttributeQuotes: true,
        removeComments: true
    },
    templateParameters(compilation, assets, options) {
        return {
            compilation: compilation,
            webpack: compilation.getStats().toJson(),
            webpackConfig: compilation.options,
            htmlWebpackPlugin: {
                files: assets,
                options: options
            },
            process,
        };
    },
    nodeModules: process.env.NODE_ENV !== 'production' ?
        path.resolve(__dirname, '../node_modules') : false
})
```

 `.electron-vue/webpack.web.config.js`

``` javascript
new HtmlWebpackPlugin({
    filename: 'index.html',
    template: path.resolve(__dirname, '../src/index.ejs'),
    templateParameters(compilation, assets, options) {
        return {
            compilation: compilation,
            webpack: compilation.getStats().toJson(),
            webpackConfig: compilation.options,
            htmlWebpackPlugin: {
                files: assets,
                options: options
            },
            process,
        };
    },
    minify: {
        collapseWhitespace: true,
        removeAttributeQuotes: true,
        removeComments: true
    },
    nodeModules: false
})
```

## 坑二

在初步使用 `electron-builder` 编译 electron-vue 项目时候会报错（PS：一般出现在 MacOS 下）

::: danger
Error: Exit code: 2. Command failed: /usr/bin/perl /private/var/folders/mj/n34f_bp95zq2_1fwll3bq70m0000gn/T/t-51hskU/1-dmgProperties.pl
Can't locate Mac/Memory.pm in @INC (you may need to install the Mac:: Memory module) (@INC contains: /Library/Perl/5.18/darwin-thread-multi-2level /Library/Perl/5.18 /Network/Library/Perl/5.18/darwin-thread-multi-2level /Network/Library/Perl/5.18 /Library/Perl/Updates/5.18.4 /System/Library/Perl/5.18/darwin-thread-multi-2level /System/Library/Perl/5.18 /System/Library/Perl/Extras/5.18/darwin-thread-multi-2level /System/Library/Perl/Extras/5.18 .) at /private/var/folders/mj/n34f_bp95zq2_1fwll3bq70m0000gn/T/t-51hskU/1-dmgProperties.pl line 4.
BEGIN failed--compilation aborted at /private/var/folders/mj/n34f_bp95zq2_1fwll3bq70m0000gn/T/t-51hskU/1-dmgProperties.pl line 4.
:::

### 解决方案

升级你的 `electron-builder` 依赖包

``` bash
npm install electron-builder@latest -D
```

## 坑三

默认项目配置的 sass/scss 编译器为 `node-sass` , 该编译器在 Windows 环境下及其的不友好, 所以这里比较推荐的是使用 `Dart Sass` , 有兴趣的小伙伴传送门在这里 [为啥要用 Dart sass](https://vxhly.github.io/views/manual/usage-of-docker.html#%E6%8B%93%E5%B1%95-2)

### 解决方案

删除你的原有的 `node-sass` 依赖项, `node-sass` 和 `Dart Sass` 几乎无缝转换, 所以不用担心兼容问题

``` bash
npm install sass -D
```

## 坑四

CI 自动编译时会报 `Node.js` 的版本过低, 导致编译错误, `Travis CI` 平台上的日志特征

::: danger
$ yarn
yarn install v1.19.2
warning You are using Node "7.10.1" which is not supported and may encounter bugs or unexpected behavior. Yarn supports the following semver range: "^4.8.0 || ^5.7.0 || ^6.2.2 || >=8.0.0"
warning package.json: No license field
warning demo@0.0.1: No license field
[1/4] Resolving packages... 
[2/4] Fetching packages... 
error getmac@4.3.0: The engine "node" is incompatible with this module. Expected version ">=8". Got "7.10.1"
error Found incompatible module.
info Visit https://yarnpkg.com/en/docs/cli/install for documentation about this command.
The command "yarn" failed and exited with 1 during .
Your build has been stopped.
:::

### 解决方案

建议在 `.travis.yml` 和 `appveyor.yml` 中修改 `Node.js` 的版本

 `.travis.yml`

``` yaml
osx_image: xcode8.3
sudo: required
dist: trusty
language: c
matrix:
  include:

  + os: osx

  # - os: linux
    env: CC=clang CXX=clang++ npm_config_clang=1
    compiler: clang
cache:
  directories:

  + node_modules
  + "$HOME/.electron"
  + "$HOME/.cache"

addons:
  apt:
    packages:

    - libgnome-keyring-dev
    - icnsutils

before_install:

* mkdir -p /tmp/git-lfs && curl -L https://github.com/github/git-lfs/releases/download/v1.2.1/git-lfs-$([

  "$TRAVIS_OS_NAME" == "linux" ] && echo "linux" || echo "darwin")-amd64-1.2.1.tar.gz
  | tar -xz -C /tmp/git-lfs --strip-components 1 && /tmp/git-lfs/git-lfs pull

* if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then sudo apt-get install --no-install-recommends -y icnsutils graphicsmagick xz-utils; fi

install:

* nvm install 12
* curl -o- -L https://yarnpkg.com/install.sh | bash
* source ~/.bashrc
* npm install -g xvfb-maybe
* yarn install

script:

* yarn run release

branches:
  only:

  + master

```

 `appveyor.yml`

``` yaml
version: 0.1.{build}

branches:
  only:

    - master

image: Visual Studio 2017
platform:

  + x64

cache:

  + node_modules
  + '%APPDATA%\npm-cache'
  + '%USERPROFILE%\.electron'
  + '%USERPROFILE%\AppData\Local\Yarn\cache'

init:

  + git config --global core.autocrlf input

install:

  + ps: Install-Product node 12 x64
  + git reset --hard HEAD
  + yarn install
  + node --version

build_script:

  + yarn run release

test: off

```

## 坑五

CI 自动编译成功但是并未成功发布到 GitHub 上, `Appveyor CI` 平台上的日志特征

::: danger
  • publishing      publisher=Github (owner: vxhly, project: demo, version: 0.0.1)
  • uploading       file=demo-setup-0.0.1.exe.blockmap provider=GitHub
  • uploading       file=demo-setup-0.0.1.exe provider=GitHub
  • skipped publishing  file=demo-setup-0.0.1.exe.blockmap reason=release doesn't exist and not created because "publish" is not "always" and build is not on tag tag=v0.0.1 version=0.0.1
  • skipped publishing  file=demo-setup-0.0.1.exe reason=release doesn't exist and not created because "publish" is not "always" and build is not on tag tag=v0.0.1 version=0.0.1
  • skipped publishing  file=latest.yml reason=release doesn't exist and not created because "publish" is not "always" and build is not on tag tag=v0.0.1 version=0.0.1
Done in 128.39s.
Updating build cache... 
Cache 'node_modules' - Updated
Cache entry not found: C:\Users\appveyor\AppData\Roaming\npm-cache
Cache 'C:\Users\appveyor\.electron' - Updated
Cache 'C:\Users\appveyor\AppData\Local\Yarn\cache' - Updated
Build success
:::

### 解决方案

可能原因一：自带的 `npm run build` 这个脚本让CI去执行构建, 但是发现无法自动上传到 GitHub 的 release 里
可能原因二：未发布 tag

步骤一：在 `package.json` 中配置运行 `script`

``` json
{
  "release": "node .electron-vue/build.js && electron-builder",
  "patch": "npm version patch && git push origin master && git push origin --tags",
  "minor": "npm version minor && git push origin master && git push origin --tags",
  "major": "npm version major && git push origin master && git push origin --tags"
}
```

步骤二：修改 `.travis.yml` 和 `appveyor.yml` 文件（PS：在坑四中已体现, 这边就不重复了）

## 坑六

在 Windows 下打包之后, 未能正确的显示安装路径而是直接安装

### 解决方案

在 `package.json` 中修改

``` json
{
  "build": {
    "win": {
      "icon": "build/icons/icon.ico",
      "target": [
        {
          "target": "nsis",
          "arch": [
            "ia32"
          ]
        }
      ]
    },
    "nsis": {
      "oneClick": false,
      "allowElevation": true,
      "allowToChangeInstallationDirectory": true,
      "installerIcon": "build/icons/icon.ico",
      "uninstallerIcon": "build/icons/icon.ico",
      "installerHeaderIcon": "build/icons/icon.ico",
      "createDesktopShortcut": true,
      "createStartMenuShortcut": true,
      "displayLanguageSelector": true,
      "multiLanguageInstaller": true,
      "installerLanguages": [
        "en_US",
        "zh_CN"
      ],
      "warningsAsErrors": false
    }
  }
}
```

## 坑七

`MacOS` 下打包未能正确识别到签名证书, 报错特征

::: danger
  • electron-builder  version=21.2.0 os=19.0.0
  • loaded configuration  file=package.json ("build" field)
  • writing effective config  file=build/builder-effective-config.yaml
  • rebuilding native dependencies  dependencies=grpc@1.24.2 platform=darwin arch=x64
  • rebuilding native dependency  name=grpc version=1.24.2
  • packaging       platform=darwin arch=x64 electron=2.0.18 appOutDir=build/mac
  • skipped macOS application code signing  reason=cannot find valid "Developer ID Application" identity or custom non-Apple code signing certificate, see https://electron.build/code-signing allIdentities=

                                                   0 identities found
                                              
                                                Valid identities only
                                                   0 valid identities found

:::

### 解决方案

使用 `X-code` 进行 mac 版本上代码签名

1. 打开xcode主界面
2. Xcode==>Preferences…

![electron-mac](https://vxhly.github.io/assets/electron-1.png)

3. Accounts==>Apple IDs==>Manage Certificates…, 如果没有登录的话, 先登录Apple ID, 注意一定要登录开发者账号。登录成功后再进行Manage Certificates；

![electron-mac](https://vxhly.github.io/assets/electron-2.png)

4. 添加“+”Developer ID Application,注意一定要添加Developer ID Application到钥匙串中, 不要选错了。

![electron-mac](https://vxhly.github.io/assets/electron-3.png)

5. 成功添加到钥匙串我的证书中, 经过以上的步骤可以成功添加一个证书到钥匙串中的我的证书中。

![electron-mac](https://vxhly.github.io/assets/electron-4.png)

![electron-mac](https://vxhly.github.io/assets/electron-5.png)

6. 正确签名之后, 打包成功！

