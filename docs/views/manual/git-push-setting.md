---
author: 星火燎原@vxhly
title: Git 远程推送配置
categories: [manual]
tags: [Liunx, Protocol]
date: 2016-07-31 16:09:21
---

<Boxx changeTime="10000"/>

::: tip 前言
Git 是一个开源的分布式版本控制系统, 可以有效、高速的处理从很小到非常大的项目版本管理。
:::
<!-- more -->

## 安装 Git

如果你的系统还没有 Git, 请通过以下命令安装

``` bash
sudo apt-get install git
```

## 创建帐号

创建你的 GitHub 帐号, 或者 Git@OSC 帐号

[GitHub](http://github.com/) 链接<br>
[Git@OSC](http://git.oschina.net/) 链接

## liunx 创建 SSH 密钥

``` bash
ssh-keygen -t rsa -C "xxxxx@xxxxx.com"
```

### 添加 public key

查看你的 public key, 并把它添加到您的账户中

如 [GitHub key](http://github.com/settings/keys) 如 [Git@OSC key](http://git.oschina.net/profile/sshkeys)

使用以下命令查看

``` bash
cat /home/youname/.ssh/id_rsa.pub
```

### 测试验证是否成功

``` bash
ssh -T git@github.com
```

或

``` bash
ssh -T git@git.oschina.net
```

返回你在 `GitHub` 或 `Git@OSC` 上注册的用户名代表成功。

## GitHub 或 Git@OSC 上创建项目

[GitHub](http://github.com/) 链接<br>
[Git@OSC](http://git.oschina.net/) 链接

## 本地配置

::: tip 提示
尽量保持你的名字和你的邮箱和 GitHub 或 Git@OSC 的注册一致。
:::

``` bash
git config --global user.name "your name"
git config --global user.email "your email"
```

## 配置大小写敏感

::: tip 提示
在 Git 中默认是不区分大小的, 因此当你修改了文件名的大小写后, git并不会认为你有修改哦, 使用下面的配置可开启大小写敏感
:::

``` bash
git config core.ignorecase false --global
```

## 本地多个 ssh-key 的配置

在 `.ssh` 目录下面创建名为 `config` 的配置文件, 配置文件内容如下

``` bash
Host me.github.com # 相当于 DNS 的作用
    HostName github.com # 解析的地址
    PreferredAuthentications publickey 
    IdentityFile ~/.ssh/me_rsa # 这是您生成密钥的文件名,注意是私钥的文件名

Host work.github.com # 相当于 DNS 的作用
    HostName github.com # 解析的地址
    Port 2222 # 如果你改变了 ssh 的端口号
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/work_rsa # 这是您生成密钥的文件名,注意是私钥的文件名
```

 `测试连接`

``` bash
ssh -T git@me.github.com  # => 将会自动解析成 git@github.com
ssh -T git@work.github.com # => 将会自动解析成 git@github.com
```

## 常用命令

### 进入项目目录, 初始化

``` bash
git init
```

### 添加项目至本地仓库

``` bash
git add .
```

### 提交并备注

``` bash
git commit -m "first commit"
git remote add origin git@github.com:[youName]/[appName].git
git push -u origin master
```

### 解决冲突

``` bash
git pull origin master
```

### 强制覆盖本地代码（更新本地源）

``` bash
git fetch --all
```

### 创建一个新的标签并提交

``` bash
git tag -a v1.0.0 -m "test@1.0.0"
git push origin --tags
```

### 查看本地标签

``` bash
git tag
```

### 删除远程 tag

``` bash
git push origin --delete tag [tagname]
git tag -d [tagname] # 删除本地 tag
git push origin :refs/tags/[tagname]
```

### 获取远程 tag

``` bash
git fetch origin tag [tagname]
```

### 修改 remote

``` bash
git remote set-url origin git@[URL]:[userName]/[projectName]
# or
git remote rm origin
git remote add origin git@[URL]:[userName]/[projectName].git
```

### 查看远程分支

``` bash
git branch -a
```

### 查看本地分支

``` bash
git branch
```

### 创建本地分支

``` bash
git branch [branchName]
```

### 创建一个空分支

``` bash
git branch --orphan [branchName] # 创建无历史提交的分支
git rm -rf . # 删除本地代码
```

### 删除远程分支

``` bash
git push origin --delete [branchName]
# or
git push origin :[branchName]
```

### 本地分支推送到远程分支

``` bash
git push origin [branchName]
```

### 切换分支

``` bash
git checkout [branchName]
```

### 合并分支

``` bash
git checkout master
git merge [branchName]
git push origin master
```

如果分支合并时出现了冲突, 那么只能强上了

``` bash
git branch -m master old-master
git branch -m [branchName] master
git push -f origin master
```

### submodule

当项目越来越庞大之后, 不可避免的要拆分成多个子模块, 我们希望各个子模块有独立的版本管理, 并且由专门的人去维护, 这时候我们就要用到 git 的 submodule 功能。

``` bash
git clone <repository> --recursive 递归的方式克隆整个项目
git submodule add <repository> <path> 添加子模块
git submodule init 初始化子模块
git submodule update 更新子模块
git submodule foreach git pull 拉取所有子模块
```

删除子模块

``` bash
# 逆初始化模块, 其中{MOD_NAME}为模块目录, 执行后可发现模块目录被清空
git submodule deinit {MOD_NAME} 
# 删除.gitmodules中记录的模块信息（--cached选项清除.git/modules中的缓存）
git rm --cached {MOD_NAME} 
# 提交更改到代码库, 可观察到'.gitmodules'内容发生变更
git commit -am "Remove a submodule." 
```

修改模块的 URL

1. 修改'.gitmodules'文件中对应模块的”url“属性; 
2. 使用 `git submodule sync` 命令, 将新的URL更新到文件 `.git/config` # git 中配置 autocrlf

## 换行符

### 源文件中是换行符是 LF

如果你的源文件中是换行符是 LF, 而 `autocrlf=true` , 此时 `Git add .` 就会遇到 `fatal: LF would be replaced by CRLF` 的错误。有两个解决办法: 

1. 将你的源文件中的 LF 转为 CRLF 即可【推荐】
2. 将 autocrlf 设置为 false

### 源文件中是换行符是 CRLF

如果你的源文件中是换行符是 CRLF, 而 `autocrlf=input` , 此时 `Git add .` 也会遇到 `fatal: CRLF would be replaced by LF` 的错误。有两个解决办法: 

1. 将你源文件中的 CRLF 转为 LF【推荐】
2. 将 autocrlf 设置为 true 或者 false

## core.autocrlf

Git 可以在你提交时自动地把行结束符 CRLF 转换成 LF, 而在签出代码时把 LF 转换成 CRLF。用 core.autocrlf 来打开此项功能, 如果是在 Windows 系统上, 把它设置成 true, 这样当签出代码时, LF 会被转换成 CRLF: 

``` bash
git config --global core.autocrlf true
```

Linux 或 Mac 系统使用 LF 作为行结束符, 因此你不想 Git 在签出文件时进行自动的转换；当一个以 CRLF 为行结束符的文件不小心被引入时你肯定想进行修正, 把 core.autocrlf 设置成 input 来告诉 Git 在提交时把 CRLF 转换成 LF, 签出时不转换: 

``` bash
git config --global core.autocrlf input
```

这样会在 Windows 系统上的签出文件中保留 CRLF, 会在 Mac 和 Linux 系统上, 包括仓库中保留 LF。

如果你是 Windows 程序员, 且正在开发仅运行在 Windows 上的项目, 可以设置 false 取消此功能, 把回车符记录在库中: 

``` bash
git config --global core.autocrlf false
```

## 建议

在 Mac/Liunx 上设置 `autocrlf = input` , 在 Windows 上设置 autocrlf = true（默认值）。

Windows: （true） 提交时, 将 CRLF 转成 LF 再提交；切出时, 自动将 LF 转为 CRLF; 

MAC/Linux: (input) 提交时, 将 CRLF 转成 LF 再提交；切出时, 保持 LF 即可

这样即可保证仓库中永远都是 LF. 而且在 Windows 工作空间都是 CRLF, 在 Mac/linux 工作空间都是 LF
