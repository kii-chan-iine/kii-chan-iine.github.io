---
author: kii
title: Linux使用札记
categories: [Linux]
tags: [Linux]
date: 2022-08-10 22:00:00
---

<Boxx changeTime="10000"/>

::: tip 前言
Linux 使用的一些笔记
:::

## shell 使用

<!-- more -->

在shell中终端中使用快捷键 “Ctr+r”，终端会出现如下提示

```
(reverse-i-search)`': 
```

然后输入历史命令中的关键字，会[自动补全](https://so.csdn.net/so/search?q=%E8%87%AA%E5%8A%A8%E8%A1%A5%E5%85%A8&spm=1001.2101.3001.7020)包含关键字的历史命令，接下来

- 按ESC键，可退出搜索模式，并保留历史命令，可再次编辑命令
- 按ENTRY键，可直接执行该历史命令

```
"auto"为关键字
(reverse-i-search)`auto': sudo apt autoremove
```

# zsh

## 安装

1. 安装zsh

```
sudo apt-get install zsh
```

2. 安装oh-my-zsh

```
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```

```bash
ZSH_THEME="candy"
ZSH_THEME="ys" #建议
```

3. autosuggestions

```bash
git clone https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions
source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh
```

# 离线安装ohmyzsh

[ohmyzsh](https://github.com/kii-chan-iine/ohmyzsh)

1. 去github下载<u><a rel="nofollow noreferrer" class="wrap external" href="https://link.zhihu.com/?target=https%3A//github.com/ohmyzsh/ohmyzsh">OH My ZSH</a></u>
2. 进入OH My ZSH项目目录并应用补丁`git apply offline_install.diff`
3. 现在你就可以运行OH My ZSH中tool/install.sh安装了

```bash
git clone https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions
source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh
```

autosuggestion 可以下载```https://github.com/zsh-users/zsh-autosuggestions```复制到```~/.zsh/zsh-autosuggestions```进行离线安装。

# centos

## 1.3 安装zsh包



`dnf -y install zsh`

安装完成后查看shell列表  
`cat /etc/shells`

返回结果如下：

```
/bin/sh
/bin/bash
/sbin/nologin
/bin/dash
/bin/tcsh
/bin/csh
/bin/zsh
```

## 1.4 切换shell至zsh

`chsh -s /bin/zsh`

返回结果：  
`Changing shell for root.`  
`Shell changed.`

按提示所述，shell已经更改为zsh了，现在查看一下系统当前使用的shell，  
`echo $SHELL`  
返回结果如下：  
`/bin/bash`  
重启过后，使用代码查看当前使用的shell  
`echo $SHELL`  
返回结果：  
`/bin/zsh`

得到如此结果，证明shell已经切换成功了。
