---
author: kii
title: Gitpages搭建总结
categories: [闲言碎语]
tags: [Git]
date: 2021-06-03 19:11:30
---

<Boxx changeTime="10000"/>

::: tip 前言
这里主要python安装的一些知识。
:::

<!-- more -->

pip更换为清华源：

```
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

update comment 注释更改：

```bash
@echo off
@set input1=
@set /p input1=ud_info:

git status
git add -A
git commit -a -m "%input1%"
git push

pause
```
