---
author: 星火燎原@vxhly
title: Liunx 学习笔记（三）文件的压缩与解压缩
categories: [liunx]
tags: [Liunx, Command]
date: 2016-09-10 14:04:39
---

<Boxx changeTime="10000"/>

::: tip 前言
本篇是 Liunx 的学习笔记系列第三篇, 这一篇将介绍在 Liunx 下压缩解压文件或目录常用命令。压缩文件及目录可以大大增加磁盘的利用率。
:::
<!-- more -->

## .gz 格式

``` bash
gzip [fileName]          # 压缩文件
gzip -d [fileName].gz    # 解压缩文件
gunzip [fileName].gz     # 解压缩文件
```

::: warning 注意
只能压缩文件而不能压缩目录, 且不保留原文件
:::

## .tar.gz 格式

``` bash
tar -czvf [folderName].tar.gz [folderName]    # 压缩目录或文件
tar -xzvf [folderName].tar.gz               # 解压目录或文件
```

### 选项解释

``` bash
-c    # 打包
-x    # 解包
-v    # 显示详细信息
-f    # 指定文件名
-z    # 打包同时压缩 / 解压缩
```

::: warning 注意
可以打包或压缩目录和文件, -f 选项是必须的, 且必须放在选项中的最后一位, 可以保留原文件
:::

## .zip 格式

``` bash
zip -r [folderName].zip [folderName]   # 压缩目录
zip [fileName].zip [fileName]          # 压缩文件
unzip [fileName].zip                 # 压缩目录或文件
```

### 选项解释

``` bash
-r    压缩目录
```

## .gz 格式

``` bash
gzip [fileName]          # 压缩文件
gzip -d [fileName].gz    # 解压缩文件
gunzip [fileName].gz     # 解压缩文件
```

::: warning 注意
注意: 只能压缩文件而不能压缩目录, 且不保留原文件
:::

## .zip 格式

``` bash
zip -r [folderName].zip [folderName]     # 压缩目录
zip [fileName].zip [fileName]            # 压缩文件
unzip -uo [fileName].zip -d [folderName] # 解压缩目录或者文件
```

选项解释

``` bash
-r    # 压缩目录
```

::: warning 注意
可以压缩目录或文件, -r 选项用于压缩目录, 可以保留原文件
:::

## .bz2 格式

``` bash
bzip2 -k [fileName]                         # 压缩文件且保留原文件
tar -cjvf [folderName].tar.bz2 [folderName]   # 压缩目录或文###件
bunzip -k [fileName].bz2                    # 解压缩文件
tar -xjvf [folderName].tar.bz2              # 解压缩文件
```

选项解释

``` bash
-k      # 产生压缩文件后保留原文件
-j      # 格式为 .bz2 的压缩包
```

::: warning 注意
bzip2 命令, 只能压缩文件, 不能压缩目录, 要使原文件保留, 使用 -k 选 项；要压缩目录使用 tar 命令的 -j 选项 用 tar 命令的 -j 选项
:::
