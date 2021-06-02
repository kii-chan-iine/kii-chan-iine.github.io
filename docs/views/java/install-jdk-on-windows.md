---
author: 星火燎原@vxhly
title: JDK 的安装和环境变量的配置
categories: [java]
tags: [Java] 
date: 2016-09-30 19:36:02
---

<Boxx changeTime="10000"/>

::: tip 前言
JDK 是 Java 语言的软件开发工具包, 主要用于移动设备、嵌入式设备上的 java 应用程序。JDK 是整个 java 开发的核心, 它包含了 JAVA 的运行环境, JAVA 工具和 JAVA 基础的类库。这里将介绍在 Windows 下 JDK 的安装和环境变量的配置。
:::
<!-- more -->

## 安装 JDK

安装 JDK 时。第一次是安装 jdk , 第二次是安装 jre 。建议两个都安装在同一个 java 文件夹中的不同文件夹中。（不能都安装在 java 文件夹的根目录下, jdk 和 jre 安装在同一文件夹会出错）。

`For Example:` <br>

![JDK 安装](https://vxhly.github.io/assets/jdk-1.png)

安装 `jdk` 随意选择目录 只需把默认安装目录 `\java` 之前的目录修改即可<br>

![JDK 安装](https://vxhly.github.io/assets/jdk-2.png)

安装 `jre` → `更改` → `\java` 之前目录和安装 `jdk` 目录相同即可<br>

![JDK 安装](https://vxhly.github.io/assets/jdk-3.png)

::: warning 注意
若无安装目录要求, 可全默认设置。无需做任何修改, 两次均直接点下一步。
:::

## 环境变量配置

安装完 JDK 后配置环境变量, 依次选择: `计算机` → `属性` → `高级系统设置` → `高级` → `环境变量` <br>

![JDK 环境变量配置](https://vxhly.github.io/assets/jdk-4.png)

### 指定 JAVA 程序目录

新建 `JAVA_HOME` 变量 。变量值填写 `jdk` 的安装目录（本人是 `D:\Java\jdk` )<br>

![JDK 环境变量配置](https://vxhly.github.io/assets/jdk-5.png)

### 修改 Path 变量

在变量值最后输入 `%JAVA_HOME%\bin;%JAVA_HOME%\jre\bin;` （注意原来 `Path` 的变量值末尾有没有 `;` 号, 如果没有, 先输入 `;` 号再输入上面的代码）<br>

![JDK 环境变量配置](https://vxhly.github.io/assets/jdk-6.png)

### 新建 CLASSPATH 变量

在变量值中输入 `.;%JAVA_HOME%\lib;%JAVA_HOME%\lib\tools.jar` （注意最前面有一点）<br>

![JDK 环境变量配置](https://vxhly.github.io/assets/jdk-7.png)

到此系统的环境变量已经全部配置完毕。

## 检验安装完成

运行 cmd 输入 `java -version` （java 和 -version 之间有空格）。 若如图所示, 显示版本信息, 则说明安装和配置成功。<br>

![JDK 环境变量配置](https://vxhly.github.io/assets/jdk-8.png)

## 附加 liunx 下环境变量配置

安装包只需在官网下载源码安装包即可, 下载之后解压在 `~/java/jdk` 下即可

在 `/etc/profile` 和 `/etc/bash.bashrc` 下粘贴以下代码

``` bash
export JAVA_HOME=~/java/jdk
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
export JRE_HOME=$JAVA_HOME
```

之后分别运行 `source /etc/profile` 、 `source /etc/bash.bashrc` 来让配置生效

生效之后可以运行 `java -version` 来验证一下
