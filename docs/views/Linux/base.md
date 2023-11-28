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

## 用户相关

```bash
useradd -c kii -u 2023 -G sudo,docker -d /home/kii -m -N -s /bin/bash kii
```

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



# jupyter

## CENTOS远程服务器安装JUPYTER：OSERROR: [ERRNO 99] CANNOT ASSIGN REQUESTED ADDRESS

标签： [环境报错及解决记录](https://www.freesion.com/tag/%E7%8E%AF%E5%A2%83%E6%8A%A5%E9%94%99%E5%8F%8A%E8%A7%A3%E5%86%B3%E8%AE%B0%E5%BD%95/ "环境报错及解决记录")  [tensorflow](https://www.freesion.com/tag/tensorflow/ "tensorflow")  [jupyter](https://www.freesion.com/tag/jupyter/ "jupyter")

Anaconda和Tensorflow的安装跳过。

问题：在tensorflow2.0-cpu版本下安装Jupyter报错。

![](https://www.freesion.com/images/187/7e72875b0afb20c6ca58384a25895f33.png)

# 解决方案：

## 

第一种（复杂的）：配置文件

### 1.打开PYTHON终端，生成**

```python
from IPython.lib import passwdpasswd()Enter password: Verify password:'sha1:xxxxxxxxx' #记住这段秘钥
```

### 2.修改/ETC/HOSTS

1.首先获取本机内网ip和本机hostname

```python
ifconfig    # 获取本机内网ipvi /etc/hostname    # 获取hostname
```

2.进入/etc/hosts，添加上一行内容

```python
内网ip hostname # 上面获取的那两个
```

### 3.生成默认配置文件---JUPYTER_NOTEBOOK_CONFIG.PY

```
jupyter notebook --generate-config
```

### 4.修改JUPYTER_NOTEBOOK_CONFIG.PY配置文件

```
vim ~/.jupyter/jupyter_notebook_config.py
```

说明：里面全是注释的配置说明，繁琐又复杂，就不看了，里面很多都用不上，这里我需要重写一些配置即可，在文件开头写入：

```
c.NotebookApp.ip = "内网ip hostname "c.NotebookAPp.open_browser = False——这项默认是True，远程登陆时要修改为 Falsec.NotebookApp.password =u 'sha1:xxxx'——前面生成的**c.NotebookApp.port= 8888——可以自己另指定一个端口
```

### 5.运行JUPYTER NOTEBOOK，浏览器访问

![](https://www.freesion.com/images/258/f2012d7edf0edbdd5a158e959bba29da.png)

## 第二种（超简单）：命令式

加入参数启动jupyter

```
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```
