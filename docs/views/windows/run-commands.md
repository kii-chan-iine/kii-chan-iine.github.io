---
author: 星火燎原@vxhly
title: 运行命令集合
categories: [windows]
tags: [Security, Command, Pentest]
date: 2016-08-11 20:46:27
---

<Boxx changeTime="10000"/>

::: tip 前言
记录 Windows 常见应用的运行快捷命令, 以下记录的命令默认为 win+R 组合键调出的窗口, 非命令提示符窗口。
:::
<!-- more -->

## 策略、管理

* `gpedit.msc` => 本地组策略
* `lusrmgr.msc` => 本机用户和组
* `rsop.msc` => 组策略结果集 **建议在 win2003 下运行**
* `wmimgmt.msc` => 打开 windows 管理体系结构(WMI)
* `secpol.msc` => 本地安全策略
* `certmgr.msc` => 证书管理实用程序

## 计算机

* `eventvwr` => 事件查看器
* `compmgmt.msc` => 计算机管理
* `winmsd` => 系统信息 **建议在 win2003 下运行**
* `perfmon.msc` => 计算机性能监测程序
* `winver` => 检查 Windows 版本
* `mmc` => 打开 Microsoft 管理控制台
* `wupdmgr` => windows 更新程序 **建议在 win2003 下运行**
* `wscript` => windows 脚本宿主设置

## 服务

* `dcomcnfg` => 打开系统组件服务
* `services.msc` => 本地服务设置
* `ciadv.msc` => 索引服务程序 **建议在 win2003 下运行**

## 系统、硬件

* `logoff` => 注销命令
* `tsshutdn` => 60 秒倒计时关机命令
* `cmd` => 命令提示符窗口
* `regedt32` => 注册表编辑器
* `regedit.exe` => 注册表编辑器
* `msconfig.exe` => 系统配置实用程序
* `ntbackup` => 系统备份和还原 **建议在 win2003 下运行**
* `drwtsn32` => 系统医生 **建议在 win2003 下运行**

## 磁盘、设备管理

* `diskmgmt.msc` => 磁盘管理实用程序
* `dfrg.msc` => 磁盘碎片整理程序 **建议在 win2003 下运行**
* `chkdsk.exe` => Chkdsk 磁盘检查 **建议在 win2003 下运行；或者使用管理员命令提示符窗口运行**
* `cleanmgr` => 垃圾整理
* `devmgmt.msc` => 设备管理器
* `ntmsmgr.msc` => 可移动存储管理器 **建议在 win2003 下运行**
* `ntmsoprq.msc` => 可移动存储管理员操作请求 **建议在 win2003 下运行**

## 应用程序

* `notepad` => 打开记事本
* `charmap` => 启动字符映射表
* `explorer` => 打开资源管理器
* `calc` => 启动计算器
* `taskmgr` => 任务管理器
* `write` => 写字板
* `mplayer2` => 简易 widnows media player
* `mspaint` => 画图板
* `mstsc` => 远程桌面连接
* `eudcedit` => 专用字符编辑程序
* `packager` => 对象包装程序 **建议在 win2003 下运行**
* `clipbrd` => 剪贴板查看器 **建议在 win2003 下运行**
* `osk` => 打开屏幕键盘
* `iexpress` => 木马捆绑工具, 系统自带

## 网络

* `nslookup` => 网络管理的工具向导
* `ncpa.pal` => 网络连接设置向导

