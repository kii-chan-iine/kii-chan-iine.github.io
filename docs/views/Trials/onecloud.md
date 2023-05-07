---
author: kii
title: 玩客云下载姬
categories: [funny]
tags: [trials]
date: 2022-08-09 17:00:00
---

<Boxx changeTime="10000"/>

::: tip 前言
玩客云下载姬。
:::

<!-- more -->

今天继续给分享第三种玩法，让它回归本质，成为一台实实在在的下载机。

**PS：本篇文章所有教程以及固件来自B站UP主@Powersee！在此表示感谢！！！**

本文教程作者@Powersee的视频地址：

🔺建议大家先去看看视频，一般来说视频就能搞定！然后在本期文章中，我将针对一些萌新和小白朋友，做一些更细化的讲解和说明！当然，你其实只看我这篇文章的图文教程，也是能直接搞定刷机过程的。

## 固件简介

我先给大家看看本固件的主界面吧！

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b184cd155913.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_2/)

🔺可以看到这个固件界面清新简洁。已经配置好了我们很多人可以直接实用的常用固件。比喻说下载需要的三大下载神器Aria2，qBittorrent，transmission全部都有，还可以搭建博客，没事写写文章什么的，还有同步神器“微力同步”可以使用，并且可以在后面的USB上连接U盘或者[移动硬盘](https://www.smzdm.com/fenlei/yidongyingpan/)，和最初的玩客云一样，做一个轻NAS，共享里面的视频或者图片给我们的手机，电视播放。

简单来说，它就是作者**@Powersee**打包的 armbian 系统，集成了常用的软件。我个人觉得这个固件非常不错！至少比原有的玩客云功能性可可玩性都要高很多。

本固件所需的所有软件我还是将它放在了天翼网盘里面：[https://cloud.189.cn/t/iUfEVfamYzum](https://go.smzdm.com/43e8cab09cd9f782/ca_bb_yc_163_86807612_14386_0_1641_0)

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b184cccbf426.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_3/)

🔺本次的刷机固件以及工具就是上面的5个文件。第1个是写盘工具，不过原作者用的是另外一个，因为我个人的原因，我换成了图中的这个，因为我觉得这个是图形化操作，看着更直观。后面的4个都是压缩包，等下使用的时候全部需要解压！

看到这里，你先把我分享的这些固件和工具下载到本地，然后把压缩包都解压出来，软件的准备工作就完成了。

话不多说，直接上教程吧！

## 刷底层包

看过我前两篇的小伙伴应该知道，玩客云刷机的前期准备工作还是需要一些硬件工具的，比如说双公头的USB线，U盘（8G以上）。而对于第一次刷机的玩客云，还要准备拆机工具（电吹风，螺丝刀，以及用于短接的镊子）。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://qnam.smzdm.com/202201/08/61d9b184d4cdd8952.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_4/)

🔺我的这台玩客云因为之前刷过第三方的包（也就是前两期给你们分享的openWRT和[电视盒](https://www.smzdm.com/fenlei/gaoqingbofangqi/)子），所以以后它再次刷机的时候就不用拆机进行短接操作了，直接用卡针什么的顶着图中红色方框位置的复位键，然后接通电源就可以直接刷机了。

**PS：前面说过，如果你的玩客云是首次刷机，需要线拆开玩客云，拿出主板进行短接操作刷入底包。短接操作不管是上文的视频还是我前两篇关于玩客云的文章，都有详细的说明。这里就不在赘述了。**

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b184cabf56207.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_5/)

🔺打开你下载到本地的烧录工具**setup_v2.1.3.exe**，然后点击该软件左上角的**文件**，选择**导入烧录包**。在接下来的窗口选择你下载到本地的**底层包(需先解压)s805_flash_snail.img**文件就可以了。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b184ce37a5634.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_6/)

🔺这个时候就需要把玩客云用双公头USB线和电脑连接起来。请注意：**双公头的一端是连接在靠近HDMI的USB接口上，等下U盘是插在靠近网口的USB接口上的，一定不要弄错了。**另一端[台式电脑](https://www.smzdm.com/fenlei/taishiji/)建议直接连在后面，以防供电不足出现刷机失败。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b184d092c2397.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_7/)

🔺因为我前面说过，我这台玩客云因为之前刷过第三方的包，所以这里只需要用卡针顶着复位键，接通电源，然后就能看到软件提示**连接成功**，软件的其它设置保持默认即可，然后点击右边的**开始**就可以了。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b186ebb313045.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_8/)

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b187369f07648.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_9/)

🔺刷机的过程还是很快的。我这里差不多20多秒就完成了。完成以后显示的是**100%烧录成功**，最后点击右边的**停止**按钮，刷底层包的这个过程就算完成了。

## 刷入镜像

刷入镜像包括两个步骤：

- **写入固件到U盘**

- **把U盘固件写入玩客云闪存（EMMC)**

**写入固件到U盘**

因为我们首先需要做的就是让玩客云通过U盘启动，所以我们先把固件写入到U盘中就可以了。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b187a43bc2760.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_10/)

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b188168486975.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_11/)

🔺打开下载到本地的**balenaEtcher-Portable-1.5.45.exe**软件，点击上图红色方框里的按钮，在接下来的方框选择你下载下来的**第二版固件Armbian_s805_powersee_2021-01-31.img**文件。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b1886f0db2293.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_12/)

🔺这里确定是你即将写入的U盘。

**PS：请注意！此步操作将会格式化你的U盘！如果你U盘有重要资料请务必备份！！！**

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b1888952d9129.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_13/)

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b188d83b87279.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_14/)

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b188eb4a43156.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_15/)

🔺点击最后的**Flash!**按钮，软件就会自动写入固件到U盘中了！等到出现上图中的**Flash Complete**，固件写入U盘就成功了。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b189347d29892.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_16/)

🔺这个时候我们将写好固件的U盘从电脑上拿下来，**插到玩客云靠近网口那边的USB接口上，**连接网线，插上电源。如果顺利的话，过个30秒左右，你就会在你的[路由器](https://www.smzdm.com/fenlei/luyouqi/)里面看到玩客云刷机的玩客云已经上线了。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b189c8e536998.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_17/)

🔺此时我们只需要在浏览器的页面直接输入玩客云在路由器中显示的IP地址，或者输入**onecloud/**，就可以显示本固件的主界面了。此时就可以说明我们写入固件到U盘已经完成了，接下来进入第二步。

**把U盘固件写入玩客云闪存**

我们把U盘固件写入到玩客云的闪存以后，就可以直接摆脱U盘的限制，不需要U盘，直接就可以运行该固件了。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b189d1b8a9178.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_18/)

🔺该固件作者贴心的加入了**网页终端**，这样，我们就不需要还下载SSH工具连接玩客云了，直接就可以通过它内置的这个**网页终端**就能连接和控制了，这真的是对我们小白太友好了。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b189e6bf56146.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_19/)

🔺打开网页终端以后，第一行和第二行分别输入账号和密码。

**账号：root**

**密码：powersee233**

输入密码的时候会没有显示的，你只管输入正确的密码，完成后回车即可。

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b18a15d043912.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_20/)

🔺然后直接输入以下代码并回车，玩客云就会自动把U盘上的固件写入到闪存当中了。

代码：**sh /boot/install/install.sh**

[![回归本质！玩客云的最终归宿还是安安静静的做一台下载机](https://imagerk.oss-cn-beijing.aliyuncs.com/img/61d9b18b0ca532341.png_e1080.jpg)](https://post.smzdm.com/p/az3e8680/pic_21/)

🔺写入过程大概三分钟左右，等到出现上图红色方框的提示，写入固件到闪存的这步也就完成了！

此时，你就可以直接拔掉U盘，重新启动玩客云，就能发现它已经可以直接运行该固件了。

**文件管理器：账号密码都是 admin**

**portainer : 账号 admin 密码 powersee**

**transmission : 账号密码都是 admin**

**qbittorrent : 账号 admin 密码 adminadmin**

好了，以上就是今天给大家分享的内容，我是爱分享的Stark-C，如果今天的内容对你有帮助请记得收藏，顺便点点关注，我会经常给大家分享各类有意思的软件和免费干货！谢谢大家~~

## 格式化新硬盘及U盘

**一、(第一种)配置外部硬盘（Armbian挂载硬盘（以及自动挂载））【格式化格式为ext4】**

1.新硬盘的格式化(如果不是新硬盘，已存有重要文件，直接跳过这步骤)

(1) 首先查看新硬盘基本信息

```
sudo fdisk -l

--会罗列出很多分区地址,自行确认自己的新硬盘识别地址在哪里，我的新硬盘被识别到了”/dev/sda”。
```

(2) 对新硬盘进行分区

```
sudo fdisk /dev/sda
 --[1] 这里的 /dev/sda 是步骤（1）中 查询出来的硬盘识别文件地址。如果你在步骤（1）中要格式化的硬盘存在于其他地址请相应改变。
 --[2] 在提示信息引导下，我选择（n） “add a new partition” 将硬盘划分为一个新分区。
(p) primary ----主分区（看个人选择）
(e) extended----扩展分区（看个人选择）
 --[3] 若整个硬盘只作为一个分区，下面三步默认回车即可；若只拿一部分空间出来当分区详细如下：
   [3-1] 第一步是分区盘号，默认回车自动分配盘号，可自己定义一下盘号例如输入4，则盘号为sda4。
   [3-2] star-是从2048字节开始，开始大小建议默认2048（默认回车即可）
  [3-3] end-输入结束字节，开始字节到结束字节为新建分区盘的大小，输入后回车即可，直接回车则默认输入最大字节。
--[4] 最后再输出（p）确认下自己创建的分区表信息是否正确。确认无误后（w）保存。
--[5] 如果成功，系统会提示“The partition table has been altered” 分区表已更改完毕 。
```

(3) 查看新硬盘识别到了哪里

    重新输入(1) 内容 ，我本地的新硬盘被识别到了 “/dev/sda1” 。

(4) 新硬盘格式化

```
sudo mkfs -t ext4 /dev/sda1

----该句将新硬盘（sda1）格式化为EXT4格式(需要点时间)，至此新硬盘的格式化操作结束。如若出现如下：
Device size reported to be zero.  Invalid partition specified, or
        partition table wasn't reread after running fdisk, due to
        a modified partition being busy and in use.  You may need to reboot
        to re-read your partition table.
----执行格式化后弹出以上提示说明没有格式化成功，需reboot重启后再执行格式化
```

2.挂载新硬盘到文件目录

(1) 新建挂载目录

```
在你想要挂载硬盘的目录下新建文件夹，我将其保存在本地用户目录下新建文件夹中。创建文件夹命令样例为：
sudo mkdir <文件夹路径>
```

(2) 挂载硬盘到目录

```
sudo mount /dev/sda1 <文件夹路径>
至此新硬盘就挂载到了自定义的目录下了。
但是发现文件夹权限不足，因此赋予其和其他普通文件夹相同权限：
sudo chmod 777 <文件夹路径>
权限如有需求相应调整，777是最高权限，然后进入文件夹新硬盘已经可以正常操作使用。
```

(3) 开机自动挂载

```
修改/etc/fstab文件，在末尾添加挂载信息。
首先打开并编辑该文件
sudo nano /etc/fstab
在最后一行添加新硬盘的挂载动作
/dev/sda1 <文件夹路径> ext4 defaults 0 0
最后reboot重启，验证开机是否自动挂载。
```

二、(第二种)配置外部硬盘（Armbian挂载硬盘（以及自动挂载））【磁盘格式不变】

(一)设置硬盘挂载在自定义目录(vic)下

1.查看系统所检测到的磁盘，这里的 sda1检测到的硬盘但是没有被挂载（注意：这里sda1 是’1’ 而不是’L’，有些可能是sda1。）

```
lsblk                            #/查看信息
在根目录新建一个目录用于挂载硬盘，命令如下:
cd /                            #/进入根目录
mkdir /vic                        #新建目录名为‘vic’ 可用'ls'查看
mount /dev/sda1 /vic/            #mount 用于挂载Linux系统外的文件。
```

(二)开机自动挂载：

```
1.下面这条命令可以显示硬盘信息，并记下UUID，为下一步做准备，这里以sda1为例
blkid /dev/sda1        #blkid命令对查询设备上所采用文件系统类型进行查询
2.执行下面命令修改 /etc/fstab 即可。例如我就是在 fstab 最后添加这行：
UUID=EC7259EC2(上面查出来的UUID值) /vic(新建硬盘挂载的目录名) ntfs defaults 0 0
vi /etc/fstab        #修改fstab
3.最后保存并应用， 则成功自定挂载，开机也会自动挂载（注意：这里只对只一个硬盘有效）
```

---
