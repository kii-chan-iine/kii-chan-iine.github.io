---
author: 星火燎原@vxhly
title: Nofollow 正确使用方式
categories: [web]
tags: [HTML5, Criterion]
date: 2017-07-05 19:07:00
---

<Boxx changeTime="10000"/>

::: tip 前言
有的网站内容比较的重要, 不希望别人所知道, 同时也防止一些无聊的 SEO 站长特意灌水, 发一些无关的信息, 所以最佳解决方式是设置 Nofollow。
:::
<!-- more -->

## Nofollow 的作用

1. 禁止爬取带有 Nofollow 标识的内容。
2. 不额外传递网站页面或者整个网站权重。
3. 为互联网打造舒适健康的网络环境。

## Nofollow 正确使用方法

### 第一种方法

通常将 nofollow 写在 head 区域, 也就是写在 meta 标签上面, 等于告诉爬虫不要抓取网页上所有标内外部链接。

``` html
<meta name="robots" content="nofollow" />
```

### 第二种方法

上面一种有所差别, 此时他是属于超文本链接的的一个属性值, 也就是超链接 a 标签, 并且带上 rel="nofollow"

``` html
<a rel="nofollow" href=""></a>
```

### Nofollow 的其他一些写法

* **index** 指令: 告诉搜索引擎抓取这个页面
* **follow** 指令: 告诉搜索引擎可以从这个页面上找到链接, 然后继续访问抓取下去。
* **noindex** 指令: 告诉搜索引擎不允许抓取这个页面
* **nofollow** 指令: 告诉搜索引擎不允许从此页找到链接、拒绝其继续访问。

根据以上的指令, 我们就有了以下的四种组合

#### 组合一

可以抓取本页, 而且可以顺着本页继续索引别的链接

``` html
<meta name="robots" content="index,follow">
```

#### 组合二

不许抓取本页, 但是可以顺着本页抓取索引别的链接

``` html
<meta name="robots" content="noindex,follow">
```

#### 组合三

可以抓取本页, 但是不许顺着本页抓取索引别的链接

``` html
<meta name="robots" content="index,nofollow">
```

#### 组合四

不许抓取本页, 也不许顺着本页抓取索引别的链接

``` html
<meta name="robots" content="noindex,nofollow">
```

#### 错误写法

不要把两个对立的反义词写到一起, 例如: 

``` html
<meta name="robots" content="index,noindex">
<!-- 或 -->
<meta name="robots" content="follow,nofollow">
```

#### 简写方式

此外对于多命令, 这里有一个简便的写法, 如果是

``` html
<meta name="robots" content="index,follow">
```

的形式的话, 可以写成: 

``` html
<meta name="robots" content="all">
```

如果是

``` html
<meta name="robots" content="noindex,nofollow">
```

的形式的话, 可以写成: 

``` html
<meta name="robots" content="none">
```

## 禁止搜索引擎建立快照

网站快照指的是搜索引擎（如百度, google 等）在抓取网站数据的时候, 对网页进行的一种缓存处理, 方便用户遇到网站打不开的时候, 也能正常查看网站的资料, 而且网站快照还能告诉站长这个网站在搜索引擎上的更新时间, 当然了, 快照的时间并不等于网站更新的时间。

### 限制所有的搜索引擎建立快照

``` html
<meta name="robots" content="noarchive">
```

### 仅限制某个搜索引擎

``` html
<meta name="Googlebot" content="noarchive"> <!-- 限制谷歌搜索引擎 -->
<meta name="Baiduspider" content="noarchive"> <!-- 限制百度搜索引擎 -->
```

### 其他写法

如果是对于单独的某个搜索引擎不允许建立快照且限制爬虫爬取链接方式, 例如谷歌, 我们就可以写成: 

``` html
<meta name="googlebot" content="index,follow,noarchive">
```

如果设置成 archive 当然是允许建立快照, 默认的也是如此

``` html
<meta name="robots" content="index,follow, archive">
```

## Nofollow 的缺点

1. Nofollow 的作用是告诫爬虫, 该链接请你老人家不要爬取, 也就意味着被添加上 Nofollow 标识的链接将失去传递权重的作用, 从而避免了权重的流失或者被均摊。
2. Nofollow 标识并不能够避免权重的流失。如果你的页面具有 4 个链接, 但是某一个链接添加上了 Nofollow 标识, 这也就导致了原来每个占有 25%的权重链接数量变成了 3 个, 但是所有的链接还是原来的 25%, 并不会因为 Nofollow 标识的存在而实现权重的累积, 也就意味着添加 Nofollow 标识的链接的群众不被考虑在内, 方便蜘蛛能够在有限的时间内爬取收录更多的页面信息。
3. Nofollow 标识是交换友情链接的重要依据之一。众所周知, 高权重友情链接的交换能够给彼此带来较高效果, 因为相互传递权重支持, 但是就有一些 SEO 站长耍点小聪明, 为了一点点的私信, 在链接上面添加上 Nofollow 标识, 从而使彼此互助的友情链接成了单方面的支持, 他的网站不给你传递权重, 而你的网站却给他传递权重, 这属于欺骗性行为, 是一种很可耻的行为, 还在现在各方面都很人性化, 一般都能够检测到 Nofollow 标识。

## Nofollow 和 External nofollow 的区别

其实这两者没有什么区别, 后者是前者的规范性写法, 意思一样, 效果一样。

## Nofollow 使用之处

1. 一般出现在博客内容区域, 评论, 论坛签名等, 防止不法粉丝狂刷屏, 一是方便用户流畅的查看到价值高的信息, 同时也给服务器减压。
2. 广告区域, 也就是广告位, 说白了就是赚钱广告费, 因为这些广告可能是与本网站无关的信心, 如果被爬虫所知道, 那么将会影响网站的相关性, 所以添加 Nofollow 标识, 这样百度搜索引擎就发觉不到了, 聪明吧。
