---
author: 星火燎原@vxhly
title: Fullcalendar 全日历 API 整理
categories: [web]
tags: [Vue.js, jQuery]
date: 2020-09-30 09:39:04
---

<Boxx changeTime="10000"/>

::: tip 前言
最近在做项目的时候发现了一个比较强大的日历插件 Fullcalendar, 由于官方只提供了英文版的 API 说明, 阅读起来确实有点不方便, 在社区找到的中文版 API 文档是 v4 版本的, 然鹅现在 Fullcalendar 已经更新到了 v5 版本了, 也有很多的 API 在 v5 版本里面已经不适用了, 所以现将常用的 API 进行整理（针对 v5 版本的）。
:::
<!-- more -->

## 常规-上文档

* [Fullcalendar 官方文档](https://fullcalendar.io/)
* [FullCalendar V4 —— （1）安装插件](https://www.jianshu.com/p/183d2fcec3fb)
* [FullCalendar V4 —— （2）日历属性](https://www.jianshu.com/p/9c9b9fb83337)
* [FullCalendar v4 日历插件中文说明文档](https://cloud.tencent.com/developer/article/1019452)

## 先附上在 Vue 中使用 Fullcalendar

``` bash
yarn add @fullcalendar/vue @fullcalendar/daygrid @fullcalendar/timegrid
```

``` vue
<template>
  <FullCalendar :options="calendarOptions" />
</template>

<script>
import FullCalendar from '@fullcalendar/vue'
import dayGridPlugin from '@fullcalendar/daygrid'
import timeGridPlugin from '@fullcalendar/timegrid'

export default {
  components: {
    FullCalendar // make the <FullCalendar> tag available
  },
  data() {
    return {
      calendarOptions: {
        plugins: [dayGridPlugin, timeGridPlugin], // 插件
        initialView: 'dayGridMonth'
      }
    }
  }
}
</script>
```

## 普通的显示设置

* [**`headerToolbar`**](https://fullcalendar.io/docs/headerToolbar)  ===>  设置日历头部信息。如果设置为 false, 则不显示头部信息。包括 left, center, right 左中右三个位置, 每个位置都可以对应以下不同的配置：title: 显示当前月份/周/日信息 prev: 用于切换到上一月/周/日视图的按钮 next: 用于切换到下一月/周/日视图的按钮 prevYear：用于切换到上一年视图的按钮 nextYear：用于切换到下一年视图的按钮
  + 默认值：
  ``` json
  {
    "start": "title", // will normally be on the left. if RTL, will be on the right
    "center": "",
    "end": "today prev,next" // will normally be on the right. if RTL, will be on the left
  }
  ```

* [**`height`**](https://fullcalendar.io/docs/height)  ===>  设置日历的高度, 包括header日历头部, 默认未设置, 高度根据aspectRatio值自适应。
* [**`contentHeight`**](https://fullcalendar.io/docs/contentHeight)  ===>  日历查看区域的高度（不包括 header 和 footer ）, 默认不设置, 日历高度由 AspectRatio 计算, 该值可以设置为：整数、函数或 "auto"
* [**`themeSystem`**](https://fullcalendar.io/docs/themeSystem)  ===>  使用给定的主题系统渲染日历。
  + 默认值： `'standard'` 、 `'bootstrap'`
* [**`buttonIcons`**](https://fullcalendar.io/docs/buttonIcons)  ===>  显示在  headerToolbar/footerToolbar 上的图标
  + 默认值：
  ``` json
  {
    "prev": "left-single-arrow",
    "next": "right-single-arrow",
    "prevYear": "left-double-arrow",
    "nextYear": "right-double-arrow"
  }
  ```

* [**`firstDay`**](https://fullcalendar.io/docs/firstDay)  ===>  每周开始的日子, 默认值为 0, 即周日
* [**`navLinks`**](https://fullcalendar.io/docs/navLinks)  ===>  日和周是否可点击, 默认值是 false
* [**`handleWindowResize`**](https://fullcalendar.io/docs/handleWindowResize)  ===>  是否随浏览器窗口大小变化而自动变化, 默认值是 false
* [**`businessHours`**](https://fullcalendar.io/docs/businessHours) ===> 在日历上强调某些时间段。参数值可以是布尔值：如果参数值为 ture, 则将强调默认的工作时间（周一 ~ 周五的上午9点到下午5点）, 如果为 false 则不强调。参数值可以是一个对象, 以便于精细的控制；也可以是多个对象的数组, 默认值是 false
* [**`editable`**](https://fullcalendar.io/docs/editable)  ===>  允许用户编辑事件（通过拖动事件位置, 或调整事件在日历上的高度）
* [**`aspectRatio`**](https://fullcalendar.io/docs/aspectRatio)  ===>  日历的宽高比, 数字越大, 日历越扁, 默认值为 1.35
* [**`events`**](https://fullcalendar.io/docs/events-array)  ===> 日历上的 event 事件

## 切换语言环境和时区

* [**`locale`**](https://fullcalendar.io/docs/locale)  ===>  修改 FullCalendar 组件的语言, 不能完全的翻译（部分的 headerToolbar 中的按钮文案并不会被翻译）
* [**`buttonText`**](https://fullcalendar.io/docs/buttonText)  ===>  headerToolbar/footerToolbar 中的按钮文案替换
  + 默认值：
  ``` json
  {
    "today": "today",
    "month": "month",
    "week": "week",
    "day": "day",
    "list": "list"
  }
  ```
 
* [**`timeZone`**](https://fullcalendar.io/docs/timeZone)  ===>  切换 FullCalendar 组件的时区, 默认值 `local`

## 操作事件

* [**`eventClick`**](https://fullcalendar.io/docs/eventClick)  ===> 当用户单击事件时触发
* [**`navLinkDayClick`**](https://fullcalendar.io/docs/navLinkDayClick)  ===> 重写点击日的回调事件, 这样配置了 navLinks 就不会导航到任何的视图

## Render Hooks

* [**`eventDidMount`**](https://fullcalendar.io/docs/event-render-hooks)  ===> 在将元素添加到DOM之后立即调用。 如果事件数据发生更改, 则不会再次调用它。