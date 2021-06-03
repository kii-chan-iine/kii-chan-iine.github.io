---
author: 星火燎原@vxhly
title: Vue.js 中的实用工具方法
categories: [vuejs]
tags: [Vue.js, Node.js, Code]
date: 2019-07-04 11:35:57
---

<Boxx changeTime="10000"/>

::: tip 前言
收集日常开发中常用到的一些工具方法, 包含 vue 的公用过滤器、公用指令等 (PS: 懒人养成记)。
:::
<!-- more -->

## 公用自定义过滤器

``` javascript
import Vue from 'vue'
import moment from 'moment'

/**
 * @filter dateFormat 时间格式化
 * @param {String, Date} value 可被 new Date 解析的字符串
 * @param {String} formatStr moment 的 format 字符串
 * 使用方法 {{ 2019-1-1 | dateFormat() }}
 */
Vue.filter('dateFormat', (value, formatStr) => {
    return moment(value).format(formatStr || 'YYYY年MM月DD日 hh:mm:ss')
})

/**
 * @filter digitUppercase 人民币金额转成汉字大写
 * @param {Number} value 金额数字
 * 使用方法 {{ 1111 | digitUppercase }}
 */
Vue.filter('digitUppercase', (value) => {
    if (Number(value)) {
        let fraction = ['角', '分']
        let digit = [
            '零', '壹', '贰', '叁', '肆',
            '伍', '陆', '柒', '捌', '玖'
        ]
        let unit = [
            ['元', '万', '亿'],
            ['', '拾', '佰', '仟']
        ]

        let head = value < 0 ? '欠' : ''
        value = Math.abs(value)
        let s = ''
        for (let i = 0; i < fraction.length; i++) {
            s += (digit[Math.floor(value * 10 * Math.pow(10, i)) % 10] + fraction[i]).replace(/零./, '')
        }
        s = s || '整'
        value = Math.floor(value)
        for (let i = 0; i < unit[0].length && value > 0; i++) {
            let p = ''
            for (let j = 0; j < unit[1].length && value > 0; j++) {
                p = digit[value % 10] + unit[1][j] + p
                value = Math.floor(value / 10)
            }
            s = p.replace(/(零.)*零$/, '').replace(/^$/, '零') + unit[0][i] + s
        }
        return head + s.replace(/(零.)*零元/, '元')
            .replace(/(零.)+/g, '零')
            .replace(/^整$/, '零元整')
    } else {
        return '零元整'
    }
})
```

## 公用自定义指令

``` javascript
import Vue from 'vue'

/**
 * @directive preventReClick 防止按钮在短时间内多次点击造成的多次请求(一般用于提交按钮)
 * @param {Element} el 绑定的元素
 * @param {Number} binding 绑定的时间
 * 使用方式 <el-button v-prevent-replace-click></el-button>
 */
Vue.directive('preventReplaceClick', {
    inserted(el, binding) {
        el.addEventListener('click', () => {
            if (!el.disabled) {
                el.classList.add('is-disabled')
                const i = document.createElement('i')
                i.classList.add('el-icon-loading')
                el.prepend(i)
                el.classList.add('is-loading')
                el.disabled = true

                setTimeout(() => {
                    el.disabled = false
                    el.classList.remove('is-disabled')
                    el.classList.remove('is-loading')
                    el.removeChild(i)
                }, binding.value || 1000)
            }
        })
    }
})
```

## 实用方法

### 节流和防抖

``` javascript
/**
 * 应用场景
 * debounce(抖动)
 * search搜索联想, 用户在不断输入值时, 用防抖来节约请求资源。
 * window触发resize的时候, 不断的调整浏览器窗口大小会不断的触发这个事件, 用防抖来让其只触发一次
 *
 * throttle(节流)
 * 鼠标不断点击触发, mousedown(单位时间内只触发一次)
 * 监听滚动事件, 比如是否滑到底部自动加载更多, 用throttle来判断
 */

// 防抖
export function debounce(fn, delay = 200) {
    let timer
    return function() {
        let th = this
        let args = arguments
        if (timer) {
            clearTimeout(timer)
        }
        timer = setTimeout(function() {
            timer = null
            fn.apply(th, args)
        }, delay)
    }
}
// 节流
export function throttle(fn, interval = 200) {
    let last
    let timer
    return function() {
        let th = this
        let args = arguments
        let now = +new Date()
        if (last && now - last < interval) {
            clearTimeout(timer)
            timer = setTimeout(function() {
                last = now
                fn.apply(th, args)
            }, interval)
        } else {
            last = now
            fn.apply(th, args)
        }
    }
}
```

### 时间格式化处理

``` javascript
// 时间格式化成 startDate 和 endDate
import moment from 'moment'
import _ from 'lodash'

/**
 * @method timerByAdd 计算相对当前时间后N个单位时间的日期(加法)
 * @param num {Number} 相对于几个时间点
 * @param timer {String} 时间单位 'days' 'months' 'years‘ 更多时间单位参考moment官方文档
 * @param formatStr {String} moment 的 format 字符串
 * @return {Object} {startDate,endDate}
 */
export function timerByAdd({
    num,
    timer = 'days'
} = {}, formatStr = 'YYYY-MM-DD') {
    let startDate = moment().format(formatStr)
    let endDate

    num ? endDate = moment().add(num, timer).format(formatStr) : startDate = endDate

    endDate = moment(endDate).subtract(1, 'days').format(formatStr)

    let result = {
        startDate,
        endDate
    }
    return result
}

/**
 * @method timerBySubtract 计算相对当前时间前N个单位时间的日期(减法)
 * @param num {Number} 相对于几个时间点
 * @param timer {String} 时间单位 'days' 'months' 'years‘ 更多时间单位参考moment官方文档
 * @param formatStr {String} moment 的 format 字符串
 * @return {Object} {startDate,endDate}
 */
export function timerBySubtract({
    num,
    timer = 'days'
} = {}, formatStr = 'YYYY-MM-DD') {
    let startDate
    let endDate = moment().format(formatStr)

    num ? startDate = moment().subtract(num, timer).format(formatStr) : startDate = endDate

    let result = {
        startDate,
        endDate
    }
    return result
}

/**
 * @method timerFormat 将对象时间转成数组形式
 * @param {Object} timer {startDate, endDate}
 */
export function timerFormat(timer) {
    if (_.isObject(timer)) {
        return _.values(timer)
    }
}

/**
 * @method timerByDiff 计算两个时间段的相差天数,精确到 N 年 N 天 精确到小数后两位
 * @param {Date} startDate 开始时间
 * @param {Date} endDate 结束时间
 */
export function timerByDiff(startDate, endDate) {
    const start = moment(startDate)
    const end = moment(endDate)
    const diffDays = end.diff(start, 'days', true).toFixed(2)
    const diffYears = Math.floor(diffDays / 365)

    let str = ''
    if (diffYears) {
        str = `${diffYears} 年 ${(diffDays - diffYears * 365) + 1} 天`
    } else {
        str = `${Math.floor(diffDays) + 1} 天`
    }
    return str
}
```

### 数值格式化

``` javascript
/**
 * @methods formatNumber 数值按一定的规则进行格式化
 *
 * @param {Number} apart 相隔几个数字
 * @param {Number} separator 添加的字符
 * @param {Number|String} value 纯数字或者数字字符串
 *
 * @return {String} 格式化好的字符串
 */
formatNumber({
    apart = 3,
    separator = ',',
    value
} = {}) {
    if (!Number(value) && value) {
        console.warn('你未正确传递值')
    } else {
        const rgx = new RegExp( `(\\d+)(\\d{${apart}})` )
        const refValue = String(value)
        const x = refValue.split('.')
        let x1 = x[0]
        const x2 = x.length > 1 ? '.' + x[1] : ''

        while (rgx.test(x1)) {
            x1 = x1.replace(rgx, '$1' + separator + '$2')
        }
        return x1 + x2
    }
}
```


``` javascript
// 数字转中文
export const chinanum = (digit) => {
  digit = typeof digit === 'number' ? String(digit) : digit
  const zh = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
  const unit = ['千', '百', '十', '']
  const quot = [
    '万',
    '亿',
    '兆',
    '京',
    '垓',
    '秭',
    '穰',
    '沟',
    '涧',
    '正',
    '载',
    '极',
    '恒河沙',
    '阿僧祗',
    '那由他',
    '不可思议',
    '无量',
    '大数'
  ]
  let breakLen = Math.ceil(digit.length / 4)
  let notBreakSegment = digit.length % 4 || 4
  let segment
  const zeroFlag = []
  const allZeroFlag = []
  let result = ''
  while (breakLen > 0) {
    if (!result) {
      // 第一次执行
      segment = digit.slice(0, notBreakSegment)
      const segmentLen = segment.length
      for (let i = 0; i < segmentLen; i++) {
        if (segment[i] !== 0) {
          if (zeroFlag.length > 0) {
            result += '零' + zh[segment[i]] + unit[4 - segmentLen + i]
            // 判断是否需要加上 quot 单位
            if (i === segmentLen - 1 && breakLen > 1) {
              result += quot[breakLen - 2]
            }
            zeroFlag.length = 0
          } else {
            result += zh[segment[i]] + unit[4 - segmentLen + i]
            if (i === segmentLen - 1 && breakLen > 1) {
              result += quot[breakLen - 2]
            }
          }
        } else {
          // 处理为 0 的情形
          if (segmentLen === 1) {
            result += zh[segment[i]]
            break
          }
          zeroFlag.push(segment[i])
          continue
        }
      }
    } else {
      segment = digit.slice(notBreakSegment, notBreakSegment + 4)
      notBreakSegment += 4
      for (let j = 0; j < segment.length; j++) {
        if (segment[j] !== 0) {
          if (zeroFlag.length > 0) {
            // 第一次执行zeroFlag长度不为0, 说明上一个分区最后有0待处理
            if (j === 0) {
              result += quot[breakLen - 1] + zh[segment[j]] + unit[j]
            } else {
              result += '零' + zh[segment[j]] + unit[j]
            }
            zeroFlag.length = 0
          } else {
            result += zh[segment[j]] + unit[j]
          }
          // 判断是否需要加上 quot 单位
          if (j === segment.length - 1 && breakLen > 1) {
            result += quot[breakLen - 2]
          }
        } else {
          // 第一次执行如果zeroFlag长度不为0, 且上一划分不全为0
          if (j === 0 && zeroFlag.length > 0 && allZeroFlag.length === 0) {
            result += quot[breakLen - 1]
            zeroFlag.length = 0
            zeroFlag.push(segment[j])
          } else if (allZeroFlag.length > 0) {
            // 执行到最后
            if (breakLen === 1) {
              result += ''
            } else {
              zeroFlag.length = 0
            }
          } else {
            zeroFlag.push(segment[j])
          }
          if (
            j === segment.length - 1 &&
            zeroFlag.length === 4 &&
            breakLen !== 1
          ) {
            // 如果执行到末尾
            if (breakLen === 1) {
              allZeroFlag.length = 0
              zeroFlag.length = 0
              result += quot[breakLen - 1]
            } else {
              allZeroFlag.push(segment[j])
            }
          }
          continue
        }
      }

      --breakLen
    }
    return result
  }
}
```

## 常量

### HTTP 状态信息

``` javascript
export const SUCCESS = {
    status: 200,
    message: '成功'
}
export const UNAUTHORIZED = {
    sattus: 401,
    message: '无效或者过期的 token'
}
export const FORBIDDEN = {
    sattus: 403,
    message: '没有访问权限'
}
export const NOT_FOUNED = {
    sattus: 404,
    message: '页面不存在或者已经被移除了'
}
export const UNPROCESSABLE_ENTITY = {
    status: 422,
    message: '无效的表单数据或者未填写完整的表单'
}
export const INTERNAL_SERVER_ERROR = {
    status: 500,
    message: '服务器内部发生不可抗拒的错误'
}
```

## 兼容处理

### 当前使用的浏览器

``` javascript
const userAgent = navigator.userAgent // 取得浏览器的userAgent字符串
const compare = s => {
    return userAgent.indexOf(s) > -1
}
if (compare('Chrome')) {
    if (compare('Edge')) {
        return 'Edge'
    }
    return 'Chrome'
}
if (compare('Opera')) {
    return 'Opera'
}
if (compare('Safari')) {
    return 'Safari'
}
if (compare('Firefox')) {
    return 'Firefox'
}
if (compare('compatible') || compare('MSIE') || compare('Trident')) {
    return 'IE'
}
```

### prepend

[MDN ParentNode.prepend](https://developer.mozilla.org/en-US/docs/Web/API/ParentNode/prepend#Polyfill)

``` javascript
(function(arr) {
    arr.forEach(function(item) {
        if (item.hasOwnProperty('prepend')) {
            return
        }
        Object.defineProperty(item, 'prepend', {
            configurable: true,
            enumerable: true,
            writable: true,
            value: function prepend() {
                var argArr = Array.prototype.slice.call(arguments)
                var docFrag = document.createDocumentFragment()

                argArr.forEach(function(argItem) {
                    var isNode = argItem instanceof Node
                    docFrag.appendChild(isNode ? argItem : document.createTextNode(String(argItem)))
                })

                this.insertBefore(docFrag, this.firstChild)
            }
        })
    })
})([Element.prototype, Document.prototype, DocumentFragment.prototype])
```

## 添加至购物车效果

[参考传送门: 利用animation和transform制作加入购物车动画](https://github.com/bigwave-cl/shopping-cart-animation)

``` javascript
const ballFly = function(opt) {
    return (function() {
        const Parabola = function(opt) {
            this.init(opt)
        }

        Parabola.prototype = {
            init: function(opt) {
                // console.log(opt)
                let flyO = this.calculatedValue(opt)
                let flyDom = this.creatHtml(flyO.site, flyO.callback)
                let flyRule = this.creatRule(flyO.coord)

                document.getElementsByTagName('head')[0].appendChild(flyRule)
                document.body.appendChild(flyDom)
            },
            creatRule: function(coord) {
                var cssAnimation = document.createElement('style')
                cssAnimation.type = 'text/css'
                var rules = `
          .parabola-animation {
            width: 100%;
          }

          .parabola-box-hor {
            position: fixed;
            z-index: 99;
            top: 0;
            left: 0;
            -webkit-animation: parabola-hor-animation 1s ease-out 1;
            animation: parabola-hor-animation 1s ease-out 1;
          }

          .parabola-box-hor.top {
            -webkit-animation-timing-function: ease-in;
            animation-timing-function: ease-in;
          }

          .parabola-box-ver {
            position: fixed;
            top: 50px;
            left: 20px;
            overflow: hidden;
            // width: 35px;
            // height: 35px;
            // background-color: red;
            -webkit-animation: parabola-ver-animation 1s ease-in 1;
            animation: parabola-ver-animation 1s ease-in 1;
            border-radius: 50%;
          }

          .parabola-box-ver.top {
            -webkit-animation-timing-function: ease-out;
            animation-timing-function: ease-out;
          }

          @-webkit-keyframes parabola-hor-animation{
              0%{
                  -webkit-transform: translate(0px, 0px);
                          transform: translate(0px, 0px);
              }
              10%{
                  -webkit-transform: translate(0px, 0px);
                          transform: translate(0px, 0px);
              }
              100%{
                  -webkit-transform: translate(${coord.x}px, 0px);
                          transform: translate(${coord.x}px, 0px);
              }
          }
          @keyframes parabola-hor-animation{
              0%{
                  -webkit-transform: translate(0px, 0px);
                          transform: translate(0px, 0px);
              }
              10%{
                  -webkit-transform: translate(0px, 0px);
                          transform: translate(0px, 0px);
              }
              100%{
                  -webkit-transform: translate(${coord.x}px, 0px);
                          transform: translate(${coord.x}px, 0px);
              }
          }
            @-webkit-keyframes parabola-ver-animation{
                0%{
                    -webkit-transform: translate(0px, 0px);
                            transform: translate(0px, 0px);
                }
                10%{
                    -webkit-transform: translate(0px, ${coord.os}px);
                            transform: translate(0px, ${coord.os}px);
                }
                100%{
                    -webkit-transform: translate(0px, ${coord.y}px);
                            transform: translate(0px, ${coord.y}px);
                }
            }
            @keyframes parabola-ver-animation{
                0%{
                    -webkit-transform: translate(0px, 0px);
                            transform: translate(0px, 0px);
                }
                10%{
                    -webkit-transform: translate(0px, ${coord.os}px);
                            transform: translate(0px, ${coord.os}px);
                }
                100%{
                    -webkit-transform: translate(0px, ${coord.y}px);
                            transform: translate(0px, ${coord.y}px);
                }
            }
        `
                cssAnimation.innerHTML = rules
                return cssAnimation
            },
            creatHtml: function(site, callback) {
                let html = `
          <div class="parabola-box-hor">
            <div class="parabola-box-ver">
            </div>
          </div>
        `
                let parentBox = document.createElement('div')
                parentBox.innerHTML = html
                parentBox.setAttribute('class', 'parabola-animation')

                const event = ['webkitAnimationEnd', 'mozAnimationEnd', 'MSAnimationEnd', 'oanimationend', 'animationend']
                event.forEach(item => {
                    parentBox.addEventListener(item, function() {
                        parentBox.remove()
                        callback()
                    })
                })

                let frag = document.createDocumentFragment()
                frag.appendChild(parentBox)

                let verBox = frag.querySelector('.parabola-box-ver')
                let horBpx = frag.querySelector('.parabola-box-hor')
                verBox.style.left = site.left + 'px'
                verBox.style.top = site.top + 'px'
                verBox.style.width = opt.size + 'px'
                verBox.style.height = opt.size + 'px'
                verBox.style.backgroundColor = opt.color

                if (site.cubic) {
                    verBox.setAttribute('class', 'parabola-box-ver top')
                    horBpx.setAttribute('class', 'parabola-box-hor top')
                }

                return frag
            },
            calculatedValue: function(opt) {
                let fly = {
                    begin: '',
                    end: '',
                    size: 12,
                    color: '#f9666b',
                    callback: function() {
                        console.log('动画完成')
                    }
                }
                let vData = {
                    site: {
                        left: 0,
                        top: 0,
                        cubic: false
                    },
                    coord: {
                        x: 0,
                        y: 0,
                        os: 0
                    },
                    callback: function() {}
                }

                if (typeof opt === 'object') {
                    fly = {
                        ...fly,
                        ...opt
                    }
                }

                if (!fly.begin || !fly.end) return vData

                /**
                 * beginCrood 获取开始元素的位置
                 * endCrood   获取结束元素的位置
                 */
                let beginCrood = fly.begin.getBoundingClientRect()
                let endCrood = fly.end.getBoundingClientRect()

                /*!
                 *  购物车动画出现的位置
                 *  left: 开始元素的left+width/2
                 *  top: 开始元素的top
                 *  购物车动画结束的位置
                 *  x: 结束元素的left+width/2 再减去购物车动画出现的位置的left
                 *  y: 结束元素的top+height/2 再减去购物车动画出现的位置的top
                 */
                /**
                 * 全部减去 18是因为购物车宽度和高度都是35px;一半难得算(-_-),就填18
                 */
                vData.site.left = beginCrood.left + parseInt(beginCrood.width / 2, 10) - 18
                vData.site.top = beginCrood.top - 18
                vData.coord.x = endCrood.left + parseInt(endCrood.width / 2, 10) - vData.site.left - 18
                vData.coord.y = endCrood.top + parseInt(endCrood.height / 2, 10) - vData.site.top - 18
                vData.coord.os = -50
                vData.callback = fly.callback
                if (beginCrood.top > endCrood.top) vData.site.cubic = true

                return vData
            }
        }

        return new Parabola(opt)
    })()
}

export default ballFly
```

使用栗子:

``` vue
<template>
    <div>
        <button class="plus-btn" @click="handlerPlus">+</button>

        <br>
        <br>
        <br>
        <br>
        <br>

        <div class="shopping-cart">购物车</div>
    </div>
</template>

<script>
    import ballFly from '@/utils/ballFly'

    export default {
        methods: {
            handlerPlus(el) {
                ballFly({
                    begin: el.querySelector('.van-stepper__plus'), // 起始位置
                    end: document.querySelector('.picking-num-info'), // 终止位置
                    size: 12, // 小球的直径,默认 12
                    color: '#f9666b' //小球的颜色,默认 #f9666b
                })
            }
        }
    }
</script>
```

## 颜色转换

``` javascript
import {
    isArray,
    isString
} from 'lodash'

/**
 * @method color2Hex RGB 颜色转成 16 进制颜色
 * @param  {String} color RGB 颜色
 * @return {Numbe} 16 进制颜色
 */
export function color2Hex(color) {
    const reg = /^#([0-9a-fA-f]{3}|[0-9a-fA-f]{6})$/
    if (/^(rgb|RGB)/.test(color)) {
        let aColor = color.replace(/(?:\(|\)|rgb|RGB)*/g, '').split(',')
        let strHex = '#'
        for (let i = 0; i < aColor.length; i++) {
            let hex = Number(aColor[i]).toString(16)
            if (hex === '0') {
                hex += hex
            }
            strHex += hex
        }
        if (strHex.length !== 7) {
            strHex = color
        }
        return strHex
    } else if (reg.test(color)) {
        let aNum = color.replace(/#/, '').split('')
        if (aNum.length === 6) {
            return color
        } else if (aNum.length === 3) {
            let numHex = '#'
            for (let i = 0; i < aNum.length; i += 1) {
                numHex += (aNum[i] + aNum[i])
            }
            return numHex
        }
    } else {
        return color
    }
}

/**
 * @method color2Rgb 16 进制颜色转化成 RGB 颜色
 * @param  {Number} color 16 进制颜色
 * @return {String} RGB 颜色
 */
export function color2Rgb(color) {
    const reg = /^#([0-9a-fA-f]{3}|[0-9a-fA-f]{6})$/
    let sColor = color.toLowerCase()
    if (sColor && reg.test(sColor)) {
        if (sColor.length === 4) {
            let sColorNew = '#'
            for (let i = 1; i < 4; i += 1) {
                sColorNew += sColor.slice(i, i + 1).concat(sColor.slice(i, i + 1))
            }
            sColor = sColorNew
        }
        // 处理六位的颜色值
        let sColorChange = []
        for (let i = 1; i < 7; i += 2) {
            sColorChange.push(parseInt('0x' + sColor.slice(i, i + 2)))
        }
        return 'rgb(' + sColorChange.join(',') + ')'
    } else {
        return sColor
    }
}

/**
 * @method setColor 返回与 color 相近的颜色
 * @param {String} color RGB 颜色
 * @param {Number} hex 调整幅度,为 1～hex 的随机数
 * @return {String | Array} RGB 颜色
 */
export function setColor(color, hex = 50) {
    const getHex = (_color) => {
        return _color.match(/\(([^)]*)\)/)[1].split(',').map(item => Number(item))
    }
    const setHex = (_color) => {
        const sColorChange = _color.map(item => {
            if (item === 255) {
                return item - Math.floor(Math.random() * hex)
            } else {
                const _item = item + Math.floor(Math.random() * hex)
                if (_item >= 255) {
                    return 255
                } else {
                    return _item
                }
            }
        })

        return 'rgb(' + sColorChange.join(',') + ')'
    }

    if (isString(color)) {
        return setHex(getHex(color))
    } else if (isArray(color)) {
        const newColor = color.map(item => {
            return setHex(getHex(item))
        })
        return Array.from(new Set(newColor))
    }
}
```

## 导出当前目录的所有 component

``` javascript
import Vue from 'vue'

const files = require.context('.', false, /\.vue$/)
files.keys().forEach(key => {
    Vue.component(files(key).default.name, files(key).default)
})
```

## 导出当前目录的所有 mock

``` javascript
const files = require.context('.', false, /\.js$/)

files.keys().forEach(key => {
    if (key !== './mock.js') {
        // 排除自己 mock.js
        files(key)
    }
})
```

## 导出当前目录的所有 modules

``` javascript
const files = require.context('.', false, /\.js$/)
const modules = {}

files.keys().forEach(key => {
    if (key === './index.js') return
    modules[key.replace(/(\.\/|\.js)/g, '')] = files(key).default
})

export default modules
```

## 关于时间的限制

::: tip 提示
这边提供 element-ui 文档中未提及的日期选择器的自定义禁用时间
:::

### 选择今天以及今天之后的日期

``` javascript
export default {
    data() {
        return {
            pickerOptions: {
                disabledDate(time) {
                    return time.getTime() < Date.now() - 8.64e7
                }
            }
        }
    }
}
```

### 设置选择今天以及今天以前的日期

``` javascript
export default {
    data() {
        return {
            pickerOptions: {
                disabledDate(time) {
                    return time.getTime() > Date.now() - 8.64e6
                }
            }
        }
    }
}
```

### 设置选择今天之后的日期（不能选择当天时间）

``` javascript
export default {
    data() {
        return {
            pickerOptions: {
                disabledDate(time) {
                    return time.getTime() < Date.now()
                }
            }
        }
    }
}
```

### 设置选择今天之前的日期（不能选择当天）

``` javascript
export default {
    data() {
        return {
            pickerOptions: {
                disabledDate(time) {
                    return time.getTime() > Date.now()
                }
            }
        }
    }
}
```

### 设置选择三个月之前到今天的日期

``` javascript
export default {
    data() {
        return {
            pickerOptions: {
                disabledDate(time) {
                    let curDate = (new Date()).getTime()
                    let three = 90 * 24 * 3600 * 1000
                    let threeMonths = curDate - three
                    return time.getTime() > Date.now() || time.getTime() < threeMonths
                }
            }
        }
    }
}
```

## require.context()

1. 场景:如页面需要导入多个组件,原始写法:

``` vue
<script>
import titleCom from '@/components/home/titleCom'
import bannerCom from '@/components/home/bannerCom'
import cellCom from '@/components/home/cellCom'

export default {
    components: {
        titleCom,bannerCom,cellCom
    }
}
</script>
```

2. 这样就写了大量重复的代码,利用 require.context 可以写成

``` vue
<script>
const path = require('path')
const files = require.context('@/components/home', false, /\.vue$/)
const modules = {}
files.keys().forEach(key => {
  const name = path.basename(key, '.vue')
  modules[name] = files(key).default || files(key)
})

export default {
    components: modules
}
</script>
```

这样不管页面引入多少组件,都可以使用这个方法

3. API 方法

``` javascript
// 实际上是 webpack 的方法,vue 工程一般基于 webpack,所以可以使用
require.context(directory,useSubdirectories,regExp)
// 接收三个参数:
// directory：说明需要检索的目录
// useSubdirectories：是否检索子目录
// regExp: 匹配文件的正则表达式,一般是文件名
```

## Object.freeze

::: tip 场景
场景: 一个长列表数据,一般不会更改,但是vue会做getter和setter的转换

用法: 是ES5新增的特性, 可以冻结一个对象, 防止对象被修改

支持: vue 1.0.18+对其提供了支持, 对于data或vuex里使用freeze冻结了的对象, vue不会做getter和setter的转换

注意: 冻结只是冻结里面的单个属性,引用地址还是可以更改
:::

```` javascript
new Vue({
    data: {
        // vue不会对list里的object做getter、setter绑定
        list: Object.freeze([
            { value: 1 },
            { value: 2 }
        ])
    },
    mounted () {
        // 界面不会有响应,因为单个属性被冻结
        this.list[0].value = 100;

        // 下面两种做法, 界面都会响应
        this.list = [
            { value: 100 },
            { value: 200 }
        ];
        this.list = Object.freeze([
            { value: 100 },
            { value: 200 }
        ]);
    }
})
```
