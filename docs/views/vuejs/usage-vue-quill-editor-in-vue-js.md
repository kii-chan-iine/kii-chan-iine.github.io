---
author: 星火燎原@vxhly
title: 在 vue.js 中使用 vue-quill-editor 富文本编辑器
categories: [vuejs]
tags: [Node.js, JavaScript]
date: 2020-08-19 14:38:09
---

<Boxx changeTime="10000"/>

::: tip 前言
富文本编辑器在后台管理系统中是一个常见而又简单的需求, 这里介绍到的是在 vue.js 项目中使用 vue-quill-editor 这一款富文本编辑器。
:::
<!-- more -->

## vue-quill-editor

至于这款编辑器并不想做过多的介绍, 本文是致力于收集网上一些关于这个编辑器的使用方法（毕竟 `quilljs` 官网挂了, 所以很多的 API 使用全部看不到官方的说明）

::: tip
上官方的演示地址 [vue-quill-editor](https://github.surmon.me/vue-quill-editor/)
官方文档(ps: 官方文档好像挂了进不去, 翻墙也进不去)：[Quill API document](https://quilljs.com/docs/quickstart/)
社区中文文档：[Quill 中文文档](https://kang-bing-kui.gitbook.io/quill/)
:::

## 下载安装

``` bash
npm install vue-quill-editor --save
# or
yarn add vue-quill-editor
```

### 全局安装

``` javascript
import Vue from 'vue'
import VueQuillEditor from 'vue-quill-editor'

import 'quill/dist/quill.core.css' // import styles
import 'quill/dist/quill.snow.css' // for snow theme
import 'quill/dist/quill.bubble.css' // for bubble theme

Vue.use(VueQuillEditor, /* { default global options } */ )
```

### 局部安装

``` javascript
import 'quill/dist/quill.core.css' // import styles
import 'quill/dist/quill.snow.css' // for snow theme
import 'quill/dist/quill.bubble.css' // for bubble theme

import {
    quillEditor
} from 'vue-quill-editor'

export default {
    components: {
        quillEditor
    }
}
```

## 使用

### 一个简单的例子

``` vue
<template>
  <quill-editor
    ref="myQuillEditor"
    v-model="content"
    :options="editorOption"
    @change="onEditorChange"
    @blur="onEditorBlur($event)"
    @focus="onEditorFocus($event)"
    @ready="onEditorReady($event)"
  />
</template>

<script>
import 'quill/dist/quill.core.css' // import styles
import 'quill/dist/quill.snow.css' // for snow theme
import 'quill/dist/quill.bubble.css' // for bubble theme

import {
    quillEditor
} from 'vue-quill-editor'

export default {
  components: {
      quillEditor
  },
  data () {
    return {
      content: '',
      editorOption: {
        // Some Quill options...
      }
    }
  },
  methods: {
    onEditorBlur(quill) {
      console.log('editor blur!', quill)
    },
    onEditorFocus(quill) {
      console.log('editor focus!', quill)
    },
    onEditorReady(quill) {
      console.log('editor ready!', quill)
    },
    onEditorChange({ quill, html, text }) {
      console.log('editor change!', quill, html, text)
    }
  }
}
</script>
```

## 二次封装

::: tip 场景一
默认的 vue-quill-editor 是采用 iframe 来嵌入视频的, 我们现需要修改成用 H5 的 video
:::

准备一个 `video.js` 文件

``` javascript
import {
    Quill
} from 'vue-quill-editor'

// 源码中是import直接倒入, 这里要用Quill.import引入
const BlockEmbed = Quill.import('blots/block/embed')
const Link = Quill.import('formats/link')

const ATTRIBUTES = ['height', 'width']

class Video extends BlockEmbed {
    static create(value) {
        const node = super.create(value)
        // 添加video标签所需的属性
        node.setAttribute('controls', 'controls') // 控制播放器
        node.setAttribute('type', 'video/mp4')
        node.setAttribute('style', 'object-fit:fill;width: 100%;')
        node.setAttribute('preload', 'auto') // auto - 当页面加载后载入整个视频  meta - 当页面加载后只载入元数据  none - 当页面加载后不载入视频
        node.setAttribute('webkit-playsinline', 'true') // 兼容ios 不全屏播放
        node.setAttribute('playsinline', 'true')
        node.setAttribute('x-webkit-airplay', 'allow')
        // node.setAttribute('x5-video-player-type', 'h5') // 启用H5播放器,是wechat安卓版特性
        node.setAttribute('x5-video-orientation', 'portraint') // 竖屏播放 声明了h5才能使用  播放器支付的方向, landscape横屏, portraint竖屏, 默认值为竖屏
        node.setAttribute('x5-playsinline', 'true') // 兼容安卓 不全屏播放
        node.setAttribute('x5-video-player-fullscreen', 'true') // 全屏设置, 设置为 true 是防止横屏
        node.setAttribute('src', this.sanitize(value))
        return node
    }

    static formats(domNode) {
        return ATTRIBUTES.reduce((formats, attribute) => {
            if (domNode.hasAttribute(attribute)) {
                formats[attribute] = domNode.getAttribute(attribute)
            }
            return formats
        }, {})
    }

    static sanitize(url) {
        return Link.sanitize(url) // eslint-disable-line import/no-named-as-default-member
    }

    static value(domNode) {
        return domNode.getAttribute('src')
    }

    format(name, value) {
        if (ATTRIBUTES.indexOf(name) > -1) {
            if (value) {
                this.domNode.setAttribute(name, value)
            } else {
                this.domNode.removeAttribute(name)
            }
        } else {
            super.format(name, value)
        }
    }

    html() {
        const {
            video
        } = this.value()
        return `<a href="${video}">${video}</a>`
    }
}
Video.blotName = 'video' // 这里不用改, 楼主不用iframe, 直接替换掉原来, 如果需要也可以保留原来的, 这里用个新的blot
Video.className = 'ql-video'
Video.tagName = 'video' // 用video标签替换iframe

export default Video
```

然后在上一节的例子之中添加以下代码

``` vue
<script>
import { quillEditor, Quill } from 'vue-quill-editor'
import Video from './video.js'

Quill.register(Video, true) // 插入线上视频

export default {
  components: {
    quillEditor
  }
}
</script>
```

::: tip 场景二
默认的 vue-quill-editor 配置的字体样式比较少, 我们需要额外的添加几种字体
:::

上一节的例子之中添加以下代码

``` vue
<script>
import { quillEditor, Quill } from 'vue-quill-editor'

export default {
  data() {
    // 额外配置的字体
    const fonts = ['SimSun', 'SimHei', 'Microsoft-YaHei', 'KaiTi', 'FangSong', 'Arial', 'Times-New-Roman', 'sans-serif']
    const Font = Quill.import('formats/font')
    Font.whitelist = fonts // 将字体加入到白名单
    Quill.register(Font, true)

    return {
      editorOption: {
        modules: {
          toolbar: {
            container: [
               [{ font: fonts }], // 字体
            ]
          }
        }
      }
    }
  }
}
</script>

<style>
  /* 额外的配置字体 */
  .ql-snow .ql-picker.ql-font .ql-picker-label[data-value='SimSun']::before,
  .ql-snow .ql-picker.ql-font .ql-picker-item[data-value='SimSun']::before {
    content: '宋体';
    font-family: 'SimSun';
  }
  .ql-snow .ql-picker.ql-font .ql-picker-label[data-value='SimHei']::before,
  .ql-snow .ql-picker.ql-font .ql-picker-item[data-value='SimHei']::before {
    content: '黑体';
    font-family: 'SimHei';
  }
  .ql-snow .ql-picker.ql-font .ql-picker-label[data-value='Microsoft-YaHei']::before,
  .ql-snow .ql-picker.ql-font .ql-picker-item[data-value='Microsoft-YaHei']::before {
    content: '微软雅黑';
    font-family: 'Microsoft YaHei';
  }
  .ql-snow .ql-picker.ql-font .ql-picker-label[data-value='KaiTi']::before,
  .ql-snow .ql-picker.ql-font .ql-picker-item[data-value='KaiTi']::before {
    content: '楷体';
    font-family: 'KaiTi';
  }
  .ql-snow .ql-picker.ql-font .ql-picker-label[data-value='FangSong']::before,
  .ql-snow .ql-picker.ql-font .ql-picker-item[data-value='FangSong']::before {
    content: '仿宋';
    font-family: 'FangSong';
  }
  .ql-snow .ql-picker.ql-font .ql-picker-label[data-value='Arial']::before,
  .ql-snow .ql-picker.ql-font .ql-picker-item[data-value='Arial']::before {
    content: 'Arial';
    font-family: 'Arial';
  }
  .ql-snow .ql-picker.ql-font .ql-picker-label[data-value='Times-New-Roman']::before,
  .ql-snow .ql-picker.ql-font .ql-picker-item[data-value='Times-New-Roman']::before {
    content: 'Times New Roman';
    font-family: 'Times New Roman';
  }
  .ql-snow .ql-picker.ql-font .ql-picker-label[data-value='sans-serif']::before,
  .ql-snow .ql-picker.ql-font .ql-picker-item[data-value='sans-serif']::before {
    content: 'sans-serif';
    font-family: 'sans-serif';
  }

  .ql-font-SimSun {
    font-family: 'SimSun';
  }
  .ql-font-SimHei {
    font-family: 'SimHei';
  }
  .ql-font-Microsoft-YaHei {
    font-family: 'Microsoft YaHei';
  }
  .ql-font-KaiTi {
    font-family: 'KaiTi';
  }
  .ql-font-FangSong {
    font-family: 'FangSong';
  }
  .ql-font-Arial {
    font-family: 'Arial';
  }
  .ql-font-Times-New-Roman {
    font-family: 'Times New Roman';
  }
  .ql-font-sans-serif {
    font-family: 'sans-serif';
  }
</style>
```

::: tip 场景三
默认的 vue-quill-editor 配置的是英文提示, 我们需要对其进行汉化处理
:::

上一节的例子之中添加以下代码

``` vue
<style>
  /* 汉化 */
  .ql-snow .ql-tooltip[data-mode='link']::before {
    content: '请输入链接地址:' !important;
  }
  .ql-snow .ql-tooltip.ql-editing a.ql-action::after {
    border-right: 0px;
    content: '保存' !important;
    padding-right: 0px;
  }
  .ql-snow .ql-tooltip[data-mode='video']::before {
    content: '请输入视频地址:' !important;
  }
  .ql-snow .ql-picker.ql-size .ql-picker-label::before,
  .ql-snow .ql-picker.ql-size .ql-picker-item::before {
    content: '14px' !important;
  }
  .ql-snow .ql-picker.ql-size .ql-picker-label[data-value='small']::before,
  .ql-snow .ql-picker.ql-size .ql-picker-item[data-value='small']::before {
    content: '10px' !important;
  }
  .ql-snow .ql-picker.ql-size .ql-picker-label[data-value='large']::before,
  .ql-snow .ql-picker.ql-size .ql-picker-item[data-value='large']::before {
    content: '18px' !important;
  }
  .ql-snow .ql-picker.ql-size .ql-picker-label[data-value='huge']::before,
  .ql-snow .ql-picker.ql-size .ql-picker-item[data-value='huge']::before {
    content: '32px' !important;
  }

  .ql-snow .ql-picker.ql-header .ql-picker-label::before,
  .ql-snow .ql-picker.ql-header .ql-picker-item::before {
    content: '正文' !important;
  }
  .ql-snow .ql-picker.ql-header .ql-picker-label[data-value='1']::before,
  .ql-snow .ql-picker.ql-header .ql-picker-item[data-value='1']::before {
    content: '标题1' !important;
  }
  .ql-snow .ql-picker.ql-header .ql-picker-label[data-value='2']::before,
  .ql-snow .ql-picker.ql-header .ql-picker-item[data-value='2']::before {
    content: '标题2' !important;
  }
  .ql-snow .ql-picker.ql-header .ql-picker-label[data-value='3']::before,
  .ql-snow .ql-picker.ql-header .ql-picker-item[data-value='3']::before {
    content: '标题3' !important;
  }
  .ql-snow .ql-picker.ql-header .ql-picker-label[data-value='4']::before,
  .ql-snow .ql-picker.ql-header .ql-picker-item[data-value='4']::before {
    content: '标题4' !important;
  }
  .ql-snow .ql-picker.ql-header .ql-picker-label[data-value='5']::before,
  .ql-snow .ql-picker.ql-header .ql-picker-item[data-value='5']::before {
    content: '标题5' !important;
  }
  .ql-snow .ql-picker.ql-header .ql-picker-label[data-value='6']::before,
  .ql-snow .ql-picker.ql-header .ql-picker-item[data-value='6']::before {
    content: '标题6' !important;
  }
</style>
```

::: tip 场景四
默认的 vue-quill-editor 设置工具栏的是没有 title 的, 我们需要对其进行设置
:::

上一节的例子之中添加以下代码

``` vue
<script>
export default {
   mounted () {
     // 设置工具栏的 title
     this.setToolbar()
   },
   methods: {
    // 设置工具栏的 title
    setToolbar () {
      const titleConfig = {
        'ql-bold': '加粗',
        'ql-color': '颜色',
        'ql-font': '字体',
        'ql-code': '插入代码',
        'ql-italic': '斜体',
        'ql-link': '添加链接',
        'ql-background': '背景颜色',
        'ql-size': '字体大小',
        'ql-strike': '删除线',
        'ql-script': '上标/下标',
        'ql-underline': '下划线',
        'ql-blockquote': '引用',
        'ql-header': '标题',
        'ql-indent': '缩进',
        'ql-list': '列表',
        'ql-align': '文本对齐',
        'ql-direction': '文本方向',
        'ql-code-block': '代码块',
        'ql-formula': '公式',
        'ql-image': '图片',
        'ql-video': '视频',
        'ql-clean': '清除字体样式'
      }
      const oToolBar = document.querySelector('.ql-toolbar')
      const aButton = oToolBar.querySelectorAll('button')
      const aSelect = oToolBar.querySelectorAll('select')
      aButton.forEach(function (item) {
        if (item.className === 'ql-script') {
          item.value === 'sub' ? (item.title = '下标') : (item.title = '上标')
        } else if (item.className === 'ql-indent') {
          item.value === '+1' ? (item.title = '向右缩进') : (item.title = '向左缩进')
        } else {
          item.title = titleConfig[item.classList[0]]
        }
      })
      aSelect.forEach(function (item) {
        item.parentNode.title = titleConfig[item.classList[0]]
      })
    }
   }
}
</script>
```

::: tip 场景五
默认的 vue-quill-editor 设置工具栏的是没有对图片进行限制大小处理的, 我们需要重写上传图片的监听
:::

``` vue
<script>
export default {
  data() {
    return {
      editorOption: {
        modules: {
          toolbar: {
            handlers: {
              // 重写图片上传监听方法, 用于控制上传图片大小和图片类型
              image: value => {
                if (value) {
                  this.doUpload('image')
                } else {
                  this.editor.format('image', false)
                }
              }
            }
          }
        }
      }
    }
  },
  methods: {
    doUpload (type) {
      const input = document.createElement('input')
      input.setAttribute('type', 'file')
      // 新建img标签, 判断是图片还是视频
      if (type === 'image') {
        // 控制图片的类型
        input.setAttribute('accept', 'image/*')
      } else {
        input.setAttribute('accept', 'video/*')
      }
      input.click()
      input.addEventListener('change', () => {
        const files = input.files
        if (type === 'image') {
          // 控制图片的大小
          if (files[0] && files[0].size > this.imageSize * 1024 * 1024) {
            this.$message.warning( `上传的图片大小不能超过${this.imageSize}M` )
          } else {
            // 这边也可以做上传到图片服务器
            this.getBase64(files[0], res => {
              const editor = this.editor
              const index = editor.getSelection().index // 获取当前的光标的位置
              // 插入图片
              editor.insertEmbed(index, 'image', res)
              // 调整光标到最后
              editor.setSelection(index + 1)
            })
          }
        }
      })
    }
  }
}
</script>
```
