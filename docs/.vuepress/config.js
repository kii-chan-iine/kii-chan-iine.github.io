module.exports = {
  'title': 'KII IINE',
  'description': '明早一起去看海 望向未来',
  'dest': '../develop/published-pages',
  'base': '/',
  'head': [
    ['link', {
      'rel': 'shortcut icon',
      'href': '/favicon.ico'
    }],
    ['link', {
      'rel': 'manifest',
      'href': '/manifest.json'
    }],
    ['meta', {
      'name': 'theme-color',
      'content': '#22979b'
    }],
    ['meta', {
      'name': 'apple-mobile-web-app-capable',
      'content': 'yes'
    }],
    ['meta', {
      'name': 'apple-mobile-web-app-status-bar-style',
      'content': '#22979b'
    }],
    ['link', {
      'rel': 'apple-touch-icon',
      'href': '/icons/apple-touch-icon-152x152.png'
    }],
    ['link', {
      'rel': 'mask-icon',
      'href': '/icons/safari-pinned-tab.svg',
      'color': '#2c2c2c'
    }],
    ['meta', {
      'name': 'msapplication-TileImage',
      'content': '/icons/msapplication-icon-144x144.png'
    }],
    ['meta', {
      'name': 'msapplication-TileColor',
      'content': '#000000'
    }],
    ['meta', {
      'name': 'viewport',
      'content': 'width=device-width,initial-scale=1,user-scalable=no'
    }],
    ['meta', {
      'name': 'google-site-verification',
      'content': 'XCppppl60fPQTlwxDodwZIhMarkybEgwVpcEz85KTuQ'
    }]
  ],
  'theme': 'reco',
  'themeConfig': {
    'themePicker': true,
    'authorAvatar': '/avatar.jpeg',
    'friendLink': [{
        'title': 'vuepress-theme-reco',
        'desc': 'A simple and beautiful vuepress Blog & Doc theme.',
        'logo': 'https://vuepress-theme-reco.recoluan.com/icon_vuepress_reco.png',
        'link': 'https://vuepress-theme-reco.recoluan.com'
      },
      {
        'title': '午后南杂',
        'desc': 'Enjoy when you can, and endure when you must.',
        'email': 'recoluan@qq.com',
        'link': 'https://www.recoluan.com'
      }
    ],
    // 'valineConfig': {
    //   'appId': 'bplAa0Fmk0scSFweIQkW1zlG-gzGzoHsz',
    //   'appKey': 'kOFtniD34zTlLvSdftu62aKu',
    //   'placeholder': '文明发言, 您的支持将是我继续下去的动力...',
    //   'notify': true,
    //   'verify': true,
    //   'visitor': true
    // },
    'nav': [{
        'text': 'Home',
        'link': '/',
        'icon': 'reco-home'
      },
      {
        'text': 'Project',
        'link': '/project/',
        'icon': 'iconfont icon-project'
      },
      {
        'text': 'TimeLine',
        'link': '/timeLine/',
        'icon': 'reco-date'
      },
      {
        'text': 'Contact',
        'icon': 'reco-message',
        'items': [
          
          // {
          //   'text': 'NPM',
          //   'link': 'https://www.npmjs.com/~vxhly',
          //   'icon': 'reco-npm'
          // },
          {
            'text': 'GitHub',
            'link': 'https://github.com/kii-chan-iine/kii-chan-iine.github.io/tree/develop',
            'icon': 'reco-github'
          },
          {
            'text': 'Mail',
            'link': 'mailto:kaichen1993@hotmail.com',
            'icon': 'iconfont icon-Gmail'
          }
          // ,
          // {
          //   'text': 'Liunx Command',
          //   'link': 'https://man.linuxde.net/',
          //   'icon': 'iconfont icon-link'
          // },
          // {
          //   'text': 'CSS 2 SASS/SCSS',
          //   'link': 'https://css2sass.herokuapp.com/',
          //   'icon': 'iconfont icon-link'
          // },
          // {
          //   'text': 'BootCDN',
          //   'link': 'https://www.bootcdn.cn/',
          //   'icon': 'iconfont icon-link'
          // }
        ]
      }
    ],
    'type': 'blog',
    'blogConfig': {
      'category': {
        'location': 2,
        'text': 'Category'
      },
      'tag': {
        'location': 3,
        'text': 'Tag'
      }
    },
    'logo': '/favicon.ico',
    'search': true,
    'searchMaxSuggestions': 10,
    'sidebar': 'auto',
    'lastUpdated': 'Last Updated',
    'author': 'KII IINE',
    'record': '',
    'startYear': '2021'
  },
  'markdown': {
    'lineNumbers': true
  },
  'plugins': [
    ['vuepress-plugin-code-preview'],
    [
      'cursor-effects',
      {
        size: 2, // size of the particle, default: 2
        shape: 'star', // shape of the particle, default: 'star' 'circle'
        zIndex: 999999999 // z-index property of the canvas, default: 999999999
      }
    ],
    ['vuepress-plugin-boxx'],
    [
      '@vuepress/google-analytics',
      {
        'ga': 'UA-135391925-1'
      }
    ],
    [
      '@vuepress/last-updated',
      {
        transformer: (timestamp, lang) => {
          // 不要忘了安装 moment
          const moment = require('moment')
          moment.locale(lang)
          return moment(timestamp).fromNow()
        }
      }
    ],
    ['@vuepress/medium-zoom', {
      'selector': '.page img',
      'options': {
        'margin': 16,
        'background': '#202124de',
        'scrollOffset': 0
      }
    }],
    ['@vuepress/pwa', {
      'serviceWorker': true,
      'updatePopup': {
        'message': '发现新内容可用.',
        'buttonText': '刷新'
      }
    }],
    ['one-click-copy', {
      'copyMessage': '复制成功了, 快去粘贴使用吧 !!!'
    }],
    ['nest', {
      'color': '255,0,255',
      'count': 100,
      'opacity': 0.7
    }],
    ['copyright', {
      'noCopy': true,
      'minLength': 150,
      'disabled': false,
      'noSelect': true
    }],
    [require('./plugin/kanbanniang/index.js'), {
      // ['blackCat', 'whiteCat', 'haru1', 'kesshouban','haru2', 'haruto', 'koharu', 'izumi', 'shizuku', 'wanko', 'miku', 'z16'],
      'theme': ['haru1','miku', 'z16'],
      'messages': {
        'welcome': ['哦尼桑, 哦尼酱, お帰りなさい !', '你好! 我是你的私人助理, 需要我的帮助吗?'],
        'home': ['准备好回家了吗? 要起航了!', '迷路了吗? 需要人家带你回家吗?', '远方漂泊的人, 常回家看看吧!'],
        'photo': ['呀! 要给人家拍照了, 羞死了羞死了!', '要拍照了, 茄子✌️', '保持微笑就好'],
        'close': ['讨厌啦! 真的要离开我吗？', '哼, 干嘛要关了人家(＞人＜;)', '呜呜呜～, 不要离开人家!', '哼! 你会后悔的!!'],
        'about': ['想知道我的一切?', '想知道关于我主人的更多信息吗？', '如果我主人的开源项目帮到你了, 别忘了给我主人来个 star!'],
        'theme': ['好吧，希望你能喜欢我的其他小伙伴。']
      },
      'messageStyle': {
        'position': 'fixed',
        'right': '75px',
        'bottom': '235px',
        'opacity': '0.75',
        'height': 'max-content',
        'width': '200px',
        'fon-szie': '16px'
      },
      'modelStyle': {
        'position': 'fixed',
        'right': '90px',
        'bottom': '-20px',
        'opacity': '1'
      },
      'width': 216,
      'height': 281.6
    }],
    ['sitemap', {
      'hostname': 'https://www.kiischan.ml',
      'exclude': ['/404.html'],
      'dateFormatter': val => {
        return new Date().toISOString()
      }
    }],
    ['reward', {
      'rewardOption': [{
          'text': '微信',
          'url': '/WeChat.png'
        },
        {
          'text': '支付宝',
          'url': '/Alipay.png'
        }
      ]
    }],
    [
      'dynamic-title',
      {
        'showText': '(/≧▽≦/)耶！主人最爱你了！',
        'hideText': '(＞人＜;)呀！主人我走丢了, 看到我了嘛？',
        'recoverTime': 2000
      }
    ]
  ]
}