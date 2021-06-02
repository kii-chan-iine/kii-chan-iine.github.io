const {
  resolve
} = require('path')

module.exports = (options, context) => ({
  define() {
    const {
      clean,
      messages,
      theme,
      modelStyle,
      width,
      height,
      messageStyle,
      waifu,
      hitokotoApi,
      showHitokoto
    } = options
    return {
      CLEAN: clean || false,
      THEME: theme || ['blackCat', 'whiteCat', 'haru1', 'haru2', 'haruto', 'koharu', 'izumi', 'shizuku', 'wanko', 'miku', 'z16'],
      MESSAGES: messages || {
        welcome: '',
        home: '心里的花, 我想要带你回家。',
        about: '想知道我的一切?',
        photo: '要给人家拍照吗？茄子！',
        theme: '好吧, 希望你能喜欢我的其他小伙伴。',
        close: '你知道我喜欢吃什么吗？痴痴地望着你。',
      },
      MESSAGE_STYLE: messageStyle || {
        position: 'fixed',
        right: '80px',
        bottom: '195px'
      },
      MODEL_STYLE: modelStyle || {
        position: 'fixed',
        right: '90px',
        bottom: '-20px',
        opacity: '0.9'
      },
      SHOW_HITOKOTO: showHitokoto || true,
      HITOKOTO_API: hitokotoApi ||'imjad.cn',// 目前支持 imjad.cn、hitokoto.cn、jinrishici.com
      WAIFU: waifu || {
        'hour_tips': {
          't5-7': ['早上好！一日之计在于晨, 美好的一天就要开始了'],
          't7-11': ['上午好！工作顺利嘛, 不要久坐, 多起来走动走动哦！'],
          't11-14': ['中午了, 工作了一个上午, 现在是午餐时间！'],
          't14-17': ['午后很容易犯困呢, 今天的运动目标完成了吗？'],
          't17-19': ['傍晚了！窗外夕阳的景色很美丽呢, 最美不过夕阳红~'],
          't19-21': ['晚上好, 今天过得怎么样？'],
          't21-23': ['已经这么晚了呀, 早点休息吧, 晚安~'],
          't23-5': ['你是夜猫子呀？这么晚还不睡觉, 明天起的来嘛'],
          'default': ['嗨~ 快来逗我玩吧！']
        },
        'mouseover': [
          {
            'selector': 'abstract-item',
            'text': ['想阅读一下这篇文章吗？']
          },
          {
            'selector': '.page a[href^=\'http\']',
            'text': ['要看看 <span style=\'color:#22979b;\'>{text}</span> 么？']
          },
          {
            'selector': '.global-ui .reward',
            'text': ['要给我主人打赏吗？']
          },
          {
            'selector': '.global-ui .back-to-ceiling',
            'text': ['回到开始的地方吧']
          },
          {
            'selector': 'pre .code-copy',
            'text': ['点击这里可以一键复制代码哟']
          }
        ],
        'click': [
          {
            'selector': 'p code',
            'text': ['这里可以被复制哦']
          }
        ],
      },
      WIDTH: width || 150,
      HEIGHT: height || 220
    }
  },
  name: '@vuepress-reco/vuepress-plugin-kan-ban-niang',
  enhanceAppFiles: resolve(__dirname, './bin/enhanceAppFile.js'),
  globalUIComponents: 'KanBanNiang'
})