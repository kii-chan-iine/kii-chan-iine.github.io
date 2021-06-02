export default {
  data() {
    return {
      getActed: false,
      hitokotoTimer: null
    }
  },
  mounted() {
    this.initTips()
    if (SHOW_HITOKOTO) {
      this.hitokotoTimer = window.setInterval(() => {
        this.showHitokotoActed()
      }, 30000);
    }
  },
  destroyed() {
    window.clearInterval(this.hitokotoTimer);
  },
  methods: {
    showHitokotoActed() {
      let url = ''
      switch (HITOKOTO_API) {
        case 'imjad.cn':
          url = 'https://api.imjad.cn/hitokoto/?cat=&charset=utf-8&length=28&encode=json'
        case 'hitokoto.cn':
          url = 'https://v1.hitokoto.cn/'
          fetch(url)
            .then((response) => {
              return response.json();
            })
            .then((result) => {
              this.message = result.hitokoto
              this.showMessage(5000)
            });
          break
        case 'jinrishici.com':
          url = 'https://v2.jinrishici.com/one.json'
          fetch(url)
            .then((response) => {
              return response.json();
            })
            .then((result) => {
              this.message = result.data.content
              this.showMessage(5000)
            });
          break
      }
    },
    getRandText(text) {
      return Array.isArray(text) ? text[Math.floor(Math.random() * text.length + 1) - 1] : text
    },
    render(str, context) {
      var tokenReg = /(\\)?\{([^\{\}\\]+)(\\)?\}/g;

      return str.replace(tokenReg, function (word, slash1, token, slash2) {
        if (slash1 || slash2) {
          return word.replace('\\', '');
        }

        var variables = token.replace(/\s/g, '').split('.');
        var currentObject = context;
        var i, length, variable;

        for (i = 0, length = variables.length; i < length; ++i) {
          variable = variables[i];
          currentObject = currentObject[variable];
          if (currentObject === undefined || currentObject === null) return '';
        }
        return currentObject;
      });
    },
    initTips() {
      console.log(`
                    _                            _ _           ____     _ 
      __      _____| | ___ ___  _ __ ___   ___  | (_)_   _____|___ \\ __| |
      \\ \\ /\\ / / _ \\ |/ __/ _ \\| '_ \` _ \\ / _ \\ | | \\ \\ / / _ \\ __) / _\` |
       \\ V  V /  __/ | (_| (_) | | | | | |  __/ | | |\\ V /  __// __/ (_| |
        \\_/\\_/ \\___|_|\\___\\___/|_| |_| |_|\\___| |_|_| \\_/ \\___|_____\\__,_|
      `)

      let now = (new Date()).getHours();
      if (now > 23 || now <= 5) this.message = this.getRandText(WAIFU.hour_tips['t23-5']);
      else if (now > 5 && now <= 7) this.message = this.getRandText(WAIFU.hour_tips['t5-7']);
      else if (now > 7 && now <= 11) this.message = this.getRandText(WAIFU.hour_tips['t7-11']);
      else if (now > 11 && now <= 14) this.message = this.getRandText(WAIFU.hour_tips['t11-14']);
      else if (now > 14 && now <= 17) this.message = this.getRandText(WAIFU.hour_tips['t14-17']);
      else if (now > 17 && now <= 19) this.message = this.getRandText(WAIFU.hour_tips['t17-19']);
      else if (now > 19 && now <= 21) this.message = this.getRandText(WAIFU.hour_tips['t19-21']);
      else if (now > 21 && now <= 23) this.message = this.getRandText(WAIFU.hour_tips['t21-23']);
      else this.message = getRandText(WAIFU.hour_tips.default);
      this.showMessage()

      setTimeout(() => {
        this.handlerMouseover()
        this.handlerClick()
      }, 2000)
    },
    showMessage(timer = 5000) {
      this.isShowMessageBox = true
      setTimeout(() => {
        this.isShowMessageBox = false
      }, timer);
    },
    handlerMouseover() {
      const {
        mouseover
      } = WAIFU
      if (mouseover.length) {
        mouseover.forEach(item => {
          this.$(item.selector).forEach(_item => {
            _item.addEventListener('mouseover', (res) => {
              this.message = this.render(this.getRandText(item.text), {
                text: _item.innerText
              })
              this.showMessage(3000)
            })
          })
        });
      }
    },
    handlerClick() {
      const {
        click
      } = WAIFU
      if (click.length) {
        click.forEach(item => {
          this.$(item.selector).forEach(_item => {
            _item.addEventListener('click', (res) => {
              console.log(1)
              this.message = this.getRandText(item.text)
              this.showMessage(3000)
            })
          })
        });
      }
    },
    $(obj) {
      return document.querySelectorAll(obj)
    }
  }
}