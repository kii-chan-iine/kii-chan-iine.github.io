<template>
  <div class="kanbanniang">
    <div
      class="banniang-container"
      v-show="isLoaded"
    >
      <div
        class="messageBox"
        :style="messageStyle"
        v-show="isShowMessageBox"
        v-html="message"
      >
        {{ message || '欢迎来到 ' + $site.title }}
      </div>
      <div
        v-show="!isShowBtns"
        class="operation"
        @mouseenter="isShowMessageBox = true"
        @mouseleave="isShowMessageBox = false"
      >
        <i
          @click="goHome"
          @mouseenter="hoverGoHome"
          @mouseleave="resetMessage"
        >
          <svg
            t="1572660425629"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="6044"
            width="16"
            height="16"
          >
            <path
              d="M577.5584 307.848533l-21.345067-18.699733c-0.062933-0.061867-0.186667-0.123733-0.280533-0.186667l-44.305067-38.689067-53.752533 47.5648L127.009067 587.424l0 177.1392 0 155.0048c0 45.886933 37.454933 83.0976 83.610667 83.0976l183.966933 0L394.586667 735.8688c0-27.512533 22.448-49.8336 50.162133-49.8336l133.7728 0c27.714133 0 50.178133 22.321067 50.178133 49.8336L628.699733 1002.666667l183.966933 0c46.170667 0 83.610667-37.211733 83.610667-83.0976L896.277333 763.9424 896.277333 586.7712 578.5216 308.688 577.5584 307.848533z"
              p-id="6045"
            ></path>
            <path
              d="M990.637867 418.164267l-94.360533-82.600533 0-181.290667c0-36.714667-29.952-66.482133-66.894933-66.482133-36.941867 0-66.893867 29.767467-66.893867 66.482133l0 64.197333L556.213333 37.911467c-25.291733-22.103467-63.165867-22.103467-88.4256 0L33.348267 418.164267c-27.730133 24.247467-30.402133 66.264533-5.9808 93.808 24.437333 27.544533 66.692267 30.219733 94.407467 5.938133L512 176.376533l390.2432 341.533867c12.7072 11.130667 28.4608 16.600533 44.181333 16.600533 18.549333 0 37.0048-7.617067 50.209067-22.538667C1021.054933 484.4128 1018.382933 442.4128 990.637867 418.164267z"
              p-id="6046"
            ></path>
          </svg>
        </i>
        <i
          @click=""
          @mouseenter="hoverMessage"
          @mouseleave="resetMessage"
        >
          <svg
            t="1572660394444"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="5299"
            width="16"
            height="16"
          >
            <path
              d="M0 202.7V631c0 83.3 68.3 150.7 152.6 150.7h228.9l8 190.3 224.9-190.3h257c84.3 0 152.6-67.4 152.6-150.7V202.7C1024 119.4 955.7 52 871.4 52H152.6C68.3 52 0 119.4 0 202.7z m658.6 237.9c0-39.7 32.1-71.4 72.3-71.4 40.2 0 72.3 31.7 72.3 71.4S771 512 730.9 512c-40.2 0-72.3-31.7-72.3-71.4z m-220.9 0c0-39.7 32.1-71.4 72.3-71.4 40.2 0 72.3 31.7 72.3 71.4S550.1 512 510 512c-40.2 0-72.3-31.7-72.3-71.4z m-216.8 0c0-39.7 32.1-71.4 72.3-71.4 40.2 0 72.3 31.7 72.3 71.4S333.3 512 293.1 512c-40.1 0-72.2-31.7-72.2-71.4z"
              p-id="5300"
            ></path>
          </svg>
        </i>
        <i
          @click="save"
          @mouseenter="hoverPhoto"
          @mouseleave="resetMessage"
        >
          <svg
            t="1572660570409"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="2153"
            width="16"
            height="16"
          >
            <path
              d="M512 393.846154c-86.646154 0-157.538462 70.892308-157.538462 157.538461s70.892308 157.538462 157.538462 157.538462 157.538462-70.892308 157.538462-157.538462-70.892308-157.538462-157.538462-157.538461z m393.846154-118.153846h-102.4c-27.569231 0-51.2-13.784615-66.953846-35.446154l-45.292308-68.923077C677.415385 137.846154 643.938462 118.153846 608.492308 118.153846h-192.984616c-35.446154 0-68.923077 19.692308-84.676923 53.169231l-45.292307 68.923077c-13.784615 21.661538-39.384615 35.446154-66.953847 35.446154H118.153846c-43.323077 0-78.769231 35.446154-78.769231 78.76923v472.615385c0 43.323077 35.446154 78.769231 78.769231 78.769231h787.692308c43.323077 0 78.769231-35.446154 78.769231-78.769231V354.461538c0-43.323077-35.446154-78.769231-78.769231-78.76923zM512 787.692308c-129.969231 0-236.307692-106.338462-236.307692-236.307693s106.338462-236.307692 236.307692-236.307692 236.307692 106.338462 236.307692 236.307692-106.338462 236.307692-236.307692 236.307693z"
              p-id="2154"
            ></path>
          </svg>
        </i>
        <i
          v-if="myTheme.length > 1"
          @click="changeTheme"
          @mouseenter="hoverChangeTheme"
          @mouseleave="resetMessage"
        >
          <svg
            t="1572660469241"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="6553"
            width="16"
            height="16"
          >
            <path
              d="M706.544835 64.021106h-6.500041a159.889547 159.889547 0 0 0-83.558068 23.557532c-54.583153 33.422204-86.949304 40.439014-104.486726 40.439014-17.538445 0-49.903573-7.016811-104.494912-40.445154a159.88136 159.88136 0 0 0-83.550905-23.551392h-6.507204a127.823224 127.823224 0 0 0-97.182366 44.702108l-172.836417 201.635323c-19.522636 22.774703-20.600177 56.050574-2.609431 80.047104l95.995331 127.994116a63.99757 63.99757 0 0 0 83.198887 17.024745v328.558038c0 52.93256 43.060725 95.995331 95.995331 95.995331h415.98011c52.934606 0 95.995331-43.062771 95.995331-95.995331V535.424502a64.028269 64.028269 0 0 0 42.240033 7.749498 64.013943 64.013943 0 0 0 46.990221-34.528398l63.996546-127.856993c11.522428-23.027459 8.125051-50.721195-8.632611-70.278623L803.743574 108.74675c-24.335245-28.421306-59.770292-44.725644-97.198739-44.725644z"
              p-id="6554"
            ></path>
          </svg>
        </i>
        <a
          target="_blank"
          href="https://github.com/vxhly"
        >
          <i
            @mouseenter="hoverMoreAbout"
            @mouseleave="resetMessage"
          >
            <svg
              t="1572660325062"
              class="icon"
              viewBox="0 0 1024 1024"
              version="1.1"
              xmlns="http://www.w3.org/2000/svg"
              p-id="3809"
              width="16"
              height="16"
            >
              <path
                d="M512 3.413333c280.849067 0 508.586667 199.273813 508.586667 444.94848 0 140.427947-74.519893 265.53344-190.65856 347.11552V1020.586667l-222.839467-135.168c-30.859947 5.147307-62.552747 8.021333-95.085227 8.021333-280.845653 0-508.586667-199.28064-508.586666-445.078187C3.413333 202.687147 231.150933 3.413333 512 3.413333z m-158.96576 603.921067h317.805227c17.578667 0 31.812267-14.2336 31.812266-31.819093a31.798613 31.798613 0 0 0-31.812266-31.80544h-317.805227c-17.578667 0-31.812267 14.2336-31.812267 31.80544 0.116053 17.585493 14.349653 31.819093 31.812267 31.819093z m-63.511893-190.665387h444.951893c17.578667 0 31.812267-14.2336 31.812267-31.812266a31.802027 31.802027 0 0 0-31.812267-31.81568H289.522347a31.802027 31.802027 0 0 0-31.81568 31.81568c0 17.578667 14.2336 31.812267 31.81568 31.812266z"
                p-id="3810"
              ></path>
            </svg>
          </i>
        </a>
        <i
          @click="closeBanNiang"
          @mouseenter="hoverCloseBanNiang"
          @mouseleave="resetMessage"
        >
          <svg
            t="1572660347392"
            class="icon"
            viewBox="0 0 1024 1024"
            version="1.1"
            xmlns="http://www.w3.org/2000/svg"
            p-id="4543"
            width="16"
            height="16"
          >
            <path
              d="M512 34.133333a486.4 486.4 0 1 0 486.4 486.4A486.4 486.4 0 0 0 512 34.133333z m209.4848 632.8064l-55.6032 55.466667-151.517867-151.125333-151.517866 151.1168-55.6032-55.466667 151.517866-151.108267L307.242667 364.714667l55.6032-55.466667 151.517866 151.125333 151.517867-151.1168 55.6032 55.466667-151.517867 151.099733z m0 0"
              p-id="4544"
            ></path>
          </svg>
        </i>
      </div>
      <canvas
        ref="banniang"
        id="banniang"
        :style="modelStyle"
        :width="style.width"
        :height="style.height"
        class="live2d"
      ></canvas>
    </div>
    <div
      class="showBanNiang"
      v-show="displayBanNiang"
      @click="showBanNiang"
    >
      看板娘
    </div>
  </div>
</template>

<script>
  import live2dJSString from "./assets/js/live2d";
  import tips from "./assets/js/waifu-tips";

  export default {
    name: "KanBanNiang",
    mixins: [tips],
    data() {
      return {
        isLoaded: true,
        displayBanNiang: false,
        isShowMessageBox: false,
        isShowBtns: CLEAN,
        message: null,
        messages: {
          welcome: MESSAGES.welcome,
          home: MESSAGES.home,
          photo: MESSAGES.photo,
          theme: MESSAGES.theme,
          close: MESSAGES.close,
          about: MESSAGES.about
        },
        currentTheme: THEME[0],
        myTheme: THEME,
        themeName: [
          "blackCat",
          "whiteCat",
          "haru1",
          "haru2",
          "haruto",
          "koharu",
          "izumi",
          "shizuku",
          "wanko",
          "miku",
          "z16",
          "kesshouban"
        ],
        // 模型地址
        model: {
          blackCat:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-hijiki/assets/hijiki.model.json",
          whiteCat:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-tororo/assets/tororo.model.json",
          haru1:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-haru/01/assets/haru01.model.json",
          haru2:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-haru/02/assets/haru02.model.json",
          haruto:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-haruto/assets/haruto.model.json",
          koharu:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-koharu/assets/koharu.model.json",
          izumi:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-izumi/assets/izumi.model.json",
          shizuku:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-shizuku/assets/shizuku.model.json",
          wanko:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-wanko/assets/wanko.model.json",
          miku:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-miku/assets/miku.model.json",
          z16:
            "https://cdn.jsdelivr.net/gh/QiShaoXuan/live2DModel@1.0.0/live2d-widget-model-z16/assets/z16.model.json",
          kesshouban: "/kesshouban/model.json"
        },
        // model的高宽
        style: {
          width: WIDTH,
          height: HEIGHT
        },
        // model的样式
        modelStyle: MODEL_STYLE,
        // messageBox的样式
        messageStyle: MESSAGE_STYLE
      };
    },
    mounted() {
      // 初始化live2d模型
      this.initBanNiang();
    },
    methods: {
      randomNum(val) {
        return Math.floor(Math.random() * val);
      },
      hoverGoHome() {
        const { home } = this.messages;
        if (typeof home === "string") {
          this.message = home;
        } else if (home instanceof Array || Array.isArray(home)) {
          this.message = home[this.randomNum(home.length)];
        } else {
          this.resetMessage();
        }
      },
      hoverMessage() {
        const { welcome } = this.messages;
        if (typeof welcome === "string") {
          this.message = welcome;
        } else if (welcome instanceof Array || Array.isArray(welcome)) {
          this.message = welcome[this.randomNum(welcome.length)];
        } else {
          this.resetMessage();
        }
      },
      hoverChangeTheme() {
        const { theme } = this.messages;
        if (typeof theme === "string") {
          this.message = theme;
        } else if (theme instanceof Array || Array.isArray(theme)) {
          this.message = theme[this.randomNum(theme.length)];
        } else {
          this.resetMessage();
        }
      },
      hoverMoreAbout() {
        const { about } = this.messages;
        if (typeof about === "string") {
          this.message = about;
        } else if (about instanceof Array || Array.isArray(about)) {
          this.message = about[this.randomNum(about.length)];
        } else {
          this.resetMessage();
        }
      },
      hoverPhoto() {
        const { photo } = this.messages;
        if (typeof photo === "string") {
          this.message = photo;
        } else if (photo instanceof Array || Array.isArray(photo)) {
          this.message = photo[this.randomNum(photo.length)];
        } else {
          this.resetMessage();
        }
      },
      hoverCloseBanNiang() {
        const { close } = this.messages;
        if (typeof close === "string") {
          this.message = close;
        } else if (close instanceof Array || Array.isArray(close)) {
          this.message = close[this.randomNum(close.length)];
        } else {
          this.resetMessage();
        }
      },
      resetMessage() {
        this.message = "";
      },
      save() {
        if (window) {
          window.Live2D.captureFrame = true;
        }
      },
      goHome() {
        if (this.$route.path !== "/") {
          this.$router.push("/");
        }
      },
      changeTheme() {
        const themes = [];
        for (var i = 0; i < this.myTheme.length; i++) {
          if (this.myTheme[i] != this.currentTheme) {
            themes.push(this.myTheme[i]);
          }
        }
        const randomNum = Math.floor(Math.random() * (this.myTheme.length - 1));
        this.currentTheme = themes[randomNum];
        this.initBanNiang();
      },
      closeBanNiang() {
        this.isLoaded = false;
        this.displayBanNiang = true;
      },
      showBanNiang() {
        this.isLoaded = true;
        this.displayBanNiang = false;
        this.initBanNiang();
      },
      initBanNiang() {
        if (this.themeName.indexOf(this.currentTheme) === -1) {
          console.log(
            "@vuepress-reco/vuepress-plugin-kan-ban-niang不支持主题" +
              this.currentTheme +
              ", 请检查主题名, 或前往https://vuepress-theme-reco.recoluan.com/views/plugins/kanbanniang.html 查看支持的主题"
          );
          document.querySelector(".kanbanniang").style.display = "none";
          return;
        }
        const isMobile = !!/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
          navigator.userAgent
        );
        if (isMobile) {
          this.isLoaded = false;
          return console.log("mobile do not load model");
        }
        if (!window.loadlive2d) {
          const script = document.createElement("script");
          script.innerHTML = live2dJSString;
          document.body.appendChild(script);
        }
        // this.style = {
        // width: (150 / 1424) * document.body.clientWidth,
        // height: ((150 / 1424) * document.body.clientWidth) / 0.8
        // }
        var ajax = new XMLHttpRequest();
        ajax.open("get", this.model[this.currentTheme]);
        ajax.send();
        ajax.onreadystatechange = function() {
          if (ajax.status !== 200) {
            console.log("看板娘的CDN资源加载失败了, 请稍后刷新页面重试！");
            document.querySelector(".kanbanniang").style.display = "none";
          }
        };
        window.loadlive2d("banniang", this.model[this.currentTheme]);
      }
    }
  };
</script>

<style lang="stylus" scoped>
  // @require './assets/iconfont/iconfont.css';
  .showBanNiang {
    position: fixed;
    right: 70px;
    bottom: 6rem;
    background-color: rgba(231, 234, 241, 0.5);
    color: $accentColor;
    width: 48px;
    height: 20px;
    padding: 10px;
    cursor: pointer;
    // box-shadow 0 9px 17px 0 rgba(0,0,0,0.2)
    border-radius: 4px;
  }

  .banniang-container {
    position: fixed;
    right: 50px;
    bottom: 100px;
    color: #00adb5;

    .messageBox {
      padding: 10px;
      height: 60px;
      width: 160px;
      border-radius: 8px;
      background-color: lighten($accentColor, 50%);
      color: $textColor;
      opacity: 0.5;
    }

    .operation {
      width: 20px;
      height: max-content;
      position: fixed;
      right: 70px;
      bottom: 65px;

      i {
        font-size: 20px;
        cursor: pointer;
        color: lighten($textColor, 50%);

        svg {
          width: 20px;
          height: 20px;
          fill: lighten($textColor, 50%);
        }

        display: inline-block;

        +i, +a {
          margin-top: 8px;
        }

        &:hover {
          color: lighten($accentColor, 50%);
        }
      }
    }

    a {
      display: inline-block;

      +i {
        margin-top: 8px;
      }
    }

    #banniang {
      z-index: 99999;
      pointer-events: none;
    }
  }
</style>
