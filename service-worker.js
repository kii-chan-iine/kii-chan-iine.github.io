/**
 * Welcome to your Workbox-powered service worker!
 *
 * You'll need to register this file in your web app and you should
 * disable HTTP caching for this file too.
 * See https://goo.gl/nhQhGp
 *
 * The rest of the code is auto-generated. Please don't update this file
 * directly; instead, make changes to your Workbox build configuration
 * and re-run your build process.
 * See https://goo.gl/2aRDsh
 */

importScripts("https://storage.googleapis.com/workbox-cdn/releases/4.3.1/workbox-sw.js");

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

/**
 * The workboxSW.precacheAndRoute() method efficiently caches and responds to
 * requests for URLs in the manifest.
 * See https://goo.gl/S9QRab
 */
self.__precacheManifest = [
  {
    "url": "404.html",
    "revision": "c3265716a6bbd27c00cb057beada45bc"
  },
  {
    "url": "Alipay.png",
    "revision": "f7366096081ffbb417eaf1a33a3cff7e"
  },
  {
    "url": "assets/box-model.gif",
    "revision": "2537725d5fa341801f2da60e27320455"
  },
  {
    "url": "assets/broswer-debug-elements-1.png",
    "revision": "752e909dc9163e254ef30c95ca7edd6c"
  },
  {
    "url": "assets/broswer-debug-elements-2.png",
    "revision": "d0b10f61b2d5222c806fe64533742922"
  },
  {
    "url": "assets/browser-debug-audits-3.png",
    "revision": "4061130903a22f790379f194423e942d"
  },
  {
    "url": "assets/browser-debug-console-1.png",
    "revision": "75e2f6f0f8b81c0e6ecbf86ffa7a783e"
  },
  {
    "url": "assets/browser-debug-console-2.png",
    "revision": "26be3e8acf46088cbc061d1c0b9cf439"
  },
  {
    "url": "assets/browser-debug-moblie-1.png",
    "revision": "422829b33cd7d856fcc0204ff9c9573c"
  },
  {
    "url": "assets/browser-debug-moblie-2.png",
    "revision": "a818523baa9652678fa1b8c93ca3cdb2"
  },
  {
    "url": "assets/browser-debug-network-1.png",
    "revision": "8f5cd3464f3c05f06994631f88ce83a1"
  },
  {
    "url": "assets/browser-debug-network-2.png",
    "revision": "fe3aed470e61fcbacc2883a87eb72305"
  },
  {
    "url": "assets/browser-debug-network-4.png",
    "revision": "48397d3f4e8acd1fa2db01425cf64ab7"
  },
  {
    "url": "assets/browser-debug-network-5.png",
    "revision": "bf82daea596ff79164e61aa11e8b8948"
  },
  {
    "url": "assets/browser-debug-network-6.png",
    "revision": "aac4ed9cefd00725244e9d0b04748df3"
  },
  {
    "url": "assets/browser-debug-sources-1.png",
    "revision": "b99e6ae0f4f1815fe2b4ebd615858351"
  },
  {
    "url": "assets/burp-1.png",
    "revision": "f1f251b01c9ddc34f0ca6f7afcdf0eba"
  },
  {
    "url": "assets/burp-10.png",
    "revision": "5f59cf59ae0fcc8b94ebe54f8f940299"
  },
  {
    "url": "assets/burp-11.png",
    "revision": "a59ae7241454a69057f40a9e3345e881"
  },
  {
    "url": "assets/burp-2.png",
    "revision": "fc20d1675e3642c323b7365e4d5e52ca"
  },
  {
    "url": "assets/burp-3.png",
    "revision": "054c0578eccaccef74bf9e10d90e6445"
  },
  {
    "url": "assets/burp-4.png",
    "revision": "22c1b3e1c09134732342b1565ed25ab9"
  },
  {
    "url": "assets/burp-5.png",
    "revision": "e0e212cad620a20f6eb709e246aabc1a"
  },
  {
    "url": "assets/burp-6.png",
    "revision": "388cf2e2be7207e6a5d74bd79524fafc"
  },
  {
    "url": "assets/burp-7.png",
    "revision": "8aa9d097fcf32f11b72d0ea8085b3eee"
  },
  {
    "url": "assets/burp-8.png",
    "revision": "d94aefcfc464f8dc4b043445e3449779"
  },
  {
    "url": "assets/burp-9.png",
    "revision": "32eb30d884ad9fc7feeb335dc81d1df2"
  },
  {
    "url": "assets/css-length-unit-1.png",
    "revision": "a28516416913ebda2a255762ce7af47f"
  },
  {
    "url": "assets/css-length-unit-2.png",
    "revision": "49e27ea32224065a9d2b53a06f62ff78"
  },
  {
    "url": "assets/css-length-unit-3.png",
    "revision": "2b7f39620daa725f50ebe2fdf0834599"
  },
  {
    "url": "assets/css-new-features-1.png",
    "revision": "1a794214665d8e4651ad768904685331"
  },
  {
    "url": "assets/css-new-features-2.gif",
    "revision": "813761c521035bdda8075141acf07197"
  },
  {
    "url": "assets/css-new-features-3.png",
    "revision": "a9e655cea767d6da06aa5c0a2aea2c1b"
  },
  {
    "url": "assets/css-new-features-4.png",
    "revision": "435d3002887192c07450923fa2504d95"
  },
  {
    "url": "assets/css-write-1.jpg",
    "revision": "5bbe7a97f5d55c84917b0832d8aff718"
  },
  {
    "url": "assets/css-write-2.jpg",
    "revision": "5bbe7a97f5d55c84917b0832d8aff718"
  },
  {
    "url": "assets/css/0.styles.c67cb200.css",
    "revision": "d80113c6a1dac5898638a1adc98ed67f"
  },
  {
    "url": "assets/debug-console-1.png",
    "revision": "1d4b0da6a944db4811a3c5b954b7d39a"
  },
  {
    "url": "assets/debug-console-2.png",
    "revision": "03c80b54bf0adf4e12c080ec1bbb96d0"
  },
  {
    "url": "assets/debug-console-3.png",
    "revision": "97c6a48d857b4a2995c7c39d456b6da7"
  },
  {
    "url": "assets/debug-console-4.png",
    "revision": "4d2008689439a2bde030ae02c8545061"
  },
  {
    "url": "assets/debug-console-5.png",
    "revision": "208ce25932556509b8d9c0bbc1fc0c83"
  },
  {
    "url": "assets/debug-console-6.png",
    "revision": "e65a0faa35fbceea85cb945530e7902c"
  },
  {
    "url": "assets/debug-console-7.png",
    "revision": "cbc3c791bdf96d4950101a6897edfc7a"
  },
  {
    "url": "assets/debug-console-8.png",
    "revision": "0f2510b657d0ddfbbba94f4acdcf2976"
  },
  {
    "url": "assets/debug-console-9.png",
    "revision": "4ed8e745acf9de704fa4a35dcbcff1eb"
  },
  {
    "url": "assets/electron-1.png",
    "revision": "f14937b84f8b82c7dc3defa2768f86cf"
  },
  {
    "url": "assets/electron-2.png",
    "revision": "6d71a400326ec458b1ce65783dab1bc2"
  },
  {
    "url": "assets/electron-3.png",
    "revision": "67ae0897d7285f1cbda5098b916c2bfd"
  },
  {
    "url": "assets/electron-4.png",
    "revision": "e0d058b3344f1dec29b83fdefc143906"
  },
  {
    "url": "assets/electron-5.png",
    "revision": "4adc876685b0b1f898797d097dfd766c"
  },
  {
    "url": "assets/fit-vue-1.png",
    "revision": "42826d04d5b2b629ef0311a46f07648a"
  },
  {
    "url": "assets/flex-align.png",
    "revision": "0ec9e81c35c66f66b23e724c6063fce8"
  },
  {
    "url": "assets/flex-content.png",
    "revision": "aade7abc9eb8c177c66d0128b1cc6ca9"
  },
  {
    "url": "assets/flex-grow.png",
    "revision": "0c40e2971edc015685f43798e9a5b90f"
  },
  {
    "url": "assets/flex-justify.png",
    "revision": "b1beedefc6a3eb52960a682ad0121926"
  },
  {
    "url": "assets/flex-layout.png",
    "revision": "8b402883445b842ca38727fc09f60d00"
  },
  {
    "url": "assets/flex-order.png",
    "revision": "70f89eba41edc0a70278c44b74747294"
  },
  {
    "url": "assets/flex-self.png",
    "revision": "0d93c40b34a77529f71ddd927dd49c82"
  },
  {
    "url": "assets/flex-shrink.jpg",
    "revision": "e24a8660e626cd488ee1e21645a92bb0"
  },
  {
    "url": "assets/git-commit-1.png",
    "revision": "5cdc2d5e57877213f8e05bad8d2cb5d4"
  },
  {
    "url": "assets/html5-1.png",
    "revision": "f242cd38d6c686c0da5185d5ab6843b0"
  },
  {
    "url": "assets/html5-2.png",
    "revision": "8cd3217339b5502df8beed1e26fa8114"
  },
  {
    "url": "assets/img/bg.2cfdbb33.svg",
    "revision": "2cfdbb338a1d44d700b493d7ecbe65d3"
  },
  {
    "url": "assets/img/iconfont.40e49907.svg",
    "revision": "40e499073350c37f960f190956a744d2"
  },
  {
    "url": "assets/img/loading.c38bb4c9.svg",
    "revision": "c38bb4c91362836bff4e41485000be83"
  },
  {
    "url": "assets/install-mongodb-1.png",
    "revision": "3706eac45398a222a6ce2c2f55d2d05c"
  },
  {
    "url": "assets/install-mongodb-2.png",
    "revision": "0178c2fa1a41776e003e6da429694a61"
  },
  {
    "url": "assets/install-mongodb-3.png",
    "revision": "a5be5999ec46b371310add0f8e41eb9a"
  },
  {
    "url": "assets/install-mongodb-4.png",
    "revision": "ba7f2761e37acc0bfad72bb319d033cc"
  },
  {
    "url": "assets/install-mongodb-5.png",
    "revision": "75b3b824bdb78c1b17c3d7d0157f0749"
  },
  {
    "url": "assets/install-mongodb-6.png",
    "revision": "0aead34bfe95bdb5f6ad6c24a5fb1bc6"
  },
  {
    "url": "assets/install-mongodb-7.png",
    "revision": "425c7a6b326c6eb8e838cf9e7d989047"
  },
  {
    "url": "assets/install-mongodb-8.png",
    "revision": "65d9b547cabb4e03c925d8b1b2d2cce8"
  },
  {
    "url": "assets/iptables-1.png",
    "revision": "1facf904b638c87d86f1f61cdeea838e"
  },
  {
    "url": "assets/iptables-2.png",
    "revision": "c3c070e3a5fd53bd5884a0bc2a38a3d7"
  },
  {
    "url": "assets/iptables-3.png",
    "revision": "f3d20fe356955ac4cd57ba10c51aed6f"
  },
  {
    "url": "assets/iptables-4.png",
    "revision": "73f2560e0d99e0a5a6b58f372bdb8402"
  },
  {
    "url": "assets/iptables-5.png",
    "revision": "137508aa932ef7734e8109e9e7f1cd26"
  },
  {
    "url": "assets/jdk-1.png",
    "revision": "afc06af71e2e18c0285ba8c5a27e1a20"
  },
  {
    "url": "assets/jdk-2.png",
    "revision": "990f965df571929bdb32d8b80ec624dc"
  },
  {
    "url": "assets/jdk-3.png",
    "revision": "070bd277fc40e4a5e4fce09f1d20d253"
  },
  {
    "url": "assets/jdk-4.png",
    "revision": "4416da5c68ec6d74f86488c24a71c6dc"
  },
  {
    "url": "assets/jdk-5.png",
    "revision": "30e3b7acf1a4cc74937b1d142f71b576"
  },
  {
    "url": "assets/jdk-6.png",
    "revision": "3225a6a4aa5340236384f73b099a27a6"
  },
  {
    "url": "assets/jdk-7.png",
    "revision": "3f43c94cdcdf7788029566744a9c2439"
  },
  {
    "url": "assets/jdk-8.png",
    "revision": "7b45a8acc9441768b445bbbf68e2ee81"
  },
  {
    "url": "assets/js/1.06133022.js",
    "revision": "7281734d2a1efc8c612d579227c9c131"
  },
  {
    "url": "assets/js/10.0c65cdf0.js",
    "revision": "cc43c4069ee41145334b84602903c6f3"
  },
  {
    "url": "assets/js/11.6a0e9e65.js",
    "revision": "2379912adf6ac79735779dc161867139"
  },
  {
    "url": "assets/js/12.4be1983b.js",
    "revision": "7db58c699869f73aa31dfc5513d6f810"
  },
  {
    "url": "assets/js/13.790bf62e.js",
    "revision": "7e8ba0f20fb59f5b1b87142f2e62fdfe"
  },
  {
    "url": "assets/js/14.dba93391.js",
    "revision": "23ba05d092be28108da5c84b0dccedac"
  },
  {
    "url": "assets/js/15.f117a77f.js",
    "revision": "47072df45c3de7b4ffa99246e8877ec3"
  },
  {
    "url": "assets/js/16.1832390e.js",
    "revision": "e65f71e7b422de3769e0dcdf26be1dbb"
  },
  {
    "url": "assets/js/17.c233510b.js",
    "revision": "354de78db06a8432a51315658ffbf430"
  },
  {
    "url": "assets/js/18.4b5864ac.js",
    "revision": "f6a6d4a98703f6f0069c5ecf7aff67e5"
  },
  {
    "url": "assets/js/19.56c98024.js",
    "revision": "a724298ed656d72a0c3256dd6063050a"
  },
  {
    "url": "assets/js/20.0334c785.js",
    "revision": "fef5eac5e2e3ffc44ac1c2782e883531"
  },
  {
    "url": "assets/js/21.39903e19.js",
    "revision": "2bd8371c189ab306f402960cb6a2f01c"
  },
  {
    "url": "assets/js/22.9e95614c.js",
    "revision": "5280dd99291863c4a375a21ec06ddde9"
  },
  {
    "url": "assets/js/23.c0e42fd1.js",
    "revision": "8b3bf74b162859048f06c00ff5ef6148"
  },
  {
    "url": "assets/js/24.0c69054b.js",
    "revision": "304af8fb69c99e676980edb7995b1d58"
  },
  {
    "url": "assets/js/25.556187b8.js",
    "revision": "967ee65cf8edefb2f312f52457cde4be"
  },
  {
    "url": "assets/js/26.150db3ff.js",
    "revision": "2a50ea82028c182a3a9a2529ec41f18d"
  },
  {
    "url": "assets/js/27.fd01dd49.js",
    "revision": "39a59fee486482c771d3cbd01a48992a"
  },
  {
    "url": "assets/js/28.c97e6d38.js",
    "revision": "88cea27b6e84915d868e1922f5f6ae7f"
  },
  {
    "url": "assets/js/29.783c6d22.js",
    "revision": "38cf1aa2e1115876a189811a4fdfb26e"
  },
  {
    "url": "assets/js/3.6c2c0b5f.js",
    "revision": "cae7eef5fcd650467a321f6bdb76a724"
  },
  {
    "url": "assets/js/30.aa2a7fb9.js",
    "revision": "6b65480b0fcc3765ad918eefe48a94d4"
  },
  {
    "url": "assets/js/31.969b8f31.js",
    "revision": "08af70c2a958912dff3f5e3da65ef794"
  },
  {
    "url": "assets/js/32.79602957.js",
    "revision": "daba9570a117b87925ccd40f70552265"
  },
  {
    "url": "assets/js/33.db85dbe1.js",
    "revision": "427cfad76ce4e8bfc00e6cb327aee431"
  },
  {
    "url": "assets/js/34.71e32573.js",
    "revision": "4dfa8401a19e7d055363a12e89c3424f"
  },
  {
    "url": "assets/js/35.a5e2bc82.js",
    "revision": "a633041b39c777f9d00a45ef6d19653e"
  },
  {
    "url": "assets/js/4.7b5c4b79.js",
    "revision": "5ffe0b266583f361ada1010030ea102f"
  },
  {
    "url": "assets/js/5.2bef9b18.js",
    "revision": "4bac520f436ed3a28b3342763a44fc80"
  },
  {
    "url": "assets/js/6.45c24d02.js",
    "revision": "b89f265a890aaaa6ffbdac5d2e8ceee8"
  },
  {
    "url": "assets/js/7.51de8bdd.js",
    "revision": "ec110c3acded95551f0a22e8d2848eca"
  },
  {
    "url": "assets/js/8.1d292254.js",
    "revision": "3b34de4b7012630d083822e9e69323d5"
  },
  {
    "url": "assets/js/9.49750083.js",
    "revision": "37499ba40bd858d9b6aa5d6a20c254d2"
  },
  {
    "url": "assets/js/app.91a157f3.js",
    "revision": "52cff11d0a5792174ca154aad17828a2"
  },
  {
    "url": "assets/layout-moblie.gif",
    "revision": "b4a5ada98973f01971807452b7bd9cb1"
  },
  {
    "url": "assets/learning-vue3-2.gif",
    "revision": "a41bd4222a783c7f58bb1826f2b809cd"
  },
  {
    "url": "assets/learning-vue3-3.gif",
    "revision": "27764adda45a5aa388cb8f55affa3178"
  },
  {
    "url": "assets/learning-vue3-4.gif",
    "revision": "1faece0b97122ffdb777650c210d4b28"
  },
  {
    "url": "assets/liunx-apache-1.png",
    "revision": "8196ea1241a5892367b052f4c0e1895c"
  },
  {
    "url": "assets/liunx-apache-10.png",
    "revision": "08c83f30f9dca0d8e58b0ef14d75cbf6"
  },
  {
    "url": "assets/liunx-apache-11.png",
    "revision": "90437f00a33b11c6545ea941715741c3"
  },
  {
    "url": "assets/liunx-apache-12.png",
    "revision": "9205d74ceca466bafe9f31160eecfe75"
  },
  {
    "url": "assets/liunx-apache-13.png",
    "revision": "ce120855b1cd59d18efd9ae2c659794a"
  },
  {
    "url": "assets/liunx-apache-15.png",
    "revision": "0e751b4a139c73dc67fa22a5a7ecfcbd"
  },
  {
    "url": "assets/liunx-apache-2.png",
    "revision": "0dd545b9be4ce400613d9c1ae6b87c5c"
  },
  {
    "url": "assets/liunx-apache-3.png",
    "revision": "383759d9b539c2edd788f3b39297547b"
  },
  {
    "url": "assets/liunx-apache-4.png",
    "revision": "ea7d399b525ed7768f681e605105a704"
  },
  {
    "url": "assets/liunx-apache-5.png",
    "revision": "a7ddf7781e080ea71e222fbe4177f103"
  },
  {
    "url": "assets/liunx-apache-6.png",
    "revision": "f9cd94d52e3eff46f0db69f0d35a2b5f"
  },
  {
    "url": "assets/liunx-apache-7.png",
    "revision": "c1c39a98b41ab573a9349fe639d6b530"
  },
  {
    "url": "assets/liunx-apache-8.png",
    "revision": "a361c5297a460291b3a90bae0c6ad59d"
  },
  {
    "url": "assets/liunx-apache-9.png",
    "revision": "fc762db9aa1d6e4b57765a59284d12c9"
  },
  {
    "url": "assets/liunx-basic-1.png",
    "revision": "3ad8d564f70af6db9d1e866f21cc72aa"
  },
  {
    "url": "assets/liunx-basic-2.png",
    "revision": "1424022a631644fd9f0e7085b3c7d264"
  },
  {
    "url": "assets/liunx-basic-3.png",
    "revision": "b54cda455b1ead5911e3244609e694d1"
  },
  {
    "url": "assets/liunx-basic-4.png",
    "revision": "cb7496d4d632b505f380894e79a7474b"
  },
  {
    "url": "assets/liunx-directory-structure.png",
    "revision": "f1d4f2aa6d51db3aaca8e32debc8d28c"
  },
  {
    "url": "assets/liunx-dns-1.png",
    "revision": "e2d7a221edcff21379ccb690bb5a83ad"
  },
  {
    "url": "assets/liunx-dns-2.png",
    "revision": "53953753db181f3c3b37feabb419cd51"
  },
  {
    "url": "assets/liunx-dns-3.png",
    "revision": "454e9863047b34cd1611c5f6d9b2a631"
  },
  {
    "url": "assets/liunx-dns-4.png",
    "revision": "56b5db5ded38baf12704119e5c8880b0"
  },
  {
    "url": "assets/liunx-dns-5.png",
    "revision": "6a10e9f08af275faf954616db71abec4"
  },
  {
    "url": "assets/liunx-ftp-1.png",
    "revision": "36d88bc374cd1a7f33aa478b7a3f71ed"
  },
  {
    "url": "assets/liunx-ftp-10.png",
    "revision": "95c1fcba9d4192cfe00a89b5a778aebe"
  },
  {
    "url": "assets/liunx-ftp-11.png",
    "revision": "d989090f0dff72cd297b98eabb6393a2"
  },
  {
    "url": "assets/liunx-ftp-12.png",
    "revision": "402af6b27b31455edda8a7cce161b356"
  },
  {
    "url": "assets/liunx-ftp-13.png",
    "revision": "adcc2f20f3eee13fddfb300508955ac8"
  },
  {
    "url": "assets/liunx-ftp-14.png",
    "revision": "ae0281fd76fbab7d39c6523bc2451a0c"
  },
  {
    "url": "assets/liunx-ftp-15.png",
    "revision": "d0f99bd172bc6f9919acc59a6e8e552d"
  },
  {
    "url": "assets/liunx-ftp-16.png",
    "revision": "d9306dbc722676484fa9fcb1f35ca3fb"
  },
  {
    "url": "assets/liunx-ftp-17.png",
    "revision": "fc1328ba69201769e2702c5430454ebc"
  },
  {
    "url": "assets/liunx-ftp-18.png",
    "revision": "92e5ea43b350aa2e2faa56326dcd0d63"
  },
  {
    "url": "assets/liunx-ftp-19.png",
    "revision": "bac98ee712e638f4636ea9028d26f934"
  },
  {
    "url": "assets/liunx-ftp-2.png",
    "revision": "191ca842fed0acb370e4a956d4706d69"
  },
  {
    "url": "assets/liunx-ftp-20.png",
    "revision": "74e72ad2842d6d0edf9d519c3b7c6ed7"
  },
  {
    "url": "assets/liunx-ftp-21.png",
    "revision": "d9846dd717947de0bb069c026432823f"
  },
  {
    "url": "assets/liunx-ftp-23.png",
    "revision": "9fce1c6152495505af40377f279ce067"
  },
  {
    "url": "assets/liunx-ftp-24.png",
    "revision": "7c9d2f8c8c72919955b074f8b081f7a3"
  },
  {
    "url": "assets/liunx-ftp-25.png",
    "revision": "1deb799c86e2cddba4a8d1c9a67a7db0"
  },
  {
    "url": "assets/liunx-ftp-3.png",
    "revision": "54d458f789929ddf0ab4ee4279ef2d8f"
  },
  {
    "url": "assets/liunx-ftp-4.png",
    "revision": "cc6d80e31ec78d47be1230002df3cf34"
  },
  {
    "url": "assets/liunx-ftp-5.png",
    "revision": "d2866a90c7547266b45ae428812b3b54"
  },
  {
    "url": "assets/liunx-ftp-6.png",
    "revision": "82809934360b88c787cfd1a6d314963b"
  },
  {
    "url": "assets/liunx-ftp-7.png",
    "revision": "8aaf890c50fedc06fdc603ebb2a9427a"
  },
  {
    "url": "assets/liunx-ftp-8.png",
    "revision": "96f924cea0b4c1702a952502e2ab65fa"
  },
  {
    "url": "assets/liunx-ftp-9.png",
    "revision": "0b24fab70703941c68ef78741a2f3be9"
  },
  {
    "url": "assets/liunx-nmap-1.png",
    "revision": "8a55649da2f285095f7ae40da5537fe1"
  },
  {
    "url": "assets/liunx-nmap-2.png",
    "revision": "83eb2704852f71775b991385683adfb5"
  },
  {
    "url": "assets/liunx-nmap-3.png",
    "revision": "5b665c8329af30f864c80d9c50e48731"
  },
  {
    "url": "assets/liunx-nmap-4.png",
    "revision": "3fec88f5da9ec38abc88e2674aa318bb"
  },
  {
    "url": "assets/liunx-nmap-5.png",
    "revision": "b85d3635c2b21382783b59deaeb4732e"
  },
  {
    "url": "assets/liunx-nmap-6.png",
    "revision": "d8cc338337fa8ffedd9a189cb05aa48f"
  },
  {
    "url": "assets/liunx-nmap-7.png",
    "revision": "65a2a48ed4cda2e23e86739b5044b2a3"
  },
  {
    "url": "assets/liunx-samba-1.png",
    "revision": "ce7071ee907fc996196f471e5ded3f52"
  },
  {
    "url": "assets/liunx-samba-2.png",
    "revision": "7616a1bb621248f87771f79548ab1d64"
  },
  {
    "url": "assets/liunx-samba-3.png",
    "revision": "5ec66f3d8525c723310f768a63eb70e4"
  },
  {
    "url": "assets/liunx-samba-4.png",
    "revision": "95c46c427c359d2ddd1c9d467f3b1784"
  },
  {
    "url": "assets/liunx-samba-5.png",
    "revision": "9b7ddbd3ff6cde20686cb81aa434a7ac"
  },
  {
    "url": "assets/liunx-samba-6.png",
    "revision": "c8570db0937938dadbe62c0209602f2e"
  },
  {
    "url": "assets/liunx-samba-7.png",
    "revision": "6f5d3c1a938a219da65b8a9641801b82"
  },
  {
    "url": "assets/liunx-samba-8.png",
    "revision": "3cf103e624453a9465c2f5b7bd59abb8"
  },
  {
    "url": "assets/liunx-samba-9.png",
    "revision": "2cfa88517f9d5dfe374080b0ced69042"
  },
  {
    "url": "assets/liunx-ssh-1.png",
    "revision": "cecccd977fae2c5a79b3ddd9b409bb83"
  },
  {
    "url": "assets/liunx-ssh-10.png",
    "revision": "e36852ebc6e2ab6c966c3b28958866d5"
  },
  {
    "url": "assets/liunx-ssh-11.png",
    "revision": "d56762a1014d606eb1fc06b1502e1e7b"
  },
  {
    "url": "assets/liunx-ssh-12.png",
    "revision": "dc66cb0985bcadedb55043e4f8dc91f8"
  },
  {
    "url": "assets/liunx-ssh-13.png",
    "revision": "9cd5de1ea512edd19d021961357b2bc3"
  },
  {
    "url": "assets/liunx-ssh-14.png",
    "revision": "2cb80d14ae414aad00ead6a2fe9daa23"
  },
  {
    "url": "assets/liunx-ssh-2.png",
    "revision": "a25019b4a26aa5d551cccea715a1ade9"
  },
  {
    "url": "assets/liunx-ssh-3.png",
    "revision": "a2069c1121156773b1e566fe388b3875"
  },
  {
    "url": "assets/liunx-ssh-4.png",
    "revision": "da5c8dc0b89f9b678749ff3ea7db3848"
  },
  {
    "url": "assets/liunx-ssh-5.png",
    "revision": "cb42ad5db175a9f4999da80bcaeefdda"
  },
  {
    "url": "assets/liunx-ssh-6.png",
    "revision": "4336360ac5762d55653d1e7fb46a99f2"
  },
  {
    "url": "assets/liunx-ssh-7.png",
    "revision": "93796fb862ab7008b17c8f3beb16892b"
  },
  {
    "url": "assets/liunx-ssh-8.png",
    "revision": "17ede0510d15eccc68553d0f53c1c105"
  },
  {
    "url": "assets/liunx-ssh-9.png",
    "revision": "77264bd2bfc6bbaea7cec9a615fada65"
  },
  {
    "url": "assets/liunx-vim.png",
    "revision": "df89d0b762906d088e699bd313464e33"
  },
  {
    "url": "assets/mysql-reinforce-1.png",
    "revision": "8508eeec236c142efaa43f7eb06f4a39"
  },
  {
    "url": "assets/mysql-reinforce-10.png",
    "revision": "1dac47ecb72c35b5202ec6ea61284ae7"
  },
  {
    "url": "assets/mysql-reinforce-11.png",
    "revision": "bbb18127ec0c0c4c7cb8b01f237a81ac"
  },
  {
    "url": "assets/mysql-reinforce-12.png",
    "revision": "f8afd0cc88ee032d61f827187ba4ec63"
  },
  {
    "url": "assets/mysql-reinforce-13.png",
    "revision": "7c34127f04432fb721eb2a0f0b308545"
  },
  {
    "url": "assets/mysql-reinforce-14.png",
    "revision": "cd7be6736eb5b2f9b26df41083085bdc"
  },
  {
    "url": "assets/mysql-reinforce-15.png",
    "revision": "274acee5f88ace9c1677358b74dff87d"
  },
  {
    "url": "assets/mysql-reinforce-16.png",
    "revision": "da54ee3c85c478791838ae1c3428e15b"
  },
  {
    "url": "assets/mysql-reinforce-17.png",
    "revision": "8d547348ea654dffdd6b0d614fbd4236"
  },
  {
    "url": "assets/mysql-reinforce-18.png",
    "revision": "e574eb607b900085def3e323bee0f092"
  },
  {
    "url": "assets/mysql-reinforce-19.png",
    "revision": "75c7e94caba1b668d38cc41a971215b7"
  },
  {
    "url": "assets/mysql-reinforce-2.png",
    "revision": "96fb88f6915dfa2f5ee3b11d77ea36e4"
  },
  {
    "url": "assets/mysql-reinforce-20.png",
    "revision": "a6cc8c17e1999a39d97bdd3d9c72dcbf"
  },
  {
    "url": "assets/mysql-reinforce-21.png",
    "revision": "bb710ba62d3c3dd93ebab2fd5cee5fb8"
  },
  {
    "url": "assets/mysql-reinforce-22.png",
    "revision": "996265ce693ae6741623b2954b20dd1a"
  },
  {
    "url": "assets/mysql-reinforce-23.png",
    "revision": "cab27d888910d78aad1558c239f8faa1"
  },
  {
    "url": "assets/mysql-reinforce-24.png",
    "revision": "e4ce2f1209b4d54a68318d77d1007da0"
  },
  {
    "url": "assets/mysql-reinforce-25.png",
    "revision": "a91f76a6da463755be6369ce26faba2c"
  },
  {
    "url": "assets/mysql-reinforce-26.png",
    "revision": "7674a06f912743f97190134c6d7a09e9"
  },
  {
    "url": "assets/mysql-reinforce-3.png",
    "revision": "0050b08df0003f6d0a70ecac33ee3864"
  },
  {
    "url": "assets/mysql-reinforce-4.png",
    "revision": "e7ff5ef2af266611ae4e1071ff5ca075"
  },
  {
    "url": "assets/mysql-reinforce-5.png",
    "revision": "8bcb4fb73646859d862c13c27046bea6"
  },
  {
    "url": "assets/mysql-reinforce-6.png",
    "revision": "abbd3b733ce98f66959d505a7a8e9609"
  },
  {
    "url": "assets/mysql-reinforce-7.png",
    "revision": "b6335c0da25aa9345d2a4a5094948c8b"
  },
  {
    "url": "assets/mysql-reinforce-8.png",
    "revision": "1059de2ab389f270dcfa4415a206cffc"
  },
  {
    "url": "assets/mysql-reinforce-9.png",
    "revision": "0375c1c424c123214390dfd0267de828"
  },
  {
    "url": "assets/nodejs-cli-1.png",
    "revision": "930338709d139777e6fd59c6a51e3998"
  },
  {
    "url": "assets/nodejs-cli-2.png",
    "revision": "87263b722f7174655ae9d67bd8198185"
  },
  {
    "url": "assets/nodejs-cli-3.png",
    "revision": "d7c9d263dea258eada1981ad438481a6"
  },
  {
    "url": "assets/open-broswer-debug-2.png",
    "revision": "accd36666d45dc359e780260c074b333"
  },
  {
    "url": "assets/open-browser-debug-1.png",
    "revision": "62d3b2c04a470b459129f4e79bbdb26f"
  },
  {
    "url": "assets/open-browser-debug-3.png",
    "revision": "c7b3b78628dcd1a164bcce44ef0b8199"
  },
  {
    "url": "assets/process-1.png",
    "revision": "d36764744a4ca27f4427ece44fcecc12"
  },
  {
    "url": "assets/process-2.png",
    "revision": "2efc9c09c5bd0294d9f55fa0d7166f43"
  },
  {
    "url": "assets/process-3.png",
    "revision": "234ce56b7a5739c8d3f129152bad01e8"
  },
  {
    "url": "assets/process-4.png",
    "revision": "5fee9b418d9a4998569916fbc544308f"
  },
  {
    "url": "assets/process-5.png",
    "revision": "36d583b5871ef77461f4f8d5268058a4"
  },
  {
    "url": "assets/process-6.png",
    "revision": "6d689fb5e0050b2465af49ed13dee5fc"
  },
  {
    "url": "assets/process-7.png",
    "revision": "beae4bac7e09c1bda278779b7b3797b2"
  },
  {
    "url": "assets/redhat-reinforce-1.png",
    "revision": "c7182f615e81965a8863fa883db4d2da"
  },
  {
    "url": "assets/redhat-reinforce-10.png",
    "revision": "6c706b0914d1e78217cb356a2b39379a"
  },
  {
    "url": "assets/redhat-reinforce-11.png",
    "revision": "5b26ba0cacc1651d56be60b6546c5390"
  },
  {
    "url": "assets/redhat-reinforce-12.png",
    "revision": "db401b83c498bc8580ad0d9e953ee96b"
  },
  {
    "url": "assets/redhat-reinforce-14.png",
    "revision": "5b980771f1642da1a496296efa1486cc"
  },
  {
    "url": "assets/redhat-reinforce-15.png",
    "revision": "062b823f329a625f7f7cf3fae26f79b3"
  },
  {
    "url": "assets/redhat-reinforce-16.png",
    "revision": "f3ca86c0a7b82180066afbb1a860916b"
  },
  {
    "url": "assets/redhat-reinforce-17.png",
    "revision": "8cdeae281194049f45b139de6593d29a"
  },
  {
    "url": "assets/redhat-reinforce-2.png",
    "revision": "ebe7ab1dc1d108a23e1ac264da9c9f85"
  },
  {
    "url": "assets/redhat-reinforce-3.png",
    "revision": "618f655f71c7c33f9eab419c7864bda5"
  },
  {
    "url": "assets/redhat-reinforce-4.png",
    "revision": "45f15640e73b4474f2ae84f13ad079ba"
  },
  {
    "url": "assets/redhat-reinforce-5.png",
    "revision": "9e9204c870132172cbb8ffaa5fab0511"
  },
  {
    "url": "assets/redhat-reinforce-6.png",
    "revision": "c022c941840637adae1a5f760709c68b"
  },
  {
    "url": "assets/redhat-reinforce-7.png",
    "revision": "c69f2cef34ac7a2a517b9f71770684f0"
  },
  {
    "url": "assets/redhat-reinforce-8.png",
    "revision": "0b01ae225380f0c005b3cc5ceae20ead"
  },
  {
    "url": "assets/redhat-reinforce-9.png",
    "revision": "ba17633083e90148a9c1b0f109be3f6f"
  },
  {
    "url": "assets/setgid.png",
    "revision": "662845766314419be8a247852ff013c7"
  },
  {
    "url": "assets/setuid-1.png",
    "revision": "b3c503491719ae8e5ecdc6ae85731686"
  },
  {
    "url": "assets/setuid-2.png",
    "revision": "d05b3fa1f430c9d4e8943ffd84b8c741"
  },
  {
    "url": "assets/shell-new-files-1.png",
    "revision": "7176a9b85666050956f2e5fab25f8759"
  },
  {
    "url": "assets/shell-new-files-2.png",
    "revision": "8aaa899f29cbee404abf73564ca25caa"
  },
  {
    "url": "assets/shell-new-files-3.png",
    "revision": "7dbb686c6afd627e7f29e3736dfcf37b"
  },
  {
    "url": "assets/shell-new-files-4.png",
    "revision": "abf13516ccf3794638e0a5566eae6b1e"
  },
  {
    "url": "assets/socks-proxy-1.png",
    "revision": "b76ddfb500bc448d1b75081b2818660b"
  },
  {
    "url": "assets/socks-proxy-2.png",
    "revision": "c9a84fa55131c9d47cf78c534822a3eb"
  },
  {
    "url": "assets/socks-proxy-3.png",
    "revision": "caf5bb1cc75bb17ef618ef3855cdebe1"
  },
  {
    "url": "assets/socks-proxy-4.png",
    "revision": "3517bd335bc893e73996138c648063be"
  },
  {
    "url": "assets/socks-proxy-5.png",
    "revision": "f4c5d8b16bc58096eee6a07de8914ba7"
  },
  {
    "url": "assets/socks-proxy-6.png",
    "revision": "880d7285267a1a1030046fd936bd76b2"
  },
  {
    "url": "assets/socks-proxy-7.png",
    "revision": "ff5e7a8d8d31cefb945e8b6a9df14f7a"
  },
  {
    "url": "assets/software.jpg",
    "revision": "687528d988f260ff65182f96bd63417d"
  },
  {
    "url": "assets/spring-boot-1.png",
    "revision": "cb1f15be376588114287ca7ff93babd1"
  },
  {
    "url": "assets/spring-boot-10.png",
    "revision": "b859accfc172562ba947f7687cd07329"
  },
  {
    "url": "assets/spring-boot-11.png",
    "revision": "3825d7d15e1ed099c361669275d53e71"
  },
  {
    "url": "assets/spring-boot-12.png",
    "revision": "d2d3663e36c36403c2186f24d74e4755"
  },
  {
    "url": "assets/spring-boot-13.png",
    "revision": "c2cf8028174bdbfca1bef440558456d1"
  },
  {
    "url": "assets/spring-boot-2.png",
    "revision": "1c4dc4f44bd9da9f1339f9143ca3e128"
  },
  {
    "url": "assets/spring-boot-3.png",
    "revision": "8b6cd45a5106ab10fd505e047470b3f8"
  },
  {
    "url": "assets/spring-boot-4.png",
    "revision": "10c0512b72e2ab9ca9b205ac92f50d75"
  },
  {
    "url": "assets/spring-boot-5.png",
    "revision": "5f3f7be1fe7d1532e0f2f951983e3668"
  },
  {
    "url": "assets/spring-boot-6.png",
    "revision": "1f85e4219991babc6ac189cd3be7037c"
  },
  {
    "url": "assets/spring-boot-7.png",
    "revision": "a739ef0847307d9b0c2d4795f4263046"
  },
  {
    "url": "assets/spring-boot-8.png",
    "revision": "681cf7d81168d01d19280b480848a5a8"
  },
  {
    "url": "assets/spring-boot-9.png",
    "revision": "cc487b98cdd1d9af5baca92eb164ffe5"
  },
  {
    "url": "assets/sqlmap-1.png",
    "revision": "ca74346e95758451d5081ba2b6d189c3"
  },
  {
    "url": "assets/sqlmap-2.png",
    "revision": "de15998198215f14f39be7beb11ab0d1"
  },
  {
    "url": "assets/sqlmap-3.png",
    "revision": "3f9fedaf9839e5d3ad3f89db2b72de1d"
  },
  {
    "url": "assets/sqlmap-4.png",
    "revision": "8525b8b66db92f4286a69e655b9bd8f5"
  },
  {
    "url": "assets/sqlmap-5.png",
    "revision": "32bb8f708f35e14ac88f8c35f51e14f8"
  },
  {
    "url": "assets/sqlmap-6.png",
    "revision": "e4543dc9c037e166c0144ba5ae0c448d"
  },
  {
    "url": "assets/sqlmap-7.png",
    "revision": "91d3d6bcbba7b7933115df92d95e8854"
  },
  {
    "url": "assets/sqlmap-8.png",
    "revision": "e019b2338b814d678a3c7308075851df"
  },
  {
    "url": "assets/sqlmap-9.png",
    "revision": "0f4c9f49caebdd8d6975168b5927cad0"
  },
  {
    "url": "assets/sticky-bit.png",
    "revision": "2c47a93032fe729326c30ad40a06a999"
  },
  {
    "url": "assets/vscode-config-1.png",
    "revision": "84ec59a4f1078e4ffd64b01822240555"
  },
  {
    "url": "assets/vscode-config-2.png",
    "revision": "45fe3310cef9304561e7266e29f9848c"
  },
  {
    "url": "assets/vscode-config-3.png",
    "revision": "e1c35c92d6ea2bfa2fca79d3ce2fba82"
  },
  {
    "url": "assets/vue-performance-1.png",
    "revision": "1cbf34a8387e6da4f696b86e52f23654"
  },
  {
    "url": "assets/vue-performance-2.png",
    "revision": "6d38f934dd45b94a7b84c6a3d2c4fa0f"
  },
  {
    "url": "assets/vue-performance-3.png",
    "revision": "6df11896f438cc10a8ad621548c7318c"
  },
  {
    "url": "assets/vuepress-blog-1.png",
    "revision": "e600eb9aba2f4118db452c70110fd01b"
  },
  {
    "url": "assets/vuepress-blog-2.png",
    "revision": "c2414c41edbc52f5cdff0ab9ce14530c"
  },
  {
    "url": "assets/vuepress-blog-3.png",
    "revision": "b5da88dcdb74cb35d6462d2c9b73ed09"
  },
  {
    "url": "assets/vuepress-blog-4.png",
    "revision": "6d2c979d2ee8cd157a9848b9cea0aa08"
  },
  {
    "url": "assets/vuepress-blog-5.png",
    "revision": "f800516c0e4f6f3ac43ceef3cbf65c1d"
  },
  {
    "url": "assets/vuepress-blog-6.png",
    "revision": "b2d422fc5fbe871ab7acad68c89aba09"
  },
  {
    "url": "assets/vuepress-blog-7.png",
    "revision": "53a48c1820d3c6870da29758b319a03f"
  },
  {
    "url": "assets/web-design-1.png",
    "revision": "6f7e0220ff201731205f92e5accb56d8"
  },
  {
    "url": "assets/web-design-2.png",
    "revision": "362daf994dfd4fd41f7c8232fbb3b396"
  },
  {
    "url": "assets/web-design-3.jpg",
    "revision": "36f17f31048308463874bc95650140d9"
  },
  {
    "url": "assets/window-object.png",
    "revision": "890edc8b4e3cd91c70695d9b210ffb7a"
  },
  {
    "url": "assets/yesterday.8e49f298.svg",
    "revision": "8e49f298844ce8a7235c197d5d12e4c4"
  },
  {
    "url": "avatar - 副本.jpeg",
    "revision": "5e8817c6e3431f1c8807602bdd03a095"
  },
  {
    "url": "avatar.png",
    "revision": "2062001bc7f62654d92de895f6da724a"
  },
  {
    "url": "banner - 副本.jpg",
    "revision": "f2bd4594cd7f6a1bb17673b51699206a"
  },
  {
    "url": "banner.jpg",
    "revision": "f2bd4594cd7f6a1bb17673b51699206a"
  },
  {
    "url": "categories/CV/index.html",
    "revision": "ba6a7ccfa3400b89470964b71808f0f4"
  },
  {
    "url": "categories/Exp/index.html",
    "revision": "037245b27719af7744c94a790f21f589"
  },
  {
    "url": "categories/Hadoop/index.html",
    "revision": "029148842e6e528d07951da9f4685ff9"
  },
  {
    "url": "categories/index.html",
    "revision": "1f15ebf3a93b9431ac8b38e57306d67d"
  },
  {
    "url": "categories/Music/index.html",
    "revision": "6424de55473fb42ace80fe491675f3b3"
  },
  {
    "url": "categories/thinks/index.html",
    "revision": "b1104b279fb0e1f7f790a83744df0ce9"
  },
  {
    "url": "categories/深度学习/index.html",
    "revision": "1cbfe3d21cb90bcf74dd96ae1c853023"
  },
  {
    "url": "categories/闲言碎语/index.html",
    "revision": "25be83ecf7aac962250ecb9e64f623c3"
  },
  {
    "url": "iconfont/iconfont.css",
    "revision": "c8b00d812608bf98f806b55fa4140795"
  },
  {
    "url": "iconfont/iconfont.eot",
    "revision": "0fe2ea06e44b4c5586cd81edfb62fa67"
  },
  {
    "url": "iconfont/iconfont.svg",
    "revision": "40e499073350c37f960f190956a744d2"
  },
  {
    "url": "iconfont/iconfont.ttf",
    "revision": "b2bb6a1eda818d2a9d922d41de55eeb1"
  },
  {
    "url": "iconfont/iconfont.woff",
    "revision": "3779cf87ccaf621f668c84335713d7dc"
  },
  {
    "url": "iconfont/iconfont.woff2",
    "revision": "66dad00c26f513668475f73f4baa29aa"
  },
  {
    "url": "icons/android-chrome-192x192.png",
    "revision": "ee9693b6d1323c35ab47222d8f2cb237"
  },
  {
    "url": "icons/android-chrome-512x512.png",
    "revision": "110f83b3656390243816823b863d19f9"
  },
  {
    "url": "icons/apple-touch-icon-120x120.png",
    "revision": "083fd5693709f1bb756d7146f92cfff0"
  },
  {
    "url": "icons/apple-touch-icon-152x152.png",
    "revision": "0ad9644b653ab367462d48e9cb31653f"
  },
  {
    "url": "icons/apple-touch-icon-180x180.png",
    "revision": "6537810f5c1f67bd9f285ab8f817dc33"
  },
  {
    "url": "icons/apple-touch-icon-60x60.png",
    "revision": "7afb2b4fd95bd147a9f9c9dcd8be96a7"
  },
  {
    "url": "icons/apple-touch-icon-76x76.png",
    "revision": "322c094e811fa22948fa838553168be6"
  },
  {
    "url": "icons/apple-touch-icon.png",
    "revision": "6537810f5c1f67bd9f285ab8f817dc33"
  },
  {
    "url": "icons/favicon-16x16.png",
    "revision": "86902dea6b16aaf02b26ef1299313344"
  },
  {
    "url": "icons/favicon-32x32.png",
    "revision": "9c4b5b6a6755765277a8d344cef51a90"
  },
  {
    "url": "icons/msapplication-icon-144x144.png",
    "revision": "e217effa9bf49048ebe5f0c3c0b9bf83"
  },
  {
    "url": "icons/mstile-150x150.png",
    "revision": "73680937b571e80d379a0d099979548f"
  },
  {
    "url": "icons/safari-pinned-tab.svg",
    "revision": "cf0c951947bdfe5abdfc0fe63e7ff297"
  },
  {
    "url": "index.html",
    "revision": "461066d2d71a3c4bb8a8a6ae732f7e30"
  },
  {
    "url": "kesshouban/model.2048/texture_00.png",
    "revision": "2e6411636e81d3e58e8315bfa586ba8d"
  },
  {
    "url": "project/index.html",
    "revision": "7cd5c1b80db0e3a6f5dd565a5536c371"
  },
  {
    "url": "tag/CV/index.html",
    "revision": "0c825054ec791c563af7b4184aa1f1da"
  },
  {
    "url": "tag/deeplearn/index.html",
    "revision": "524696d542a45fe4761ee286984f5fa6"
  },
  {
    "url": "tag/DL/index.html",
    "revision": "a457ce5a93d8cc0e66708f05274e1c14"
  },
  {
    "url": "tag/Exp/index.html",
    "revision": "e35285007fdf837fef15d19f1a29c567"
  },
  {
    "url": "tag/Git/index.html",
    "revision": "35153dd5d0da42b6481a3d7a8f125868"
  },
  {
    "url": "tag/Hadoop/index.html",
    "revision": "f9b362f452033a198d6fb8e281138604"
  },
  {
    "url": "tag/Hive/index.html",
    "revision": "a1d4502c426e2ec12292d1f55c58f8a0"
  },
  {
    "url": "tag/index.html",
    "revision": "fa819d63e2d6a4041f2d82caa81906bd"
  },
  {
    "url": "tag/LeetCode/index.html",
    "revision": "a0eab4f67af332321d8f40ffaa93e455"
  },
  {
    "url": "tag/music/index.html",
    "revision": "885ecbe9cefea1f6d10ebff10f875712"
  },
  {
    "url": "tag/NLP/index.html",
    "revision": "d87341f09d42b91c518be538c08d2a0d"
  },
  {
    "url": "tag/trials/index.html",
    "revision": "4c796ea98e4adf4e3f461675f4a54163"
  },
  {
    "url": "timeline/index.html",
    "revision": "2a64971ac4120f5e092c3486f782a4de"
  },
  {
    "url": "views/CV/VAE.html",
    "revision": "6b3f6fdff452d44465a844cdd74c6241"
  },
  {
    "url": "views/Deeplearn/Basic.html",
    "revision": "dfde292aa1d86d1f96b817596d80de66"
  },
  {
    "url": "views/Deeplearn/Besic_ec.html",
    "revision": "71373004be9b581496885e44f0f520b4"
  },
  {
    "url": "views/Deeplearn/Different_conv.html",
    "revision": "e34ae8cbe5586b311b88afacc91b081a"
  },
  {
    "url": "views/Deeplearn/DisLearn.html",
    "revision": "48645336bbb9c8d9a89fc9bb0625546f"
  },
  {
    "url": "views/Deeplearn/FewShortLearning.html",
    "revision": "4c04b2216b308aa07ce6b8b48df5334e"
  },
  {
    "url": "views/Deeplearn/ReadingList.html",
    "revision": "9c59882cbe1841edbf1f1d307585c98f"
  },
  {
    "url": "views/Deeplearn/Self-attention-Bert_EC.html",
    "revision": "7eb3827e526641d96dc75ef9dcd47871"
  },
  {
    "url": "views/Deeplearn/Uneven_Data.html",
    "revision": "cb29d86af992ed05fb95f76bfc87e7eb"
  },
  {
    "url": "views/Deeplearn/why_use_module.html",
    "revision": "5ae856abee513f8f24969af50575861c"
  },
  {
    "url": "views/Exp/Exp.html",
    "revision": "f04b5f318a171c0bb9cf99775996f990"
  },
  {
    "url": "views/Git/Vue.html",
    "revision": "cee5e3057a3b566d60b8151b873a2ddf"
  },
  {
    "url": "views/Hadoop/Hive.html",
    "revision": "cab2f89770d02bc932a06bba94f10065"
  },
  {
    "url": "views/Hadoop/Mynote.html",
    "revision": "8a2f812eec7be2ca1bff6195e27b1af9"
  },
  {
    "url": "views/LeetCode/Leetcode.html",
    "revision": "572fe023ea9ab3a5fa09b820827c1a26"
  },
  {
    "url": "views/Trials/blah.html",
    "revision": "5959d8803206f329b1d68d81aa49d31f"
  },
  {
    "url": "views/Trials/lifeiine.html",
    "revision": "b9bd36380b8d3f97fd27b92290e61494"
  },
  {
    "url": "views/Trials/piano.html",
    "revision": "a7521e3ada468395d9f675863dd7083d"
  },
  {
    "url": "WeChat.png",
    "revision": "f7366096081ffbb417eaf1a33a3cff7e"
  }
].concat(self.__precacheManifest || []);
workbox.precaching.precacheAndRoute(self.__precacheManifest, {});
addEventListener('message', event => {
  const replyPort = event.ports[0]
  const message = event.data
  if (replyPort && message && message.type === 'skip-waiting') {
    event.waitUntil(
      self.skipWaiting().then(
        () => replyPort.postMessage({ error: null }),
        error => replyPort.postMessage({ error })
      )
    )
  }
})
