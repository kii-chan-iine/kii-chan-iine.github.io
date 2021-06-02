---
author: 星火燎原@vxhly
title: 二次封装 axios
categories: [vuejs]
tags: [Node.js, WEB, Vue.js, JavaScript]
date: 2020-02-16 16:43:59
---

<Boxx changeTime="10000"/>

::: tip 前言
前后端分离的时候, 通常后端要求前端发送 JSON 请求, 而后端返回的格式也是 JSON 的, 这时候就需要一个中间插件 axios 上场, 本篇文章主要介绍在 vuejs 项目中是如何二次封装 axios 的, 本篇适宜阅读人群为有一些的 Vue.js 的开发基础经验。
:::
<!-- more -->

## 关于 axios

::: tip 介绍
Axios 是一个基于promise 的HTTP 库, 可以用在浏览器和node.js 中。更多的介绍, 米娜桑上 GitHub 查看官方说明, [axios](https://github.com/axios/axios), 这里不做过多的阐述
:::

## 二次封装 axios 

### 安装依赖

``` bash
npm i axios
# or 
yarn add axios
```

### 基础请求地址

一般来说一个项目至少对应一个后端 API 服务地址, 所以需要一个配置文件来存储这个后端 API 地址

在 `src/config.js` 中写入以下配置

``` javascript
export const baseHost = {
    API1: {
        dev_host: 'http://[ip]:[port]/v[n]/api/', // 调试模式
        pro_host: 'http://[ip]:[port]/v[n]/api/', // 线上模式
        file_host: 'http://[ip]:[port]/v[n]/file/' // 文件存储前置 URL
    },
    API2: {
        dev_host: 'http://[ip]:[port]/v[n]/api/', // 调试模式
        pro_host: 'http://[ip]:[port]/v[n]/api/', // 线上模式
        file_host: 'http://[ip]:[port]/v[n]/file/' // 文件存储前置 URL
    }
    // 还有其他的 API 往下加
}
```

在 `src/API.vue` 中添加后端的 API 地址, 此时添加的 API 地址无需在前面的 URL 部分写出来, 参考如下

``` vue
<script>
export default {
  name: 'API',

  user: {
    UserLogin: '/user/UserLogin', // GET 用户登录

    GetUserInfo: '/user/GetUserInfo' // GET 获取用户信息
  },

  logout: '/logout', // 登出账号
}
</script>
```

在你的 `src/main.js` 中注册这些 API 地址

``` javascript
import API from './API.vue' // 后端API接口地址

Vue.prototype.API = API // 挂载到Vue实例上面
```

### 正式封装 axios

创建目录结构 `src/libraries/axios` , 并在这个目录下创建两个核心文件 `featch.js` 和 `http.js`
`featch.js` 文件主要编写 axios 的基础配置

``` javascript
import axios from 'axios'

// 创建 axios 实例
const instance = axios.create({
    responseType: 'json', // 返回数据类型
    // baseURL: serverURL, // 请求 URL
    timeout: 7000, // 超时时间
    maxContentLength: 2000, // 最大发包长度
    retry: 3, // 重试次数
    retryDelay: 3000, // 重试延时, 3秒重试一次
    shouldRetry: () => true // 重试条件, 默认只要是错误都需要重试
})

// http request 拦截器
instance.interceptors.request.use(
    config => {
        // 在 headers 头上添加参数
        config.headers['Content-Type'] = 'application/json;charset=UTF-8'
        const token = ''
        if (token) {
            config.headers['token'] = token
        }

        return config
    },
    error => {
        return Promise.reject(error)
    }
)

// http response 拦截器
instance.interceptors.response.use(
    response => {
        const {
            code
        } = response.data
        switch (code) {
            // 这边可以对后端的一些状态码进行处理
        }
    },
    error => {
        const {
            config
        } = error
        // If config does not exist or the retry option is not set, reject
        if (!config || !config.retry) return Promise.reject(error)

        // Set the variable for keeping track of the retry count
        config.__retryCount = config.__retryCount || 0

        // Check if we've maxed out the total number of retries
        if (config.__retryCount >= config.retry) {
            // Reject with the error
            Notification({
                title: '请求超时',
                message: '当前网络不佳, 请稍后刷新重试'
            })
            return Promise.reject(error)
        }

        // Increase the retry count
        config.__retryCount += 1

        // Create new promise to handle exponential backoff
        var backoff = new Promise(function(resolve) {
            setTimeout(function() {
                resolve()
            }, config.retryDelay || 1)
        })

        // Return the promise in which recalls axios to retry the request
        return backoff.then(function() {
            return instance(config)
        })
    }
)

export default instance
```

`http.js` 主要封装 `GET` 、 `POST` 、 `PATCH` 、 `DELETE` 、 `PUT` 方法

``` javascript
import featch from './featch'
import {
    omitBy
} from 'lodash'
import {
    baseHost
} from '@/config'
import axios from 'axios'
let serverURL = ''
const env = process.env.NODE_ENV

const getURL = (type) => {
    switch (env) {
        case 'production':
            serverURL = baseHost[type].pro_host
            break
        case 'development':
            serverURL = baseHost[type].dev_host
            break
        case 'testing':
            serverURL = baseHost.test_host
            break
    }
    return serverURL
}

/**
 * @method httpGet
 *
 * import { httpGet } from '@/libraries/axios/http'
 *
 * httpGet('', payload).then().catch()
 *
 * @param {String} url 请求地址
 * @param {Object} payload 请求参数
 * @param {String} type 设置 baseURL
 */

export const httpGet = async (url, payload, type = 'API1') => {
    try {
        serverURL = getURL(type)
        axios.defaults.baseURL = serverURL
        const response = await featch.get( `${url}` , {
            params: payload
        })

        const result = response.data
        return result
    } catch (err) {
        throw new Error(err)
    }
}

/**
 * @method httpPost
 *
 * import { httpPost } from '@/libraries/axios/http'
 *
 * httpPost('', payload).then().catch()
 *
 * @param {String} url 请求地址
 * @param {Object} payload 请求参数
 */
export const httpPost = async (url, payload, type = 'API1') => {
    try {
        serverURL = getURL(type)
        axios.defaults.baseURL = serverURL
        const response = await featch.post( `${url}` , payload)

        const result = response.data
        return result
    } catch (err) {
        throw new Error(err)
    }
}

/**
 * @method httpPut
 *
 * import { httpPut } from '@/libraries/axios/http'
 *
 * httpPut('', payload).then().catch()
 *
 * @param {String} url 请求地址
 * @param {Object} payload 请求参数
 */
export const httpPut = async (url, payload, type = 'API1') => {
    try {
        serverURL = getURL(type)
        axios.defaults.baseURL = serverURL
        const response = await featch.put( `${url}` , payload)

        const result = response.data
        return result
    } catch (err) {
        throw new Error(err)
    }
}

/**
 * @method httpPatch
 *
 * import { httpPatch } from '@/libraries/axios/http'
 *
 * httpPatch('', payload).then().catch()
 *
 * @param {String} url 请求地址
 * @param {Object} payload 请求参数
 */
export const httpPatch = async (url, payload, type = 'API1') => {
    try {
        serverURL = getURL(type)
        axios.defaults.baseURL = serverURL
        const response = await featch.patch( `${url}` , payload)

        const result = response.data
        return result
    } catch (err) {
        throw new Error(err)
    }
}

/**
 * @method httpDelte
 *
 * import { httpDelte } from '@/libraries/axios/http'
 *
 * httpDelte('', payload).then().catch()
 *
 * @param {String} url 请求地址
 * @param {Object} payload 请求参数
 */
export const httpDelte = async (url, payload, type = 'API1') => {
    try {
        serverURL = getURL(type)
        axios.defaults.baseURL = serverURL
        const response = await featch.delete( `${url}` , {
            data: payload
        })

        const result = response.data
        return result
    } catch (err) {
        throw new Error(err)
    }
}
```

### 使用案例

``` vue
<script>
import { httpPost } from '@/libraries/axios/http'

export default {
  mounted() {
    const url = this.API.user.UserLogin
    const payload = {
        userName: 'admin',
        userPassword: 'admin'
    }

    httpPost(url, payload)
        .then(res => {
            // 这里的 res 即是后端返回的数据结构
        })
  }
}
</script>
```
