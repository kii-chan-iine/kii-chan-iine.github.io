---
author: 星火燎原@vxhly
title: RESTful API 设计指南（补充）
categories: [manual]
tags: [Criterion] 
date: 2017-09-29 12:17:33
---

<Boxx changeTime="10000"/>

::: tip 前言
网络应用程序, 分为前端和后端两个部分。这时候必须有一种统一的机制, 方便不同的前端设备与后端进行通信。这导致 API 构架的流行, 甚至出现 "API First" 的设计思想。[ `RESTful API` ](http://en.wikipedia.org/wiki/Representational_state_transfer) 是目前比较成熟的一套互联网应用程序的 API 设计理论。
:::
<!-- more -->

## 先言

::: tip 提示
该篇博文为补充 [RESTful API 设计指南- 阮一峰的网络日志](http://www.ruanyifeng.com/blog/2014/05/restful_api.html)

[RESTful API 设计参考文献列表, 可帮助你更加彻底的了解REST风格的接口设计](https://github.com/aisuhua/restful-api-design-references)。

REST API一般用来将某种资源和允许的对资源的操作暴露给外界, 使调用者能够以正确的方式操作资源。这里, 在输入输出的处理上, 要符合HTTP/1.1（不久的将来, 要符合HTTP/2.0）的RFC, 保证接口的一致性。
:::

## 协议

API 与用户的通信协议, 总是使用 [https](http://www.ruanyifeng.com/blog/2014/02/ssl_tls.html) 协议。

## 域名

应该尽量将 API 部署在专用域名之下。

<https://api.example.com>

如果确定 API 很简单, 不会有进一步扩展, 可以考虑放在主域名下。

<https://example.org/api/>

## 版本（Versioning）

应该将 API 的版本号放入 URL。

<https://api.example.com/v1/>

另一种做法是, 将版本号放在 HTTP 头信息中, 但不如放入 URL 方便和直观。[Github](http://developer.github.com/v3/media/#request-specific-version) 采用这种做法。

## 路径（Endpoint）

路径又称"终点"（endpoint）, 表示 API 的具体网址。 在 RESTful 架构中, 每个网址代表一种资源（resource）, 所以网址中不能有动词, 只能有名词, 而且所用的名词往往与数据库的表格名对应。一般来说, 数据库中的表都是同种记录的"集合"（collection）, 所以 API 中的名词也应该使用复数。 举例来说, 有一个 API 提供动物园（zoo）的信息, 还包括各种动物和雇员的信息, 则它的路径应该设计成下面这样。

* <https://api.example.com/v1/zoos>
* <https://api.example.com/v1/animals>
* <https://api.example.com/v1/employees>、

## RESTful API 规范

* URI 请求采用小写字母, 数字, 部分特殊符号（非制表符, 该特殊符号特指 _、&、?）组成。
* URI 请求中不采用大小写混合的驼峰命名方式, 尽量采用全小写单词, 如果需要连接多个单词, 则采用连接符 `_` 连接单词
* URI 中不包含文件名, 例如 `/api/test.php`
* URI 中不包含 HTTP 动词, 动作由请求包体中的 method 决定, 语义化能够一眼看出当前的操作
* URI 不能过长, 任何 HTTP 协议客户端和服务器都应当支持 8000 字节的首行（方法名 + URI + HTTP 版本号）, 可以理解成支持 7980+ 字节的 URI
* 非 ID 形式的全部通过使用 ? 传参形式, 并且使用 encodeURI 或者 encodeURIComponent 进行编码
* 使用复数形式的单词表示大量的数据, 例如 `/api/users`
* 根据需求添加 版本号, 例如 `/api/v1/users`
* 根据需求添加 SecretKey, 特指某些对外服务盈利的接口
* 过深的导航容易导致url膨胀, 不易维护, 如 GET /zoos/1/areas/3/animals/4, 尽量使用查询参数代替路径中的实体导航, 如GET /animals?zoo=1&area=3
* 对于一些需要用户验证的查用 `OAuth` 或者 `JWT` , 并且添加 HTTP Authorization 请求头
* 关于 `Content-Type` 采用 `application/json;charset=UTF-8` 或者 ` application/x-www-form-urlencoded;charset=UTF-8`

## HTTP 动词

对于资源的具体操作类型, 由 HTTP 动词表示。 常用的 HTTP 动词有下面五个（括号里是对应的 SQL 命令）。

* **GET（SELECT）** => 从服务器取出资源（一项或多项）。
* **POST（CREATE）** => 在服务器新建一个资源。
* **PUT（UPDATE）** => 在服务器更新资源（客户端提供改变后的完整资源）。
* **PATCH（UPDATE）** => 在服务器更新资源（客户端提供改变的属性）。
* **DELETE（DELETE）** => 从服务器删除资源。

还有两个不常用的 HTTP 动词。

* **HEAD** => 获取资源的元数据。
* **OPTIONS** => 获取信息, 关于资源的哪些属性是客户端可以改变的。

下面是一些例子。

* **GET** `/zoos` => 列出所有动物园
* **POST** `/zoos` => 新建一个动物园
* **GET** `/zoos/ID` => 获取某个指定动物园的信息
* **PUT** `/zoos/ID` => 更新某个指定动物园的信息（提供该动物园的全部信息）
* **PATCH** `/zoos/ID` => 更新某个指定动物园的信息（提供该动物园的部分信息）
* **DELETE** `/zoos/ID` => 删除某个动物园
* **GET** `/zoos/ID/animals` => 列出某个指定动物园的所有动物
* **DELETE** `/zoos/ID/animals/ID` => 删除某个指定动物园的指定动物

## 过滤信息（Filtering）

如果记录数量很多, 服务器不可能都将它们返回给用户。AP I 应该提供参数, 过滤返回结果。 下面是一些常见的参数。

* `?limit=10` => 指定返回记录的数量
* `?offset=10` => 指定返回记录的开始位置。
* `?page=2&per_page=100` => 指定第几页, 以及每页的记录数。
* `?sortby=nameℴ=asc` => 指定返回结果按照哪个属性排序, 以及排序顺序。
* `?animal_type_id=1` => 指定筛选条件

参数的设计允许存在冗余, 即允许 API 路径和 URL 参数偶尔有重复。比如, GET `/zoo/ID/animals` 与 GET `/animals?zoo_id=ID` 的含义是相同的。

## 状态码（Status Codes）

服务器向用户返回的状态码和提示信息, 常见的有以下一些（方括号中是该状态码对应的 HTTP 动词）。

### 请求成功

* **200** `OK` -> 【 `GET` 】 => 服务器成功返回用户请求的数据, 该操作是幂等的（Idempotent）。
* **201** `CREATED` -> 【 `POST/PUT/PATCH` 】 => 对象创建成功并返回相应资源数据, 如 `POST` 成功；创建完成后响应头中应该携带头标 `Location` , 指向新建资源的地址
* **202** `Accepted` -> 【 `*` 】 => 接受请求, 但无法立即完成创建行为, 比如其中涉及到一个需要花费若干小时才能完成的任务。返回的实体中应该包含当前状态的信息, 以及指向处理状态监视器或状态预测的指针, 以便客户端能够获取最新状态。
* **204** `No Content ` -> 【 `DELETE/PATCH` 】 => 请求执行成功, 不返回相应资源数据

### 重定向

重定向的新地址都需要在响应头 Location 中返回

* **301** `Moved Permanently ` -> 【 `*` 】 => 被请求的资源已永久移动到新位置
* **302** `Found` -> 【 `*` 】 => 请求的资源现在临时从不同的 URI 响应请求
* **303** `See Other` -> 【 `*` 】 => 应当前请求的响应可以在另一个 URI 上被找到, 客户端应该使用 GET 方法进行请求。比如在创建已经被创建的资源时, 可以返回 303
* **307** `Temporary Redirect ` -> 【 `*` 】 => 对应当前请求的响应可以在另一个 URI 上被找到, 客户端应该保持原有的请求方法进行请求

### 条件请求

* **304** `Not Modified` -> 【 `GET/HEAD` 】 => 未改变说明无需再次传输请求的内容, 也就是说可以使用缓存的内容。这通常是在一些安全的方法（safe）, 资源自从上次请求后没有再次发生变化, 主要使用场景在于实现数据缓存
* **409** `Conflict ` -> 【 `GET/HEAD` 】 => 请求操作和资源的当前状态存在冲突。主要使用场景在于实现并发控制
* **412** `Precondition Failed` -> 【 `GET/HEAD` 】 => 服务器在验证在请求的头字段中给出先决条件时, 没能满足其中的一个或多个。主要使用场景在于实现并发控制

### 客户端错误

* **400** `Bad Request` -> 【 `POST/PUT/PATCH` 】 => 请求体包含语法错误, 用户发出的请求有错误, 服务器没有进行新建或修改数据的操作, 该操作是幂等的。
* **401** `Unauthorized` -> 【 `*` 】 => 表示用户没有权限（令牌、用户名、密码错误）。需要验证用户身份, 如果服务器就算是身份验证后也不允许客户访问资源, 应该响应 403 Forbidden 。如果请求里有 Authorization 头, 那么必须返回一个 WWW-Authenticate 头
* **403** `Forbidden` -> 【 `*` 】 => 服务器拒绝执行, 表示用户得到授权（与 401 错误相对）, 但是访问是被禁止的。
* **404** `Not Found` -> 【 `*` 】 => 用户发出的请求针对的是不存在的记录, 服务器没有进行操作, 该操作是幂等的。
* **406** `Not Acceptable` -> 【 `GET` 】 => 用户请求的格式不可得（比如用户请求 JSON 格式, 但是只有 XML 格式）。服务器不支持客户端请求的内容格式, 但响应里会包含服务端能够给出的格式的数据, 并在 Content-Type 中声明格式名称
* **410** `Gone` -> 【 `GET` 】 => 用户请求的资源被永久删除, 且不会再得到的。被请求的资源已被删除, 只有在确定了这种情况是永久性的时候才可以使用, 否则建议使用 404 Not Found
* **413** `Payload Too Large` -> 【 `POST/PUT` 】=> POST 或者 PUT 请求的消息实体过大
* **415** `Unsupported Media Type` -> 【 `GET` 】=> 服务器不支持请求中提交的数据的格式
* **422** `Unprocessable Entity` -> 【 `POST/PUT/PATCH` 】=> 请求格式正确, 但是由于含有语义错误, 无法响应
* **428** `Precondition Required` -> 【 `GET` 】 => 要求先决条件, 如果想要请求能成功必须满足一些预设的条件

### 服务端错误

* **500** `INTERNAL SERVER ERROR` -> 【 `*` 】 => 服务器遇到了一个未曾预料的状况, 导致了它无法完成对请求的处理。
* **501** `Not Implemented` -> 【 `*` 】 => 服务器不支持当前请求所需要的某个功能。
* **502** `Bad Gateway` -> 【 `*` 】 => 作为网关或者代理工作的服务器尝试执行请求时, 从上游服务器接收到无效的响应。
* **503** `Service Unavailable` -> 【 `*` 】 => 由于临时的服务器维护或者过载, 服务器当前无法处理请求。这个状况是临时的, 并且将在一段时间以后恢复。

状态码的完全列表参见[这里](http://www.w3.org/Protocols/rfc2616/rfc2616-sec10.html)。

## 错误处理（Error handling）

1. 不要发生了错误但给2xx响应, 客户端可能会缓存成功的http请求；
2. 正确设置http状态码, 不要自定义；
3. Response body 提供 1) 错误的代码（日志/问题追查）；2) 错误的描述文本（展示给用户）。
4. 如果状态码是 4xx, 就应该向用户返回出错信息。一般来说, 返回的信息中将 error 作为键名, 出错信息作为键值即可。

业务类异常必须提供2种信息: 

* 如果抛出该类异常, HTTP 响应状态码应该设成什么；
* 异常的文本描述；

在Controller层使用统一的异常拦截器: 

* 设置 HTTP 响应状态码: 对业务类异常, 用它指定的 HTTP code；对非业务类异常, 统一500；
* Response Body 的错误码: 异常类名
* Response Body 的错误描述: 对业务类异常, 用它指定的错误文本；对非业务类异常, 线上可以统一文案如“服务器端错误, 请稍后再试”, 开发或测试环境中用异常的 stacktrace, 服务器端提供该行为的开关。

## 返回结果

针对不同操作, 服务器向用户返回的结果应该符合以下规范。

* **Get** `/collection` => 返回资源对象的列表（数组）
* **Get** `/collection/resource` => 返回单个资源对象
* **POST** `/collection` => 返回新生成的资源对象
* **PUT** `/collection/resource` => 返回完整的资源对象
* **PATCH** `/collection/resource` => 返回完整的资源对象
* **DELETE** `/collection/resource` => 返回一个空文档

## Hypermedia API

RESTful API 最好做到 Hypermedia, 即返回结果中提供链接, 连向其他 API 方法, 使得用户不查文档, 也知道下一步应该做什么。 比如, 当用户向 [api.github.com](http://api.github.com/) 的根目录发出请求, 会得到这样一个文档。

``` json
{
  "link": {
    "rel": "collection https://www.example.com/zoos",
    "href": "https://api.example.com/zoos",
    "title": "List of zoos",
    "type": "application/vnd.yourformat+json"
  }
}
```

上面代码表示, 文档中有一个 link 属性, 用户读取这个属性就知道下一步该调用什么 API 了。rel 表示这个 API 与当前网址的关系（collection 关系, 并给出该 collection 的网址）, href 表示 API 的路径, title 表示 API 的标题, type 表示返回类型。

Hypermedia API 的设计被称为 [HATEOAS](http://en.wikipedia.org/wiki/HATEOAS)。Github 的 API 就是这种设计, 访问 [api.github.com](http://api.github.com/) 会得到一个所有可用 API 的网址列表。

``` json
{
  "current_user_url": "https://api.github.com/user",
  "authorizations_url": "https://api.github.com/authorizations"
}
```

从上面可以看到, 如果想获取当前用户的信息, 应该去访问 [api.github.com/user](http://api.github.com/user), 然后就得到了下面结果。

``` json
{
  "message": "Requires authentication",
  "documentation_url": "https://developer.github.com/v3"
}
```

上面代码表示, 服务器给出了提示信息, 以及文档的网址。

## 其他

* （1）API 的身份认证应该使用 [OAuth 2.0](http://www.ruanyifeng.com/blog/2014/05/oauth_2_0.html) 框架。
* （2）服务器返回的数据格式, 应该尽量使用 JSON, 避免使用 XML。
