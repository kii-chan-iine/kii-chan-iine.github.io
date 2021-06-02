---
author: 星火燎原@vxhly
title: ES6 学习笔记
categories: [web]
tags: [JavaScript, Node.js, ES6] 
date: 2017-11-17 21:35:51
---

<Boxx changeTime="10000"/>

::: tip 前言
本篇学习笔记纯粹记录学习 ES6 的过程以及一些坑点, 下面的例子没有先后之分, 也是很简单的例子, 想了解更多的直接戳 [ECMAScript 6 入门](http://es6.ruanyifeng.com/)。
:::
<!-- more -->

## 建立一个 ES6 专属项目

创建 package.json, 将以下的 json 复制到你的文件中, 然后运行 `npm insatll`

``` json
{
  "name": "es6-demo",
  "version": "1.0.0",
  "description": "学习 ES6",
  "main": "src/index.js",
  "devDependencies": {
    "babel-cli": "^6.26.0",
    "babel-eslint": "^8.0.2",
    "babel-preset-latest": "^6.24.1",
    "babel-preset-react": "^6.24.1",
    "babel-preset-stage-2": "^6.24.1",
    "eslint": "^4.10.0",
    "eslint-friendly-formatter": "^3.0.0",
    "eslint-loader": "^1.9.0",
    "eslint-plugin-promise": "^3.6.0",
    "eslint-plugin-standard": "^3.0.1",
    "eslint-config-standard": "^10.2.1",
    "eslint-plugin-html": "^3.2.2",
    "nodemon": "^1.12.1",
    "eslint-plugin-import": "^2.8.0",
    "eslint-plugin-node": "^5.2.1"
  },
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1",
    "dev": "nodemon ./src/index.js --exec babel-node",
    "babel": "babel-node ./src/index.js",
    "lint": "eslint --ext .js src"
  },
  "author": "",
  "license": "ISC"
}
```

可以看到上面的 package.json 是有 eslint 代码校验的, 所以还需在项目目录下新建 `.babelrc` 、 `.editorconfig` 、 `.eslintrc.js`
 `.babelrc`

``` json
{
  "presets": ["es2015"],
  "plugins": []
}
```

 `.editorconfig`

``` bash
root = true

[*]
charset = utf-8
indent_style = space
indent_size = 2
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
```

 `.eslintrc.js`

``` javascript
// http://eslint.org/docs/user-guide/configuring

module.exports = {
    root: true,
    parser: "babel-eslint",
    parserOptions: {
        sourceType: "module"
    },
    env: {
        browser: true
    },
    // https://github.com/standard/standard/blob/master/docs/RULES-en.md
    extends: "standard",
    // required to lint *.vue files
    plugins: ["html"],
    // add your custom rules here
    rules: {
        // allow paren-less arrow functions
        "arrow-parens": 0,
        // allow async-await
        "generator-star-spacing": 0,
        // allow debugger during development
        "no-debugger": process.env.NODE_ENV === "production" ? 2 : 0
    }
};
```

### 运行你的项目

``` bash
npm run dev
```

因为安装了 nodemon 该模块, 所以我们不需要每次保存都去运行 `node src/index.js` , nodemon 该模块会在保存后自动执行命令

## ES6 块级作用域

这一小节让我想到很多年前阅读到 Nicholas 的那本红宝书, 所讲的作用域以及 Var。也许至今, 还有一些人搞不明白变量提升的含义。如果你有幸在大学里写过 C 语言, 其实这就很好理解了。一个我们所说明的变量其实包含声明, 赋值两个部分, 是否有看过在头文件里声明, 或者在 C 函数体内先声明, 比如: 

``` javascript
int sayB() {
    int b;
    b = 1;
    return b;
}
sayB()
```

而 JS 中所提到的变量提升, 与其非常类似, 你在函数体内定义的变量, 无论在哪里定义, 都会提升到函数的顶部, 比如: 

``` javascript
function b() {
    console.log(v); // ?会报错吗？
    if (true) {
        var v = 1;
    }
}
```

其实不会, 执行顺序会变成

``` javascript
function b() {
    var v;
    console.log(v);
    if (true) {
        v = 1;
    }
}
```

好了, 我们不复习之前的知识了, 块级作用域顾名思义的其作用域在一个小块中, 其中跟两个关键字有关系 let 和 const , 于是乎你用它们定义的变量, 也就只能存在于块级作用域内了。

``` javascript
function b() {
    console.log(v);
    if (true) {
        let v = 1;
    }
}
```

这个时候再打印 v 是会报错的, 不像之前那样给你一个 undefined。

通过这一小节的阅读, 这应该是其中一个很大的知识点, 以及知晓了一些有趣的事情, 这和 let const 有关。let 和 const 定义的变量不会像 var 一样, 覆盖到全局。以前我们用 var 来定义变量时, 如果多写一个重名的, 只会是最后一个覆盖之前的。但是, 这里如果你用 let 或者 const 来定义时, 必然会报错。而且 const 定义的变量是不允许再赋值的, 但是它允许对于键的再赋值, 比如: 

``` javascript
const b = {
    a: 1
};

// b.a = 2 (YES)
```

可以说对于以前我印象中最深刻的是如果从数组里可以正确的获取到其数值, 需要借助闭包来完成, 而现在因为块级作用域的存在, 你完全可以不必要像 ES5 那样借助闭包了。

``` javascript
for (let i = 0; i < j; i++) {
    //i
}
```

这个时候可以很顺利的完成从[0...n] 的过程。

很明显, 块级作用域在某些时候帮助我们节约了很多事情, 不会像因为变量提升而带来的某些不可预知的奇怪问题。最后这一小节给出了最佳实践, 同名而言, 如果你定义的是一个预知的值（不再修改）那么你应该使用 const, 反之你应该使用 let, 尽量的避免使用 var, 当然你想定义一个全局变量除外。

## 字符串处理与函数

说到字符串我想 ES6 提供了最好的便利是"模板语法", 这一特性让我们深受鼓舞, 想着终于不是那么搓的来拼接变量了。当然, 这一版本的 JS 在字符串上也提供了很多起他的方法, 比起它来倒是要黯然神伤的多。比如我们几乎很少用到的 codePointAt, fromCodePoint, normalize, 像这几个一般在国际化的产品中会有的比较多, 一般正常情况下几乎很少会接触到。

以前我们使用 indexOf > -1 来判断一个字符是否在这个字符串中, 现在好了 ES6 提供了三个方法它们都会返回一个布尔值来告诉你是否存在: includes, startsWith, endsWith。从字面上几乎可以理解第一个方法在字符串中检查指定的文本, 有返回 true, 没有返回 false。第二个和第三个方法只是在你检查的方向上有所区别, 从首部开始还是从尾部开始而已。当然, 如果你还是想要文本的位置, 那么你可以继续使用 indexOf 或 lastIndexOf。哦, 对了字符串还提供了一个 repeat 方法, 用于返回一个将当前字符串重复几次之后的字符串, 看起来, 这玩意有些鸡肋, 但是你要是开发编辑器之类的应用, 它倒是非常有用。

说说"模板语法"吧。

``` javascript
const name = "icepy";
const message = `${name} call u` ;
console.log(message);
```

如果在以前我们想完成谁 call 你这样的动态消息, 都需要如此: 

``` javascript
var name = "icepy";
var message = name + "call u";
console.log(message);
```

粗看起来也没有什么不好, 但是如果这个消息会很长很长, 那么就要用+来拼接几行了。那时候看起来, 就比较搓了。如果有模板语法之后, 我们会怎样？

``` javascript
const name = "icepy";
const phone = "186xxxxxx";
const message = `
${name} call u
Phone Number: ${phone}`;
```

会不会舒服很多？当然这个特性带给你的不仅仅是这些, 很重要的一点是完全可以支持"运算"式的表达式, 你可以在完全可以支持"运算"式的表达式, 你可以在{}来运算一些结果。看到这里, 是不是感觉到和一些模板库很像？幸运的是, 从今之后再也不用引入一些基础模板库了。

在模板语法中我可能用的比较多的就是 tag, 有时候我会比较偷懒, 将字符串的转换写成函数, 然后不同类型的转换, 直接用 tag 就好, 比如: 

``` javascript
function message(literals, ...subs) {
    // do
}
const name = "icepy";
const phone = "186xxxxx";
const mg = message `${name} call u, Phone Number: ${phone}` ;
```

也许, 从这个小例子里并不能看出来什么, 你可以想象一下, 如果你想自己控制结果的输出, 在 message 函数中, 你可以转换成任意的结果, 完全取决于你写的这个 message 函数。比如在消息中, 文本, 会员, 等级, 之类的, 你都可以偷懒在这个 message 函数中完成, 结合某些判断来给出你想要的结果。

ES6 的函数有几个显著的特点, 方便了传入参数的处理, 又不失优雅。提供的箭头函数虽然有诸多限制, 但是在表达上减少了代码量还解决了我们在 ES5 中遇到的 `this` 丢失的问题。比较乐观的说, 这是我非常喜欢使用的 ES6 新特性之一。

在参数处理上, ES6 给大家提供了默认参数和不定参数的选择, 而默认参数在某个形式上对于解决默认值的问题, 终于不用再去处理很多的判断了, 想想 JavaScript 在"布尔"逻辑判断上的隐式转换, 比如"0" == false, 如果你的参数传入的是 0, 本意它就是一个真实存在的参数, 当在逻辑判断中时, 这个参数被隐式的转换了, 从而获取了默认值。

``` javascript
function logName(name = "icepy") {
    console.log(name);
}
logName(); // icepy
logName("wower"); // wower
```

在你细心的处理默认参数时, 有一点需要注意: 

``` javascript
function logName(name = "icepy", Qname = name) {
    // name
    // Qname
}
logName(); //icepy
```

你可以将第二个参数的默认值设置成第一个参数, 但是不能将第一个参数的默认值设置成第二个参数。

关于不定参数我想和 Function 一起说说, 先来看一段 Weex 执行业务 Code 的代码: 

``` javascript
/**
 * Call a new function body with some global objects.
 * @param  {object} globalObjects
 * @param  {string} code
 * @return {any}
 */
function callFunction(globalObjects, body) {
    const globalKeys = [];
    const globalValues = [];
    for (const key in globalObjects) {
        globalKeys.push(key);
        globalValues.push(globalObjects[key]);
    }
    globalKeys.push(body);

    const result = new Function(...globalKeys);
    return result(...globalValues);
}
```

这个巧妙的 callFunction 就包含了我们想看到的不定参数与 Function 构造函数, 不定参数使用... 来表示, 它用一个数组包含了其中的参数, 相信读到这里, 你会很快的联想到 ES5 中的 arguments, 你可以说和它类似, 又非常不同。

如果在函数定义时写了不定参数, 那么这个函数就只能接受唯一的一个, 如果函数定义时定义了多个参数呢？

``` javascript
function(name, ...profile) {
    // profile []
    // arguments
    // name
}
```

你可以通过这样的关系看到不定参数与 arguments 之间的关系, arguments 永远包含所有的参数, 这也意味着它包含了 name, 而 profile 并不包含 name。回到上面的那段 Weex 核心代码, 细心阅读一下, 相信你会有收获。

1. body 是 bundle.js 的业务代码, 将它加入到 globalKeys 最后一个位置。
2. 定义一个拥有不定参数 globalKeys 的 Function 来生成一个函数
3. Function 你可以知道, 它接受参数和函数体, 并且最后一个永远是函数体
4. 最后执行生成的函数, 将 globalValues 当不定参数传入 说起来... 还可以用来处理其他的事情, 来多实践一下吧。

说完参数, 我们再来说说平时我们可能大量使用的 () => {} 箭头函数, 它可能是我们非常乐见的"语法糖"之一, 不过在此之前有一些要说明的是: 

* 自身没有 this, super 等, 而这些需要外层的非箭头函数来决定
* 不能实例化
* 没有原型
* 不能改变 this, arguments 等

如果你写过 React + Redux, 相信你一定对箭头函数使用的非常频繁, 比如: 

``` javascript
const title = data => ({
    type: TITLE,
    payload: {
        data
    }
});

export const getDefaultData = () => dispatch => {};
```

是不是有种似曾相识的感觉？

有时候我们在写 React 组件时, 可能会用到立即调用模式: 

``` javascript
...{
    render() {
        return ( <
            div >
            (() => {
                    return ( < div > icepy < /div>)
                    })() <
                /div>
            );
        }
    }
```

不过, 说真的, 我常常配合着数组用, Why？表达的很简单: 

``` javascript
values.sort((a, b) => a - b);
```

## 解构和扩展对象的功能性

如果你正在开发使用数据驱动的应用, 那么这一小节的内容对你来讲, 就非常重要了, 欢迎来到解构的世界。

何为"解构"？当我们定义了很多对象和数组, 又必须从中提取有价值的数据时, 这种行为在 ES6 中被称为"解构", 解构分为两种: 对象解构和数组解构。

如果你定义过这样的对象: 

``` javascript
const obj = {
    name: "icepy",
    work: "alibaba"
};

const {
    name,
    work
} = obj;
```

这是一种非常常见的解构, 将值提取出来, 假设有一天 icepy 离职了, 没有工作呢？

``` javascript
const obj = {
    name: "icepy"
};
const {
    name,
    work
} = obj;
```

这个时候 work 会是一个 undefined, 但是我想给予他一份"新的工作", 骑摩托车旅行: 

``` javascript
const obj = {
    name: "icepy"
};
const {
    name,
    work = "骑摩托车旅行"
} = obj;
```

当然, 如果可以, 我们也可以重新定义一个名字, 比如 work 变成 travel: 

``` javascript
const obj = {
    name: "icepy"
};
const {
    name,
    work: travel = "骑摩托车旅行"
} = obj;
```

解构在函数的参数中也可以定义, 当然传参时需要传入的是对象了。如果你写过一些 Redux, 那么相信对这个会无比的熟悉: 

``` javascript
export const changeDate = ({
    mode,
    date
}) => dispatch => {
    dispatch(setTime({
        mode,
        date
    }));
};

// 在action里可以直接使用mode和date
```

如果你有多重嵌套的对象, 解构也可以很方便的提取出来你想要的数据: 

``` javascript
const obj = {
    home: {
        ad: "hunan 湘西",
        mz: "m"
    },
    name: "icepy"
};

const {
    home: {
        ad,
        mz = "h"
    },
    name
} = obj;
```

那么, 如果你这样定义过数组: 

``` javascript
const arr = [1, 2, 3];

let [f, s, t] = arr;

console.log(f); //1
console.log(s); //2
console.log(t); //3
```

当然如果你只想取第三个元素, 只需要用, 占位即可, 如果你想从嵌套的数组中提取数据, 其实和对象非常类似: 

``` javascript
const arr = [1, 2, 3];

let [, , t] = arr;

const arr = [1, [2], 3];

let [, [f]] = arr;

// f
```

曾经我们可能会有这样的一个需求, 有一个数组, 需要第五个元素之后的元素, 返回一个新数组: 

``` javascript
const arr = [1, 2, 3, 4, 5, 6, 7, 8, 9];
let [, , , , , ...f] = arr;
//f
```

而现在, 这样多简单。

当然在这一小节中, 主要的内容都是这些实践操作性的内容, 如果有时间, 不妨试试。

在《红宝书》里是这样来说对象初始值的: 

``` javascript
var name = "icepy";
var age = 18;
var obj = {
    name: name,
    age: age,
    say: function() {
        console.log(this.name);
    }
};
```

到了 ES6 中, 你可以使用一个更简写的方式: 

``` javascript
let name = "icepy";
let age = 18;
let obj = {
    name,
    age,
    say() {
        console.log(this.name);
    }
};
```

比之 ES5, 这拥有了极致的简化, 而且使用 say(){}定义的属性可以调用 super。

这一小节里讲的内容, 比较重要的是讲原型的增强, 其他的比如对对象的增强, 仅是添加了几个我们已经比较常用的方法了: 

``` javascript
Object.is;
Object.assign;
Object.setPrototypeOf;
```

可能某些情况下会出现这样的状况, A 类是 B 类的父类, A 类定义了 say 方法, B 类也定义了 say 方法, 在 B 类的 parentSay 中调用 A 类的 say 方法, 实例化 B 类, 直接调用 parentSay 方法, 这个时候我们就很尴尬了, 如果调用 this.say, 不一定能调用到 A 类定义的 say 方法。在 ES5 中给我们提供了一个 getPrototypeOf 方法来获取任意一个对象的原型, 根据 JS 的继承机制我们可以知道如果调用 Object.getPrototypeOf(this)正好就可以获取到父类的原型, 然后再调用 say 方法就可以了。

但, 这也有一个很尴尬的问题, 于是, ES6 给我们提供了 super 来代表父类, 这个时候, 我们直接使用 super.say()就很方便的区别调用 A 类的 say 方法和 B 类的 say 方法了。

有趣的是, ES6 还为我们提供了一个设置原型的方法"setPrototypeOf", 这个方法可以设置原型对象。

如果曾经你使用过类似 jQuery 的 extend 方法, 那么就应该对 assign 会有相当熟悉了感觉, 它和 extend 非常类型, 可以将一对象中的属性, 方法赋值给另一个对象。至于 is, 就是一个正确的布尔值比对, 大部分情况下与 === 行为一致。

## 什么是 Symbol 以及怎么使用

Symbol 其实解决了两个问题: 

* 唯一性
* 通过 Symbol 的属性来操作 JavaScript 内部的逻辑

这个特性的出现为我们定义程序有了普遍的意义, 它和原来的六个原始类型一样, 也是一个原始类型, 你可以调用 Symbol()来初始化。

``` javascript
let v = Symbol("icepy");
```

唯一性

我们知道在 JavaScript 的世界里, 唯一性是很难被描述的, 哪怕你用任意的加密函数生成的 Key 也有可能被撞库, Symbol 为我们带来了这样的唯一性: 

``` javascript
let name = Symbol("name");
let max = Symbol();
let obj = {
    [name]: "icepy",
    [max]: 29
};

// Symbol()
```

Symbol 可以接收一个参数, 这个参数用来 des, 主要用于程序调试时的跟踪, 当然你也可以不传入参数, 同样的我们可以通过 typeof 来判断是否为 Symbol 类型。

Symbol 属性中提供了 for 方法来处理全局共享的问题, 它可以从指定的 Symbol 注册表中来搜索到具体的内容, 反之你可以用 keyFor 来推断某个 Symbol 关联的键。

通过 Symbol 的属性来操作 JavaScript 内部的逻辑

以前我们比较难去操作 JavaScript 本身语言内部的逻辑, 最多也是在原型链上去定义或者修改某个方法的实现, Symbol 的属性中提供了很多去处理程序内部执行的逻辑, 如果你有兴趣, 可以仔细阅读一下 MDN 上的文档:  Symbol , 这里主要举一个例子来说明: 

在以前我们想判断一个数组, 可能会这样写: 

``` javascript
let f = [];
Object.prototype.toString.call(f);
// "[object Array]"
```

我们可以定义"[object Array]"来判断, 但是, 假设今天我们写的某些应用程序中, 我想将某些类定义出来类型, 通过类型的判断, 想解决一些问题, 最开始我们可能可以用 instance 来判断是哪个实例, 但, 这解决不了问题。

有了 Symbol 属性之后, 我们可以改变 Object.prototype.toString 的返回值。

``` javascript
function Pre() {}
Pro.prototype[Symbol.toStringTag] = "Pre";

const me = new Pre();

// Object.prototype.toString.call(me)
// [object Pre]
```

可能在这个粗躁的例子中, 很难看出来 Symbol 的作用。

但是, 在之前有说到可以通过 instance 来判断是否为某个类或函数的实例, 有一天我想给予它就算它是真实的实例, 我也想让它返回 false, 即: 不是一个实例。

``` javascript
function Pre() {}
Object.defineProperty(Pre, Symbol.hasInstance, {
    value: function() {
        return false;
    }
});
```

这就是 Symbol 的作用之一, 来改变语言程序内部的逻辑。

也许, 这些特性在你真实的应用中, 很少会用到, 不过, 看一看, 也是没有坏处的。

## 新的 Set 集合, Map 集合

曾经在 JavaScript 中就只有一种集合可用: 数组, 如果可以你将对象也称之为一种集合。对于老程序员来说, 数组可能会勾起很多大家的回忆, 毕竟我们用数组, 对象实现了诸如去重复, 缓存等等, 而今天, ES6 带来了 Set 集合和 Map 集合, 让我们更方便的去操作这些。

* [Set](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Set)
* [WeakSet](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/WeakSet)
* [Map](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Map)
* [WeakMap](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/WeakMap)

关于 API 有兴趣的朋友, 可以去翻一翻文档, 特别简单。 在使用 Set 或者 WeakSet 之前, 你应该需要了解一下 Set 或 WeakSet 的特点: 

它是一个不可重复的无序列表 值与值之间的比较是通过 Object.is 方法来进行的, 这也意味着 5 和"5"会占据两个空间 Set 和 WeakSet 区别在于, WeakSet 只提供了基础的方法, 并且是一个弱引用。 如果你有兴趣对一个数组去除重复的值: 

``` javascript
[...new Set([1, 1, 1, 2, 2, 3, 3, 2, 5])];
```

这就是 Set 集合给予的魅力。

关于 Map 和 WeakMap, 你应该也需要知晓其中的特点: 

* 它是一个由多个无序键值对组成的集合
* 键名可以是任意的数据类型
* WeakMap 的键名只能是对象, 与 WeakSet 类似, 它也是一个弱引用
* Map 比 WeakMap 多了几个方法, 但是基础的 set get 都是相同的

目前, 已知的 Map 的用途, 可能就是来存储私有数据, 存储对象引用（可以自动释放）, 大部分情况下, 你基本不会用到 Map。

``` javascript
let privateData = new WeakMap();

class Pre {
    constructor(name) {
        privateData.set(this, {
            name
        });
    }
    preName() {
        return privateData.get(this).name;
    }
}

export default Pre;
```

这些对于内存管理倒是很方便了, 不需要自己手动的去跟踪信息, 再手动的去设置 null 来断开引用。

## iterator 和 generator 的使用

这一小节的内容, 比较鼓舞的是终于可以在 JS 语言层面, 能看见 Iterator 和 Generator 了。说到迭代器, 也许你会有疑问, 可以预期的, 你能看到 Generator 的实现也是依赖迭代器。我所接触到的编程语言中, 最早让我理解这个特性的是 Python。迭代器是一种特殊的对象, 它的设计有专门的接口（描述）来完成我们常说的迭代（循环）过程。

每一个迭代器对象, 都具备 next 方法（当然它也具备一些比如 throw 方法）, 当你执行 next 方法时, 会返回一个（描述）对象, 这个对象中, 存在 value, done 属性, 你想要的值就是 value, 而 done 则是用来描述整个迭代过程是否结束, 可以想象, 当迭代过程结束之后, value 的值必然会是一个"空"（不同语言的描述不同）, JS 会给你一个 undefined。

目前我们所知道的数组, Set, Map 这三个集合是会存在迭代器, 而且它们都有内建的迭代器: entries, values, keys。不知道, 你在前几个小节中是否有印象 Symbol 中存在一个 iterator 属性, 如果你想知道, 不妨翻一翻之前分享的 Symbol。在这里, 我们可以通过 Symbol.iterator 来获取默认的迭代器（如果具备 Symbol.iterator 属性, 那么意味着它必然具备默认的迭代器）。

``` javascript
let sets = [1, 2, 3];
let iterator = sets[Symbol.iterator]();
console.log(iterator.next());
console.log(iterator.next());
console.log(iterator.next());
```

如果你不想调用三下迭代器, 你可以自己写一个小的循环, 比如: 

``` javascript
let res = iterator.next();
while (!res.done) {
    res = iterator.next();
    console.log(res);
}
```

所谓的 Generator 也就是一种可以返回迭代器的函数, 只不过它用*和 yield 来表示（描述过程）。

``` javascript
function* createIterator() {
    yield 1;
    yield 2;
}

let iterator = createIterator();

console.log(iterator.next());
console.log(iterator.next());
```

其实这很好理解, （语言）在背后帮我们对这个函数进行了包装, 每一个 yield 都会返回一个迭代器对象, 你想执行真正的逻辑, 或者你想获取值, 都需要通过这个迭代器对象来获取。

Generator 可以辅助我们完成很多复杂的任务, 而这些基础知识, 又与 iterator 息息相关, 举一个很简单的例子, 相信有很多朋友, 应该使用过 co 这个异步编程的库, 它就是用 Generator 来实现, 当然它的设计会比例子要复杂的多, 我们先来看一个 co 简单的用法: 

``` javascript
import co from "co";
co(function*() {
    var result = yield Promise.resolve(true);
    return result;
}).then(
    function(value) {
        console.log(value);
    },
    function(err) {
        console.error(err.stack);
    }
);
```

相应的, 我们来实现一个简化的版本: 

``` javascript
function co(task) {
    let _task = task()
    let resl = _task.next();
    while (!resl.done) {
        console.log(resl);
        resl = _task.next(resl.value);
    }
}

function sayName() {
    return {
        name: 'icepy'
    }
}

function assign * (f) {
    console.log(f)
    let g = yield sayName()
    return Object.assign(g, {
        age: f
    });
}

co(function*() {
    let info = yield* assign(18)
    console.log(info)
})
```

虽然, 这个例子中, 还不能很好的看出来"异步"的场景, 但是它很好的描述了 Generator 的使用方式。

从最开始的定义中, 已经和大家说明了, Generator 最终返回的依然是一个迭代器对象, 有了这个迭代器对象, 当你在处理某些场景时, 你可以通过 yield 来控制, 流程的走向。通过 co 函数, 我们可以看出, 先来执行 next 方法, 然后通过一个 while 循环, 来判断 done 是否为 true, 如果为 true 则代表整个迭代过程的结束, 于是, 这里就可以退出循环了。在 Generator 中的返回值, 可以通过给 next 方法传递参数的方式来实现, 也就是遇上第一个 yield 的返回值。

有逻辑, 自然会存在错误, 在 Generator 捕获错误的时机与执行 throw 方法的顺序有关系, 一个小例子: 

``` javascript
let hu = function*() {
    let g = yield 1;
    try {
        let j = yield 2;
    } catch (e) {
        console.log(e);
    }
    return 34;
};

let _it = hu();

console.log(_it.next());
console.log(_it.next());
console.log(_it.throw(new Error("hu error")));
```

当我能捕获到错误的时机是允许完第二次的 yield, 这个时候就可以 try 了。

Iterator 和 generator 给了我们很多启发, 在编程的维度上, 我能想到的就是去处理异步代码时为我们提供便捷的方式。当然迭代器, 这个方向上, 可以做的事情有很多, 如果你悉心去寻找, 相信, 很快能找到答案。

## 怎么写类, 以及 Promise 和如何封装处理异步

大部分面向对象的语言都支持类和类继承的特性 从 ECMA1-ECMA5 的版本都不支持类和类继承的特性, 于是开发者们通过原型, 构造函数等来模拟类和类继承特性, 这里不在复述, 如果你有兴趣的话, 可以阅读一下红包书（JavaScript 高级程序设计）中关于类, 类继承这两章。ECMA6 终于至少在语言层（依然是基于原型的语法糖）面看起来支持了类和类继承, 理解类的基本原理有助于理解 ES6 关于类的设计。

**基本的类声明**

``` javascript
class Human {
    constructor(name) {
        this.name = name;
    }

    sayName() {
        console.log(this.name);
    }
}

let man = new Human("icepy");
man.sayName();

console.log(typeof Human); // function
```

类的基本定义与一个构造函数非常的类似, 通过 typeof 也能看得出来 class 定义的也是一个 function, 在声明中, 我们定义的属性, 定义了原型上的方法 sayName。但是, 类声明的比之函数声明, 又多了一些不同的东西, 比如类声明是不允许提升的, 且内部运行的环境 100%的严格模式不允许强制降级。

类与函数一样, 也可以是表达式, 也可以将类当一个参数传给另外的函数或者类: 

``` javascript
let Human = class {
    constructor(name) {
        this.name = name;
    }
    sayName() {
        console.log(this.name);
    }
};

function fetchObj(HumanClass) {
    return new HumanClass("icepy");
}

let man = fetchObj(Human);
```

但是匿名的类表达式, 有一个不好的地方, 就是在调试的时候很难定位, 不过我们可以像函数一样给表达式加上一个 name: 

``` javascript
let Human = class Human {
    ...
}
```

**高阶知识**

我们可以为类中的属性创建访问器, 就像使用 Object.defineProperty 给对象的属性创建访问器一样的含义。

``` javascript
class Human {
    constructor(name) {
        this.name = name;
    }
    get name() {
        return this.manName;
    }

    set name(newValue) {
        this.manName = newValue;
    }
}
```

唯一需要注意的是关于"死循环"的问题。

类中的方法, 访问器, 都可以和对象一样, 可计算, 也就是使用[]这些方法, 属性的名字可以用变量来代替。

有趣的知识点, 是可以在类中声明一个 generator 方法, 这一点倒是可以和上一小节的内容结合起来, 不过一般情况下, 在类中声明一个 generator 方法的情况非常少见: 

``` javascript
class Human {
    constructor(name) {
            this.name = name;
        }

        -
        sayName() {

            yield 1;
        }
}
```

你如果有兴趣的话, 可以把 sayName 改造成可支持异步的。

关于"静态"也就是说不必实例化就可以调用的方法, 类语法也支持了这个: 

``` javascript
class Human {
    constructor(name) {
        this.name = name;
    }

    static sayWork(name) {
        return new Human(name);
    }
}

Human.sayWork();
```

有趣的是, 如果是在同一个类中两个静态方法, 其中一个方法想调用另外一个静态方法, 这个时候, 也可以使用 this。

``` javascript
class Human {
    constructor(name) {
        this.name = name;
    }
    static sayWork(name) {
        return new Human(this.whichName());
    }
    static whichName() {
        return "icepy";
    }
}
```

既然说到了类, 我们不可避免的要谈到继承, 在 ES6 中的类继承几乎简化到了一个关键字 extends, 而且从前面的可计算来说, 继承这个地方也是可以被计算的。

``` javascript
class Human {
    constructor(name) {
        this.name = name;
    }
}

class Icepy extends Human {
    constructor(name) {
        super(name);
    }

    sayName() {
        console.log(this.name);
    }
}
```

关于 super 使用的时机, 在前几章中有谈到, 类中如果有继承, 并且指定了 constructor, 那么就必须在使用 this 之前先调用 super 来初始化 this。这个继承不仅仅是原型上, 也包括静态成员, 而且就算父类与子类都有同样的方法名, 也不怕被覆盖, 可以用<http://this.xxx和http://super.xxx来分别调用, 在以往我们用原型模拟时, 就非常难界定这个方法到底调用来自哪里。>

如果你想知道类是否被实例化, 也可以通过 new.target 来确定, 在别的语言中有抽象类的概念, 也就是只定义描述不搞实现, 并且不能被实例化, 只能被继承。这个时候, new.target 就能排上用场了。

``` javascript
class Human {
    constructor() {
        if (new.target === Human) {
            throw new Error("抽象类不可以使用new");
        }
    }
    sayName() {}
}
```

最后一个想说一下的是关于 Symbol.species 属性, 这个属性用来返回函数的静态访问器属, 说得更直白一些, 你想用 instanceof 来判断一个对象属于哪个类, 用它就能改变这个逻辑的内部实现, 从而判定出可能不是原来访问器的类名。

``` javascript
class MyArrar extends Array {
    static get[Symbol.species]() {
        return Array;
    }
}
```

正常情况下 MyArray 返回的必然是 MyArray, 如果我想让它返回 Array, 那么这个时候就需要用到 Symbol.species 属性了。

类属于 ES6 的新特性, 它让我们可以更方便, 安全的定义类, 使用类, 而不是像 ES5 一样, 需要搞那么多复杂的东西来模拟这个特性。

在异步编程概念已经普及的今天, 我们依然要谈一谈它, 对于我们做前端代码的意义。当你通过 Ajax 请求数据使用回调函数来获取数据时, 这就是一种异步编程。

``` javascript
$.ajax({
    url: "xxx",
    success: function(data) {
        // data
    }
});
```

考虑到 JavaScript 属于单线程的特点, 异步对于这门语言就可见多么的重要。你可以试着想想, 当我们没有异步时, 我正准备提交一个信息, 界面就卡住了, 一直要等待信息返回结果才能有接下来的其他操作, 这很痛苦。在 JavaScript 的世界里, 异步是非常重要的一件事情。

Promise 正是想来处理这样的异步编程, 如果我们用 Promise 该如何处理这段 Ajax？

``` javascript
function fetch() {
    return new Promise(function(resolve, reject) {
        $.ajax({
            url: "xxx",
            success: function(data) {
                resolve(data);
            },
            error: function(error) {
                reject(error);
            }
        });
    });
}

fetch()
    .then(function(data) {})
    .catch(function(error) {});
```

这辅助我们避免了回调地狱式的恐惧, 更优雅的处理了回调。如果, 你有兴趣, 可以配合着 Promise 来学习【理论上的内容, 可以仔细阅读】。

**Promise 声明周期**

* 进行中（pending）
* 已经完成（fulfilled）
* 拒绝（rejected）

如同上面 Ajax 的例子, 我们可以很好的包装一个函数, 让 fetch 函数返回一个 Promise 对象。在 Promise 构造函数里, 可以传入一个 callback, 并且在这里完成主体逻辑的编写。唯一需要注意的是: Promise 对象只能通过 resolve 和 reject 函数来返回, 在外部使用 then 或 catch 来获取。如果你直接抛出一个错误（throw new Error('error')）, catch 也是可以正确的捕获到的。

**Promise 其他的方法**

Promise.all（当所有在可迭代参数中的 promises 已完成, 或者第一个传递的 promise（指 reject）失败时, 返回 promise。）

``` javascript
var p1 = Promise.resolve(3);
var p2 = 1337;
var p3 = new Promise((resolve, reject) => {
    setTimeout(resolve, 100, "foo");
});

Promise.all([p1, p2, p3]).then(values => {
    console.log(values); // [3, 1337, "foo"]
});
```

Promise.race（返回一个新的 promise, 参数 iterable 中只要有一个 promise 对象"完成（resolve）"或"失败（reject）", 新的 promise 就会立刻"完成（resolve）"或者"失败（reject）", 并获得之前那个 promise 对象的返回值或者错误原因。）

``` javascript
var p1 = new Promise(function(resolve, reject) {
    setTimeout(resolve, 500, "one");
});
var p2 = new Promise(function(resolve, reject) {
    setTimeout(resolve, 100, "two");
});

Promise.race([p1, p2]).then(function(value) {
    console.log(value); // "two"
    // 两个都完成,但 p2 更快
});
```

有趣的是如果你使用 ES6 的 class, 你是可以去派生 Promise 的, 就像我们想派生 Array 一样: 

``` javascript
class MePromise extends Promise {
    // 处理 ...
}
```

那么, 除了 Promise 还有没有更牛逼的方案？这是有的, 目前的异步编程已经从最初的 callback 进化到了如下的顺序。

> callback -> Promise -> generator -> async await

未来, 肯定属于 async await 了。

## 代理和反射

这两个 API 在实际的业务场景中几乎非常少的使用, 至于具体的场景可能还需要大家来挖掘了。看起来每一个 API 都有一些其他的 API 可模拟, 可使用, 它们具体的内涵就是在于拦截, 执行, 返回结果。

**Proxy（代理）**

从字面上来看, 这应该很好理解, 就像 nginx 一样, 将请求代理。我们来看一个非常小的例子: 

``` javascript
let handler = {
    get: function(target, name) {
        return name in target ? target[name] : 37;
    }
};

let p = new Proxy({}, handler);

p.a = 1;
p.b = undefined;

console.log(p.a, p.b);
// 1, undefined

console.log("c" in p, p.c);
```

通过 new Proxy 来创建一个自定义的行为, handler 做为捕获器可以捕获对象操作的行为。这一点上, 其实和 Object.defineProperty 来创建一个 get, set 方法非常像。由于 b 已经存在, 所以返回的是 undefined, 当我们打印 c 时, p.c 就会返回 37。

有兴趣的话, 可以仔细阅读: [Proxy](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Proxy)

**Reflect**

Reflect（反射）其实也和 Proxy 的作用非常类似, 它提供的是拦截操作 JavaScript 的方法, 唯一和代理不同的是, Reflect 提供的是静态方法, 无法使用 new 运算符。

获取一个属性: 

``` javascript
let obj = {
    a: 1
};

console.log(Reflect.get(obj, "a"));
```

设置一个属性: 

``` javascript
let obj = {};
Reflect.set(obj, "prop", "value"); // true
obj.prop; // "value"
```

对函数进行调用: 

``` javascript
Reflect.apply(Math.floor, undefined, [1.75]);
```

这和函数的 apply 唯一不同的是第一个参数必须是要调用的函数, 后两个参数与函数的 apply 保持一致。

更多的方法, 有兴趣的话可以阅读: [Reflect](https://developer.mozilla.org/zh-CN/docs/Web/JavaScript/Reference/Global_Objects/Reflect)

说实话, 这两个 API 的应用场景, 其实我也有些糊涂, 可能要经常使用, 比对, 才能发现比之对应有相同功能的方法的区别。

## 使用模块封装代码

不知不觉《深入理解 ES6》阅读笔记就写到了最后一篇, 完结之后可能会开启另外的一个系列, 分享自己的知识点, 让阅读到的人有一点点的收获, 以及自己的成长。最后一篇主要是来写一写用模块封装代码的事情, 回顾历史, 从最早的立即执行函数, 再到 require.js, 以及 commonjs, 今天我们面对的是语言标准给我们带来的模块化方案。如果说模块, 我们应该可以从字面的意思上来看, 这是一种可以自动运行在严格模式下并且没有办法退出的代码块。这种代码块在作用域上可以避免互相之间的污染, 以及更好的组织代码运行。

模块语法

ES6 定义了一对非常简单的关键字来描述模块的导入和导出, 在此之间, 又为了方便而定义了一些导出导入的方式。

``` javascript
import from
export
```

基本的导出, 我们可以用 export 来将函数暴露出去: 

``` javascript
export function a() {}
```

当然这样的方式, 也可以导出类, 对象, 变量等。

有了导出, 自然会有导入: 

``` javascript
// 假设a.js

export function a() {}

// b.js使用a.js

import {
    a
} from "a.js";

a();
```

这样, 也就构成了 ES6 的模块系统。

除此之外, 整个模块系统给了我们很多其他的方式来操作, 比如导出一个默认的函数: 

``` javascript
// a.js
export default function a() {}

// b.js

import a from "a.js";
```

如果可以, 我想给 a 起一个别名: 

``` javascript
// a.js

export function a() {}
import * as query from "a.js";
```

当然了, 我们也可以导出多个函数: 

``` javascript
export function a() {}
export function b() {}

import {
    a,
    b
} from "a.js";
```

至于其他的用法, 有兴趣的朋友, 可以阅读一下 export 和 import 。

在浏览器中不借助 webpack 这样的工具也可以使用模块系统, 只需要将 type=module, 这个模块需要注意的, 可能就是关于路径的问题了, 其他都和 export import 一样。
