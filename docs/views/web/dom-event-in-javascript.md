---
author: 星火燎原@vxhly
title: JavaScript 中的 DOM 事件
categories: [web]
tags: [JavaScript, jQuery]
date: 2016-10-28 16:59:57
---

<Boxx changeTime="10000"/>

::: tip 前言
这篇文章主要介绍了 JavaScript 中 DOM 0 级事件和 DOM 2 级事件的区别介绍。
:::
<!-- more -->

## DOM 0 级事件

### 写在标签里

``` html
<a href="#" id="main" onclick="fn();fn1();">
    <button type="button">this is button</button>
</a>
```

如上所示, 把 `onclick` 写在标签内, 都是 DOM 0 级事件, 方法 fn 和 fn1 依次执行

### 绑定事件

``` javascript
var ele = document.getElementById("main");
ele.onclick = function() {
    console.log("做第一件事");
};
```

这也是 DOM 0 级事件方法, 我们需要用 `XXX.onYYY=ZZZ` 这种形式来绑定事件, 当点击元素时, 会输出 `"做第一件事"` , 恩, 很理想。

接下来, 当我们希望在点击该元素的时候再做另一件事怎么办呢？下面这样: 

``` javascript
ele.onclick = function() {
    console.log("做第二件事");
};
```

这样肯定不行, 因为我们是希望点击时既输出 `“做一件事”` , 又要输出 `“做第二件事”` 。而这样的话后面的方法将前面的覆盖掉了, 因为 `XXX.onYYY=ZZZ` 只能给元素的某个事件类型(如例子中的 `click 事件` )绑定一个方法 `ZZZ` ；这个问题难不到我们, 既然这样, 那我们就把要做的事情全都放在 `ZZZ` 里不就行了么；

``` javascript
function fn1() {
    console.log("做第一件事");
}

function fn2() {
    console.log("做第二件事");
}
ele.onclick = function() {
    fn1();
    fn2();
};
```

## dom 2 级事件

### jQuery 绑定事件

``` javascript
$("#hash").click(function() {
    alert("jQuery 的 DOM 2 级点击第一次");
});
$("#hash").click(function() {
    alert("jQuery 的 DOM 2 级点击第二次");
});
```

::: tip 提示
通过 jQuery 方法来绑定事件, 不会覆盖之前方法体里写的事件。
:::

### JavaScript 绑定事件

``` javascript
btn.addEventListener(
    "click",
    function() {
        alert("原生 DOM 2 级第一次 click");
    },
    false
);
btn.addEventListener(
    "click",
    function() {
        alert("原生 DOM 2 级第二次click");
    },
    false
);
```

::: tip 提示
通过原生的 JavaScript 方法来绑定事件, 会覆盖之前方法体里写的事件。
:::

## DOM 0 级和 DOM 2 级共存

``` html
<a href="#" id="hash" onclick="fn();fn1();">
    <button type="button">返回上面进行开通</button>
</a>

<script type="text/javascript">
    function fn() {
        alert('ade');
    }

    function fn1() {
        alert('ade111');
    }
    var btn = $('#hash').get(0);
    btn.onclick = function() {
        alert('111');
    };
    $('#hash').click(function() {
        alert('jQuery 的 DOM 2 级点击第一次');
    });
    btn.addEventListener('click', function() {
        alert('原生 DOM 2 级第一次 click')
    }, false);
</script>
```

上面的例子有一个两个 DOM 0 级和两个 DOM 2 级绑定事件, js 里面写的 DOM 0 级会覆盖行内的 fn 和 fn1 方法, 但是 js 里面的 DOM 0 可以和 dom2 共存。
