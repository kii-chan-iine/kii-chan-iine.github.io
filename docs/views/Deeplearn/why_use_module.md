---
author: kii
title: why_use_module
categories: [深度学习]
tags: [deeplearn]
date: 2022-01-18 15:48:00
---

<Boxx changeTime="10000"/>

::: tip 前言
Pytorch提供了几个设计得非常棒的模块和类，比如 torch.nn，torch.optim，Dataset 以及 DataLoader，来帮助你设计和训练神经网络。为了充分利用他们来解决你的问题，你需要明白他们具体是做什么的。为了帮助大家理解这些内容，我们首先基于MNIST数据集，不用以上提到的模块和类来训练一个基础的神经网络，只用到基本的PyTorch tensor 函数。然后我们会逐渐地使用来自torch.nn，torch.optim，Dataset 以及 DataLoader的功能。展示每个模块具体的功能，他的运作过程。这样来使得代码逐渐简洁和灵活。
:::
<!-- more -->

# 1、设置[MNIST](https://so.csdn.net/so/search?q=MNIST&spm=1001.2101.3001.7020)数据

使用经典的 `MNIST` 数据集，该数据集由手写数字（0-9）的黑白图像组成。

使用 `pathlib` 来处理路径（Python3标准库的一部分），用 `requests` 下载数据。

```
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
1234567891011121314
```

该数据集的格式为`NumPy array`，使用 `pickle` 存储。

```
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
12345
```

每个图片大小为28x28，并存储为长度为784（=28x28）的扁平行。

查看其中的一个图片：

```
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)
12345
```

输出为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190828233837271.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NwcmluZ18yNA==,size_16,color_FFFFFF,t_70)

```
(50000, 784)
1
```

PyTorch使用 `tensor` 而不是 NumPy `array`，所以我们需要将其转换。

```
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())
12345678910
```

输出：

```
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]]) tensor([5, 0, 4,  ..., 8, 4, 8])
torch.Size([50000, 784])
tensor(0) tensor(9)
123456789
```

# 2、从头构建神经网络（不使用 `torch.nn`）

首先只使用PyTorch `tensor` 操作创建一个模型。

```
#initializing the weights with Xavier initialisation (by multiplying with 1/sqrt(n)).

import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
    
loss_func = nll

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]

preds = model(xb)  # predictions

print(preds[0], preds.shape)
print(loss_func(preds, yb))
print(accuracy(preds, yb))
123456789101112131415161718192021222324252627282930313233
```

输出：

```
tensor([-1.7022, -3.0342, -2.4138, -2.6452, -2.7764, -2.0892, -2.2945, -2.5480,
        -2.3732, -1.8915], grad_fn=<SelectBackward>) torch.Size([64, 10])

tensor(2.3783, grad_fn=<NegBackward>)
tensor(0.0938)
12345
```

现在我们可以进行训练。对于每次迭代，将会做以下几件事：

- 选择一批数据（mini-batch）
- 使用模型进行预测
- 计算损失
- `loss.backward()` 更新模型的梯度，即权重和偏置

```
from IPython.core.debugger import set_trace

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
       #set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))
1234567891011121314151617181920212223
```

输出：

```
tensor(0.0806, grad_fn=<NegBackward>) tensor(1.)
1
```

# 3、使用 `torch.nn.functional`

如果使用了负对数似然损失函数和 `log softnax` 激活函数，那么Pytorch提供的`F.cross_entropy` 结合了两者。所以我们甚至可以从我们的模型中移除激活函数。

```
import torch.nn.functional as F

loss_func = F.cross_entropy

def model(xb):
    return xb @ weights + bias
123456
```

注意，在 `model` 函数中我们不再需要调用 `log_softmax`。让我们确认一下，损失和精确度与前边计算的一样：

```
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
1
```

输出：

```
tensor(0.0806, grad_fn=<NllLossBackward>) tensor(1.)
1
```

# 4、使用 `nn.Module` 重构

继承 `nn.Module`（它本身是一个类并且能够跟踪状态）建立子类，并实例化模型：

```
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias
        
model = Mnist_Logistic()

print(loss_func(model(xb), yb))
1234567891011121314
```

输出：

```
tensor(2.3558, grad_fn=<NllLossBackward>)
1
```

将训练循环包装到一个 `fit` 函数中，以便我们以后运行。

```
def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()

print(loss_func(model(xb), yb))
12345678910111213141516171819
```

输出：

```
tensor(0.0826, grad_fn=<NllLossBackward>)
1
```

# 5、使用 `nn.Linear` 重构

使用PyTorch 的 `nn.Linear` 类建立一个线性层，以替代手动定义和初始化 `self.weights` 和 `self.bias`、计算 `xb @ self.weights + self.bias` 等工作。

```
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

model = Mnist_Logistic()
print(loss_func(model(xb), yb))
12345678910
```

输出：

```
tensor(2.3156, grad_fn=<NllLossBackward>)
1
```

我们仍然能够像之前那样使用 `fit` 方法

```
fit()

print(loss_func(model(xb), yb))
123
```

输出：

```
tensor(0.0809, grad_fn=<NllLossBackward>)
1
```

# 6、使用 `optim` 重构

定义一个函数来创建模型和优化器，以便将来可以重用它。

```
from torch import optim

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
1234567891011121314151617181920212223
```

输出：

```
tensor(2.2861, grad_fn=<NllLossBackward>)
tensor(0.0815, grad_fn=<NllLossBackward>)
12
```

# 7、使用 `Dataset` 重构

```
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)
model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
12345678910111213141516
```

输出：

```
tensor(0.0800, grad_fn=<NllLossBackward>)
1
```

# 8、使用 `DataLoader` 重构

```
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
123456789101112131415
```

输出：

```
tensor(0.0821, grad_fn=<NllLossBackward>)
1
```

# 9、增加验证

```
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)
12345
```

我们将在每个epoch结束时计算和打印验证损失。（注意，我们总是在训练之前调用`model.train()`，在推理之前调用 `model.eval()`，因为这些由诸如 `nn.BatchNorm2d` 和`nn.Dropout` 等层使用，以确保这些不同阶段的适当行为。）

```
model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
1234567891011121314151617
```

输出：

```
0 tensor(0.2981)
1 tensor(0.3033)
12
```

# 10、创建 `fit()` 和 `get_data()`

`loss_batch` 函数计算每个批次的损失。

```
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
123456789
```

`fit` 运行必要的操作来训练我们的模型并计算每个epoch的训练和验证损失。

```
import numpy as np

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
12345678910111213141516
```

`get_data` 为训练集合验证集返回 `DataLoader`。

```
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )
12345
```

现在，我们获取 `DataLoader` 和拟合模型的整个过程可以在3行代码中运行：

```
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
123
```

输出：

```
0 0.3055081913471222
1 0.31777948439121245
12
```

# 11、总结

我们现在有一个通用数据流水线和训练循环，你可以使用它来训练多种类型PyTorch模型。 各部分的功能总结如下：

- ```
  torch.nn
  ```

  - `Module`：创建一个可调用的对象，其行为类似于一个函数，但也可以包含状态（例如神经网络层权重）。 它知道它包含哪些参数，并且可以将所有梯度归零，循环遍历它们更新权重等。
  - `Parameter`：`tensor` 的包装器（wrapper），它告诉 `Module` 它具有在反向传播期间需要更新的权重。 只更新具有 `requires_grad` 属性的 `tensor`。
  - `functional`：一个模块（通常按惯例导入到F命名空间中），它包含激活函数，损失函数等，以及非状态（non-stateful）版本的层，如卷积层和线性层。

- `torch.optim`：包含 `SGD` 等优化器，可在后向传播步骤中更新 `Parameter` 的权重。

- `Dataset`：带有 `__len__` 和 `__getitem__` 的对象的抽象接口，包括 `PyTorch` 提供的类，如`TensorDataset`。

- `DataLoader`：获取任何 `Dataset` 并创建一个返回批量数据的迭代器。







---



# Python的几个知识



这篇文章希望把 __*init*_*, _*_*call__, __new*__ 3个魔术方法一起讲明白，以及他们的正确使用方式和应用场景。

python中所有对象都有一个从创建，被使用，再到消亡的过程，不同的阶段由不同的方法负责执行。

定义一个类时，大家用得最多的就是 `__init__` 方法，而 `__new__` 和 `__call__` 使用得比较少

本文代码都是基于 Python3 来讨论

### __init__方法

`__init__`方法负责对象的初始化，系统执行该方法前，其实该对象已经存在了，要不然初始化什么东西呢？先看例子：

```python
# class A(object): 
class A:
    def __init__(self):
        print("__init__ ")
        super(A, self).__init__()

    def __new__(cls):
        print("__new__ ")
        return super(A, cls).__new__(cls)

    def __call__(self):  # 可以定义任意参数
        print('__call__ ')

A()
```

输出

```text
__new__
__init__
```

从输出结果来看， `__new__`方法先被调用，返回一个实例对象，接着 `__init__` 被调用。 `__call__`方法并没有被调用，这个我们放到最后说，先来说说前面两个方法，稍微改写成：

```python
def __init__(self):
    print("__init__ ")
    print(self)
    super(A, self).__init__()

def __new__(cls):
    print("__new__ ")
    self = super(A, cls).__new__(cls)
    print(self)
    return self
```

输出：

```text
__new__ 
<__main__.A object at 0x1007a95f8>
__init__ 
<__main__.A object at 0x1007a95f8>
```

从输出结果来看，`__new__` 方法的返回值就是类的实例对象，这个实例对象会传递给 `__init__` 方法中定义的 self 参数，以便实例对象可以被正确地初始化。

如果 `__new__` 方法不返回值（或者说返回 None）那么 `__init__` 将不会得到调用，这个也说得通，因为实例对象都没创建出来，调用 init 也没什么意义，此外，Python 还规定，`__init__` 只能返回 None 值，否则报错，这个留给大家去试。

`__init__`方法可以用来做一些初始化工作，比如给实例对象的状态进行初始化：

```python
def __init__(self, a, b):
    self.a = a
    self.b = b
    super(A, self).__init__()
```

另外，`__init__`方法中除了self之外定义的参数，都将与`__new__`方法中除cls参数之外的参数是必须保持一致或者等效。

```text
class B:
    def __init__(self, *args, **kwargs):
        print("init", args, kwargs)

    def __new__(cls, *args, **kwargs):
        print("new", args, kwargs)
        return super().__new__(cls)

B(1, 2, 3)

# 输出

new (1, 2, 3) {}
init (1, 2, 3) {}
```

### __new__ 方法

一般我们不会去重写该方法，除非你确切知道怎么做，什么时候你会去关心它呢，它作为构造函数用于创建对象，是一个[工厂函数](https://www.zhihu.com/search?q=工厂函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1842466508})，专用于生产实例对象。著名的设计模式之一，单例模式，就可以通过此方法来实现。在自己写框架级的代码时，可能你会用到它，我们也可以从开源代码中找到它的应用场景，例如微型 Web 框架 Bootle 就用到了。

```python
class BaseController(object):
    _singleton = None
    def __new__(cls, *a, **k):
        if not cls._singleton:
            cls._singleton = object.__new__(cls, *a, **k)
        return cls._singleton
```

这段代码出自 [https://github.com/bottlepy/bottle/blob/release-0.6/bottle.py](https://link.zhihu.com/?target=https%3A//github.com/bottlepy/bottle/blob/release-0.6/bottle.py)

这就是通过 `__new__` 方法是实现单例模式的的一种方式，如果实例对象存在了就直接返回该实例即可，如果还没有，那么就先创建一个实例，再返回。当然，实现单例模式的方法不只一种，Python之禅有说：

> There should be one-- and preferably only one --obvious way to do it.
> 用一种方法，最好是只有一种方法来做一件事

### __call__ 方法

关于 `__call__` 方法，不得不先提到一个概念，就是*可调用对象（callable）*，我们平时自定义的函数、内置函数和类都属于可调用对象，但凡是可以把一对括号`()`应用到某个对象身上都可称之为可调用对象，判断对象是否为可调用对象可以用函数 `callable`

如果在类中实现了 `__call__` 方法，那么[实例对象](https://www.zhihu.com/search?q=实例对象&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1842466508})也将成为一个可调用对象，我们回到最开始的那个例子：

```python
a = A()
print(callable(a))  # True
```

`a`是实例对象，同时还是可调用对象，那么我就可以像函数一样调用它。试试：

```text
a()  # __call__
```

很神奇不是，实例对象也可以像函数一样作为可调用对象来用，那么，这个特点在什么场景用得上呢？这个要结合类的特性来说，类可以记录数据（属性），而函数不行（闭包某种意义上也可行），利用这种特性可以实现基于类的装饰器，在类里面记录状态，比如，下面这个例子用于记录函数被调用的次数：

```python
class Counter:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@Counter
def foo():
    pass

for i in range(10):
    foo()

print(foo.count)  # 10
```

在 Bottle 中也有 call 方法 的使用案例，另外，[stackoverflow](https://link.zhihu.com/?target=https%3A//stackoverflow.com/questions/5824881/python-call-special-method-practical-example) 也有一些关于 call 的实践例子，推荐看看，如果你的项目中，需要更加抽象化、[框架代码](https://www.zhihu.com/search?q=框架代码&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1842466508})，那么这些高级特性往往能发挥出它作用。





---

作为典型的**面向对象**的语言，Python中 **类** 的**定义**和**使用**是不可或缺的一部分知识。对于有面向对象的经验、对**类**和**实例**的概念已经足够清晰的人，学习Python的这套定义规则不过是语法的迁移。但对新手小白而言，要想相对快速地跨过`__init__`这道坎，还是结合一个简单例子来说比较好。

以创建一个“学生”**类**为例，最简单的语句是

```python3
class Student():
    pass
```

当然，这样定义的类没有包含任何预定义的数据和功能。除了名字叫Student以外，它没有体现出任何“学生”应该具有的特点。但它是可用的，上述代码运行过后，通过类似

```python3
stu_1 = Student()
```

这样的语句，我们可以创建一个“学生”**实例**，即一个具体的“学生”对象。

通过`class`语句定义的类`Student`，就好像一个**“模具”**，它可以定义作为一个学生应该具有的各种特点（这里暂未具体定义）；

而在类名`Student`后加圆括号`()`，组成一个**类似函数调用**的形式`Student()`，则执行了一个叫做**实例化**的过程，即根据定义好的规则，创建一个包含具体数据的学生对象（实例）。

为了使用创建的学生实例`stu_1`，我们可以继续为它添加或修改属性，比如添加一组成绩`scores` ，由三个整数组成：

```python3
stu_1.scores = [80, 90, 85]
```

但这样明显存在很多问题，一旦我们需要处理很多学生实例，比如`stu_2`, `stu_3`, `...`，这样不但带来书写上的麻烦，还容易带来错误，万一某些地方`scores`打错了，或者干脆忘记了，相应的学生实例就会缺少正确的`scores`属性。更重要的是，**这样的`scores`属性是暴露出来的，它的使用完全被外面控制着，没有起到“封装”的效果，既不方便也不靠谱**。

一个自然的解决方案是允许我们在执行实例化过程`Student()`时**传入一些参数**，以方便且正确地初始化/设置一些属性值，那么如何定义这种初始化行为呢？答案就是在类内部定义一个`__init__`函数。这时，`Student`的定义将变成（我们先用一段注释占着`__init__`函数内的位置）。

```python3
class Student():
    def __init__(self, score1, score2, score3):
        # 相关初始化语句
```

定义`__init__`后，执行实例化的过程须变成`Student(arg1, arg2, arg3)`，**新建的实例本身，连带其中的参数，会一并传给`__init__`函数自动并执行它**。所以**`__init__`函数的参数列表会在开头多出一项，它永远指代新建的那个实例对象**，Python语法要求这个参数**必须要有**，而名称随意，习惯上就命为`self`。

新建的实例传给`self`后，就可以在`__init__`函数内创建并初始化它的属性了，比如之前的`scores`，就可以写为

```python3
class Student():
    def __init__(self, score1, score2, score3):
        self.scores = [score1, score2, score3]
```

此时，若再要创建拥有具体成绩的学生实例，就只需

```python3
stu_1 = Student(80, 90, 85)
```

此时，`stu_1`将已经具有设置好的`scores`属性。并且由于`__init__`规定了实例化时的参数，若传入的参数数目不正确，解释器可以报错提醒。你也可以在其内部添加必要的参数检查，以避免错误或不合理的参数传递。

> 在其他方面，`__init__`就与普通函数无异了。考虑到新手可能对“函数”也掌握得很模糊，这里特别指出几个“无异”之处：
> **独立的命名空间**，也就是说**函数内新引入的变量均为局部变量**，新建的实例对象对这个函数来说也只是通过第一参数self从外部传入的，故无论设置还是使用它的属性都得利用`self.<属性名>`。如果将上面的初始化语句写成
> `scores = [score1, score2, score3]`（少了`self.`），
> 则只是在函数内部创建了一个[scores变量](https://www.zhihu.com/search?q=scores变量&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A767530541})，它在函数执行完就会消失，对新建的实例没有任何影响；
> 与此对应，**`self`的属性名和函数内其他名称(包括参数)也是不冲突的**，所以你可能经常见到类似这种写法，它正确而且规范。

```python3
class Student():
    def __init__(self, name, scores):
        # 这里增加了属性name，并将所有成绩作为一个参数scores传入
        # self.name是self的属性，单独的name是函数内的局部变量，参数也是局部变量
        self.name = name
        if len(scores) == 3:
            self.scores = scores
        else:
            self.scores = [0] * 3
```

> 从第二参数开始均可设置**变长参数**、**默认值**等，相应地将允许实例化过程`Student()`中灵活地传入需要数量的参数；
> 其他……

说到最后，`__init__`还是有个特殊之处，那就是它**不允许有返回值**。如果你的`__init__`过于复杂有可能要提前结束的话，使用**单独的`return`**就好，不要带返回值。
