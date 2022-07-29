---
author: kii
title: Pytorch学习札记
categories: [CV]
tags: [CV,DL]
date: 2022-07-26 23:13:30
---

<Boxx changeTime="10000"/>

::: tip 前言

记录pytorch学习的心得。

:::

<!-- more -->

# Dataloader

# Backbone

# Loss

PyTorch中，损失函数可以看做是网络的某一层而放到模型定义中，但在实际使用时更偏向于作为功能函数而放到前向传播过程中。

### 各种Loss

#### MSELoss

```python
#接上篇神经网络
out = net(input)
net.zero_grad()
out.backward(torch.randn(1, 10))
target = torch.randn(10) # a dummy target, for example
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)
#输出结果
#tensor(1.7306, grad_fn=<MseLossBackward0>)
#“因为input和target的是随机torch阵，所以loss结果不固定”
```



#### Softmax loss+交叉熵

![](https://imagerk.oss-cn-beijing.aliyuncs.com/img/2022-07-28-23-07-07-image.png)

```python
import torch
import torch.nn as nn
x_input=torch.tensor([[4,2,0.2],[3,7,1],[6,3,5]])
print('x_input:\n',x_input)
y_target=torch.tensor([1,2,0])#设置输出具体值 print('y_target\n',y_target)

#softmax，此时可以看到每一行加到一起结果都是1
soft_output=nn.Softmax(dim=1)(x_input)
print('softmax:\n',soft_output)
#取对数
log_output=torch.log(soft_output)
print('softmax取对数:\n',log_output)



# Step1 LogSoftmaxloss
#nn.LogSoftmaxloss(负对数似然损失)
logsoftmax_func=nn.LogSoftmax(dim=1)
logsoftmax_output=logsoftmax_func(x_input)
print('logsoftmax_output:\n',logsoftmax_output)
# Step2 NLLoss
#pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
nllloss_func=nn.NLLLoss() # reduction='sum',注意交叉熵是使用的mean
nlloss_output=nllloss_func(logsoftmax_output,y_target)
print('nlloss_output:\n',nlloss_output)

#直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
crossentropyloss=nn.CrossEntropyLoss()
crossentropyloss_output=crossentropyloss(x_input,y_target)
print('crossentropyloss_output:\n',crossentropyloss_output)
```

#### FocalLoss

先放出Focal loss的公式：

$$FL(pt)=−(1−pt)γlog(pt)$$

其中：

pt={pif y=11−potherwise

γ 为常数，当其为0时，FL就和普通的交叉熵损失函数一致了。 γ 不同取值，FL曲线如下：

![img](https://pic2.zhimg.com/80/v2-fb5a4c4f9586c5351075888d8ce135e9_720w.jpg)

![image-20220729000021125](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220729000021125.png)

二分类及多分类实现

```python
class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
```



```python
class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 1, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.elipson = 0.000001
    
    def forward(self, logits, labels):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        if labels.dim() > 2:
            labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
            labels = labels.transpose(1, 2)
            labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
        if logits.dim() > 3:
            logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
            logits = logits.transpose(2, 3)
            logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
        assert(logits.size(0) == labels.size(0))
        assert(logits.size(2) == labels.size(1))
        batch_size = logits.size(0)
        labels_length = logits.size(1)
        seq_length = logits.size(2)
        # transpose labels into labels onehot
        new_label = labels.unsqueeze(1)
        label_onehot = torch.zeros([batch_size, labels_length, seq_length]).scatter_(1, new_label, 1)
        # calculate log
        log_p = F.log_softmax(logits)
        pt = label_onehot * log_p
        sub_pt = 1 - pt
        fl = -self.alpha * (sub_pt)**self.gamma * log_p
        if self.size_average:
            return fl.mean()
        else:
            return fl.sum()
```



### 优化器和损失函数

#### 损失函数Loss



```python
fc=torch.nn.Linear(n_features,1)
criterion=torch.nn.BCEWithLogitsLoss() # Loss 
optimizer=torch.optim.Adam(fc.parameters()) # optimizer 

for step in range(n_steps):
    if step:
        optimizer.zero_grad() # optimizer 
        loss.backward() # Loss  
        optimizer.step() # optimizer 

    pred=fc(x)
    loss=criterion(pred[train_start:train_end],y[train_start:train_end])
```

下面：当我们在使用torch.tensor()对 自变量x进行构建的时候，曾经对参数`requires_grad=True`进行设置为True，意思就相当于说，这个Tensor类保留了一个盛放自己导数值的单位tensor。即，这个x有两个单位tensor：

- **一个是主体的单位tensor**：用于存放的是实实在在的这个自变量x的数值
- **另外一个是辅助的单位tensor**：用于存放的是这个自变量x被求导之后的导函数值--x.gard
  当我们通过一些正常手段（例如使用Tensor类的函数）进行操作的时候都是操作在主体那个单位tensor上，剩下那个辅助的单位tensor，我们可以通过`x.grad`来访问那个辅助的单位tensor。

如果自变量x不经过导数计算，`x.grad`是不会有数据的，你访问得到是None。  
所以`f.backward()`所做的就是把导数值存放在`x.grad`里面。

```python
import torch

def func(x):
    return x**3

x=torch.tensor([4],requires_grad=True,dtype=torch.float32) # 构建Tensor类
f=func(x) #把 Tensor 类作为参数传入,类比loss函数
f.backward() #求导数
print(x.grad)
```

需要注意的是`x.grad`不会自动清零的，他只会不断把新得到的数值累加到旧的数值上面，这就需要我们利用`optimizer.zero_grad()`来给他清零。

#### 反向传播

在神经网络的学习中，寻找最优参数（权重和偏置）时，要寻找使损失函数的值尽可能小的参数。为了找到使损失函数的值尽可能小的地方，需要计算参数的导数（确切地讲是梯度），然后以这个导数为指引，逐步更新参数的值。数值微分可以计算神经网络的权重参数的梯度（严格来说，是损失函数关于权重参数的梯度），但是计算上比较费时间。误差反向传播法则是一个能够高效计算权重参数的梯度的方法。

为了实现反向传播损失，我们所有需要做的事情仅仅是使用 loss.backward()。你需要清空现存的梯度```net.zero_grad()```，要不然梯度将会和现存的梯度累计到一起。

> 如果 optimizer=optim.Optimizer(net.parameters())， optimizer.zero_grad()和net.zero_grad()是等价的,原因在于zero_grad()函数的定义：
>
> ```python
> def zero_grad(self):
>     """Sets gradients of all model parameters to zero."""
>     for p in self.parameters():
>         if p.grad is not None:
>             p.grad.data.zero_()
> ```

#### 优化器

- 优化器

利用反向传播，优化器应运而生。优化器可以更新参数即网络中的权重，进行模型优化、加速收敛。

常用的优化器算法SGD, Nesterov-SGD, Adam,RMSProp等。算法在算法包torch.optim。

```python
import torch
import torch.optim

def func(x):
    return x**5 #x的5次方

x=torch.tensor([2],requires_grad=True,dtype=torch.float32)
optimizer=torch.optim.Adam([x,])

f=func(x)
optimizer.zero_grad() # 梯度清零
f.backward(retain_graph=True) #第一次反向传播，链式求导，retain_graph 为了不让计算图释放
optimizer.step() # 迭代更新
print(x.grad)

optimizer.zero_grad() # 梯度清零==》   x.grad=None
f.backward() #第二次反向传播
optimizer.step() #迭代更新
print(x.grad)
---
tensor([80.])
tensor([79.8401])
```

#### （1）zero_grad() 到底干了什么？

我们前文提到过Tensor类的grad数值只会在原有基础上增加，自己是不会覆盖的，所以我们每计算一次就应该清零。实际上，```zero_grad()```做的就是一个清理工作。  
如果我们把`optim.zero_grad()`换成`x.grad=None`，会发现得到一样的输出结果。

#### （2）step()函数到底干了什么？

首先，我们必须明白优化器是干什么的？现在我们有一个函数，我们现在需要到达一个谷点，也就是我们需要知道当函数在谷点的时候，自变量x到底等于多少，但是我们只能一个个点去尝试，所以我们必须要让梯度下降，也就是不断迭代x，每一次更换一次x的值，来促使梯度的值不断变小。

> 这个决定了迭代的方向

我们在运算step()的时是参考了`x.grad`，也就是`f.backward()`的结果，这也就是为啥backward会先执行，再执行step。

我们在最开始初始化的时候就已经把自变量x托管给optim类了，无论是我们自己编写的函数，还是已经封装好的损失函数。也就是说我们的optim类只需要损失函数的自变量即可。  
我们自己的函数使用优化器：

```python
def func(x):
    return x**5 #x的5次方

x=torch.tensor([2],requires_grad=True,dtype=torch.float32) 
criterion=func(x) #我们自己写的loss???我的理解，是否正确？
optimizer=torch.optim.Adam([x,])
```

封装的损失函数使用优化器：

```python
fc=torch.nn.Linear(n_features,1)
criterion=torch.nn.BCEWithLogitsLoss() # Loss 类
optimizer=torch.optim.Adam(fc.parameters()) # optimizer 类
```

我们已经知道`weights,bias=fc.parameters()`，对于损失函数而言，weights和bias就是他的自变量，所以这里用weights和bias来初始化优化器就能说得通了。

所有的优化都是围绕损失函数来转的，我们想要损失降到最小，我们想要损失函数最小的时候的那个自变量的值，就是我们需要的权值。整个训练的过程就是在求权值的过程。

<img title="" src="https://img-blog.csdnimg.cn/20200205210102349.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9qaW50dXpoZW5nLmJsb2cuY3Nkbi5uZXQ=,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" width="528" data-align="center">

## optimizer.step()和[scheduler](https://so.csdn.net/so/search?q=scheduler&spm=1001.2101.3001.7020).step()的区别

optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面,但是不绝对，可以根据具体的需求来做。只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整。通常我们有

```python
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
model = net.train(model, loss_function, optimizer, scheduler, num_epochs = 100)
```

在scheduler的step_size表示scheduler.step()每调用step_size次，对应的学习率就会按照策略调整一次。所以如果scheduler.step()是放在mini-batch里面，那么step_size指的是经过这么多次迭代，学习率改变一次。



## Resnet example

```python

# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential( #其实这个Sequential就是相当于把里面的东西打包了，将网络层和激活函数结合起来。
                nn.Conv1d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes, self.expansion*planes,kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )
            #这块还是不是很理解，这个的输出是？

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):#block: block method
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=3,stride=1, padding=1, bias=False)
        # conv1的输入为图像是3层，而输出是64层，核函数的大小不是7？ stride是2
        # max pool 3x3的吧 我看你写的还是3x3
        # 这里缺一个pool层
        self.bn1 = nn.BatchNorm1d(64)
        # self.maxpool=nn.MaxPool1d(kernel_size=3, stride=2, padding=1)#EC
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.fc1 = nn.Linear(in_features= 512*block.expansion*16, out_features=1) #自己查看flatten的输出，需要改
        # self.fc2 = nn.Linear(in_features= 512, out_features=1)
        # self.linear = nn.Linear(1233, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)#对于每一layer，只有第一个block的stride是2，后面的都是1，该句定义了这个，如[2,1,1]
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride)
                )
            self.in_planes = planes * block.expansion #每层的一个block（如bottleneck）结束后是planes * block.expansion层的输入
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out=self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print('tt',out.size())
        out = F.avg_pool1d(out, 4)#input:（batch_size,channels,width）,假设kernel_size=2,则每俩列相加求平均，stride默认和kernel_size保持一致，越界则丢弃
        # print('ff',out.size())
        out = out.view(out.size(0), -1) #falttern,打平，channel*feature num
        #在pytorch中view函数的作用为重构张量的维度，相当于numpy中resize（）的功能，但是用法可能不太一样。
        # -1 是指根据前面的数自动调整维度。这里是指根据out.size(0)自动调整维度
        # print('kk',out.size())
        out=self.fc1(out)
        # out=self.fc2(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def ResNetec():
    return ResNet(BasicBlock, [1,2,2,1])
# net = ResNet18()
# print(net)
'''
##############################################################
#
##############################################################
'''
fa_num=0
# %所有的一起预测
# 0% C12:0# 1% C14:0# 2% C15:0# 3% C16:4# 4% C16:1# 5% C16:0# 6% C18:3# 7% C18:2# 8% C18:1# 9% C18:0
time_start=time.time()

#%% Train_test_split
X_train,X_test, y_train, y_test =train_test_split(data,fa[:,fa_num],test_size=0.3, random_state=10)
# 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，
# 其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
print('\n')
print(X_train.shape)
print(X_test.shape)
[m,n]=X_train.shape
train_bc_size=76
n_epochs = 2000
'''
Batch Size从小到大的变化对网络影响
1、没有Batch Size，梯度准确，只适用于小样本数据库
2、Batch Size=1，梯度变来变去，非常不准确，网络很难收敛。
3、Batch Size增大，梯度变准确，
4、Batch Size增大，梯度已经非常准确，再增加Batch Size也没有用
EC：数据量小的话，尽量保证每个batch大小一样吧，我感觉会影响精度。
————————————————
'''


# To tensor
X_train = torch.unsqueeze(torch.from_numpy(np.array(X_train)), dim=1)
X_test = torch.unsqueeze(torch.from_numpy(np.array(X_test)), dim=1)
y_train = torch.unsqueeze(torch.from_numpy(np.array(y_train)),dim=1)
y_test = torch.unsqueeze(torch.from_numpy(np.array(y_test)),dim=1)
train_size = X_train.size(0)
test_size = X_test.size(0)

# TensorDataset
train_dataset = TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_bc_size, shuffle=True)
#这里batch的大小先暂时用训练集的大小，最好是小于,这里是67个，原先是90
test_dataset = TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_size, shuffle=False)

train_loader2=torch.utils.data.DataLoader(train_dataset, batch_size=train_size, shuffle=False)#,pin_memory=True
'''
##############################################################
#
##############################################################
'''
# %% 实例化，这里可以修改不同的网络ResNet18,ResNet34等
net = ResNet18()
# print(net)
net.to(device)
lr = 1e-4#学习率，这里是默认值，可修改，大了可加快学习速度，但是结果可能会跑飞，但会收敛的性能会降低，反之亦然
criterion = nn.MSELoss(reduction='sum')
lamda = 0.01 #L1正则化

# criterion = nn.MSELoss() #损失函数
#均方损失函数，这里sum代表最后的损失都加起来，不除以n
# optimizer = optim.Adam(net.parameters(), lr=lr)#, weight_decay=0.01)Adam优化器
# optimizer = optim.Rprop(net.parameters(),lr=lr)

# optimizer=optim.SGD(net.parameters(),lr=lr)#有问题
# optimizer=optim.SGD(net.parameters(),lr=lr,momentum=0.8)#momentum方式
# optimizer=optim.RMSprop(net.parameters(),lr=lr,alpha=0.9)
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)  #weight_decay=0.01-L2正则
# optimizer=optim.Adam(net.parameters(),lr=lr,betas=(0.9,0.99))
min_rmse_tr=50
max_R2_test=-10
# %%

loss_train = np.zeros(n_epochs) 
loss_test = np.zeros(n_epochs)
rmse_train = np.zeros(n_epochs) 
rmse_test = np.zeros(n_epochs)
R2_test = np.zeros(n_epochs)
R2_train = np.zeros(n_epochs)


for epoch_num in tqdm(range(n_epochs),ncols=50):
    net.train()
    train_running_loss = 0 
    train_running_R2 = 0 
    train_epoch_rmse = 0
    train_epoch_loss = 0

    test_running_loss = 0
    test_running_R2 = 0
    test_epoch_loss = 0
    test_epoch_rmse = 0
    
    for X_train, y_train in train_loader:
        X_train = X_train.to(device).float()
        y_train = y_train.to(device).float()
        y_train_pred_batch = net(X_train)
        loss = criterion(y_train_pred_batch, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()
    train_epoch_loss = train_running_loss / train_size#记录损失
    train_epoch_rmse = math.sqrt(train_epoch_loss)
    loss_train[epoch_num] = train_epoch_loss
    rmse_train[epoch_num] = train_epoch_rmse


    net.eval()#用于测试
    for X_train, y_train in train_loader2:
        X_train = X_train.to(device).float()
        y_train = y_train.to(device).float()
        y_train_pred_batch = net(X_train)
        trloss1 = criterion(y_train_pred_batch, y_train)
        trR2=metrics.r2_score(y_train.detach().cpu(), y_train_pred_batch.detach().cpu()) #GPU
        train_running_loss += trloss1.item()
        train_running_R2 += trR2.item()
    for X_test, y_test in test_loader:
        X_test = X_test.to(device).float()
        y_test = y_test.to(device).float()
        y_test_pred_batch = net(X_test)
        loss1 = criterion(y_test_pred_batch, y_test)
        R2=metrics.r2_score(y_test.detach().cpu(), y_test_pred_batch.detach().cpu()) #GPU
        test_running_loss += loss1.item()
        test_running_R2 += R2.item()
    test_epoch_loss = test_running_loss / test_size
    test_epoch_rmse = math.sqrt(test_epoch_loss)
    loss_test[epoch_num] = test_epoch_loss
    rmse_test[epoch_num] = test_epoch_rmse  
    R2_test[epoch_num] = test_running_R2
    R2_train[epoch_num] = train_running_R2

    if test_running_R2>max_R2_test:
        max_R2_test=test_running_R2
        torch.save(net.state_dict(),'./My Drive/Colab Notebooks/res_model_1w')

print('\n')
print('R2_max_index: {:.4f},RMSR_train: {:.4f},RMSR_test: {:.4f},R2_train: {:.4f},R2_test: {:.4f},'.format(np.argmax(R2_test), rmse_train[np.argmin(rmse_train)],rmse_test[np.argmin(rmse_train)] ,R2_train[np.argmin(rmse_train)],max(R2_test)))

# %%
figsize=(3, 1)
plt.subplot(211)
plt.plot(rmse_train,'b')
plt.plot(rmse_test,'r')
plt.legend(['rmse_train','rmse_test'], fontsize=8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.rcParams['figure.dpi'] = 72


figsize=(3, 1)
c = R2_train.copy()
cc = R2_test.copy()
c[c < 0] = 0
cc[cc < 0] = 0
plt.subplot(212)
plt.plot(c,'b')
plt.plot(cc,'r')
plt.legend(['R2_train','R2_test'], fontsize=8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('R2', fontsize=12)
plt.title('R2_train')
plt.rcParams['figure.dpi'] = 72
plt.show()

time_end=time.time()


print('Time cost',time_end-time_start,'s')
```



# 推荐阅读

https://cloud.tencent.com/developer/column/77164

