(window.webpackJsonp=window.webpackJsonp||[]).push([[16],{627:function(t,a,s){"use strict";s.r(a);var _=s(3),r=Object(_.a)({},(function(){var t=this,a=t.$createElement,s=t._self._c||a;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("Boxx",{attrs:{changeTime:"10000"}}),t._v(" "),s("div",{staticClass:"custom-block tip"},[s("p",{staticClass:"title"},[t._v("前言")]),s("p",[t._v("这里主要讲深度学习的一些基础知识。")])]),t._v(" "),s("h1",{attrs:{id:"_1-深度学学习的一些知识"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_1-深度学学习的一些知识"}},[t._v("#")]),t._v(" 1 深度学学习的一些知识")]),t._v(" "),s("h2",{attrs:{id:"bn"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#bn"}},[t._v("#")]),t._v(" BN")]),t._v(" "),s("p",[t._v("我们假设有一批图像的feature maps传入网络中（如上）。其中，N表示batch_size，9*9表示图像的大小，5表示channel。")]),t._v(" "),s("p",[t._v("BN做了一件什么事呢。")]),t._v(" "),s("p",[t._v("（1）把不同batch_size的同一个channel的feature map进行求均值，得到mean")]),t._v(" "),s("p",[t._v("（2）把不同batch的同一个channel的feature map进行求标准差，得到std")]),t._v(" "),s("p",[t._v("（3）最后对每一个channel的每一个feature map减去对应channel的mean，再除以std，就得到了新的N"),s("em",[t._v("9")]),t._v("9*5的feature maps")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://img-blog.csdnimg.cn/20190601170251915.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjA3ODYxOA==,size_16,color_FFFFFF,t_70",alt:"img"}})]),t._v(" "),s("blockquote",[s("p",[t._v("BN层的的好处：")]),t._v(" "),s("ol",[s("li",[t._v("Internal Covariate Shift(本层的输出会导致下一层的输入的分布发生变化，因而导致训练效果变差)")]),t._v(" "),s("li",[s("a",{attrs:{href:"https://blog.csdn.net/ygfrancois/article/details/90382459",target:"_blank",rel:"noopener noreferrer"}},[t._v("减轻了梯度消失"),s("OutboundLink")],1),t._v("和梯度爆炸的问题：")]),t._v(" "),s("li",[t._v("BN可以支持更多的激活函数")]),t._v(" "),s("li",[t._v("BN层一定程度上增加了 泛化能力")])])]),t._v(" "),s("h2",{attrs:{id:"layernormalization-ln"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#layernormalization-ln"}},[t._v("#")]),t._v(" LayerNormalization（LN）")]),t._v(" "),s("h3",{attrs:{id:"理论"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#理论"}},[t._v("#")]),t._v(" 理论")]),t._v(" "),s("p",[t._v("好，各位，刚才的任务完成了，我们进行下一项任务，名叫：LN。。。")]),t._v(" "),s("p",[t._v("（1）第一梯队的"),s("strong",[t._v("所有通道的第一列")]),t._v("，听清楚了，是第一列，给到我你们的均值（mean）")]),t._v(" "),s("p",[t._v("（2）给完以后，给到我你们的标准差（std）")]),t._v(" "),s("p",[t._v("（3）然后：把你们的数值减去mean，再除以std")]),t._v(" "),s("p",[t._v("（4）接着我会给你们一个gamma，把结果乘上去；还有一个beta，加上去")]),t._v(" "),s("p",[t._v("（5）OK，第一梯队的所有通道的第一列，给我最终结果。")]),t._v(" "),s("p",[t._v("（6）接下来，第一梯队的所有通道的其他列，按照第一列的步骤，开始！")]),t._v(" "),s("p",[t._v("（7）其他梯队，按照第一梯队的流程，GO！")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://img-blog.csdnimg.cn/20190601170311687.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjA3ODYxOA==,size_16,color_FFFFFF,t_70",alt:"img"}})]),t._v(" "),s("h3",{attrs:{id:"实际"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#实际"}},[t._v("#")]),t._v(" 实际")]),t._v(" "),s("p",[t._v("1）第一梯队的所有通道的第一列的第一行，听清楚了，是"),s("strong",[t._v("第一列的第一行")]),t._v("，给到我你们的均值（mean）")]),t._v(" "),s("p",[t._v("（2）给完以后，给到我你们的标准差（std）")]),t._v(" "),s("p",[t._v("（3）然后：把你们的数值减去mean，再除以std")]),t._v(" "),s("p",[t._v("（4）接着我会给你们一个gamma，把结果乘上去；还有一个beta，加上去")]),t._v(" "),s("p",[t._v("（5）OK，第一梯队的所有通道的第一列的第一行，给我最终结果。")]),t._v(" "),s("p",[t._v("（6）接下来，第一梯队的所有通道的第一列的其他行，按照第一列第一行的步骤，开始！")]),t._v(" "),s("p",[t._v("（7）接下来，第一梯队的所有通道的其他列，按照第一列步骤，开始！")]),t._v(" "),s("p",[t._v("（8）其他梯队，按照第一梯队的流程，GO！")]),t._v(" "),s("p",[t._v("这就是（才是实战中的）LN")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://img-blog.csdnimg.cn/20190601170337230.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjA3ODYxOA==,size_16,color_FFFFFF,t_70",alt:"img"}})]),t._v(" "),s("h2",{attrs:{id:"in"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#in"}},[t._v("#")]),t._v(" IN")]),t._v(" "),s("p",[t._v("好，接下来是最后一个任务了，IN")]),t._v(" "),s("p",[t._v("（1）第一梯队，给出你们所有通道所有行所有列的均值（mean）")]),t._v(" "),s("p",[t._v("（2）第一梯队，给出标准差（std）")]),t._v(" "),s("p",[t._v("（3）乘上gamma和beta，在给到我")]),t._v(" "),s("p",[t._v("（4）其他梯队，跟上！")]),t._v(" "),s("p",[t._v("（5）任务完成，开饭！")]),t._v(" "),s("p",[t._v("这就是IN")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://img-blog.csdnimg.cn/20190601170351203.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjA3ODYxOA==,size_16,color_FFFFFF,t_70",alt:"img"}})]),t._v(" "),s("h2",{attrs:{id:"总结"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#总结"}},[t._v("#")]),t._v(" 总结")]),t._v(" "),s("p",[t._v("BN做的事情是沿着channel方向，把每个channel的特征图做标准化")]),t._v(" "),s("p",[t._v("LN做的事情是沿着batch方向，同时也沿着time_step方向，把每一个单词做标准化")]),t._v(" "),s("p",[t._v("IN则很生猛，直接把单个输入特征图直接整体做标准化")]),t._v(" "),s("p",[t._v("因此呢，BN对CNN效果很好，因为CNN本身的目的就是结合不同batch_size的特征，做特征提取；LN对单个词做的标准化，对时序特征的效果特别好；IN是对整体单个输入做标准化，所以在风格迁移的时候，能对单一风格做到非常好的特征提取。")]),t._v(" "),s("p",[s("strong",[t._v("BN的缺陷")])]),t._v(" "),s("p",[s("strong",[t._v("缺陷如下：")])]),t._v(" "),s("p",[t._v("1、BN是在batch size样本上各个维度做标准化的，所以size越大肯定越能得出合理的μ和σ来做标准化，因此BN比较依赖size的大小。\n2、在训练的时候，是分批量进行填入模型的，但是在预测的时候，如果只有一个样本或者很少量的样本来做inference，这个时候用BN显然偏差很大，例如在"),s("strong",[t._v("线学习场景")]),t._v("。\n3、RNN是一个动态的网络，也就是size是变化的，可大可小，造成多样本维度都没法对齐，所以不适合用BN。")]),t._v(" "),s("p",[s("strong",[t._v("LN带来的优势：")])]),t._v(" "),s("p",[t._v("1、Layer Normalization是每个样本内部做标准化，跟size没关系，不受其影响。\n2、RNN中LN也不受影响，内部自己做标准化，所以LN的应用面更广。")]),t._v(" "),s("h2",{attrs:{id:"_1-激活函数"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_1-激活函数"}},[t._v("#")]),t._v(" 1 激活函数")]),t._v(" "),s("p",[t._v("所谓激活函数（Activation Function），就是在"),s("a",{attrs:{href:"https://baike.baidu.com/item/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/382460",target:"_blank",rel:"noopener noreferrer"}},[t._v("人工神经网络"),s("OutboundLink")],1),t._v("的神经元上运行的"),s("a",{attrs:{href:"https://baike.baidu.com/item/%E5%87%BD%E6%95%B0/301912",target:"_blank",rel:"noopener noreferrer"}},[t._v("函数"),s("OutboundLink")],1),t._v("，负责将神经元的输入映射到输出端。")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://bkimg.cdn.bcebos.com/pic/0eb30f2442a7d933739ff390a14bd11373f00119?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2U4MA==,g_7,xp_5,yp_5/format,f_auto",alt:"img"}})]),t._v(" "),s("h3",{attrs:{id:"softmax"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#softmax"}},[t._v("#")]),t._v(" Softmax")]),t._v(" "),s("p",[t._v("$$\nS_i=\\frac{e^i}{\\sum e^j }\n$$")]),t._v(" "),s("p",[t._v("映射区间[0,1],主要用于：离散化概率分布。")]),t._v(" "),s("p",[t._v("https://blog.csdn.net/bitcarmanlee/article/details/82320853")]),t._v(" "),s("p",[t._v("softmax函数，又称归一化指数函数。它"),s("strong",[t._v("是二分类函数sigmoid在多分类上的推广")]),t._v("，目的是将多分类的结果以概率的形式展现出来。下图展示了softmax的计算方法：")]),t._v(" "),s("p",[t._v("softmax就是先把输出用指数表示，保证输出非负，然后在加权平均，获得0-1之间的预测结果概率。具体如下：")]),t._v(" "),s("p",[t._v("1）分子：通过指数函数，将实数输出映射到零到正无穷。")]),t._v(" "),s("p",[t._v("2）分母：将所有结果相加，进行归一化。")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://i.loli.net/2021/07/16/tvy3bHkmUTRO2QP.png",alt:"image-20210716155825571"}})]),t._v(" "),s("p",[t._v("之其中，这里的$W_yx$就是某个的输出结果(softmax之前)")]),t._v(" "),s("h3",{attrs:{id:"sigmoid"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#sigmoid"}},[t._v("#")]),t._v(" Sigmoid")]),t._v(" "),s("ul",[s("li",[t._v("映射区间(0, 1)")]),t._v(" "),s("li",[t._v("也称logistic函数")])]),t._v(" "),s("p",[t._v("$$\nf(x)=\\frac{1}{1+e^{-x}}\n$$")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://img-blog.csdnimg.cn/20181130114706469.gif",alt:"sigmoid函数"}})]),t._v(" "),s("ul",[s("li",[t._v("映射区间(0, 1)")]),t._v(" "),s("li",[t._v("也称logistic函数")]),t._v(" "),s("li",[t._v("存在三个"),s("strong",[t._v("问题")]),t._v(":\n"),s("ol",[s("li",[t._v('饱和的神经元会"杀死"梯度,指离中心点较远的x处的导数接近于0,停止反向传播的学习过程.')]),t._v(" "),s("li",[t._v("sigmoid的输出不是以0为中心,而是0.5,这样在求权重w的梯度时,梯度总是正或负的.")]),t._v(" "),s("li",[t._v("指数计算耗时")])])])]),t._v(" "),s("h3",{attrs:{id:"relu"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#relu"}},[t._v("#")]),t._v(" Relu")]),t._v(" "),s("img",{staticStyle:{zoom:"50%"},attrs:{src:"https://bkimg.cdn.bcebos.com/pic/d788d43f8794a4c25b5e4dd902f41bd5ac6e39c6?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2U5Mg==,g_7,xp_5,yp_5/format,f_auto",alt:"img"}}),t._v(" "),s("h3",{attrs:{id:"leaky-relus"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#leaky-relus"}},[t._v("#")]),t._v(" Leaky ReLUs")]),t._v(" "),s("hr"),t._v(" "),s("p",[t._v("ReLU是将所有的负值都设为零，相反，Leaky ReLU是给所有负值赋予一个非零斜率。Leaky ReLU激活函数是在声学模型（2013）中首次提出的。以数学的方式我们可以表示为：")]),t._v(" "),s("p",[s("img",{staticStyle:{zoom:"33%"},attrs:{src:"http://p0.ifengimg.com/pmop/2017/0701/CFC5A1C95A84A6D8CF3FFC1DD30597782AEEAE57_size20_w740_h231.jpeg",alt:"img"}}),t._v("ai是（1，+∞）区间内的固定参数。")]),t._v(" "),s("p",[s("img",{attrs:{src:"http://p0.ifengimg.com/pmop/2017/0701/C56E5C6FCBB36E70BA5EBC90CBD142BA320B3DF6_size19_w740_h217.jpeg",alt:"img"}})]),t._v(" "),s("h3",{attrs:{id:"prelu"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#prelu"}},[t._v("#")]),t._v(" PRelu")]),t._v(" "),s("p",[t._v("负值部分的斜率是根据数据来定的")]),t._v(" "),s("h3",{attrs:{id:"rrelu"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#rrelu"}},[t._v("#")]),t._v(" "),s("strong",[t._v("RReLU")])]),t._v(" "),s("p",[s("strong",[t._v("随机纠正线性单元（RReLU）")]),t._v(",训练的时候负数部分的斜率是不固定的。a_ji是从一个均匀的分布U(I,u)中随机抽取的数值")]),t._v(" "),s("h3",{attrs:{id:"elu"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#elu"}},[t._v("#")]),t._v(" ELU")]),t._v(" "),s("p",[t._v("ELU函数公式和曲线如下图")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://img-blog.csdn.net/20180104121207844?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast",alt:"elu函数公式"}})]),t._v(" "),s("p",[s("img",{attrs:{src:"https://img-blog.csdn.net/20180104121237935?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast",alt:"elu函数图"}})]),t._v(" "),s("p",[t._v("是有一定的输出的，而且这部分输出还具有一定的抗干扰能力。这样可以消除ReLU死掉的问题，不过还是有梯度饱和和指数运算的问题。"),s("strong",[t._v("ELU对于输入特征只定性不定量")]),t._v("。")]),t._v(" "),s("h3",{attrs:{id:"selu-就是在elu前面乘以一个-lambda-并告诉你-lambda-alpha-是多少"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#selu-就是在elu前面乘以一个-lambda-并告诉你-lambda-alpha-是多少"}},[t._v("#")]),t._v(" SELU(就是在ELU前面乘以一个$\\lambda$，并告诉你$\\lambda，\\alpha$是多少)")]),t._v(" "),s("p",[t._v("上面那个ELU，"),s("img",{attrs:{src:"https://math.jianshu.com/math?formula=%CE%B1",alt:"α"}}),t._v("要设多少？后来又出现一种新的方法，叫做：SELU。它相对于ELU做了一个新的变化：就是现在把每一个值的前面都乘上一个"),s("img",{attrs:{src:"https://math.jianshu.com/math?formula=%CE%BB",alt:"λ"}}),t._v("，然后他告诉你说"),s("img",{attrs:{src:"https://math.jianshu.com/math?formula=%CE%BB",alt:"λ"}}),t._v("跟"),s("img",{attrs:{src:"https://math.jianshu.com/math?formula=%CE%B1",alt:"α"}}),t._v("应该设多少，"),s("img",{attrs:{src:"https://math.jianshu.com/math?formula=%CE%B1%3D1.67326324%E2%80%A6%E2%80%A6",alt:"α=1.67326324……"}}),t._v("，然后"),s("img",{attrs:{src:"https://math.jianshu.com/math?formula=%CE%BB%3D1.050700987%E2%80%A6%E2%80%A6",alt:"λ=1.050700987……"}}),t._v("。")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://upload-images.jianshu.io/upload_images/5631876-a163982aad9150ed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/404/format/webp",alt:"img"}})]),t._v(" "),s("h3",{attrs:{id:"gelu"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#gelu"}},[t._v("#")]),t._v(" "),s("font",{attrs:{color:"red"}},[t._v("GELU")])],1),t._v(" "),s("p",[t._v("GELU（高斯误差线性单元）是一个非初等函数形式的激活函数，是RELU的变种。由16年论文 "),s("a",{attrs:{href:"https://arxiv.org/abs/1606.08415",target:"_blank",rel:"noopener noreferrer"}},[t._v("Gaussian Error Linear Units (GELUs)"),s("OutboundLink")],1),t._v(" 提出，随后被GPT-2、BERT、RoBERTa、ALBERT 等NLP模型所采用。")]),t._v(" "),s("h3",{attrs:{id:"全家福"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#全家福"}},[t._v("#")]),t._v(" 全家福")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://images2017.cnblogs.com/blog/606386/201711/606386-20171102101447857-1756364198.png",alt:"ReLU系列对比"}})]),t._v(" "),s("h3",{attrs:{id:"swish"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#swish"}},[t._v("#")]),t._v(" "),s("a",{attrs:{href:"https://arxiv.org/abs/1710.05941",target:"_blank",rel:"noopener noreferrer"}},[t._v("Swish"),s("OutboundLink")],1)]),t._v(" "),s("p",[t._v("还有一个新的激活函数叫做"),s("strong",[t._v("Swish")]),t._v("。这个"),s("strong",[t._v("Swish")]),t._v("激活函数长什么样子，它是一个非常神奇的激活函数，他把"),s("strong",[t._v("sigmoid")]),t._v("乘上"),s("img",{attrs:{src:"https://math.jianshu.com/math?formula=z",alt:"z"}}),t._v("得到她的output。")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210716172123869.png",alt:"image-20210716172123869"}})]),t._v(" "),s("p",[t._v("β是个常数或可训练的参数.Swish 具备无上界有下界、平滑、非单调的特性。\nSwish 在深层模型上的效果优于 ReLU。")]),t._v(" "),s("hr"),t._v(" "),s("p",[t._v("GELU 与 Swish 激活函数（x · σ(βx)）的函数形式和性质非常相像，一个是固定系数 1.702，另一个是可变系数 β（可以是可训练的参数，也可以是通过搜索来确定的常数），两者的实际应用表现也相差不大。")]),t._v(" "),s("h3",{attrs:{id:"tanh"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#tanh"}},[t._v("#")]),t._v(" tanh")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210716172707674.png",alt:"image-20210716172707674"}})]),t._v(" "),s("p",[s("img",{attrs:{src:"https://images2018.cnblogs.com/blog/606386/201807/606386-20180712202915278-1408388561.png",alt:"蓝色sigmoid-红色tanh"}})]),t._v(" "),s("h3",{attrs:{id:"为什么tanh相比sigmoid收敛更快"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#为什么tanh相比sigmoid收敛更快"}},[t._v("#")]),t._v(" 为什么tanh相比sigmoid收敛更快:")]),t._v(" "),s("ol",[s("li",[s("p",[t._v("梯度消失问题程度\n$tanh′(x)=1−tanh(x)^2∈(0,1)$")]),t._v(" "),s("p",[t._v("sigmoid:$ s′(x)=s(x)×(1−s(x))∈(0,1/4)$\n可以看出tanh(x)的"),s("font",{attrs:{color:"red"}},[t._v("梯度消失问题比sigmoid要轻")]),t._v(".梯度如果过早消失,收敛速度较慢.")],1)]),t._v(" "),s("li",[s("p",[s("font",{attrs:{color:"red"}},[t._v("以零为中心的影响")]),t._v("\n如果当前参数(w0,w1)的最佳优化方向是(+d0, -d1),则根据反向传播计算公式,我们希望 x0 和 x1 符号相反。但是如果上一级神经元采用 Sigmoid 函数作为激活函数，sigmoid不以0为中心，输出值恒为正，那么我们无法进行最快的参数更新，而是走 Z 字形逼近最优解。["),s("a",{attrs:{href:"https://www.cnblogs.com/makefile/p/activation-function.html#fn4",target:"_blank",rel:"noopener noreferrer"}},[t._v("4]"),s("OutboundLink")],1)],1)])]),t._v(" "),s("h2",{attrs:{id:"激活函数的作用"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#激活函数的作用"}},[t._v("#")]),t._v(" 激活函数的作用")]),t._v(" "),s("ol",[s("li",[s("p",[t._v("加入非线性因素")])]),t._v(" "),s("li",[s("p",[t._v("充分组合特征")]),t._v(" "),s("p",[t._v("**为什么ReLU,Maxout等能够提供网络的非线性建模能力？**它们看起来是分段线性函数，然而并不满足完整的线性要求：加法f(x+y)=f(x)+f(y)和乘法f(ax)=a×f(x)或者写作f(αx1+βx2)=αf(x1)+βf(x2)f(αx1+βx2)=αf(x1)+βf(x2)。非线性意味着得到的输出不可能由输入的线性组合重新得到（重现）。**假如网络中不使用非线性激活函数，那么这个网络可以被一个单层感知器代替得到相同的输出，**因为线性层加起来后还是线性的，可以被另一个线性函数替代。")])])]),t._v(" "),s("h2",{attrs:{id:"梯度消失与梯度爆炸"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#梯度消失与梯度爆炸"}},[t._v("#")]),t._v(" "),s("font",{attrs:{color:"red"}},[t._v("梯度消失与梯度爆炸")])],1),t._v(" "),s("p",[t._v("​\t在反向传播过程中需要对激活han函数进行求导，如果导数大于1，那么随着网络层数的增加梯度更新将会朝着指数爆炸的方式增加这就是梯度爆炸。同样如果导数小于1，那么随着网络层数的增加梯度更新信息会朝着指数衰减的方式减少这就是梯度消失。因此，梯度消失、爆炸，其根本原因在于反向传播训练法则，属于先天不足。")]),t._v(" "),s("p",[s("strong",[t._v("【"),s("font",{attrs:{color:"red"}},[t._v("梯度消失")]),t._v("】"),s("strong",[t._v("原因有：一是在")]),t._v("深层网络")],1),t._v("中，二是采用了"),s("strong",[t._v("不合适的损失函数")]),t._v("，比如sigmoid。当梯度消失发生时，接近于输出层的隐藏层由于其梯度相对正常，所以权值更新时也就相对正常，但是当越靠近输入层时，由于梯度消失现象，会导致靠近输入层的隐藏层权值更新缓慢或者更新停滞。这就导致在训练时，只等价于后面几层的浅层网络的学习。")]),t._v(" "),s("p",[s("strong",[t._v("【"),s("font",{attrs:{color:"red"}},[t._v("梯度爆炸")]),t._v("】"),s("strong",[t._v("一般出现在")]),t._v("深层网络")],1),t._v("和"),s("strong",[t._v("权值初始化值太大")]),t._v("的情况下。在深层神经网络或循环神经网络中，"),s("font",{attrs:{color:"blue"}},[s("strong",[t._v("误差的梯度可在更新中累积相乘")])]),t._v("。如果网络层之间的"),s("strong",[t._v("梯度值大于 1.0")]),t._v("，那么"),s("strong",[t._v("重复相乘会导致梯度呈指数级增长")]),t._v("，梯度变的非常大，然后导致网络权重的大幅更新，并因此使网络变得不稳定。")],1),t._v(" "),s("h3",{attrs:{id:"原因"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#原因"}},[t._v("#")]),t._v(" 原因")]),t._v(" "),s("h4",{attrs:{id:"深层网络"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#深层网络"}},[t._v("#")]),t._v(" 深层网络")]),t._v(" "),s("p",[t._v("深度网络是多层非线性函数的堆砌，整个深度网络可以视为是一个"),s("strong",[t._v("复合的非线性多元函数")]),t._v("。（这些非线性多元函数其实就是每层的激活函数），那么对loss function求不同层的权值偏导，相当于应用梯度下降的链式法则，链式法则是一个连乘的形式，所以当层数越深的时候，梯度将以指数传播。")]),t._v(" "),s("p",[t._v("如果接近"),s("font",{attrs:{color:"cyan"}},[t._v("输出层")]),t._v("的激活函数求导后梯度值大于1，那么层数增多的时候，最终求出的梯度很容易指数级增长，就会产生"),s("strong",[t._v("梯度爆炸")]),t._v("；相反，如果小于1，那么经过链式法则的连乘形式，也会很容易衰减至0，就会产生"),s("strong",[t._v("梯度消失")]),t._v("。")],1),t._v(" "),s("p",[t._v("从深层网络角度来讲，不同的层学习的速度差异很大，表现为网络中靠近输出的层学习的情况很好，靠近输入的层学习的很慢，有时甚至训练了很久，前几层的权值和刚开始随机初始化的值差不多。因此，"),s("font",{attrs:{color:"cyan"}},[t._v("梯度消失、爆炸，其根本原因在于反向传播训练法则，属于先天不足")]),t._v("。")],1),t._v(" "),s("h4",{attrs:{id:"激活函数"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#激活函数"}},[t._v("#")]),t._v(" "),s("strong",[t._v("激活函数")])]),t._v(" "),s("p",[t._v("以下图的反向传播为例（假设每一层只有一个神经元且对于每一层"),s("img",{attrs:{src:"https://www.zhihu.com/equation?tex=y_i%3D%5Csigma%5Cleft%28z_i%5Cright%29%3D%5Csigma%5Cleft%28w_ix_i%2Bb_i%5Cright%29",alt:"[公式]"}}),t._v("，其中"),s("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%5Csigma",alt:"[公式]"}}),t._v("为sigmoid函数）")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://pic3.zhimg.com/80/v2-ea9beb6c28c7d4e89be89dc5f4cbae2e_720w.png",alt:"img"}})]),t._v(" "),s("p",[t._v("可以推导出：")]),t._v(" "),s("img",{staticStyle:{zoom:"67%"},attrs:{src:"https://pic1.zhimg.com/80/v2-8e6665fb67f086c0864583caa48c8d30_720w.jpg",alt:"img"}}),t._v(" "),s("p",[t._v("原因看下图，sigmoid导数的图像。")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://pic3.zhimg.com/80/v2-cd452d42a0f5dcad974098dda44c4622_720w.jpg",alt:"img"}})]),t._v(" "),s("p",[t._v("如果使用sigmoid作为损失函数，其梯度是不可能超过0.25的，而我们初始化的网络权值"),s("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%7Cw%7C",alt:"[公式]"}}),t._v("通常都小于1，因此"),s("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%7C%5Csigma%27%5Cleft%28z%5Cright%29w%7C%5Cleq%5Cfrac%7B1%7D%7B4%7D",alt:"[公式]"}}),t._v("，因此对于上面的链式求导，层数越多，求导结果"),s("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+C%7D%7B%5Cpartial+b_1%7D",alt:"[公式]"}}),t._v("越小，因而很容易发生梯度消失。")]),t._v(" "),s("h4",{attrs:{id:"初始化权重的值过大"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#初始化权重的值过大"}},[t._v("#")]),t._v(" "),s("strong",[t._v("初始化权重的值过大")])]),t._v(" "),s("img",{staticStyle:{zoom:"80%"},attrs:{src:"https://pic1.zhimg.com/80/v2-8e6665fb67f086c0864583caa48c8d30_720w.jpg",alt:"img"}}),t._v(" "),s("p",[t._v("如上图所示，当"),s("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%7C%5Csigma%27%5Cleft%28z%5Cright%29w%7C%3E1",alt:"[公式]"}}),t._v("，也就是"),s("img",{attrs:{src:"https://www.zhihu.com/equation?tex=w",alt:"[公式]"}}),t._v("比较大的情况。根据链式相乘(反向传播)可得，则前面的网络层比后面的网络层梯度变化更快，很容易发生梯度爆炸的问题。（再理解下)")]),t._v(" "),s("h3",{attrs:{id:"解决办法"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#解决办法"}},[t._v("#")]),t._v(" 解决办法")]),t._v(" "),s("p",[t._v("梯度消失和梯度爆炸本质上是一样的，都是因为网络层数太深而引发的梯度反向传播中的连乘效应。")]),t._v(" "),s("p",[t._v("解决梯度消失、爆炸主要有以下几种方案：")]),t._v(" "),s("h4",{attrs:{id:"换用relu、leakyrelu、elu等激活函数-梯度大部分落在常数上"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#换用relu、leakyrelu、elu等激活函数-梯度大部分落在常数上"}},[t._v("#")]),t._v(" 换用Relu、LeakyRelu、Elu等激活函数("),s("strong",[t._v("梯度大部分落在常数上")]),t._v(")")]),t._v(" "),s("p",[t._v("ReLu：让激活函数的导数为1")]),t._v(" "),s("p",[t._v("LeakyReLu：包含了ReLu的几乎所有有点，同时解决了ReLu中0区间带来的影响")]),t._v(" "),s("p",[t._v("ELU：和LeakyReLu一样，都是为了解决0区间问题，相对于来，elu计算更耗时一些（为什么）")]),t._v(" "),s("p",[t._v("具体可以看"),s("a",{attrs:{href:"#activation"}},[t._v("关于各种激活函数的解析与讨论")])]),t._v(" "),s("h4",{attrs:{id:"batchnormalization"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#batchnormalization"}},[t._v("#")]),t._v(" BatchNormalization")]),t._v(" "),s("p",[t._v("BN本质上是解决传播过程中的梯度问题，具体待补充完善，查看"),s("a",{attrs:{href:"..."}},[t._v("BN")])]),t._v(" "),s("h4",{attrs:{id:"resnet残差结构"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#resnet残差结构"}},[t._v("#")]),t._v(" ResNet残差结构")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://pic4.zhimg.com/80/v2-68f5136f96c6ecce7ccc7b9e9a569f63_720w.jpg",alt:"img"}})]),t._v(" "),s("h4",{attrs:{id:"lstm结构"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#lstm结构"}},[t._v("#")]),t._v(" LSTM结构")]),t._v(" "),s("p",[s("strong",[t._v("STM")]),t._v("全称是长短期记忆网络（long-short term memory networks），LSTM的结构设计可以改善RNN中的梯度消失的问题。主要原因在于LSTM内部复杂的“门”(gates)，如下图所示。")]),t._v(" "),s("p",[s("img",{attrs:{src:"https://pic1.zhimg.com/80/v2-2b5e5e1f76374c764d24ae5d70e94288_720w.jpg",alt:"img"}})]),t._v(" "),s("p",[t._v("LSTM 通过它内部的“门”可以在接下来更新的时候“记住”前几次训练的”残留记忆“。")]),t._v(" "),s("h4",{attrs:{id:"预训练加finetunning"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#预训练加finetunning"}},[t._v("#")]),t._v(" 预训练加finetunning")]),t._v(" "),s("p",[t._v("此方法来自Hinton在06年发表的论文上，其基本思想是每次训练一层隐藏层节点，将上一层隐藏层的输出作为输入，而本层的输出作为下一层的输入，这就是逐层预训练。")]),t._v(" "),s("p",[t._v("训练完成后，再对整个网络进行“微调（fine-tunning）”。")]),t._v(" "),s("p",[t._v("目前应用的不是很多了。")]),t._v(" "),s("p",[t._v("此方法相当于是找全局最优，然后整合起来寻找全局最优，但是现在"),s("font",{attrs:{color:"red"}},[t._v("基本都是直接拿imagenet的预训练模型直接进行fine-tunning")]),t._v("。")],1),t._v(" "),s("h4",{attrs:{id:"梯度剪切、正则"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#梯度剪切、正则"}},[t._v("#")]),t._v(" 梯度剪切、正则")]),t._v(" "),s("p",[s("font",{attrs:{color:"blue"}},[t._v("梯度剪切")]),t._v("，其思想是"),s("strong",[t._v("设值一个剪切阈值，如果更新梯度时，梯度超过了这个阈值，那么就将其强制限制在这个范围之内")]),t._v("。这样可以防止梯度爆炸。")],1),t._v(" "),s("p",[s("font",{attrs:{color:"blue"}},[t._v("另一种防止梯度爆炸的手段是采用权重正则化")]),t._v("，正则化主要是通过"),s("strong",[t._v("对网络权重做正则")]),t._v("来限制过拟合，但是根据正则项在损失函数中的形式：")],1),t._v(" "),s("p",[t._v("可以看出，如果发生梯度爆炸，那么权值的范数就会变的非常大，反过来，通过限制正则化项的大小，也可以在一定程度上限制梯度爆炸的发生。")]),t._v(" "),s("p",[t._v("参考：")]),t._v(" "),s("p",[t._v("https://zhuanlan.zhihu.com/p/72589432")]),t._v(" "),s("p",[t._v("https://www.jianshu.com/p/3f35e555d5ba")]),t._v(" "),s("p",[t._v("https://baike.baidu.com/item/%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98/22761355?fr=aladdin")]),t._v(" "),s("p",[t._v("https://zhuanlan.zhihu.com/p/51490163")]),t._v(" "),s("h1",{attrs:{id:"_2-优化算法"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_2-优化算法"}},[t._v("#")]),t._v(" 2 优化算法")]),t._v(" "),s("p",[t._v("https://blog.csdn.net/qunnie_yi/article/details/80129952")]),t._v(" "),s("p",[t._v("https://blog.csdn.net/fengchao03/article/details/78208414")]),t._v(" "),s("h2",{attrs:{id:"adam"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#adam"}},[t._v("#")]),t._v(" Adam")]),t._v(" "),s("p",[t._v("https://baijiahao.baidu.com/s?id=1668617930732883837&wfr=spider&for=pc")]),t._v(" "),s("h1",{attrs:{id:"_3-损失函数-cost-fun-损失函数-loss-func"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_3-损失函数-cost-fun-损失函数-loss-func"}},[t._v("#")]),t._v(" 3 损失函数（cost fun）/损失函数 （loss func）")]),t._v(" "),s("h2",{attrs:{id:"二次代价函数"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#二次代价函数"}},[t._v("#")]),t._v(" 二次代价函数")]),t._v(" "),s("img",{attrs:{src:"http://latex.codecogs.com/gif.latex?C=\\frac{1}{2n}\\sum_x||y(x)-a^L(x)||^2"}}),t._v(" "),s("h2",{attrs:{id:"交叉熵"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#交叉熵"}},[t._v("#")]),t._v(" 交叉熵")]),t._v(" "),s("img",{attrs:{src:"http://latex.codecogs.com/gif.latex?C=-\\frac{1}{n}\\sum_x[ylna+(1-y)ln(1-a)]"}}),t._v(" "),s("p",[t._v("MSEloss")]),t._v(" "),s("h1",{attrs:{id:"_4-几个网络"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#_4-几个网络"}},[t._v("#")]),t._v(" 4 几个网络")]),t._v(" "),s("h2",{attrs:{id:"resnet"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#resnet"}},[t._v("#")]),t._v(" resnet")]),t._v(" "),s("p",[t._v("模型退化原因：")]),t._v(" "),s("ol",[s("li",[t._v("过拟合，层数越多，参数越复杂，泛化能力弱")]),t._v(" "),s("li",[t._v("梯度消失/梯度爆炸，层数过多，"),s("strong",[t._v("梯度反向传播时由于链式求导连乘")]),t._v("使得梯度过大或者过小，使得梯度出现消失/爆炸，对于这种情况，可以通过BN(batch normalization)可以解决")]),t._v(" "),s("li",[t._v("由深度网络带来的退化问题，一般情况下，网络层数越深越容易学到一些复杂特征，理论上模型效果越好，但是由于深层网络中含有大量非线性变化，每次变化相当于丢失了特征的一些原始信息，从而导致层数越深退化现象越严重。")])]),t._v(" "),s("p",[s("img",{attrs:{src:"https://s3.us-west-2.amazonaws.com/secure.notion-static.com/d15d749d-1836-49e7-84fd-7f35b37e4385/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210716%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210716T034740Z&X-Amz-Expires=86400&X-Amz-Signature=fe8596bd65a531bf1d958bfab5890439624a052258077ead6270bb16f3dc9d8d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Untitled.png%22",alt:"img"}})])],1)}),[],!1,null,null,null);a.default=r.exports}}]);