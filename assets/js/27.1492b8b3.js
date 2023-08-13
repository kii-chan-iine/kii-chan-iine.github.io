(window.webpackJsonp=window.webpackJsonp||[]).push([[27],{642:function(t,a,e){"use strict";e.r(a);var r=e(3),s=Object(r.a)({},(function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("Boxx",{attrs:{changeTime:"10000"}}),t._v(" "),e("div",{staticClass:"custom-block tip"},[e("p",{staticClass:"title"},[t._v("前言")]),e("p",[t._v("整理了图像和NLP领域处理小样本的方法。")])]),t._v(" "),e("h1",{attrs:{id:"few-short-learning-小样本学习"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#few-short-learning-小样本学习"}},[t._v("#")]),t._v(" Few-short Learning (小样本学习)")]),t._v(" "),e("h2",{attrs:{id:"总述"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#总述"}},[t._v("#")]),t._v(" 总述")]),t._v(" "),e("p",[t._v("Few-short learning的目标不是让机器识别训练集中图片并且泛化到测试集,而是"),e("font",{attrs:{color:"red"}},[t._v("让机器自己学会学习")]),t._v("。")],1),t._v(" "),e("p",[t._v("在few-shot learning中有两个常用的术语：")]),t._v(" "),e("ul",[e("li",[e("p",[t._v("k-way：the support set has k classes.")])]),t._v(" "),e("li",[e("p",[t._v("n-shot：every class has n exampples.")])])]),t._v(" "),e("h3",{attrs:{id:"流程"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#流程"}},[t._v("#")]),t._v(" 流程")]),t._v(" "),e("ol",[e("li",[t._v("在一个很大的数据集上学习一个相似度函数。比如imagenet")]),t._v(" "),e("li",[t._v("利用相似度进行预测。选择相似度高的作为预测结果。")])]),t._v(" "),e("h3",{attrs:{id:"几个数据集"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#几个数据集"}},[t._v("#")]),t._v(" 几个数据集")]),t._v(" "),e("p",[t._v("omniglot")]),t._v(" "),e("p",[t._v("Mini-Imagenet")]),t._v(" "),e("h2",{attrs:{id:"预训练模型"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#预训练模型"}},[t._v("#")]),t._v(" "),e("font",{attrs:{color:"orange"}},[t._v("预训练模型")])],1),t._v(" "),e("p",[t._v("预训练（通用领域），然后(特定领域)Finetune：获得一定量的标注数据，然后基于一个基础网络进行微调。（这里有很多训练的trick，包括如何设置固定层和学习率等），如图3。这个方法可以相对较快，依赖数据量也不必太多，效果还行。")]),t._v(" "),e("h3",{attrs:{id:"余弦相似度"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#余弦相似度"}},[t._v("#")]),t._v(" 余弦相似度")]),t._v(" "),e("p",[t._v("**余弦相似度Cosine Similarity，**假设有两个单位向量，它们的夹角记为 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%5Ctheta",alt:"[公式]"}}),t._v(" ，此时余弦相似度为： "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=cos+%5Ctheta+%3D+x%5ET+w",alt:"[公式]"}}),t._v(" ，由此可以看出其实就是x在w方向上的投影长度，因此它的取值范围就是 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%5B-1%2C+1%5D",alt:"[公式]"}}),t._v(" 。")]),t._v(" "),e("img",{staticStyle:{zoom:"33%"},attrs:{src:"https://pic2.zhimg.com/80/v2-8ff7c2824659e0e3d04fc09f45432fd1_720w.jpg",alt:"img"}}),t._v(" "),e("p",[t._v("那么如果两个向量不是单位向量，此时就需要对其进行归一化。此时计算余弦相似度的公式为：")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=cos+%5Ctheta+%3D+%5Cfrac%7Bx%5ETw%7D%7B+%5Cparallel+x+%5Cparallel+_2+%2B+%5Cparallel+w+%5Cparallel+_2+%7D",alt:"[公式]"}})]),t._v(" "),e("h3",{attrs:{id:"softmax"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#softmax"}},[t._v("#")]),t._v(" Softmax")]),t._v(" "),e("h3",{attrs:{id:"few-shot-prediction-using-pretrain-cnn"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#few-shot-prediction-using-pretrain-cnn"}},[t._v("#")]),t._v(" "),e("strong",[t._v("Few-shot Prediction using pretrain CNN")])]),t._v(" "),e("p",[e("img",{attrs:{src:"https://pic2.zhimg.com/80/v2-22028b8acc7be9c1bb8b6e3713d0218d_720w.jpg",alt:"img"}})]),t._v(" "),e("p",[t._v("通过神经网络输出的特征向量进行求平均和归一化得到三个类别的表征 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%5Cmu_1%2C+%5Cmu_2%2C+%5Cmu_3",alt:"[公式]"}}),t._v(" ,且他们三个的2范围都为单位向量。做分类的时候，使用query的特征向量 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=q",alt:"[公式]"}}),t._v(" 和三个表征进行计算余弦相似度。")]),t._v(" "),e("p",[t._v("令 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=M+%3D+%5Cbegin+%7Bbmatrix%7D++%5Cmu_1+%5C%5C+%5Cmu_2+%5C%5C+%5Cmu_3+%5Cend+%7Bbmatrix%7D",alt:"[公式]"}}),t._v(" ，则有 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=p+%3D+softmax%28Mq%29+%3D+%5Cbegin+%7Bbmatrix%7D+%5Cmu_1%5ETq+%5C%5C+%5Cmu_2%5ETq+%5C%5C+%5Cmu_3%5ETq+%5C%5C+%5Cend+%7Bbmatrix%7D",alt:"[公式]"}})]),t._v(" "),e("p",[e("strong",[t._v("Fine-Tuning流程：")])]),t._v(" "),e("ul",[e("li",[t._v("将support set中每个样本记做 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%28x_j%2C+y_j%29",alt:"[公式]"}}),t._v(" 其中 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=x_j",alt:"[公式]"}}),t._v(" 表示图片， "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=y_j",alt:"[公式]"}}),t._v(" 表示标签。")]),t._v(" "),e("li",[t._v("预训练的神经网络"),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=f%28x_j%29+",alt:"[公式]"}}),t._v(" 对support set中的样本进行特征提取，得到对应的特征向量")]),t._v(" "),e("li",[e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=p_j+%3D+softmax%28w+%5Ccdot+f%28x_j%29+%2B+b",alt:"[公式]"}}),t._v(" 就是预测结果。通常会在support set上对w和b进行fine tuning，这样能大幅度提升准确率。")]),t._v(" "),e("li",[e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%5Cmin+%5Csum_j+CrossEntropy%28y_j%2C+p_j%29+%2B+Regularization",alt:"[公式]"}})])]),t._v(" "),e("h2",{attrs:{id:"基于metric-大概是与样本之间的距离有关"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#基于metric-大概是与样本之间的距离有关"}},[t._v("#")]),t._v(" 基于metric：（大概是与样本之间的距离有关）")]),t._v(" "),e("p",[t._v("该方法是对"),e("font",{attrs:{color:"red"}},[t._v("样本间距离分布")]),t._v("进行建模，使得属于同类样本靠近，异类样本远离。简单地，我们可以采用无参估计的方法，如"),e("strong",[t._v("KNN")]),t._v("。KNN虽然不需要训练，但效果依赖距离度量的选取, 一般采用的是一个比较随意的距离计算（L2）。另一种，也是目前比较好的方法，即通过学习"),e("strong",[t._v("一个端到端的最近邻分类器")]),t._v("，它同时受益于带参数和无参数的优点，使得不但能快速的学习到新的样本，而且能对已知样本有很好的泛化性。下面介绍3个相关的方法。")],1),t._v(" "),e("h3",{attrs:{id:"孪生网络-siamese-neural-networks-1"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#孪生网络-siamese-neural-networks-1"}},[t._v("#")]),t._v(" "),e("font",{attrs:{color:"orange"}},[e("strong",[t._v("孪生网络 （Siamese Neural Networks）")]),t._v(" "),e("a",{attrs:{href:"#refer-anchor-1"}},[e("sup",[t._v("1")])])])],1),t._v(" "),e("h4",{attrs:{id:"learning-pari-wise-similarity-score"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#learning-pari-wise-similarity-score"}},[t._v("#")]),t._v(" learning pari-wise Similarity score")]),t._v(" "),e("ul",[e("li",[e("strong",[t._v("构建数据集")])])]),t._v(" "),e("p",[t._v("每次取两个样本取比较他们的相似度。数据集要求比较大，利用数据集带标签，利用"),e("strong",[t._v("训练集")]),t._v("构建正负样本。正样本可以使得神经网络什么东西是同一类，负样本可以使神经网络了解事物的区别。")]),t._v(" "),e("ul",[e("li",[e("strong",[t._v("构建CNN for Feature Extraction")])])]),t._v(" "),e("p",[t._v("输入一张图片x，通过CNN提取特征输出一个特征向量f(x)--flatten之后的结果。")]),t._v(" "),e("ul",[e("li",[e("strong",[t._v("训练Siamese network")])])]),t._v(" "),e("p",[t._v("通过对两张图片的特征提取得到两个特征向量，然后使用全连接层对特征向量进行处理得到一个标量，使用非线性激活函数对标量进行激活，得到的输出在0-1之间，作为两张图片的相似度度量。")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://pic1.zhimg.com/80/v2-d92fd4c422119e8dbde6a0029d63c5d8_720w.jpg",alt:"img"}})]),t._v(" "),e("p",[t._v("模型训练完成之后，进行one-shot prediction（当然也可使是n-shot这里只是举例）6-way 1-shot prediction，训练siamese network的训练数据集中并不包括这6个类别，这就是few shot learning的难点所在。")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://pic3.zhimg.com/80/v2-6f8fc94ff2b6342ef54f2dd3b7bd2cbe_720w.jpg",alt:"img"}})]),t._v(" "),e("h4",{attrs:{id:"triplet-loss-学习到的是一个好的embedding"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#triplet-loss-学习到的是一个好的embedding"}},[t._v("#")]),t._v(" Triplet Loss--学习到的是一个好的embedding")]),t._v(" "),e("p",[t._v("第二种训练siamese network的方法。首先在训练样本集中随机选取一张图片作为anchor，然后从选中样本的类别中再选一张图片作为正样本（positive sample），然后在除此类别之外的类别中选取张图片作为负样本（negtive sample）。")]),t._v(" "),e("p",[t._v("由此得到三张图片，分别给入神经网络进行特征提取，由此进行计算损失时， "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=d%5E%7B%2B%7D",alt:"[公式]"}}),t._v(" 应该比较小，而 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=d%5E%7B-%7D",alt:"[公式]"}}),t._v(" 应该很大。")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://pic3.zhimg.com/80/v2-471ba5e8ffdc4bef4ce526b6cd24f5ce_720w.jpg",alt:"img"}})]),t._v(" "),e("p",[t._v("由此也可以理解为"),e("strong",[t._v("经过神经网络进行特征映射")]),t._v("，最终映射到同一特征空间中，进行举例度量：")]),t._v(" "),e("img",{staticStyle:{zoom:"67%"},attrs:{src:"https://pic2.zhimg.com/80/v2-5ce2cdc5eefaf6abedae07f4984b1f11_720w.jpg",alt:"img"}}),t._v(" "),e("p",[t._v("也即是模型鼓励损失函数中"),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=d%5E%7B%2B%7D",alt:"[公式]"}}),t._v("尽可能地小，"),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=d%5E%7B-%7D",alt:"[公式]"}}),t._v("尽可能地大，这样才能在特征空间中将其分开。在这个过程中定义一个阈值 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=%5Calpha",alt:"[公式]"}}),t._v(" ，如果 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=d%5E%7B-%7D+%5Cge+d%5E%7B%2B%7D+%2B+%5Calpha",alt:"[公式]"}}),t._v(" 此时就没有损失，否则损失就为 "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=d%5E%7B%2B%7D+%2B+%5Calpha+-+d%5E%7B-%7D",alt:"[公式]"}}),t._v(" ，即是： "),e("img",{attrs:{src:"https://www.zhihu.com/equation?tex=Loss%28x%5Ea%2C+x%5E%2B%2C+x%5E%7B-%7D%29+%3D+%5Cmax%5C%7B+0%2C++d%5E%7B%2B%7D+%2B+%5Calpha+-+d%5E%7B-%7D%5C%7D",alt:"[公式]"}}),t._v(" 。")]),t._v(" "),e("hr"),t._v(" "),e("p",[t._v("对输入的结构进行限制，自动发现新样本上的泛化特征。")]),t._v(" "),e("p",[t._v("通过一个有监督的基于孪生网络的度量学习来训练，然后重用那个网络所提取的特征进行one/few-shot学习。它是一个双路的神经网络，训练时，通过组合不同类的样本成对，同时输入网络进行训练，在最上层通过一个距离的交叉熵进行loss的计算。在预测的时候，以5way-5shot为例，从5个类中随机抽取5个样本，把这个mini-batch=25的数据输入网络，最后获得25个值，取分数最高对应的类别作为预测结果。")]),t._v(" "),e("h4",{attrs:{id:"小结"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#小结"}},[t._v("#")]),t._v(" 小结")]),t._v(" "),e("p",[t._v("总结一下训练siamese network的思路：")]),t._v(" "),e("ul",[e("li",[t._v("首先使用一个比较大的数据集训练siamese network。")]),t._v(" "),e("li",[t._v("然后给定一个k-way n-shot的support set，其特点在于训练集中并不包含support set中的k个类别。")]),t._v(" "),e("li",[t._v("给定一个query，去预测其类别。使用Siamese network进行计算相似度或距离。")])]),t._v(" "),e("h3",{attrs:{id:"匹配网络-matching-networks-2"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#匹配网络-matching-networks-2"}},[t._v("#")]),t._v(" 匹配网络（matching networks） "),e("a",{attrs:{href:"#refer-anchor-2"}},[e("sup",[t._v("2")])])]),t._v(" "),e("p",[t._v("不改变网络模型的前提下能对未知类别生成标签，其主要创新体现在建模过程和训练过程上。")]),t._v(" "),e("p",[t._v("对于建模过程的创新，文章提出了基于memory和attantion的matching nets，使得可以快速学习。")]),t._v(" "),e("p",[t._v("对于训练过程的创新，文章基于传统机器学习的一个原则，即训练和测试是要在同样条件下进行的，提出在训练的时候不断地让网络只看每一类的少量样本，这将和测试的过程是一致的。")]),t._v(" "),e("h3",{attrs:{id:"原型网络-prototypical-networks-3"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#原型网络-prototypical-networks-3"}},[t._v("#")]),t._v(" 原型网络 （Prototypical Networks） "),e("a",{attrs:{href:"#refer-anchor-3"}},[e("sup",[t._v("3")])])]),t._v(" "),e("p",[t._v("该方法思想十分简单高效，效果也非常好。它学习一个度量空间， 通过计算和每个类别的原型表达的距离来进行分类。文章基于这样的想法：每个类别都存在一个聚在某单个原型表达周围的embedding，该类的原型是support set在embedding空间中的均值。然后，分类问题变成在"),e("font",{attrs:{color:"red"}},[t._v("embedding空间中的最近邻")]),t._v("。")],1),t._v(" "),e("h2",{attrs:{id:"基于graph-neural-network-4"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#基于graph-neural-network-4"}},[t._v("#")]),t._v(" 基于graph neural network "),e("a",{attrs:{href:"#refer-anchor-4"}},[e("sup",[t._v("4")])])]),t._v(" "),e("p",[t._v("这是一篇比较新的文章，提交到ICLR 2018[4]。他定义了一个图神经网络框架，端到端地学习消息传递的“关系”型任务。在这里，每个样本看成图的节点，该方法不仅学习节点的embedding，也学习边的embedding。如图9，在网络第一层5个样本通过边模型A～构建了图，接着通过图卷积（graph conv）获得了节点的embedding，然后在后面的几层继续用A～更新图、用graph conv更新节点embedding, 这样便构成了一个深度GNN，最后输出样本的预测标签。")]),t._v(" "),e("h2",{attrs:{id:"基于元学习meta-learning-learn-to-learn"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#基于元学习meta-learning-learn-to-learn"}},[t._v("#")]),t._v(" 基于"),e("font",{attrs:{color:"red"}},[t._v("元学习")]),t._v("meta learning--\x3elearn to learn")],1),t._v(" "),e("p",[t._v("就是避免从0开始学习的方法。"),e("strong",[t._v("依靠少量的样本完成推理")]),t._v("。")]),t._v(" "),e("p",[t._v("我们希望它能够从之前的经验快速地学习新的技能，而不是把新的任务孤立地考虑。这个方法，我们称为元学习（learning to learn,或meta learning）, 使得我们的系统在它的整个生命周期中可以持续地学习各种各样的任务。")]),t._v(" "),e("p",[t._v("meta learning是机器学习的一个子领域，它自动学习一些应用于机器学习实验的元数据，主要目的是使用这些元数据来自动学习如何在解决不同类型的学习问题时变得灵活，从而提高现有的学习算法。灵活性是非常重要的，因为"),e("strong",[t._v("每个学习算法都是基于一组有关数据的假设，即它是归纳偏(bias)的。"),e("strong",[t._v("这意味着如果bias与学习问题中的数据相匹配，那么学习就会很好。学习算法在一个学习问题上表现得非常好，但在下一个学习问题上表现得非常糟糕。这对机器学习或数据挖掘技术的使用造成了很大的限制，因为学习问题与不同学习算法的有效性之间的关系尚不清楚。\n通过使用不同类型的元数据，如学习问题的属性，算法属性（如性能测量）或从之前数据推导出的模式，可以选择、更改或组合不同的学习算法，以有效地解决给定的学习问题。\n元学习一般有两级，第一级是")]),t._v("快速地获得每个任务中的知识")]),t._v("，第二级是较慢地"),e("strong",[t._v("提取所有任务中学到的信息")]),t._v("。下面从不同角度解释了元学习的方法")]),t._v(" "),e("ul",[e("li",[t._v("通过知识诱导来表达每种学习方法如何在不同的学习问题上执行，从而发现元知识。元数据是由学习问题中的数据特征（一般的，统计的，信息论的......）以及学习算法的特征（类型，参数设置，性能测量...）形成的。然后，另一个学习算法学习数据特征如何与算法特征相关。给定一个新的学习问题，测量数据特征，并且可以预测不同学习算法的性能。因此，至少在诱导关系成立的情况下，可以选择最适合新问题的算法。")]),t._v(" "),e("li",[t._v("stacking. 通过组合一些（不同的）学习算法，即堆叠泛化。元数据是由这些不同算法的预测而形成的。然后，另一个学习算法从这个元数据中学习，以预测哪些算法的组合会给出好的结果。在给定新的学习问题的情况下，所选择的一组算法的预测被组合（例如通过加权投票）以提供最终的预测。由于每种算法都被认为是在一个问题子集上工作，所以希望这种组合能够更加灵活，并且能够做出好的预测。")]),t._v(" "),e("li",[t._v("boosting. 多次使用相同的算法，训练数据中的示例在每次运行中获得不同的权重。这产生了不同的预测，每个预测都集中于正确预测数据的一个子集，并且结合这些预测导致更好（但更昂贵）的结果。")]),t._v(" "),e("li",[t._v("动态偏选择(Dynamic bias selection)通过改变学习算法的感应偏来匹配给定的问题。这通过改变学习算法的关键方面来完成，例如假设表示，启发式公式或参数。")]),t._v(" "),e("li",[t._v("learning to learn，研究如何随着时间的推移改进学习过程。元数据由关于以前的学习事件的知识组成，并被用于高效地开发新任务的有效假设。"),e("strong",[t._v("其目标是使用从一个领域获得的知识来帮助其他领域的学习")]),t._v("。")])]),t._v(" "),e("p",[t._v("在meta learning中，我们在训练集上训练一个训练过程(meta learner)来生产生一个分类器（learner）使得learner在测试集上获得高的精度。如下图")]),t._v(" "),e("h3",{attrs:{id:"递归记忆模型-memory-augmented-neural-networks-5"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#递归记忆模型-memory-augmented-neural-networks-5"}},[t._v("#")]),t._v(" 递归记忆模型 （Memory-Augmented Neural Networks） "),e("a",{attrs:{href:"#refer-anchor-5"}},[e("sup",[t._v("5")])])]),t._v(" "),e("p",[t._v("文章基于神经网络图灵机（NTMs）的思想，因为NTMs能通过外部存储（external memory）进行短时记忆，并能通过缓慢权值更新来进行长时记忆，NTMs可以学习将表达存入记忆的策略，并如何用这些表达来进行预测。由此，文章方法可以快速准确地预测那些只出现过一次的数据。文章基于LSTM等RNN的模型，将数据看成序列来训练，在测试时输入新的类的样本进行分类。具体地，网络的输入把上一次的y (label)也作为输入，并且添加了external memory存储上一次的x输入，这使得下一次输入后进行反向传播时，可以让y (label)和x建立联系，使得之后的x能够通过外部记忆获取相关图像进行比对来实现更好的预测。这里的RNN就是meta-learner。")]),t._v(" "),e("h3",{attrs:{id:"优化器学习-meta-learning-lstm-6"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#优化器学习-meta-learning-lstm-6"}},[t._v("#")]),t._v(" 优化器学习  （meta-learning LSTM） "),e("a",{attrs:{href:"#refer-anchor-6"}},[e("sup",[t._v("6")])])]),t._v(" "),e("p",[t._v("这些梯度优化算法包括momentum, adagrad, adadelta, ADAM等，无法在几步内完成优化，特别是在非凸的问题上，多种超参的选取无法保证收敛的速度。其次，不同任务分别随机初始化会影响任务收敛到好的解上。虽然finetune这种迁移学习能缓解这个问题，但**当新数据相对原始数据偏差比较大时，迁移学习的性能会大大下降。**我们需要一个系统的学习通用初始化，使得训练从一个好的点开始，它和迁移学习不同的是，它能保证该初始化能让finetune从一个好的点开始。")]),t._v(" "),e("p",[t._v("**文章学习的是一个模新参数的更新函数或更新规则。**它不是在多轮的episodes学习一个单模型，而是在每个episode学习特定的模型。具体地，学习基于梯度下降的参数更新算法，采用LSTM表达meta learner，用其状态表达目标分类器的参数的更新，最终学会如何在新的分类任务上，对分类器网络(learner)进行初始化和参数更新。这个优化算法同时考虑一个任务的短时知识和跨多个任务的长时知识。文章设定目标为通过少量的迭代步骤捕获优化算法的泛化能力，由此meta learner可以训练让learner在每个任务上收敛到一个好的解。另外，通过捕获所有任务之前共享的基础知识，进而更好地初始化learner。")]),t._v(" "),e("h3",{attrs:{id:"模型无关自适应-model-agnostic-7"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#模型无关自适应-model-agnostic-7"}},[t._v("#")]),t._v(" 模型无关自适应（Model-Agnostic） "),e("a",{attrs:{href:"#refer-anchor-7"}},[e("sup",[t._v("7")])])]),t._v(" "),e("p",[t._v("meta learning 的目标是在各种不同的学习任务上学出一个模型，使得可以仅用少量的样本就能解决一些新的学习任务。这种任务的挑战是模型需要结合之前的经验和当前新任务的少量样本信息，并避免在新数据上过拟合。")]),t._v(" "),e("p",[e("strong",[t._v("这个方法无需关心模型的形式")]),t._v("，也不需要为meta learning增加新的参数，"),e("strong",[t._v("直接用梯度下降来训练learner")]),t._v("。")]),t._v(" "),e("p",[t._v("文章的核心思想是"),e("font",{attrs:{color:"green"}},[e("strong",[t._v("学习模型的初始化参数使得在一步或几步迭代后在新任务上的精度最大化")])]),t._v("。它学的不是模型参数的更新函数或是规则，它不局限于参数的规模和模型架构（比如用RNN或siamese）。"),e("strong",[t._v("它本质上也是学习一个好的特征")]),t._v("使得可以适合很多任务（包括分类、回归、增强学习），并通过fine-tune来获得好的效果。")],1),t._v(" "),e("p",[t._v("文章提出的方法，可以学习任意标准模型的参数，并让该模型能快速适配。方法认为，一些中间表达更加适合迁移，比如神经网络的内部特征。因此面向泛化性的表达是有益的。")]),t._v(" "),e("p",[t._v("是要找到一些对任务变化敏感的参数，使得当改变梯度方向，小的参数改动也会产生较大的loss.")]),t._v(" "),e("h1",{attrs:{id:"长尾分布问题"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#长尾分布问题"}},[t._v("#")]),t._v(" 长尾分布问题")]),t._v(" "),e("p",[t._v("https://zhuanlan.zhihu.com/p/158638078")]),t._v(" "),e("p",[t._v("1）重采样（re-sampling）相关")]),t._v(" "),e("p",[t._v("2）重加权（re-weighting）相关")]),t._v(" "),e("p",[t._v("3）迁移学习（transfer learning）相关")]),t._v(" "),e("h1",{attrs:{id:"nlp领域-文本增强"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#nlp领域-文本增强"}},[t._v("#")]),t._v(" NLP领域-文本增强")]),t._v(" "),e("h2",{attrs:{id:"back-translation"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#back-translation"}},[t._v("#")]),t._v(" back-translation")]),t._v(" "),e("p",[t._v("标注文本->翻译->在翻译回来。")]),t._v(" "),e("h2",{attrs:{id:"eda-easy-data-augmentation"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#eda-easy-data-augmentation"}},[t._v("#")]),t._v(" EDA-easy data augmentation")]),t._v(" "),e("p",[t._v("四种操作：")]),t._v(" "),e("ol",[e("li",[t._v("同义词替换：随机选n个同义词替换，(非停用词)")]),t._v(" "),e("li",[t._v("随机插入：插入文本中某个非停用词的同义词")]),t._v(" "),e("li",[t._v("随机交换：")]),t._v(" "),e("li",[t._v("随机删除：按概率p随机删除")])]),t._v(" "),e("p",[t._v("增强值$\\alpha$")]),t._v(" "),e("p",[t._v("没论证是否能够保证标签不变。只是通过一个图显示影响不大")]),t._v(" "),e("p",[t._v("t-SNE 降维度")]),t._v(" "),e("p",[t._v("数据量大，提升效果不好。预训练复杂模型效果可能不行。")]),t._v(" "),e("p",[t._v("数据量少的时候，可能有几个点的提升")]),t._v(" "),e("h2",{attrs:{id:"contextual-augmentation"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#contextual-augmentation"}},[t._v("#")]),t._v(" Contextual Augmentation")]),t._v(" "),e("ol",[e("li",[e("p",[t._v("使用语言模型进行文本的替换")]),t._v(" "),e("ol",[e("li",[t._v("语言模型：用语言模型评价一句话是否合理或是人话")]),t._v(" "),e("li",[t._v("数学上讲：P（合理句子）>P（不合理句子）")]),t._v(" "),e("li",[t._v("用文本中前n个字预测下一个字")])])]),t._v(" "),e("li",[e("p",[t._v("语言模型结构：双向LSTM")])]),t._v(" "),e("li",[e("p",[t._v("修改训练目标融入标签信息")])]),t._v(" "),e("li",[e("p",[t._v("利用了语境信息")])])]),t._v(" "),e("p",[t._v("现有的方法：基于word-Net库，不好")]),t._v(" "),e("p",[t._v("把标签也序列化，融入到训练过程中，从而保证替换后不会对原有的标签有损伤。")]),t._v(" "),e("h2",{attrs:{id:"conditional-bert"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#conditional-bert"}},[t._v("#")]),t._v(" Conditional Bert")]),t._v(" "),e("ol",[e("li",[t._v("使用Bert模型结构")])]),t._v(" "),e("h2",{attrs:{id:"lambada"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#lambada"}},[t._v("#")]),t._v(" LAMBADA")]),t._v(" "),e("p",[t._v("Do Not Have Enough Data? Deep Learning to the Rescue!")]),t._v(" "),e("p",[t._v("基于generative pre-training 2（GPT2）")]),t._v(" "),e("p",[t._v("GPT也是深层的transformer模型，更复杂，深度更深")]),t._v(" "),e("img",{staticStyle:{zoom:"50%"},attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210712173432884.png",alt:"image-20210712173432884"}}),t._v(" "),e("h2",{attrs:{id:"uda-半监督学习"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#uda-半监督学习"}},[t._v("#")]),t._v(" UDA-半监督学习")]),t._v(" "),e("p",[t._v("半监督学习：如何结合有标注数据，直接利用无标签数据")]),t._v(" "),e("p",[t._v("数据增强是：如花使用有标注数据，构造更多有标签数据")]),t._v(" "),e("h3",{attrs:{id:"平滑假设"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#平滑假设"}},[t._v("#")]),t._v(" 平滑假设")]),t._v(" "),e("ol",[e("li",[e("p",[t._v("如果两个输入样本相似，那么模型输出结果也应当相似")])]),t._v(" "),e("li",[e("p",[t._v("对样本做某种很小的扰动，得到x2")])]),t._v(" "),e("li",[e("p",[t._v("训练目标："),e("strong",[t._v("调整模型w，使得w1、w2接近")])])]),t._v(" "),e("li",[e("p",[t._v("在这个过程中，y1和y2的实际值并不重要")])])]),t._v(" "),e("h1",{attrs:{id:"参考"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#参考"}},[t._v("#")]),t._v(" 参考")]),t._v(" "),e("div",{attrs:{id:"refer-anchor-1"}}),t._v(" "),e("ul",[e("li",[t._v("[1] G Koch, R Zemel, and R Salakhutdinov. Siamese neural networks for one-shot image recognition. In ICML Deep Learning workshop, 2015.")])]),t._v(" "),e("div",{attrs:{id:"refer-anchor-2"}}),t._v(" "),e("ul",[e("li",[t._v("[2] Oriol Vinyals, Charles Blundell, Tim Lillicrap, Daan Wierstra, et al. Matching networks for one shot learning. In Advances in Neural Information Processing Systems, pages 3630–3638, 2016.")])]),t._v(" "),e("div",{attrs:{id:"refer-anchor-3"}}),t._v(" "),e("ul",[e("li",[t._v("[3] Jake Snell, Kevin Swersky, and Richard S Zemel. Prototypical networks for few-shot learning. arXiv preprint arXiv:1703.05175, 2017.")])]),t._v(" "),e("div",{attrs:{id:"refer-anchor-4"}}),t._v(" "),e("ul",[e("li",[t._v("[4] Victor Garcia, Joan Bruna. Few-shot learning with graph neural networs. Under review as a conference paper at ICLR 2018.")])]),t._v(" "),e("div",{attrs:{id:"refer-anchor-5"}}),t._v(" "),e("ul",[e("li",[t._v("[5] Santoro, Adam, Bartunov, Sergey, Botvinick, Matthew, Wierstra, Daan, and Lillicrap, Timothy. Meta-learning with memory-augmented neural networks. In International Conference on Machine Learning (ICML), 2016.")])]),t._v(" "),e("div",{attrs:{id:"refer-anchor-6"}}),t._v(" "),e("ul",[e("li",[t._v("[6] Ravi, Sachin and Larochelle, Hugo. Optimization as a model for few-shot learning. In International Conference on Learning Representations (ICLR), 2017.")])])],1)}),[],!1,null,null,null);a.default=s.exports}}]);