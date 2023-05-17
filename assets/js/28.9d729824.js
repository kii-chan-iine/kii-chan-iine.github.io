(window.webpackJsonp=window.webpackJsonp||[]).push([[28],{641:function(a,s,t){"use strict";t.r(s);var e=t(3),i=Object(e.a)({},(function(){var a=this,s=a.$createElement,t=a._self._c||s;return t("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[t("Boxx",{attrs:{changeTime:"10000"}}),a._v(" "),t("div",{staticClass:"custom-block tip"},[t("p",{staticClass:"title"},[a._v("前言")]),t("p",[a._v("通过某种方法使得不同类别的样本对于模型学习中的Loss（或梯度）贡献是比较均衡的。具体可以从"),t("strong",[a._v("数据样本、模型算法、目标函数、评估指标")]),a._v("等方面进行优化，其中数据增强、代价敏感学习及采样+集成学习是比较常用的，效果也是比较明显的。其实，不均衡问题解决也是"),t("strong",[a._v("结合实际")]),a._v("再做方法选择、组合及调整，在验证中调优的过程。")])]),a._v(" "),t("h1",{attrs:{id:"样本不均衡"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#样本不均衡"}},[a._v("#")]),a._v(" 样本不均衡")]),a._v(" "),t("h2",{attrs:{id:"影响"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#影响"}},[a._v("#")]),a._v(" 影响")]),a._v(" "),t("p",[a._v("具体举个例子，在一个欺诈识别的案例中，好坏样本的占比是1000：1，而如果我们直接拿这个比例去学习模型的话，因为扔进去模型学习的样本大部分都是好的，就很容易学出一个把所有样本都预测为好的模型，而且这样预测的概率准确率还是非常高的。而模型最终学习的并不是如何分辨好坏，而是学习到了”好 远比 坏的多“这样的先验信息，凭着这个信息把所有样本都判定为“好”就可以了。这样就背离了模型学习去分辨好坏的初衷了。")]),a._v(" "),t("blockquote",[t("p",[a._v("EC：对于一般情况问题不大，但是如果场景是挑选次品等关注于少样本的应用时，样本不均匀带来的影响就比较大了。")])]),a._v(" "),t("p",[a._v("所以，样本不均衡带来的根本影响是："),t("strong",[a._v("模型会学习到训练集中样本比例的这种先验性信息")]),a._v("，以致于实际预测时就会对多数类别有侧重（可能导致多数类精度更好，而少数类比较差）。")]),a._v(" "),t("p",[a._v("总结一下也就是，"),t("strong",[a._v("我们通过解决样本不均衡，可以减少模型学习样本比例的先验信息，以获得能学习到辨别好坏本质特征的模型")]),a._v("。")]),a._v(" "),t("h2",{attrs:{id:"判断样本不均衡必要性的场景"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#判断样本不均衡必要性的场景"}},[a._v("#")]),a._v(" 判断样本不均衡必要性的场景")]),a._v(" "),t("p",[a._v("从分类效果出发，通过上面的例子可知，不均衡对于分类结果的影响不一定是不好的，那"),t("strong",[a._v("什么时候需要解决样本不均衡")]),a._v("？")]),a._v(" "),t("ul",[t("li",[t("p",[a._v("判断任务是否复杂："),t("strong",[a._v("复杂度")]),a._v(" 学习任务的复杂度与样本不平衡的敏感度是成正比的（参见《Survey on deep learning with class imbalance》），对于简单线性可分任务，样本是否均衡影响不大。需要注意的是，学习任务的复杂度是"),t("strong",[a._v("相对意义")]),a._v("上的，得从特征强弱、数据噪音情况以及模型容量等方面综合评估。")])]),a._v(" "),t("li",[t("p",[a._v("判断训练样本的分布与真实样本"),t("strong",[a._v("分布是否一致且稳定")]),a._v("，如果分布是一致的，带着这种正确点的先验对预测结果影响不大。但是，还需要考虑到，如果后面真实样本分布变了，这个样本比例的先验就有副作用了。")])]),a._v(" "),t("li",[t("p",[a._v("判断是否出现某一类别样本数目非常稀少的情况，这时模型很有可能学习不好，类别不均衡是需要解决的，如选择一些数据增强的方法，或者尝试如异常检测的单分类模型。")])])]),a._v(" "),t("blockquote",[t("p",[a._v("Summary:")]),a._v(" "),t("ol",[t("li",[a._v("复杂度高的学习任务、训练样本与总体分布不一致必须要考虑样本不均衡")])])]),a._v(" "),t("h1",{attrs:{id:"解决方案"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#解决方案"}},[a._v("#")]),a._v(" 解决方案")]),a._v(" "),t("p",[a._v("基本上，在学习任务有些难度的前提下，不均衡解决方法可以归结为："),t("strong",[a._v("通过某种方法使得不同类别的样本对于模型学习中的Loss（或梯度）贡献是比较均衡的")]),a._v("。以消除模型对不同类别的偏向性，学习到更为本质的特征。本文从"),t("strong",[a._v("数据样本、模型算法、目标（损失）函数、评估指标")]),a._v("等方面，对个中的解决方法进行探讨。")]),a._v(" "),t("h2",{attrs:{id:"样本层面"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#样本层面"}},[a._v("#")]),a._v(" 样本层面")]),a._v(" "),t("h3",{attrs:{id:"欠采样、过采样"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#欠采样、过采样"}},[a._v("#")]),a._v(" 欠采样、过采样")]),a._v(" "),t("p",[a._v("最直接的处理方式就是样本数量的调整了，常用的可以：")]),a._v(" "),t("ul",[t("li",[t("p",[a._v("欠采样：减少多数类的数量（如随机欠采样、NearMiss、ENN）。")])]),a._v(" "),t("li",[t("p",[a._v("过采样：尽量多地增加少数类的的样本数量（如随机过采样、以及2.1.2数据增强方法），以达到类别间数目均衡。")])]),a._v(" "),t("li",[t("p",[a._v("还可结合两者做混合采样（如Smote+ENN）。")]),a._v(" "),t("p",[a._v("具体还可以参见【scikit-learn的imbalanced-learn.org/stable/user_guide.html以及github的awesome-imbalanced-learning】")])])]),a._v(" "),t("h3",{attrs:{id:"数据增强"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#数据增强"}},[a._v("#")]),a._v(" 数据增强")]),a._v(" "),t("p",[a._v("数据增强（Data Augmentation）是在不实质性的增加数据的情况下，从原始数据加工出更多数据的表示，提高原数据的数量及质量，以接近于更多数据量产生的价值，从而提高模型的学习效果（"),t("font",{attrs:{color:"blue"}},[a._v("其实也是过采样的方法的一种")]),a._v(")。如下列举常用的方法：")],1),a._v(" "),t("h4",{attrs:{id:"基于样本变换的数据增强"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#基于样本变换的数据增强"}},[a._v("#")]),a._v(" 基于样本变换的数据增强")]),a._v(" "),t("p",[a._v("样本变换数据增强即采用预设的数据变换规则进行已有数据的扩增，包含单样本数据增强和多样本数据增强。"),t("strong",[a._v("单样本增强(主要用于图像)")]),a._v("：主要有"),t("font",{attrs:{color:"red"}},[a._v("几何操作、颜色变换、随机擦除、添加噪声")]),a._v("等方法产生新的样本，可参见imgaug开源库。")],1),a._v(" "),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161002927.png",alt:"image-20220124161002927"}})]),a._v(" "),t("p",[t("strong",[a._v("多样本增强")]),a._v("：是通过组合及转换多个样本，主要有Smote类（可见imbalanced-learn.org/stable/references/over_sampling.html）、SamplePairing、Mixup等方法在特征空间内构造已知样本的邻域值样本。(> 这块看一下<)")]),a._v(" "),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161021861.png",alt:"image-20220124161021861"}})]),a._v(" "),t("h4",{attrs:{id:"基于深度学习的数据增强"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#基于深度学习的数据增强"}},[a._v("#")]),a._v(" 基于深度学习的数据增强")]),a._v(" "),t("p",[a._v("生成模型如变分自编码网络(Variational Auto-Encoding network, VAE)和生成对抗网络(Generative Adversarial Network, GAN)，其生成样本的方法也可以用于数据增强。这种基于网络合成的方法相比于传统的数据增强技术虽然过程更加复杂, 但是生成的样本更加多样。")]),a._v(" "),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161042531.png",alt:"image-20220124161042531"}})]),a._v(" "),t("p",[t("strong",[a._v("数据样本层面解决不均衡的方法，需要关注的是：")])]),a._v(" "),t("ul",[t("li",[a._v("随机欠采样可能会导致丢弃含有重要信息的样本。在计算性能足够下，可以考虑"),t("strong",[a._v("数据的分布信息")]),a._v("（通常是基于距离的邻域关系）的采样方法，如ENN、NearMiss等。")]),a._v(" "),t("li",[a._v("随机过采样或数据增强样本也有可能是强调（或引入）片面噪声，导致过拟合。也可能是引入信息量不大的样本。此时需要考虑的是调整采样方法，或者通过半监督算法(可借鉴Pu-Learning思路)选择增强数据的"),t("strong",[a._v("较优子集")]),a._v("，以提高模型的泛化能力。")])]),a._v(" "),t("h2",{attrs:{id:"损失函数的层面"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#损失函数的层面"}},[a._v("#")]),a._v(" 损失函数的层面")]),a._v(" "),t("p",[a._v("损失函数层面主流的方法也就是常用的代价敏感学习（cost-sensitive），"),t("font",{attrs:{color:"red"}},[a._v("为不同的分类错误给予不同惩罚力度（权重）")]),a._v("，在调节类别平衡的同时，也不会增加计算复杂度。如下常用方法：")],1),a._v(" "),t("h3",{attrs:{id:"class-weight"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#class-weight"}},[a._v("#")]),a._v(" class weight")]),a._v(" "),t("p",[a._v("这最常用也就是scikit模型的’class weight‘方法，"),t("code",[a._v("If ‘balanced’, class weights will be given by n_samples / (n_classes * np.bincount(y)). If a dictionary is given, keys are classes and values are corresponding class weights. If None is given, the class weights will be uniform.")]),a._v("，class weight可以为不同类别的样本提供不同的权重（少数类有更高的权重），从而模型可以平衡各类别的学习。如下图通过为少数类做更高的权重，以避免决策偏重多数类的现象（类别权重除了设定为balanced，还可以作为一个超参搜索。示例代码请见（github.com/aialgorithm）：")]),a._v(" "),t("div",{staticClass:"language- line-numbers-mode"},[t("pre",{pre:!0,attrs:{class:"language-text"}},[t("code",[a._v("clf2 = LogisticRegression(class_weight={0:1,1:10}) # 代价敏感学习\n")])]),a._v(" "),t("div",{staticClass:"line-numbers-wrapper"},[t("span",{staticClass:"line-number"},[a._v("1")]),t("br")])]),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161101003.png",alt:"image-20220124161101003"}})]),a._v(" "),t("h3",{attrs:{id:"ohem-和-focal-loss"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#ohem-和-focal-loss"}},[a._v("#")]),a._v(" OHEM 和 Focal Loss")]),a._v(" "),t("blockquote",[t("p",[a._v("In this work, we first point out that the class imbalance can be summarized to the imbalance in difficulty and the imbalance in difficulty can be summarized to the imbalance in gradient norm distribution.")]),a._v(" "),t("p",[a._v("——原文可见《Gradient Harmonized Single-stage Detector》")])]),a._v(" "),t("p",[a._v("上文的大意是，"),t("strong",[a._v("类别的不平衡可以归结为难易样本的不平衡，而难易样本的不平衡可以归结为梯度的不平衡")]),a._v("。按照这个思路，OHEM和Focal loss都做了两件事：难样本挖掘以及类别的平衡。（另外的有 GHM、 PISA等方法，可以自行了解）")]),a._v(" "),t("ul",[t("li",[a._v("OHEM（Online Hard Example Mining）算法的核心是选择一些hard examples（多样性和高损失的样本）作为训练的样本，针对性地改善模型学习效果。对于数据的类别不平衡问题，OHEM的针对性更强。")]),a._v(" "),t("li",[a._v("Focal loss的核心思想是在交叉熵损失函数（CE）的基础上增加了类别的不同权重以及困难（高损失）样本的权重（如下公式），以改善模型学习效果。")])]),a._v(" "),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161118681.png",alt:"image-20220124161118681"}})]),a._v(" "),t("h2",{attrs:{id:"模型层面"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#模型层面"}},[a._v("#")]),a._v(" 模型层面")]),a._v(" "),t("p",[a._v("模型方面主要是选择一些对不均衡比较不敏感的模型，比如，对比逻辑回归模型（lr学习的是全量训练样本的最小损失，自然会比较偏向去减少多数类样本造成的损失），决策树在不平衡数据上面表现相对好一些，树模型是按照增益递归地划分数据（如下图），划分过程考虑的是局部的增益，全局样本是不均衡，局部空间就不一定，所以比较不敏感一些（但还是会有偏向性）。相关实验可见arxiv.org/abs/2104.02240。")]),a._v(" "),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161135940.png",alt:"image-20220124161135940"}})]),a._v(" "),t("p",[t("font",{attrs:{color:"red"}},[a._v("解决不均衡问题，更为优秀的是基于采样+集成树模型等方法，可以在类别不均衡数据上表现良好")]),a._v("。")],1),a._v(" "),t("h3",{attrs:{id:"采样-集成学习"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#采样-集成学习"}},[a._v("#")]),a._v(" 采样+集成学习")]),a._v(" "),t("p",[a._v("这类方法简单来说，通过重复组合少数类样本与抽样的同样数量的多数类样本，训练若干的分类器进行集成学习。")]),a._v(" "),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161157928.png",alt:"image-20220124161157928"}})]),a._v(" "),t("ul",[t("li",[a._v("BalanceCascade BalanceCascade基于Adaboost作为基分类器，核心思路是在每一轮训练时都使用多数类与少数类数量上相等的训练集，然后使用该分类器对全体多数类进行预测，通过控制分类阈值来控制FP（False Positive）率，将所有判断正确的类删除，然后进入下一轮迭代继续降低多数类数量。")]),a._v(" "),t("li",[a._v("EasyEnsemble EasyEnsemble也是基于Adaboost作为基分类器，就是将多数类样本集随机分成 N 个子集，且每一个子集样本与少数类样本相同，然后分别将各个多数类样本子集与少数类样本进行组合，使用AdaBoost基分类模型进行训练，最后bagging集成各基分类器，得到最终模型。示例代码可见：www.kaggle.com/orange90/ensemble-test-credit-score-model-example")])]),a._v(" "),t("p",[a._v("通常，在数据集"),t("strong",[a._v("噪声较小")]),a._v("的情况下，可以用"),t("strong",[a._v("BalanceCascade")]),a._v("，可以用较少的基分类器数量得到较好的表现（基于串行的集成学习方法，对噪声敏感容易过拟合）。"),t("strong",[a._v("噪声大的情况")]),a._v("下，可以用"),t("strong",[a._v("EasyEnsemble")]),a._v("，基于串行+并行的集成学习方法，bagging多个Adaboost过程可以抵消一些噪声影响。此外还有RUSB、SmoteBoost、balanced RF等其他集成方法可以自行了解。")]),a._v(" "),t("h3",{attrs:{id:"异常检测"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#异常检测"}},[a._v("#")]),a._v(" 异常检测")]),a._v(" "),t("p",[a._v("类别不平衡很极端的情况下（比如少数类只有几十个样本），将分类问题考虑成异常检测（anomaly detection）问题可能会更好。异常检测是通过数据挖掘方法发现与数据集分布不一致的异常数据，也被称为离群点、异常值检测等等。无监督异常检测按其算法思想大致可分为几类：基于聚类的方法、基于统计的方法、基于深度的方法(孤立森林)、基于分类模型（one-class SVM）以及基于神经网络的方法（自编码器AE）等等。")]),a._v(" "),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161215561.png",alt:"image-20220124161215561"}})]),a._v(" "),t("h2",{attrs:{id:"决策及评估指标"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#决策及评估指标"}},[a._v("#")]),a._v(" 决策及评估指标")]),a._v(" "),t("p",[a._v("本节关注的重点是，当我们采用不平衡数据训练模型，如何更好决策以及客观地评估不平衡数据下的模型表现。对于分类常用的precision、recall、F1、混淆矩阵，样本不均衡的不同程度，都会明显改变这些指标的表现。对于类别不均衡下模型的预测，我们可以做分类阈值移动，以调整模型对于不同类别偏好的情况（如模型偏好预测负样本，偏向0，对应的我们的分类阈值也往下调整），"),t("strong",[a._v("达到决策时类别平衡的目的")]),a._v("。这里，通常可以通过P-R曲线，选择到较优表现的阈值。")]),a._v(" "),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161243405.png",alt:"image-20220124161243405"}})]),a._v(" "),t("p",[a._v("对于类别不均衡下的模型评估，可以采用AUC、AUPRC(更优)评估模型表现。AUC的含义是ROC曲线的面积，其数值的物理意义是：随机给定一正一负两个样本，将正样本预测分值大于负样本的概率大小。"),t("strong",[a._v("AUC对样本的正负样本比例情况是不敏感")]),a._v("，即使正例与负例的比例发生了很大变化，ROC曲线面积也不会产生大的变化。")]),a._v(" "),t("p",[t("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20220124161306945.png",alt:"image-20220124161306945"}})]),a._v(" "),t("h2",{attrs:{id:"小结"}},[t("a",{staticClass:"header-anchor",attrs:{href:"#小结"}},[a._v("#")]),a._v(" 小结")]),a._v(" "),t("p",[a._v("不均衡样本解决方法：")]),a._v(" "),t("ol",[t("li",[a._v("数据层面\n"),t("ol",[t("li",[a._v("采样方法：欠采样、过采样、混合采样（如Smote+ENN）")]),a._v(" "),t("li",[a._v("数据增强："),t("strong",[a._v("传统方法")]),a._v("(几何操作、颜色变换、随机擦除、添加噪声)，"),t("strong",[a._v("深度学习")]),a._v("(VAE、GEN)")])])]),a._v(" "),t("li",[a._v("损失函数层面\n"),t("ol",[t("li",[a._v("class weight")]),a._v(" "),t("li",[a._v("OHEM")]),a._v(" "),t("li",[a._v("Focal loss")])])]),a._v(" "),t("li",[a._v("模型层面\n"),t("ol",[t("li",[a._v("逻辑回归、决策树")]),a._v(" "),t("li",[a._v("采样+集成学习：BalanceCascade BalanceCascade (噪声小的时候用可以得到较好的效果)\\EasyEnsemble EasyEnsemble(for big noise)")]),a._v(" "),t("li",[a._v("异常检测：对于特定场景，可以直接将分类模型替换为异常检测模型")])])]),a._v(" "),t("li",[a._v("评估指标层面\n"),t("ol",[t("li",[a._v("P-R曲线")]),a._v(" "),t("li",[a._v("AUC")])])])])],1)}),[],!1,null,null,null);s.default=i.exports}}]);