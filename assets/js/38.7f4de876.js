(window.webpackJsonp=window.webpackJsonp||[]).push([[38],{651:function(a,t,v){"use strict";v.r(t);var _=v(3),s=Object(_.a)({},(function(){var a=this,t=a.$createElement,v=a._self._c||t;return v("ContentSlotsDistributor",{attrs:{"slot-key":a.$parent.slotKey}},[v("Boxx",{attrs:{changeTime:"10000"}}),a._v(" "),v("div",{staticClass:"custom-block tip"},[v("p",{staticClass:"title"},[a._v("前言")]),v("p",[a._v("实验的一些东西。")])]),a._v(" "),v("p",[a._v("文本增强任务：-宋学林")]),a._v(" "),v("h1",{attrs:{id:"常用文本增强技术-eda、uda、cbert、lambada"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#常用文本增强技术-eda、uda、cbert、lambada"}},[a._v("#")]),a._v(" 常用文本增强技术：EDA、UDA、CBert、Lambada")]),a._v(" "),v("p",[a._v("低资源和冷启动问题")]),a._v(" "),v("h1",{attrs:{id:"数据"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#数据"}},[a._v("#")]),a._v(" 数据")]),a._v(" "),v("p",[a._v("标注图片费时费力；数据标注难免有错误")]),a._v(" "),v("p",[a._v("数据增强任务：Data augmentation")]),a._v(" "),v("p",[a._v("少量数据--(Data augmentation)->大量标注数据")]),a._v(" "),v("h2",{attrs:{id:"图像领域"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#图像领域"}},[a._v("#")]),a._v(" 图像领域")]),a._v(" "),v("p",[a._v("评议、旋转、剪裁、遮挡、反转、放缩、灰度")]),a._v(" "),v("p",[a._v("gnn")]),a._v(" "),v("h2",{attrs:{id:"语音"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#语音"}},[a._v("#")]),a._v(" 语音")]),a._v(" "),v("p",[a._v("音速扰动、音量扰动")]),a._v(" "),v("p",[a._v("频率遮蔽、时间遮蔽")]),a._v(" "),v("p",[a._v("加入噪声")]),a._v(" "),v("h2",{attrs:{id:"文本增强"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#文本增强"}},[a._v("#")]),a._v(" 文本增强")]),a._v(" "),v("p",[a._v("难度>图像和语音领域")]),a._v(" "),v("p",[a._v("微小的改动可能带来语义的改变")]),a._v(" "),v("p",[a._v("文本增强的核心：改变文本内容&保持标签不变")]),a._v(" "),v("h1",{attrs:{id:"具体方法"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#具体方法"}},[a._v("#")]),a._v(" 具体方法")]),a._v(" "),v("h2",{attrs:{id:"back-translation"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#back-translation"}},[a._v("#")]),a._v(" back-translation")]),a._v(" "),v("p",[a._v("标注文本->翻译->在翻译回来。")]),a._v(" "),v("h2",{attrs:{id:"eda-easy-data-augmentation"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#eda-easy-data-augmentation"}},[a._v("#")]),a._v(" EDA-easy data augmentation")]),a._v(" "),v("p",[a._v("四种操作：")]),a._v(" "),v("ol",[v("li",[a._v("同义词替换：随机选n个同义词替换，(非停用词)")]),a._v(" "),v("li",[a._v("随机插入：插入文本中某个非停用词的同义词")]),a._v(" "),v("li",[a._v("随机交换：")]),a._v(" "),v("li",[a._v("随机删除：按概率p随机删除")])]),a._v(" "),v("p",[a._v("增强值$\\alpha$")]),a._v(" "),v("p",[a._v("没论证是否能够保证标签不变。只是通过一个图显示影响不大")]),a._v(" "),v("p",[a._v("t-SNE 降维度")]),a._v(" "),v("p",[a._v("数据量大，提升效果不好。预训练复杂模型效果可能不行。")]),a._v(" "),v("p",[a._v("数据量少的时候，可能有几个点的提升")]),a._v(" "),v("h2",{attrs:{id:"contextual-augmentation"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#contextual-augmentation"}},[a._v("#")]),a._v(" Contextual Augmentation")]),a._v(" "),v("ol",[v("li",[v("p",[a._v("使用语言模型进行文本的替换")]),a._v(" "),v("ol",[v("li",[a._v("语言模型：用语言模型评价一句话是否合理或是人话")]),a._v(" "),v("li",[a._v("数学上讲：P（合理句子）>P（不合理句子）")]),a._v(" "),v("li",[a._v("用文本中前n个字预测下一个字")])])]),a._v(" "),v("li",[v("p",[a._v("语言模型结构：双向LSTM")])]),a._v(" "),v("li",[v("p",[a._v("修改训练目标融入标签信息")])]),a._v(" "),v("li",[v("p",[a._v("利用了语境信息")])])]),a._v(" "),v("p",[a._v("现有的方法：基于word-Net库，不好")]),a._v(" "),v("p",[a._v("把标签也序列化，融入到训练过程中，从而保证替换后不会对原有的标签有损伤。")]),a._v(" "),v("h2",{attrs:{id:"conditional-bert"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#conditional-bert"}},[a._v("#")]),a._v(" Conditional Bert")]),a._v(" "),v("ol",[v("li",[a._v("使用Bert模型结构")])]),a._v(" "),v("p",[a._v("bert-预训练")]),a._v(" "),v("ol",[v("li",[a._v("mask language model；依照一定的概率，用mask掩盖文本中的某个词，用剩下的预测这个mask的词")]),a._v(" "),v("li",[a._v("next sentence prediction：挨着的两句话为正样本，不挨着的为负样本")])]),a._v(" "),v("p",[v("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706211307471.png",alt:"image-20210706211307471"}})]),a._v(" "),v("p",[v("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706211444673.png",alt:"image-20210706211444673"}})]),a._v(" "),v("p",[v("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706211637664.png",alt:"image-20210706211637664"}})]),a._v(" "),v("p",[a._v("Conditional bert data augmentation")]),a._v(" "),v("p",[a._v("如何保证标签不变？")]),a._v(" "),v("p",[a._v("把label 加入到输入中")]),a._v(" "),v("p",[a._v("训练的时候是使用的bert的结构")]),a._v(" "),v("h2",{attrs:{id:"lambada"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#lambada"}},[a._v("#")]),a._v(" LAMBADA")]),a._v(" "),v("p",[a._v("Do Not Have Enough Data? Deep Learning to the Rescue!")]),a._v(" "),v("p",[a._v("基于generative pre-training 2（GPT2）")]),a._v(" "),v("p",[a._v("GPT也是深层的transformer模型，更复杂，深度更深")]),a._v(" "),v("p",[a._v("GPT-写东西的能力很强")]),a._v(" "),v("p",[v("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706215121764.png",alt:"image-20210706215121764"}})]),a._v(" "),v("p",[v("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706215153910.png",alt:"image-20210706215153910"}})]),a._v(" "),v("h1",{attrs:{id:"uda-半监督学习"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#uda-半监督学习"}},[a._v("#")]),a._v(" UDA-半监督学习")]),a._v(" "),v("p",[a._v("半监督学习：如何结合有标注数据，直接利用无标签数据")]),a._v(" "),v("p",[a._v("数据增强是：如花使用有标注数据，构造更多有标签数据")]),a._v(" "),v("h2",{attrs:{id:"平滑假设"}},[v("a",{staticClass:"header-anchor",attrs:{href:"#平滑假设"}},[a._v("#")]),a._v(" 平滑假设")]),a._v(" "),v("ol",[v("li",[v("p",[a._v("如果两个输入样本相似，那么模型输出结果也应当相似")])]),a._v(" "),v("li",[v("p",[a._v("对样本做某种很小的扰动，得到x2")])]),a._v(" "),v("li",[v("p",[a._v("训练目标：调整模型w，使得w1、w2接近")])]),a._v(" "),v("li",[v("p",[a._v("在这个过程中，y1和y2的实际值并不重要")]),a._v(" "),v("p",[v("img",{attrs:{src:"https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210706220236419.png",alt:"image-20210706220236419"}})])])])],1)}),[],!1,null,null,null);t.default=s.exports}}]);