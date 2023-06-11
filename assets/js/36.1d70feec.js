(window.webpackJsonp=window.webpackJsonp||[]).push([[36],{652:function(s,e,a){"use strict";a.r(e);var t=a(3),r=Object(t.a)({},(function(){var s=this,e=s.$createElement,a=s._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":s.$parent.slotKey}},[a("Boxx",{attrs:{changeTime:"10000"}}),s._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"title"},[s._v("前言")]),a("p",[s._v("Hive")])]),s._v(" "),a("h1",{attrs:{id:"hive"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hive"}},[s._v("#")]),s._v(" Hive")]),s._v(" "),a("ol",[a("li",[s._v("Hive不是一个计算引擎，它是将SQL语句转成其他计算引擎的执行方式，如Mapreduce和Spark")]),s._v(" "),a("li",[s._v("Hive 通过sql的方式进行操作，Hive只是一个吧sql语句转换成mr任务的SQL解析引擎\n"),a("ol",[a("li",[s._v("实际上在hadoop中，Hbase是分布式数据库")]),s._v(" "),a("li",[s._v("实际上可以直接用sql操作hbase，就是利用hive实现的")])])]),s._v(" "),a("li",[s._v("Hive中数据表实际上只是一个纯逻辑表，只有表的定义，而没有表的数据。")]),s._v(" "),a("li",[s._v("hive中的sql和传统数据库中的sql区别：\na. hive中的sql不支持改写和删除\nb. hive的sql支持很多的扩展能力，\ni. UDF（用户自定义函数）--1vs1的服务，通常用来做格式化处理，map\nii. UDAF：用户自定义聚合函数--多vs1的服务，需要与groupby联合使用\niii. UDTF：用户自定义表生成函数：1对多\nc. 数据检查方式不一样\ni. Hive sql：读时模式\n1) 只有读的时候，才检查数据（类型：字段缺失）\n2) 写的时候，不检查数据。\nii. 传统SQL：写时模式\n1) 写的时候做检查，为了提升查询的性能")]),s._v(" "),a("li",[s._v("Hive的体系架构\n"),a("ol",[a("li",[s._v("用户接口 client：终端，页面，dashboard")]),s._v(" "),a("li",[s._v("语句转换：将用户的sql编译转化成Mapreduce进行执行，driver是整个hive的核心")]),s._v(" "),a("li",[s._v("数据存储：元数据（表的结构定义）的存储和实际数据（HDFS）的存储\n"),a("ol",[a("li",[s._v("元数据：metastore：不存储在hive中，而是存储在第三方数据库（mysql）中")]),s._v(" "),a("li",[s._v("默认 derby（不建议用，单用户模式）")]),s._v(" "),a("li",[s._v("建议MYSQL：多用户模式")]),s._v(" "),a("li",[s._v("本地")]),s._v(" "),a("li",[s._v("远程--MYSQL这个集群可以和HDFS完全不影响")])])])])]),s._v(" "),a("li",[s._v("Hive的数据管理--使用者必须会\n"),a("ol",[a("li",[s._v("Table：内表\n"),a("ol",[a("li",[s._v("表格的创建和数据的加载，分为2个不同的过程")]),s._v(" "),a("li",[s._v("可以用一条语句统一的完成")]),s._v(" "),a("li",[s._v("当删除表格的时候，"),a("strong",[s._v("数据是一起被删除的，所以工作中建议用外表")])])])]),s._v(" "),a("li",[s._v("External Table\n"),a("ol",[a("li",[s._v("删除表格的时候，数据是不删除的")])])]),s._v(" "),a("li",[s._v("Partition（加快检索速度）\n"),a("ol",[a("li",[a("p",[s._v("辅助查询，为了缩小查询范围，加快查询速度")])]),s._v(" "),a("li",[a("p",[s._v("是一个优化手段，不一定所有的数据都加partition。粒度不能太细，建议选择有穷集合，否则创建的partition的文件夹太多")])])])])])])]),s._v(" "),a("div",{staticClass:"language-mermaid line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("graph LR\nA1[Partition] --\x3eB1(0801 目录)\nA2[Partition] --\x3eB2(0801 目录)\nA3[Partition] --\x3eB3(0801 目录)\n    B2 --\x3e C[select xxx from table where dt=8082]\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br"),a("span",{staticClass:"line-number"},[s._v("3")]),a("br"),a("span",{staticClass:"line-number"},[s._v("4")]),a("br"),a("span",{staticClass:"line-number"},[s._v("5")]),a("br")])]),a("ol",{attrs:{start:"4"}},[a("li",[a("p",[s._v("Bucket（不一定会用，知道）设置输出part文件的个数，以及基于part文件怎么进行采样")]),s._v(" "),a("ol",[a("li",[s._v("控制part分区个数、类似于MR中的task_reduce_num")]),s._v(" "),a("li",[s._v("采样（数据非常多。采样方式）")])])]),s._v(" "),a("li",[a("p",[s._v("优化（不一定用）\n8. 控制map和reduce个数")]),s._v(" "),a("ol",{attrs:{start:"9"}},[a("li",[a("p",[s._v("Group By，避免产生一个reduce的情况")]),s._v(" "),a("ol",[a("li",[a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("Select userid,count(*) from table group by userid;---只会产生多个reduce\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])])]),s._v(" "),a("li",[a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("Select userid,count(*) from table;---只会产出一个reduce\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])])])])]),s._v(" "),a("li",[a("p",[s._v("在做join的时候如果不加on或者无效on， 也会产生一个reduce执行的情况（慢）")]),s._v(" "),a("div",{staticClass:"language-sql line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-sql"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("Select")]),s._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("From")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("table")]),s._v(" A\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("Join")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("table")]),s._v(" B"),a("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#on 只有Join的时候可这样代替")]),s._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("where")]),s._v(" A"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("userid "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),s._v(" B"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("userid\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br"),a("span",{staticClass:"line-number"},[s._v("3")]),a("br"),a("span",{staticClass:"line-number"},[s._v("4")]),a("br")])])]),s._v(" "),a("li",[a("p",[s._v("Partition 是个非常有效果的优化手段")])]),s._v(" "),a("li",[a("p",[s._v("Join，适合大小表关联")])]),s._v(" "),a("li",[a("p",[s._v("大表关联（看一下大表的结构，可以删除掉大量的bot ID）")])])])])]),s._v(" "),a("hr"),s._v(" "),a("h1",{attrs:{id:"_2021重新认知"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_2021重新认知"}},[s._v("#")]),s._v(" 2021重新认知")]),s._v(" "),a("hr"),s._v(" "),a("p",[s._v("第三个优化点：")]),s._v(" "),a("p",[s._v("慎用distinct，改用"),a("strong",[s._v("group by")]),s._v("。")]),s._v(" "),a("div",{staticClass:"language-sql line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-sql"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("Select")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[s._v("count")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("distinct")]),s._v(" uid"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" user_sum "),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("from")]),s._v(" movies\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("p",[a("strong",[s._v("distinct")]),s._v("和"),a("strong",[s._v("order by")]),s._v(" 只启动一个reduce。")]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("select count(t1.uid) from (select uid from movies group by uid)t1\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("p",[s._v("括号部分就是启用多个reduce来处理一个计算任务。")]),s._v(" "),a("p",[s._v("第四个优化点：小文件，但不用管。小文件越多，map个数也越多，每一个map都会开启一个java虚拟机。")]),s._v(" "),a("p",[s._v("hive 中每个map最大输入大小是256M。")]),s._v(" "),a("blockquote",[a("p",[s._v("为啥选orc和par，优缺点？")])]),s._v(" "),a("p",[s._v("ORC 和 Parquet 都是 Hadoop 生态系统中流行的开源列文件存储格式，在效率和速度方面非常相似，最重要的是，它们旨在加快大数据分析工作负载。")]),s._v(" "),a("p",[a("code",[s._v("ORC (Optimized Row Columnar)")]),s._v("，是专为 Hadoop 工作负载设计的免费开源列存储格式。")]),s._v(" "),a("p",[a("code",[s._v("Parquet")]),s._v(" 是 Cloudera 与 Twitter 合作支持的 Hadoop 生态系统中另一种面向开源列的文件格式。")]),s._v(" "),a("p",[s._v("ORC 和 Parquet 都是 Hadoop 生态系统中最流行的两种面向列的文件存储格式，旨在很好地处理数据分析工作负载。\nParquet 由 Cloudera 和 Twitter 共同开发，用于存储具有高列的大型数据集的问题。\nORC 是传统 RCFile 规范的后续产品，存储在 ORC 文件格式中的数据被组织成条带，这些条带高度优化了 HDFS 读取操作。\n如果您在 Hadoop 生态系统中使用多种工具，则 Parquet 在适应性方面是一个更好的选择。\nParquet更好地优化了与Spark的使用，而 ORC 则针对Hive进行了优化。但在大多数情况下，两者非常相似，两者之间没有显著差异。")]),s._v(" "),a("p",[s._v("Join优化：")]),s._v(" "),a("p",[a("img",{attrs:{src:"C:%5CUsers%5Ckaich%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210612224702223.png",alt:"image-20210612224702223"}})]),s._v(" "),a("h1",{attrs:{id:"tf-idf"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#tf-idf"}},[s._v("#")]),s._v(" TF-IDF")]),s._v(" "),a("p",[s._v("文章中有4个词，共10个词：TF-> 4/10=0.4")]),s._v(" "),a("p",[s._v("LDA：主题分析")]),s._v(" "),a("h1",{attrs:{id:"学习网站推荐"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#学习网站推荐"}},[s._v("#")]),s._v(" 学习网站推荐")]),s._v(" "),a("ol",[a("li",[s._v("刘建平(理论讲的很好):")])]),s._v(" "),a("p",[s._v("https://www.cnblogs.com/pinard/")]),s._v(" "),a("ol",{attrs:{start:"2"}},[a("li",[s._v("lda数学八卦pdf")]),s._v(" "),a("li",[s._v("刘焕勇")]),s._v(" "),a("li",[s._v("模范的")])]),s._v(" "),a("p",[s._v("Es框架，搜索，内部很生硬，大厂很少用")])],1)}),[],!1,null,null,null);e.default=r.exports}}]);