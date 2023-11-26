(window.webpackJsonp=window.webpackJsonp||[]).push([[40],{656:function(s,a,e){"use strict";e.r(a);var t=e(3),n=Object(t.a)({},(function(){var s=this,a=s.$createElement,e=s._self._c||a;return e("ContentSlotsDistributor",{attrs:{"slot-key":s.$parent.slotKey}},[e("Boxx",{attrs:{changeTime:"10000"}}),s._v(" "),e("div",{staticClass:"custom-block tip"},[e("p",{staticClass:"title"},[s._v("前言")]),e("p",[s._v("学习的过程中的笔记")])]),s._v(" "),e("h1",{attrs:{id:"配置环境问题"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#配置环境问题"}},[s._v("#")]),s._v(" 配置环境问题")]),s._v(" "),e("ol",[e("li",[s._v("虚拟机开机pw：12345678？")]),s._v(" "),e("li",[s._v("ifconfig")]),s._v(" "),e("li",[s._v("修改ip：")])]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("cd /etc/sysconfig/network-scripts/\n\nvim ifcfg-eth0   / Centos7:   vim ifcfg-ens33\n\n# ethh0 是当前使用的网卡/ ifcfg-ens33\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br"),e("span",{staticClass:"line-number"},[s._v("2")]),e("br"),e("span",{staticClass:"line-number"},[s._v("3")]),e("br"),e("span",{staticClass:"line-number"},[s._v("4")]),e("br"),e("span",{staticClass:"line-number"},[s._v("5")]),e("br")])]),e("p",[s._v("这个是学校的DNS,这里配了2个防止无法使用")]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("DNS1=10.10.0.21  \nDNS2=114.114.114.114\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br"),e("span",{staticClass:"line-number"},[s._v("2")]),e("br")])]),e("ol",[e("li",[s._v("重启网络:")])]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("/etc/init.d/network restart\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br")])]),e("ol",[e("li",[e("p",[s._v("主节点master在生产过程中，应当多分配一些内存资源。而对于从节点则需要更多的磁盘开销，应当分配更多的磁盘资源。:CentOS 7,复制从节点后，配置ip后需要移除网卡再添加。这个过程会出现新的ifcfg-ens37网卡，但是这个网卡没有配置文件需要用 mv A B 重命名一下。")]),s._v(" "),e("p",[s._v("补充：一些linux命令")]),s._v(" "),e("div",{staticClass:"language-jsx line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-jsx"}},[e("code",[e("span",{pre:!0,attrs:{class:"token number"}},[s._v("1.")]),s._v(" 使用nmcli con show命令，查看网卡的"),e("span",{pre:!0,attrs:{class:"token constant"}},[s._v("UUID")]),s._v("信息，记下"),e("span",{pre:!0,attrs:{class:"token constant"}},[s._v("UUID")]),s._v("值\n"),e("span",{pre:!0,attrs:{class:"token number"}},[s._v("2.")]),s._v(" 使用ip addr命令查看网卡信息\n"),e("span",{pre:!0,attrs:{class:"token number"}},[s._v("3.")]),s._v(" ifconfig "),e("span",{pre:!0,attrs:{class:"token operator"}},[s._v("/")]),s._v("  ifconfig "),e("span",{pre:!0,attrs:{class:"token operator"}},[s._v("-")]),s._v("a\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br"),e("span",{staticClass:"line-number"},[s._v("2")]),e("br"),e("span",{staticClass:"line-number"},[s._v("3")]),e("br")])]),e("p",[s._v("CentOS7修改hostname")]),s._v(" "),e("div",{staticClass:"language-bash line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-bash"}},[e("code",[s._v("hostnamectl set-hostname master "),e("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#立即切重启也会生效")]),s._v("\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br")])])]),s._v(" "),e("li",[e("p",[s._v("将所用的jave安装文件（bin）格式和hadoop的安装包拷贝到 "),e("code",[s._v("/usr/bin/src/")]),s._v("目录下。")])])]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("#cp * /usr/local/src/\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br")])]),e("ol",[e("li",[e("code",[s._v("ll")]),s._v(",查看权限，是否有可执行权限")]),s._v(" "),e("li",[e("code",[s._v("./xxxx.bin")]),s._v(",安装java")]),s._v(" "),e("li",[s._v("编辑环境变量")])]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("vim ~/.bashrc\n%该文件在/usr/local/src目录下\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br"),e("span",{staticClass:"line-number"},[s._v("2")]),e("br")])]),e("p",[s._v("添加：")]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("export JAVA_HOME=/usr/local/src/jdk1.6.0_45\nexport CLASSPATH=.:CLASSPATH:$JAVA_HOME/lib\nexport PATH=$PATH:$JAVA_HOME/bin\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br"),e("span",{staticClass:"line-number"},[s._v("2")]),e("br"),e("span",{staticClass:"line-number"},[s._v("3")]),e("br")])]),e("p",[s._v("然后退出， 终端下输入")]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("#source ~/.bashrc\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br")])]),e("ol",[e("li",[s._v("查看java的安装位置： "),e("code",[s._v("which java")])]),s._v(" "),e("li",[s._v("这时，回到master的/usr/local/src/下，将文件远程复制到slave的/usr/lovcal/src/目录下")])]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("scp -rp jdk-6u45-linux-x64.bin 192.168.220.129:/usr/local/src/\ncat ~/.bashrc\n#该语句可以查看bashrc的内部的内容\ncat命令是linux下的一个文本输出命令，通常是用于观看某个文件的内容的；\ncat主要有三大功能：\n1.一次显示整个文件。\n$ cat   filename\n2.从键盘创建一个文件。\n$ cat  >  filename\n只能创建新文件,不能编辑已有文件.\n3.将几个文件合并为一个文件。\n$cat   file1   file2  > file\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br"),e("span",{staticClass:"line-number"},[s._v("2")]),e("br"),e("span",{staticClass:"line-number"},[s._v("3")]),e("br"),e("span",{staticClass:"line-number"},[s._v("4")]),e("br"),e("span",{staticClass:"line-number"},[s._v("5")]),e("br"),e("span",{staticClass:"line-number"},[s._v("6")]),e("br"),e("span",{staticClass:"line-number"},[s._v("7")]),e("br"),e("span",{staticClass:"line-number"},[s._v("8")]),e("br"),e("span",{staticClass:"line-number"},[s._v("9")]),e("br"),e("span",{staticClass:"line-number"},[s._v("10")]),e("br"),e("span",{staticClass:"line-number"},[s._v("11")]),e("br"),e("span",{staticClass:"line-number"},[s._v("12")]),e("br")])]),e("h1",{attrs:{id:"hive"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#hive"}},[s._v("#")]),s._v(" HIVE")]),s._v(" "),e("p",[s._v("优化的目的：尽量让多台机器去均匀的计算总的数据")]),s._v(" "),e("ol",[e("li",[e("p",[s._v("如果在一个sql中多次查询一张表，请用from table")])]),s._v(" "),e("li",[e("p",[s._v("慎用count （distinct col），这个尤其是distinct全局去重，只启动一个MR，会造成单个机器压力过大。")])]),s._v(" "),e("li",[e("blockquote",[e("p",[s._v("小文件的处理方式。（在工作中不需要过多的关心，有"),e("strong",[s._v("运维")]),s._v("去处理，可以配置最小核名数据大小，还可以基于orc parquet）")]),s._v(" "),e("p",[s._v("orc， seq合并 par合并")])])]),s._v(" "),e("li",[e("p",[s._v("过滤 where过滤：谓词下推（提前过滤那些可以早早过滤的数据，减少下一个任务的io）；having；distinct")])]),s._v(" "),e("li",[e("p",[s._v("分区过滤：在读数据的时候，直接去指定的分区里面读数据，where不过mapper进程。select * from table（如果在公用平台上，有的是不符合规的，必须加）")]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("select * from table where dt=''\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br")])]),e("p",[s._v("甚至有的大厂，不允许select * 这种写法")])]),s._v(" "),e("li",[e("p",[s._v("列过滤，orc parquet 就是这种格式的存储(列存储)会大大减少读取的数据量。直接用schema信息，读取的时候就可以指定读取的列")])]),s._v(" "),e("li",[e("p",[s._v("count(col)--不统计空值，它会将col序列化和反序列化，==  count(1) where col is not null,count(*),count(1)")])]),s._v(" "),e("li",[e("p",[s._v("默认开启combiner")])]),s._v(" "),e("li",[e("p",[s._v("大表join小表：启动mapjoin，在工作中自己要去显式的开启mapjoin。聚类里面有个加载聚类中心的代码，就是每个mapper调用setup，spark里面的一个广播。就是把每个小表全部加载到每个mapper的内存中去，然后join发生在map中。")])])]),s._v(" "),e("hr"),s._v(" "),e("h2",{attrs:{id:"数据倾斜"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#数据倾斜"}},[s._v("#")]),s._v(" 数据倾斜")]),s._v(" "),e("p",[s._v("不怕数据量大，就怕数据倾斜。")]),s._v(" "),e("p",[s._v("核心点其实就是对应着每个机器干的活不一样多，尤其是1-2个机器干的活太多了。")]),s._v(" "),e("p",[s._v("hash进行分发的，吧key的字段hash到不同的机器上，使得均匀。")]),s._v(" "),e("p",[s._v("如果一个热点新闻（key）里面有大量的数据，分发如果不均匀的话，会全部将这些数据分发到1个reduce上。表现为，99%的reduce完成，一直在刷99%。可能经过很长时间才处理完。")]),s._v(" "),e("div",{staticClass:"language-mermaid line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("graph LR\nA1[Key1-新闻] --\x3eB1(80%-数据量)\nA2[Key2-10000] --\x3eB2(20%)\n\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br"),e("span",{staticClass:"line-number"},[s._v("2")]),e("br"),e("span",{staticClass:"line-number"},[s._v("3")]),e("br"),e("span",{staticClass:"line-number"},[s._v("4")]),e("br")])]),e("p",[s._v("数据倾斜：根本原因：key分布不均。")]),s._v(" "),e("h3",{attrs:{id:"group-by-会导致数据倾斜"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#group-by-会导致数据倾斜"}},[s._v("#")]),s._v(" "),e("strong",[s._v("group by 会导致数据倾斜")]),s._v(" :")]),s._v(" "),e("p",[s._v("因为group by会分不同的字段，但这些字段不一定会均匀的分开数据。")]),s._v(" "),e("blockquote",[e("p",[s._v('select item, count(*) form tb where dt="group by item, area, age"')]),s._v(" "),e("p",[s._v("借助其他字段进行key分发")])]),s._v(" "),e("p",[s._v("sum,count等不容易产生数据倾斜。")]),s._v(" "),e("p",[s._v("插一个知识点：group by加盐，join不加盐。如100000个item1，如果改用item+random*1000，那么就可以分成1000个分发到reduce上")]),s._v(" "),e("h3",{attrs:{id:"表join的时候会导致数据倾斜-join就是把两张表中相同key分发到同一个reduce上"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#表join的时候会导致数据倾斜-join就是把两张表中相同key分发到同一个reduce上"}},[s._v("#")]),s._v(" **表join的时候会导致数据倾斜：**join就是把两张表中相同key分发到同一个reduce上")]),s._v(" "),e("ol",[e("li",[s._v("MAPJOIN")]),s._v(" "),e("li",[s._v("set hive.groupby.skewindata=true   先随机分发")]),s._v(" "),e("li",[e("img",{attrs:{src:"C:%5CUsers%5Ckaich%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210614221423039.png",alt:"image-20210614221423039"}})])]),s._v(" "),e("p",[s._v("优化点：尽量保证key不冗余")]),s._v(" "),e("p",[e("img",{attrs:{src:"C:%5CUsers%5Ckaich%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210614215753516.png",alt:"image-20210614215753516"}})]),s._v(" "),e("h4",{attrs:{id:"大表join小表"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#大表join小表"}},[s._v("#")]),s._v(" 大表join小表")]),s._v(" "),e("blockquote",[e("p",[s._v("大表join小表之mapjoin这块不是很理解，多看看")])]),s._v(" "),e("h4",{attrs:{id:"大表join大表"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#大表join大表"}},[s._v("#")]),s._v(" 大表join大表")]),s._v(" "),e("p",[s._v("必须得过reduce。处理null：")]),s._v(" "),e("div",{staticClass:"language- line-numbers-mode"},[e("pre",{pre:!0,attrs:{class:"language-text"}},[e("code",[s._v("select col_a,col_b from a left join b on coalesce(a.key,rand()*9999) on coalesce(b.key,rand()*9999)\n")])]),s._v(" "),e("div",{staticClass:"line-numbers-wrapper"},[e("span",{staticClass:"line-number"},[s._v("1")]),e("br")])]),e("h4",{attrs:{id:"多表联合join"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#多表联合join"}},[s._v("#")]),s._v(" 多表联合join")]),s._v(" "),e("p",[e("img",{attrs:{src:"C:%5CUsers%5Ckaich%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20210614224617325.png",alt:"image-20210614224617325"}})]),s._v(" "),e("p",[s._v("威尔逊平滑：实现ctr的修正")])],1)}),[],!1,null,null,null);a.default=n.exports}}]);