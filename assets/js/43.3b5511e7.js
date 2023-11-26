(window.webpackJsonp=window.webpackJsonp||[]).push([[43],{657:function(s,t,a){"use strict";a.r(t);var e=a(3),r=Object(e.a)({},(function(){var s=this,t=s.$createElement,a=s._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":s.$parent.slotKey}},[a("Boxx",{attrs:{changeTime:"10000"}}),s._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"title"},[s._v("前言")]),a("p",[s._v("Linux 使用的一些笔记")])]),a("h2",{attrs:{id:"用户相关"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#用户相关"}},[s._v("#")]),s._v(" 用户相关")]),s._v(" "),a("div",{staticClass:"language-bash line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-bash"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[s._v("useradd")]),s._v(" -c kii -u "),a("span",{pre:!0,attrs:{class:"token number"}},[s._v("2023")]),s._v(" -G sudo,docker -d /home/kii -m -N -s /bin/bash kii\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("h2",{attrs:{id:"shell-使用"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#shell-使用"}},[s._v("#")]),s._v(" shell 使用")]),s._v(" "),a("p",[s._v("在shell中终端中使用快捷键 “Ctr+r”，终端会出现如下提示")]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("(reverse-i-search)`': \n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("p",[s._v("然后输入历史命令中的关键字，会"),a("a",{attrs:{href:"https://so.csdn.net/so/search?q=%E8%87%AA%E5%8A%A8%E8%A1%A5%E5%85%A8&spm=1001.2101.3001.7020",target:"_blank",rel:"noopener noreferrer"}},[s._v("自动补全"),a("OutboundLink")],1),s._v("包含关键字的历史命令，接下来")]),s._v(" "),a("ul",[a("li",[s._v("按ESC键，可退出搜索模式，并保留历史命令，可再次编辑命令")]),s._v(" "),a("li",[s._v("按ENTRY键，可直接执行该历史命令")])]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v('"auto"为关键字\n(reverse-i-search)`auto\': sudo apt autoremove\n')])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br")])]),a("h1",{attrs:{id:"zsh"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#zsh"}},[s._v("#")]),s._v(" zsh")]),s._v(" "),a("h2",{attrs:{id:"安装"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#安装"}},[s._v("#")]),s._v(" 安装")]),s._v(" "),a("ol",[a("li",[s._v("安装zsh")])]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("sudo apt-get install zsh\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("ol",{attrs:{start:"2"}},[a("li",[s._v("安装oh-my-zsh")])]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v('sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"\n')])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("div",{staticClass:"language-bash line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-bash"}},[a("code",[a("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("ZSH_THEME")]),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v('"candy"')]),s._v("\n"),a("span",{pre:!0,attrs:{class:"token assign-left variable"}},[s._v("ZSH_THEME")]),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v('"ys"')]),s._v(" "),a("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#建议")]),s._v("\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br")])]),a("ol",{attrs:{start:"3"}},[a("li",[s._v("autosuggestions")])]),s._v(" "),a("div",{staticClass:"language-bash line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-bash"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[s._v("git")]),s._v(" clone https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions\n"),a("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("source")]),s._v(" ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br")])]),a("h1",{attrs:{id:"离线安装ohmyzsh"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#离线安装ohmyzsh"}},[s._v("#")]),s._v(" 离线安装ohmyzsh")]),s._v(" "),a("p",[a("a",{attrs:{href:"https://github.com/kii-chan-iine/ohmyzsh",target:"_blank",rel:"noopener noreferrer"}},[s._v("ohmyzsh"),a("OutboundLink")],1)]),s._v(" "),a("ol",[a("li",[s._v("去github下载"),a("u",[a("a",{staticClass:"wrap external",attrs:{rel:"nofollow noreferrer",href:"https://link.zhihu.com/?target=https%3A//github.com/ohmyzsh/ohmyzsh"}},[s._v("OH My ZSH")])])]),s._v(" "),a("li",[s._v("进入OH My ZSH项目目录并应用补丁"),a("code",[s._v("git apply offline_install.diff")])]),s._v(" "),a("li",[s._v("现在你就可以运行OH My ZSH中tool/install.sh安装了")])]),s._v(" "),a("div",{staticClass:"language-bash line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-bash"}},[a("code",[a("span",{pre:!0,attrs:{class:"token function"}},[s._v("git")]),s._v(" clone https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions\n"),a("span",{pre:!0,attrs:{class:"token builtin class-name"}},[s._v("source")]),s._v(" ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br")])]),a("p",[s._v("autosuggestion 可以下载"),a("code",[s._v("https://github.com/zsh-users/zsh-autosuggestions")]),s._v("复制到"),a("code",[s._v("~/.zsh/zsh-autosuggestions")]),s._v("进行离线安装。")]),s._v(" "),a("h1",{attrs:{id:"centos"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#centos"}},[s._v("#")]),s._v(" centos")]),s._v(" "),a("h2",{attrs:{id:"_1-3-安装zsh包"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-3-安装zsh包"}},[s._v("#")]),s._v(" 1.3 安装zsh包")]),s._v(" "),a("p",[a("code",[s._v("dnf -y install zsh")])]),s._v(" "),a("p",[s._v("安装完成后查看shell列表"),a("br"),s._v(" "),a("code",[s._v("cat /etc/shells")])]),s._v(" "),a("p",[s._v("返回结果如下：")]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("/bin/sh\n/bin/bash\n/sbin/nologin\n/bin/dash\n/bin/tcsh\n/bin/csh\n/bin/zsh\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br"),a("span",{staticClass:"line-number"},[s._v("3")]),a("br"),a("span",{staticClass:"line-number"},[s._v("4")]),a("br"),a("span",{staticClass:"line-number"},[s._v("5")]),a("br"),a("span",{staticClass:"line-number"},[s._v("6")]),a("br"),a("span",{staticClass:"line-number"},[s._v("7")]),a("br")])]),a("h2",{attrs:{id:"_1-4-切换shell至zsh"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-4-切换shell至zsh"}},[s._v("#")]),s._v(" 1.4 切换shell至zsh")]),s._v(" "),a("p",[a("code",[s._v("chsh -s /bin/zsh")])]),s._v(" "),a("p",[s._v("返回结果："),a("br"),s._v(" "),a("code",[s._v("Changing shell for root.")]),a("br"),s._v(" "),a("code",[s._v("Shell changed.")])]),s._v(" "),a("p",[s._v("按提示所述，shell已经更改为zsh了，现在查看一下系统当前使用的shell，"),a("br"),s._v(" "),a("code",[s._v("echo $SHELL")]),a("br"),s._v("\n返回结果如下："),a("br"),s._v(" "),a("code",[s._v("/bin/bash")]),a("br"),s._v("\n重启过后，使用代码查看当前使用的shell"),a("br"),s._v(" "),a("code",[s._v("echo $SHELL")]),a("br"),s._v("\n返回结果："),a("br"),s._v(" "),a("code",[s._v("/bin/zsh")])]),s._v(" "),a("p",[s._v("得到如此结果，证明shell已经切换成功了。")]),s._v(" "),a("h1",{attrs:{id:"jupyter"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#jupyter"}},[s._v("#")]),s._v(" jupyter")]),s._v(" "),a("h2",{attrs:{id:"centos远程服务器安装jupyter-oserror-errno-99-cannot-assign-requested-address"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#centos远程服务器安装jupyter-oserror-errno-99-cannot-assign-requested-address"}},[s._v("#")]),s._v(" CENTOS远程服务器安装JUPYTER：OSERROR: [ERRNO 99] CANNOT ASSIGN REQUESTED ADDRESS")]),s._v(" "),a("p",[s._v("标签： "),a("a",{attrs:{href:"https://www.freesion.com/tag/%E7%8E%AF%E5%A2%83%E6%8A%A5%E9%94%99%E5%8F%8A%E8%A7%A3%E5%86%B3%E8%AE%B0%E5%BD%95/",title:"环境报错及解决记录",target:"_blank",rel:"noopener noreferrer"}},[s._v("环境报错及解决记录"),a("OutboundLink")],1),s._v(" "),a("a",{attrs:{href:"https://www.freesion.com/tag/tensorflow/",title:"tensorflow",target:"_blank",rel:"noopener noreferrer"}},[s._v("tensorflow"),a("OutboundLink")],1),s._v(" "),a("a",{attrs:{href:"https://www.freesion.com/tag/jupyter/",title:"jupyter",target:"_blank",rel:"noopener noreferrer"}},[s._v("jupyter"),a("OutboundLink")],1)]),s._v(" "),a("p",[s._v("Anaconda和Tensorflow的安装跳过。")]),s._v(" "),a("p",[s._v("问题：在tensorflow2.0-cpu版本下安装Jupyter报错。")]),s._v(" "),a("p",[a("img",{attrs:{src:"https://www.freesion.com/images/187/7e72875b0afb20c6ca58384a25895f33.png",alt:""}})]),s._v(" "),a("h1",{attrs:{id:"解决方案"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#解决方案"}},[s._v("#")]),s._v(" 解决方案：")]),s._v(" "),a("h2",{attrs:{id:""}},[a("a",{staticClass:"header-anchor",attrs:{href:"#"}},[s._v("#")])]),s._v(" "),a("p",[s._v("第一种（复杂的）：配置文件")]),s._v(" "),a("h3",{attrs:{id:"_1-打开python终端-生成"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_1-打开python终端-生成"}},[s._v("#")]),s._v(" 1.打开PYTHON终端，生成**")]),s._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("from")]),s._v(" IPython"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("lib "),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("import")]),s._v(" passwdpasswd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v("Enter password"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(":")]),s._v(" Verify password"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(":")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v("'sha1:xxxxxxxxx'")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token comment"}},[s._v("#记住这段秘钥")]),s._v("\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("h3",{attrs:{id:"_2-修改-etc-hosts"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_2-修改-etc-hosts"}},[s._v("#")]),s._v(" 2.修改/ETC/HOSTS")]),s._v(" "),a("p",[s._v("1.首先获取本机内网ip和本机hostname")]),s._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[s._v("ifconfig    "),a("span",{pre:!0,attrs:{class:"token comment"}},[s._v("# 获取本机内网ipvi /etc/hostname    # 获取hostname")]),s._v("\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("p",[s._v("2.进入/etc/hosts，添加上一行内容")]),s._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[s._v("内网ip hostname "),a("span",{pre:!0,attrs:{class:"token comment"}},[s._v("# 上面获取的那两个")]),s._v("\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("h3",{attrs:{id:"_3-生成默认配置文件-jupyter-notebook-config-py"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_3-生成默认配置文件-jupyter-notebook-config-py"}},[s._v("#")]),s._v(" 3.生成默认配置文件---JUPYTER_NOTEBOOK_CONFIG.PY")]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("jupyter notebook --generate-config\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("h3",{attrs:{id:"_4-修改jupyter-notebook-config-py配置文件"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_4-修改jupyter-notebook-config-py配置文件"}},[s._v("#")]),s._v(" 4.修改JUPYTER_NOTEBOOK_CONFIG.PY配置文件")]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("vim ~/.jupyter/jupyter_notebook_config.py\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("p",[s._v("说明：里面全是注释的配置说明，繁琐又复杂，就不看了，里面很多都用不上，这里我需要重写一些配置即可，在文件开头写入：")]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("c.NotebookApp.ip = \"内网ip hostname \"c.NotebookAPp.open_browser = False——这项默认是True，远程登陆时要修改为 Falsec.NotebookApp.password =u 'sha1:xxxx'——前面生成的**c.NotebookApp.port= 8888——可以自己另指定一个端口\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])]),a("h3",{attrs:{id:"_5-运行jupyter-notebook-浏览器访问"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#_5-运行jupyter-notebook-浏览器访问"}},[s._v("#")]),s._v(" 5.运行JUPYTER NOTEBOOK，浏览器访问")]),s._v(" "),a("p",[a("img",{attrs:{src:"https://www.freesion.com/images/258/f2012d7edf0edbdd5a158e959bba29da.png",alt:""}})]),s._v(" "),a("h2",{attrs:{id:"第二种-超简单-命令式"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#第二种-超简单-命令式"}},[s._v("#")]),s._v(" 第二种（超简单）：命令式")]),s._v(" "),a("p",[s._v("加入参数启动jupyter")]),s._v(" "),a("div",{staticClass:"language- line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-text"}},[a("code",[s._v("jupyter notebook --ip=0.0.0.0 --no-browser --allow-root\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br")])])],1)}),[],!1,null,null,null);t.default=r.exports}}]);