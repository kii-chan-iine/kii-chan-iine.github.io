(window.webpackJsonp=window.webpackJsonp||[]).push([[38],{652:function(e,t,a){"use strict";a.r(t);var s=a(3),n=Object(s.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("Boxx",{attrs:{changeTime:"10000"}}),e._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"title"},[e._v("前言")]),a("p",[e._v("涉及参考文献中BST文件的制作和修改")])]),e._v(" "),a("h1",{attrs:{id:"制作和修改需要的参考文献格式-bst"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#制作和修改需要的参考文献格式-bst"}},[e._v("#")]),e._v(" 制作和修改需要的参考文献格式（.bst）")]),e._v(" "),a("p",[e._v("一般的期刊或者会议都会提供参考文献格式模板(.bst)，但是如果没有提供的话，你可以选择自己制作一个bst或者寻找类似的再修改bst文件。这是一篇教你制作需要的"),a("strong",[e._v("参考文献格式")]),e._v("(.bst)的文章。文章主要包括两方面内容：1、从头制作.bst文件，包括对book,article,proceedings等等进行格式设置；2、根据需要微调做好的bst文件。(建议看下文档"),a("a",{attrs:{href:"https://www.docin.com/p-700531479.html",target:"_blank",rel:"noopener noreferrer"}},[e._v("A BibTEX Guide via Examples"),a("OutboundLink")],1),e._v(")")]),e._v(" "),a("p",[e._v("如果参考文献格式要求排序按出现顺序，引用方式为数字，作者只出现三个，超过三个用et al 表示。例要求参考文献格式为：\n"),a("strong",[e._v("Journal articles")]),e._v("\n[6] Borwn, L., Thomas, H., James, C., et al.:'The title of the paper, IET Communications, 2012, 6, (5), pp 125-138")]),e._v(" "),a("h2",{attrs:{id:"制作bst文件"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#制作bst文件"}},[e._v("#")]),e._v(" 制作bst文件")]),e._v(" "),a("ol",[a("li",[a("strong",[e._v("准备工作")]),e._v(" ，安装CTEX，如果已安装就跳过这步。从清华镜像网站下载CTEX套装"),a("a",{attrs:{href:"https://mirrors.tuna.tsinghua.edu.cn/ctex/legacy/2.9/",target:"_blank",rel:"noopener noreferrer"}},[e._v("下载点这里"),a("OutboundLink")],1),e._v("；下载Full版本。")]),e._v(" "),a("li",[e._v("win+R快捷键输入cmd后回车，键入"),a("strong",[e._v("latex makebst")]),e._v("。回答出现的一系列问题就可以制作自己的bst文件了。如果对问题不确定，可以选择默认的选项(*)，"),a("strong",[e._v("直接按回车表示选择默认选项")]),e._v("。")])]),e._v(" "),a("p",[e._v("以下"),a("strong",[e._v("xx")]),e._v("表示选择的内容")]),e._v(" "),a("ol",[a("li",[e._v("Do you want a description of the usage? 是否需要使用说明？"),a("strong",[e._v("yes")]),e._v("；")]),e._v(" "),a("li",[e._v("Enter the name of the MASTER file (default=merlin.mbs)输入主文件名，"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("Name of the final OUTPUT .bst file? (default extension=bst)，给输出的bst文件命名，这里我输入"),a("strong",[e._v("fly")]),e._v("来命名我的文件；")]),e._v(" "),a("li",[e._v("Give a comment line to include in the style file.Something like for which journals it is applicable.写在bst文件开头描述这个bst文件的用途等，可以写可以不写，我这里写*For CSDN**；")]),e._v(" "),a("li",[e._v("Do you want verbose comments? "),a("strong",[e._v("yes")]),e._v("可以在路径下的mbs文件中查看关于问题不同选项的意思")]),e._v(" "),a("li",[e._v("Name of language definition file (default=merlin.mbs)给定义语言的文件命名，我这里选"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("Include file(s) for extra journal names? "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("<<INTERNAL LANGUAGE SUPPORT (if no external language file) (*) English words used explicitly (b) Babel (words replaced by commands defined in babelbst.tex)\n"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("STYLE OF CITATIONS: (*) Numerical as in standard LaTeX (a) Author-year with some non-standard interface (b) Alpha style, Jon90 or JWB90 for single or multiple authors (o) Alpha style, Jon90 even for multiple authors (f) Alpha style, Jones90 (full name of first author) © Cite key (special for listing contents of bib file) 文章中引用的格式，这里我选择"),a("strong",[e._v("默认")]),e._v(",以数字出现")]),e._v(" "),a("li",[e._v("HTML OUTPUT (if non author-year citations) (*) Normal LaTeX output (h) Hypertext output, in HTML code, in paragraphs (n) Hypertext list with sequence numbers (k) Hypertext with keys for viewing databases 超文本的输出方式，是和正文一样还是实现超链接？"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("LANGUAGE FIELD (*) No language field (l) Add language field to switch hyphenation patterns temporarily 语言字段，"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("ANNOTATIONS: (*) No annotations will be recognized (a) Annotations in annote field or in .tex file of citekey name 注释，"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("PRESENTATIONS: (*) Do not add presentation type for conference talks § Add presentation, speaker not highlighted (b) Presentation, speaker bold face (i) Presentaion, speaker italic © Presentaion, speaker in small caps"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("ORDERING OF REFERENCES (if non-author/year and non-alph)（参考文献排序方式） (*) Alphabetical by all authors（按作者字母） ©Citation order (unsorted, like unsrt.bst)（按引用顺序） (d) Year ordered and then by authors（先按年再按作者） ® Reverse year ordered and then by authors 选"),a("strong",[e._v("c")])]),e._v(" "),a("li",[e._v("ORDE ON VON PART (if not citation order) (*) Sort on von part (de la Maire before Defoe) (x) Sort without von part (de la Maire after Mahone) Select:"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("AUTHOR NAMES: (*) Full, surname last (John Frederick Smith) (f) Full, surname first (Smith, John Frederick) (i) Initials + surname (J. F. Smith) ® Surname + initials (Smith, J. F.) (s) Surname + dotless initials (Smith J F) (w) Surname + comma + spaceless initials (Smith, J.F.) (x) Surname + pure initials (Smith JF) (y) Surname + comma + pure initials (Smith, JF) (z) Surname + spaceless initials (Smith J.F.) (a) Only first name reversed, initials (AGU style: Smith, J. F., H. K. Jones) (b) First name reversed, with full names (Smith, John Fred, Harry Kab Jones) 选择需要的作者名格式，这里我选择"),a("strong",[e._v("w")])]),e._v(" "),a("li",[e._v("PUNCTUATION BETWEEN AUTHOR NAMES: (*) Author names separated by commas (s) Names separated by semi-colon (h) Names separated by slash / 作者名之间的符号，我选择"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("<<ADJACENT REFERENCES WITH REPEATED NAMES: (*) Author/editor names always present (d) Repeated author/editor names replaced by dash (2) Repeated author/editor names replaced by 2 dashes (3) Repeated author/editor names replaced by 3 dashes 重复名称的相邻引用，"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("NUMBER OF AUTHORS IN BIBLIOGRAPHY: (*) All authors included in listing (l) Limited authors (et al replaces missing names) 参考文献中列出的作者名，"),a("strong",[e._v("l")])]),e._v(" "),a("li",[e._v("NUMBER OF AUTHORS IN BIBLIOGRAPHY: Maximum number of authors (1-99) 最多列出几个作者，这里我选"),a("strong",[e._v("3")])]),e._v(" "),a("li",[e._v("Minimum number (before et al given) (1-3)et al 放在第几个作者后面，"),a("strong",[e._v("3")])]),e._v(" "),a("li",[e._v("AUTHORS IN CITATIONS: (*) One author et al for three or more authors (m) Some other truncation scheme 引用中的作者数"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("TYPEFACE FOR AUTHORS IN LIST OF REFERENCES: (*) Normal font for author names (s) Small caps authors (\\sc) (i) Italic authors (\\it or \\em) (b) Bold authors (\\bf) (u) User defined author font (\\bibnamefont) 参考文献列表中作者的字体，"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("DATE POSITION: (*) Date at end (b) Date after authors (j) Date part of journal spec. (as 1994;45:34-40) else at end (e) Date at very end after any notes 日期的位置，这里我选"),a("strong",[e._v("j")])]),e._v(" "),a("li",[e._v("DATE FORMAT (if non author-year citations) (*) Plain month and year without any brackets § Date in parentheses as (May 1993) (b) Date in brackets as [May 1993] © Date preceded by colon as "),a("code",[e._v(": May 1993' (d) Date preceded by period as")]),e._v(". May 1993’ (m) Date preceded by comma as "),a("code",[e._v(", May 1993' (s) Date preceded by space only, as")]),e._v(" May 1993’ 日期格式，按自己的需要，我选"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("SUPPRESS MONTH: (*) Date is month and year (x) Date is year only 取消月份，我选"),a("strong",[e._v("x")])]),e._v(" "),a("li",[e._v("DATE PUNCTUATION (if date not at end) (*) Date with standard block punctuation (comma or period) © Colon after date as 1994: (s) Semi-colon after date as 1994; § Period after date even when blocks use commas (x) No punct. after date 日期之后的标点，我选"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("BLANK AFTER DATE: (*) Space after date and punctuation (x) No space after date as 1994:45 日期之后的间隔，我选"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("DATE FONT: (*) Date in normal font (b) Date in bold face 日期的字体，我选"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("TITLE OF ARTICLE ("),a("em",[e._v(") Title plain with no special font (i) Title italic (\\em) (q) Title and punctuation in single quotes ("),a("code",[e._v("Title,' ..) (d) Title and punctuation in double quotes (``Title,'' ..) (g) Title and punctuation in guillemets (<<Title,>> ..) (x) Title in single quotes (")]),e._v("Title’, …) (y) Title in double quotes (``Title’’, …) (z) Title in guillemets (<")]),e._v(") Quote collection and proceedings titles too (x) Collection and proceedings titles not in quotes 会议是否显示title，"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("CAPITALIZATION OF ARTICLE TITLE: (*) Sentence style (capitalize first word and those in braces) (t) Title style (just as in bib entry) 标题首字母大写问题，"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("ARTICLE TITLE PRESENT: (*) Article title present in journals and proceedings (x) No article title "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("JOURNAL NAMES: (*) Periods in journal names are retained, as "),a("code",[e._v("Phys. Rev.' (x) Dotless journal names as")]),e._v("Phys Rev’ "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("JOURNAL NAME FONT: (*) Journal name italics ® Journal name normal font 期刊名字体，"),a("strong",[e._v("r")])]),e._v(" "),a("li",[e._v("THESIS TITLE: (*) Thesis titles like books (a) Thesis title like article (x) No thesis title "),a("strong",[e._v("a")])]),e._v(" "),a("li",[e._v("TECHNICAL REPORT TITLE: (*) Tech. report title like articles (b) Tech. report title like books "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("TECHNICAL REPORT NUMBER: (*) Tech. report and number plain as "),a("code",[e._v("Tech. Rep. 123' (i) Tech. report and number italic as")]),e._v("{\\it Tech. Rep. 123’} "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("JOURNAL VOLUME: (*) Volume plain as vol(num) (i) Volume italic as {\\em vol}(num) (b) Volume bold as {\\bf vol}(num) (d) Volume and number bold as {\\bf vol(num)} "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("JOURNAL VOL AND NUMBER: (*) Journal vol(num) as 34(2) (s) Journal vol (num) as 34 (2) © Journal vol, num as 34, 2 (n) Journal vol, no. num as 34, no. 2 (h) Journal vol, # number as 34, #2 (b) Journal vol number as 34 2 (x) Journal vol, without number as 34 这里我需要的格式是34, (2)，但这里没有，我先选个相近的"),a("strong",[e._v("s")])]),e._v(" "),a("li",[e._v("VOLUME PUNCTUATION: (*) Volume with colon as vol(num):ppp (s) Volume with colon and space as vol(num): ppp (h) Volume with semi-colon as vol(num); ppp © Volume with comma as vol(num), ppp (b) Volume with blank as vol(num) ppp "),a("strong",[e._v("c")])]),e._v(" "),a("li",[e._v("YEAR IN JOURNAL SPECIFICATION: (*) Journal year like others as given by date position (v) Journal vol(year) as 34(1995) (s) Journal vol (year) as 34 (1995) § Year with pages as 34(2), (1995) 1345–1387 © Year, comma, pages as 34(2), (1995), 1345–1387 期刊年份，"),a("strong",[e._v("默认")]),e._v("以上面的date定义为准")]),e._v(" "),a("li",[e._v("PAGE NUMBERS: (*) Start and stop page numbers given (f) Only start page number "),a("strong",[e._v("默认")])]),e._v(" "),a("li"),e._v(" "),a("li",[e._v("WORD PAGE IN ARTICLES: (*) Article pages numbers only as 234-256 § Include page in articles as pp. 234–256 "),a("strong",[e._v("p")])]),e._v(" "),a("li",[e._v("POSITION OF PAGES: (*) Pages given mid text as is normal (e) Pages at end but before any notes 按需要选"),a("strong",[e._v("e")])]),e._v(" "),a("li",[e._v("WORD VOLUME IN ARTICLES: (*) Article volume as number only as 21 § Include volume in articles as vol. 21 "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("NUMBER AND SERIES FOR COLLECTIONS: (*) Allows number without series and suppresses word “number” (s) Standard BibTeX as: “number 123 in Total Works”; error if number and no ser ies "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("POSITION OF NUMBER AND SERIES: (*) After chapter and pages as in standard BibTeX (t) Just before publisher or organization "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("VOLUME AND SERIES FOR BOOKS/COLLECTIONS: (*) Vol. 23 of Series as in standard BibTeX (s) Series, vol. 23 "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("POSITION OF VOLUME AND SERIES FOR INCOLLECTIONS: (*) Series and volume after the editors (e) Series and volume after booktitle and before editors "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("JOURNAL NAME PUNCTUATION: (*) Comma after journal name (x) Space after journal name "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("BOOK TITLE: (*) Book title italic (\\em) § Book title plain (no font command) "),a("strong",[e._v("p")])]),e._v(" "),a("li",[e._v("PAGES IN BOOKS: (*) Pages in book plain as pp. 50-55 § Pages in book in parentheses as (pp. 50-55) (x) Pages in book bare as 50-55 "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("TOTAL PAGES OF A BOOK: (*) Total book pages not printed § For book: 345 pages or pp. (a) Total book pages before publisher "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("PUBLISHER ADDRESS: ("),a("em",[e._v(") Publisher, address as Harcourt, New York (a) Address: Publisher as New York: Harcourt "),a("strong",[e._v("默认")]),e._v("\n55.PUBLISHER IN PARENTHESES: (")]),e._v(") Publisher as normal block without parentheses § Publisher in parentheses (d) Publisher and date in parentheses (Oxford, 1994) © Publisher and date in parentheses, no comma (Oxford 1994) (f) Publisher and date without parentheses Oxford, 1994 (k) Publisher and date, no parentheses, no comma Oxford 1994 "),a("strong",[e._v("默认")])]),e._v(" "),a("li"),e._v(" "),a("li",[e._v("ISBN NUMBER: ("),a("em",[e._v(") Include ISBN for books, booklets, etc. (x) No ISBN "),a("strong",[e._v("x")]),e._v("\n57.ISSN NUMBER: (")]),e._v(") Include ISSN for periodicals (x) No ISSN "),a("strong",[e._v("x")])]),e._v(" "),a("li",[e._v("DOI NUMBER: (*) Include DOI as “doi: number” (u) Format DOI as URL //dx.doi.org/doi (must give url options!) (a) Insert DOI AGU style as part of page number (x) No DOI "),a("strong",[e._v("x")])]),e._v(" "),a("li",[a("code",[e._v("EDITOR' AFTER NAMES (EDITED BOOKS WITHOUT AUTHORS): (*) Word")]),e._v("editor’ after name (a) "),a("code",[e._v("Name (editor),' in parentheses, after name, comma after (b)")]),e._v("Name (Editor),’ as above, editor upper case © "),a("code",[e._v("Name, (editor)' in parentheses, after name, comma between (d)")]),e._v("Name, (Editor)’ as above, editor upper case (e) "),a("code",[e._v("Name (editor)' in parentheses, after name, no commas (f)")]),e._v("Name (Editor)’ as above, editor upper case "),a("strong",[e._v("d")])]),e._v(" "),a("li",[e._v("EDITOR IN COLLECTIONS: (*) Same as for edited book (names before booktitle) (b) In booktitle, edited by … (where … is names) § In booktitle (edited by …) © In booktitle, (edited by …) (e) In booktitle, editor … (f) In booktitle, (editor) … (k) In booktitle (editor…) (g) In booktitle, (editor…) (j) In booktitle, …, editor (m) In booktitle (…, editor) "),a("em",[e._v("默认")])]),e._v(" "),a("li",[e._v("PUNCTUATION BETWEEN SECTIONS (BLOCKS): (*) \\newblock after blocks (periods or new lines with openbib option) © Comma between blocks (s) Semi-colon between blocks (b) Blanks between blocks (t) Period after titles of articles, books, etc else commas (u) Colon after titles of articles, books, etc else commas (a) Period after titles of articles else commas (d) Colon after titles of articles else commas 每一块之间的符号，"),a("strong",[e._v("c")])]),e._v(" "),a("li",[e._v("PUNCTUATION BEFORE NOTES (if not using \\newblock) (*) Notes have regular punctuation like all other blocks § Notes preceded by period "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("PUNCTUATION AFTER AUTHORS:\n(*) Author block normal with regular block punctuation\n© Author block with colon\nSelect:"),a("strong",[e._v("c")]),e._v("，如果不需要作者不需要用冒号隔开，就用"),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("': (*) Space after`in’ for incollection or inproceedings © Colon after `in' (as`In: …’) (i) Italic `in' and space (d) Italic`in’ and colon (x) No word `in’ for edited works **默认**\n")]),e._v(" "),a("li"),e._v(" "),a("li",[e._v("ABBREVIATE WORD "),a("code",[e._v("PAGES' (if not using external language file) (*)")]),e._v("Page(s)’ (no abbreviation) (a) "),a("code",[e._v("Page' abbreviated as p. or pp. (x)")]),e._v("Page’ omitted "),a("strong",[e._v("a")])]),e._v(" "),a("li",[e._v("ABBREVIATE WORD "),a("code",[e._v("EDITORS': (*)")]),e._v("Editor(s)’ (no abbreviation) (a) `Editor’ abbreviated as ed. or eds. "),a("strong",[e._v("a")])]),e._v(" "),a("li",[e._v("OTHER ABBREVIATIONS: (*) No abbreviations of volume, edition, chapter, etc (a) Abbreviations of such words "),a("strong",[e._v("a")]),e._v("\n67.ABBREVIATION FOR "),a("code",[e._v("EDITION' (if abbreviating words) (*)")]),e._v("Edition’ abbreviated as "),a("code",[e._v("edn' (a)")]),e._v("Edition’ abbreviated as `ed’ "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("MONTHS WITH DOTS: (*) Months with dots as Jan. (x) Months without dots as Feb Mar "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("EDITION NUMBERS: (*) Editions as in database saving much processing memory (w) Write out editions as first, second, third, etc (n) Numerical editions as 1st, 2nd, 3rd, etc "),a("strong",[e._v("n")])]),e._v(" "),a("li",[e._v("ABBREVIATE WORD `PAGES’ (if not using external language file) <<STORED JOURNAL NAMES: (*) Full journal names for prestored journals (a) Abbreviated journal names (s) Abbreviated with astronomy shorthands like ApJ and AJ "),a("strong",[e._v("a")])]),e._v(" "),a("li",[e._v("AMPERSAND: (*) Use word "),a("code",[e._v("and' in author lists (a) Use ampersand in place of")]),e._v("and’ (v) Use \\BIBand in place of `and’ "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("COMMA BEFORE "),a("code",[e._v("AND': (*) Comma before")]),e._v("and’ as "),a("code",[e._v("Tom, Dick, and Harry' (n) No comma before")]),e._v("and’ as "),a("code",[e._v("Tom, Dick and Harry' (c) Comma even with 2 authors as")]),e._v("Tom, and Harry’ "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("NO "),a("code",[e._v("AND' IN REFERENCE LIST: (*) With")]),e._v("and’ before last author in reference list (x) No "),a("code",[e._v("and' as")]),e._v("Tom, Dick, Harry’ Select: x\n73.COMMA BEFORE "),a("code",[e._v("ET AL': (*) Comma before")]),e._v("et al’ in reference list (x) No comma before `et al’ "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("FONT OF `ET AL’: (*) Plain et al (i) Italic et al ® Roman et al even when authors something else "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("ADDITIONAL REVTeX DATA FIELDS: (*) No additional fields for REVTeX ® Include REVTeX data fields collaboration, eid, eprint, archive, numpages, u rl "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("E-PRINT DATA FIELD: (without REVTeX fields) (*) Do not include eprint field (e) Include eprint and archive fields for electronic publications "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("E-PRINT DATA FIELD: (without REVTeX fields) <<URL ADDRESS: (without REVTeX fields) (*) No URL for electronic (Internet) documents (u) Include URL as regular item block (n) URL as note (l) URL on new line after rest of reference "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("REFERENCE COMPONENT TAGS: (*) No reference component tags in the \\bibitem entries (b) Reference component tags like \\bibinfo in the content of \\bibitem Select: "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("EMPHASIS: (affects all so-called italics) (*) Use emphasis ie, \\em, allows font switching (i) Use true italics ie, \\it, absolute italics (x) No italics at all (u) Underlining in place of italics, best with ulem package **x **")]),e._v(" "),a("li",[e._v("COMPATIBILITY WITH PLAIN TEX: (*) Use LaTeX commands which may not work with Plain TeX (t) Use only Plain TeX commands for fonts and testing "),a("strong",[e._v("默认")])]),e._v(" "),a("li",[e._v("COMPATIBILITY WITH PLAIN TEX: ) Finished!! Batch job written to file `fly.dbj’")]),e._v(" "),a("li",[e._v("Shall I now run this batch job? (NO) \\yn=yes")])]),e._v(" "),a("p",[a("strong",[e._v("回答完上述所有问题，在当前路径下可以得到一个makebst文档记录了你之前回答的问题，还有一个一个fly.bst文件。将这个fly.bst文件复制到编辑的latex所在的文件夹中。")])]),e._v(" "),a("h2",{attrs:{id:"修改bst文件-在期和卷之间加逗号"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#修改bst文件-在期和卷之间加逗号"}},[e._v("#")]),e._v(" 修改bst文件：在期和卷之间加逗号")]),e._v(" "),a("p",[e._v("编译.tex文件，得到如下列出的参考文献如下：\n[6] Borwn, L., Thomas, H., James, C., et al.:'The title of the paper, IET Communications, 2012, 6 (5), pp 125-138\n但是要求的格式是vol, num。还记得在问题39里面，只选了个相近的vol (num)，不符合要求。因此需要修改bst。\n用Winedt打开bst文件，"),a("strong",[e._v("ctrl+F")]),e._v("查找"),a("strong",[e._v("format.vol.num.pages")]),e._v(",")]),e._v(" "),a("div",{staticClass:"language-typescript line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-typescript"}},[a("code",[a("span",{pre:!0,attrs:{class:"token constant"}},[e._v("FUNCTION")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("{")]),e._v("format"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),e._v("vol"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),e._v("num"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),e._v("pages"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("}")]),e._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("{")]),e._v(" volume field"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),e._v("or"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[e._v("null")]),e._v("\n  duplicate$ empty$ 'skip$\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("{")]),e._v("\n      "),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('"volume"')]),e._v(" bibinfo"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),e._v("check\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("}")]),e._v("\n  "),a("span",{pre:!0,attrs:{class:"token keyword"}},[e._v("if")]),e._v("$\n  "),a("span",{pre:!0,attrs:{class:"token builtin"}},[e._v("number")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('"number"')]),e._v(" bibinfo"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(".")]),e._v("check duplicate$ empty$ 'skip$\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("{")]),e._v("\n      swap$ duplicate$ empty$\n        "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("{")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('"there\'s a number but no volume in "')]),e._v(" cite$ "),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("*")]),e._v(" warning$ "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("}")]),e._v("\n        'skip$\n      "),a("span",{pre:!0,attrs:{class:"token keyword"}},[e._v("if")]),e._v("$\n      swap$\n      "),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('"~("')]),e._v(" swap$ "),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("*")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('")"')]),e._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("*")]),e._v("\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("}")]),e._v("\n  "),a("span",{pre:!0,attrs:{class:"token keyword"}},[e._v("if")]),e._v("$ "),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("*")]),e._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v("}")]),e._v("\n")])]),e._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[e._v("1")]),a("br"),a("span",{staticClass:"line-number"},[e._v("2")]),a("br"),a("span",{staticClass:"line-number"},[e._v("3")]),a("br"),a("span",{staticClass:"line-number"},[e._v("4")]),a("br"),a("span",{staticClass:"line-number"},[e._v("5")]),a("br"),a("span",{staticClass:"line-number"},[e._v("6")]),a("br"),a("span",{staticClass:"line-number"},[e._v("7")]),a("br"),a("span",{staticClass:"line-number"},[e._v("8")]),a("br"),a("span",{staticClass:"line-number"},[e._v("9")]),a("br"),a("span",{staticClass:"line-number"},[e._v("10")]),a("br"),a("span",{staticClass:"line-number"},[e._v("11")]),a("br"),a("span",{staticClass:"line-number"},[e._v("12")]),a("br"),a("span",{staticClass:"line-number"},[e._v("13")]),a("br"),a("span",{staticClass:"line-number"},[e._v("14")]),a("br"),a("span",{staticClass:"line-number"},[e._v("15")]),a("br"),a("span",{staticClass:"line-number"},[e._v("16")]),a("br"),a("span",{staticClass:"line-number"},[e._v("17")]),a("br"),a("span",{staticClass:"line-number"},[e._v("18")]),a("br")])]),a("p",[e._v("修改方法如下：")]),e._v(" "),a("div",{staticClass:"language-typescript line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-typescript"}},[a("code",[e._v("将"),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('"~("')]),e._v(" swap$ "),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("*")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('")"')]),e._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("*")]),e._v("修改为"),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('",~("')]),e._v(" swap$ "),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("*")]),e._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[e._v('")"')]),e._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[e._v("*")]),e._v(" 添加的逗号就是vol"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[e._v(",")]),e._v("num中的逗号\n"),a("span",{pre:!0,attrs:{class:"token number"}},[e._v("1")]),e._v("\n")])]),e._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[e._v("1")]),a("br"),a("span",{staticClass:"line-number"},[e._v("2")]),a("br")])]),a("p",[e._v("将上次编译生成的bbl删除，就可以得到正确的参考文献格式\n[6] Borwn, L., Thomas, H., James, C., et al.:'The title of the paper, IET Communications, 2012, 6, (5), pp 125-138")])],1)}),[],!1,null,null,null);t.default=n.exports}}]);