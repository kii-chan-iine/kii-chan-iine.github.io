---
author: kii
title: 制作和修改需要的参考文献格式
categories: [Hadoop]
tags: [Hadoop]
date: 2021-06-12 19:11:30
---

<Boxx changeTime="10000"/>

::: tip 前言
涉及参考文献中BST文件的制作和修改
:::
<!-- more -->

# 制作和修改需要的参考文献格式（.bst）

一般的期刊或者会议都会提供参考文献格式模板(.bst)，但是如果没有提供的话，你可以选择自己制作一个bst或者寻找类似的再修改bst文件。这是一篇教你制作需要的**参考文献格式**(.bst)的文章。文章主要包括两方面内容：1、从头制作.bst文件，包括对book,article,proceedings等等进行格式设置；2、根据需要微调做好的bst文件。(建议看下文档[A BibTEX Guide via Examples](https://www.docin.com/p-700531479.html))

如果参考文献格式要求排序按出现顺序，引用方式为数字，作者只出现三个，超过三个用et al 表示。例要求参考文献格式为：
**Journal articles**
[6] Borwn, L., Thomas, H., James, C., et al.:'The title of the paper, IET Communications, 2012, 6, (5), pp 125-138

## 制作bst文件

1. **准备工作** ，安装CTEX，如果已安装就跳过这步。从清华镜像网站下载CTEX套装[下载点这里](https://mirrors.tuna.tsinghua.edu.cn/ctex/legacy/2.9/)；下载Full版本。
2. win+R快捷键输入cmd后回车，键入**latex makebst**。回答出现的一系列问题就可以制作自己的bst文件了。如果对问题不确定，可以选择默认的选项(*)，**直接按回车表示选择默认选项**。

以下**xx**表示选择的内容

1. Do you want a description of the usage? 是否需要使用说明？**yes**；
2. Enter the name of the MASTER file (default=merlin.mbs)输入主文件名，**默认**
3. Name of the final OUTPUT .bst file? (default extension=bst)，给输出的bst文件命名，这里我输入**fly**来命名我的文件；
4. Give a comment line to include in the style file.Something like for which journals it is applicable.写在bst文件开头描述这个bst文件的用途等，可以写可以不写，我这里写*For CSDN**；
5. Do you want verbose comments? **yes**可以在路径下的mbs文件中查看关于问题不同选项的意思
6. Name of language definition file (default=merlin.mbs)给定义语言的文件命名，我这里选**默认**
7. Include file(s) for extra journal names? **默认**
8. <<INTERNAL LANGUAGE SUPPORT (if no external language file) (*) English words used explicitly (b) Babel (words replaced by commands defined in babelbst.tex)
   **默认**
9. STYLE OF CITATIONS: (*) Numerical as in standard LaTeX (a) Author-year with some non-standard interface (b) Alpha style, Jon90 or JWB90 for single or multiple authors (o) Alpha style, Jon90 even for multiple authors (f) Alpha style, Jones90 (full name of first author) © Cite key (special for listing contents of bib file) 文章中引用的格式，这里我选择**默认**,以数字出现
10. HTML OUTPUT (if non author-year citations) (*) Normal LaTeX output (h) Hypertext output, in HTML code, in paragraphs (n) Hypertext list with sequence numbers (k) Hypertext with keys for viewing databases 超文本的输出方式，是和正文一样还是实现超链接？**默认**
11. LANGUAGE FIELD (*) No language field (l) Add language field to switch hyphenation patterns temporarily 语言字段，**默认**
12. ANNOTATIONS: (*) No annotations will be recognized (a) Annotations in annote field or in .tex file of citekey name 注释，**默认**
13. PRESENTATIONS: (*) Do not add presentation type for conference talks § Add presentation, speaker not highlighted (b) Presentation, speaker bold face (i) Presentaion, speaker italic © Presentaion, speaker in small caps**默认**
14. ORDERING OF REFERENCES (if non-author/year and non-alph)（参考文献排序方式） (*) Alphabetical by all authors（按作者字母） ©Citation order (unsorted, like unsrt.bst)（按引用顺序） (d) Year ordered and then by authors（先按年再按作者） ® Reverse year ordered and then by authors 选**c**
15. ORDE ON VON PART (if not citation order) (*) Sort on von part (de la Maire before Defoe) (x) Sort without von part (de la Maire after Mahone) Select:**默认**
16. AUTHOR NAMES: (*) Full, surname last (John Frederick Smith) (f) Full, surname first (Smith, John Frederick) (i) Initials + surname (J. F. Smith) ® Surname + initials (Smith, J. F.) (s) Surname + dotless initials (Smith J F) (w) Surname + comma + spaceless initials (Smith, J.F.) (x) Surname + pure initials (Smith JF) (y) Surname + comma + pure initials (Smith, JF) (z) Surname + spaceless initials (Smith J.F.) (a) Only first name reversed, initials (AGU style: Smith, J. F., H. K. Jones) (b) First name reversed, with full names (Smith, John Fred, Harry Kab Jones) 选择需要的作者名格式，这里我选择**w**
17. PUNCTUATION BETWEEN AUTHOR NAMES: (*) Author names separated by commas (s) Names separated by semi-colon (h) Names separated by slash / 作者名之间的符号，我选择**默认**
18. <<ADJACENT REFERENCES WITH REPEATED NAMES: (*) Author/editor names always present (d) Repeated author/editor names replaced by dash (2) Repeated author/editor names replaced by 2 dashes (3) Repeated author/editor names replaced by 3 dashes 重复名称的相邻引用，**默认**
19. NUMBER OF AUTHORS IN BIBLIOGRAPHY: (*) All authors included in listing (l) Limited authors (et al replaces missing names) 参考文献中列出的作者名，**l**
20. NUMBER OF AUTHORS IN BIBLIOGRAPHY: Maximum number of authors (1-99) 最多列出几个作者，这里我选**3**
21. Minimum number (before et al given) (1-3)et al 放在第几个作者后面，**3**
22. AUTHORS IN CITATIONS: (*) One author et al for three or more authors (m) Some other truncation scheme 引用中的作者数**默认**
23. TYPEFACE FOR AUTHORS IN LIST OF REFERENCES: (*) Normal font for author names (s) Small caps authors (\sc) (i) Italic authors (\it or \em) (b) Bold authors (\bf) (u) User defined author font (\bibnamefont) 参考文献列表中作者的字体，**默认**
24. DATE POSITION: (*) Date at end (b) Date after authors (j) Date part of journal spec. (as 1994;45:34-40) else at end (e) Date at very end after any notes 日期的位置，这里我选**j**
25. DATE FORMAT (if non author-year citations) (*) Plain month and year without any brackets § Date in parentheses as (May 1993) (b) Date in brackets as [May 1993] © Date preceded by colon as `: May 1993' (d) Date preceded by period as`. May 1993’ (m) Date preceded by comma as `, May 1993' (s) Date preceded by space only, as` May 1993’ 日期格式，按自己的需要，我选**默认**
26. SUPPRESS MONTH: (*) Date is month and year (x) Date is year only 取消月份，我选**x**
27. DATE PUNCTUATION (if date not at end) (*) Date with standard block punctuation (comma or period) © Colon after date as 1994: (s) Semi-colon after date as 1994; § Period after date even when blocks use commas (x) No punct. after date 日期之后的标点，我选**默认**
28. BLANK AFTER DATE: (*) Space after date and punctuation (x) No space after date as 1994:45 日期之后的间隔，我选**默认**
29. DATE FONT: (*) Date in normal font (b) Date in bold face 日期的字体，我选**默认**
30. TITLE OF ARTICLE (*) Title plain with no special font (i) Title italic (\em) (q) Title and punctuation in single quotes (`Title,' ..) (d) Title and punctuation in double quotes (``Title,'' ..) (g) Title and punctuation in guillemets (<<Title,>> ..) (x) Title in single quotes (`Title’, …) (y) Title in double quotes (``Title’’, …) (z) Title in guillemets (<*) Quote collection and proceedings titles too (x) Collection and proceedings titles not in quotes 会议是否显示title，**默认**
31. CAPITALIZATION OF ARTICLE TITLE: (*) Sentence style (capitalize first word and those in braces) (t) Title style (just as in bib entry) 标题首字母大写问题，**默认**
32. ARTICLE TITLE PRESENT: (*) Article title present in journals and proceedings (x) No article title **默认**
33. JOURNAL NAMES: (*) Periods in journal names are retained, as `Phys. Rev.' (x) Dotless journal names as`Phys Rev’ **默认**
34. JOURNAL NAME FONT: (*) Journal name italics ® Journal name normal font 期刊名字体，**r**
35. THESIS TITLE: (*) Thesis titles like books (a) Thesis title like article (x) No thesis title **a**
36. TECHNICAL REPORT TITLE: (*) Tech. report title like articles (b) Tech. report title like books **默认**
37. TECHNICAL REPORT NUMBER: (*) Tech. report and number plain as `Tech. Rep. 123' (i) Tech. report and number italic as`{\it Tech. Rep. 123’} **默认**
38. JOURNAL VOLUME: (*) Volume plain as vol(num) (i) Volume italic as {\em vol}(num) (b) Volume bold as {\bf vol}(num) (d) Volume and number bold as {\bf vol(num)} **默认**
39. JOURNAL VOL AND NUMBER: (*) Journal vol(num) as 34(2) (s) Journal vol (num) as 34 (2) © Journal vol, num as 34, 2 (n) Journal vol, no. num as 34, no. 2 (h) Journal vol, # number as 34, #2 (b) Journal vol number as 34 2 (x) Journal vol, without number as 34 这里我需要的格式是34, (2)，但这里没有，我先选个相近的**s**
40. VOLUME PUNCTUATION: (*) Volume with colon as vol(num):ppp (s) Volume with colon and space as vol(num): ppp (h) Volume with semi-colon as vol(num); ppp © Volume with comma as vol(num), ppp (b) Volume with blank as vol(num) ppp **c**
41. YEAR IN JOURNAL SPECIFICATION: (*) Journal year like others as given by date position (v) Journal vol(year) as 34(1995) (s) Journal vol (year) as 34 (1995) § Year with pages as 34(2), (1995) 1345–1387 © Year, comma, pages as 34(2), (1995), 1345–1387 期刊年份，**默认**以上面的date定义为准
42. PAGE NUMBERS: (*) Start and stop page numbers given (f) Only start page number **默认**
43. <LARGE PAGE NUMBERS: (*) No separators for large page numbers © Comma inserted over 9999 as 11,234 (s) Thin space inserted over 9999 as 11 234 § Period inserted over 9999 as 11.234 页码数字是否用逗号隔开，**默认**
44. WORD PAGE IN ARTICLES: (*) Article pages numbers only as 234-256 § Include page in articles as pp. 234–256 **p**
45. POSITION OF PAGES: (*) Pages given mid text as is normal (e) Pages at end but before any notes 按需要选**e**
46. WORD VOLUME IN ARTICLES: (*) Article volume as number only as 21 § Include volume in articles as vol. 21 **默认**
47. NUMBER AND SERIES FOR COLLECTIONS: (*) Allows number without series and suppresses word “number” (s) Standard BibTeX as: “number 123 in Total Works”; error if number and no ser ies **默认**
48. POSITION OF NUMBER AND SERIES: (*) After chapter and pages as in standard BibTeX (t) Just before publisher or organization **默认**
49. VOLUME AND SERIES FOR BOOKS/COLLECTIONS: (*) Vol. 23 of Series as in standard BibTeX (s) Series, vol. 23 **默认**
50. POSITION OF VOLUME AND SERIES FOR INCOLLECTIONS: (*) Series and volume after the editors (e) Series and volume after booktitle and before editors **默认**
51. JOURNAL NAME PUNCTUATION: (*) Comma after journal name (x) Space after journal name **默认**
52. BOOK TITLE: (*) Book title italic (\em) § Book title plain (no font command) **p**
53. PAGES IN BOOKS: (*) Pages in book plain as pp. 50-55 § Pages in book in parentheses as (pp. 50-55) (x) Pages in book bare as 50-55 **默认**
54. TOTAL PAGES OF A BOOK: (*) Total book pages not printed § For book: 345 pages or pp. (a) Total book pages before publisher **默认**
55. PUBLISHER ADDRESS: (*) Publisher, address as Harcourt, New York (a) Address: Publisher as New York: Harcourt **默认**
    55.PUBLISHER IN PARENTHESES: (*) Publisher as normal block without parentheses § Publisher in parentheses (d) Publisher and date in parentheses (Oxford, 1994) © Publisher and date in parentheses, no comma (Oxford 1994) (f) Publisher and date without parentheses Oxford, 1994 (k) Publisher and date, no parentheses, no comma Oxford 1994 **默认**
56. <PUBLISHER POSITION: (*) Publisher after chapter, pages § Publisher before chapter, pages (e) Publisher after edition **p**
57. ISBN NUMBER: (*) Include ISBN for books, booklets, etc. (x) No ISBN **x**
    57.ISSN NUMBER: (*) Include ISSN for periodicals (x) No ISSN **x**
58. DOI NUMBER: (*) Include DOI as “doi: number” (u) Format DOI as URL //dx.doi.org/doi (must give url options!) (a) Insert DOI AGU style as part of page number (x) No DOI **x**
59. `EDITOR' AFTER NAMES (EDITED BOOKS WITHOUT AUTHORS): (*) Word`editor’ after name (a) `Name (editor),' in parentheses, after name, comma after (b)`Name (Editor),’ as above, editor upper case © `Name, (editor)' in parentheses, after name, comma between (d)`Name, (Editor)’ as above, editor upper case (e) `Name (editor)' in parentheses, after name, no commas (f)`Name (Editor)’ as above, editor upper case **d**
60. EDITOR IN COLLECTIONS: (*) Same as for edited book (names before booktitle) (b) In booktitle, edited by … (where … is names) § In booktitle (edited by …) © In booktitle, (edited by …) (e) In booktitle, editor … (f) In booktitle, (editor) … (k) In booktitle (editor…) (g) In booktitle, (editor…) (j) In booktitle, …, editor (m) In booktitle (…, editor) *默认*
61. PUNCTUATION BETWEEN SECTIONS (BLOCKS): (*) \newblock after blocks (periods or new lines with openbib option) © Comma between blocks (s) Semi-colon between blocks (b) Blanks between blocks (t) Period after titles of articles, books, etc else commas (u) Colon after titles of articles, books, etc else commas (a) Period after titles of articles else commas (d) Colon after titles of articles else commas 每一块之间的符号，**c**
62. PUNCTUATION BEFORE NOTES (if not using \newblock) (*) Notes have regular punctuation like all other blocks § Notes preceded by period **默认**
63. PUNCTUATION AFTER AUTHORS:
    (*) Author block normal with regular block punctuation
    © Author block with colon
    Select:**c**，如果不需要作者不需要用冒号隔开，就用**默认**
64. <PUNCTUATION AFTER `IN': (*) Space after`in’ for incollection or inproceedings © Colon after `in' (as`In: …’) (i) Italic `in' and space (d) Italic`in’ and colon (x) No word `in’ for edited works **默认**
65. <FINAL PUNCTUATION: (*) Period at very end of the listed reference (x) No period at end 最后的标点，**x**
66. ABBREVIATE WORD `PAGES' (if not using external language file) (*)`Page(s)’ (no abbreviation) (a) `Page' abbreviated as p. or pp. (x)`Page’ omitted **a**
67. ABBREVIATE WORD `EDITORS': (*)`Editor(s)’ (no abbreviation) (a) `Editor’ abbreviated as ed. or eds. **a**
68. OTHER ABBREVIATIONS: (*) No abbreviations of volume, edition, chapter, etc (a) Abbreviations of such words **a**
    67.ABBREVIATION FOR `EDITION' (if abbreviating words) (*)`Edition’ abbreviated as `edn' (a)`Edition’ abbreviated as `ed’ **默认**
69. MONTHS WITH DOTS: (*) Months with dots as Jan. (x) Months without dots as Feb Mar **默认**
70. EDITION NUMBERS: (*) Editions as in database saving much processing memory (w) Write out editions as first, second, third, etc (n) Numerical editions as 1st, 2nd, 3rd, etc **n**
71. ABBREVIATE WORD `PAGES’ (if not using external language file) <<STORED JOURNAL NAMES: (*) Full journal names for prestored journals (a) Abbreviated journal names (s) Abbreviated with astronomy shorthands like ApJ and AJ **a**
72. AMPERSAND: (*) Use word `and' in author lists (a) Use ampersand in place of`and’ (v) Use \BIBand in place of `and’ **默认**
73. COMMA BEFORE `AND': (*) Comma before`and’ as `Tom, Dick, and Harry' (n) No comma before`and’ as `Tom, Dick and Harry' (c) Comma even with 2 authors as`Tom, and Harry’ **默认**
74. NO `AND' IN REFERENCE LIST: (*) With`and’ before last author in reference list (x) No `and' as`Tom, Dick, Harry’ Select: x
    73.COMMA BEFORE `ET AL': (*) Comma before`et al’ in reference list (x) No comma before `et al’ **默认**
75. FONT OF `ET AL’: (*) Plain et al (i) Italic et al ® Roman et al even when authors something else **默认**
76. ADDITIONAL REVTeX DATA FIELDS: (*) No additional fields for REVTeX ® Include REVTeX data fields collaboration, eid, eprint, archive, numpages, u rl **默认**
77. E-PRINT DATA FIELD: (without REVTeX fields) (*) Do not include eprint field (e) Include eprint and archive fields for electronic publications **默认**
78. E-PRINT DATA FIELD: (without REVTeX fields) <<URL ADDRESS: (without REVTeX fields) (*) No URL for electronic (Internet) documents (u) Include URL as regular item block (n) URL as note (l) URL on new line after rest of reference **默认**
79. REFERENCE COMPONENT TAGS: (*) No reference component tags in the \bibitem entries (b) Reference component tags like \bibinfo in the content of \bibitem Select: **默认**
80. EMPHASIS: (affects all so-called italics) (*) Use emphasis ie, \em, allows font switching (i) Use true italics ie, \it, absolute italics (x) No italics at all (u) Underlining in place of italics, best with ulem package **x **
81. COMPATIBILITY WITH PLAIN TEX: (*) Use LaTeX commands which may not work with Plain TeX (t) Use only Plain TeX commands for fonts and testing **默认**
82. COMPATIBILITY WITH PLAIN TEX: ) Finished!! Batch job written to file `fly.dbj’
83. Shall I now run this batch job? (NO) \yn=yes

**回答完上述所有问题，在当前路径下可以得到一个makebst文档记录了你之前回答的问题，还有一个一个fly.bst文件。将这个fly.bst文件复制到编辑的latex所在的文件夹中。**

## 修改bst文件：在期和卷之间加逗号

编译.tex文件，得到如下列出的参考文献如下：
[6] Borwn, L., Thomas, H., James, C., et al.:'The title of the paper, IET Communications, 2012, 6 (5), pp 125-138
但是要求的格式是vol, num。还记得在问题39里面，只选了个相近的vol (num)，不符合要求。因此需要修改bst。
用Winedt打开bst文件，**ctrl+F**查找**format.vol.num.pages**,

```typescript
FUNCTION {format.vol.num.pages}
{ volume field.or.null
  duplicate$ empty$ 'skip$
    {
      "volume" bibinfo.check
    }
  if$
  number "number" bibinfo.check duplicate$ empty$ 'skip$
    {
      swap$ duplicate$ empty$
        { "there's a number but no volume in " cite$ * warning$ }
        'skip$
      if$
      swap$
      "~(" swap$ * ")" *
    }
  if$ *
}
```

修改方法如下：

```typescript
将"~(" swap$ * ")" *修改为",~(" swap$ * ")" * 添加的逗号就是vol,num中的逗号
1
```

将上次编译生成的bbl删除，就可以得到正确的参考文献格式
[6] Borwn, L., Thomas, H., James, C., et al.:'The title of the paper, IET Communications, 2012, 6, (5), pp 125-138

