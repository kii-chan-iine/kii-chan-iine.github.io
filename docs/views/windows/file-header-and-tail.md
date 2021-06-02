---
author: 星火燎原@vxhly
title: 常见的文件头和文件尾
categories: [windows]
tags: [Windows, Pentest] 
date: 2016-11-23 10:05:44
---

<Boxx changeTime="10000"/>

::: tip 前言
文件头是位于文件开头的一段承担一定任务的数据, 一般都在开头的部分。而文件尾对应的就在 文件的结束部分, 本篇将介绍常见的文件头和文件尾的十六进制数, 这些文件的文件头和文件尾几乎多是一样的。
:::
<!-- more -->

## 从 Ultraedit 提取的文件头信息

| 文件类型                       | 文件头                                    | 文件尾               |
|--------------------------------|-------------------------------------------|----------------------|
| JPEG ( `jpg` )                   | FF D8 FF                                  | FF D9                |
| PNG ( `png` )                    | 89 50 4E 47                               | AE 42 60 82          |
| GIF ( `gif` )                    | 47 49 46 38                               | 00 3B                |
| TIFF ( `tif` )                   | 49 49 2A 00                               |                      |
| Windows Bitmap ( `bmp` )         | 42 4D                                     |                      |
| ZIP Archive ( `zip` )            | 50 4B 03 04                               | 50 4B                |
| RAR Archive ( `rar` )            | 52 61 72 21                               | 07 00                |
| CAD ( `dwg` )                    | 41 43 31 30                               |                      |
| Adobe Photoshop ( `psd` )        | 38 42 50 53                               |                      |
| Rich Text Format ( `rtf` )       | 7B 5C 72 74 66                            |                      |
| XML ( `xml` )                    | 3C 3F 78 6D 6C                            |                      |
| HTML ( `html` )                  | 3C 68 74 6D 6C                            | 3C 2F 68 74 6D 6C 3E |
| Email thorough only ( `eml` )    | 44 65 6C 69 76 65 72 79 2D 64 61 74 65 3A |                      |
| Outlook Express ( `dbx` )        | CF AD 12 FE C5 FD 74 6F                   |                      |
| Outlook ( `pst` )                | 21 42 44 4E                               |                      |
| MS Word/Excel ( `xls` or `doc` ) | D0 CF 11 E0                               |                      |
| MS Access ( `mdb` )              | 53 74 61 6E 64 61 72 64 20 4A             |                      |
| WordPerfect ( `wpd` )            | FF 57 50 43                               |                      |
| Postscript ( `eps` or `ps` )     | 25 21 50 53 2D 41 64 6F 62 65             |                      |
| Adobe Acrobat ( `pdf` )          | 25 50 44 46 2D 31 2E                      |                      |
| Quicken ( `qdf` )                | AC 9E BD 8F                               |                      |
| Windows Password ( `pwl` )       | AC 9E BD 8F                               |                      |
| Wave ( `wav` )                   | 57 41 56 45                               |                      |
| AVI ( `avi` )                    | 41 56 49 20                               |                      |
| AVI ( `avi` )                    | 41 56 49 20                               |                      |
| Real Audio ( `ram` )             | 2E 72 61 FD                               |                      |
| Real Media ( `rm` )              | 2E 52 4D 46                               |                      |
| MPEG ( `mpg` )                   | 00 00 01 BA                               |                      |
| MPEG ( `mpg` )                   | 00 00 01 B3                               |                      |
| Quicktime ( `mov` )              | 6D 6F 6F 76                               |                      |
| Windows Media ( `asf` )          | 30 26 B2 75 8E 66 CF 11                   |                      |
| MIDI ( `mid` )                   | 4D 54 68 64                               |                      |

## 从 winhex 提取的文件头信息

| 文件                   | 文件后缀                     | 文件头             |
|------------------------|------------------------------|--------------------|
| JPEG                   | `jpg` ; `jpeg` | 0xFFD8FF           |
| PNG                    | `png` | 0x89504E470D0A1A0A |
| GIF                    | `gif` | GIF8               |
| TIFF                   | `tif` ; `tiff` | 0x49492A00         |
| Bit map                | `bmp` | BM                 |
| AOL ART                | `art` | 0x4A47040E000000   |
| PC Paintbrush          | `pcx` | 0x0A050108         |
| Graphics Metafile      | `wmf` | 0xD7CDC69A         |
| Enhanced Metafile      | `emf` | 0x0100000058000000 |
| Corel Draw             | `cdr` | CDR                |
| CAD                    | `dwg` | 0x41433130         |
| Adobe Photoshop        | `psd` | 8BPS               |
| Rich Text Format       | `rtf` | rtf                |
| XML                    | `xml` |                    |
| HTML                   | `html` ; `htm` ; `php` | type               |
| Email                  | `eml` | Delivery-date:     |
| Outlook Express        | `dbx` | 0xCFAD12FE         |
| MS Office/OLE2         | `doc` ; `xls` ; `dot` ; `ppt` ; `db` | 0xD0CF11E0A1B11AE1 |
| MS Access              | `mdb` ; `mda` ; `mde` ; `mdt` | Standard J         |
| WordPerfect            | `wpd` | 0xFF575043         |
| OpenOffice Writer      | `sxw` | writer             |
| OpenOffice Calc        | `sxc` | calc               |
| WordPerfect            | `wpd` | 0xFF575043         |
| OpenOffice Writer      | `sxw` | writer             |
| OpenOffice Calc        | `sxc` | calc               |
| OpenOffice Math        | `sxm` | math               |
| Windows Password       | `pwl` | 0xE3828596         |
| ZIP Archive            | `zip` ; `jar` | 0x504B0304         |
| ZIP Archive (outdated) | `zip` | 0x504B3030         |
| RAR Archive            | `rar` | Rar!               |
| GZ Archive             | `gz` ; `tgz` | 0x1F8B08           |
| BZIP Archive           | `bz2` | BZh                |
| ARJ Archive            | `arj` | 0x60EA             |
| 7－ZIP Archive         | `7z` | 7z 集'             |
| Wave                   | `wav` | WAVE               |
| AVI                    | `avi` | AVI                |
| Real Audio             | `ram` ; `ra` | .ra?0              |
| Real Media             | `rm` | . RMF               |
| MPEG                   | `mpg` ; `mpeg` | 0x000001BA         |
| MPEG                   | `mpg` ; `mpeg` | 0x000001B3         |
| F575043                |                              |                    |
| OpenOffice Writer      | `sxw` | writer             |
| OpenOffice Calc        | `sxc` | calc               |
| OpenOffice Math        | `sxm` | math               |
| Windows Password       | `pwl` | 0xE3828596         |
| ZIP Archive            | `zip` ; `jar` | 0x504B0304         |
| ZIP Archive (outdated) | `zip` | 0x504B3030         |
| RAR Archive            | `rar` | Rar!               |
| GZ Archive             | `gz` ; `tgz` | 0x1F8B08           |
| BZIP Archive           | `bz2` | BZh                |
| ARJ Archive            | `arj` | 0x60EA             |
| 7－ZIP Archive         | `7z` | 7z 集'             |
| Wave                   | `wav` | WAVE               |
| AVI                    | `avi` | AVI                |
| Real Audio             | `ram` ; `ra` | .ra?0              |
| Real Media             | `rm` | . RMF               |
| MPEG                   | `mpg` ; `mpeg` | 0x000001BA         |
| MPEG                   | `mpg` ; `mpeg` | 0x000001B3         |
| F575043                |                              |                    |
| OpenOffice Writer      | `sxw` | writer             |
| OpenOffice Calc        | `sxc` | calc               |
| OpenOffice Math        | `sxm` | math               |
| Windows Password       | `pwl` | 0xE3828596         |
| ZIP Archive            | `zip` ; `jar` | 0x504B0304         |
| ZIP Archive (outdated) | `zip` | 0x504B3030         |
| RAR Archive            | `rar` | Rar!               |
| GZ Archive             | `gz` ; `tgz` | 0x1F8B08           |
| BZIP Archive           | `bz2` | BZh                |
| ARJ Archive            | `arj` | 0x60EA             |
| 7－ZIP Archive         | `7z` | 7z 集'             |
| Wave                   | `wav` | WAVE               |
| AVI                    | `avi` | AVI                |
| Real Audio             | `ram` ; `ra` | .ra?0              |
| Real Media             | `rm` | . RMF               |
| MPEG                   | `mpg` ; `mpeg` | 0x000001BA         |
| MPEG                   | `mpg` ; `mpeg` | 0x000001B3         |

