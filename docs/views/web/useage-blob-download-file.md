---
author: 星火燎原@vxhly
title: 使用 Blob 来下载文件
categories: [web]
tags: [Code, JavaScript]
date: 2020-10-13 18:00:26
---

<Boxx changeTime="10000"/>

::: tip 前言
当你的网页只有文件的 URL 时, 我们可以通过 Blob 来创建文件流下载文件。
:::
<!-- more -->

## 上代码

``` javascript
export function downloadFile(url, fileName = '') {
  return downloadBlob(url, fileName)
    .then(resp => {
      if (resp.blob) {
        return resp.blob()
      } else {
        return new Blob([resp])
      }
    })
    .then(blob => URL.createObjectURL(blob))
    .then(url => {
      downloadURL(url, fileName)
      URL.revokeObjectURL(url)
    })
    .catch(err => {
      throw new Error(err.message)
    })
}

function downloadBlob(url) {
  return new Promise((resolve, reject) => {
    var xhr = new XMLHttpRequest()
    xhr.open('GET', url)
    xhr.responseType = 'blob'

    xhr.onload = function () {
      if (xhr.status === 200) {
        resolve(xhr.response)
      } else {
        reject(new Error(xhr.statusText || 'Download failed.'))
      }
    }
    xhr.onerror = function () {
      reject(new Error('Download failed.'))
    }
    xhr.send()
  })
}

function downloadURL(url, name = '') {
  const link = document.createElement('a')
  link.download = name
  link.href = url
  if ('download' in document.createElement('a')) {
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  } else {
    // 对不支持download进行兼容
    click(link, (link.target = '_blank'))
  }
}

function click(node) {
  try {
    node.dispatchEvent(new MouseEvent("click"));
  } catch (e) {
    var evt = document.createEvent("MouseEvents");
    evt.initMouseEvent(
      "click",
      true,
      true,
      window,
      0,
      0,
      0,
      80,
      20,
      false,
      false,
      false,
      false,
      0,
      null
    );
    node.dispatchEvent(evt);
  }
}
```