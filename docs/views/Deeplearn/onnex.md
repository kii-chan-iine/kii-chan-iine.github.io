---
author: kii
title: Onnx
categories: [深度学习]
tags: [deeplearn]
date: 2023-05-07 17:00:00
---

<Boxx changeTime="10000"/>

::: tip 前言
onnx。
:::

<!-- more -->

# Onnx

## 单个图片

```python
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.1307], std=[0.3081]),transforms.Resize(size=[112,112]),transforms.ToPILImage()])

img=Image.open(lists[0])
print(np.array(img).shape)


img = transform(img)

# ts=transforms.ToTensor()
# img=ts(img)
# ss=transforms.ToPILImage()
# img=ss(img)

# ress = transforms.Resize(size=[112,112])
# img = ress(img)

print(np.array(img).shape)

plt.imshow(img)
plt.show()
```


## 多个图片

```python
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

lists=[]
for r,d,f in os.walk('../../datasets/CASIA-FaceV5/CASIA-FaceV5 (000-099)/'):
    for file in f:
#         lists.append(os.path.join(r,file)+'\n')
        lists.append(os.path.join(r,file))
transform=transforms.Compose([transforms.ToTensor(),transforms.Resize(size=[112,112])])

data=[]
for i in lists:
    img=Image.open(i)
    img=transform(img)
    img=np.array(img)
#     img=torch.unsqueeze(transform(img),0)
#     print(img.shape)
    data.append(img)

data=np.array(data)

sess = rt.InferenceSession('r100.onnx',providers=['CUDAExecutionProvider'])
input_name = sess.get_inputs()[0].name  

output_name = sess.get_outputs()[0].name
pred_onnx = sess.run([output_name], {input_name: data})

print("outputs:")
print(np.array(pred_onnx)[0].shape)
```


## 多层

```python
import cv2
import os
import glob
import subprocess
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import onnx
import sys
import shutil
import copy
from collections import OrderedDict

def normalize(img,scale=None,mean=None,std=None):
    
    if isinstance(scale, str):
        scale = eval(scale)

    scale = np.float32(scale if scale is not None else 1.0 / 255.0)

    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]

    shape =  (1, 1, 3)
    mean = np.array(mean).reshape(shape).astype('float32')
    std = np.array(std).reshape(shape).astype('float32')
    assert isinstance(img,
                    np.ndarray), "invalid input 'img' in normalize"
    img = (img.astype('float32') * scale - mean) / std
    return img
model = onnx.load("r18.onnx")
# 模型推理
ori_output = copy.deepcopy(model .graph.output)
# 输出模型每层的输出
for node in model.graph.node:
    for output in node.output:
        if output not in ori_output:
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])

#进行配置
if ort.get_device()=="CPU":
    config = ort.SessionOptions()
    ret,val=subprocess.getstatusoutput("cat /proc/cpuinfo | grep 'core id' |sort |uniq | wc -l")
    if ret==0:
        cpu_num_thread = int(val)
    else:
        cpu_num_thread=4
    config.intra_op_num_threads = cpu_num_thread
    config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers=["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(model.SerializeToString(),providers=providers,sess_options=config)
elif ort.get_device()=="GPU":
    providers=["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(model.SerializeToString(),providers=providers)

image_list=["1.jpeg",'2.jpeg']
# for root,dir,files in os.walk('need_test/crop/'):
#     if len(files):
#         for ff in files:
#             n = os.path.join(root,ff)
#             image_list.append(n)

for img_path in tqdm(image_list):
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = img[:, :, ::-1]
    img = cv2.resize(img,(112,112))
    img = normalize(img)
    img = img.transpose((2,0,1))
    image = np.expand_dims(img,axis=0)
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    #获取所有节点输出
    outputs = [x.name for x in ort_session.get_outputs()] #ort_session.get_outputs()[0].name是原模型的单一输出
    print(outputs)
    ort_outs = ort_session.run(output_names=outputs, input_feed=ort_inputs)    
    # 生成字典，便于查找层对应输出
    ort_outs = OrderedDict(zip(outputs, ort_outs))
    
# ort_outs
```