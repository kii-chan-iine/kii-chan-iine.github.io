---
author: kii
title: 学习札记
categories: [CV]
tags: [CV,DL]
date: 2023-12-04 00:44:30
---

<Boxx changeTime="10000"/>

::: tip 前言

rush KEXN

:::

<!-- more -->

Data Process

```python
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

age = [67.0,  22.0,  49.0,  45.0,  53.0,  35.0, 53.0, 35.0, 61.0,
       28.0,  25.0, 24.0, 22.0, 60.0, 28.0]
target = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0]

dic = {'age':age, 'target':target}

df = pd.DataFrame(dic)
df.head(3)
```

main

```python
class ChiSqureMerge:
    def __cal_chi2(self, df, col, target):
        '''
        用来计算区间内的卡方值，其中期望频率是按照区间内样本分布来计算的。该函数为私有函数，不在类外使用。
        df：DataFrame，列联表，至少包含:col, target, good_target, bad_cnt, good_cnt这几列
        col：str，待分箱列名
        target：str，label值列名，0、1
        return：
            chi2：float，卡方值
        '''
        bindf = df.copy()
        bad_rate = bindf["bad_cnt"].sum() / (bindf["bad_cnt"].sum() + bindf["good_cnt"].sum())
        good_rate = 1 - bad_rate
        bindf["expect_good_cnt"] = (bindf["bad_cnt"] + bindf["good_cnt"]) * good_rate
        bindf["expect_bad_cnt"] = (bindf["bad_cnt"] + bindf["good_cnt"]) * bad_rate
        # 这里为了防止期望频数为0导致零除报错，简单处理将分母+1
        bindf["chi2"] = (bindf["good_cnt"] - bindf["expect_good_cnt"]) ** 2 / (bindf["expect_good_cnt"] + 1) + \
(bindf["bad_cnt"] - bindf["expect_bad_cnt"]) ** 2 / (bindf["expect_bad_cnt"] + 1)
        return bindf["chi2"].sum()

    def chi2_merge(self, df, col, target, maxInterval=5, chi_threshold=None):
        '''
        1、根据最大区间数限制
        2、最小卡方值限制
        df：DataFrame，原始数据集，至少包含：col, target
        col：str，待分箱列
        target：str， label值列名，0、1
        maxInterval：int，最少箱数
        chi_threshold：float，卡方阈值，当前轮最小的卡方值如果大于阈值就停止
        return:
            bins：list in list箱。
        '''
        bindf = df[[col, target]].copy()
        bindf["good_target"] = 1 - bindf[target]
        # 获得频数列联表
        bindf = pd.crosstab(bindf[col], bindf[target]).reset_index()
        # print(bindf)

        bindf.columns = [col, "good_cnt", "bad_cnt"]
        # print(bindf)

        bin_cnt = len(bindf) #初始化段的个数
        # 初始化桶
        bins = [[i] for i in bindf[col].values.tolist()]
        if chi_threshold is None:
            chi_threshold = 1e9
        min_chiSquare = 0.
        while (min_chiSquare < chi_threshold and bin_cnt > maxInterval): #两个终止条件
            bins_chi2 = []
            for idx, val in enumerate(bins[:-1]):
                # 合并区间
                start_index = bindf[bindf[col] == val[0]].index.tolist()[0]
                end_index = bindf[bindf[col] == bins[idx+1][-1]].index.tolist()[0]
                bins_chi2.append((idx, self.__cal_chi2(bindf.loc[start_index:end_index, :], col, target)))
            min_dix, min_chiSquare = sorted(bins_chi2, key=lambda x:x[1])[0] #每次合并最小的，如果多个区间之间的卡方均小，不用处理
            bins[min_dix] += bins[min_dix+1]
            bins.pop(min_dix + 1)
            bin_cnt = len(bins)
        return bins
```

result

```python
chi2m = ChiSqureMerge()
bins = chi2m.chi2_merge(df, "age", "target")
bins

#%%
# 共产生了5个箱，即(19, 25]、(25, 34]、(34,52]、(52, 61]、(61, 75]，这里最大值 75 和最小值 19 只是针对这1000个数据而言的，为了使分好的箱兼容将来新的数据，可以对界限稍作处理。即(-1e9, 25]、(25, 34]、(34,52]、(52, 61]、(61, 1e9]，这里正负1e9代表以上（或以下）。

bins = [b[-1] for b in bins]
print(bins)
bins = [-1e9, ] + bins[:-1] + [1e9]
bins
#%%
df["age_cut"] = pd.cut(df["age"], bins=bins, labels=["a", "b", "c", "d", "e"])
# df.head()
```

numpy

```python
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

age = [67.0,  22.0,  49.0,  45.0,  53.0,  35.0, 53.0, 35.0, 61.0,
       28.0,  25.0, 24.0, 22.0, 60.0, 28.0]
target = [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0]

xrange = sorted(list(set(age)))
xrange.append(200000)
yrange = [0,1,2]
crosstab = np.histogram2d(age, target, bins=(xrange, yrange))
# crosstab

def calx2(data):
    data =data.copy()
    gr = sum(data[:,0]) / np.sum(data)
    br = sum(data[:,1]) / np.sum(data)
    gexp = np.sum(data, axis=1) * gr
    bexp = np.sum(data, axis=1) * br

    x2 = (data[:,0] - gexp)**2/(gexp + 1) + (data[:,1] - bexp)**2/(bexp + 1)
    return sum(x2)

def merge(x, y, maxnum=5, thr=None):
    x = x.copy()

    # init
    bins = [[i] for i in y]
    bins_len = len(bins)
    if thr == None:
        thr = 10e10
    thrmin = 0

    yy = list(y)
    while (bins_len > maxnum and thrmin < thr):
        bins_cnt = []
        for k,v in enumerate(bins[:-1]):
            start_ = yy.index(v[0])
            end_ = yy.index(bins[k + 1][-1])
            bins_cnt.append((k, calx2(x[start_:end_+1,:])))
        ind, thrmin = sorted(bins_cnt, key=lambda x:x[1])[0]
        bins[ind] += bins[ind + 1]
        bins.pop(ind + 1)
        bins_len = len(bins)
    return bins


x0 = crosstab[0][:2,:]
calx2(x0)

merge(crosstab[0], crosstab[1][:-1])
```

ms

```python
import numpy as np


def iou(a,b):
    mx0 = max(a[0], b[0]) #左上 max
    mx1 = max(a[1], b[1])
    mn2 = min(a[2], b[2]) #右下 min
    mn3 = min(a[3], b[3])

    intra = max(0, (mn2 -mx0 + 1)) * max(0, mn3 -mx1 + 1)  #容易错
    areaa = (a[2] -a[0] + 1)*(a[3] - a[1] + 1)
    areab = (b[2] -b[0] + 1)*(b[3] - b[1] + 1)
    return intra/(areaa + areab -intra)

def gaussscore(score, iou, rho=0.5):
    score = score * np.exp(-(iou *iou)/rho)
    return score

def get_res(boxx,thr=0.3):
    res = []
    while (boxx):
        boxx.sort(key = lambda x:-x[4])
        res.append(boxx[0])
        new = map(lambda x:str(x), boxx[0][:-1])
        print(' '.join(new))
        tmp1 = boxx[1:].copy()
        tmp2 = boxx[1:].copy()

        for i in range(len(tmp2)):
            tiou = iou(res[-1][:-1], tmp2[i][:-1])
            if tiou > 0.3:
                new_score = gaussscore(tmp2[i][-1], tiou)
                if new_score < 0.3:
                    tmp1.remove(tmp2[i])
                else:
                    tmp1[i][-1] = new_score
            elif tmp2[i][-1] < 0.3:
                tmp1.remove(tmp2[i])
        boxx = tmp1

    return res


box1 = [
    [2, 2, 32, 32, 0.9], 
    [3, 3, 30, 30, 0.8], 
    [45, 45, 60, 60, 0.88], 
    [50, 50, 60, 60, 0.75],
    [33,33,46,46,0.85]
]

box2 = [
    [2, 2, 50, 50, 0.88],
    [3, 3, 49, 49, 0.9],
    [3, 3, 30, 30, 0.25],
    [5, 5, 51, 51, 0.97],
    [6, 6,47, 47, 0.65],
    [10,10,80,80, 0.34],
    [20,20,70,70, 0.46]
]

box3 =[
    [5, 5, 35, 35, 0.9],
    [3, 3, 30, 30, 0.82],
    [45, 45, 60, 60, 0.88],
    [60, 60, 80, 80, 0.75],
    [33, 33,46, 46, 0.85],
    [2,2,20,20, 0.95],
    [4,4,50,40, 0.35],
    [40,40,100,100, 0.65],
    [21,21,91,95, 0.73],
    [15,30,71,74, 0.99],
]
boxs = [box1, box2, box3]

for i in boxs:
    get_res(i)
    print('--')
```
