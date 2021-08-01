---
author: kii
title: LeetCode
categories: [深度学习]
tags: [LeetCode]
date: 2021-06-15 20:11:30
---

<Boxx changeTime="10000"/>

::: tip 前言
Leetcode
:::
<!-- more -->

# 动态规划

## 台阶跳，斐波那契数列





## 使用最小价怕楼梯

> 数组的每个下标作为一个阶梯，第 i 个阶梯对应着一个非负数的体力花费值 cost[i]（下标从 0 开始）。
>
> 每当你爬上一个阶梯你都要花费对应的体力值，一旦支付了相应的体力值，你就可以选择向上爬一个阶梯或者爬两个阶梯。
>
> 请你找出达到楼层顶部的最低花费。在开始时，你可以选择从下标为 0 或 1 的元素作为初始阶梯。
>
> <font color='red'>我认为官方题有误</font>

```
class Solution:
    def minCostClimbingStairs(self, cost):


        if len(cost)>4:
            dp1,dp2=cost[0],cost[1]
            for i in range(4,len(cost)):#注意
                dp1,dp2=dp2,min(dp2+cost[i-1],dp1+cost[i-2])#逻辑
        elif len(cost)==4:
            return min(cost[0]+cost[3],cost[2])
        elif len(cost)==3:
            return min(cost[0],cost[1])
        else:
            return 0

        return dp2

x=Solution()
ff=x.minCostClimbingStairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1])



# ff=x.minCostClimbingStairs([10, 15, 20,31,40,16])
print(ff)
```



## 编辑距离

Q：

给定两个单词 word1 和 word2，计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符
示例 1:

输入: word1 = "horse", word2 = "ros"
输出: 3
解释: 
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
示例 2:

输入: word1 = "intention", word2 = "execution"
输出: 5
解释: 
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')

### 动态规划

动态规划状态矩阵填充word1-> word2

以下矩阵行为word2，列为word1

|      |      | S    | a    | t    | u    | r    | d    | a    | y(m) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|      | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    |
| S    | 1    | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
| u    | 2    | 1    | 1    | 2    | 2    | 3    | ?4   |      |      |
| n    | 3    |      |      |      |      |      |      |      |      |
| d    | 4    |      |      |      |      |      |      |      |      |
| a    | 5    |      |      |      |      |      |      |      |      |
| y(n) | 6    |      |      |      |      |      |      |      |      |

> 其实就是计算 `min(res[i-1][j]+1,res[i][j-1]+1,res[i-1][j-1]+cost)`,其中`cost`是所在位置`res[i][j]`对应的字幕是否相等，如果相等，那么就不需要替换，cost=0；否则cost=1。注意`res[i][j]`对应的是word的`[i-1][j-1]`

分析：从以上的递归算法（初始位置在字符串尾部）中，可推断出状态转移方程如下。

```python
res[i][j] = res[i-1][j-1] if word1[i] = word2[j]
res[i][j] = min(res[i][j-1], res[i-1][j], res[i-1][j-1])+1 if word1[i] != word2[j]
```

**动态转移方向**为从上往下，从左往右

**初始值**，比较`` word1[0]`` 与 ``word2[0]``，此时需要知道 ``res[-1][-1]`` 的情况，添加第 0 行和第 0 列，设置大小为 (n+1)*(m+1) 的 dres数组方便计算，n 为 word1 的长度，m 为 word2的长度。

- 设置 `` res[0][0]=0 ``表示 word1 和 word2 皆为空，操作数为 0；

- 设置 ``res[1][0]~res[n][0]`` 为 0~n，表示 word2 为空时，word1 需要删除的操作数；

- 设置 ``res[0][1]~res[0][m]`` 为 0~m，表示 word1 为空时，word1 需要插入的操作数。

**返回值**，``res[n][m]`` 表示 word1 与 word2已遍历结束的最小操作数，即 ``word1[n-1]`` 与 ``word2[m-1]``处。

```python
class Solution:
    # def minDistance(self, word1: str, word2: str) -> int:
    def minDistance(self, word1, word2):
        n,m=len(word1),len(word2)
        #初始化res数组
        #res = [[0]*(m+1)  for _ in range(n + 1)]
        res = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
        for i in range(n+1):
            res[i][0] = i
        for j in range(m+1):
            res[0][j] = j
        # 更新 res 数组
        for i in range(1, n+1):
            for j in range(1, m+1):
                res[i][j] = min(res[i - 1][j] + 1, res[i][j - 1] + 1)
                if word1[i - 1] == word2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                res[i][j] = min(res[i][j], res[i - 1][j - 1] + cost)
        return res[n][m]
```

```python
test = Solution()
word1_li = ["horse", "intention"]
word2_li = ["ros", "execution"]
for word1, word2 in zip(word1_li, word2_li):
    print(test.minDistance(word1, word2))
```

### 记忆递归
分析：普通的递归方案中存在大量的重叠子问题，如下示例，因此可采用携带记忆的递归方式进行剪枝。

示例：目的为 `dp[i][j] --> dp[i-1][j-1]` 可通过如下 3 种路线到达

``dp[i][j] --> dp[i-1][j-1]``
`dp[i][j] --> dp[i-1][j] --> dp[i][j-1]`
`dp[i][j] --> dp[i][j-1] --> dp[i-1][j]`

```python
# 携带记忆的递归
class Solution:
    def minDistance(self, word1, word2):
        memo = dict()                   # 记忆
        def dp_memo(i, j):
            if i==-1: 
                memo[(i, j)] = j+1
                return memo[(i, j)]    
            if j==-1: 
                memo[(i, j)] = i+1
                return memo[(i, j)]    
            
            if (i, j) in memo:         # 若该状态存在记忆中，则直接返回
                return memo[(i, j)]
            
            if word1[i] == word2[j]:       
                memo[(i, j)] = dp_memo(i-1, j-1)
                return memo[(i, j)]
            else:                          
                memo[(i, j)] = min(dp_memo(i, j-1)+1, dp_memo(i-1, j)+1, dp_memo(i-1, j-1)+1)
                return memo[(i, j)] 
        # 调用携带记忆的递归函数 
        return dp_memo(len(word1)-1, len(word2)-1)
```


### 递归（超时）
 分析背景：通过改变 word1 使其与 word2 相同，计算需要操作的次数。
i、j 分别指向 word1、word2 中的某位置（初始指向字符串尾部）
若 word1[i]=word2[j]，则编辑距离为 0，不需要进行操作，此时需要同时将 i、j 左移。
若 word1[i]！=word2[j]，则需要进行插入、删除、替换操作使得对应字符相同：
对 word1 的 i 位置后进行插入字符操作，此时将 j 左移，操作数 +1；
对 word1 的 i 位置处字符进行替换操作，此时将 i 和 j 同时左移，操作数 +1；
对 word1 的 i 位置处字符进行删除操作，此时将 i 左移，操作数 +1。
最终取word1=word2时最小的操作数

使用递归将所有情况遍历，返回满足条件的最小操作数
```python
# 递归
class Solution:
    def minDistance(self, word1, word2):
        n1, n2 = len(word1), len(word2)
        
        def dp(i, j):
            if i==-1: return j+1    # word1 遍历完，返回 word2 的长度，即需要添加的步数
            if j==-1: return i+1    # word2 遍历完，返回 word1 的长度，即需要删除的步数
            
            if word1[i] == word2[j]:       # 若字符串对应位置相等，则指针左移，不做操作
                return dp(i-1, j-1)
            else:                          # 若字符串对应位置不相等，则进行插入、删除、替换操作
                return min(
                            dp(i, j-1)+1,     # 插入操作
                            dp(i-1, j)+1,     # 删除操作
                            dp(i-1, j-1)+1    # 替换操作
                            )
        # 调用递归函数 
        return dp(n1-1, n2-1)
```

# 动态规划

## 最大子序和 53





### 动态规划

*f*(*i*)=max{*f*(*i*−1)+*nums*[*i*],*nums*[*i*]}

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        pre,maxre=0,nums[0]
        for i in nums:
            pre=max(i,pre+i)
            maxre=max(maxre,pre)
        return maxre
```

这里有一个动态转移方程 fn=max(fn-1+nums[i],nums[i])，此外还得有一个记录最大的和。

### 分治(**线段树求解最长公共上升子序列问题**的pushup)

```
```

 ### s

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_ = float('-inf')
        sum_ = 0
        for i in range(len(nums)):
            sum_ += nums[i]
            if sum_ > max_:
                max_ = sum_
            if sum_ < 0:
                sum_ = 0
        return max_
```

## 打家劫舍 198

> 用 \textit{dp}[i]dp[i] 表示前 ii 间房屋能偷窃到的最高总金额，那么就有如下的状态转移方程：
>
> \textit{dp}[i] = \max(\textit{dp}[i-2]+\textit{nums}[i], \textit{dp}[i-1])
> dp[i]=max(dp[i−2]+nums[i],dp[i−1])
>
> 边界条件为：
>
> \begin{cases} \textit{dp}[0] = \textit{nums}[0] & 只有一间房屋，则偷窃该房屋 \\ \textit{dp}[1] = \max(\textit{nums}[0], \textit{nums}[1]) & 只有两间房屋，选择其中金额较高的房屋进行偷窃 \end{cases}
> { 
> dp[0]=nums[0]
> dp[1]=max(nums[0],nums[1])
>
>
> 只有一间房屋，则偷窃该房屋
> 只有两间房屋，选择其中金额较高的房屋进行偷窃
>
>  
>
> 最终的答案即为 \textit{dp}[n-1]dp[n−1]，其中 nn 是数组的长度。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums)<3: #先处理边界
            return max(nums)
        fn=0
        fnext=0
        # max_=0
        for i in range(len(nums)):
            fn,fnext=fnext,max(fn+nums[i],fnext)# 动态转移
        return fnext

#f(i)=max(f(i-2)+nums[i],f(i-1))
#fn,fnext=fnext,max(fn+nums[i],fnext)


```

## 打家劫舍2 213

数组首位相连

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums)<3:
            return max(nums)
        fn,fnext=0,0
        for i in range(len(nums)-1):
            fn,fnext=fnext,max(fn+nums[i],fnext)
        gn,gnext=0,0
        for i in range(1,len(nums)):
            gn,gnext=gnext,max(gn+nums[i],gnext)
        return max(fnext,gnext)
```

> 其实就是把环拆成两个队列，一个是从0到n-1，另一个是从1到n，然后返回两个结果最大的。

##  256. 粉刷房子

![image-20210801165441692](https://imagerk.oss-cn-beijing.aliyuncs.com/img/image-20210801165441692.png)

> 这里可以使用 3 个长度的数组或者变量。数组在于可以编写较简洁的代码，如果你被要求使用 mm 个颜色的算法，它们也会更好修改。我们在这里使用数组，因为跟踪 6 个独立的变量太麻烦了。
>
> 在每个步骤中，我们通过添加前一行的值来更新当前行中的值。然后，我们将前一行设置为当前行，并转到下一个值。重复该过程。
>

```python
import copy

class Solution:
    def minCost(self, costs: List[List[int]]) -> int:

        if len(costs) == 0: return 0

        previous_row = costs[-1]
        for n in reversed(range(len(costs) - 1)):

            current_row = copy.deepcopy(costs[n])#这里保证操作不对costs中的数据造成影响
            # Total cost of painting nth house red?
            current_row[0] += min(previous_row[1], previous_row[2])
            # Total cost of painting nth house green?
            current_row[1] += min(previous_row[0], previous_row[2])
            # Total cost of painting nth house blue?
            current_row[2] += min(previous_row[0], previous_row[1])
            previous_row = current_row

        return min(previous_row)

作者：LeetCode-官方题解
```

```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        n = len(costs)
        dp = [[0] * 3 for _ in range(n)]
        for i in range(3):
            dp[0][i] = costs[0][i]
        for i in range(1, n):
            dp[i][0] = min(dp[i - 1][1], dp[i - 1][2]) + costs[i][0]
            dp[i][1] = min(dp[i - 1][0], dp[i - 1][2]) + costs[i][1]
            dp[i][2] = min(dp[i - 1][1], dp[i - 1][0]) + costs[i][2]
        return min(dp[-1])
```

```python
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        if len(costs)==0: return 0
        previous_row=costs[-1]
        for n in range(len(costs)-2,-1,-1):
            current_row=costs[n]
            current_row[0]+=min(previous_row[1],previous_row[2])
            current_row[1]+=min(previous_row[0],previous_row[2])
            current_row[2]+=min(previous_row[1],previous_row[0])
            previous_row=current_row
        return min(previous_row)
```



##  265. 粉刷房子 II

把前面的题改成k种颜色

```python
import copy
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        if len(costs)==0:
            return 0
        pre_=costs[-1]
        for i in range(len(costs)-2,-1,-1):
            cur_=costs[i]
            for j in range(len(costs[0])):
                temp=copy.deepcopy(pre_)
                temp.remove(temp[j])
                cur_[j]+=min(temp)
            pre_=cur_
        return min(pre_)
#My
```





```python
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        n, k = len(costs), len(costs[0])
        if k == 1:
            return sum(costs[0])
        bestCost = [-1, 0]
        sbCost = [-1, 0]
        for i in range(n):
            nbCost = [-1, float('inf')]
            snbCost = [-1, float('inf')]
            for j in range(k):
                if j != bestCost[0]:
                    myCost = bestCost[1] + costs[i][j]
                else:
                    myCost = sbCost[1] + costs[i][j]
                if myCost < nbCost[1]:
                    snbCost = nbCost
                    nbCost = [j, myCost]
                elif myCost < snbCost[1]:
                    snbCost = [j, myCost]
            bestCost, sbCost = nbCost, snbCost
        return min(bestCost[1], sbCost[1])
```





## 121. 买卖股票的最佳时机

```
显然，如果我们真的在买卖股票，我们肯定会想：如果我是在历史最低点买的股票就好了！太好了，在题目中，我们只要用一个变量记录一个历史最低价格 minprice，我们就可以假设自己的股票是在那天买的。那么我们在第 i 天卖出股票能得到的利润就是 prices[i] - minprice。

因此，我们只需要遍历价格数组一遍，记录历史最低点，然后在每一天考虑这么一个问题：如果我是在历史最低点买进的，那么我今天卖出能赚多少钱？当考虑完所有天数之时，我们就得到了最好的答案。
```



```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        inf = int(1e9)
        minprice = inf
        maxprofit = 0
        for price in prices:
            maxprofit = max(price - minprice, maxprofit)#更新最大收益
            minprice = min(price, minprice)#更新最小价格
        return maxprofit
```



##  714. 买卖股票的最佳时机含手续费

##  309. 最佳买卖股票时机含冷冻期

##  152. 乘积最大子数组


##  最大连续1的个数 II



##  376. 摆动序列

## 1746. 经过一次操作后的最大子数

## 1230. 抛掷硬币


<font color='red'>字符串上的动态规划 / 最长递增子序列 / 最长公共子序列</font>


## 1143. 最长公共子序列
## 1035. 不相交的线
## 712. 两个字符串的最小ASCII删除和
## 300. 最长递增子序列
## 673. 最长递增子序列的个数
## 1048. 最长字符串链
## 646. 最长数对链
## 5. 最长回文子串
## 1055. 形成字符串的最短路径
## 516. 最长回文子序列

<font color='red'>最小（最大）路径目标</font>
## 64. 最小路径和
## 562. 矩阵中最长的连续1线段
## 1182. 与目标颜色间的最短距离

<font color='red'>经典动态规划</font>

## 343. 整数拆分
## 238. 除自身以外数组的乘积

<font color='red'>记忆化</font>
## 139. 单词拆分

## 254. 因子的组合
## 329. 矩阵中的最长递增路径

<font color='red'>计数动态规划</font>
## 62. 不同路径
## 63. 不同路径 II
## 576. 出界的路径数
## 650. 只有两个键的键盘
## 361. 轰炸敌人


<font color='red'>合并间隔</font>

## 96. 不同的二叉搜索树
## 1130. 叶值的最小代价生成树

<font color='red'>硬币兑换 / 组合和</font>

## 322. 零钱兑换

## 518. 零钱兑换 II

## 39. 组合总和

## 279. 完全平方数

<font color='red'>背包问题</font>

## 416. 分割等和子集
## 494. 目标和

# 链表

## 环形链表，返回入环点

> 先找到fast=slow的点，然后将slow=head,然后两个指针一步一步前进，重合点就是入环点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head==None:
            return 
        fast=head
        slow=head
        while fast:
            try:
                fast=fast.next.next
                slow=slow.next
            except:
                return 
            if fast==slow:
                slow2=head
                fast2=fast
                while fast2:
                    if fast2==slow2:
                        return slow2
                    fast2=fast2.next
                    slow2=slow2.next
            elif fast== None or slow==None:
                return 
```





# 查找

## [寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

```python
class Solution:
    def findDuplicate(self, nums):
        b=nums
        b.sort()
        pointer=len(b)//2

        while 3<len(b):
            if b[pointer+1]==b[pointer] or b[pointer]==b[pointer-1]:
                return b[pointer]
            else:
                if len(set(b[:pointer]))==pointer:
                    b=b[pointer:]
                    pointer=len(b)//2
                else:
                    b=b[:pointer]
                    pointer=len(b)//2

        if len(b)==3:
            if b[0]==b[1] or b[1]==b[2]:
                return b[1]
        elif len(b)==2:
            if b[0]==b[1]:
                return b[0]
```

```python
#二进制，但是结果却很慢，想想原因
class Solution:
    def findDuplicate(self, nums):
        ll=list(set(nums))

        b=['%032d'%int(bin(i)[2:]) for i in nums]
        c=['%032d'%int(bin(i)[2:]) for i in ll]

        def getcount(b):
            temp=[]
            for i in range(len(b[0])):
                templayer=[]
                for j in range(len(b)):
                    templayer.append(int(b[j][i]))
                temp.append(templayer)
            countre=[]
            for i in range(len(temp)):
                countre.append(sum(temp[i]))

            return countre

        orr=getcount(b)
        setr=getcount(c)

        finre=[]
        for i in range(len(orr)):
            if orr[i]>setr[i]:
                finre.append('1')
            else:
                finre.append('0')
        result=''.join(finre)
        return int(result,2)
```





# 排序

## 冒泡

```python
import random

L1=random.sample(range(1,20),8)
# L1=[18,14,10,3,13,16]

print(L1)
class BubbleSort():
    def BSort(self,L1):
        if len(L1)<1:
            return L1
        for i in range(1,len(L1)):
            for j in range(0,len(L1)-i):
                if L1[j]>L1[j+1]:
                    L1[j],L1[j+1]=L1[j+1],L1[j]
        return L1

x=BubbleSort()
print(x.BSort(L1))
```

注意这个的逻辑（有次写成了这个）

```python
#%%
import random

L1=random.sample(range(1,20),8)


class BubbleSort():
    def BSort(self,L1):
        if len(L1)<1:
            return L1
        for i in range(len(L1)):
            for j in range(0,len(L1)-1):
                if L1[j]>L1[i]:
                    L1[i],L1[j]=L1[j],L1[i]
        return L1

x=BubbleSort()
print(x.BSort(L1))

```







# 牛客网模拟笔试题目

## 输入一个数字n，输出精简数列使得这个数列中的数的加和就是1-n，如1 2 4 7

```python
k=14


l=list(range(3,k+1))
if k<=2:
    ans=list(range(1,k+1))
else:
    ans=[1,2]


def getnewlist(sl,k):
    temp=[]
    for i in range(k-2):
        for j in range(i+1,k-1):
            temp.append(sl[i]+sl[j])
        if sl[-1] in temp:
            sl=sl[:-1]
    return sl
            
for i in range(3,k+1):
    ans.append(l.pop(0))
    ans=getnewlist(ans, len(ans))
    
   
print(ans)
```





# hw-0728

## 服务器资源分配

```python
matrix=[[9,9,4],[6,6,8],[2,1,1]]
matrix.sort(key=lambda x:x[1])
```



## 陈大师吃药

## Leetcode 980 不同路径



# shopee 0730

## 台阶跳问题

> 假设你正在爬楼梯。需要 *n* 阶你才能到达楼顶。
>
> 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        a,b=1,1
        for _ in range(n):
            a,b=b,a+b
        return a
```

##  一个字符串中字串之积的最大值

>设字符串s存在两个子字符串s1,s2。
>使得s1的长度Xs2的长度 的最大值
>s1 , s2 没有相交元素
>举例：s=‘adcbadcbedbadedcbacbcadbc’
>s1=‘ded’ s2 = 'cbacbca’最大
>结果为 21



分析，求字符串的子字符串相乘，先求两个子字符串。
取得两个子字符串的过程用`滑动窗口`的思想

```
一
1,求字符串s 的子字符串 a 组成的所有对。如 [a,bcbc] 为一对。
2，求ab的所有对
3，求abc的所有对
...
二
1，求b 组成的所有对。如 [b,c].
...
依次求完。
三
在求子串的同时，将两个子串长度相乘，取最大值。
```

```python
class Solution:
    def getMaxSubstrLenProd(self, inputStr):
        ans=0
        i=0#string1 ini
        for i in range(len(inputStr)):
            for j in range(i+1,len(inputStr)):
                m=n=j
                string1=inputStr[i:j]
                string2=[]
                while n<len(inputStr):
                    if inputStr[m] in string1:
                        m+=1
                        n=m
                    elif inputStr[n] in string1:
                        ans=max(ans,len(string1)*len(string2))
                        string2=[]
                        m=n+1
                        n=m
                    else:
                        string2.append(inputStr[n])
                        n+=1
                ans=max(ans, len(string2)*len(string1))
        return ans
x=Solution()
hhh=x.getMaxSubstrLenProd('adcbadcbedbadedcbacbcadbc')
print(hhh)


```

##  Leecode329



