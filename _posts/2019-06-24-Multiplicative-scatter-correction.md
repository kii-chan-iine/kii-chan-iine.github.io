# 多元散射校正


Multiplicative scatter correction，主要为了消除光谱散射的影响。

该方法需一个“理想光谱”（即，光谱与样品成分呈直接的线性关系）来对其他光谱的基线平移和偏移进行校正。理想光谱难以获得，通常使用平均光谱替代。

过程：
1. 计算平均光谱![](http://latex.codecogs.com/gif.latex?\\R_m)
2. 每个样品光谱![](http://latex.codecogs.com/gif.latex?\\R_i)同![](http://latex.codecogs.com/gif.latex?\\R_m)进行一元线性回归(![](http://latex.codecogs.com/gif.latex?\\R_i=m_i R_m+b_i))，获得各光谱相对于标准光谱的线性平移量(回归常数)和倾斜偏移量(回归系数)
3. ![](http://latex.codecogs.com/gif.latex?\\R_{msc}=\frac{R_i-b_i}{m_i})



