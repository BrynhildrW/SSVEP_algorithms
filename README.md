# SSVEP_algorithms 
更新进展：今天写了公式变量说明，代码更新了CCA、eCCA和msCCA部分（2022/7/12）<p>
近期计划：编写eCCA、msCCA的说明内容。<p>
**建议各位同僚读完硕士赶紧去就业吧，千万不要盲目投身火海。**

***
## 公式变量符号及说明
| 符号名称 | 物理含义 |
| --- | --- |
| $\pmb{\chi}$ | 单个受试者的全体训练数据 |
| $\pmb{X}$ | EEG训练数据矩阵 |
| $\pmb{x}$ | EEG训练数据序列 |
| $\pmb{Y}$ | 人工构建正余弦模板 |
| $N_e$ | 刺激类别数 |
| $N_t$ | 训练样本数 |
| $N_c$ | 导联数 |
| $N_p$ | 单试次采样点数 |
| $N_h$ | 正余弦信号谐波个数 |
| $N_k$ | 保留子空间个数 |
| $\pmb{X}^i, \pmb{x}^i$ | 第 $i$ 试次或第 $i$ 导联数据，详见各段说明|
| $\pmb{X}_k, \pmb{x}_k$ | 第 $k$ 类别数据 |
| $\={\pmb{X}}_k$, $\={\pmb{x}}_k$ | 类别样本中心，由 $\pmb{X}_k$ 或 $\pmb{x}_k$ 按试次叠加平均得到 |
| $\={\={\pmb{X}}}, \={\={\pmb{x}}}$ | 总体样本中心，由 $\={\pmb{X}}_k$ 或 $\={\pmb{x}}_k$ 按类别叠加平均得到 |
| $f_s$ | EEG信号采样率 |
| $\pmb{\omega}, \pmb{U}, \pmb{V}$ etc | 低维空间滤波器 |
| $\pmb{W}$ | 高维空间滤波器，由低维空间滤波器集成而得 |

（在无特殊说明的情况下，所有训练数据默认经过了零均值化处理）
***
## 常见SSVEP信号处理算法（空间滤波器）
### 1. 典型相关性分析 | Canonical correlation analysis, CCA
#### 1.1 [标准CCA][CCA]
> 对于第 $k$ 目标、第 $i$ 试次数据 $\pmb{X}_k^i \in \mathbb{R}^{N_c \times N_p}$，其对应频率的人工构建正余弦模板 $\pmb{Y}_k \in \mathbb{R}^{(2N_h) \times N_p}$ 可表示为： <p>
$
 \pmb{Y}_k = 
  \begin{pmatrix}
   \sin(2 \pi fn)\\
   \cos(2 \pi fn)\\
   \sin(4 \pi fn)\\
   \cos(4 \pi fn)\\
   ...\\
   \sin(2 N_h \pi fn)\\
   \cos(2 N_h \pi fn)\\
  \end{pmatrix}
 , 
 n=[\dfrac{1}{f_s}, \dfrac{2}{f_s}, ..., \dfrac{N_p}{f_s}]
$ <p>
> CCA的优化目标为 $\hat{\pmb{U}}_k^i$ 和 $\hat{\pmb{V}}_k^i$，使得一维信号 $\hat{\pmb{U}}_k^i \pmb{X}_k^i$ 与 $\hat{\pmb{V}}_k^i \pmb{Y}_k$ 之间相关性最大化，其目标函数为： <p>
$
 \hat{\pmb{U}}_k^i, \hat{\pmb{V}}_k^i =
 \underset{\pmb{U}_k^i, \pmb{V}_k^i} \argmax 
  \dfrac{Cov(\pmb{U}_k^i \pmb{X}_k^i, \pmb{V}_k^i \pmb{Y}_k)}
        {\sqrt{Var(\pmb{U}_k^i \pmb{X}_k^i)}
         \sqrt{Var(\pmb{V}_k^i \pmb{Y}_k)}}
 = 
 \underset{\pmb{U}_k^i, \pmb{V}_k^i} \argmax
  \dfrac{\pmb{U}_k^i \pmb{C}_{\pmb{XY}} {\pmb{{V}}_k^i}^T}
       {\sqrt{\pmb{U}_k^i \pmb{C}_{\pmb{XX}} {\pmb{{U}}_k^i}^T}
        \sqrt{\pmb{V}_k^i \pmb{C}_{\pmb{YY}} {\pmb{{V}}_k^i}^T}}
$ <p>
$ 
 \begin{cases}
  \pmb{C}_{\pmb{XX}} = \pmb{X}_k^i {\pmb{X}_k^i}^T
   \in \mathbb{R}^{N_c \times N_c}\\
  \pmb{C}_{\pmb{YY}} = \pmb{Y}_k {\pmb{Y}_k}^T
   \in \mathbb{R}^{(2N_h) \times (2N_h)}\\
  \pmb{C}_{\pmb{XY}} = \pmb{X}_k^i {\pmb{Y}_k}^T
   \in \mathbb{R}^{N_c \times (2N_h)}\\
  \pmb{C}_{\pmb{YX}} = \pmb{Y}_k {\pmb{X}_k}^T
   \in \mathbb{R}^{(2N_h) \times N_c}\\
 \end{cases}
$ <p>
根据最优化理论，上述问题的等效形式为：
$
 \begin{cases}
  \underset{\pmb{U}_k^i, \pmb{V}_k^i} \max \ 
   \pmb{U}_k^i \pmb{C}_{\pmb{XY}} {\pmb{{V}}_k^i}^T\\
  s.t.\  \pmb{U}_k^i \pmb{C}_{\pmb{XX}} {\pmb{{U}}_k^i}^T = 
       \pmb{V}_k^i \pmb{C}_{\pmb{YY}} {\pmb{{V}}_k^i}^T = 1
 \end{cases}
$ <p>
利用Lagrandian乘子法构建多元函数 $J(\pmb{U}_k^i, \pmb{V}_k^i, \lambda, \theta)$：<p>
$
 J = \pmb{U}_k^i \pmb{C}_{\pmb{XY}} {\pmb{{V}}_k^i}^T - 
     \dfrac{1}{2} \lambda (\pmb{U}_k^i \pmb{C}_{\pmb{XX}} {\pmb{{U}}_k^i}^T - 1) - 
     \dfrac{1}{2} \theta (\pmb{V}_k^i \pmb{C}_{\pmb{YY}} {\pmb{{V}}_k^i}^T - 1)
$ <p>
对函数$J$求偏导数并置零、化简：<p>
$
 \begin{cases}
  \dfrac{\partial J}{\partial \pmb{{U}}_k^i} = 
   \pmb{C}_{\pmb{XY}} {\pmb{{V}}_k^i}^T - 
    \lambda \pmb{C}_{\pmb{XX}} {\pmb{{U}}_k^i}^T = 0\\
  \dfrac{\partial J}{\partial \pmb{{V}}_k^i} = 
   \pmb{C}_{\pmb{YX}} {\pmb{{U}}_k^i}^T - 
    \theta \pmb{C}_{\pmb{YY}} {\pmb{{V}}_k^i}^T = 0
 \end{cases}
$ <p>
$
 \begin{cases}
  {\pmb{C}_{\pmb{XX}}}^{-1} \pmb{C}_{\pmb{XY}} 
  {\pmb{C}_{\pmb{YY}}}^{-1} \pmb{C}_{\pmb{YX}} \pmb{U}_k^i
   = {\lambda}^2 \pmb{U}_k^i\\
  {\pmb{C}_{\pmb{YY}}}^{-1} \pmb{C}_{\pmb{YX}} 
  {\pmb{C}_{\pmb{XX}}}^{-1} \pmb{C}_{\pmb{XY}} \pmb{V}_k^i
   = {\theta}^2 \pmb{V}_k^i
 \end{cases}
$ <p>
对上式中的两个Hermitte矩阵分别进行特征值分解，取最大特征值对应的特征向量作为投影向量，即为所求。


#### 1.2 [扩展CCA][eCCA] | Extended CCA, eCCA
#### 1.3 [多重刺激CCA][msCCA] | Multi-stimulus CCA, msCCA
#### 1.4 [跨个体空间滤波器迁移][CSSFT] | Cross-subject spatial filter transfer method, CSSFT

[CCA]: http://ieeexplore.ieee.org/document/4203016/
[eCCA]: http://www.pnas.org/lookup/doi/10.1073/pnas.1508080112
[msCCA]: https://ieeexplore.ieee.org/document/9006809/
[CSSFT]: http://iopscience.iop.org/article/10.1088/1741-2552/ac6b57

***
### 2. 多变量同步化系数 | Multivariate synchronization index, MSI
#### 2.1 [标准MSI][MSI]
#### 2.2 [时域局部MSI][tMSI] | Temporally MSI, tMSI
#### 2.3 [扩展MSI][eMSI] | Extended MSI, eMSI

[MSI]: temp
[tMSI]: temp
[eMSI]: temp

***
### 3. 任务相关成分分析 | Task-related component analysis, TRCA
#### 3.1 [（集成）TRCA][TRCA] | (Ensemble) TRCA, (e)TRCA
#### 3.2 [正余弦扩展TRCA][TRCA-R] | (e)TRCA-R
#### 3.3 [多重刺激TRCA][ms-TRCA] | Multi-stimulus (e)TRCA, ms-(e)TRCA
#### 3.4 [相似度约束TRCA][sc-TRCA] | Similarity-constrained (e)TRCA, sc-(e)TRCA

[TRCA]: https://ieeexplore.ieee.org/document/7904641/
[TRCA-R]: https://ieeexplore.ieee.org/document/9006809/
[ms-TRCA]: temp
[sc-TRCA]: temp
***






