---
html:
    toc: true
print_background: true
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>

# SSVEP_algorithms 
Github的在线Markdown公式编译模块有bug，想看公式过程的建议下载**算法说明**或者**html**、**md**等文件本地查看。鉴于部分公式推导步骤属于本人硕士学位论文内容，在此提醒各位，ctrl+C/V 请慎重。
更新进展：今天写了TRCA部分的公式说明（2022/7/18）
近期计划：编写eCCA、msCCA的说明内容。

**建议各位同僚读完硕士赶紧去就业吧，千万不要盲目读博、投身火海。**

***
## 公式变量符号及说明
| 符号名称 | 物理含义 |
| --- | --- |
| $\pmb{\chi}$ | EEG测试数据矩阵 |
| $\pmb{X}$ | EEG训练数据矩阵 |
| $\pmb{x}$ | EEG训练数据序列 |
| $\pmb{Y}$ | 人工构建正余弦模板 |
| $N_e$ | 刺激类别数 |
| $N_t$ | 训练样本数 |
| $N_c$ | 导联数 |
| $N_p$ | 单试次采样点数 |
| $N_h$ | 正余弦信号谐波个数 |
| $N_k$ | 保留子空间个数 |
| $\pmb{X}^i, \pmb{x}^i$ | 第 $i$ 试次或第 $i$ 导联数据，详见各部分具体说明|
| $\pmb{X}_k, \pmb{x}_k$ | 第 $k$ 类别数据 |
| $\bar{\pmb{X}}_k$, $\bar{\pmb{x}}_k$ | 类别样本中心，由 $\pmb{X}_k$ 或 $\pmb{x}_k$ 按试次叠加平均获得 |
| $\bar{\bar{\pmb{X}}}, \bar{\bar{\pmb{x}}}$ | 总体样本中心，由 $\bar{\pmb{X}}_k$ 或 $\bar{\pmb{x}}_k$ 按类别叠加平均获得 |
| $f_s$ | EEG 信号采样率 |
| $\pmb{\omega}, \pmb{U}, \pmb{V}$ ... | 低维空间滤波器 |
| $\pmb{W}$ | 高维空间滤波器，由低维空间滤波器集成获得 |
（在无特殊说明的情况下，所有训练数据默认经过了零均值化处理）

***
## 常见SSVEP信号处理算法（空间滤波器）
### 1. 典型相关性分析
**Canonical correlation analysis, CCA**

#### 1.1 标准CCA：CCA
**[论文链接][CCA] | 代码：[cca][cca(code)].cca()**

对于第 $k$ 类别、第 $i$ 试次数据 $\pmb{X}_k^i \in \mathbb{R}^{N_c \times N_p}$，其对应频率的人工构建正余弦模板 $\pmb{Y}_k \in \mathbb{R}^{(2N_h) \times N_p}$ 可表示为：
$$
  \pmb{Y}_k = 
  \begin{pmatrix}
    \sin(2 \pi fn)\\
    \cos(2 \pi fn)\\
    \sin(4 \pi fn)\\
    \cos(4 \pi fn)\\
    ...\\
    \sin(2 N_h \pi fn)\\
    \cos(2 N_h \pi fn)\\
  \end{pmatrix}, 
  n=[\dfrac{1}{f_s}, \dfrac{2}{f_s}, ..., \dfrac{N_p}{f_s}]
  \tag{1-1}
$$
CCA的优化目标为 $\hat{\pmb{U}}_k^i$ 和 $\hat{\pmb{V}}_k^i$，使得一维信号 $\hat{\pmb{U}}_k^i \pmb{X}_k^i$ 与 $\hat{\pmb{V}}_k^i \pmb{Y}_k$ 之间相关性最大化，其目标函数为：
$$
  \hat{\pmb{U}}_k^i, \hat{\pmb{V}}_k^i =
  \underset{\pmb{U}_k^i, \pmb{V}_k^i} \argmax 
    \dfrac{Cov(\pmb{U}_k^i \pmb{X}_k^i, \pmb{V}_k^i \pmb{Y}_k)}
          {\sqrt{Var(\pmb{U}_k^i \pmb{X}_k^i)} \sqrt{Var(\pmb{V}_k^i \pmb{Y}_k)}} = 
    \underset{\pmb{U}_k^i, \pmb{V}_k^i} \argmax
      \dfrac{\pmb{U}_k^i \pmb{C}_{\pmb{XY}} {\pmb{{V}}_k^i}^T}
            {\sqrt{\pmb{U}_k^i \pmb{C}_{\pmb{XX}} {\pmb{{U}}_k^i}^T} \sqrt{\pmb{V}_k^i \pmb{C}_{\pmb{YY}} {\pmb{{V}}_k^i}^T}}
  \\
  \tag {1-2}
$$
$$
  \begin{cases}
    \pmb{C}_{\pmb{XX}} = \dfrac{1}{N_p-1} \pmb{X}_k^i {\pmb{X}_k^i}^T \in \mathbb{R}^{N_c \times N_c}\\
    \pmb{C}_{\pmb{YY}} = \dfrac{1}{N_p-1} \pmb{Y}_k {\pmb{Y}_k}^T \in \mathbb{R}^{(2N_h) \times (2N_h)}\\
    \pmb{C}_{\pmb{XY}} = \dfrac{1}{N_p-1} \pmb{X}_k^i {\pmb{Y}_k}^T \in \mathbb{R}^{N_c \times (2N_h)}\\
    \pmb{C}_{\pmb{YX}} = \dfrac{1}{N_p-1} \pmb{Y}_k {\pmb{X}_k}^T \in \mathbb{R}^{(2N_h) \times N_c}\\
  \end{cases}
  \tag {1-3}
$$
根据最优化理论，函数 (1-2) 的等效形式为：
$$
  \begin{cases}
    \underset{\pmb{U}_k^i, \pmb{V}_k^i} \max \ \pmb{U}_k^i \pmb{C}_{\pmb{XY}} {\pmb{{V}}_k^i}^T\\
    \\
    s.t.\ \pmb{U}_k^i \pmb{C}_{\pmb{XX}} {\pmb{{U}}_k^i}^T =
    \pmb{V}_k^i \pmb{C}_{\pmb{YY}} {\pmb{{V}}_k^i}^T = 1
  \end{cases}
  \tag {1-4}
$$
利用*Lagrandian*乘子法构建多元函数 $J(\pmb{U}_k^i, \pmb{V}_k^i, \lambda, \theta)$：
$$
  J = \pmb{U}_k^i \pmb{C}_{\pmb{XY}} {\pmb{{V}}_k^i}^T - \dfrac{1}{2} \lambda (\pmb{U}_k^i \pmb{C}_{\pmb{XX}} {\pmb{{U}}_k^i}^T - 1) - \dfrac{1}{2} \theta (\pmb{V}_k^i \pmb{C}_{\pmb{YY}} {\pmb{{V}}_k^i}^T - 1)
  \tag {1-5}
$$
对函数 $J$ 求偏导数并置零、化简：
$$
  \begin{cases}
    \dfrac{\partial J}{\partial \pmb{{U}}_k^i} = 
    \pmb{C}_{\pmb{XY}} {\pmb{{V}}_k^i}^T - \lambda \pmb{C}_{\pmb{XX}} {\pmb{{U}}_k^i}^T = 0\\
    \\
    \dfrac{\partial J}{\partial \pmb{{V}}_k^i} = 
    \pmb{C}_{\pmb{YX}} {\pmb{{U}}_k^i}^T - \theta \pmb{C}_{\pmb{YY}} {\pmb{{V}}_k^i}^T = 0
  \end{cases}
  \tag {1-6}
$$
$$
  \begin{cases}
    {\pmb{C}_{\pmb{XX}}}^{-1} \pmb{C}_{\pmb{XY}} {\pmb{C}_{\pmb{YY}}}^{-1} \pmb{C}_{\pmb{YX}} \pmb{U}_k^i = {\lambda}^2 \pmb{U}_k^i\\
    \\
    {\pmb{C}_{\pmb{YY}}}^{-1} \pmb{C}_{\pmb{YX}} {\pmb{C}_{\pmb{XX}}}^{-1} \pmb{C}_{\pmb{XY}} \pmb{V}_k^i = {\theta}^2 \pmb{V}_k^i
 \end{cases}
 \tag {1-7}
$$
对式 (1-7) 中的两个*Hermitte*矩阵分别进行特征值分解，取最大特征值对应的特征向量作为投影向量，即为所求。

#### 1.2 扩展CCA：eCCA
**(Extended CCA)**

**[论文链接][eCCA] | 代码：[cca][cca(code)].ecca()**


#### 1.3 多重刺激CCA：msCCA
**(Multi-stimulus CCA)**

**[论文链接][msCCA] | 代码：[cca][cca(code)].mscca()**


#### 1.x 跨个体空间滤波器迁移：CSSFT
**(Cross-subject spatial filter transfer method)**

**[论文链接][CSSFT] | 代码：[cca][cca(code)].cssft()**


[cca(code)]: https://github.com/BrynhildrW/SSVEP_algorithms/blob/main/cca.py
[CCA]: http://ieeexplore.ieee.org/document/4203016/
[eCCA]: http://www.pnas.org/lookup/doi/10.1073/pnas.1508080112
[msCCA]: https://ieeexplore.ieee.org/document/9006809/
[CSSFT]: http://iopscience.iop.org/article/10.1088/1741-2552/ac6b57

***
### 2. 多变量同步化系数
**Multivariate synchronization index, MSI**

#### 2.1 标准MSI：MSI
**[论文链接][MSI] | 代码：[msi][msi(code)].msi()**


#### 2.2 时域局部MSI：tMSI
**(Temporally MSI)**

**[论文链接][tMSI] | 代码：[msi][msi(code)].tmsi()**


#### 2.3 扩展MSI：eMSI
**(Extended MSI)**

**[论文链接][MSI] | 代码：[msi][msi(code)].emsi()**


[msi(code)]: temp
[MSI]: https://linkinghub.elsevier.com/retrieve/pii/S0165027013002677
[tMSI]: http://link.springer.com/10.1007/s11571-016-9398-9
[eMSI]: https://linkinghub.elsevier.com/retrieve/pii/S0925231217309980

***
### 3. 任务相关成分分析
**Task-related component analysis, TRCA**

#### 3.1 普通/集成TRCA：(e)TRCA
**( (Ensemble) TRCA, (e)TRCA）**

**[论文链接][TRCA] | 代码：[trca][trca(code)].etrca()**

与此前基于CCA改进的SSVEP算法相比，TRCA在构建思路上存在较大差别，具体表现在其关注对象（即信号模板）不再限定为具有正余弦波动性质的传统模型，而是充分包含了个体信息的 “任务相关成分” (Task-related components, TRCs)。关于TRC可以简单理解为：当受试者在多次接受相同任务时，其EEG信号中应当包含具有相同性质的诱发成分。由此可见，TRCA在理论上适用于任何诱发信号具有稳定波形特征的BCI范式特征信号解码。

*Nakanishi* 等人首次将TRCA应用至SSVEP信号解码上时，在公式推导部分使用了一个非常讨巧的办法：**跨试次信号相关性最大化**。之所以称其“讨巧”，是因为原版TRCA公式分子中强调的**跨试次协方差计算**操作，在实际编程过程中产生了大量冗余计算步骤；其分母的**矩阵拼接**操作也缺乏明确的物理意义对应说明。而上述“瑕疵”在后续算法改进工作中被不断研究透彻。因此本文不再按照原文思路推导算法，仅给出相对成熟的阐释：

对于第 $k$ 类别、第 $i$、$j$ 试次数据 $\pmb{X}_k^i,\pmb{X}_k^j \in \mathbb{R}^{N_c \times N_p}$ (假定 $i \ne j$)，其跨试次样本协方差以及单试次样本方差（自协方差）分别为：
$$
  \begin{cases}
    Cov(\pmb{\omega}_k \pmb{X}_k^i, \pmb{\omega}_k \pmb{X}_k^j) = \dfrac{1} {N_p-1} \pmb{\omega}_k \pmb{X}_k^i {\pmb{X}_k^j}^T {\pmb{\omega}_k}^T, i \ne j\\
    \\
    Var(\pmb{\omega}_k \pmb{X}_k^i) = Cov(\pmb{\omega}_k \pmb{X}_k^i, \pmb{\omega}_k \pmb{X}_k^j) = \dfrac{1}{N_p-1} \pmb{\omega}_k \pmb{X}_k^i {\pmb{X}_k^i}^T {\pmb{\omega}_k}^T\\
  \end{cases}
  \tag {3-1}
$$
因此，TRCA的目标函数可写为：
$$
  \hat{\pmb{\omega}}_k = 
  \underset{\pmb{\omega}_k} \argmax 
    \dfrac{\sum_{j=1, j \ne i}^{N_t} \sum_{i=1}^{N_t} Cov(\pmb{\omega}_k \pmb{X}_k^i, \pmb{\omega}_k \pmb{X}_k^j)} {\sum_{i=1}^{N_t} Var(\pmb{\omega}_k \pmb{X}_k^i)} = 
  \underset{\pmb{\omega}_k} \argmax 
    \dfrac{\pmb{\omega}_k \pmb{S}_k {\pmb{\omega}_k}^T} {\pmb{\omega}_k \pmb{Q}_k {\pmb{\omega}_k}^T}
  \tag {3-2}
$$
$$
  \begin{cases}
    \pmb{S}_k = \sum_{j=1,j \ne i}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^j}^T, i \ne j\\
    \\
    \pmb{Q}_k = \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T
  \end{cases}
  \tag {3-3}
$$
根据广义瑞丽商 (*Generalized Rayleigh quotient*) 的结论，上述目标函数的单维度最优解即为方阵 ${\pmb{Q}_k}^{-1} \pmb{S}_k$ 的**最大特征值对应的特征向量**。接下来对TRCA的目标函数作进一步分析：
$$
  \pmb{S}_k = \sum_{j=1}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^j}^T - \pmb{Q}_k = 
  {N_t}^2 \bar{X}_k {\bar{X}_k}^T - \pmb{Q}_k = 
  \pmb{S}_k^{'} - \pmb{Q}_k
  \tag {3-4}
$$
$$
  \dfrac{\pmb{\omega}_k \pmb{S}_k {\pmb{\omega}_k}^T} {\pmb{\omega}_k \pmb{Q}_k {\pmb{\omega}_k}^T} = 
  \dfrac{\pmb{\omega}_k \pmb{S}_k^{'} {\pmb{\omega}_k}^T} {\pmb{\omega}_k \pmb{Q}_k {\pmb{\omega}_k}^T} - 1
  \tag {3-5}
$$

相比于直接计算 $\pmb{S}_k$，经由 $\pmb{S}_k^{'}$ 替换或计算得到 $\pmb{S}_k$ 能够大幅提升运算速度。其原因如下：将单次浮点数相乘与相加设为两种单位操作，其耗时分别为 $T_{\times}$ 和 $T_+$，对应时间复杂度分别为 $O_{\times}$ 与 $O_+$。则针对 $\pmb{X}_k^i$ 执行一次矩阵乘法 $\pmb{X}_k^i {\pmb{X}_k^i}^T$ 或矩阵加法 $\pmb{X}_k^i + \pmb{X}_k^j$ 所需的理论运行时间 $T_{M \times}$、$T_{M+}$ 分别为：
$$
  \begin{cases}
    T_{M \times} = ({N_c}^2 N_p)T_+ + [{N_c}^2 (N_p-1)]T_{\times}\\
    \\
    T_{M+} = (N_c N_p)T_+
  \end{cases}
  \tag {3-6}
$$
对于具有 $\mathbb{R}^{N_t \times N_c \times N_p}$ 维度的训练数据张量 $\pmb{X}_k$，求解 $\pmb{S}_k$ 的总计理论时间 $T_1$ 与时间复杂度 $O_1$ 分别为：
$$
  \begin{cases}
    T_1 = N_t(N_t-1)T_{M \times} + [N_t(N_t-1)-1]T_{M+}\\
    \\
    O_1 = O_{\times}({N_t}^2 {N_c}^2 N_p) + O_+({N_t}^2 {N_c}^2 N_p)
  \end{cases}
  \tag {3-7}
$$
而使用 $\pmb{S}_k^{'}$ 时，首先计算按试次平均后的个体模板 $\bar{\pmb{X}}_k$，其理论运行时间 $T_0$ 为：
$$
  T_0 = (N_c N_p)T_{\times} + (N_t-1)T_{M+}
  \tag {3-9}
$$
$\pmb{S}_k^{'}$ 的总计理论计算时间 $T_2$ 与时间复杂度 $O_2$ 分别为：
$$
  \begin{cases}
    T_2 = T_0 + T_{M \times}\\
    \\
    O_2 = O_{\times}({N_c}^2 N_p) + O_+(\max \{N_t N_c N_p, {N_c}^2 N_p \})
  \end{cases}
  \tag {3-10}
$$
对比 $O_1$ 与 $O_2$ 可见，样本数量越多，采用该种替换方法与原始情况所产生的偏差越小、速度提升越大。

综上所述，通过训练数据获取当前类别专属的空间滤波器 $\hat{\pmb{\omega}}_k$ 以及信号模板 $\hat{\pmb{\omega}}_k \bar{\pmb{X}}_k$，基于一维*Pearson*相关系数公式，对单试次测试数据 $\pmb{\chi}$ 应用空间滤波后与模板信号计算判别系数：
$$
  \rho_k = corr(\hat{\pmb{\omega}}_k \bar{\pmb{X}}_k, \hat{\pmb{\omega}}_k \pmb{\chi})
  \tag {3-11}
$$
eTRCA是基于TRCA的集成学习版本，它把各类别 $\hat{\pmb{\omega}}_k \in \mathbb{R}^{1 \times N_c}$ 按行拼接在一起，在空间维度上扩增了信号模板：
$$
  \begin{cases}
    \hat{\pmb{W}} = {[{\pmb{\omega}_1}^T, {\pmb{\omega}_2}^T,..., {\pmb{\omega}_{N_e}}^T]}^T \in \mathbb{R}^{N_e \times N_c}\\
    \\
    \rho_k = corr2(\hat{\pmb{W}} \bar{\pmb{X}}_k, \hat{\pmb{W}} \pmb{\chi})
  \end{cases}
  \tag {3-12}
$$

笔者认为，eTRCA虽然性能更为强劲，但该算法可能存在原理性缺陷：容易产生冗余成分。在刺激目标较多时，全类别集成并无必要。具体研究工作正在进行中。

至此我们有必要再回顾一下TRCA的目标函数：

（1）分子中 $\pmb{\omega}_k \bar{X}_k {\bar{X}_k}^T {\pmb{\omega}_k}^T$ 的本质为“**滤波后特征信号的能量**”。训练样本数目越多，叠加平均操作获取的信号模板质量越高，即随机信号成分削减越充分。而且分子能够决定目标函数的最终优化上限。

（2）分母 $\pmb{\omega}_k (\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T) {\pmb{\omega}_k}^T$ 的本质为“**滤波后各试次信号能量之和**”。

（3）结合上述两点可见，TRCA的性能优越是原理性的，其结构相当完善。唯一的缺陷在于训练样本数目：当 $N_t$ 较小时，由（1）可知优化目标将产生无法弥补的偏差。因此后续关于TRCA的改进，大多针对少样本下获取更稳健的信号模板估计入手，我们将在(e)TRCA-R、sc-(e)TRCA等算法中观察到这一倾向。

#### 3.2 正余弦扩展TRCA：(e)TRCA-R
**[论文链接][TRCA-R] | 代码：[trca][trca(code)].etrca_r()**


#### 3.3 多重刺激TRCA：ms-(e)TRCA
**(Multi-stimulus (e)TRCA)**

**[论文链接][ms-TRCA] | 代码：[trca][trca(code)].ms_etrca()**


#### 3.4 相似度约束TRCA：sc-(e)TRCA
**(Similarity-constrained (e)TRCA)**

**[论文链接][sc-TRCA] | 代码：[trca][trca(code)].sc_etrca()**


#### 3.5 组TRCA：gTRCA
**(Group TRCA)**

**[论文链接][gTRCA] | 代码：[trca][trca(code)].gtrca()**


#### 3.6 交叉相关性TRCA：xtrca
**(Cross-correlation TRCA)**

**[论文链接][xTRCA] | 代码：[trca][trca(code)].xtrca()**



[trca(code)]: temp
[TRCA]: https://ieeexplore.ieee.org/document/7904641/
[TRCA-R]: https://ieeexplore.ieee.org/document/9006809/
[ms-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
[sc-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/abfdfa
[gTRCA]: temp
[xTRCA]: temp


***
### x. 其它早期算法
#### x.1 [最小能量组合][MEC]：[other][other(code)].mec()
**Minimun energy combination, MEC**


#### x.2 [最大对比度组合][MCC]：[other][other(code)].mcc()
**Maximun contrast combination, MCC**


[other(code)]: temp
[MEC]: http://ieeexplore.ieee.org/document/4132932/
[MCC]: http://ieeexplore.ieee.org/document/4132932/

***






