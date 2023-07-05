# 3. 任务相关成分分析
**Task-related component analysis, TRCA**



## 3.2 多重刺激 TRCA：ms-(e)TRCA
**(Multi-stimulus (e)TRCA)**

**[论文链接][ms-TRCA] | 代码：[trca][trca(code)].ms_etrca()**

该算法是我们提到的第一种关于TRCA的改进算法，与 msCCA、ms-eCCA 一样出自 Wong 的之手，行文风格也一以贯之。由 3.1 节末尾分析可知，TRCA 类型的算法需要较多的训练数据以保证协方差矩阵估计的准确性，在原文中亦有相关描述：
> To guarantee good performance, the number of calibration trials for each visual stimulus cannot be small for the eCCA method and the eTRCA method; otherwise, their recognition accuracies would decrease dramatically $\dots$ A major reason is that the estimate of covariance matrices in the CCA-based (or the TRCA-based) methods could become unreliable in the case of small training data so that the resulting spatial filters may not be accurate. As a matter of fact, the algorithms relying on the estimator of covariance matrix also suffer from insufficient training data problem $\dots$

然而对于具有较多指令集（32、40 或更多）的 SSVEP-BCI 系统而言，为每个指令获取足量训练试次（$N_t>10$）在时间上通常是不允许的。而且刺激时间如果过长，诱发信号的质量会由于受试者累积的视觉疲劳而急剧下降，不利于系统实际应用。因此，Wong 提出了一种学习方法，从目标 $A$ 的相邻刺激（$B$、$C$ 等）的诱发信号中学习 $A$ 的特征。以一个简单的 4 指令集系统为例，ms- 技术的学习流程见下图（原文 fig.1）：

![ms-eTRCA示意图](figures/ms-eTRCA_fg1.png)

原文对于这一套流程的可行性做了连篇累牍的证明，在此仅提炼其最终结论：

**去除 0.14 s 的人眼视觉通路延迟之后，SSVEP 采集信号与闪烁刺激相位基本一致、不同刺激间的相位差也与闪烁刺激相位差基本一致**。

听君一席话，如听一席话是吧？这不是 SSVEP 信号处理的常规步骤吗？此外关于 Benchmark 数据集的刺激参数设定（频率、相位等），他们也特地写了三个**方程**来说明，总之一通操作着实看得人头昏脑胀。鉴于本人有幸曾经听过 Wong 的线上报告，视频里他只消三言两语便把 ms-eTRCA 的原理描述得鞭辟入里，所以我相信这种反人类的数学语言其实出自 Feng Wan 老师的手笔。

接下来我们先看看原文中给出的公式。ms-eTRCA 与 ms-TRCA 的关系类似 eTRCA 之于 TRCA，因此前两者的目标函数是共通的：
$$
    \hat{\pmb{w}}_k = \underset{\pmb{w}_k} \argmax 
    \dfrac{\pmb{w}_k \pmb{A}_k {\pmb{A}_k}^T {\pmb{w}_k}^T} {\pmb{w}_k \pmb{B}_k {\pmb{B}_k}^T {\pmb{w}_k}^T}
    \tag{3-2-1}
$$
分子分母中的协方差矩阵与 (e)TRCA 略有差异，具体如下：
$$
    \begin{cases}
        \pmb{A}_k = 
        \begin{bmatrix}
            \bar{\pmb{X}}_{k-m} & \bar{\pmb{X}}_{k-m+1} & \cdots & \bar{\pmb{X}}_{k+n}
        \end{bmatrix} \in \mathbb{R}^{N_c \times [(m+n+1)N_p]}\\
        \\
        \pmb{B}_k = 
        \begin{bmatrix}
            \pmb{\chi}_{k-m} & \pmb{\chi}_{k-m+1} & \cdots & \pmb{\chi}_{k+n}
        \end{bmatrix} \in \mathbb{R}^{N_c \times [(m+n+1) N_t N_p]}\\
    \end{cases}
    \tag{3-2-2}
$$
$$
    \pmb{\chi}_{k} = 
    \begin{bmatrix}
            \pmb{X}_k^1 & \pmb{X}_k^2 & \cdots & \pmb{X}_k^{N_t}
    \end{bmatrix} \in \mathbb{R}^{N_c \times (N_t N_p)}
    \tag{3-2-3}
$$
可以看出，multi-stimulus 技术的本质就是把目标刺激前 $m$、后 $n$ 个不同频率的刺激信号**顺次拼接**起来，在时间维度上对训练数据进行扩增，其范围为 $d$（含自身），具体拼接个数（$m$、$n$）依不同情况各有一番规定。

（1）关于 $m$ 与 $n$ 的大小分配（一般情况）：若 $d$ 为奇数，则前向与后向扩增范围相等；若 $d$ 为偶数，则前向扩增比后向多一位，即有：
$$
    \begin{cases}
        m = n = \dfrac{1} {2} (d-1), \ \ d=2n+1 | n \in \mathbb{N^+}\\
        \\
        m = \dfrac{1} {2} d, \ n = \dfrac{1} {2} d - 1, \ \ d=2n | n \in \mathbb{N^+}\\
    \end{cases}
    \tag{3-2-4}
$$

（2）假设刺激目标 $k$ 处于相对**靠前**的位置，由于前向没有足够类别的信号用于拼接，因此需向后顺延扩增位数。例如 $d=5,k=2$，应向后顺延一位（$m=1,n=3$）；若 $d=6,k=2$，则向后顺延两位（$m=1,n=4$）。综上可总结出式 (3-2-7)：
$$
    \begin{cases}
        m = k - 1\\
        \\
        n = d - k\\
    \end{cases}, \ k \in \left[ 1, \ \dfrac{1} {2} d \right]
    \tag{3-2-5}
$$
（3）假设刺激目标 $k$ 处于**中部**位置，即 (1) 中所述的“一般情况”，则有式 (3-2-8)：
$$
    \begin{cases}
        m = \left[\dfrac{1}{2}d\right]\\
        \\
        n = d - \left[\dfrac{1}{2}d\right] - 1
    \end{cases}, \ k \in \left( \left[ \dfrac{1}{2}d \right], \ N_e-\left(d-\left[\dfrac{1}{2}d\right]\right)\right)
    \tag{3-2-6}
$$
（4）假设刺激目标 $k$ 位于**尾部**位置，此时与（2）相反，需向前顺延扩增位数，即有式 (3-2-9)：
$$
    \begin{cases}
        m = d - 1 - \left(N_e - k \right)\\
        \\
        n = N_e - k\\
    \end{cases},  \ k \in \left[N_e - \left(d - \left[\dfrac{1}{2} d \right] - 1 \right), N_e \right]
    \tag{3-2-7}
$$
好了，我们再回过头去看式 (3-2-3)，该式分子、分母中的协方差矩阵可以通过与式 (3-1-5) 联动，进一步改写为如下形式：
$$
    \begin{cases}
        \pmb{A}_k {\pmb{A}_k}^T = \sum_{i=-m}^{n+1} \bar{\pmb{X}}_{k+i} {\bar{\pmb{X}}_{k+i}}^T = \dfrac{1}{{N_t}^2} \sum_{i=-m}^{n+1} {\pmb{S}_{k+i}^{'}}\\
        \\
        \pmb{B}_k {\pmb{B}_k}^T = \sum_{i=-m}^{n+1} \sum_{j=1}^{N_t} \pmb{X}_i^j {\pmb{X}_i^j}^T = \sum_{i=-m}^{n+1} \pmb{Q}_{k+i}
    \end{cases}
    \tag{3-2-8}
$$
一般来说，非零常系数是不影响矩阵特征值分解结果的。所以我们看 ms-(e)TRCA 的目标函数式，它就是把不同频率信号对应的 (e)TRCA 目标函数的分子、分母各自相加组成新的分式。再直白一点，就是“**把多个频率的信号当一个频率去训练**”，强行增加了可用样本数目。
$$
    \hat{\pmb{w}}_k = 
    \underset{\pmb{w}_k} \argmax 
    \dfrac{\pmb{w}_k \left(\sum_{i=-m}^{n+1} \bar{\pmb{X}}_{k+i} {\bar{\pmb{X}}_{k+i}}^T \right) {\pmb{w}_k}^T} {\pmb{w}_k \left(\sum_{i=-m}^{n+1} \sum_{j=1}^{N_t} \pmb{X}_i^j {\pmb{X}_i^j}^T \right) {\pmb{w}_k}^T} = 
    \underset{\pmb{w}_k} \argmax 
    \dfrac{\sum_{i=-m}^{n+1} \pmb{w}_k {\pmb{S}_{k+i}^{'}} {\pmb{w}_k}^T} {\sum_{i=-m}^{n+1} \pmb{w}_k \pmb{Q}_{k+i} {\pmb{w}_k}^T}
    \tag{3-2-9}
$$

我们有一点需要注意，根据文章里网格筛选的结果（原文 Fig.3）， $d$ 的范围并非是越大越好，在 (e)TRCA 算法上体现得尤为明显。

![ms-etrca筛选结果](figures/ms-eTRCA-net.png)

说来也令人感慨，ms- 的思路不可谓不简单，但是 *Wong* 等人之所以成功，一方面是因为敢想敢做，另一方面也要归功于砌墙的数学功底，能够把简单的内核包装成高大上的复杂操作，让人一眼完全看不透其内在关联。

## 3.3 正余弦扩展 TRCA：(e)TRCA-R
**[论文链接][TRCA-R] | 代码：[trca][trca(code)].etrca_r()**

该算法依旧出自 *Wong* 的手笔。按照其提出的设计框架，-R 技术就是将原本为单位阵的空间投影矩阵替换为正余弦信号张成的投影空间 $\pmb{\mathcal{P}}$，与之类似的算法还有 MsetCCA1-R（未来更新）。在讲解 (e)TRCA-R 之前，我们先来观察 (e)TRCA 的目标函数在统一框架（ 1.3 节式 (1-3-4)，**Type I** ) 下的各部分组成：
$$
    \begin{cases}
        \pmb{\mathcal{Z}} = \pmb{\mathcal{I}}_{N_t,N_c} \left(\oplus_{i=1}^{N_t} \pmb{X}_k^i \right) \in \mathbb{R}^{N_c \times \left(N_t N_p \right)}\\
        \\
        \pmb{\mathcal{D}} = \pmb{I}_{N_t N_p} \in \mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}\\
        \\
        \pmb{\mathcal{P}} = {\pmb{\mathcal{I}}_{N_t,N_p}}^T \pmb{\mathcal{I}}_{N_t,N_p} \in \mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}
    \end{cases}
    \tag{3-3-1}
$$
为了更清楚地让大家明白这个框架到底干了什么事，我们来依次画一下各步骤的展开形态：
$$
    \pmb{\mathcal{Z}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_c} & \pmb{I}_{N_c} & \cdots & \pmb{I}_{N_c}
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_c \right)}}
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \pmb{0} & \cdots & \pmb{0}\\
            \pmb{0} & \pmb{X}_k^2 & \cdots & \pmb{0}\\
            \vdots & \vdots & \ddots & \vdots\\
            \pmb{0} & \pmb{0} & \cdots & \pmb{X}_k^{N_t}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_c \right) \times \left(N_t N_p \right)}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \pmb{X}_k^2 & \cdots & \pmb{X}_k^{N_t}
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \tag{3-3-2}
$$
$$
    \pmb{\mathcal{Z}} \pmb{\mathcal{D}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \pmb{X}_k^2 & \cdots & \pmb{X}_k^{N_t}
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p} & \pmb{0} & \cdots & \pmb{0}\\
            \pmb{0} & \pmb{I}_{N_p} & \cdots & \pmb{0}\\
            \vdots & \vdots & \ddots & \vdots\\
            \pmb{0} & \pmb{0} & \cdots & \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}} = \pmb{\mathcal{Z}}
    \tag{3-3-3}
$$
$$
    \pmb{\mathcal{P}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p}\\
            \pmb{I}_{N_p}\\
            \vdots\\
            \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times N_p}}
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p} & \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{N_p \times \left(N_t N_p \right)}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p} & \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
            \pmb{I}_{N_p} & \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
            \vdots & \vdots & \ddots & \vdots\\
            \pmb{I}_{N_p} & \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}}
    \tag{3-3-4}
$$
$$
    \pmb{\mathcal{Z}} \pmb{\mathcal{D}} \pmb{\mathcal{P}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \cdots & \pmb{X}_k^{N_t}\\
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
            \vdots & \ddots & \vdots\\
            \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}} = 
    \underbrace{
        \begin{bmatrix}
            \sum_{i=1}^{N_t}{\pmb{X}_k^i} & \cdots & \sum_{i=1}^{N_t}{\pmb{X}_k^i}\\
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \tag{3-3-5}
$$
$$
    \pmb{\mathcal{Z}} \pmb{\mathcal{D}} \pmb{\mathcal{P}} {\pmb{\mathcal{P}}}^T {\pmb{\mathcal{D}}}^T {\pmb{\mathcal{Z}}}^T = 
    \underbrace{
        \begin{bmatrix}
            \sum_{i=1}^{N_t}{\pmb{X}_k^i} & \cdots & \sum_{i=1}^{N_t}{\pmb{X}_k^i}\\
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \underbrace{
        \begin{bmatrix}
            \sum_{i=1}^{N_t}{\pmb{X}_k^i}\\
            \vdots\\
            \sum_{i=1}^{N_t}{\pmb{X}_k^i}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times N_c}} = 
    N_t\sum_{j=1}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T
    \tag{3-3-6}
$$
$$
    \pmb{\mathcal{Z}} \pmb{\mathcal{D}} {\pmb{\mathcal{D}}}^T {\pmb{\mathcal{Z}}}^T = \pmb{\mathcal{Z}} \pmb{\mathcal{Z}}^T = 
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \pmb{X}_k^2 & \cdots & \pmb{X}_k^{N_t}\\
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1\\
            \pmb{X}_k^2\\
            \vdots\\
            \pmb{X}_k^{N_t}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times N_c}} = 
    \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T
    \tag{3-3-7}
$$
综上所述，仅需一维投影向量的情况下，*GEP* 方程可表示为式 (3-3-8)，忽略常系数影响后可发现该式与 (e)TRCA 的目标函数完全吻合。
$$
    \left(N_t\sum_{j=1}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) \pmb{w} = 
    \lambda \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) \pmb{w}
    \tag{3-3-8}
$$
在常规 (e)TRCA 中，正交投影矩阵 $\pmb{\mathcal{P}}$ 的本质作用仅是叠加。而在 (e)TRCA-R 中，*Wong* 将其改为了正余弦信号张成的投影空间，其余均与 (e)TRCA 保持一致：
$$
    \pmb{\mathcal{P}} = 
    \begin{bmatrix}
        \pmb{Q}_{\pmb{Y}_k}\\
        \vdots\\
        \pmb{Q}_{\pmb{Y}_k}\\
    \end{bmatrix}
    \begin{bmatrix}
        \pmb{Q}_{\pmb{Y}_k} & \cdots & \pmb{Q}_{\pmb{Y}_k}\\
    \end{bmatrix} = 
    \begin{bmatrix}
        \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T & \cdots & \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T\\
        \vdots & \ddots & \vdots\\
        \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T & \cdots & \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T\\
    \end{bmatrix}
    \tag{3-3-9}
$$
注意有 $\pmb{\mathcal{P}} = \pmb{\mathcal{P}} {\pmb{\mathcal{P}}}^T$，所以 (e)TRCA-R 的 *GEP* 方程可表示为：
$$
    \left[N_t\sum_{j=1}^{N_t} \sum_{i=1}^{N_t} \left(\pmb{X}_k^i \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T\right) \left(\pmb{X}_k^j \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T\right)^T \right] \pmb{w}^T = 
    \lambda \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) \pmb{w}^T
    \tag{3-3-10}
$$
目标函数可以写为：
$$
    \hat{\pmb{w}}_k = 
    \underset{\pmb{w}_k} \argmax 
        \dfrac{\pmb{w}_k \left(\bar{\pmb{X}}_k \pmb{P}_k \right) \left(\bar{\pmb{X}}_k \pmb{P}_k \right)^T {\pmb{w}_k}^T} {\pmb{w}_k \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T\right) {\pmb{w}_k}^T} = 
        \dfrac{\pmb{w}_k \bar{\pmb{X}}_k \pmb{Y}_k^T \pmb{Y}_k \bar{\pmb{X}}_k^T {\pmb{w}_k}^T} {\pmb{w}_k \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T\right) {\pmb{w}_k}^T}
    \tag{3-3-11}
$$
通过（3-3-11）可以观察到，-R 技术的改进点在于通过正交投影矩阵 $\pmb{P}_k$ 进一步约束了优化函数的目标 $\bar{\pmb{X}}_k$ 。关于投影矩阵的作用，在 ms-eCCA 章节中已有介绍，此处不再赘述。

## 3.4 相似度约束 TRCA：sc-(e)TRCA
**(Similarity-constrained (e)TRCA)**

**[论文链接][sc-TRCA] | 代码：[trca][trca(code)].sc_etrca()**

2021 年发表在 JNE 上的这篇文章，似乎是横空出世一般。没有 *Yijun Wang*，没有 *Xiaogang Chen*，没有 *Nakanish Masaki*，更没有 *Chiman Wong*，作者都是清一色的新面孔，让人眼前一亮。在公式推导方面，该团队似乎受到 *Hirokazu Tanaka* 的风格影响更多一些，在描述相关性时特意标注清楚了常系数，这一点对于明确变量的物理含义大有裨益。

按照原文的思路，sc-TRCA 的主要改进点在于通过人工构造正余弦矩阵 $\pmb{Y}_k$ 对目标函数分子、分母中的协方差矩阵进行扩增。我们先来看看原文对于扩增方式的描述：
$$
    \begin{align}
        \notag
        \widetilde{\pmb{Q}}_k &= 
        \begin{bmatrix}
            \pmb{Q}_1 & \pmb{0}\\ \ \\
            \pmb{0} & \pmb{Q}_2\\
        \end{bmatrix}, \ 
        \begin{cases}
            \pmb{Q}_1 = \dfrac{1}{N_t N_p} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \in \mathbb{R}^{N_c \times N_c}\\
            \ \\
            \pmb{Q}_2 = \dfrac{1}{N_p} \pmb{Y}_k {\pmb{Y}_k}^T \in \mathbb{R}^{\left(2N_h \right) \times \left(2N_h \right)}
        \end{cases}\\
        \notag \ \\
        \notag
        \widetilde{\pmb{S}}_k &= 
        \begin{bmatrix}
            \pmb{S}_{11} & \pmb{S}_{12}\\ \ \\
            \pmb{S}_{21} & \pmb{S}_{22}\\
        \end{bmatrix}, \ 
        \begin{cases}
            \pmb{S}_{11} = \dfrac{1}{N_t (N_t-1) N_p} \sum_{j=1,j \ne i}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \in \mathbb{R}^{N_c \times N_c}\\
            \ \\
            \pmb{S}_{12} = {\pmb{S}_{21}}^T = \dfrac{1}{N_t N_p} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{Y}_k}^T \in \mathbb{R}^{N_c \times \left(2N_h \right)}\\
            \ \\
            \pmb{S}_{22} = \dfrac{1}{N_p} \pmb{Y}_k {\pmb{Y}_k}^T \in \mathbb{R}^{\left(2N_h \right) \times \left(2N_h \right)}
        \end{cases}
    \end{align}
    \tag{3-4-1}
$$
sc-TRCA 的目标函数为：
$$
    \hat{\pmb{w}}_k = \underset{\pmb{w}_k} \argmax \dfrac{\pmb{w}_k \widetilde{\pmb{S}}_k {\pmb{w}_k}^T}{\pmb{w}_k \widetilde{\pmb{Q}}_k {\pmb{w}_k}^T} = 
    \begin{bmatrix}
        \hat{\pmb{u}}_k & \hat{\pmb{v}}_k\\
    \end{bmatrix} \ 
    \in \mathbb{R}^{1 \times \left(N_c + 2N_h \right)}
    \tag{3-4-2}
$$
不难发现，$\pmb{S}_{11}$ 与 $\pmb{Q}_1$ 与 TRCA 目标函数中的 $\pmb{S}_k$、$\pmb{Q}_k$ 是相等的。维度扩增后得到的滤波器需要分割为两个部分：（1）适用于 EEG 数据的 $\hat{\pmb{u}}_k$；（2）适用于正余弦模板的 $\hat{\pmb{v}}_k$。因此判别系数也分为两部分：
$$
    \begin{cases}
        \rho_k^1 = {\rm corr} \left(\hat{\pmb{u}}_k \bar{\pmb{X}}_k, \hat{\pmb{u}}_k \pmb{\mathcal{X}} \right)\\
        \ \\
        \rho_k^2 = {\rm corr} \left(\hat{\pmb{v}}_k \pmb{Y}_k, \hat{\pmb{u}}_k \pmb{\mathcal{X}} \right)\\
    \end{cases}, \ \rho_k = \sum_{i=1}^2 sign \left(\rho_k^i \right) \left(\rho_k^i \right)^2
    \tag{3-4-3}
$$
原文的公式推导不可谓不直观，但是我总觉得差点意思。核心问题在于，我们不明白为什么要这样操作，也很难从中学到什么经验。此外 sc- 技术与原版 TRCA 的联系似乎并不是很紧密，因此接下来我们将从另外一个角度审视上述扩增过程。

根据 3.1 节末尾的分析，我们知道想要改善 TRCA 的模型性能，关键在于对 $\bar{\pmb{X}}_k$ 的信息优化：（1）-R 技术通过正交投影约束了其中的 SSVEP 频率特征成分；（2）ms- 技术通过合并其它频率的样本，约束随机噪声成分的占比。sc- 技术也不例外，它通过空间维度扩增强化 $\bar{\pmb{X}}_k$：
$$
    \widetilde{\pmb{X}}_k = 
    \begin{bmatrix}
        \bar{\pmb{X}}_k\\ \ \\ \pmb{Y}_k\\
    \end{bmatrix} \in \mathbb{R}^{\left(N_c + 2N_h \right) \times N_p}
    \tag{3-4-4}
$$
之后 $\pmb{S}_k$ 的扩增以及 $\hat{\pmb{u}}_k$、$\hat{\pmb{v}}_k$ 的分割就犹如母兔产崽一般顺理成章了：
$$
    \widetilde{\pmb{S}}_k = \widetilde{\pmb{X}}_k {\widetilde{\pmb{X}}_k}^T \ \Longrightarrow \ 
    \begin{bmatrix}
        \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T & \bar{\pmb{X}}_k {\pmb{Y}_k}^T\\
        \ \\
        \pmb{Y}_k {\bar{\pmb{X}}_k}^T & \pmb{Y}_k {\pmb{Y}_k}^T
    \end{bmatrix} \ \Longrightarrow \
    \begin{bmatrix}
        \pmb{S}_{11} & \pmb{S}_{12}\\
        \ \\
        \pmb{S}_{21} & \pmb{S}_{22}\\
    \end{bmatrix}
    \tag{3-4-5}
$$
$$
    \pmb{w}_k \widetilde{\pmb{X}}_k = 
    \begin{bmatrix}
        \hat{\pmb{u}}_k & \hat{\pmb{v}}_k\\
    \end{bmatrix}
    \begin{bmatrix}
        \bar{\pmb{X}}_k\\ \ \\ \pmb{Y}_k\\
    \end{bmatrix} = \hat{\pmb{u}}_k \bar{\pmb{X}}_k + \hat{\pmb{v}}_k \pmb{Y}_k
    \tag{3-4-6}
$$
目标函数分子的物理意义转变为“滤波后**扩增模板**的能量”：
$$
    \begin{align}
        \notag
        \pmb{w}_k \widetilde{\pmb{S}}_k {\pmb{w}_k}^T &= \pmb{w}_k \widetilde{\pmb{X}}_k \left(\pmb{w}_k \widetilde{\pmb{X}}_k \right)^T\\
        \notag \ \\
        \notag
        &= \hat{\pmb{u}}_k \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T {\hat{\pmb{u}}_k}^T + \hat{\pmb{u}}_k \bar{\pmb{X}}_k {\pmb{Y}_k}^T {\hat{\pmb{v}}_k}^T + \hat{\pmb{v}}_k \pmb{Y}_k {\bar{\pmb{X}}_k}^T { \hat{\pmb{u}}_k}^T + \hat{\pmb{v}}_k \pmb{Y}_k {\pmb{Y}_k}^T {\hat{\pmb{v}}_k}^T\\
        \notag \ \\
        \notag
        &= {\rm Var} \left(\hat{\pmb{u}}_k \bar{\pmb{X}}_k \right) + 2 \times {\rm Cov} \left(\hat{\pmb{u}}_k \bar{\pmb{X}}_k, \hat{\pmb{v}}_k \pmb{Y}_k \right) + {\rm Var} \left(\hat{\pmb{v}}_k \pmb{Y}_k \right)\\
    \end{align}
    \tag{3-4-7}
$$
通过（3-4-7）我们可以很清晰地看到分子的能量组成：滤波后 EEG 信号能量、滤波后正余弦模板能量以及二者的互能量（协方差）。其中协方差的存在我个人认为是至关重要的，尽管没有经过测试证实。在从能量角度思考问题时，我们很容易陷入想当然的场景：一段含噪信号的能量（通过幅值平方求和的方式计算）就等于信号能量加上噪声能量。这一观点往往是不准确的，因为“**信号与噪声完全不相关**”这一前提太过严格，大多数生理信号其实并不满足该条件。因此，将信号与噪声的协方差考虑进优化目标是非常有必要的。

尽管如此，sc-TRCA 的分母还是狠狠地给了我一巴掌，同上分析可知：
$$
    \begin{align}
        \notag
        \pmb{w}_k \widetilde{\pmb{Q}}_k {\pmb{w}_k}^T &= \hat{\pmb{u}}_k \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) {\hat{\pmb{u}}_k}^T + N_t \hat{\pmb{v}}_k \pmb{Y}_k {\pmb{Y}_k}^T {\hat{\pmb{v}}_k}^T\\
        \notag \ \\
        \notag
        &= \sum_{i=1}^{N_t} {\rm Var} \left(\hat{\pmb{u}}_k \pmb{X}_k^i \right) + \sum_{i=1}^{N_t} {\rm Var} \left(\hat{\pmb{v}}_k \pmb{Y}_k \right)\\
    \end{align}
    \tag{3-4-8}
$$
可以看到，分母建立在“采集信号中的非 SSVEP 噪声与正余弦信号毫不相关”这一强假设基础上，而且在“各个试次中均存在”，这一点目前恕我不能苟同（~~没准等我测试完以后再次被打脸~~）。

争议暂且搁置，我们来总结一下 sc- 技术的本质，即引入正余弦模板，在空间维度扩增训练数据的信息容量，从而强化模型稳定性。重要的话我再重复一遍：只要对 $\bar{\pmb{X}}$ 进行有效改进（扩增、纯化等），就有希望提高 TRCA 空间滤波器的分类性能，未来各位将在其它 TRCA 改进算法中看到类似的趋势（~~求求各位大佬多发文章没准我还能水一篇综述~~）。


## 3.5 组 TRCA：gTRCA
**(Group TRCA)**

**[论文链接][gTRCA] | 代码：[trca][trca(code)].gtrca()**


## 3.6 交叉相关性 TRCA：xTRCA
**(Cross-{\rm corr}elation TRCA)**

**[论文链接][xTRCA] | 代码：[trca][trca(code)].xtrca()**


[trca(code)]:  https://github.com/BrynhildrW/SSVEP_algorithms/blob/main/trca.py

[ms-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
[Benchmark]:https://ieeexplore.ieee.org/document/7740878/
[UCSD]:https://dx.plos.org/10.1371/journal.pone.0140703
[BETA]:https://www.frontiersin.org/article/10.3389/fnins.2020.00627/full
[TRCA-R]: https://ieeexplore.ieee.org/document/9006809/
[sc-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/abfdfa
[gTRCA]: temp
[xTRCA]: temp
[TDCA]: https://ieeexplore.ieee.org/document/9541393/

***