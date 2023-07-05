# 最小能量组合 & 最大对比度组合
## Minimum Energy Combinations & Maximum Contrast Combinations, MEC & MCC
***

#### [论文链接][MEC_MCC]

本节介绍的两种算法，与 CCA 同属一个年代，甚至经过数学推理后三者之间具备一定的相通性。然而 MEC、MCC 对 SSVEP 信号的建模方法以及“能量滤波”的思想显然比简单粗暴的 CCA 更有意思。

首先我们尽量按原文方式介绍信号模型，部分参数及变量表示方法根据现有研究标准进行了更新。由 $N_e$ 个刺激频率中的第 $k$ 种 $f_k$（单位 Hz）诱发的单试次（索引 $i$ ）、单导联（索引 $j$ ）观测信号 $x_{j,k}^i(t)$ 可表示为三个部分的组合：
$$
    x_{j,k}^i(t) = \underbrace{\sum_{n=1}^{N_h}a_{j,k,n}^i \sin \left( 2 \pi n f_k t + \phi_{j,k}^{i,n} \right)}_{\rm part \ 1}\ + \underbrace{\sum_{m} b_{j,k}^{i,m}z_{j,k}^{i,m}(t)}_{\rm part \ 2} + \underbrace{e_{j,k}^i(t)}_{\rm part \ 3} \tag{1}
$$
公式中有很多符号及上下角标，且听慢慢道来：

（i）**诱发响应信号**：由频-相调制的、刺激频率及其高次谐频正弦信号 $\sin \left( 2 \pi n f_k t + \phi_{j,k}^{i,n} \right)$ 组成，其中 $n$ 表示谐波次数（最高 $N_h$ 次），$a_{j,k}^i$ 为幅值调制系数，$\phi_{j,k}^{i,n}$ 表示相位调制系数。原文中相位 $\phi$ 仅与导联 $j$ 、谐波次数 $n$ 有关。实际研究指出，单次诱发信号的潜伏期是浮动的，即 $\phi$ 与试次 $i$ 有关。而且目前 SSVEP 常采用 JFPM 编码方法，即频率 $f_k$ 对应于独特的初始相位，因此 $\phi$ 也与 $k$ 有关。当然在后续推导过程中，需视情况削减变量个数。

（ii）**生理噪声信号**：该信号是一组干扰成分 $z_{j,k}^{i,m}(t)$ 的线性加权和，包括内部噪声（如背景脑电活动）与外部噪声（如呼吸伪迹、肌电干扰、工频干扰等）共计 $N_m$ 类，其中 $b_{j,k}^{i,m}$ 是噪声权重系数。原文中 $b$ 仅与导联 $j$、噪声类别索引 $m$ 有关。式 (1) 中新增了时间、刺激因素，后续过程中视情况简化。

（iii）**采集设备噪声**：该信号不可避免，不可量化评估，是硬件设备固定存在的基底噪声。

信号模型进一步可升维至多导联矩阵维度，包括观测信号 $\pmb{X}_k^i \in \mathbb{R}^{N_c \times N_p}$、正弦信号权重矩阵 $\pmb{A}_k^i \in \mathbb{R}^{N_c \times 2N_h}$、特征响应信号 $\pmb{Y}_k^i \in \mathbb{R}^{2N_h \times N_p}$、生理噪声分布权重矩阵 $\pmb{B}_k^i \in \mathbb{R}^{N_c \times N_m}$、生理噪声信号 $\pmb{Z}_k^i \in \mathbb{R}^{N_m \times N_p}$ 以及采集噪声 $\pmb{E}_k^i \in \mathbb{R}^{N_c \times N_p}$：
$$
    \pmb{X}_k^i = \pmb{A}_k^i \pmb{Y}_k^i + \pmb{B}_k^i \pmb{Z}_k^i + \pmb{E}_k^i
    \tag{2}
$$
该模型默认数据 $\pmb{X}_k^i$ 经过了零均值化（带通/带陷滤波）、方差标准化预处理。接下来作者在文中提出了多达 **6** 种空间滤波方法（原文描述是“信道筛选\构成方法”），第 5、第 6 种分别为 MEC、MCC（结果自然是这俩效果最好啦）。虽然这篇文章是 07 年的老古董了，但是这六种设计可谓是环环相扣、层层递进，其推理过程精彩纷呈。出于对作者团队的尊重，我认为有必要在此介绍全部方案。

## Method I: Average Combination
实行该方案需满足两个前提：

一是特征信号相位与各影响因素均无关，即 $\phi$ 始终恒定；

二是参与空间滤波的各导联相关性很低，即生理噪声信号 $\pmb{Z}_k^i$ 稀疏分布于各导联。

在此基础上对各导联信号进行简单叠加平均（等权重）可有效抑制导联特异性噪声成分，进而提升理论信噪比。相应的空间滤波器为：
$$
    \pmb{w}_k^i = 
        \begin{bmatrix}
            1 & 1 & \cdots & 1
        \end{bmatrix} \in \mathbb{R}^{1 \times N_c}
    \tag{3}
$$

## Method II: Native Combination
Method I 存在一个致命缺陷，即 SSVEP 正弦态响应的相位在头皮上各导联的分布情况并非均一不变，这是由脑电信号的传播性质决定的。尽管原文洋洋洒洒解释了一整段 Method II 的思想，总结来看其实就是两个字：“**摆烂**”——将 $\pmb{W}$ 设置为单位矩阵 $\pmb{I}$，空间滤波名存实亡：
$$
    \pmb{w}_k^i = \begin{bmatrix}
        1 & 0 & \cdots & 0\\
        0 & 1 & \ddots & 0\\
        \vdots & \vdots & \ddots & \vdots\\
        0 & 0 & \cdots & 1\\
    \end{bmatrix} \in \mathbb{R}^{N_c \times N_c}
    \tag{4}
$$

## Method III: Bipolar Combination
Method III 直译就是“双极组合”，这是一种（~~很久以前~~）常见的差分信号提取方法，曾用于 SSVEP 信号特征的增强（[1][ref1]、[2][ref2]）。其原理是通过差分输出抑制相邻信道中的共模噪声，突出差异信号特征。这种思想常见于拉普拉斯电极设计，即后续将要介绍的 Method IV。双极差分的特点决定了参与滤波的导联数目必须是偶数，且仅进行一次差分，即双通道压缩为单通道、四通道压缩为双通道，以此类推：
$$
    \pmb{w}_k^i = \begin{bmatrix}
        1 & -1\\
    \end{bmatrix}, \ 
    \begin{bmatrix}
        1 & -1 & 0 & 0\\
        0 & 0 & 1 & -1\\
    \end{bmatrix}, \ 
    \begin{bmatrix}
        1 & -1 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & -1 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & -1\\
    \end{bmatrix}, \ \cdots
    \tag{5}
$$

## Method IV: Laplacian Combination
表面 Laplacian 组合是双极差分组合的优化形式，具体步骤为设立单导联为中心、周围环绕导联与之进行差分，并最终输出单通道数据。通常选用“1+4”的组合模式，即：
$$
    \pmb{w}_k^i = \begin{bmatrix}
        4 & -1 & -1 & -1 & -1\\
    \end{bmatrix}
    \tag{6}
$$

## Method V: Minimum Energy Combination
终于来到了本节的重头戏。MEC 可视为双极差分和拉普拉斯差分的拓展，即寻找一种导联线性组合的方式使各导联间的共模噪声能量得以衰减（所谓“最小能量组合”）。首先通过正交投影尽量移除观测信号 $\pmb{X}_k^i$ 中的 SSVEP 响应成分 $\pmb{Y}_k$。此处推导简化了试次影响因素 $i$，即认为诱发信号的试次间相位（潜伏期）保持恒定。剔除有效成分后的 $\tilde{\pmb{X}}_k^i$ 可视为仅包含各类噪声的信号成分：
$$
    \tilde{\pmb{X}}_k^i = \pmb{X}_k^i - \pmb{X}_k^i \underbrace{{\pmb{Y}_k}^T \left( \pmb{Y}_k {\pmb{Y}_k}^T \right)^{-1} \pmb{Y}_k}_{\pmb{P}_k^i}, \
    \ \tilde{\pmb{X}}_k^i \approx \pmb{B}_k^i \pmb{Z}_k^i + \pmb{E}_k^i
    \tag{7}
$$
接下来通过空间滤波使得该部分能量最小：
$$
    \hat{\pmb{w}}_k^i = \underset{\pmb{w}_k^i} \argmin \|\pmb{w}_k^i \tilde{\pmb{X}}_k^i\|_2^2
        = \underset{\pmb{w}_k^i} \argmin \left( \pmb{w}_k^i \tilde{\pmb{X}}_k^i {\tilde{\pmb{X}}_k^i}^T {\pmb{w}_k^i}^T \right) \in \mathbb{R}^{1 \times N_c}
    \tag{8}
$$
严格意义来说，式 (8) 还有一个关于空间滤波器本身的隐藏约束条件（文中并未指出），即：
$$
    \pmb{w}_k^i {\pmb{w}_k^i}^T = 1
    \tag{9}
$$
如若不然，大家可以设想一下当 $\pmb{w}_k^i=\pmb{0}$ 时会发生什么情况。所以式 (8) 的最优化问题实质为：
$$
    \hat{\pmb{w}}_k^i = \underset{\pmb{w}_k^i} \argmin \dfrac{\pmb{w}_k^i \tilde{\pmb{X}}_k^i {\tilde{\pmb{X}}_k^i}^T {\pmb{w}_k^i}^T} {\pmb{w}_k^i {\pmb{w}_k^i}^T}
    \tag{10}
$$
式 (10) 是一个标准的 Rayleigh Quotient 问题，其求解过程不再赘述。在特征向量数目（投影子空间维度）上，文章提出了一个颇具创意的设计。首先将特征向量按对应特征值大小**升序**排列并拼接成矩阵 $\hat{\pmb{W}}_k^i$，对 $\hat{\pmb{W}}_k^i$ 进行幅值平衡处理：
$$
    \hat{\pmb{W}}_k^i = \begin{bmatrix}
        \dfrac{\hat{\pmb{w}}_k^i(:,1)}{\sqrt{\lambda_1}} & \dfrac{\hat{\pmb{w}}_k^i(:,2)}{\sqrt{\lambda_2}} & \cdots & \dfrac{\hat{\pmb{w}}_k^i(:,N_c)}{\sqrt{\lambda_{N_c}}}
    \end{bmatrix} \in \mathbb{R}^{N_c \times N_c}
    \tag{11}
$$
接着依照如下标准确定特征向量数目 $N_k$：
$$
    N_k = \min \{ N | \dfrac{\sum_{i=1}^N \lambda_i}{\sum_{j=1}^{N_c} \lambda_j} > 0.1\}
    \tag{12}
$$
按作者的设想，这样的筛选能够保证接近 90% 的噪声能量衰减率。有效信号虽然有所损失，但总收益依然是正的。

## Method VI: Maximum Contrast Combination
显然作者还是知道 Method V 是一柄双刃剑，所以他们提出了一个更理想化的方案：在削减噪声能量的同时增强 SSVEP 的有效频率成分：
$$
    \hat{\pmb{w}}_k^i = \underset{\pmb{w}_k^i} \argmax \dfrac{\| \pmb{w}_k^i \pmb{X}_k^i \|_2^2}{\| \pmb{w}_k^i \tilde{\pmb{X}}_k^i \|_2^2}
        = \underset{\pmb{w}_k^i} \argmax \dfrac{\pmb{w}_k^i \pmb{X}_k^i {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T} {\pmb{w}_k^i \tilde{\pmb{X}}_k^i {\tilde{\pmb{X}}_k^i}^T {\pmb{w}_k^i}^T}
    \tag{13}
$$
考虑到正交投影矩阵 $\pmb{P}_k^i \in \mathbb{R}^{M \times M}$ 通常具备性质 $\pmb{P}_k^i = {\pmb{P}_k^i}^T = \pmb{P}_k^i {\pmb{P}_k^i}^T$，因此式 (13) 可作如下转化：
$$
    \begin{align}
        \notag \pmb{w}_k^i \tilde{\pmb{X}}_k^i {\tilde{\pmb{X}}_k^i}^T {\pmb{w}_k^i}^T &= \left(\pmb{w}_k^i \pmb{X}_k^i - \pmb{w}_k^i \pmb{X}_k^i \pmb{P}_k^i \right) \left(\pmb{w}_k^i \pmb{X}_k^i - \pmb{w}_k^i \pmb{X}_k^i \pmb{P}_k^i \right)^T\\
        \notag \\
        \notag &= \pmb{w}_k^i \pmb{X}_k^i {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T - 2 \pmb{w}_k^i \pmb{X}_k^i \pmb{P}_k^i {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T + \pmb{w}_k^i \pmb{X}_k^i \pmb{P}_k^i {\pmb{P}_k^i }^T {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T\\
        \notag \\
        \notag &= \pmb{w}_k^i \pmb{X}_k^i {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T - \pmb{w}_k^i \pmb{X}_k^i \pmb{P}_k^i {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T
        \tag{14}
    \end{align}
$$
$$
    \hat{\pmb{w}}_k^i = \underset{\pmb{w}_k^i} \argmin \left( 1 - \dfrac{\pmb{w}_k^i \pmb{X}_k^i \pmb{P}_k^i {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T} {\pmb{w}_k^i \pmb{X}_k^i {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T} \right) =
        \underset{\pmb{w}_k^i} \argmax \dfrac{\pmb{w}_k^i \pmb{X}_k^i \pmb{P}_k^i {\pmb{P}_k^i}^T {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T} {\pmb{w}_k^i \pmb{X}_k^i {\pmb{X}_k^i}^T {\pmb{w}_k^i}^T}
    \tag{15}
$$
式 (15) 的形式与 (e)TRCA-R 非常相似，可惜当时研究者没有想到使用叠加平均模板替代标准正余弦信号成为 SSVEP 响应，否则历史的进程有可能提前整整 13 年，哈利·波特都能见证阿兹卡班的囚徒重获新生了。也正因为当时研究者对于信号模板的认识不够全面，所以后续分类判据依然选择了落后的频域能量判据，也许限制了该模型分类性能的充分释放。
***

[MEC_MCC]: https://ieeexplore.ieee.org/document/4132932/
[ref1]: http://ieeexplore.ieee.org/document/847819/
[ref2]: https://iopscience.iop.org/article/10.1088/1741-2560/2/4/008
