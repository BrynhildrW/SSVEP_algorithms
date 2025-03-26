# 多重刺激 CCA：msCCA
## Multi-stimulus CCA, msCCA
---

### [论文链接][msCCA]

**<font color="#dd0000">（大量数学前置知识警告）</font>**

今天我们来膜拜澳门大学的内卷发动机 *Chi Man Wong* 和 *Feng Wan* 老师团队。之所以对他们“赞誉有加”，主要有三方面原因：

（1）**算法有用，但只有一点用**：他们提出的一系列 SSVEP 算法在公开数据集与竞赛数据集中具有极大优势（即样本量不足的情况）。不过在数据样本量充足的情况下，与传统的 (e)TRCA 算法难以拉开差距；

（2）**强悍如斯，地都快耕坏了**：他们的每篇论文都充分（~~往死里~~）挖掘了公开数据集的可用潜力，从 [Benchmark][Benchmark]、[UCSD][UCSD] 再到 [BETA][BETA] 都能看到他的身影，从 CCA 到 ms-(e)TRCA 各种花里胡哨的算法都测了个遍（~~根本不给人活路~~），低频 SSVEP-BCI 系统的解码被他卷得翻江倒海，再无探索空间。

（3）**故弄玄虚，堆砌数学壁垒**：该团队 2020 年发表的一篇关于[空间滤波器构建框架][TRCA-R]的综述性论文就是万恶之源。在其文章中，经常使用怪僻的希腊字母、花体英文字母作为变量名称，为了形式简约而把简单的实际操作过程复杂化。例如明明是人畜无害的试次叠加平均：
$$
\bar{\pmb{X}} = \dfrac{1} {N_t} \sum_{n=1}^{N_t} \pmb{X}_i, \ \pmb{X}_i \in \mathbb{R}^{Nc \times Np}
\tag{1}
$$
为了凑上自己提出的框架，硬要给你表演一套天书：
$$
\oplus_{i=1}^{N_t} \pmb{X}_i = 
\begin{bmatrix}
\pmb{X}_1 & \pmb{0} & \cdots & \pmb{0}\\
\pmb{0} & \pmb{X}_2 & \cdots & \pmb{0}\\
\vdots & \vdots & \ddots & \vdots\\
\pmb{0} & \cdots & \cdots & \pmb{X}_{N_t}
\end{bmatrix}, \ 
\bar{\pmb{X}} = \dfrac{1} {N_t} \pmb{\mathcal{I}}_{N_t, N_c} \cdot \left[\oplus_{i=1}^{N_t} \pmb{X}_i \right] \cdot \pmb{\mathcal{I}}_{N_t, N_p}^T
\tag{2}
$$
组里有些萌新，一板一眼地照着论文公式复现算法，结果训练样本一多，程序不仅运行慢，还动不动就内存溢出。从原始数据结构中创建 $\oplus_{i=1}^{N_t} \pmb{X}_i$、$\pmb{\mathcal{I}}_{N_t, N_p}$ 这样的大矩阵送去后端运算，相当于先做一把电锯去再杀鸡炖汤，能不慢吗？家庭厨房里接上工业用电，能不溢出吗？

不可否认的是，*Chi Man Wong* 及其团队对于 SSVEP 信号解码的研究是成体系的、步骤严谨的，他们提出的空间滤波器框架适配了自 CCA 以来的各种算法，为后续研究工作打开了思路。更重要的是，他们团队以及清华大学 *Yijun Wang*、*Xiaogang Chen* 等老师带领的团队，都不搞弯道超车，不搞非对称竞争，每一个研究思路都是建立在已有研究基础上，每一步新的收获都会切实体现为文章成果。这样的团队对于学生培养是利好的，学生不用担心梭哈一个老板异想天开的课题而愁于毕业困境。因此再让我跑题一次：**<font color="#dd0000">但凡遇到老板鼓吹自己手握多少项目、每年经费多少万、带领多少优秀青年教师团队、手下多少研究生之类的话术，一定要慎之又慎</font>**。你要知道，牛逼吹得不够大是不能吸引上边的人投资的，牛逼吹起来了就是在梭哈你自己的学术生涯与宝贵光阴。老板项目结不了题顶多延期，少赚一点经费，少评一些名声，日子一分都不会难受。你延期延掉的是什么还请自己掂量清楚。

言归正传，我们首先有必要介绍一下 *Wong* 提出的统一框架。*Wong* 以及部分研究者喜欢按列展示向量，而本文中向量统一按行排布（~~我偏不，老子就喜欢按行~~），因此部分公式可能在形式上与原文有所出入，但本质是一样的：
$$
\pmb{\mathcal{Z}} \pmb{\mathcal{D}} \pmb{\mathcal{P}} \pmb{\mathcal{P}}^T {\pmb{\mathcal{D}}}^T \pmb{\mathcal{Z}}^T \pmb{W} = 
\begin{cases}
\pmb{\mathcal{Z}} \pmb{\mathcal{D}} \pmb{\mathcal{D}}^T \pmb{\mathcal{Z}}^T \pmb{W} \pmb{\Lambda}, \ {\rm Type I}\\
\\
\pmb{W} \pmb{\Lambda}, \ {\rm Type II}
\end{cases}
\tag{3}
$$
这里 $\pmb{W}$ 与 $\pmb{\Lambda}$ 之所以写成矩阵而不是“向量+标量”形式，是因为空间滤波器并不总是将多通道信号压缩至一维，对于需要进行压缩的一般情况，只需取方阵 $\pmb{\mathcal{Z}} \pmb{\mathcal{D}} \pmb{\mathcal{P}} \pmb{\mathcal{P}}^T {\pmb{\mathcal{D}}}^T \pmb{\mathcal{Z}}^T$ 的最大（小）特征值对应的特征向量即可；而当需要保留多个维度（投影子空间）时， $\pmb{W}$ 的最优解为多个特征向量的拼接，拼接顺序以对应特征向量的大小顺序为准。

接下来我不想再去复述他们文章中对各种算法的具体匹配方式，仅在此对式 (1-4-3) 中的主要成分进行简单介绍：

$\pmb{\mathcal{Z}}$ 是数据（默认按列排布）的集合矩阵，可能是（1）单个数据矩阵；（2）形如 $\bigoplus_{i=1}^{N_t} \pmb{X}_i$ 的多种数据块对角拼接组成的联合矩阵。一般来说（2）中的对角联合矩阵，在整体公式中需要经过 $\pmb{\mathcal{I}}$ 矩阵的变形处理,将其转换为由多个数据块**横向**或**纵向**拼接而成的大矩阵，如式 (11) ；

$\pmb{\mathcal{D}}$ 是时域滤波矩阵，除了滤波器组技术以外，通常预处理（带通滤波）结束后的数据无需再进行时域滤波，即 $\pmb{\mathcal{D}} = \pmb{I}$；

$\pmb{\mathcal{P}}$ 为正交投影矩阵，通常满足 $\pmb{\mathcal{P}} = {\pmb{\mathcal{P}}}^T = \pmb{\mathcal{P}} {\pmb{\mathcal{P}}}^T = {\pmb{\mathcal{P}}}^T \pmb{\mathcal{P}}$。根据给定的投影方向（$\pmb{T}$），可表示为:
$$
\begin{cases}
\pmb{\mathcal{P}} = \pmb{T}^T \left(\pmb{T} \pmb{T}^T \right)^{-1} \pmb{T}
    = \pmb{Q}_{\pmb{T}} {\pmb{Q}_{\pmb{T}}}^T\\
\\
\pmb{T} = \pmb{Q}_{\pmb{T}} \pmb{R}_{\pmb{T}}, \ Reduced \ QR \ decomposition
\end{cases}
\tag{4}
$$
不难发现，该框架的数学本质是一系列**广义特征值** ( *Generalized eigenvalue problems, GEPs* ) 方程，而空间滤波器构建过程中常见的**广义瑞利商** ( *Generalized Rayleigh quotient* ) 问题通常又可以转化为 *GEP* 方程加以求解，因此该框架几乎能够契合现有各种 SSVEP-BCI 系统中常见的空间滤波器算法。尽管如此，除了 *Wong* 设计的算法，我基本上不会使用这个框架来展示公式。原因除了之前吐槽过的“数学墙”以外，还有很重要的一点，即**凭空创造**、**设计**更好的时域滤波矩阵 $\pmb{\mathcal{D}}$ 或正交投影矩阵 $\pmb{\mathcal{P}}$ 都是不现实的，想要从数学上证明某个投影空间具有某些数学特性或优势都极具挑战性。我个人更倾向于通过**直观物理含义**的途径来阐释算法原理，希望通过我的讲解，能够让大家实现数学原理、编程实践与性能优化的三者合一，从而更好地掌握算法的精髓、洞察未来发展方向。

在本节前置数学知识的最后，给大家简单介绍一下广义瑞利商及其与 *GEP* 问题的关系。形如式 (5) 所示的函数称为瑞利商 ( *Rayleigh quotient* )，其中 $\pmb{A}$ 为 *Hermitte* 矩阵：
$$
f(\pmb{\omega}) = \dfrac{\pmb{\omega} \pmb{A} {\pmb{\omega}}^T} {\pmb{\omega} {\pmb{\omega}}^T}, \ 
\pmb{A} \in \mathbb{R}^{N \times N}, \ 
\pmb{\omega} \in \mathbb{R}^{1 \times N}
\tag{5}
$$
一般最优化问题需要求解的是瑞利商的最值。将式 (5) 转化为最优化问题的标准描述形式，之后利用 *Lagrandian* 乘子法构建函数 $J$：
$$
\begin{cases}
\underset{\pmb{\omega}} \max \ \pmb{\omega} \pmb{A} {\pmb{\omega}}^T\\
s.t.\ \pmb{\omega} {\pmb{\omega}}^T = 1
\end{cases} \ \Longrightarrow \ 
J(\pmb{\omega}) = \pmb{\omega} \pmb{A} {\pmb{\omega}}^T - \lambda \left(\pmb{\omega} {\pmb{\omega}}^T - 1 \right)
\tag{6}
$$
对 $J$ 求导并置零，最终可得特征值方程：
$$
\dfrac{dJ(\pmb{\omega})}{d \pmb{\omega}} = 2 \pmb{A} {\pmb{\omega}}^T - 2 \lambda {\pmb{\omega}}^T = 0
\to 
\pmb{A} {\pmb{\omega}}^T = \lambda {\pmb{\omega}}^T
\tag{7}
$$
至此可以看出，瑞利商的最值即为方阵 $\pmb{A}$ 最大（小）特征值，取最值时的解即为该特征值对应的特征向量。至于广义瑞利商，其形如式 (8) 所示的函数，$\pmb{B}$ 同样也是 *Hermitte* 矩阵：
$$
f(\pmb{\omega}) = \dfrac{\pmb{\omega} \pmb{A} {\pmb{\omega}}^T} {\pmb{\omega} \pmb{B} {\pmb{\omega}}^T}, \ 
\pmb{A},\pmb{B} \in \mathbb{R}^{N \times N}, \ 
\pmb{\omega} \in \mathbb{R}^{1 \times N}
\tag{8}
$$
同上进行类似操作，可以得到式 (9) 所示的 *GEP* 方程。由此可知广义瑞利商的最值即为方阵 $\pmb{B}^{-1} \pmb{A}$ 最大（小）特征值对应的特征向量：
$$
\pmb{A} {\pmb{\omega}}^T = \lambda \pmb{B} {\pmb{\omega}}^T \ 
\Longrightarrow \ 
\left(\pmb{B}^{-1} \pmb{A} \right) {\pmb{\omega}}^T  = \lambda {\pmb{\omega}}^T
\tag{9}
$$
接下来我们来看 msCCA 算法。首先给出统一框架 （**Type I**） 下的各部分组成：
$$
\begin{cases}
\pmb{\mathcal{Z}} = \pmb{\mathcal{I}}_{N_e,N_c} \left( \oplus_{k=1}^{N_e}{\bar{\pmb{X}}_k} \right) \in \mathbb{R}^{N_c \times (N_e N_p)}\\
\\
\pmb{\mathcal{D}} = \pmb{I}_{N_e N_p} \in \mathbb{R}^{\left(N_e N_p \right) \times \left(N_e N_p \right)}\\
\\
\pmb{\mathcal{P}} = \pmb{Q}_{\pmb{\mathcal{Y}}} {\pmb{Q}_{\pmb{\mathcal{Y}}}}^T =  \in \mathbb{R}^{\left(N_e N_p \right) \times \left(N_e N_p \right)}
\end{cases}
\tag{10}
$$
其中：
$$
\pmb{\mathcal{Y}} = 
\begin{bmatrix}
\pmb{Y}_1 & \pmb{Y}_2 & \cdots & \pmb{Y}_{N_e}
\end{bmatrix} \in \mathbb{R}^{\left(2N_h \right) \times \left(N_e N_p \right)}
\tag{11}
$$
$$
\pmb{Q}_{\pmb{\mathcal{Y}}} {\pmb{Q}_{\pmb{\mathcal{Y}}}}^T = 
\begin{bmatrix}
\pmb{Q}_{\pmb{Y}_1} {\pmb{Q}_{\pmb{Y}_1}}^T & \pmb{Q}_{\pmb{Y}_1} {\pmb{Q}_{\pmb{Y}_2}}^T & \cdots & \pmb{Q}_{\pmb{Y}_1} {\pmb{Q}_{\pmb{Y}_{N_e}}}^T\\
\pmb{Q}_{\pmb{Y}_2} {\pmb{Q}_{\pmb{Y}_1}}^T & \pmb{Q}_{\pmb{Y}_2} {\pmb{Q}_{\pmb{Y}_2}}^T & \cdots & \pmb{Q}_{\pmb{Y}_2} {\pmb{Q}_{\pmb{Y}_{N_e}}}^T\\
\vdots & \vdots & \ddots & \vdots\\
\pmb{Q}_{\pmb{Y}_{N_e}} {\pmb{Q}_{\pmb{Y}_1}}^T & \pmb{Q}_{\pmb{Y}_{N_e}} {\pmb{Q}_{\pmb{Y}_2}}^T & \cdots & \pmb{Q}_{\pmb{Y}_{N_e}} {\pmb{Q}_{\pmb{Y}_{N_e}}}^T\\
\end{bmatrix}
\tag{12}
$$
不要被这些花里胡哨的公式迷乱了双眼，我们来看看每一步都发生了什么：
$$
\pmb{\mathcal{Z}} = 
\underbrace{
\begin{bmatrix}
\pmb{I}_{N_c} & \pmb{I}_{N_c} & \cdots & \pmb{I}_{N_c}
\end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_e N_c \right)}}
\underbrace{
\begin{bmatrix}
\bar{\pmb{X}}_1 & \pmb{0} & \cdots & \pmb{0}\\
\pmb{0} & \bar{\pmb{X}}_2 & \cdots & \pmb{0}\\
\vdots & \vdots & \ddots & \vdots\\
\pmb{0} & \pmb{0} & \cdots & \bar{\pmb{X}}_{N_e}\\
\end{bmatrix}}_{\mathbb{R}^{\left(N_e N_c \right) \times \left(N_e N_p \right)}} = 
\underbrace{
\begin{bmatrix}
\bar{\pmb{X}}_1 & \bar{\pmb{X}}_2 & \cdots & \bar{\pmb{X}}_{N_e}
\end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_e N_p \right)}}
\tag{13}
$$
$$
\pmb{\mathcal{Z}} \pmb{\mathcal{D}} = 
\underbrace{
\begin{bmatrix}
\bar{\pmb{X}}_1 & \bar{\pmb{X}}_2 & \cdots & \bar{\pmb{X}}_{N_e}
\end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_e N_p \right)}}
\underbrace{
\begin{bmatrix}
\pmb{I}_{N_p} & \pmb{0} & \cdots & \pmb{0}\\
\pmb{0} & \pmb{I}_{N_p} & \cdots & \pmb{0}\\
\vdots & \vdots & \ddots & \vdots\\
\pmb{0} & \pmb{0} & \cdots & \pmb{I}_{N_p}\\
\end{bmatrix}}_{\mathbb{R}^{\left(N_e N_p \right) \times \left(N_e N_p \right)}} = \pmb{\mathcal{Z}}
\tag{14}
$$
$$
\pmb{\mathcal{Z}} \pmb{\mathcal{D}} \pmb{\mathcal{P}} = 
\underbrace{
\begin{bmatrix}
\sum_k{\bar{\pmb{X}}_k \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_1}}^T} & \sum_k{\bar{\pmb{X}}_k \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_2}}^T} & \cdots & \sum_k{\bar{\pmb{X}}_k \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_{N_e}}}^T}\\
\end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_e N_p \right)}}
\tag{15}
$$
$$
\pmb{\mathcal{Z}} \pmb{\mathcal{D}} \pmb{\mathcal{P}} {\pmb{\mathcal{P}}}^T {\pmb{\mathcal{D}}}^T {\pmb{\mathcal{Z}}}^T =
\sum_{c=1}^{N_e} \left[\sum_{a=1}^{N_e} \bar{\pmb{X}}_a \pmb{Q}_{\pmb{Y}_a} {\pmb{Q}_{\pmb{Y}_c}}^T \left(\sum_{b=1}^{N_e} \bar{\pmb{X}}_b \pmb{Q}_{\pmb{Y}_b} {\pmb{Q}_{\pmb{Y}_c}}^T \right)^T \right]\\
\ \\
\xrightarrow{{\pmb{Q}_{\pmb{Y}_c}}^T \pmb{Q}_{\pmb{Y}_c} = \pmb{I}} N_e \sum_{b=1}^{N_e} \sum_{a=1}^{N_e} \bar{\pmb{X}}_a \pmb{Q}_{\pmb{Y}_a} {\pmb{Q}_{\pmb{Y}_b}}^T \bar{\pmb{X}}_b
\tag{16}
$$
$$
\pmb{\mathcal{Z}} \pmb{\mathcal{D}} {\pmb{\mathcal{D}}}^T {\pmb{\mathcal{Z}}}^T = \pmb{\mathcal{Z}} \pmb{\mathcal{Z}}^T = 
\underbrace{
\begin{bmatrix}
\bar{\pmb{X}}_1 & \bar{\pmb{X}}_2 & \cdots & \bar{\pmb{X}}_{N_e}\\
\end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_e N_p \right)}}
\underbrace{
\begin{bmatrix}
\bar{\pmb{X}}_1\\
\bar{\pmb{X}}_2\\
\vdots\\
\bar{\pmb{X}}_{N_e}\\
\end{bmatrix}}_{\mathbb{R}^{\left(N_e N_p \right) \times N_c}} = 
\sum_{k=1}^{N_e} \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T
\tag{17}
$$
可真是费了好一番力气才完成。最终 *GEP* 方程为（仅需一维投影向量）：
$$
\left(N_e \sum_{b=1}^{N_e} \sum_{a=1}^{N_e} \bar{\pmb{X}}_a \pmb{Q}_{\pmb{Y}_a} {\pmb{Q}_{\pmb{Y}_b}}^T {\bar{\pmb{X}}_b}^T \right) \pmb{\omega}^T = 
\lambda \left(\sum_{k=1}^{N_e} \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T \right) \pmb{\omega}^T
\tag{18}
$$
判别过程为：
$$
\rho_k = corr \left(\hat{\pmb{u}}_k \pmb{\mathcal{X}}, \hat{\pmb{v}}_k \pmb{Y}_k \right), \ \hat{k} = \underset{k} \argmax{\{\rho_k}\}
\tag{19}
$$
相信各位很快就会发现，这个公式没有为我们提供任何直白的、一般人能阅读的有效信息。坦白地说，仅靠式 (18) 设计的滤波器以及相应的模板匹配方法是不完整的，具体原因请各位移步 ms-eCCA，我们将从另一个更合理的角度审视这个算法。

不得不说 *Wong* 这一手阉割刀法堪比老黄，先在 *JNE* 上发表精心推导设计的两大 ms- 算法，再在 *IEEE TBME* 上发表统一框架，顺带蜻蜓点水一般地，用框架小增小改就套出了这个丐版 msCCA，还比老旧的 itCCA 强上一截，以此彰显框架的“易用性”，其实根本没这么浅显，套模型套框架也并非研学之道。

---

[msCCA]: https://ieeexplore.ieee.org/document/9006809/
[Benchmark]: https://ieeexplore.ieee.org/document/7740878/
[UCSD]: https://dx.plos.org/10.1371/journal.pone.0140703
[BETA]: https://www.frontiersin.org/article/10.3389/fnins.2020.00627/full