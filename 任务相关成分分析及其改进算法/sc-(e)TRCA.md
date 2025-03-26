# 相似度约束 TRCA
## Similarity-constrained (e)TRCA

---

[论文链接][sc-TRCA]

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
\tag{1}
$$
sc-TRCA 的目标函数为：
$$
\hat{\pmb{w}}_k = \underset{\pmb{w}_k} \argmax \dfrac{\pmb{w}_k \widetilde{\pmb{S}}_k {\pmb{w}_k}^T}{\pmb{w}_k \widetilde{\pmb{Q}}_k {\pmb{w}_k}^T} = 
\begin{bmatrix}
\hat{\pmb{u}}_k & \hat{\pmb{v}}_k\\
\end{bmatrix} \ 
\in \mathbb{R}^{1 \times \left(N_c + 2N_h \right)}
\tag{2}
$$
不难发现，$\pmb{S}_{11}$ 与 $\pmb{Q}_1$ 与 TRCA 目标函数中的 $\pmb{S}_k$、$\pmb{Q}_k$ 是相等的。维度扩增后得到的滤波器需要分割为两个部分：（1）适用于 EEG 数据的 $\hat{\pmb{u}}_k$；（2）适用于正余弦模板的 $\hat{\pmb{v}}_k$。因此判别系数也分为两部分：
$$
\begin{cases}
\rho_k^1 = {\rm corr} \left(\hat{\pmb{u}}_k \bar{\pmb{X}}_k, \hat{\pmb{u}}_k \pmb{\mathcal{X}} \right)\\
\ \\
\rho_k^2 = {\rm corr} \left(\hat{\pmb{v}}_k \pmb{Y}_k, \hat{\pmb{u}}_k \pmb{\mathcal{X}} \right)\\
\end{cases}, \ \rho_k = \sum_{i=1}^2 sign \left(\rho_k^i \right) \left(\rho_k^i \right)^2
\tag{3}
$$
原文的公式推导不可谓不直观，但是我总觉得差点意思。核心问题在于，我们不明白为什么要这样操作，也很难从中学到什么经验。此外 sc- 技术与原版 TRCA 的联系似乎并不是很紧密，因此接下来我们将从另外一个角度审视上述扩增过程。

想要改善 TRCA 的模型性能，关键在于对 $\bar{\pmb{X}}_k$ 的信息优化：（1）-R 技术通过正交投影约束了其中的 SSVEP 频率特征成分；（2）ms- 技术通过合并其它频率的样本，约束随机噪声成分的占比。sc- 技术也不例外，它通过空间维度扩增强化 $\bar{\pmb{X}}_k$：
$$
\widetilde{\pmb{X}}_k = 
\begin{bmatrix}
\bar{\pmb{X}}_k\\ \ \\ \pmb{Y}_k\\
\end{bmatrix} \in \mathbb{R}^{\left(N_c + 2N_h \right) \times N_p}
\tag{4}
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
\tag{5}
$$
$$
\pmb{w}_k \widetilde{\pmb{X}}_k = 
\begin{bmatrix}
\hat{\pmb{u}}_k & \hat{\pmb{v}}_k\\
\end{bmatrix}
\begin{bmatrix}
\bar{\pmb{X}}_k\\ \ \\ \pmb{Y}_k\\
\end{bmatrix} = \hat{\pmb{u}}_k \bar{\pmb{X}}_k + \hat{\pmb{v}}_k \pmb{Y}_k
\tag{6}
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
\tag{7}
$$
通过式（7）我们可以很清晰地看到分子的能量组成：滤波后 EEG 信号能量、滤波后正余弦模板能量以及二者的互能量（协方差）。其中协方差的存在我个人认为是至关重要的，尽管没有经过测试证实。在从能量角度思考问题时，我们很容易陷入想当然的场景：一段含噪信号的能量（通过幅值平方求和的方式计算）就等于信号能量加上噪声能量。这一观点往往是不准确的，因为“**信号与噪声完全不相关**”这一前提太过严格，大多数生理信号其实并不满足该条件。因此，将信号与噪声的协方差考虑进优化目标是非常有必要的。

尽管如此，sc-TRCA 的分母还是狠狠地给了我一巴掌，同上分析可知：
$$
\begin{align}
\notag
\pmb{w}_k \widetilde{\pmb{Q}}_k {\pmb{w}_k}^T &= \hat{\pmb{u}}_k \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) {\hat{\pmb{u}}_k}^T + N_t \hat{\pmb{v}}_k \pmb{Y}_k {\pmb{Y}_k}^T {\hat{\pmb{v}}_k}^T\\
\notag \ \\
\notag
&= \sum_{i=1}^{N_t} {\rm Var} \left(\hat{\pmb{u}}_k \pmb{X}_k^i \right) + \sum_{i=1}^{N_t} {\rm Var} \left(\hat{\pmb{v}}_k \pmb{Y}_k \right)\\
\end{align}
\tag{8}
$$
可以看到，分母建立在“采集信号中的非 SSVEP 噪声与正余弦信号毫不相关”这一强假设基础上，而且在“各个试次中均存在”，这一点目前恕我不能苟同（~~没准等我测试完以后再次被打脸~~）。

争议暂且搁置，我们来总结一下 sc- 技术的本质，即引入正余弦模板，在空间维度扩增训练数据的信息容量，从而强化模型稳定性。重要的话我再重复一遍：只要对 $\bar{\pmb{X}}$ 进行有效改进（扩增、纯化等），就有希望提高 TRCA 空间滤波器的分类性能，未来各位将在其它 TRCA 改进算法中看到类似的趋势（~~求求各位大佬多发文章没准我还能水一篇综述~~）。

---

[sc-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/abfdfa