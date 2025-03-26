# 个体内/间相关性最大化
## Inter- and Intra-Subject Maximal Correlation, IISMC
***

### [论文链接][IISMC]

这篇 2021 年发表的文章是我在学习另外一篇 2023 年文章的时候，从引言部分的引用文献中找到的。从结果来看，本文的工作还算扎实，但是孤陋寡闻的我就是没有关注到。单就外部原因而言，可能是同期的 TDCA 算法性能过于耀眼，抑或是该算法的名字（缩写）起得不好：大家都是某某 CA，什么 CCA、TRCA、TDCA 的，你来一个 IISMC，这谁记得住呀。

IISMC 的主要思路来源于 TRCA 与 CORCA，滤波目的在于提取个体特异性信息与任务相关的通用性信息，在研究思路上有点类似 stCCA，即 subject-specific 与 task-invariant（或者 mutually-invariant）。

![IISMC流程图](/数据增强%20&%20迁移学习算法/IISMC-1.png)

简单起见，我们先从两个受试者 $S_1$、$S_2$ 开始说明。IISMC 的**第一步**是 **Inter-Subject Maximal Correlation**，即**受试者个体间相关性最大化**，本质上是一种迁移学习，其主要目的是求解一个 $S_1$、$S_2$ 在进行第 $k$ 类任务时通用的空间滤波器 $\pmb{u}_k^{(1),(2)} \in \mathbb{R}^{1 \times N_c}$。记受试者 $S_1$ 的第 $k$ 类、第 $i$ 个样本为 $\pmb{X}_k^{i,(1)} \in \mathbb{R}^{N_c \times N_p}$，样本类别总数为 $N_e$ 且全个体通用，样本容量为 $N_{t,k}^{(1)}$。考虑到书写简洁性与数据集构成的一般情况，我们在讨论生成式模型时一般不考虑类间样本不平衡的情况，因此下文中将 $N_{t,k}^{(1)}$ 简记为 $N_t^{(1)}$。

接下来计算跨个体交叉协方差（inter-subject cross-covariance）矩阵 $\pmb{C}_{12}$、$\pmb{C}_{21}$ 与个体内自协方差（intra-subject auto-covariance）矩阵 $\pmb{C}_{11}$、$\pmb{C}_{22}$：（注意，原文公式推导有误，$\pmb{R}_{12}$ 是数字，无法再经过 $\pmb{w}$ 的滤波）
$$
\begin{align}
\notag \pmb{C}_{12} &= \dfrac{1}{N_t^{(1)} N_t^{(2)}} \sum_{i=1}^{N_t^{(1)}} \sum_{j=1}^{N_t^{(2)}} \pmb{X}_k^{i,(1)} {\pmb{X}_k^{j,(2)}}^T, \ \ \pmb{C}_{21} = {\pmb{C}_{12}}^T\\
\notag \ \\ 
\notag \pmb{C}_{11} &= \dfrac{1}{N_t^{(1)}} \sum_{i=1}^{N_t^{(1)}} \pmb{X}_k^{i,(1)} {\pmb{X}_k^{i,(1)}}^T, \ \ \pmb{C}_{22} = \dfrac{1}{N_t^{(2)}} \sum_{j=1}^{N_t^{(2)}} \pmb{X}_k^{j,(2)} {\pmb{X}_k^{j,(2)}}^T
\end{align}
\tag{1}
$$
在此基础上，依照 Pearson 相关系数的计算形式，IISMC 在第一步设计的跨个体空间滤波器目标函数为：
$$
\hat{\pmb{u}}_k^{(1),(2)} = \underset{\pmb{u}_k^{(1),(2)}} \argmax \dfrac{1}{2} \dfrac{\pmb{u}_k^{(1),(2)} \left( \pmb{C}_{12} + \pmb{C}_{21} \right) {\pmb{u}_k^{(1),(2)}}^T} {\sqrt{\pmb{u}_k^{(1),(2)} \pmb{C}_{11} {\pmb{u}_k^{(1),(2)}}^T} \sqrt{\pmb{u}_k^{(1),(2)} \pmb{C}_{22} {\pmb{u}_k^{(1),(2)}}^T}}
\tag{2}
$$
在原始数据标准化的前提下，我们可以近似认为 $\pmb{u}_k^{(1),(2)} \pmb{C}_{11} {\pmb{u}_k^{(1),(2)}}^T$ 与 $\pmb{u}_k^{(1),(2)} \pmb{C}_{22} {\pmb{u}_k^{(1),(2)}}^T$ 是相等的，即滤波后两名受试者的信号能量总体水平大致相当，即有：
$$
\hat{\pmb{u}}_k^{(1),(2)} = \underset{\pmb{u}_k^{(1),(2)}} \argmax \dfrac{\pmb{u}_k^{(1),(2)} \left( \pmb{C}_{12} + \pmb{C}_{21} \right) {\pmb{u}_k^{(1),(2)}}^T} {\pmb{u}_k^{(1),(2)} \left( \pmb{C}_{11} + \pmb{C}_{22} \right) {\pmb{u}_k^{(1),(2)}}^T}
\tag{3}
$$
与之前讨论过的广义 Rayleigh Quotient 结论一致，取 $\left( \pmb{C}_{11} + \pmb{C}_{22} \right)^{-1} \left( \pmb{C}_{12} + \pmb{C}_{21} \right)$ 的最大特征值对应的特征向量作为最优解 $\hat{\pmb{u}}_k$。

**第二步**是 **Intra-Subject Maximal Correlation**，即**受试者个体内相关性最大化**。以受试者 $\pmb{S}_1$ 为例，第二步最终将求解一个使得 $\pmb{S}_1$ 的第 $k$ 类数据中，任意两个试次数据相关性最大的空间滤波器 $\pmb{v}_k^{(1)} \in \mathbb{R}^{1 \times N_c}$。有浏览过 CORCA 那篇论文的朋友应该发现了，其实这一步（包括 CORCA 那篇论文）与 TRCA 的目标函数在数理上是一样的，只不过原文按照 CORCA 的形式重构了一下训练过程（在 CORCA 论文里，则是用余弦相似度替换了 Pearson 相关系数，当原始数据满足某些分布条件时，余弦相似度与 Pearson 相关系数是等价度量）。这里我不想用原文那拙劣的表达式，就稍微改写了一下。当然，我实际编程时参照的表达式还是 TRCA 章节写的那个版本：
$$
\hat{\pmb{v}}_k^{(1)} = \underset{\pmb{v}_k^{(1)}} \argmax \dfrac{\pmb{v}_k^{(1)} \pmb{S}_k^{(1)} {\pmb{v}_k^{(1)}}^T} {\pmb{v}_k^{(1)} \pmb{Q}_k^{(1)} {\pmb{v}_k^{(1)}}^T}, \ \ \
\begin{cases}
\pmb{S}_k^{(1)} = \dfrac{4}{N_t (N_t - 1)} \sum_{j=1,j \ne i}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^{i,(1)} {\pmb{X}_k^{j,(1)}}^T\\
\ \\
\pmb{Q}_k^{(1)} = \dfrac{2}{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^{i,(1)} {\pmb{X}_k^{i,(1)}}^T
\end{cases}
\tag{4}
$$
同样地，取 ${\pmb{Q}_k^{(1)}}^{-1} \pmb{S}_k^{(1)}$ 的最大特征值对应的特征向量作为最优解 $\hat{\pmb{v}}_k^{(1)}$。

**第三步**是**Combined Inter- and Intra-Subject Maximal Correlation**，即所谓 IISMC，本质上是模板匹配与系数融合。记源域受试者 $s$ 的第 $k$ 类、第 $i$ 个样本数据为 $\pmb{X}_k^{i,(s)}$，目标域受试者的上标为 $(\tau)$，源域受试者总数为 $N_s$。对于目标域未知类别的测试数据 $\pmb{\mathcal{X}} \in \mathbb{R}^{N_c \times N_p}$，IISMC 给出了一个四种相关系数的融合方案：
$$
\rho_k = \begin{bmatrix}
\rho_{k,1}\\ \ \\ \rho_{k,2}\\ \ \\ \rho_{k,3} \\ \ \\ \rho_{k,4}\\
\end{bmatrix} = 
\begin{bmatrix}
{\rm corr} \left( \hat{\pmb{v}}_k^{(\tau)} \pmb{\mathcal{X}}, \ \hat{\pmb{v}}_k^{(\tau)} \bar{\pmb{X}}_k^{(\tau)} \right)\\
\ \\
\dfrac{1}{N_s} \sum_{i=1}^{N_s} {\rm corr} \left( \hat{\pmb{u}}_k^{(\tau),(i)} \pmb{\mathcal{X}}, \ \hat{\pmb{u}}_k^{(\tau),(i)} \bar{\pmb{X}}_k^{(\tau)} \right)\\
\ \\
\dfrac{1}{N_s} \sum_{i=1}^{N_s} {\rm corr} \left( \hat{\pmb{v}}_k^{(\tau)} \pmb{\mathcal{X}}, \ \hat{\pmb{v}}_k^{(\tau)} \bar{\pmb{X}}_k^{(i)} \right)\\
\ \\
\dfrac{1}{N_s} \sum_{i=1}^{N_s} {\rm corr} \left( \hat{\pmb{u}}_k^{(\tau),(i)} \pmb{\mathcal{X}}, \ \hat{\pmb{u}}_k^{(\tau),(i)} \bar{\pmb{X}}_k^{(i)} \right)\\
\end{bmatrix}, \ \ \ \hat{k} = \underset{k} \argmax \sum_{n=1}^{4} {\rm sign} (\rho_{k,n}) \rho_{k,n}^2
\tag{5}
$$
我们挨个看看这四种系数：（1）$\rho_{k,1}$ 其实就是 TRCA 的决策系数；（2）$\rho_{k,2}$ 在 $\rho_{k,1}$ 的基础上替换了空间滤波器，由目标域数据限定的 $\hat{\pmb{v}}_k^{(\tau)}$ 换成了目标域与源域数据共同训练的 $\hat{\pmb{u}}_k^{(\tau),(i)}$，且各源域受试者的迁移权重是相同的（这一点或许存在些微的改进空间？）；（3）$\rho_{k,3}$ 在 $\rho_{k,1}$ 的基础上替换了匹配模板，由目标域模板 $\bar{\pmb{X}}_k^{(\tau)}$ 换成了源域模板 $\bar{\pmb{X}}_k^{(i)}$，且迁移权重是均等的；（4）$\rho_{k,4}$ 则是用源域模型（滤波器与模板）换掉了目标域模型。

我很好奇的一点是，这四类系数到底哪一类的决策能力最高？可惜文章没有给出相关的描述。而且作者使用的交叉验证与迭代方式非常“巧妙”，是蒙特卡洛交叉验证法，即“有放回随机抽样”，这也就意味着我几乎不可能在没有源码的情况下完全复现出他的结果。据我所知，这一类算法的最大局限性在于源域数据的质量问题。文中按照数据集的先验知识（Experienced 受试者与 Naive 受试者），对比了源域数据在 Experienced 受试者群体和完整群体中选取的算法测试情况（IISMC-EG 与 IISMC），IISMC-EG 的结果是要高于 IISMC 的。而这已经是作者“选取”了 8 名源域受试者之后的结果了。如果我们盲目地将 34 位受试者（Benchmark 数据集共 35 名受试者）的数据统统作为源域数据，可能迁移效果会非常难看。这也给我们未来研究迁移学习算法指出了一个值得探索的方向，即如何有效筛选适合迁移的数据。

![IISMC测试结果](/数据增强%20&%20迁移学习算法/IISMC-2.png)

[IISMC]: https://ieeexplore.ieee.org/document/9350285/