# SSVEP 空间滤波器的最小二乘统一框架
## Least-Square Unified Framework
***

[论文链接][refer]

恐怕只有 Wong Chi Man 以及 Wang Feng 老师团队才能完成这种数学复杂程度的工作，而且还是两次。诚如作者在引言中所述：“我们团队于 2020 年基于广义特征值问题（GEP）构建的统一框架，仅从数学角度把现有的 SSVEP 空间滤波算法进行了重构与整合，并没有从机器学习的角度阐释这些方法的原理与彼此间的联系”。我非常认同这一论断，同时这也是我设立一个专栏介绍本文框架的主要原因：个人认为，Wong Chi Man 在上一篇工作（基于 GEP 的统一框架）中，炫技的成分较多；而此文在数学公式的复杂度上虽然为之更甚，但该框架在机器学习可解释性的角度上明显有了长足的进步，值得仔细研读。

---
## LS 框架的数学描述
本文提出了一种基于最小二乘问题（Least-square, LS）的统一框架，严格来说是降秩回归问题（Reduced rank regression, RRR）。相比 GEP 问题，LS 问题在数学形式上具备更好的包容性，可以更方便地结合约束、正则化、非线性变换以及人工神经网络等数学技巧或模型；而 RRR 问题是 LS 问题的扩展，它能够在回归矩阵的约束下最小化 LS 误差，减少训练参数的数目。该研究提出的 RRR 框架可表述为如下形式：
$$
\hat{\pmb{W}}, \hat{\pmb{M}} = \underset{\pmb{W},\pmb{M}}{\arg\min} \left\| \pmb{E} \pmb{W} {\pmb{M}}^T - \pmb{T} \right\|_F^2, \ \ \ \ s.t. \ \ \pmb{M}^T \pmb{M} = \pmb{I}
\tag{1}
$$
其中，$\pmb{W}$ 表示空间滤波器；$\pmb{E}$ 一般表示类间 EEG 特征的组合，由拼接矩阵 $\pmb{L}_{\pmb{E}}$、特征提取矩阵（正交矩阵） $\pmb{P}_{\pmb{E}}$ 以及 EEG 数据矩阵 $\pmb{Z}$ 之间的矩阵乘法运算构成：
$$
\pmb{E} = \pmb{L}_{\pmb{E}} \pmb{P}_{\pmb{E}} \pmb{Z}
\tag{2}
$$
类似地可有类内特征组合矩阵 $\pmb{K}$，其中 $\pmb{L}_{\pmb{K}}$、$\pmb{P}_{\pmb{K}}$ 分别为相应的拼接矩阵与特征提取矩阵：
$$
\pmb{K} = \pmb{L}_{\pmb{K}} \pmb{P}_{\pmb{K}} \pmb{Z}
\tag{3}
$$
$\pmb{T}$ 是另一种包含了类间 EEG 特征（$\pmb{E}$）以及类内 EEG 特征（$\mathcal{S} (\pmb{K})$）的矩阵：
$$
\begin{align}
\notag \pmb{K} \ &\xlongequal{{\rm SVD}} \ \pmb{U}_{\pmb{K}} \pmb{\Sigma}_{\pmb{K}} {\pmb{V}_{\pmb{K}}}^T, \ \ \ \ \mathcal{S} (\pmb{K}) = \pmb{V}_{\pmb{K}} {\pmb{\Sigma}_{\pmb{K}}}^{-1} {\pmb{V}_{\pmb{K}}}^T\\
\notag \ \\
\notag \pmb{T} &= \pmb{E} \mathcal{S} (\pmb{K}) = \pmb{E} \pmb{V}_{\pmb{K}} {\pmb{\Sigma}_{\pmb{K}}}^{-1} {\pmb{V}_{\pmb{K}}}^T
\end{align}
\tag{4}
$$
由于 $\pmb{T}$、$\pmb{E}$ 与变量 $\pmb{W}$ 与 $\pmb{M}$ 均无关，因此在某种意义上可视为常数项。关于这些矩阵为什么要这样设计，此处暂不予详述，我们在之后结合一些经典算法案例加以分析。类似基于 SVD 分解构建的 $\mathcal{S} (\pmb{K})$，文中还有利用 QR 分解构建的正交投影矩阵：
$$
\pmb{A} \ \xlongequal{{\rm QR}} \ \pmb{Q}_{\pmb{A}} \pmb{R}_{\pmb{A}}, \ \ \ \ \mathcal{P} (\pmb{A}) = \pmb{Q}_{\pmb{A}} {\pmb{Q}_{\pmb{A}}}^T
\tag{5}
$$
接下来从数学的角度推导一下式（1）的求解方法，首先根据 $\pmb{M}$ 的约束条件改写一下式（1）：
$$
\underset{\pmb{W},\pmb{M}}{\arg\min} \left\| \pmb{E} \pmb{W} {\pmb{M}}^T - \pmb{T} \right\|_F^2 \ \xlongequal{\pmb{M}^T \pmb{M} = \pmb{I}} \ \underset{\pmb{W},\pmb{M}}{\arg\min} \left\| \pmb{E} \pmb{W} - \pmb{T} \pmb{M} \right\|_F^2
\tag{6}
$$
把 $\pmb{T} \pmb{M}$ 与 $\pmb{E} \pmb{W}$ 分别视为常量，即可通过交替迭代的最小二乘法求解：
$$
\hat{\pmb{W}} = {\left( {\pmb{E}}^T \pmb{E} \right)}^{-1} {\pmb{E}}^T \left( \pmb{T} \pmb{M} \right)
\tag{7}
$$
$$
\hat{\pmb{M}} = {\left( {\pmb{T}}^T \pmb{T} \right)}^{-1} {\pmb{T}}^T \left( \pmb{E} \pmb{W} \right)
\tag{8}
$$
具体迭代流程如下。首先初始化变量 $\hat{\pmb{M}}_0$：
$$
\left\| \pmb{E} \pmb{W} {\pmb{M}}^T - \pmb{T} \right\|_F^2 \approx \left\| \pmb{T} \pmb{M} {\pmb{M}}^T - \pmb{T} \right\|_F^2, \ \ \ \ \hat{\pmb{M}}_0 = \pmb{V}_{\pmb{T}}
$$
按式（7）初始化变量 $\hat{\pmb{W}}_0$，接下来按照先式（8）后式（7）的顺序进行交替迭代。假设 $t$ 轮后迭代终止，终止条件为：
$$
\begin{cases}
\left\| \hat{\pmb{M}}_t - \hat{\pmb{M}}_{t-1} \right\|_1 < \theta_{\pmb{M}}\\
\ \\
\left\| \hat{\pmb{W}}_t - \hat{\pmb{W}}_{t-1} \right\|_1 < \theta_{\pmb{W}}
\end{cases}
\tag{9}
$$
其中 $\theta_{\pmb{M}}$ 与 $\theta_{\pmb{W}}$ 为预设的边界条件，$\|*\|_1$ 表示 $*$ 中全体元素的绝对值（或模）之和。最终获取该模型对应的空间滤波器（$\hat{\pmb{W}}_t$）。

---
## LS 框架在几种典型算法中的应用
按照原文的思路，LS 框架通常是由 GEP 问题转化而来的。但是我实在是不想把那个玩意拉出来再单独介绍一遍，在此仅给出 LS 框架下的各算法目标函数以及相应的推导证明。**为了尽量尊重原著**（~~改写格式实在是太复杂了~~）（~~反正也不会有人真的用这个格式去写代码~~），本专栏接下来的向量表示都是**列向量**。

### 标准 CCA
将单试次测试样本记为 $\mathcal{\pmb{X}} \in \mathbb{R}^{N_p \times N_c}$，$N_c$、$N_p$ 分别表示导联数与采样点数；第 $k$ 类刺激对应的人工正余弦模板记为 $\pmb{Y}_k \in \mathbb{R}^{N_p \times 2 N_h}$，其中 $N_h$ 表示谐波数。以 EEG 侧空间滤波器为例（记为 $\hat{\pmb{W}}_k \in \mathbb{R}^{N_c \times N_k}$，$N_k$ 表示子空间维度）：
$$
\hat{\pmb{W}}_k = \underset{\pmb{W}_k}{\arg\min} \left\| \underbrace{\mathcal{P}(\pmb{Y}_k) \mathcal{\pmb{X}}}_{\pmb{E}} \ \ \pmb{W}_k \ \ \underbrace{{\left( \mathcal{\pmb{X}} \pmb{W}_k \right)}^T}_{\pmb{M}^T} \ \ - \ \ \underbrace{\mathcal{P}(\pmb{Y}_k) \mathcal{\pmb{X}} \mathcal{S}(\mathcal{\pmb{X}})}_{\pmb{T}} \right\|_F^2
\tag{10}
$$
首先来验证一下 $\pmb{M}^T \pmb{M} = \pmb{I}$：
$$
\pmb{M}^T \pmb{M} = {\left( \mathcal{\pmb{X}} \pmb{W}_k \right)}^T \mathcal{\pmb{X}} \pmb{W}_k = {\pmb{W}_k}^T {\mathcal{\pmb{X}}}^T \mathcal{\pmb{X}} \pmb{W}_k = \pmb{I}_{N_k}
$$
上式是成立的，详情可参考 CCA 专栏，简单来说 ${\pmb{W}_k}^T {\mathcal{\pmb{X}}}^T \mathcal{\pmb{X}} \pmb{W}_k$ 表示滤波后测试信号的方差（能量），当 CCA 的目标函数由线性相关系数形式向最优化问题形式转换时，需要满足的边界约束条件之一就是该能量为单位数值（单位阵）。接下来把式（10）展开：
$$

$$



[refer]: https://ieeexplore.ieee.org/document/10587150/