# 个体迁移学习 CCA
## Subject Transfer based CCA, stCCA
***
### [论文链接][stCCA]

Wong 于 2020 年发表的 stCCA 算法，开启了他在 SSVEP 迁移算法领域研究的“传奇生涯”。至本文落笔之日，他先后提出了 Subject Transfer based CCA（stCCA）、Transfer Learning CCA （tlCCA）、Different Phase Multi-channel Adaptive Fourier Decomposition（DP-MAFD）三篇面向 SSVEP 迁移学习主题的算法，将迁移训练的可行域范围逐步扩大。不出意外的话我能在今年暑假结束之前把这三部曲都完成（~~主要是最后一个难，真的难~~）。

本期专栏我们来看 stCCA。stCCA 面向的应用场景需求是：已有源域受试者 $n$ 的完整数据 $\pmb{X}_k^{(n)} \in \mathbb{R}^{N_c^{(n)} \times N_p}$（全类别 $N_e$，训练样本 $N_t^{(n)}$ 相对充分，导联、采样点数分别为 $N_c^{(n)}$、$N_p$），已有目标域受试者 $\tau$ 的训练数据 $\pmb{X}_k^{(\tau)} \in \mathbb{R}^{N_c^{(\tau)} \times N_p}$（类别不全 $N_e^{(\tau)} \leqslant N_e$，样本 $N_t^{(\tau)}$ 不充分）。stCCA 的主要目标是：通过将源域数据 $\pmb{X}_k^{(n)}$ 迁移至目标域 $\pmb{X}_k^{(\tau)}$，补充目标域数据的建模信息。具体来说，stCCA 分为三个主要阶段：Intra-Subject Spatial Filter、Inter-Subject SSVEP Template 以及 Pattern Matching。

![stCCA训练目标](stCCA-1.png)

首先是受试者个体空间滤波器。Wong 提出了一种前提假设：对于一个固定的受试者而言，面向不同频率刺激数据构建的空间滤波器是具有较高相似度，即可以共享的。因此 Wong 利用 msCCA 方法计算目标域受试者 $\tau$ 的类别非特异、个体特异（class-nonspecific, subject-specific）空间滤波器（transferred spatial filters）$\hat{\pmb{u}}_k^{(\tau)} \in \mathbb{R}^{1 \times N_c^{(\tau)}}$、$\hat{\pmb{v}}_k^{(\tau)} \in \mathbb{R}^{1 \times 2N_h}$，以实现基于参数的迁移学习。：
$$
    \hat{\pmb{u}}_k^{(\tau)}, \hat{\pmb{v}}_k^{(\tau)} = \underset{\pmb{u}_k^{(\tau)},\pmb{v}_k^{(\tau)}} \argmax \dfrac{\pmb{u}_k^{(\tau)} \left( \sum_{k=1}^{N_e^{(\tau)}} \bar{\pmb{X}}_k^{(\tau)} {\pmb{Y}_k}^T \right) {\pmb{v}_k^{(\tau)}}^T} {\sqrt{\pmb{u}_k^{(\tau)} \left( \sum_{k=1}^{N_e^{(\tau)}} \bar{\pmb{X}}_k^{(\tau)} {\bar{\pmb{X}}_k^{(\tau)}}^T \right) {\pmb{u}_k^{(\tau)}}^T} \sqrt{\pmb{v}_k^{(\tau)} \left( \sum_{k=1}^{N_e^{(\tau)}} \pmb{Y}_k {\pmb{Y}_k}^T \right) {\pmb{v}_k^{(\tau)}}^T}}
    \tag{1}
$$

[stCCA]: https://ieeexplore.ieee.org/document/9177172/