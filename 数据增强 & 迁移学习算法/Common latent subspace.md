# 基于公共潜在子空间的时空域适应方法
## Spatiotemporal domain adaptation based on common latent subspace, CLS-SDA
---

这是我的一位同行大佬设计的跨域机器学习方法，在 P300 数据集上表现出了不俗的性能。鄙人不才，现对其中的部分细节进行修改调整，尝试应用在 SSVEP 数据上。

一般认为，脑电信号同时具有时-空域复合信息，且二者相互耦合。然而传统的跨域数据对齐方法往往仅关注某一个维度的统计量（如代表空域信息的协方差矩阵），所以我的同僚认为，想要充分描述具有丰富信息的脑电数据，需要首先解耦脑电信号的时、空域信息，避免在跨域对齐过程中产生互相干扰，之后分别从时域波形与空域传播两个方面进行模型迁移。本文算法的第一步，就是构建脑电特征响应的传播模型。

按照惯例，首先定义本文所需的基本变量表示方法。目标受试者的代号（上标）为 $\tau$，源域受试者的索引为 $s$，共有 $N_s$ 名源域受试者（$s$ 的取值范围为 $1$ 至 $N_s$)；目标受试者第 $k$ 类、第 $i$ 试次的多导联数据记为 $\pmb{X}_{k}^{i,(\tau)} \in \mathbb{R}^{N_c^{(\tau)} \times N_p}$，导联数为 $N_c^{(\tau)}$，采样点数为 $N_p$；$\pmb{X}_{k}^{i,(\tau)}$ 的试次平均记为 $\bar{\pmb{X}}_k^{(\tau)}$，样本数目为 $N_t^{(\tau)}$（简单起见，此处不考虑受试者个体内样本不均衡的情况）。接下来进入算法的具体步骤。

---
### 脑电特征响应的传播模型

根据 2014 年一篇发表在 [Neuroimage 期刊上的研究][refer1]表明，传统意义上的“空间滤波器”在数值上并不等价于信源到接受端（采集导联）的传播模式。例如对于一个双极采集系统，两个导联的信息分布分别为 $\pmb{s} + \pmb{n}$、$\pmb{n}$，其中 $\pmb{s}$ 表示来自脑内信源的有效信息（序列），$\pmb{n}$ 表示任务无关噪声。此时空间滤波器的系数应当为 $1$、$-1$，从数值上看两个导联的数据信息占比是相等的，但显然只有第一个接受端才包含有效信息。在本专栏后文中还将出现多次来源于这一研究的结论，我将在另一篇专栏中加以详细介绍与说明。简而言之，空间滤波器代表了信源信息的“**反向传播模式**”，信息的“**正向传播模式**”与空间滤波器之间存在一定的数学转换关系，二者结合可视为**脑电特征响应的传播模型**。

以最常用的 TRCA 模型构建**反向传播模型** $\pmb{w}_k^{(\tau)} \in \mathbb{R}^{N_k \times N_c^{(\tau)}}$，其中 $N_k$ 表示空间滤波器的子空间个数（在 SSVEP 信号解码应用中通常取 1），该模型的简化公式与等价广义特征值方程为（公式推导的详情请移步 TRCA 章节）：
$$
\begin{align}
\notag \hat{\pmb{w}}_k^{(\tau)} = \underset{\pmb{w}_k^{(\tau)}} \argmax \dfrac{\pmb{w}_k^{(\tau)} \bar{\pmb{X}}_k^{(\tau)} {\bar{\pmb{X}}_k^{(\tau)}}^T {\pmb{w}_k^{(\tau)}}^T} {\pmb{w}_k^{(\tau)} \left( \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} \pmb{X}_k^{i, (\tau)} {\pmb{X}_k^{i, (\tau)}}^T \right) {\pmb{w}_k^{(\tau)}}^T}\\
\notag \ \\
\notag \left( \bar{\pmb{X}}_k^{(\tau)} {\bar{\pmb{X}}_k^{(\tau)}}^T \right) {\pmb{w}_k^{(\tau)}}^T = \lambda \pmb{\Sigma}_{\pmb{X},k}^{(\tau)} {\pmb{w}_k^{(\tau)}}^T, \ \ \ \ \pmb{\Sigma}_{\pmb{X},k}^{(\tau)} =  \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t} \pmb{X}_k^{i, (\tau)} {\pmb{X}_k^{i, (\tau)}}^T
\end{align}
\tag{1}
$$
考虑到在需要使用数据域适应方法的场景下，目标域数据样本可能出现不足（即 $\bar{\pmb{X}}_k^{(\tau)}$ 对样本中心的估计不准确），因此引入源域受试者 $s$ 的数据进行联合求解，对（1）式进行调整：
$$
\begin{align}
\notag \hat{\pmb{w}}_k^{(\tau, s)} = \underset{\pmb{w}_k^{(\tau, s)}} \argmax \dfrac{\pmb{w}_k^{(\tau, s)} \left[ (1 - \theta) \bar{\pmb{X}}_k^{(\tau)} {\bar{\pmb{X}}_k^{(\tau)}}^T + \theta \bar{\pmb{X}}_k^{(s)} {\bar{\pmb{X}}_k^{(s)}}^T \right] {\pmb{w}_k^{(\tau)}}^T} {\pmb{w}_k^{(\tau)} \left[ (1 - \theta) \pmb{\Sigma}_{\pmb{X},k}^{(\tau)} + \theta \pmb{\Sigma}_{\pmb{X},k}^{(s)} \right] {\pmb{w}_k^{(\tau)}}^T}\\
\notag \ \\
\notag \left[ (1 - \theta) \bar{\pmb{X}}_k^{(\tau)} {\bar{\pmb{X}}_k^{(\tau)}}^T + \theta \bar{\pmb{X}}_k^{(s)} {\bar{\pmb{X}}_k^{(s)}}^T \right] {\pmb{w}_k^{(\tau, s)}}^T = \lambda \left[ (1 - \theta) \pmb{\Sigma}_k^{(\tau)} + \theta \pmb{\Sigma}_k^{(s)} \right] {\pmb{w}_k^{(\tau, s)}}^T
\end{align}
\tag{2}
$$
混合系数 $\theta$ 为平衡信息来源比例的超参数，在本算法的原始工作中，$\theta$ 被经验性地设置为 0.2。需要强调的是，该方法默认源域数据 $\pmb{X}_k^{i, (s)}$ 与目标域数据 $\pmb{X}_k^{i, (\tau)}$ 来自同一批数据集，即二者满足导联（空间）维度相同（$N_c^{(\tau)} = N_c^{(s)}$）、采集设备相同、参考电极相同等条件。考虑到需要数据迁移的场景通常是源域数据的采集条件优于目标域（源域数据拥有更完整的导联组：$C^{(\tau)} \subset C^{(s)}$），因此倘若二者除导联数目外其余采集条件均相同，可以考虑在式（2）中强行将 $\pmb{X}_k^{i, (s)}$ 的导联维度与 $\pmb{X}_k^{i, (\tau)}$ 统一之后（俗称删导联）再进行计算。但如果实际情况就是那么惨淡，源域数据的导联数反而更少，则不建议使用式（2）所示的联合求解方法，甚至应该重新考虑使用源域数据进行迁移的可行性与实用性。

在获得空间滤波器 $\hat{\pmb{w}}_k^{(\tau, s)} \in \mathbb{R}^{N_k \times N_c^{\tau}}$（或 $\hat{\pmb{w}}_k^{(\tau)}$）之后，我们可以得到目标域脑电特征响应的单试次源活动信息 $\pmb{s}_k^{i, (\tau)} \in \mathbb{R}^{N_k \times N_p}$ ：
$$
\pmb{s}_k^{i, (\tau)} = \hat{\pmb{w}}_k^{(\tau, s)} \pmb{X}_k^{i,(\tau)} \ \ \ \ {\rm or} \ \ \ \ \hat{\pmb{w}}_k^{(\tau)} \pmb{X}_k^{i,(\tau)}
\tag{3}
$$
此处 $N_k$ 是数学意义上的子空间维度，与实际脑电数据的空间维度（导联数目 $N_c^{(\tau)}$）无关，因此不包含物理意义上的空间信息。在这种 $N_k$ 维度的统一数学空间进行特征对齐，能够避免很多对齐算法物理可解释性的困难。接下来对源活动 $\pmb{s}_k^{i, (\tau)}$ 的**正向传播过程**进行模拟求解，通过计算传播（混叠）矩阵（或向量）$\pmb{A}_k^{(\tau)} \in \mathbb{R}^{N_c^{(\tau)} \times N_k}$，使得空间传播后的多通道源活动与原始多导联信号之误差的欧式空间范数最小：
$$
\hat{\pmb{A}}_k^{(\tau)} = \underset{\pmb{A}_k^{(\tau)}} \argmin \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} \left\| \pmb{A}_k^{(\tau)} \pmb{s}_k^{i,(\tau)} - \pmb{X}_k^{i,(\tau)} \right\|_F^2
\tag{4}
$$
此处容易产生的一个误解是：传播模式 $\hat{\pmb{A}}_k^{(\tau)}$ 看上去很像是空间滤波器 $\hat{\pmb{w}}_k^{(\tau, s)}$ 的 Penrose 伪逆 ${\hat{\pmb{w}}_k^{(\tau, s)}}^+$。当然实际结果并非如此，该结论的说明请参考后文中的 **讨论 1**。式（4）的解析解如下所示，其证明过程请参考后文中的 **推导 1**：
$$
\hat{\pmb{A}}_k^{(\tau)} = \left( \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} \pmb{X}_{k}^{i,(\tau)} {\pmb{X}_{k}^{i,(\tau)}}^T \right) {\hat{\pmb{w}}_k^{(\tau, s)}}^T \left( \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} \pmb{s}_k^{i,(\tau)} {\pmb{s}_k^{i,(\tau)}}^T \right)^{-1} = \pmb{\Sigma}_{\pmb{X},k}^{(\tau)} {\hat{\pmb{w}}_k^{(\tau, s)}}^T {\pmb{\Sigma}_{\pmb{s}, k}^{(\tau)}}^{-1} \ \ \ \ {\rm or} \ \ \ \ \pmb{\Sigma}_{\pmb{X},k}^{(\tau)} {\hat{\pmb{w}}_k^{(\tau)}}^T {\pmb{\Sigma}_{\pmb{s}, k}^{(\tau)}}^{-1}
\tag{5}
$$

---
### 公共潜在子空间内的空域对齐

在上一小节中，我们获得了数学意义上的、子空间内的目标域脑电源活动估计 $\pmb{s}_k^{i,(\tau)} \in \mathbb{R}^{N_k \times N_p}$，该空间通常称为潜在子空间（latent subspace）。在潜在子空间内，$\pmb{s}_k^{i,(\tau)}$ 摒弃了空间维度信息，仅含有时间维度信息，因此很适合作为个体信息的“中转媒介”——来自不同个体的数据在这个 $N_k$ 维度的潜在子空间内进行对齐时，不会再损失空间维度信息（~~毕竟没有这个维度了~~）。同时源域数据 $\pmb{X}_k^{i,(s)}$ 投影到潜在子空间并与目标域源活动对齐后，能够通过目标域的传播模式 $\hat{\pmb{A}}_k^{(\tau)}$ 拟合出具有相似空间分布的多导联信号。这种基于正向传播模型设计的数据扩增方法，据说相比于最小二乘对齐法（如 2020 年发表在 JNE 期刊上的 [LST 对齐方法]、2023 年发表在 TNSRE 期刊的 [sd-LST 算法]、[基于 TRCA 的迁移学习算法]等）有更好的鲁棒性，能够保留更多有效信息。同时在跨设备迁移问题上（主要面向 $C^{(\tau)} \subsetneqq C^{(s)}$ 场景）具有更深的应用潜力。

选定待迁移对象（对应源域个体的原始数据） $\pmb{X}_k^{i,(s)}$ 与数据对齐的标靶（目标域源活动的试次平均） $\bar{\pmb{s}}_k^{(\tau)} \in \mathbb{R}^{N_k \times N_p}$：
$$
\bar{\pmb{s}}_k^{(\tau)} = \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} \pmb{s}_k^{i, (\tau)}
\tag{6}
$$
对前者向 $N_k$ 维的公共潜在子空间投影，使得该空间内二者的欧式距离最小。关于这一步骤所需的投影矩阵（向量） $\pmb{U}_k^{(\tau, s)} \in \mathbb{R}^{N_k \times N_c^{(s)}}$，现有两种计算方法。首先介绍相对简单的一种：
$$
\hat{\pmb{U}}_k^{(\tau, s)} = \underset{\pmb{U}_k^{(\tau, s)}}{\argmin} \dfrac{1}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} \left\| \pmb{U}_k^{(\tau, s)} \pmb{X}_k^{i, (s)} - \bar{\pmb{s}}_k^{(\tau)} \right\|_F^2
\tag{7}
$$
式（7）的解析解如下所示。其证明过程请参考后文中的 **推导 2**：
$$
\hat{\pmb{U}}_k^{(\tau, s)} = \left( \sum_{i=1}^{N_t^{(s)}} \bar{\pmb{s}}_k^{(\tau)} {\pmb{X}_k^{i,(s)}}^T \right) \left( \sum_{i=1}^{N_t^{(s)}} \pmb{X}_k^{i,(s)} {\pmb{X}_k^{i,(s)}}^T \right)^{-1} = \bar{\pmb{s}}_k^{(\tau)} {\bar{\pmb{X}}_k^{(s)}}^T {\pmb{\Sigma}_{\pmb{X}, k}^{(s)}}^{-1}
\tag{8}
$$
根据大佬的测试经验，在部分 P300 数据集上，上述求解方法获取的投影矩阵效果有限。这或许是由于 P300 特征信号的试次间变异性较大导致的。根据式（7）的设计，并不能保证源域数据的每个试次 $\pmb{X}_k^{i, (s)}$ 经过投影后都能与目标域源活动 $\bar{\pmb{s}}_k^{(\tau)}$ 较好地对齐。因此引申出了第二种相对繁琐一些的求解方法。首先对源域个体 $s$ 的每个样本 $\pmb{X}_k^{i, (s)}$ 单独求解一个投影矩阵（向量）$\hat{\pmb{U}}_k^{i, (\tau, s)}$（证明过程见 **推导 2**），并对其进行标准化处理：
$$
\hat{\pmb{U}}_k^{i, (\tau, s)} = \underset{\pmb{U}_k^{i, (\tau, s)}}{\argmin} \left\| \pmb{U}_k^{i, (\tau, s)} \pmb{X}_k^{i, (s)} - \bar{\pmb{s}}_k^{(\tau)} \right\|_F^2 = \bar{\pmb{s}}_k^{(\tau)} {\pmb{X}_k^{i, (s)}}^T \left( \pmb{X}_k^{i, (s)} {\pmb{X}_k^{i, (s)}}^T \right)^{-1}, \ \ \ \ \left\| \hat{\pmb{U}}_k^{i, (\tau, s)} \right\| = 1
\tag{9}
$$
接下来基于余弦相似度平方最大化准则来计算这批投影矩阵（向量）$\hat{\pmb{U}}_k^{i, (\tau, s)}$ 的[原型滤波器]（Prototype spatial filter, PSF）$\hat{\pmb{U}}_k^{(\tau, s)}$：
$$
\hat{\pmb{U}}_k^{(\tau, s)} = \underset{\pmb{U}_k^{(\tau, s)}}{\argmax} \sum_{i=1}^{N_t^{(s)}} \left( \dfrac{\pmb{U}_k^{(\tau, s)} {\hat{\pmb{U}}_k^{i, (\tau, s)}}^T} {\left\| \pmb{U}_k^{(\tau, s)} \right\| \times \left\| \hat{\pmb{U}}_k^{i, (\tau, s)} \right\|} \right)^2 = \underset{\pmb{U}_k^{(\tau, s)}}{\argmax} \dfrac{\pmb{U}_k^{(\tau, s)} \left( \sum_{i=1}^{N_t^{(s)}} {\hat{\pmb{U}}_k^{i, (\tau, s)}}^T \hat{\pmb{U}}_k^{i, (\tau, s)} \right) {\pmb{U}_k^{(\tau, s)}}^T} {\pmb{U}_k^{(\tau, s)} {\pmb{U}_k^{(\tau, s)}}^T}
\tag{10}
$$
从定义来看，余弦相似度是面向向量的数学概念，而 $\hat{\pmb{U}}_k^{i, (\tau, s)}$ 可能具有多个维度（即 $N_k \ne 1$），因此最好基于 $\hat{\pmb{U}}_k^{i, (\tau, s)}$ 的每个维度分别进行上述步骤后，将各步骤结果拼接成 $\hat{\pmb{U}}_k^{(\tau, s)}$。式（10）是典型的 Rayleigh Quotient 问题，对方阵 $\sum_{i=1}^{N_t^{(s)}} {\hat{\pmb{U}}_k^{i, (\tau, s)}}^T \hat{\pmb{U}}_k^{i, (\tau, s)}$ 进行特征值分解、取最大特征值对应的特征向量即可获得式（10）的解析解。

需要注意的是，投影矩阵 $\hat{\pmb{U}}_k^{(\tau, s)}$ 并非总是必须的。根据本小节首部的说明可知，$\hat{\pmb{U}}_k^{(\tau, s)}$ 在面对源域、目标域数据集空间维度不匹配的场景时具有较好的适应能力，但是假如二者来源于同一数据集（$C^{(\tau)} = C^{(s)}$），且使用式（2）计算了面向公共潜在子空间的空间滤波器 $\hat{\pmb{w}}_k^{(\tau, s)}$，则仅需对源域数据 $\pmb{X}_k^{i, (s)}$ 应用 $\hat{\pmb{w}}_k^{(\tau, s)}$ 进行降维，即可将其投影至公共潜在子空间中。最终获得空域对齐后、源域的源活动信号 $\pmb{s}_k^{i, (s)} \in \mathbb{R}^{N_k \times N_p}$：
$$
\pmb{s}_k^{i, (s)} = \hat{\pmb{U}}_k^{(\tau, s)} \pmb{X}_k^{i, (s)} \ \ \ \ {\rm or} \ \ \ \ \hat{\pmb{w}}_k^{(\tau, s)} \pmb{X}_k^{i, (s)}
\tag{11}
$$

---
### 公共潜在子空间内的时域对齐

在前述步骤中，我们通过空间滤波器 $\hat{\pmb{w}}_k^{(\tau, s)}$（或 $\hat{\pmb{w}}_k^{(\tau)}$）求得公共潜在子空间内的目标域响应 $\pmb{s}_k^{i,(\tau)}$，通过空间投影矩阵（向量） $\hat{\pmb{U}}_k^{(\tau, s)}$（或 $\hat{\pmb{w}}_k^{(\tau, s)}$ ）求得源域响应 $\pmb{s}_k^{i,(s)}$。二者不再含有与实际采集设备电极位置有关的空间维度信息，在此基础上能够更好地处理特征信号的时域波形信息。与空间对齐类似地，我们需要在公共潜在子空间内求解一个时域投影矩阵 $\pmb{P}_k^{(\tau, s)} \in \mathbb{R}^{N_p \times N_p}$，使得 $\pmb{s}_k^{i,(s)}$ 与 $\pmb{s}_k^{i,(\tau)}$ 的时域波形尽量相似：
$$
\hat{\pmb{P}}_k^{(\tau, s)} = \underset{\pmb{P}_k^{(\tau, s)}}{\argmin} \dfrac{1 - \rho}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} \left\| \pmb{s}_k^{i,(s)} \pmb{P}_k^{(\tau, s)} - \bar{\pmb{s}}_k^{(\tau)} \right\|_F^2 + \rho \left\| \pmb{P}_k^{(\tau, s)} \right\|_F^2
\tag{12}
$$
由于时域投影矩阵的维度较大，因此最好通过正则化方法降低模型过拟合程度。式（12）中的 $\rho \left\| \pmb{P}_k^{(\tau, s)} \right\|_F^2$ 即为 L2 正则化项，用于约束投影矩阵的数值大小，减少模型过度依赖某一段时域特征的情况。超参数 $\rho$ 是惩罚系数，其具体数值可经验性地设置为 0.1。式（12）的解析解如下所示（证明过程见 **推导 3**）：
$$
\hat{\pmb{P}}_k^{(\tau, s)} = \left( 1 - \rho \right) {\left[ \left( 1 - \rho \right) \pmb{\Sigma}_{\pmb{s}, k}^{(s)} + \rho \pmb{I} \right]}^{-1} {\bar{\pmb{s}}_k^{(s)}}^T \pmb{s}_k^{(\tau)}
\tag{13}
$$

---
### 公共潜在子空间向目标域现实空间的正向传播
此时我们终于来到了domain adaptation 的最后一步：将公共潜在子空间内经过空间维度统一、时间维度对齐的源域源活动信号 $\hat{\pmb{U}}_k^{(\tau, s)} \pmb{X}_k^{i, (s)} \hat{\pmb{P}}_k^{(\tau, s)}$ 通过目标域的正向传播模式 $\hat{\pmb{A}}_k^{(\tau)}$ 映射到目标域的现实物理空间中，形成具有相似空间分布、相似时域波形的扩增特征信号 $\widetilde{\pmb{X}}_k^{i, (\tau, s)} \in \mathbb{R}^{N_c^{(s)} \times N_p}$：
$$
\widetilde{\pmb{X}}_k^{i, (\tau, s)} = \pmb{\Lambda}_k^{(\tau, s)} \left( \hat{\pmb{A}}_k^{(\tau)} \hat{\pmb{U}}_k^{(\tau, s)} \pmb{X}_k^{i, (s)} \hat{\pmb{P}}_k^{(\tau, s)} + \pmb{N}_k^{(\tau)} \right) \ \ \ \ or \ \ \ \ \pmb{\Lambda}_k^{(\tau, s)} \left( \hat{\pmb{A}}_k^{(\tau)} \hat{\pmb{w}}_k^{(\tau, s)} \pmb{X}_k^{i, (s)} \hat{\pmb{P}}_k^{(\tau, s)} + \pmb{N}_k^{(\tau)} \right)
\tag{14}
$$
不难发现，式（14）在空间滤波、时域对齐以及正向传播的基础上新增加了两个步骤：背景噪声估计 $\pmb{N}_k^{(\tau)} \in \mathbb{R}^{N_c^{(\tau)} \times N_p}$ 与幅值控制 $\pmb{\Lambda}_k^{(\tau, s)} \in \mathbb{R}^{N_c^{(\tau)} \times N_c^{(\tau)}}$。关于这两项平衡措施，目前还在技术迭代过程中，相关内容未来可能随时会更新。

首先来谈谈背景噪声估计 $\pmb{N}_k^{(\tau)}$。由于源活动通常可视为高度纯化的特征信号，而原始脑电信号（不论哪个导联）的信噪比往往要比源活动低得多，因此经过正向传播 $\hat{\pmb{A}}_k^{(\tau)} \hat{\pmb{U}}_k^{(\tau, s)} \pmb{X}_k^{i, (s)} \hat{\pmb{P}}_k^{(\tau, s)}$ 形成的扩增信号相比目标域单试次信号 $\pmb{X}_k^{i, (s)}$ 而言可能“过于干净”了。在 domain adaptation 过程中，将这种“过于干净”的信号与目标域原始信号直接合并，可能会影响、甚至干扰后续分类模型的训练过程。因此基于合理假设设计、添加噪声扰动是一种常见的、提高鲁棒性的操作方法（亦可视为一种正则化方法）。其中，空间（导联）分布独立的随机高斯噪声是最常用的噪声假设之一：
$$
\pmb{\Sigma}_{\pmb{N}, k}^{(\tau)} = \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} \left( \pmb{X}_k^{i, (\tau)} - \hat{\pmb{A}}_k^{(\tau)} \pmb{s}_k^{i, (\tau)} \right) {\left( \pmb{X}_k^{i, (\tau)} - \hat{\pmb{A}}_k^{(\tau)} \pmb{s}_k^{i, (\tau)} \right)}^T, \ \ \ \ \pmb{N}_k^{(\tau)} \sim N \left( 0, \dfrac{{\rm diag} \left( \pmb{\Sigma}_{\pmb{N}, k}^{(\tau)} \right) } {{\rm tr} \left( \pmb{\Sigma}_{\pmb{N}, k}^{(\tau)} \right)} \pmb{I} \right)
\tag{15}
$$
其中，${\rm diag} \left( \pmb{A} \right)$ 表示取方阵 $\pmb{A}$ 的对角线元素组成新的对角阵。当然，添加噪声的方式有很多种，在不同数据集上的最优方法可能各不相同，目前很难确定哪一种方法是广义上的最优解，甚至也可能不加噪声会更好。因此在实践操作中，建议不要把思维局限在某一种方式上，多多尝试。

接下来谈谈幅值控制 $\pmb{\Lambda}_k^{(\tau, s)}$。包含空间滤波、时域投影在内的诸多投影方法，在实际操作过程中，往往会出现投影过后信号幅值偏小的情况，这一点几乎是无法在求解投影矩阵的同时加以约束的（见 **讨论 2**）。因此，最好通过数值放缩降低扩增信号与目标域原始信号的幅值差异：
$$
\pmb{\Sigma}_{\widetilde{\pmb{X}}, k}^{(\tau)} = \dfrac{1}{N_t^{s}} \sum_{i=1}^{N_t^{(s)}} \left( \hat{\pmb{A}}_k^{(\tau)} \hat{\pmb{U}}_k^{(\tau, s)} \pmb{X}_k^{i, (s)} \hat{\pmb{P}}_k^{(\tau, s)} {\hat{\pmb{P}}_k^{(\tau, s)}}^T {\pmb{X}_k^{i, (s)}}^T {\hat{\pmb{U}}_k^{(\tau, s)}}^T {\hat{\pmb{A}}_k^{(\tau)}}^T \right), \ \ \ \ \pmb{\Lambda}_k^{(\tau, s)} = \sqrt{ \dfrac{{\rm diag} \left( \pmb{\Sigma}_{\pmb{X}, k}^{(\tau)} \right)} {{\rm diag} \left( \pmb{\Sigma}_{\widetilde{\pmb{X}}, k}^{(\tau)} \right)}}
\tag{16}
$$


---
### 讨论 1：传播矩阵 $\pmb{A}$ 与 Penrose 伪逆 $\pmb{w}^+$

1955 年，Penrose 给出了伪逆的数学定义：对于实矩阵 $\pmb{A} \in \mathbb{R}^{m \times n}$，若存在矩阵 $\pmb{X} \in \mathbb{R}^{n \times m}$ 满足如下方程组：
$$
\begin{align}
\notag \pmb{AXA} &= \pmb{A}\\
\notag \pmb{XAX} &= \pmb{X} \tag{x-2}\\
\notag \pmb{AX} &= \left( \pmb{AX} \right)^T\\
\notag \pmb{XA} &= \left( \pmb{XA} \right)^T\\
\end{align}
$$
则称 $\pmb{X}$ 为 $\pmb{A}$ 的广义逆（或伪逆）。其中，满足全部方程的伪逆称为 Penrose 伪逆（Moore-Penrose pseudoinverse），Penrose 伪逆存在且唯一，在 Python 语句中，可通过调用 `scipy.linalg.pinv()` 来进行计算；仅满足部分方程的广义逆可能不唯一。

简单起见，我们依然参照文首处那个双极系统的案例。在这个系统中，原始数据为 $\pmb{X} = [\pmb{s} + \pmb{n}, \pmb{n}]^T$，默认 $\pmb{s}$ 与 $\pmb{n}$ 不相关且均为一维序列，二者已经过零均值化处理。那么显然有 $\pmb{w} = [1, -1]$，Penrose 伪逆为 $\pmb{w}^+ = [0.5, -0.5]^T$，而依照（5）式计算的结果为（假设单试次场景）：
$$
\pmb{A} = \pmb{X} \pmb{X}^T \pmb{w}^T \left( \pmb{s} \pmb{s}^T \right)^{-1} =
\dfrac{1}{\pmb{s} \pmb{s}^T}
\begin{bmatrix}
\pmb{s} \pmb{s}^T + \pmb{n} \pmb{n}^T & 0\\
\ \\
0 & \pmb{n} \pmb{n}^T
\end{bmatrix}
\begin{bmatrix}
1\\ \ \\ -1\\
\end{bmatrix} = 
\begin{bmatrix}
1 + \dfrac{\pmb{n} \pmb{n}^T}{\pmb{s} \pmb{s}^T}\\
\ \\
-\dfrac{\pmb{n} \pmb{n}^T}{\pmb{s} \pmb{s}^T}
\end{bmatrix}
$$
不难发现，$\pmb{A}$ 并不是一个固定值，而是随着噪声与信号的能量变化而变化的：如果噪声能量（$\pmb{n} \pmb{n}^T$）远大于信号能量（$\pmb{s} \pmb{s}^T$），$\pmb{A}$ 与 $\pmb{w}^+$ 在数值比例上比较接近；但是当 $\pmb{n} \pmb{n}^T$ 相对较小时，显然 $\pmb{A}$ 对于信源信息 $\pmb{s}$ 的传播模式 $[1, 0]^T$ 估计得更为准确。

---
### 讨论 2：上下标与步骤可行性检查
由于本算法涉及到目标域与源域内不同个体间的数据对齐过程，因此在模型变量的上标往往会出现 $(\tau, s)$、$(\tau)$、$(s)$ 一类的标识。同时在跨设备（不同导联数）的应用场景下，本算法的部分步骤是无法进行的。因此针对上述情况，在此对具有复杂上下标记的变量及其维度其进行说明。

首先是式（1）的 $\hat{\pmb{w}}_k^{(\tau)} \in \mathbb{R}^{N_k \times N_c^{(\tau)}}$，。

接下来是式


---
### 推导 0：前置数学知识

首先是一些矩阵求导的常用公式：
$$
\begin{align}
\notag \dfrac{\partial \left( {\pmb{x}}^T \pmb{A} \right)}{\partial \left( \pmb{x} \right)} &= \dfrac{\partial \left( {\pmb{A}}^T \pmb{x} \right)}{\partial \left( \pmb{x} \right)} = \pmb{A}\\
\notag \ \\
\notag \dfrac{\partial \left( {\pmb{x}}^T \pmb{x} \right)}{\partial \pmb{x}} &= 2 \pmb{x}\\
\notag \ \\
\notag \dfrac{\partial \left( {\pmb{x}}^T \pmb{A} \pmb{x} \right)}{\partial \left( \pmb{x} \right)} &= \pmb{Ax} + {\pmb{A}}^T \pmb{x}\\
\notag \ \\
\notag \dfrac{\partial \left( {\pmb{A}}^T \pmb{x} {\pmb{x}}^T \pmb{B} \right)}{\partial \pmb{x}} &= \pmb{A} {\pmb{B}}^T \pmb{x} + \pmb{B} {\pmb{A}}^T \pmb{x}\\
\notag \ \\
\notag {\rm d} \left\{ {\rm tr} \left[ f (\pmb{X}) \right] \right\} &= {\rm tr} \left[ d f \left( \pmb{X} \right) \right] = {\rm d} f \left( \pmb{X} \right)
\end{align}
$$
此外，矩阵（乘积）的迹满足如下变换规则（$\pmb{A}$、$\pmb{B} \in \mathbb{R}^{m \times n}$）：
$$
\begin{align}
\notag {\rm tr} \left( \pmb{A} \right) &= {\rm tr} \left( {\pmb{A}}^T \right)\\
\notag \ \\
\notag {\rm tr} \left( \pmb{A} \pmb{B} \right) &= \sum_{i=1}^{m} \left( \pmb{A} \pmb{B} \right)_{ii} = \sum_{i=1}^{m} \sum_{j=1} ^{n} \pmb{A}_{ij} \pmb{B}_{ji} = \sum_{j=1}^{n} \sum_{i=1}^{m} \pmb{B}_{ji} \pmb{A}_{ij} = \sum_{j=1}^{m} \left( \pmb{B} \pmb{A} \right)_{jj} =  {\rm tr} \left( \pmb{B} \pmb{A} \right)\\
\notag \ \\
\notag {\rm tr} \left( \pmb{A} {\pmb{B}}^T \right) &= {\rm tr} \left( {\pmb{B}}^T \pmb{A} \right) = {\rm tr} \left( {\pmb{A}}^T \pmb{B} \right) = {\rm tr} \left( \pmb{B} {\pmb{A}}^T \right)
\end{align}
$$
最后，矩阵的 Frobenius 范数可展开成矩阵迹形式：
$$
\left\| \pmb{A} \right\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} \left| \pmb{A}_{ij} \right|^2} = \sqrt{{\rm tr} \left( \pmb{A} \pmb{A}^T \right)}, \ \ \pmb{A} \in \mathbb{R}^{m \times n}
$$

---
### 推导 1：式（4）-（5）

式（4）如下：
$$
\hat{\pmb{A}}_k^{(\tau, s)} = \underset{\pmb{A}_k^{(\tau, s)}} \argmin \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} \left\| \pmb{A}_k^{(\tau, s)} \pmb{s}_k^{i,(\tau, s)} - \pmb{X}_k^{i,(\tau)} \right\|_F^2
$$
构建矩阵函数 $f \left( \pmb{A}_k^{(\tau, s)} \right)$：
$$
\begin{align}
\notag f \left( \pmb{A}_k^{(\tau, s)} \right) &= \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} {\rm tr} \left[ \left( \pmb{A}_k^{(\tau, s)} \pmb{s}_k^{i,(\tau, s)} - \pmb{X}_k^{i,(\tau)} \right) \left( \pmb{A}_k^{(\tau, s)} \pmb{s}_k^{i,(\tau, s)} - \pmb{X}_k^{i,(\tau)} \right)^T \right]\\
\notag \ \\
\notag &= \dfrac{1}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} {\rm tr} \left( \pmb{A}_k^{(\tau, s)} \pmb{s}_k^{i,(\tau, s)} {\pmb{s}_k^{i,(\tau, s)}}^T {\pmb{A}_k^{(\tau, s)}}^T - \pmb{A}_k^{(\tau, s)} \pmb{s}_k^{i,(\tau, s)} {\pmb{X}_k^{i,(\tau)}}^T - \pmb{X}_k^{i,(\tau)} {\pmb{s}_k^{i,(\tau, s)}}^T {\pmb{A}_k^{(\tau, s)}}^T + \pmb{X}_k^{i,(\tau)} {\pmb{X}_k^{i,(\tau)}}^T \right)\\
\notag \ \\
\end{align}
$$
对 $f \left( \pmb{A}_k^{(\tau, s)} \right)$ 求导并置零：
$$
\dfrac{d f \left( \pmb{A}_k^{(\tau, s)} \right)} {d \pmb{A}_k^{(\tau, s)}} = \dfrac{2}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} \pmb{A}_k^{(\tau, s)} \pmb{s}_k^{i,(\tau, s)} {\pmb{s}_k^{i,(\tau, s)}}^T - \dfrac{2}{N_t^{(\tau)}} \sum_{i=1}^{N_t^{(\tau)}} \pmb{X}_k^{i,(\tau)} {\pmb{s}_k^{i, (\tau, s)}}^T = 0
$$
求解可得：
$$
\hat{\pmb{A}}_k^{(\tau, s)} = \left( \sum_{i=1}^{N_t^{(\tau)}} \pmb{X}_k^{i,(\tau)} {\pmb{s}_k^{i, (\tau, s)}}^T \right) \left( \sum_{i=1}^{N_t^{(\tau)}} \pmb{s}_k^{i,(\tau, s)} {\pmb{s}_k^{i,(\tau, s)}}^T \right)^{-1} = \left( \sum_{i=1}^{N_t^{(\tau)}} \pmb{X}_k^{i,(\tau)} {\pmb{X}_k^{i,(\tau)}}^T \right) {\hat{\pmb{w}}_k^{(\tau,s)}}^T \left( \sum_{i=1}^{N_t^{(\tau)}} \pmb{s}_k^{i,(\tau, s)} {\pmb{s}_k^{i,(\tau, s)}}^T \right)^{-1} = \pmb{\Sigma}_{\pmb{X}, k}^{(\tau)} {\hat{\pmb{w}}_k^{(\tau,s)}}^T {\pmb{\Sigma}_{\pmb{s}, k}^{(\tau, s)}}^{-1}
$$

---
### 推导 2：式（7）-（9）

式（7）如下：
$$
\hat{\pmb{U}}_k^{(\tau, s)} = \underset{\pmb{U}_k^{(\tau, s)}}{\argmin} \dfrac{1}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} \left\| \pmb{U}_k^{(\tau, s)} \pmb{X}_k^{i,(s)} - \bar{\pmb{s}}_k^{(\tau, s)} \right\|_F^2
$$
以类似 **推导 1** 的方法构建矩阵函数 $f \left( \pmb{U}_k^{(\tau, s)} \right)$ 并求导置零：
$$
\begin{align}
\notag f \left( \pmb{U}_k^{(\tau, s)} \right) &= \dfrac{1}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} {\rm tr} \left( \pmb{U}_k^{(\tau, s)} \pmb{X}_k^{i,(s)} {\pmb{X}_k^{i,(s)}}^T {\pmb{U}_k^{(\tau, s)}}^T - \pmb{U}_k^{(\tau, s)} \pmb{X}_k^{i,(s)} {\bar{\pmb{s}}_k^{(\tau, s)}}^T - \bar{\pmb{s}}_k^{(\tau, s)} {\pmb{X}_k^{i,(s)}}^T {\pmb{U}_k^{(\tau, s)}}^T + \bar{\pmb{s}}_k^{(\tau, s)} {\bar{\pmb{s}}_k^{(\tau, s)}}^T \right)\\
\notag \ \\
\notag \dfrac{d f \left( \pmb{U}_k^{(\tau, s)} \right)}{d \pmb{U}_k^{(\tau, s)}} &= \dfrac{2}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} \pmb{U}_k^{(\tau, s)} \pmb{X}_k^{i,(s)} {\pmb{X}_k^{i,(s)}}^T - \dfrac{2}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} \bar{\pmb{s}}_k^{(\tau, s)} {\pmb{X}_k^{i,(s)}}^T = 0
\end{align}
$$
求解可得：
$$
\hat{\pmb{U}}_k^{(\tau, s)} = \left( \sum_{i=1}^{N_t^{(s)}} \bar{\pmb{s}}_k^{(\tau, s)} {\pmb{X}_k^{i,(s)}}^T \right) \left( \sum_{i=1}^{N_t^{(s)}} \pmb{X}_k^{i,(s)} {\pmb{X}_k^{i,(s)}}^T \right)^{-1} = \bar{\pmb{s}}_k^{(\tau, s)} {\bar{\pmb{X}}_k^{(s)}}^T {\pmb{\Sigma}_{\pmb{X}, k}^{(s)}}^{-1}
$$
式（9）如下：
$$
\hat{\pmb{U}}_k^{i, (\tau, s)} = \underset{\pmb{U}_k^{i, (\tau, s)}}{\argmin} \left\| \pmb{U}_k^{i, (\tau, s)} \pmb{X}_k^{i,(s)} - \bar{\pmb{s}}_k^{(\tau, s)} \right\|_F^2
$$
同理进行上述操作，可以得到 $f \left( \pmb{U}_k^{i, (\tau, s)} \right)$：
$$
f \left( \pmb{U}_k^{i, (\tau, s)} \right) = {\rm tr} \left( \pmb{U}_k^{i, (\tau, s)} \pmb{X}_k^{i, (s)} {\pmb{X}_k^{i, (s)}}^T {\pmb{U}_k^{i, (\tau, s)}}^T - \pmb{U}_k^{i, (\tau, s)} \pmb{X}_k^{i, (s)} {\bar{\pmb{s}}_k^{(\tau, s)}}^T - \bar{\pmb{s}}_k^{(\tau, s)} {\pmb{X}_k^{i, (s)}}^T {\pmb{U}_k^{i, (\tau, s)}}^T + \bar{\pmb{s}}_k^{(\tau, s)} {\bar{\pmb{s}}_k^{(\tau, s)}}^T \right)
$$
求导后置零可得：
$$
\dfrac{d f \left( \pmb{U}_k^{i, (\tau, s)} \right)}{d \pmb{U}_k^{i, (\tau, s)}} = 2 \pmb{U}_k^{i, (\tau, s)} \pmb{X}_k^{i, (s)} {\pmb{X}_k^{i, (s)}}^T - 2 \bar{\pmb{s}}_k^{(\tau, s)} {\pmb{X}_k^{i, (s)}}^T = 0
$$
求解可得：
$$
\hat{\pmb{U}}_k^{i, (\tau, s)} = \bar{\pmb{s}}_k^{(\tau, s)} {\pmb{X}_k^{i, (s)}}^T \left( \pmb{X}_k^{i, (s)} {\pmb{X}_k^{i, (s)}}^T \right)^{-1}
$$

---
### 推导 3：式（12）-（13）

式（12）如下：
$$
\hat{\pmb{P}}_k^{(\tau, s)} = \underset{\pmb{P}_k^{(\tau, s)}}{\argmin} \dfrac{1 - \rho}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} \left\| \pmb{s}_k^{i,(s)} \pmb{P}_k^{(\tau, s)} - \bar{\pmb{s}}_k^{(\tau)} \right\|_F^2 + \rho \left\| \pmb{P}_k^{(\tau, s)} \right\|_F^2
$$
将 Frobenius 范数展开成迹形式，构建矩阵函数 $f \left( \pmb{P}_k^{(\tau, s)} \right)$：
$$
f \left( \pmb{P}_k^{(\tau, s)} \right) = \dfrac{1 - \rho}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} {\rm tr} \left[ \left( \pmb{s}_k^{i,(s)} \pmb{P}_k^{(\tau, s)} - \bar{\pmb{s}}_k^{(\tau)} \right) {\left( \pmb{s}_k^{i,(s)} \pmb{P}_k^{(\tau, s)} - \bar{\pmb{s}}_k^{(\tau)} \right)}^T \right] + \rho {\rm tr} \left( \pmb{P}_k^{(\tau, s)} {\pmb{P}_k^{(\tau, s)}}^T \right)
$$
根据矩阵乘积的迹的变换规则，$f \left( \pmb{P}_k^{(\tau, s)} \right)$ 可以改写成：
$$
\begin{align}
\notag f \left( \pmb{P}_k^{(\tau, s)} \right) &= \dfrac{1 - \rho}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} {\rm tr} \left[ \left( {\pmb{P}_k^{(\tau, s)}}^T {\pmb{s}_k^{i,(s)}}^T - {\bar{\pmb{s}}_k^{(\tau)}}^T \right) \left( \pmb{s}_k^{i,(s)} \pmb{P}_k^{(\tau, s)} - \bar{\pmb{s}}_k^{(\tau)} \right) \right] + \rho {\rm tr} \left( {\pmb{P}_k^{(\tau, s)}}^T \pmb{P}_k^{(\tau, s)} \right)\\
\notag \ \\
\notag &= \dfrac{1 - \rho}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} {\rm tr} \left( {\pmb{P}_k^{(\tau, s)}}^T {\pmb{s}_k^{i,(s)}}^T \pmb{s}_k^{i,(s)} \pmb{P}_k^{(\tau, s)} - {\pmb{P}_k^{(\tau, s)}}^T {\pmb{s}_k^{i,(s)}}^T \bar{\pmb{s}}_k^{(\tau)} - {\bar{\pmb{s}}_k^{(\tau)}}^T \pmb{s}_k^{i,(s)} \pmb{P}_k^{(\tau, s)} + {\bar{\pmb{s}}_k^{(\tau)}}^T \pmb{s}_k^{i,(s)} \right) + \rho {\rm tr} \left( {\pmb{P}_k^{(\tau, s)}}^T \pmb{P}_k^{(\tau, s)} \right)
\end{align}
$$
求导并置零：
$$
\begin{align}
\notag \dfrac{d f \left( \pmb{P}_k^{(\tau, s)} \right)}{d \pmb{P}_k^{(\tau, s)}} &= \dfrac{2 \left( 1 - \rho \right)}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} {\pmb{s}_k^{i,(s)}}^T \pmb{s}_k^{i,(s)} \pmb{P}_k^{(\tau, s)} - \dfrac{2 \left( 1 - \rho \right)}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} {\pmb{s}_k^{i,(s)}}^T \bar{\pmb{s}}_k^{(\tau)} + 2 \rho \pmb{P}_k^{(\tau, s)}\\
\notag \ \\
\notag &= 2 \left[ \left( 1 - \rho \right) \pmb{\Sigma}_{\pmb{s}, k}^{(s)} \pmb{P}_k^{(\tau, s)} + \rho \pmb{P}_k^{(\tau, s)} \right] - 2 \left( 1 - \rho \right) {\bar{\pmb{s}}_k^{(s)}}^T \pmb{s}_k^{(\tau)} = 0
\end{align}
$$
化简后可得：
$$
\hat{\pmb{P}}_k^{(\tau, s)} = \left( 1 - \rho \right) {\left[ \left( 1 - \rho \right) \pmb{\Sigma}_{\pmb{s}, k}^{(s)} + \rho \pmb{I} \right]}^{-1} {\bar{\pmb{s}}_k^{(s)}}^T \pmb{s}_k^{(\tau)}
$$




$$
\begin{align}
\notag \rho_{k,1} &= {\rm corr} \left( \hat{\pmb{w}}_k^{(\tau)} \pmb{\mathcal{X}}, \ \ \hat{\pmb{w}}_k^{(\tau)} \bar{\pmb{X}}_k^{(\tau)} \right)\\
\notag \ \\
\notag \rho_{k,2} &= {\rm corr} \left( \hat{\pmb{w}}_k^{(\tau)} \pmb{\mathcal{X}}, \ \ \sum_{s=1}^{N_s} \hat{\psi}(s) \hat{\pmb{w}}_k^{(s)} \bar{\pmb{X}}_k^{(s)} \right)\\
\notag \ \\
\notag \rho_{k,3} &= {\rm corr} \left( \hat{\pmb{w}}_k^{(\tau)} \pmb{\mathcal{X}} \hat{\pmb{P}}_k, \ \ \sum_{s=1}^{N_s} \hat{\psi}(s) \hat{\pmb{w}}_k^{(s)} \bar{\pmb{X}}_k^{(s)} \right)\\
\notag \ \\
\notag \rho_{k,4} &= {\rm corr} \left( \hat{\pmb{w}}_k^{(\tau)} \pmb{\mathcal{X}} \hat{\pmb{P}}_k, \ \ \hat{\pmb{w}}_k^{(\tau)} \bar{\pmb{X}}_k^{(\tau)} \hat{\pmb{P}}_k \right)
\end{align}
$$



[alpha]: https://ieeexplore.ieee.org/document/9516951/
[refer1]: https://linkinghub.elsevier.com/retrieve/pii/S1053811913010914
[LST 对齐方法]: https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e
[sd-LST 算法]: https://ieeexplore.ieee.org/document/9967845/
[基于 TRCA 的迁移学习算法]: https://ieeexplore.ieee.org/document/10057002/
[原型滤波器]: https://ieeexplore.ieee.org/document/8616087/