# EEG 采集设备数据域适应的对齐与池化方法
## Align and Pool for EEG Headset Domain Adaptation, ALPHA
***

[论文链接][alpha]

2022 年这篇发表在 TBME 期刊上的域适应算法，给 SSVEP 领域的迁移学习（广义）算法带来了一些新的概念，比如子空间（Subspace）、对齐空间模式（Align spatial pattern, ASP）、对齐协方差矩阵（Align covariance, AC）、子空间池化（Subspace pooling）等。当然这些术语基本上都算是“舶来品”，在运动想象或其它机器学习领域里曾得到广泛应用。这些新概念（对于 SSVEP 而言）非常有助于拓展后续研究的思路，不然大家总是绞尽脑汁搞一些 gTRCA、xTRCA、LA-TRCA 之类的改进，在这种道路上的创新空间早已越来越小了。

该文献首先为 ALPHA 适用的场景进行了抽象化定义：对于一个有标签源数据域 $\mathcal{D}^{(s)}$，它可表示为数据随机变量 $\mathcal{X}^{(s)}$ 与标签随机变量 $\mathcal{Y}^{(s)}$ 的联合分布：
$$
\mathcal{D}^{(s)} = \left\{ \mathcal{X}^{(s)}, \mathcal{Y}^{(s)} \right\} = \left\{ \left( \pmb{X}^{i,(s)}, y^{i,(s)} \right) \right\}, \ \ \ \ i=1,2,\cdots,N_s
\tag{1}
$$
其中 $\mathcal{X}^{(s)}$ 的元素为数据矩阵（或张量）$\pmb{X}^{i,(s)}$，$\mathcal{Y}^{(s)}$ 的元素为标签（离散整型变量 $y^{i,(s)}$）。$(s)$ 为源索引（$s=1,2,\cdots$），通常表示第 $s$ 个受试者的数据或第 $s$ 个数据源；$i$ 为样本索引，$N_s$ 为样本总数，在面向 SSVEP 公开数据集的应用场景下，$N_s$ 通常等于 $N_t \times N_e$，即每类样本数乘上类别总数。同理目标数据域也有类似的数学表示 $\mathcal{D}^{(\tau)}$。

插一句题外话，在我的专栏中，通常会假设全体类别数据（$N_e$ 类）拥有相同数目的训练样本（$N_t$ 个），但其实在我个人的代码库中，函数接口要求输入数据的格式规定为“**数据** + **标签**”，即两个变量：`(n_trials, n_chans, n_points)`& `(n_trials,)`，这与 scikit-learn 公开库的标准是一致的。这种设计主要是考虑到实际数据采集完成后，在数据集清洗过程中可能会剔除部分样本从而导致微弱的样本不均衡现象，同时很多实验范式会刻意设计不均衡的样本数目来观察偶发事件下产生的各种特殊效应。但是部分算法又要求为每类数据的模型训练使用相同数目的样本（如 ASS-IISCCA 等），所以我特地写了一个数据集格式转换的函数`utils.reshape_dataset()`，可实现两种`style`的数据集格式转换（`'public'`和`'sklearn'`）。当然从前者向后者转，一般是无损的；后者向前者转，则根据最小的样本数目决定每个类的最终样本数目，因此有时候是有损的。

言归正传，ALPHA 所关注的跨设备应用问题，可表示为 $\mathcal{D}^{(s)}$ 与 $\mathcal{D}^{(\tau)}$ 来源于不同的设备（$\mathcal{D}^{(s)} \ne \mathcal{D}^{(\tau)}$），该问题还应满足以下条件：

（1）$\mathcal{X}^{(s)} \ne \mathcal{X}^{(\tau)}$，即源域个体与目标域个体的数据分布不同，根据脑电信号的随机性可知，只要没有数据泄露，这个条件基本上是必定满足的；

（2）$\mathcal{Y}^{(s)} = \mathcal{Y}^{(\tau)}$，源域与目标域刺激的类别是相同的，其比例应该均衡且相似。举个不太恰当的例子，假如目标域的绝大部分测试样本都是第 $k$ 类的（样本极端不均衡），那就没必要训练模型了，直接无脑判断任何输入样本都是第 $k$ 类，最后分类准确率肯定不差，显然这样是没有意义的；

（3）$| \mathcal{Y} | = N_e$，其中 $|·|$ 表示集合的基数，即目标域不应该出现源域中完全不包含的全新类别。

此外，ALPHA 还限定了目标域是无标签的，即：
$$
\mathcal{D}^{(\tau)} = \left\{ \mathcal{X}^{(\tau)} \right\} = \left\{ \pmb{X}^{(\tau)} \right\}
\tag{2}
$$
作者在文中表示，对于跨设备迁移学习，由于数据统计分布的差异（Domain gap）导致源域数据训练的模型通常很难直接应用于目标域，因此他们提出了 ALPHA 算法来尝试解决这种问题。这套话术着实不错，事实上别说跨设备了，跨个体、跨时间、跨范式，甚至同一个受试者前一组实验的时候犯困了，后一组精神了，两组实验数据之间可能都存在明显的 domain gap。

接下来我们正式开始拆解这个算法。首先定义本专栏所需的变量符号表示：将所属类别为 $k$（类别总数为 $N_e$）、样本索引为 $i$（单类别样本总数为 $N_t$）、来源于第 $s$ 个源受试者的多导联数据矩阵记为 $\pmb{X}_k^{i,(s)} \in \mathbb{R}^{N_c \times N_p}$，$N_c$、$N_p$ 分别表示导联数目和采样点数；单试次目标受试者数据记为 $\pmb{X}_k^{(\tau)}$，第 $k$ 类刺激对应的人工正余弦模板记为 $\pmb{Y}_k \in \mathbb{R}^{2N_h \times N_p}$，其中 $N_h$ 表示谐波数目。

如下图所示，ALPHA 的总体步骤分为三个部分：**子空间分解**（Subspace decomposition）、**子空间对齐**（Subspace alignment）以及**子空间池化**（Subspace pooling）。接下来我们依次讲述每一步骤的公式细节。

![ALPHA算法流程图](/SSVEP_algorithms/数据增强%20&%20迁移学习算法/ALPHA-1.png)

---
### 子空间分解（Subspace decomposition）
此处的“decomposition”可以理解为“矩阵分解”（Matrix decomposition）的一部分，实际上指的是协方差矩阵的特征值分解以及特征向量选取，也就是空间滤波器计算步骤中的最后一步。ALPHA 使用的主要是基于 CCA 的分解方法，这种方法目前依旧是无监督 SSVEP 机器学习解码算法中不可避免的环节之一。本专栏中不对该算法的步骤进行详述，有需要请参考典型相关性分析章节中的 CCA 部分，下文将简记为 ${\rm CCA} \left( \pmb{X},\pmb{Y} \right)$。ALPHA 一共设计了五套矩阵分解，其中前 4 种均为 CCA 分解，它们的具体形式（或实质）分别为：

（1）面向目标域未知类别测试样本 $\pmb{X}^{(\tau)}$ 的标准 CCA：
$$
\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}, \ \ \hat{\pmb{V}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} = {\rm CCA} \left( \pmb{X}^{(\tau)}, \pmb{Y}_k \right), \ \ \ \ \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \in \mathbb{R}^{N_k \times N_c}, \ \ \hat{\pmb{V}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \in \mathbb{R}^{N_k \times 2N_h}
\tag{3}
$$
需要注意的是，根据原文的定义，此处我特地用了大写字母 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$、$\hat{\pmb{V}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$ 表示矩阵分解结果，即此处获得的空间滤波器是全部特征向量构成的矩阵。根据我（~~桀骜不驯~~）的**向量行式表示法则**，在 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$ 中，每一个行向量表示一个特征向量，即空间滤波过程表示为 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \pmb{X}^{(\tau)}$ 而不是 ${\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}}^T \pmb{X}^{(\tau)}$，$N_k$ 表示子空间维度，特地与 $N_c$ 在物理含义上加以区分（在 ALPHA 中 $N_k = N_c$）

（2）面向源域模板（按试次叠加平均） $\bar{\pmb{X}}_k^{(s)}$ 的 CCA（eCCA 的一部分）：
$$
\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}, \ \ \hat{\pmb{V}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k} = {\rm CCA} \left( \bar{\pmb{X}}_k^{(s)}, \pmb{Y}_k \right), \ \ \ \ \hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k} \in \mathbb{R}^{N_k \times N_c}
\tag{4}
$$

（3）面向源域训练样本 $\pmb{X}_k^{i,(s)}$ 的 TDCCA（目前专栏未收录，未来会更新）：
$$
\hat{\pmb{U}}_{\pmb{X}_k^{i,(s)} \bar{\pmb{X}}_k^{(s)}}, \ \ \hat{\pmb{V}}_{\pmb{X}_k^{i,(s)} \bar{\pmb{X}}_k^{(s)}} = {\rm CCA} \left( \pmb{X}_k^{i,(s)}, \bar{\pmb{X}}_k^{(s)} \right), \ \ \ \ \hat{\pmb{U}}_{\pmb{X}_k^{i,(s)} \bar{\pmb{X}}_k^{(s)}} \in \mathbb{R}^{N_k \times N_c}
\tag{5}
$$

（4）面向源域模板 $\bar{\pmb{X}}_k^{(s)}$ 与目标域样本 $\pmb{X}_k^{(\tau)}$ 的 ttCCA。需要注意的是，保留的滤波器 $\hat{\pmb{U}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}}$ 是面向目标域的：
$$
\hat{\pmb{U}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}}, \ \ \hat{\pmb{V}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}} = {\rm CCA} \left( \pmb{X}_k^{(\tau)}, \bar{\pmb{X}}_k^{(s)} \right), \ \ \ \ \hat{\pmb{U}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}} \in \mathbb{R}^{N_k \times N_c}
\tag{6}
$$

在上述分解的基础上，ALPHA 还新增了一种 LDA 类型的滤波器，其实本质就是 DSP（目前专栏未收录，未来会更新），或者说是不加空间维度、时间维度扩增的 TDCA（目前专栏未收录，未来会更新），这里将该过程简记为 ${\rm DSP} (\pmb{X})$：
$$
\hat{\pmb{W}}_{\pmb{X}^{(s)}} = {\rm DSP} \left(\pmb{X}^{(s)} \right), \ \ \ \ \hat{\pmb{W}}_{\pmb{X}^{(s)}} \in \mathbb{R}^{N_k \times N_c}
\tag{7}
$$
总之，在子空间分解阶段，ALPHA 需要从源域训练数据集 $\mathcal{D}^{(s)}$ 与目标域单试次测试信号 $\pmb{X}^{(\tau)}$ 中获取五个空间滤波器（矩阵）：$\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$、$\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}$、$\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{X}_k^{i,(s)}}$、$\hat{\pmb{U}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}}$ 以及 $\hat{\pmb{W}}_{\pmb{X}^{(s)}}$。

---
### 子空间对齐（Subspace alignment）
子空间对齐的主要目的是将 $\mathcal{D}^{(s)}$ 与 $\mathcal{D}^{(\tau)}$ 在相同的子空间内进行对齐，从而减少数据域差异的影响。ALPHA 使用了两种子空间对齐方法：对齐空间模式（ASP）与对齐协方差矩阵（AC）。

#### Align Spatial Pattern, ASP
在 ASP 过程中，顾名思义，对齐的是“空间模式”（Spatial pattern）。这个概念引入自 2014 年一篇非常重要的 [Neuroimage 文献]，其大意是说从 EEG 的反向传播模型（Backward-propagation model）中可以重建出正向传播模型（Forward-propagation model）。这里的“**反向**”指的是通过观测数据推断信号源（或信号能量）分布，而“**正向**”指的是通过建立的源模型估计在采集通道（头皮导联）获得的混叠观测数据。根据以上定义，空间滤波器显然属于反向模型。文献指出，反向模型的数值并不能反映信号源分布或信息传播情况，例如对于一个双通道系统 $[S+N,N]^T$，其中 $S$ 和 $N$ 分别表示有效信号与噪声，其对应的最优空间滤波器为 $[1,-1]$，但我们并不能由此说该系统的两个通道包含了等量的信号成分，空间滤波器的数值是不具备直接物理意义的。文章指出，从空间滤波器中可以计算得到空间模式，详细原理与证明细节请参考源文献。

关于空间模式的计算，ALPHA 文献中给出的标记 $\pmb{W}_X^{-T}$ 可能表示直接求逆后转置（转置是因为他们用列向量形式记录空间滤波器）。这种方法在 Neuroimage 文献中似乎被拿来做了对比，且并不推荐。因此接下来我参照 Neuroimage 文献的说明给出计算方法，以 $\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}$ 为例，经过空间滤波后的单试次信号源活动估计 $\pmb{S}_k^{i,(s)} \in \mathbb{R}^{N_k \times N_p}$ 为:
$$
\pmb{S}_k^{i,(s)} = \hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k} \pmb{X}_k^{i,(s)}
\tag{8}
$$
基于 Frobenius 范数计算源信号 $\pmb{S}_k^{(s)}$ 传播到头皮各采集导联的传播模式（矩阵）$\pmb{A}_k^{(s)} \in \mathbb{R}^{N_c \times N_k}$：
$$
\begin{align}
\notag \hat{\pmb{A}}_k^{(s)} &= \underset{\pmb{A}_k^{(s)}} {\arg\min} \dfrac{1}{N_t} \sum_{i=1}^{N_t} \left\| \pmb{A}_k^{(s)} \pmb{S}_k^{i,(s)} - \pmb{X}_k^{i,(s)} \right\|_F^2\\
\notag \ \\
\notag &= \left( \dfrac{1}{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^{i,(s)} {\pmb{X}_k^{i,(s)}}^T \right) {\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}}^T \left( \dfrac{1}{N_t} \sum_{i=1}^{N_t} \pmb{S}_k^{i,(s)} {\pmb{S}_k^{i,(s)}}^T \right)^{-1} = \pmb{\Sigma}_{\pmb{X},k}^{(s)} {\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}}^T {\pmb{\Sigma}_{\pmb{S},k}^{(s)}}^{-1}
\end{align}
\tag{9}
$$
其中 $\pmb{\Sigma}$ 表示方差矩阵。类似地，对 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$ 有：
$$
\hat{\pmb{A}}_k^{(\tau)} = \pmb{\Sigma}_{\pmb{X}}^{(\tau)} {\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}}^T \pmb{\Sigma}_{\pmb{S},k}^{(\tau)}, \ \ \ \ \pmb{S}_k^{(\tau)} = \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \pmb{X}^{(\tau)}
\tag{10}
$$
接下来，ALPHA 假设 $\hat{\pmb{A}}_k^{(\tau)}$ 与 $\hat{\pmb{A}}_k^{(s)}$ 经过旋转（Rotation）之后可以达到更加一致的状态，因此需要计算一个正交转换矩阵 $\pmb{P}_k \in \mathbb{R}^{N_k \times N_k}$ 使得：
$$
\hat{\pmb{P}}_k = \underset{\pmb{P}_k} {\arg\min} \left\| \hat{\pmb{A}}_k^{(s)} - \hat{\pmb{A}}_k^{(\tau)} {\pmb{P}_k}^T \right\|_F^2, \ \ \ \ s.t. \ \ \hat{\pmb{P}}_k {\hat{\pmb{P}}_k}^T = \pmb{I}_{N_c}
\tag{11}
$$
式（11）在数学形式上是典型的 Procrustes 问题，此处不讨论证明过程，仅给出可行的解析解：
$$
\hat{\pmb{P}}_k = \pmb{\mathcal{U}}_k {\pmb{\mathcal{V}}_k}^T
\tag{12}
$$
其中 $\pmb{\mathcal{U}}_k$、$\pmb{\mathcal{V}}_k$ 是对 ${\hat{\pmb{A}}_k^{(s)}}^T \hat{\pmb{A}}_k^{(\tau)}$ 进行奇异值分解（Singular value decomposition, SVD）得到的左、右奇异向量（矩阵）。文中没有指出保留奇异值的个数，因此推测是全部保留。最后得到投影后的目标域空间滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \pmb{Y}_k}^{{\rm ASP}}$，并将上述步骤简记为 ${\rm ASP} (*)$：
$$
\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \pmb{Y}_k}^{{\rm ASP}} = {\hat{\pmb{P}}_k}^T \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} = {\rm ASP} \left( \hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}, \ \ \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \right)
\tag{13}
$$
类似地，对 $\hat{\pmb{U}}_{\pmb{X}_k^{i,(s)} \bar{\pmb{X}}_k^{(s)}}$、$\hat{\pmb{U}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}}$ 进行 ASP 操作，获得 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \bar{\pmb{X}}_k^{(s)}}^{{\rm ASP}} = {\rm ASP} \left( \hat{\pmb{U}}_{\pmb{X}_k^{i,(s)} \bar{\pmb{X}}_k^{(s)}}, \ \ \hat{\pmb{U}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}} \right)$。

需要注意的是，空间传播矩阵的计算结果与使用了何种空间滤波器是紧密相关的。因此式（13）中使用的旋转矩阵 ${\hat{\pmb{P}_k}}^T$ 不可以用于构建 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \bar{\pmb{X}}_k^{(s)}}^{{\rm ASP}}$，这也是我要用 ${\rm ASP} (*)$ 的形式概括上述步骤的主要原因。此外，之所以使用这两组空间滤波器分别进行 ASP 步骤，是因为 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$ 与 $\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}$ 是面向相同的正余弦模板 $\pmb{Y}_k$ 构建的，因此两者之间的差异有望通过对齐手段进行弥补；同理 $\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{X}_k^{i,(s)}}$ 与 $\hat{\pmb{U}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}}$ 也属于一组。

这里我们有必要先停下来想一想为什么要按式（13）对滤波器进行投影变换。回顾式（9）与（11）可以发现，传播矩阵（简记为 $\pmb{A}$）的作用是将空间滤波后的高质量源信号估计（简记为 $\pmb{wX}$）正向传播到各个采集导联上，其主要作用形式为 $\pmb{AwX}$。而变换矩阵 $\hat{\pmb{P}}_k$ 的作用是对目标域的传播矩阵 $\hat{\pmb{A}}_k^{(\tau)}$ 进行投影 $\hat{\pmb{A}}_k^{(\tau)} {\hat{\pmb{P}}_k}^T$，使得它更接近源域的传播模式 $\hat{\pmb{A}}_k^{(s)}$。所以式（13）的投影变换过程应当放在如下场景，也许能更好地帮助理解：
$$
\hat{\pmb{A}}_k^{(\tau)} {\hat{\pmb{P}}_k}^T \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \pmb{X}^{(\tau)} \sim \hat{\pmb{A}}_k^{(s)} \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \pmb{X}^{(\tau)}
\tag{14}
$$
变换矩阵 $\hat{\pmb{P}}_k$ 并不是针对目标域空间滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$ 的，而是通过修饰目标域的正向传播矩阵 $\hat{\pmb{A}}_k^{(\tau)}$ ，模拟目标域源信号 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \pmb{X}^{(\tau)}$ 在源域下的传播模式 $\hat{\pmb{A}}_k^{(s)}$ ，希望更好地与源域模板 $\bar{\pmb{X}}_k^{(s)}$ 进行匹配。至于对目标域空间滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$ 进行投影变换 ${\hat{\pmb{P}}_k}^T \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$ ，只是式（14）展示的完整视角下的一个切片而已。

从另一方面来看，由前所述，空间滤波器数值并不直接反映信号能量分布或传播模式，正向传播矩阵才真正具备了相关的生理意义，而在模板匹配阶段又不需要用到传播矩阵，只需要用到空间滤波器。因此以往算法直接针对空间滤波器或者滤波后模板进行数据对齐可能达不到理想效果，但是通过传播矩阵的对齐，并作为修饰投影至空间滤波器上，可能会起到更好的迁移效果。

可能有读者会问，既然式（11）的误差越小越好，为什么不直接在式（14）中用 $\hat{\pmb{A}}_k^{(s)}$ 替换掉 $\hat{\pmb{A}}_k^{(\tau)}$ 呢？关于这一点我的理解是，别人的毕竟还是别人的，直接拿来用总是会有各种各样的不匹配问题。真实条件下式（11）的误差几乎不可能做到非常非常小，而且 $\hat{\pmb{P}}_k$ 同时包含了源域与目标域的信息，既是向源域信息对齐的“桥梁”，也是保留目标域特性的一种权衡手段。如果未知样本数据可以完全转化为已有数据，那从信息论的角度来看，新样本没有提供任何新的信息，其信息熵不就是 0 了吗。所以不论是从数学理论上，还是实践目的上，直接使用目标域的传播模式都是不太可取的。

#### Align covariance, AC
一般认为，协方差矩阵不仅包含了不同导联信号总体能量信息，还包含了导联间信号相关性（互相关）信息。除了空间模式的偏移，作者认为在协方差上，不同的数据域之间同样存在偏移，而且这种偏移是可以通过旋转变换加以缓解的。因此 ALPHA 使用了一种基于相关性对齐（Corelation alignment, CORAL）的方法，通过寻找线性变换矩阵 $\pmb{Q} \in \mathbb{R}^{N_c \times N_c}$ 使得 $\mathcal{D}^{(s)}$ 与 $\mathcal{D}^{(\tau)}$ 的二阶统计量（$\pmb{C}^{(s)}$、$\pmb{C}^{(\tau)}$）差异最小，其目标函数与解析解如下所示：
$$
\hat{\pmb{Q}} = \underset{\pmb{Q}} {\arg\min} \left\| \pmb{Q}^T \pmb{C}^{(s)} \pmb{Q} - \pmb{C}^{(\tau)} \right\|_F^2, \ \ \ \ \hat{\pmb{Q}} = {\pmb{C}^{(s)}}^{-\frac{1}{2}} {\pmb{C}^{(\tau)}}^{\frac{1}{2}}
\tag{15}
$$
其中二阶统计量（协方差矩阵）分别为：
$$
\pmb{C}_k^{(s)} = \dfrac{1}{N_t N_p - 1} \sum_{i=1}^{N_t} \pmb{X}_k^{i,(s)} {\pmb{X}_k^{i,(s)}}^T, \ \ \ \ \pmb{C}^{(\tau)} = \dfrac{1}{N_p - 1} \pmb{X}^{(\tau)} {\pmb{X}^{(\tau)}}^T
\tag{16}
$$
需要注意的是，按照统计学习领域的术语，这种协方差矩阵对齐方法属于“**边缘分布对齐**”，而正统的边缘分布对齐是不考虑类别差异的（考虑类别差异的叫“**条件分布对齐**”），即应有：
$$
\pmb{C}^{(s)} = \dfrac{1}{N_e} \sum_{k=1}^{N_e} \pmb{C}_k^{(s)}
\tag{17}
$$
但是关于这一细节，本文并没有具体说明，换句话说这个 $\pmb{Q}$ 是否需要加下标（$\pmb{Q}_k$）尚不可知。建议有复现代码需求的读者可以测试一下是否有明显差别，本人 Github 上的代码是按照式（17）实现的。综上所述，对于 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$，将其经过 $\pmb{Q}$ 投影处理后理论上可适用于源域数据 $\bar{\pmb{X}}_k^{(s)}$，即有：
$$
\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}^{\rm AC} = \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \pmb{Q}^T
\tag{18}
$$
类似地，将 $\hat{\pmb{W}}_{\pmb{X}^{(s)}}$ 经过 $\pmb{Q}^{-1}$ 投影处理后理论上可适用于目标域数据 $\pmb{X}^{(\tau)}$，即有：
$$
\hat{\pmb{W}}_{\pmb{X}^{(s)}}^{\rm AC} = \hat{\pmb{W}}_{\pmb{X}^{(s)}} \pmb{Q}^{-T}
\tag{19}
$$
再一次地，我们停下来思考一下为什么要执行式（18）与（19）。结合式（15）、式（16）可以发现，线性变换矩阵 $\hat{\pmb{Q}}$ 的作用形式是 $\hat{\pmb{Q}}^T \pmb{X}_k^{i,(s)}$，把这一部分代入到空间滤波过程中（以空间滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$ 为例），则有：
$$
\begin{align}
\notag \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \pmb{X}_k^{(\tau)} &\sim \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \hat{\pmb{Q}}^T \pmb{X}_k^{i,(s)}\\
\notag \ \\
\notag \hat{\pmb{W}}_{\pmb{X}^{(s)}} \pmb{X}_k^{i,(s)} &\sim \hat{\pmb{W}}_{\pmb{X}^{(s)}} \pmb{Q}^{-T} \pmb{X}_k^{(\tau)}
\end{align}
\tag{20}
$$
在 ALPHA 使用场景中，目标域样本是无标签的（没有下标 $k$）。式（20）仅作为一种定性的表述以阐释 AC 的理论假设。由于 domain gap 的存在，目标域与源域的空间滤波器（$\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$、$\hat{\pmb{W}}_{\pmb{X}^{(s)}}$）是不互通的，数据空间信息分布也存在差异。但是通过线性变换后，源域数据（$\hat{\pmb{Q}}^T \pmb{X}_k^{i,(s)}$）的空间信息分布可以更加接近目标域（$\pmb{X}_k^{(\tau)}$）（反之则有 $\pmb{Q}^{-T} \pmb{X}_k^{(\tau)}$ 更接近 $\pmb{X}_k^{i,(s)}$），此时再用目标域的空间滤波器（$\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$）对其进行处理就更合理了。因此与式（14）类似地，投影矩阵 $\pmb{Q}$ 的实际作用对象是源域数据，而不是目标域空间滤波器，式（18）、式（19）只是完整空间滤波过程中的一个切片。

通俗来讲，由目标域数据 $\pmb{X}^{(\tau)}$ 构建的滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$ 想应用到源域数据 $\pmb{X}_k^{i,(s)}$ 上，需要经过投影修饰 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k} \hat{\pmb{Q}}^T$；相反地，源域数据 $\pmb{X}_k^{i,(s)}$ 构建的滤波器 $\hat{\pmb{W}}_{\pmb{X}^{(s)}}$ 想用在目标域数据 $\pmb{X}^{(\tau)}$ 上，也需要经过投影修饰 $\hat{\pmb{W}}_{\pmb{X}^{(s)}} \pmb{Q}^{-T}$。

总结一下，在子空间对齐阶段，ALPHA 构建了 4 种新的空间滤波器，它们的构建过程及适用对象分别如下所示：

（1）通过对目标域正向传播矩阵的旋转变换，对标源域空间传播模式，从而完成目标域空间滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \pmb{Y}_k}$ 的修饰，获得新滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \pmb{Y}_k}^{{\rm ASP}}$，适用于源域模板 $\bar{\pmb{X}}_k^{(s)}$；

（2）通过对目标域正向传播矩阵的旋转变换，对标源域空间传播模式，从而完成目标域空间滤波器 $\hat{\pmb{U}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}}$ 的修饰，获得新滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \bar{\pmb{X}}_k^{(s)}}^{{\rm ASP}}$，适用于源域模板 $\bar{\pmb{X}}_k^{(s)}$；

（3）通过对源域数据的旋转变换 $\hat{\pmb{Q}}^T \pmb{X}_k^{i,(s)}$，对标目标域数据 $\pmb{X}^{(\tau)}$ 的空间分布，从而完成目标域空间滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \pmb{Y}_k}$ 的修饰，获得新滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}^{\rm AC}$，适用于源域模板 $\bar{\pmb{X}}_k^{(s)}$；

（4）通过对目标域数据的旋转变换 $\pmb{Q}^{-T} \pmb{X}_k^{(\tau)}$，对标源域数据 $\pmb{X}_k^{i,(s)}$ 的空间分布，从而完成源域空间滤波器 $\hat{\pmb{W}}_{\pmb{X}^{(s)}}$ 的修饰，获得新滤波器 $\hat{\pmb{W}}_{\pmb{X}^{(s)}}^{\rm AC}$，适用于目标域数据 $\pmb{X}^{(\tau)}$。
 


---
### 子空间池化（Subspace pooling）
到这一步，ALPHA 通过子空间分解获得了 **6** 种（待用）空间滤波器：$\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$、$\hat{\pmb{V}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$、$\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}$、$\hat{\pmb{U}}_{\pmb{X}_k^{i,(s)} \bar{\pmb{X}}_k^{(s)}}$、$\hat{\pmb{U}}_{\pmb{X}_k^{(\tau)} \bar{\pmb{X}}_k^{(s)}}$ 以及 $\hat{\pmb{W}}_{\pmb{X}^{(s)}}$；通过子空间对齐进一步获得了 **4** 种滤波器：$\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \pmb{Y}_k}^{{\rm ASP}}$、$\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \bar{\pmb{X}}_k^{(s)}}^{{\rm ASP}}$、$\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}^{\rm AC}$ 以及 $\hat{\pmb{W}}_{\pmb{X}^{(s)}}^{\rm AC}$。为此，ALPHA 设计了 **5** 种判别系数向量。鉴于变量符号的上下标较为复杂，我们依次来看各个系数：

（1）$\rho_{k,1}^{(s),n}$ 表示面向目标域测试样本 $\pmb{X}^{(\tau)}$ 的标准 CCA 判别过程。其中下标 $k$ 为类别索引（$k = 1, 2, \cdots, N_e$），上标 $n$ 为滤波器子空间维度索引（$n = 1, 2, \cdots, N_k$），$(n,:)$ 表示取滤波器矩阵的第 $n$ 维度向量进行空间滤波。在 5 种判别系数中，这是唯一仅使用目标域数据 $\pmb{X}^{(\tau)}$ 的判别过程，剩余 4 种系数都围绕目标域数据 $\pmb{X}^{(\tau)}$ 与源域模板 $\bar{\pmb{X}}_k^{(s)}$ 的滤波匹配构建。
$$
\rho_{k,1}^{(s),n} = {\rm corr} \left( \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}(n,:) \pmb{X}^{(\tau)}, \ \ \hat{\pmb{V}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}(n,:) \pmb{Y}_k \right)
\tag{21}
$$
（2）$\rho_{k,2}^{(s),n}$ 使用目标域标准 CCA 过程中产生的 EEG 侧滤波器 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}$（下文简称目标域 CCA 滤波器）对目标域数据 $\pmb{X}^{(\tau)}$ 进行滤波，使用 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}^{\rm AC}$ 对源域模板 $\bar{\pmb{X}}_k^{(s)}$ 进行滤波，之后对二者计算 Pearson 相关系数：
$$
\rho_{k,2}^{(s),n} = {\rm corr} \left( \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}(n,:) \pmb{X}^{(\tau)}, \ \ \hat{\pmb{U}}_{\pmb{X}^{(\tau)}\pmb{Y}_k}^{\rm AC}(n,:) \bar{\pmb{X}}_k^{(s)} \right)
\tag{22}
$$
（3）$\rho_{k,3}^{(s),n}$ 使用源域 CCA 滤波器 $\hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}$ 对源域模板 $\bar{\pmb{X}}_k^{(s)}$ 进行滤波，使用 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \pmb{Y}_k}^{{\rm ASP}}$ 对目标域数据 $\pmb{X}^{(\tau)}$ 进行滤波，之后计算相关系数：
$$
\rho_{k,3}^{(s),n} = {\rm corr} \left( \hat{\pmb{U}}_{\pmb{X}^{(\tau)} \pmb{Y}_k}^{{\rm ASP}}(n,:) \pmb{X}^{(\tau)}, \ \ \hat{\pmb{U}}_{\bar{\pmb{X}}_k^{(s)} \pmb{Y}_k}(n,:) \bar{\pmb{X}}_k^{(s)} \right)
\tag{23}
$$
（4）$\rho_{k,4}^{(s),n}$ 使用源域 TDCCA 滤波器 $\hat{\pmb{U}}_{\pmb{X}_k^{i,(s)} \bar{\pmb{X}}_k^{(s)}}$ 对源域模板 $\bar{\pmb{X}}_k^{(s)}$ 进行滤波，使用 $\hat{\pmb{U}}_{\pmb{X}^{(\tau)} \bar{\pmb{X}}_k^{(s)}}^{{\rm ASP}}$ 对目标域数据 $\pmb{X}^{(\tau)}$ 进行滤波，之后计算相关系数：
$$
\rho_{k,4}^{(s),n} = {\rm corr} \left( \hat{\pmb{U}}_{\pmb{X}^{(\tau)} \bar{\pmb{X}}_k^{(s)}}^{{\rm ASP}}(n,:) \pmb{X}^{(\tau)}, \ \ \hat{\pmb{U}}_{\pmb{X}_k^{i,(s)} \bar{\pmb{X}}_k^{(s)}}(n,:) \bar{\pmb{X}}_k^{(s)} \right)
\tag{24}
$$
（5）$\rho_{k,5}^{(s),n}$ 使用源域 DSP 滤波器 $\hat{\pmb{W}}_{\pmb{X}^{(s)}}$ 对源域模板 $\bar{\pmb{X}}_k^{(s)}$ 进行滤波，使用 $\hat{\pmb{W}}_{\pmb{X}^{(s)}}^{\rm AC}$ 对目标域数据 $\pmb{X}^{(\tau)}$ 进行滤波，之后计算相关系数：
$$
\rho_{k,5}^{(s),n} = {\rm corr} \left( \hat{\pmb{W}}_{\pmb{X}^{(s)}}^{\rm AC}(n,:) \pmb{X}^{(\tau)}, \ \ \hat{\pmb{W}}_{\pmb{X}^{(s)}}(n,:) \bar{\pmb{X}}_k^{(s)} \right)
\tag{25}
$$
最后组合 5 种系数向量 $\pmb{\rho}_{k,j}^{(s)}$ 构成总判别系数向量 $\pmb{\rho}_{k}^{(s)}$：
$$
\begin{align}
\notag \pmb{\rho}_{k,j}^{(s)} &=
\begin{bmatrix}
\rho_{k,j}^{(s),1} & \rho_{k,j}^{(s),2} & \cdots & \rho_{k,j}^{(s),N_k}
\end{bmatrix}, \ \ \ \ j = 1, 2, \cdots, 5\\
\notag \ \\
\notag \pmb{\rho}_{k}^{(s)} &= \sum_{j=1}^{5} \left[ {\rm sign} \left( \pmb{\rho}_{k,j}^{(s)} \right) \odot \pmb{\rho}_{k,j}^{(s)} \odot \pmb{\rho}_{k,j}^{(s)} \right] \in \mathbb{R}^{1 \times N_k}
\end{align}
\tag{26}
$$
其中 $\odot$ 表示哈达玛积（Hadamard product），$\odot$ 的作用为两个同形状矩阵（向量）的各元素分别相乘，形成一个同形状的新矩阵（向量），其本质就是把以前 eCCA 中用过的系数组合方法扩展到向量 $\pmb{\rho}_{k,j}^{(s)}$ 上，压缩系数种类维度（5），保留子空间维度（$N_k$）。

接下来开始说明子空间维度的池化步骤。目前判别系数中存在的子空间维度（$N_k$）需要通过合理方式进行压缩，ALPHA 为之设计了一种别出心裁的方法：在源域数据集中执行留一法（Leave-one-block-out）交叉验证，根据分类结果计算面向子空间维度的最佳投影向量 $\pmb{\omega}_k^{(s)} \in \mathbb{R}^{N_k \times 1}$。

具体来说，以 Benchmark 数据集为例，每个受试者完整数据的有效部分维度为 $(40,6,64,1500)$，即 40 类别、每类 6 个样本（block）、共 64 个导联（不一定全用）、采样点数为 1500（不一定全用）。在源域数据集中，每次交叉验证使用 5 个 block 作为训练集（即共计 200 个样本），剩余 1 个 block 作为验证集（共计 40 个样本），并执行一遍上述的子空间分解、对齐以及判别系数计算过程，每次交叉验证应获得维度为 $\mathbb{R}^{40 \times N_k \times 40}$ 的判别系数张量（样本数 $\times$ 子空间个数 $\times$ 类别数）。交叉验证（6 折）结束后，在样本数维度上拼接各轮验证结果，并获得总判别系数张量 $\pmb{\mathcal{R}}^{(s)} \in \mathbb{R}^{240 \times N_k \times 40}$。

接下来针对 $\pmb{\mathcal{R}}^{(s)}$ 中每一类的信息求解投影向量 $\pmb{\omega}_k^{(s)}$。ALPHA 原文设计了一种 selecting matrix，它的主要作用就是在 $\pmb{\rho}^{(s)}$ 中筛选出某一类样本数据在面向当前类别模板时的判别系数矩阵，但原文公式（eq.18）的数学描述比较容易让人迷惑，实际编程也用不上这个表达式，因此接下来我基于自己的理解重新描述一遍后续步骤：ALPHA 需要利用投影向量 $\pmb{\omega}_k$ 最大化第 $k$ 类测试样本的第 $k$ 组判别系数。例如对于真实标签为第 1 类的验证样本（源域数据所有样本真实标签都是已知的），它们计算得到的判别系数张量维度应为 $\mathbb{R}^{6 \times N_k \times 40}$，从中（最后一维）提取到预测为第 1 类时的判别系数矩阵并记为 $\pmb{\mathcal{R}}_{1}^{(s)} \in \mathbb{R}^{6 \times N_k}$，此时寻找向量 $\pmb{\omega}_1 \in \mathbb{R}^{N_k \times 1}$，使得 $\pmb{\mathcal{R}}_{1}^{(s)} \pmb{\omega}_1$ 的 Frobenius 范数（综合了不同交叉验证轮次信息）最大化。将以上步骤拓展到全体类别，则有：
$$
\hat{\pmb{\omega}}_k = \underset{\pmb{\omega}_k}{\arg\max} \left\| \pmb{\mathcal{R}}_{k}^{(s)} \pmb{\omega}_k \right\|_F^2, \ \ \ \ {\pmb{\omega}_k}^T \pmb{\omega}_k = 1
\tag{27}
$$
我们来完整描述一遍 ALPHA 算法的应用流程：对于一个未知类别的目标域测试样本 $\pmb{X}^{(\tau)}$，选定源域受试者 $s$，依次进行子空间分解（得到 5 个空间滤波器）、子空间对齐（再得到 4 个空间滤波器）、子空间池化步骤至式（26）时，对于每一个预测类别，都会得到一个系数向量 $\pmb{\rho}_k^{(s)}$，到这为止都是分类算法的常规流程。这里存在一个细节，很容易被新手朋友们忽视：由于测试样本类别是未知的，我们需要首先预设它属于某一类（假定为 $k$），然后使用第 $k$ 类训练数据模型的空间滤波器，与第 $k$ 类模板进行匹配，获得第 $k$ 个判别系数，此时是不会与其它类的模板进行匹配的。假如这个样本真的属于第 $k$ 类（而且数据质量不错），那么第 $k$ 类系数（靶系数）的值就会比第 $k-1$ 个、第 $k+1$ 个等系数（非靶系数）都大。根据这种定义，我们可以**划分**出 $N_e$ 种靶系数向量（即 $\pmb{\rho}_{k}^{(s)}$）以及各自对应的非靶系数（记为 $\pmb{\rho}_{k,{\rm ntar}}^{(s)}$），但是判别系数的总数据依然是那 $N_e$ 个 $\pmb{\rho}_k^{(s)}$，并不需要额外计算。ALPHA 输出的预测类别 $\hat{k}$ 如下所示：
$$
\hat{k} = \underset{k}{\arg\max} \left[ \left( \pmb{\rho}_{k}^{(s)} - \pmb{\rho}_{k,{\rm, ntar}}^{(s)} \right) \pmb{\omega}_k \right]
\tag{28}
$$

[alpha]: https://ieeexplore.ieee.org/document/9516951/
[Neuroimage 文献]: https://linkinghub.elsevier.com/retrieve/pii/S1053811913010914