# 基于数据域泛化的跨个体迁移学习
## Cross-Subject Transfer Method Based on Domain Generalization
***

[论文链接][CSTMBDG]

这次我们来学习一篇产自西安电子科技大学的文章，除了数学公式的符号略有些不明所以之外（但是慢慢看还是能看懂的，比 TDCA 那篇好一些），我认为他们的工作量压缩在短短的一篇 TNSRE 属实是有些明珠暗投（~~不会跟 TL-TRCA 一样又是调参怪吧~~）。另外文章中存在一些诡异的细节，更进一步加深了我的怀疑。不过在测试之前，总的来看他们的工作量还是相当充实的，算法设计理念也令人耳目一新。

除去 Domain Generalization 的部分不提，本文设计的个体模型训练方法（详见 TL-TRCA 章节我对迁移学习算法框架的描述）在理念上与 ms-TRCA 比较相似，都是利用了多个邻近目标对注视目标数据进行补充和优化。但是 ms-TRCA 对于“邻近”的定义是基于刺激频率的，而本文的定义是基于刺激实际空间位置的。例如下图所示的经典低频 40 指令集 JFPM 编码范式：

![40指令JFPM编码示意图](/SSVEP_algorithms/数据增强%20&%20迁移学习算法/20230903.png)

假设邻近目标数目设定为 4，按照 ms-TRCA 的思路，9.2 Hz 刺激的“邻近目标”是 8.8 Hz、9.0 Hz、9.4 Hz 以及 9.6 Hz。而本文设计的“邻近目标”为 9.0 Hz、8.2 Hz、9.4 Hz 以及 10.2 Hz。非常惭愧的是，本人在一年前（2022年 9 月）也曾想到这样的设计，当时对比的结果是与 ms-TRCA 差别不大（自采数据集，样本较多），因此未能继续深入探索。该团队的一项针对公开数据集的研究表明，这种差别其实还是存在的。我们会在另一篇简短的专栏中看到这项研究。

![邻近目标定义](/SSVEP_algorithms/数据增强%20&%20迁移学习算法/20230925-1.png)

本文在此基础上设计了一套无监督迁移学习算法，通过一系列操作获得了三种无监督空间滤波器以及两种基于源域真实数据训练获得的迁移模板，在 UCSD 数据集、Benchmark 数据集以及一个自采数据集上都取得了亮眼的效果。记源受试者个数为 $N_s$（索引 $s$）、数据类别总数为 $N_e$（无类别缺失情况，索引 $k$），源域个体可用训练样本数目为 $N_t^{(s)}$（索引 $i$）、可用导联总数为 $N_c$、单试次采样点数为 $N_p$。第 $s$ 号源受试者的第 $k$ 类别、第 $i$ 试次多导联数据记为 $\pmb{X}_k^{i,(s)} \in \mathbb{R}^{N_c \times N_p}$。将每一个目标试次 $i$ 视为未知的“测试数据”，源受试者的其余试次（$N_t^{(s)}-1$ ）被用作“训练数据”，因此本文设计的算法步骤将以“留一法”的方式多次执行，并通过叠加平均获得最终模型。根据前述说明，$\pmb{X}_k^{i,(s)}$ 的周边区域存在 $N_n$ 个邻近单元数据，以 $n$ 为索引记为 $\pmb{X}_{k,(n)}^{i,(s)} \in \mathbb{R}^{N_c \times N_p}$。

接下来正式进入算法公式的说明部分。首先是算法第一步：
> Learning the **internally-invariant** spatial filter and template for each source subject to extract common frequency information across neighboring stimuli.

原文中的 Internally-invariant spatial filter 可以理解为受试者个体内、周边区域内通用的空间滤波器（类似 ms-(e)TRCA）。第 $k$ 目标、第 $i$ 试次的训练数据集 $\left\{ \pmb{X}_k^{i_r,(s)} \right\}$（排除了第 $i$ 试次的其余试次数据）与相同试次索引、第 $n$ 个邻近目标的训练数据集 $\left\{ \pmb{X}_{k,(n)}^{i_r,(s)} \right\}$ 分别为：
$$
    \left\{ \pmb{X}_k^{i_r,(s)} \right\} = \left\{ \pmb{X}_k^{j,(s)} \ | \ j \in [1,N_t^{(s)}], j \ne i \right\}, \ \ \ \left\{ \pmb{X}_{k,(n)}^{i_r,(s)} \right\} = \left\{ \pmb{X}_{k,(n)}^{j,(s)} \ | \ j \in [1,N_t^{(s)}], j \ne i \right\}
    \tag{1}
$$
对照原文的示意图应该更好理解：本节专栏中的 $\left\{ \pmb{X}_k^{i_r,(s)} \right\}$ 对应于图中的 $\pmb{\mathcal{X}}_n^b$（$k$ 对应 $n$，$i_r$ 对应 $b$），$\left\{ \pmb{X}_{k,(n)}^{i_r,(s)} \right\}$ 对应图中的 $\pmb{\mathcal{X}}_{n^h}^b$（$n$ 对应 $h$）。

![数据划分示意图](/SSVEP_algorithms/数据增强%20&%20迁移学习算法/20230903-2.png)

之所以不使用原文的字母标记，一是因为不符合我个人的标记习惯，二是我认为文章的字符标记略显随意，没有与英文单词首字母对应，后面还更换过一次标记方法。当然字符标记都是细枝末节的小问题，我个人认为这里牵涉到一个更为关键的问题：**为什么邻近目标的试次索引要与注视目标保持一致**？**为什么不同受试者的试次索引也要保持一致**？这些数据明明在时间上是完全隔离的呀？而且如果经过前期数据清洗后，各类别数据的样本容量不一致怎么办？本文使用的公开数据集是包装完好的，能够避免这个问题，但依然无法解释索引一致的奇怪要求。这样的问题居然没有审稿人指出，文章也没有据此进行讨论，属实诡异。按原文描述，在 internally-invariant 训练阶段，$\left\{ \pmb{X}_k^{i_r,(s)} \right\}$ 与 $\left\{ \pmb{X}_{k,(n)}^{i_r,(s)} \right\}$ 整合构成面向 $\pmb{X}_k^{i,(s)}$ 的训练数据集，按照 ms-TRCA 的方式构建目标函数：
$$
    \bar{\pmb{X}}_k^{i_r,(s)} = \dfrac{1}{N_t^{(s)}-1} \sum_{j=1,j \ne i}^{N_t^{(s)}} \pmb{X}_k^{j,(s)}, \ \ \ \bar{\pmb{X}}_{k,(n)}^{i_r,(s)} = \dfrac{1}{N_t^{(s)}-1} \sum_{j=1,j \ne i}^{N_t^{(s)}} \pmb{X}_{k,(n)}^{j,(s)}
    \tag{2}
$$
$$
    \begin{align}
        \notag
        \pmb{S}_k^{i_r,(s)} &= \bar{\pmb{X}}_k^{i_r,(s)} {\bar{\pmb{X}}_k^{i_r,(s)}}^T + \sum_{n=1}^{N_n} \bar{\pmb{X}}_{k,(n)}^{i_r,(s)} {\bar{\pmb{X}}_{k,(n)}^{i_r,(s)}}^T\\
        \notag \ \\
        \notag
        \pmb{Q}_k^{i_r,(s)} &= \sum_{j=1,j \ne i}^{N_t^{(s)}} \pmb{X}_k^{j,(s)} {\pmb{X}_k^{j,(s)}}^T + \sum_{n=1}^{N_n} \sum_{j=1,j \ne i}^{N_t^{(s)}} \pmb{X}_{k,(n)}^{j,(s)} {\pmb{X}_{k,(n)}^{j,(s)}}^T
    \end{align}
    \tag{3}
$$
$$
    \hat{\pmb{w}}_k^{i_r,(s)} = \underset{\pmb{w}_k^{i_r,(s)}} \argmax \dfrac{\pmb{w}_k^{i_r,(s)} \pmb{S}_k^{i_r,(s)} {\pmb{w}_k^{i_r,(s)}}^T} {\pmb{w}_k^{i_r,(s)} \pmb{Q}_k^{i_r,(s)} {\pmb{w}_k^{i_r,(s)}}^T}
    \tag{4}
$$
获得空间滤波器 $\hat{\pmb{w}}_k^{i_r,(s)} \in \mathbb{R}^{1 \times N_c}$ 之后，对第 $i$ 试次训练数据集 $\left\{ \pmb{X}_k^{i_r,(s)} \right\}$ 的叠加平均结果计算 internally-invariant 模板 $\pmb{T}_k^{i_r,(s)}$：
$$
    \pmb{T}_k^{i_r,(s)} = \hat{\pmb{w}}_k^{i_r,(s)} \bar{\pmb{X}}_k^{i_r,(s)} \in \mathbb{R}^{1 \times N_p}
    \tag{5}
$$
这些公式看着很复杂，但其实我们只需理清一个关键思路：ms-TRCA 干了什么？它把对应不同类别索引的数据协方差矩阵加在一起，计算得到的空间滤波器给其中某一个类别用，表面上是扩增了目标类别的训练样本量，本质上是挪用了其它类别数据的相关信息。本文在 internally-invariant 阶段所做的事与 ms-TRCA 基本一致，唯二的区别在于：

（1）ms-TRCA 划分类别组的依据是索引顺序，本文算法的依据是空间位置，最后叠加了 $N_n +1$ 个类别的数据协方差矩阵；

（2）ms-TRCA 把能用的训练试次全部用上，做一锤子买卖，本文算法根据留一法思想，每次剔除一个试次 $i$，该试次索引通用于目标类别和周边类别数据集，剩下的数据（上标 $i_r$）训练模型，最后叠加平均多个轮次的训练结果。

接下来是第二步工作：

> Calculating the **mutually-invariant** spatial filter and template from all source subjects to learn common knowledge shared across subjects.

文中提到的 mutually-invariant 空间滤波器是一种面向源域全体受试者的、单类别的个体间相关性最大化处理方法，其构建思路是使得数据的个体间协方差最大化、个体内方差最小化，可以理解为 TRCA 模型在个体间层面的延续。个体间协方差矩阵每次只计算两位受试者（共计 $C_{N_s}^2$ 次）：
$$
    \pmb{P}_k^{i_r} = \dfrac{1}{N_s(N_s-1)} \sum_{s_1=1}^{N_s} \sum_{s_2=1, \ s_2 \ne s_1}^{N_s} \left( \bar{\pmb{X}}_k^{i_r,(s_1)} {\bar{\pmb{X}}_k^{i_r,(s_2)}}^T + \sum_{n=1}^{N_n} \bar{\pmb{X}}_{k,(n)}^{i_r,(s_1)} {\bar{\pmb{X}}_{k,(n)}^{i_r,(s_2)}}^T \right)
    \tag{6}
$$
$$
    \pmb{R}_k^{i_r} = \dfrac{1}{N_s} \sum_{s=1}^{N_s} \left( \bar{\pmb{X}}_k^{i_r,(s)} {\bar{\pmb{X}}_k^{i_r,(s)}}^T + \sum_{n=1}^{N_n} \bar{\pmb{X}}_{k,(n)}^{i_r,(s)} {\bar{\pmb{X}}_{k,(n)}^{i_r,(s)}}^T \right)
    \tag{7}
$$
类似 TRCA 公式的化简步骤，令 $\bar{\pmb{X}}_k^{i_r,a}=\dfrac{1}{N_s} \sum_{s=1}^{N_s} \bar{\pmb{X}}_k^{i_r,(s)}$、$\bar{\pmb{X}}_{k,(n)}^{i_r,a}=\dfrac{1}{N_s} \sum_{s=1}^{N_s} \bar{\pmb{X}}_{k,(n)}^{i_r,(s)}$，则有：
$$
    \begin{align}
        \notag
        \pmb{P}_k^{i_r} &= \dfrac{1}{N_s(N_s-1)} \left[ \left( N_s^2 \bar{\pmb{X}}_k^{i_r,a} {\bar{\pmb{X}}_k^{i_r,a}}^T - \sum_{s=1}^{N_s} \bar{\pmb{X}}_k^{i_r,(s)} {\bar{\pmb{X}}_k^{i_r,(s)}}^T \right) + \sum_{n=1}^{N_n} \left( N_s^2 \bar{\pmb{X}}_{k,(n)}^{i_r,a} {\bar{\pmb{X}}_{k,(n)}^{i_r,a}}^T - \sum_{s=1}^{N_s} \bar{\pmb{X}}_{k,(n)}^{i_r,(s)} {\bar{\pmb{X}}_{k,(n)}^{i_r,(s)}}^T \right) \right]\\
        \notag \ \\
        \notag
        &= \dfrac{1}{N_s-1} \left( N_s \bar{\pmb{X}}_k^{i_r,a} {\bar{\pmb{X}}_k^{i_r,a}}^T + \sum_{n=1}^{N_n} N_s \bar{\pmb{X}}_{k,(n)}^{i_r,a} {\bar{\pmb{X}}_{k,(n)}^{i_r,a}}^T - \pmb{R}_k^{i_r} \right)\\
        \notag \ \\
        \notag
        &= \dfrac{N_s}{N_s-1} \left( \bar{\pmb{X}}_k^{i_r,a} {\bar{\pmb{X}}_k^{i_r,a}}^T + \sum_{n=1}^{N_n}\bar{\pmb{X}}_{k,(n)}^{i_r,a} {\bar{\pmb{X}}_{k,(n)}^{i_r,a}}^T \right) - \dfrac{1}{N_s-1} \pmb{R}_k^{i_r}
    \end{align}
    \tag{8}
$$
mutually-invariant 空间滤波器 $\hat{\pmb{v}}_k^{i_r} \in \mathbb{R}^{1 \times N_c}$ 的目标函数为：
$$
    \begin{align}
        \notag
        \hat{\pmb{v}}_k^{i_r} &= \underset{\pmb{v}_k^{i_r}} \argmax \dfrac{\pmb{v}_k^{i_r} \pmb{P}_k^{i_r} {\pmb{v}_k^{i_r}}^T} {\pmb{v}_k^{i_r} \pmb{R}_k^{i_r} {\pmb{v}_k^{i_r}}^T}\\
        \notag \ \\
        \notag
        &= \underset{\pmb{v}_k^{i_r}} \argmax \left[ \dfrac{N_s} {N_s-1} \dfrac{\pmb{v}_k^{i_r} \left( \bar{\pmb{X}}_k^{i_r,a} {\bar{\pmb{X}}_k^{i_r,a}}^T + \sum_{n=1}^{N_n}\bar{\pmb{X}}_{k,(n)}^{i_r,a} {\bar{\pmb{X}}_{k,(n)}^{i_r,a}}^T \right) {\pmb{v}_k^{i_r}}^T} {\pmb{v}_k^{i_r} \pmb{R}_k^{i_r} {\pmb{v}_k^{i_r}}^T} - \dfrac{1}{N_s-1} \right]\\
        \notag \ \\
        \notag
        &= \underset{\pmb{v}_k^{i_r}} \argmax \dfrac{\pmb{v}_k^{i_r} \left( \bar{\pmb{X}}_k^{i_r,a} {\bar{\pmb{X}}_k^{i_r,a}}^T + \sum_{n=1}^{N_n}\bar{\pmb{X}}_{k,(n)}^{i_r,a} {\bar{\pmb{X}}_{k,(n)}^{i_r,a}}^T \right) {\pmb{v}_k^{i_r}}^T} {\pmb{v}_k^{i_r} \pmb{R}_k^{i_r} {\pmb{v}_k^{i_r}}^T}
    \end{align}
    \tag{9}
$$
与第一步和接下来描述的第三步不同，第二步只需要计算 $N_e \times N_t$ 次（变量上下标只有 $k$、$i_r$，没有 $(s)$），而其它两步都需要结合具体的源受试者，即多次执行。因此在编程实践中，优先进行第二步运算（一步到位），之后再进行一、三两步的循环。$\hat{\pmb{v}}_k^{i_r}$ 是全体源受试者通用、类别不通用的空间滤波器，因此对目标类别 $k$ 的多个体平均模板进行空间滤波，获得 mutually-invariant 模板 $\pmb{Z}_k^{i_r,a}$：
$$
    \pmb{Z}_k^{i_r,a} = \hat{\pmb{v}}_k^{i_r} \bar{\pmb{X}}_k^{i_r,a} \in \mathbb{R}^{1 \times N_p}
    \tag{10}
$$
最后是第三步工作：

> Training a **test-trial** spatial filter to improve the SNR of test-trial data by incorporating the internally- and mutually-invariant templates.

第三步设计的 test-trial 空间滤波器 $\hat{\pmb{u}}_k^{i,(s)} \in \mathbb{R}^{1 \times N_c}$ 用来处理最初被挑出来的试次 $\pmb{X}_k^{i,(s)}$（注意此处变量的上标是 $i$ 而不是 $i_r$，因为作用对象是 $\pmb{X}_k^{i,(s)}$ 而不是 $\left\{ \pmb{X}_k^{i_r,(s)} \right\}$）。此处该团队描述了一个有约束多目标最优化问题：
$$
    \begin{align}
        \notag
        \mathcal{F} \left( \pmb{u}_k^{i,(s)} \right) &= 
        \begin{bmatrix}
        \rho \left( \pmb{u}_k^{i,(s)} \pmb{X}_k^{i,(s)}, \pmb{T}_k^{i_r,(s)} \right)\\
        \ \\
        \rho \left( \pmb{u}_k^{i,(s)} \pmb{X}_k^{i,(s)}, \pmb{Z}_k^{i_r,a} \right)
        \end{bmatrix}\\
        \notag \ \\
        \notag
        \hat{\pmb{u}}_k^{i,(s)} &=
        \underset{\pmb{u}_k^{i,(s)}} \argmax \mathcal{F} \left( \pmb{u}_k^{i,(s)} \right), \ \ s.t. \ \sum_{c=1}^{N_c} \pmb{u}_k^{i,(s)}(:,c) = 0
    \end{align}
    \tag{11}
$$
解决这种 multi-objective optimization 问题有很多种方法，本文使用的是基于 MATLAB 软件的 fgoalattain 函数。根据我有限的调研，Python 环境下解决该类问题可以使用 Pymoo 或者 SciPy 的 optimize 库，但是具体使用什么寻优算法或策略我还没有复现成功。

根据前述可知，$\hat{\pmb{u}}_k^{i,(s)}$、$\pmb{Z}_k^{i_r,a}$、$\hat{\pmb{v}}_k^{i_r}$、$\pmb{T}_k^{i_r,(s)}$、$\hat{\pmb{w}}_k^{i_r,(s)}$ 都只是“留一法”中的一小步，在经过 $N_t$ 次建模运算之后，通过叠加平均获得面向源受试者 $s$ 的、第 $k$ 类别数据的迁移学习模型：
$$
    \begin{align}
        \notag
        \bar{\pmb{w}}_k^{(s)} &= \dfrac{1}{N_t} \sum_{i=1}^{N_t} \hat{\pmb{w}}_k^{i_r,(s)}, \ \ \bar{\pmb{T}}_k^{(s)} = \dfrac{1}{N_t} \sum_{i=1}^{N_t} \pmb{T}_k^{i_r,(s)}\\
        \notag \ \\
        \notag
        \bar{\pmb{v}}_k &= \dfrac{1}{N_t} \sum_{i=1}^{N_t} \hat{\pmb{v}}_k^{i_r}, \ \ \bar{\pmb{Z}}_k = \dfrac{1}{N_t} \sum_{i=1}^{N_t} \pmb{Z}_k^{i_r,a}\\
        \notag \ \\
        \notag
        \bar{\pmb{u}}_k^{(s)} &= \dfrac{1}{N_t} \sum_{i=1}^{N_t} \hat{\pmb{u}}_k^{i,(s)}
    \end{align}
    \tag{12}
$$
在无监督源域模型训练完成后，本文利用上述三类空间滤波器、两类模板设计了一种包含四个判别系数的特征融合方法，对于目标域受试者未知类别的测试数据 $\pmb{\mathcal{X}} \in \mathbb{R}^{N_c \times N_p}$：
$$
    \begin{align}
        \notag
        \rho_{k,1} &= {\rm corr} \left( \bar{\pmb{v}}_k \pmb{\mathcal{X}}, \ \bar{\pmb{Z}}_k \right)\\
        \notag
        \rho_{k,2} &= \dfrac{1}{N_s} \sum_{s=1}^{N_s} {\rm corr} \left( \bar{\pmb{w}}_k^{(s)} \pmb{\mathcal{X}}, \ \bar{\pmb{T}}_k^{(s)} \right)\\
        \notag
        \rho_{k,3} &= \dfrac{1}{N_s} \sum_{s=1}^{N_s} {\rm corr} \left( \bar{\pmb{u}}_k^{(s)} \pmb{\mathcal{X}}, \ \bar{\pmb{Z}}_k \right)\\
        \notag
        \rho_{k,4} &= \dfrac{1}{N_s} \sum_{s=1}^{N_s} {\rm corr} \left( \bar{\pmb{u}}_k^{(s)} \pmb{\mathcal{X}}, \ \bar{\pmb{T}}_k^{(s)} \right)\\
    \end{align}, \ \ \ \ \rho_k = \sum_{i=1}^{4} {\rm sign} \left( \rho_{k,i} \right) \rho_{k,i}^2, \ \ \ \ \hat{k} = \underset{k} \argmax \rho_k
    \tag{13}
$$
由于本文是一种源域有监督建模 + 目标域无监督应用的无监督迁移学习算法，所以在源域上用什么模型对以上策略构成的算法最终性能起到了关键作用。本文特地对比了在三个数据集上，源域使用 TRCA、ms-TRCA、TDCA 模型、目标域使用 CCA 策略的迁移学习方法与本研究方法的性能差异，结果均是本研究方法占优。一方面说明了本文设计的空间滤波器与模板的数据泛化性更强，另一方面说明了本文设计的四特征融合方法优于普通 CCA 的双特征融合。另外，文章还对比了去除单个特征与全体融合特征的分类效果差异，结果表明 $\rho_{k,1}$ 对分类结果的影响最大，即 mutually-invariant 模型对于无监督迁移学习的作用最好；$\rho_{k,4}$ 对分类结果的影响相对最小，但综合来看，去掉任一系数的结果与其它结果差别并不大，融合四个系数能够达到最佳效果。

[CSTMBDG]: https://ieeexplore.ieee.org/document/10216996/