# 多重刺激 TRCA
## Multi-stimulus (e)TRCA

[论文链接][ms-TRCA]

该算法是我们提到的第一种关于TRCA的改进算法，与 msCCA、ms-eCCA 一样出自 Wong 的之手，行文风格也一以贯之。由 TRCA 节末尾分析可知，该类算法需要较多的训练数据以保证协方差矩阵估计的准确性，在原文中亦有相关描述：
> To guarantee good performance, the number of calibration trials for each visual stimulus cannot be small for the eCCA method and the eTRCA method; otherwise, their recognition accuracies would decrease dramatically $\dots$ A major reason is that the estimate of covariance matrices in the CCA-based (or the TRCA-based) methods could become unreliable in the case of small training data so that the resulting spatial filters may not be accurate. As a matter of fact, the algorithms relying on the estimator of covariance matrix also suffer from insufficient training data problem $\dots$

然而对于具有较多指令集（32、40 或更多）的 SSVEP-BCI 系统而言，为每个指令获取足量训练试次（$N_t>10$）在时间上通常是不允许的。而且刺激时间如果过长，诱发信号的质量会由于受试者累积的视觉疲劳而急剧下降，不利于系统实际应用。因此，Wong 提出了一种学习方法，从目标 $A$ 的相邻刺激（$B$、$C$ 等）的诱发信号中学习 $A$ 的特征。以一个简单的 4 指令集系统为例，ms- 技术的学习流程见下图（原文 fig.1）：

![ms-eTRCA示意图](/figures/ms-eTRCA_fg1.png)

原文对于这一套流程的可行性做了连篇累牍的证明，在此仅提炼其最终结论：

**去除 0.14 s 的人眼视觉通路延迟之后，SSVEP 采集信号与闪烁刺激相位基本一致、不同刺激间的相位差也与闪烁刺激相位差基本一致**。

听君一席话，如听一席话是吧？这不是 SSVEP 信号处理的常规步骤吗？此外关于 Benchmark 数据集的刺激参数设定（频率、相位等），他们也特地写了三个**方程**来说明，总之一通操作着实看得人头昏脑胀。鉴于本人有幸曾经听过 Wong 的线上报告，视频里他只消三言两语便把 ms-eTRCA 的原理描述得鞭辟入里，所以我相信这种反人类的数学语言其实出自 Feng Wan 老师的手笔。

接下来我们先看看原文中给出的公式。ms-eTRCA 与 ms-TRCA 的关系类似 eTRCA 之于 TRCA，因此前两者的目标函数是共通的：
$$
\hat{\pmb{w}}_k = \underset{\pmb{w}_k} \argmax 
\dfrac{\pmb{w}_k \pmb{A}_k {\pmb{A}_k}^T {\pmb{w}_k}^T} {\pmb{w}_k \pmb{B}_k {\pmb{B}_k}^T {\pmb{w}_k}^T}
\tag{1}
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
\tag{2}
$$
$$
\pmb{\chi}_{k} = 
\begin{bmatrix}
\pmb{X}_k^1 & \pmb{X}_k^2 & \cdots & \pmb{X}_k^{N_t}
\end{bmatrix} \in \mathbb{R}^{N_c \times (N_t N_p)}
\tag{3}
$$
可以看出，multi-stimulus 技术的本质就是把目标刺激前 $m$、后 $n$ 个不同频率的刺激信号**顺次拼接**起来，在时间维度上对训练数据进行扩增，其范围为 $d$（含自身），具体拼接个数（$m$、$n$）依不同情况各有一番规定。

（1）关于 $m$ 与 $n$ 的大小分配（一般情况）：若 $d$ 为奇数，则前向与后向扩增范围相等；若 $d$ 为偶数，则前向扩增比后向多一位，即有：
$$
\begin{cases}
m = n = \dfrac{1} {2} (d-1), \ \ d=2n+1 | n \in \mathbb{N^+}\\
\\
m = \dfrac{1} {2} d, \ n = \dfrac{1} {2} d - 1, \ \ d=2n | n \in \mathbb{N^+}\\
\end{cases}
\tag{4}
$$

（2）假设刺激目标 $k$ 处于相对**靠前**的位置，由于前向没有足够类别的信号用于拼接，因此需向后顺延扩增位数。例如 $d=5,k=2$，应向后顺延一位（$m=1,n=3$）；若 $d=6,k=2$，则向后顺延两位（$m=1,n=4$）。综上可总结出式 (3-2-7)：
$$
\begin{cases}
m = k - 1\\
\\
n = d - k\\
\end{cases}, \ k \in \left[ 1, \ \dfrac{1} {2} d \right]
\tag{5}
$$
（3）假设刺激目标 $k$ 处于**中部**位置，即 (1) 中所述的“一般情况”，则有式 (3-2-8)：
$$
\begin{cases}
m = \left[\dfrac{1}{2}d\right]\\
\\
n = d - \left[\dfrac{1}{2}d\right] - 1
\end{cases}, \ k \in \left( \left[ \dfrac{1}{2}d \right], \ N_e-\left(d-\left[\dfrac{1}{2}d\right]\right)\right)
\tag{6}
$$
（4）假设刺激目标 $k$ 位于**尾部**位置，此时与（2）相反，需向前顺延扩增位数，即有式 (3-2-9)：
$$
\begin{cases}
m = d - 1 - \left(N_e - k \right)\\
\\
n = N_e - k\\
\end{cases},  \ k \in \left[N_e - \left(d - \left[\dfrac{1}{2} d \right] - 1 \right), N_e \right]
\tag{7}
$$
好了，我们再回过头去看式 (3-2-3)，该式分子、分母中的协方差矩阵可以通过与式 (3-1-5) 联动，进一步改写为如下形式：
$$
\begin{cases}
\pmb{A}_k {\pmb{A}_k}^T = \sum_{i=-m}^{n+1} \bar{\pmb{X}}_{k+i} {\bar{\pmb{X}}_{k+i}}^T = \dfrac{1}{{N_t}^2} \sum_{i=-m}^{n+1} {\pmb{S}_{k+i}^{'}}\\
\\
\pmb{B}_k {\pmb{B}_k}^T = \sum_{i=-m}^{n+1} \sum_{j=1}^{N_t} \pmb{X}_i^j {\pmb{X}_i^j}^T = \sum_{i=-m}^{n+1} \pmb{Q}_{k+i}
\end{cases}
\tag{8}
$$
一般来说，非零常系数是不影响矩阵特征值分解结果的。所以我们看 ms-(e)TRCA 的目标函数式，它就是把不同频率信号对应的 (e)TRCA 目标函数的分子、分母各自相加组成新的分式。再直白一点，就是“**把多个频率的信号当一个频率去训练**”，强行增加了可用样本数目。
$$
\hat{\pmb{w}}_k = 
\underset{\pmb{w}_k} \argmax 
\dfrac{\pmb{w}_k \left(\sum_{i=-m}^{n+1} \bar{\pmb{X}}_{k+i} {\bar{\pmb{X}}_{k+i}}^T \right) {\pmb{w}_k}^T} {\pmb{w}_k \left(\sum_{i=-m}^{n+1} \sum_{j=1}^{N_t} \pmb{X}_i^j {\pmb{X}_i^j}^T \right) {\pmb{w}_k}^T} = 
\underset{\pmb{w}_k} \argmax 
\dfrac{\sum_{i=-m}^{n+1} \pmb{w}_k {\pmb{S}_{k+i}^{'}} {\pmb{w}_k}^T} {\sum_{i=-m}^{n+1} \pmb{w}_k \pmb{Q}_{k+i} {\pmb{w}_k}^T}
\tag{9}
$$

我们有一点需要注意，根据文章里网格筛选的结果（原文 Fig.3）， $d$ 的范围并非是越大越好，在 (e)TRCA 算法上体现得尤为明显。

![ms-etrca筛选结果](/figures/ms-eTRCA-net.png)

说来也令人感慨，ms- 的思路不可谓不简单，但是 *Wong* 等人之所以成功，一方面是因为敢想敢做，另一方面也要归功于砌墙的数学功底，能够把简单的内核包装成高大上的复杂操作，让人一眼完全看不透其内在关联。

---
[ms-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373