## 1.5 多重刺激扩展 CCA：ms-eCCA
**(Multi-stimulus extened CCA)**

**[论文链接][ms-TRCA] | 代码：[cca][cca(code)].msecca()**

顾名思义，ms-eCCA 是 msCCA 的扩展，在模板匹配上类似 eCCA 之于 CCA，此外在扩增的刺激目标选择上有所约束。阅读本节内容之前，建议阅读 3.2 节以了解部分背景知识，因为本节将从另外一个角度解释ms-eCCA 公式的由来。ms-eCCA 与 ms-eTRCA 是 *Wong* 在同一篇论文中提出的方法，在发表时间上也先于 msCCA。与 3.2 节所述一致，ms- 技术假定一套滤波器同时适用于目标频率以及周边少数频率的信号，通过合并一定范围内的多类别信号，强行扩增可用训练样本数目。

“合并样本”的操作，既可以理解为**不同类别样本模板在时间顺序上的拼接**，见式(1-5-2)；也可以理解为**不同类别样本各自协方差矩阵的叠加**，见式 (1-5-4)。结合 3.2 节 ms-(e)TRCA 与 1.1 节 CCA 的相关内容，ms-eCCA 的目标函数可表示为：
$$
\hat{\pmb{u}}_k, \hat{\pmb{v}}_k =
\underset{\pmb{u}_k, \pmb{v}_k} \argmax 
\dfrac{Cov \left(\pmb{u}_k \pmb{\mathcal{Z}}_k, \pmb{\mathcal{Y}}_k {\pmb{v}_k} \right)}
{\sqrt{Var \left(\pmb{u}_k \pmb{\mathcal{Z}}_k \right)} \sqrt{Var \left(\pmb{v}_k \pmb{\mathcal{Y}}_k \right)}} = 
\underset{\pmb{u}_k, \pmb{v}_k} \argmax 
\dfrac{\pmb{u}_k \pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Y}}_k} {\pmb{v}_k}^T}
{\sqrt{\left(\pmb{u}_k \pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Z}}_k} {\pmb{u}_k}^T \right)} \sqrt{\left(\pmb{v}_k \pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Y}}_k} {\pmb{v}_k}^T \right)}}\\
\tag{1-5-1}
$$
$$
\begin{cases}
\pmb{\mathcal{Z}}_k = 
\begin{bmatrix}
\bar{\pmb{X}}_{k-m} & \bar{\pmb{X}}_{k-m+1} & \cdots & \bar{\pmb{X}}_{k+n}
\end{bmatrix} \in \mathbb{R}^{N_c \times [(m+n+1)N_p]}\\
\\
\pmb{\mathcal{Y}}_k = 
\begin{bmatrix}
\pmb{Y}_{k-m} & \pmb{Y}_{k-m+1} & \cdots & \pmb{Y}_{k+n}
\end{bmatrix} \in \mathbb{R}^{N_c \times [(m+n+1) N_t N_p]}\\
\end{cases}
\tag{1-5-2}
$$
$$
\pmb{Y}_k = 
\begin{bmatrix}
\sin(2 \pi f_k \pmb{t} + \phi_k)\\
\cos(2 \pi f_k \pmb{t} + \phi_k)\\
\vdots\\
\sin(2 \pi N_h f_k \pmb{t} + N_h \phi_k)\\
\cos(2 \pi N_h f_k \pmb{t} + N_h \phi_k)\\
\end{bmatrix} \in \mathbb{R}^{(2N_h) \times N_p}, \ 
\pmb{t} = 
\begin{bmatrix}
\dfrac{1}{f_s} & \dfrac{2}{f_s} & \cdots & \dfrac{N_p}{f_s}
\end{bmatrix}
\tag{1-5-3}
$$
$$
\begin{cases}
\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Z}}_k} = \dfrac{1} {N_p-1} \sum_{i=-n}^{m} \bar{\pmb{X}}_{k+i} {\bar{\pmb{X}}_{k+i}}^T \in \mathbb{R}^{N_c \times N_c}\\
\\
\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Y}}_k} = \dfrac{1} {N_p-1} \sum_{i=-n}^{m} \pmb{Y}_{k+i} {\pmb{Y}_{k+i}}^T \in \mathbb{R}^{(2N_h) \times (2N_h)}\\
\\
\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Y}}_k} = \dfrac{1} {N_p-1} \sum_{i=-n}^{m} \bar{\pmb{X}}_{k+i} {\pmb{Y}_{k+i}}^T \in \mathbb{R}^{N_c \times (2N_h)}\\
\\
\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Z}}_k} = \dfrac{1} {N_p-1} \sum_{i=-n}^{m} \pmb{Y}_{k+i} {\bar{\pmb{X}}_{k+i}}^T \in \mathbb{R}^{(2N_h) \times N_c}\\
\end{cases}
\tag{1-5-4}
$$
简而言之，在空间滤波器构建上，ms-eCCA 与 ms-(e)TRCA 的思路一致，即把不同类别信号按时间维度顺次拼接，以起到数据扩增的作用。类比 1.1 节推导过程可知，式 (1-5-1) 所对应的两个 *GEP* 方程为：
$$
\begin{cases}
{\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Z}}_k}}^{-1}
\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Y}}_k}
{\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Y}}_k}}^{-1}
\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Z}}_k}
\pmb{u}_k = {\lambda}^2 {\pmb{u}_k}^T \ \ (I)\\
\\
{\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Y}}_k}}^{-1}
\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Z}}_k}
{\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Z}}_k}}^{-1}
\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Y}}_k}
\pmb{v}_k = {\theta}^2 {\pmb{v}_k}^T \ \ (II)\\
\end{cases}
\tag{1-5-5}
$$
首先来看方程组中的 $(I)$：
$$
\begin{align}
\notag
\left(\sum_k \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T \right)^{-1}
\left(\sum_a \bar{\pmb{X}}_a {\pmb{Y}_a}^T \right)
\left(\sum_c \pmb{Y}_c {\pmb{Y}_c}^T \right)^{-1}
\left(\sum_b \pmb{Y}_b {\bar{\pmb{X}}_b}^T \right) {\pmb{u}_k}^T &= 
{\lambda}^2 {\pmb{u}_k}^T\\ 
\notag
\ \\
\notag
\left(\sum_a \bar{\pmb{X}}_a {\pmb{Y}_a}^T \right)
\left(\sum_c \pmb{Y}_c {\pmb{Y}_c}^T \right)^{-1}
\left(\sum_b \pmb{Y}_b {\bar{\pmb{X}}_b}^T \right) {\pmb{u}_k}^T &= 
{\lambda}^2 \left(\sum_k \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T \right) {\pmb{u}_k}^T\\
\notag
\ \\
\notag
\left[\sum_b \sum_a \bar{\pmb{X}}_a {\pmb{Y}_a}^T \left(\sum_c \pmb{Y}_c {\pmb{Y}_c}^T \right)^{-1} \pmb{Y}_b {\bar{\pmb{X}}_b}^T \right] {\pmb{u}_k}^T &= {\lambda}^2 \left(\sum_k \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T \right) {\pmb{u}_k}^T
\end{align}
\tag{1-5-6}
$$
到这一步我们不难发现，其结果已经与上一节末尾的式 (1-4-18) 形式非常相似，仅剩的问题在于正交投影 $\pmb{Q}_k {\pmb{Q}_k}^T$ 与正余弦矩阵 $\pmb{Y}_k$ 之间的关系。

因此我们有必要介绍一下何为 QR 分解 ( *QR decomposition* )。QR 分解又称正交三角分解，通常用于求解矩阵的特征值与特征向量。QR 分解的作用是将实（复）非奇异矩阵 $\pmb{A}$ 转化为正交（酉）矩阵 $\pmb{Q}$ 与实（复）非奇异上三角矩阵 $\pmb{R}$ 的乘积。从操作结果来看，QR 矩阵分为全分解 ( *full decomposition* ) 与约化分解 ( *reduced decomposition* ) 两种，二者的差异见下图：

![QR分解示意图](figures/QR_decomposition.png)

一般情况下，我们需要的是约化 QR 分解结果。计算方法有很多，这里给出其中一种方便理解与实操的方法—— *Gram-Schmidt* 正交化。首先我们获取矩阵 $\pmb{A} \in \mathbb{R}^{m \times n}$ 的列向量组 $\pmb{A}(:,j)$，之后单位化 $\pmb{A}(:,1)$，并将其作为第一个正交基，对后续列向量依次进行投影分解：
$$
\begin{cases}
\pmb{A}(:,1) = r_{1,1} \pmb{q}_1\\
\pmb{A}(:,2) = r_{2,1} \pmb{q}_1 + r_{2,2} \pmb{q}_2\\
\cdots \\
\pmb{A}(:,n) = r_{n,1} \pmb{q}_1 + r_{n,2} \pmb{q}_2 + \cdots + r_{n,n} \pmb{q}_{n}\\
\end{cases}
\tag{1-5-7}
$$
$$
\pmb{A} = 
\begin{bmatrix}
\pmb{q}_1 & \pmb{q}_2 & \cdots & \pmb{q}_{n}
\end{bmatrix}
\begin{bmatrix}
r_{1,1} & r_{1,2} & \cdots & r_{1,n}\\
0 & r_{2,2} & \cdots & r_{2,n}\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & r_{n,n}\\
\end{bmatrix} = \pmb{Q_A R_A}
\tag{1-5-8}
$$
我们在 -R 技术、ms- 技术乃至未来将更新的 TDCA 算法中都经常见到正交投影 $\pmb{\mathcal{P}}_k = \pmb{Q}_k {\pmb{Q}_k}^T$ 的身影，我们来仔细研究一下这个矩阵。已知 ${\pmb{Y}_k}^T = \pmb{Q}_k \pmb{R}_k$，对照式 (1-5-8) 不难看出：

（1）$\pmb{Q}_k$ 本质上就是 ${\pmb{Y}_k}^T$，二者在数值上存在一定的系数比例。因为对于正余弦模板矩阵而言，其列向量本来就是正交的，只是不满足单位化而已；

（2）$\pmb{R}_k$ 是个对角阵，主对角线上的系数绝对值均相等，且为 ${\pmb{Y}_k}^T$ 中任一列向量的**内积平方根**，$\pmb{R}_k$ 的唯一作用就是将 $\pmb{Q}_k$ 压缩至单位化水平。这一点很容易证明：
$$
{\pmb{Y}_k}^T \pmb{Y}_k = \pmb{Q}_k \left(\pmb{R}_k {\pmb{R}_k}^T \right) {\pmb{Q}_k}^T = {r_{1,1}}^2 \pmb{Q}_k {\pmb{Q}_k}^T
\tag{1-5-9}
$$

又因为式 (1-4-4) 指出，$\pmb{\mathcal{P}} = \pmb{T}^T \left(\pmb{T} \pmb{T}^T \right)^{-1} \pmb{T} = \pmb{Q}_{\pmb{T}} {\pmb{Q}_{\pmb{T}}}^T$，我们知道 $\pmb{Q}_k {\pmb{Q}_k}^T = {\pmb{Y}_k}^T \left(\pmb{Y}_k {\pmb{Y}_k}^T \right)^{-1} \pmb{Y}_k$，接下来我们证明这个矩阵能够起到正交投影的作用。

关于正交投影，从泛函的角度而言可以给出如下定义：令 $\pmb{H}$ 为向量空间，$\pmb{M}$ 是 $\pmb{H}$ 内的 $n$ 维子空间。若对于 $\pmb{H}$ 中的向量 $\pmb{x}$：
$$
\exists \ \hat{\pmb{x}} \in \pmb{M}, \  \ s.t. \ \forall \ \pmb{y} \in \pmb{M}, \ \left<\pmb{x} - \hat{\pmb{x}}, \pmb{y} \right> = 0
\tag{1-5-10}
$$
则称 $\hat{\pmb{x}}$ 是 $\pmb{x}$ 在子空间 $\pmb{M}$ 上的投影。对于某一频率、各次谐波的正余弦信号张成的向量空间 $\pmb{Y}_k$，矩阵中每一个行向量都是该空间内的一个向量。我们来看投影过程：
$$
\pmb{X} {\pmb{Y}_k}^T \left(\pmb{Y}_k {\pmb{Y}_k}^T \right)^{-1} \pmb{Y}_k = \hat{\pmb{X}} \ \Longrightarrow \ 
\pmb{X} {\pmb{Y}_k}^T = \hat{\pmb{X}} {\pmb{Y}_k}^T\\
\ \\
\therefore \ \left(\pmb{X} - \hat{\pmb{X}} \right) {\pmb{Y}_k}^T = 0 \ \Longrightarrow \ \left<\pmb{X} - \hat{\pmb{X}}, \pmb{Y}_k \right>=0
\tag{1-5-11}
$$
由上可知，经过投影后的信号 $\hat{\pmb{X}}$ 位于向量空间 $\pmb{Y}_k$ 内，即纯化了 EEG 信号中与正余弦刺激相关的成分，尽管这个成分也许并不能完全覆盖刺激诱发信号的共性或个性特征。结合此前关于 QR 分解的描述，我们知道 $\pmb{Q}_k {\pmb{Q}_k}^T$ 与 ${\pmb{Y}_k}^T \pmb{Y}_k$ 的差别仅在于一个数值平衡，起到该效果的部分显然就是 $\left(\pmb{Y}_k {\pmb{Y}_k}^T \right)^{-1}$ 了：
$$
\left<\pmb{Y}_k(i,:), \pmb{Y}_k(i,:) \right> = \left\|\pmb{Y}_k(i,:) \right\|_2^2\\
\ \\
\left\|\pmb{Y}_k(i,:) \right\|_2^2 = \left\|\pmb{Y}_k(j,:) \right\|_2^2, \ \forall i,j \in \left[1,2N_h \right]\\
\ \\
\left(\pmb{Y}_k {\pmb{Y}_k}^T \right)^{-1} = 
\begin{bmatrix}
\dfrac{1}{\left\|\pmb{Y}_k(1,:) \right\|_2^2} & \cdots & 0\\
\vdots & \ddots & \vdots\\
0 & \cdots & \dfrac{1}{\left\|\pmb{Y}_k(2N_h,:) \right\|_2^2}\\
\end{bmatrix} = \dfrac{\pmb{I}_{2N_h}} {\left\|\pmb{Y}_k(i,:) \right\|_2^2} 
\tag{1-5-12}
$$
这里需要额外指出，$\left\|\pmb{Y}_k(i,:) \right\|_2^2$ 的数值与**正余弦信号的频率或相位**是无关的（这一点可以自行编程测试，不再证明），且人工构建正余弦信号的峰峰值均为 2（即幅值相等），因此该范数只与**数据长度**有关。换句话说在同一批次数据中，对任意 $k$ 该范数均相等：
$$
\left(\sum_c \pmb{Y}_c {\pmb{Y}_c}^T \right)^{-1} = 
\dfrac{\pmb{I}_{2N_h}}{N_e \left\|\pmb{Y}_k(i,:) \right\|_2^2}
\tag{1-5-13}
$$
因此，式 (1-5-6) 可简化为：
$$
\begin{align}
\notag
\dfrac{1}{N_e} \left[\sum_b \sum_a \bar{\pmb{X}}_a \dfrac{{\pmb{Y}_a}^T} {\left\|\pmb{Y}_k(i,:) \right\|_2} \dfrac{\pmb{Y}_b} {\left\|\pmb{Y}_k(i,:) \right\|_2} {\bar{\pmb{X}}_b}^T \right] {\pmb{u}_k}^T &= {\lambda}^2 \left(\sum_k \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T \right) {\pmb{u}_k}^T\\
\notag \ \\
\left(\dfrac{1}{N_e} \sum_b \sum_a \bar{\pmb{X}}_a \pmb{Q}_{\pmb{Y}_a} { \pmb{Q}_{\pmb{Y}_b}}^T {\bar{\pmb{X}}_b}^T \right) {\pmb{u}_k}^T &= {\lambda}^2 \left(\sum_k \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T \right) {\pmb{u}_k}^T
\end{align}
\tag{1-5-14}
$$
这样我们就来到了式 (1-4-18)。条条大路通罗马，不是吗？但比起式 (1-5-14) 这种惊险刺激的过山车，我个人还是更喜欢式 (1-5-5) 这种平静祥和的旅途。

对比可知，上一节中提到的 msCCA 只是式 (1-5-5) 中的 $(I)$。尽管它可以进行模板匹配，但是其背后的设计思维显然不止于此，这也是我们通过统一框架无法了解到的信息。式 (1-5-1) 指出，msCCA 的训练目标应当是为训练数据与正余弦信号（二者均为合并样本）各自寻找一组投影向量以满足后续需求，而不是单方面优化 EEG 数据。

结合上述关于投影矩阵相关的说明，我们可以剔除 ${\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Y}}_k}}^{-1}$ 以进一步简化 $(I)$：
$$
{\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Z}}_k}}^{-1} \pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Y}}_k}
\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Z}}_k} \pmb{u}_k = 
{\lambda}^2 {\pmb{u}_k}^T
\tag{1-5-15}
$$
我们来看看这个 *GEP* 方程对应的广义瑞利商形式：
$$
\hat{\pmb{u}}_k = \underset{\pmb{u}_k} \argmax
\dfrac{\pmb{u}_k \pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Y}}_k} \pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Z}}_k} {\pmb{u}_k}^T} {\pmb{u}_k \left(\sum_k \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T \right) {\pmb{u}_k}^T} 
\tag{1-5-16}
$$
这个方程的分母是**滤波后合并样本的能量**，但是分子就很有意思了，显然 $\pmb{u}_k \pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Y}}_k}$ 并不能表示某种信号，因为它的长度已经与采样信号不一样了，这一点在所有 CCA 系列算法中其实都存在。目前我还无法给它一个合理的物理解释，如果各位观众姥爷有高见，希望邮件联系我指点迷津。$(I)$ 的问题到此为止了，可 $(II)$ 真不是一位善茬，不论如何都无法缩减运算量，只得就此作罢。最后总结一下 ms-eCCA 的空间滤波器构建函数（*GEP* 方程）以及模板匹配所需的步骤：
$$
\begin{cases}
{\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Z}}_k}}^{-1}
\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Y}}_k}
\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Z}}_k}
\pmb{u}_k = {\lambda}^2 {\pmb{u}_k}^T \ \ (I)\\
\\
{\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Y}}_k}}^{-1}
\pmb{C}_{\pmb{\mathcal{Y}}_k \pmb{\mathcal{Z}}_k}
{\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Z}}_k}}^{-1}
\pmb{C}_{\pmb{\mathcal{Z}}_k \pmb{\mathcal{Y}}_k}
\pmb{v}_k = {\theta}^2 {\pmb{v}_k}^T \ \ (II)\\
\end{cases}
\tag{1-5-17}
$$
$$
\rho_k = \sum_{i=1}^2 sign \left(\rho_{k,i} \right) \times \rho_{k,i}^2, \ 
\begin{cases}
\rho_{k,1} = corr \left(\pmb{u}_k \pmb{\mathcal{X}}, \pmb{v}_k \pmb{Y}_k \right)\\
\rho_{k,2} = corr \left(\pmb{u}_k \pmb{\mathcal{X}}, \pmb{u}_k \bar{\pmb{X}}_k \right)\\
\end{cases}
\tag{1-5-18}
$$
***

## 1.6 多重数据集 CCA：MsetCCA
**(Multi-set CCA)**

**[论文链接][MsetCCA] | 代码：[cca][cca(code)].msetcca()**

标准 CCA 算法使用人工构建的标准正余弦模板来描述 SSVEP 的正弦波动性质。在有条件（可接受训练校准）的情况下，这种设计无法充分利用受试者个体的差异性信息，因此性能受到了极大的限制。包括 it-CCA、eCCA 等多种基于 CCA 的改进算法都在尝试引入真实脑电数据以提高模型的有效性，MsetCCA 也是其中一种。该算法由 *Zhang* 等人于 2014 年提出，对标算法是标准 CCA 以及 MwayCCA，在实际分类性能上与 eCCA、TRCA 等后来者有一定的差距，但是算法设计思路依然值得参考。

对于第 $k$ 类别、具有 $N_t$ 个试次的训练数据集 $\pmb{X}_k \in \mathbb{R}^{N_t \times N_c \times N_p}$，MsetCCA 为其中每一个试次 $\pmb{X}_k^i \in \mathbb{R}^{N_c \times N_p}$ 都训练一个滤波器 $\pmb{\omega}_k^i$，最终使得**滤波后信号**的叠加平均模板能量与原始信号能量的比值最大：
$$
\begin{cases}
\underset{\pmb{\omega}_k^1, \pmb{\omega}_k^2, \cdots, \pmb{\omega}_k^{N_t}} \max \sum_{i \ne j, i=1}^{N_t} \sum_{j=1}^{N_t} \pmb{\omega}_k^i \pmb{X}_k^i {\pmb{X}_k^j}^T {\pmb{\omega}_k^j}^T\\
\\
s.t.\ \dfrac{1}{N_t} \sum_{i=1}^{N_t} \pmb{\omega}_k^i \pmb{X}_k^i {\pmb{X}_k^i}^T {\pmb{\omega}_k^i}^T = 1
\end{cases}
\tag{1-6-1}
$$
$$
\begin{align}
\notag
\sum_{i=1}^{N_t} \sum_{j=1}^{N_t} \pmb{\omega}_i \pmb{X}_i {\pmb{X}_j}^T {\pmb{\omega}_j}^T &=
\underbrace{
\begin{bmatrix}
\pmb{\omega}_1 & \pmb{\omega}_2 & \cdots & \pmb{\omega}_{N_t}\\
\end{bmatrix}}_{\pmb{W}}
\underbrace{
\begin{bmatrix}
\pmb{X}_1 \pmb{X}_1^T & \pmb{X}_1 \pmb{X}_2^T & \cdots & \pmb{X}_1 \pmb{X}_{N_t}^T\\
\pmb{X}_2 \pmb{X}_1^T & \pmb{X}_2 \pmb{X}_2^T & \cdots & \pmb{X}_2 \pmb{X}_{N_t}^T\\
\vdots & \vdots & \ddots & \vdots\\
\pmb{X}_{N_t} \pmb{X}_1^T & \pmb{X}_{N_t} \pmb{X}_2^T & \cdots & \pmb{X}_{N_t} \pmb{X}_{N_t}^T\\
\end{bmatrix}}_{\pmb{R}}
\underbrace{
\begin{bmatrix}
\pmb{\omega}_1^T\\ \pmb{\omega}_2^T\\ \vdots\\ \pmb{\omega}_{N_t}^T\\
\end{bmatrix}}_{\pmb{W}^T} \tag{1-6-2}\\
\notag \ \\
\notag
\sum_{i=1}^{N_t}\pmb{\omega}_i \pmb{X}_i {\pmb{X}_i}^T {\pmb{\omega}_i}^T &=
\underbrace{
\begin{bmatrix}
\pmb{\omega}_1 & \pmb{\omega}_2 & \cdots & \pmb{\omega}_{N_t}\\
\end{bmatrix}}_{\pmb{W}}
\underbrace{
\begin{bmatrix}
\pmb{X}_1 \pmb{X}_2^T & 0 & \cdots & 0\\
0 & \pmb{X}_2 \pmb{X}_2^T & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & \pmb{X}_{N_t} \pmb{X}_{N_t}^T\\
\end{bmatrix}}_{\pmb{S}}
\underbrace{
\begin{bmatrix}
\pmb{\omega}_1^T\\ \pmb{\omega}_2^T\\ \vdots\\ \pmb{\omega}_{N_t}^T\\
\end{bmatrix}}_{\pmb{W}^T} \tag{1-6-3}
\end{align}
$$
$$
\hat{\pmb{W}} = \underset{\pmb{W}} \argmax \dfrac{\pmb{W} (\pmb{R}-\pmb{S}) \pmb{W}^T} {\pmb{W} \pmb{S} \pmb{W}^T} = \underset{\pmb{W}} \argmax \dfrac{\pmb{W} \pmb{R} \pmb{W}^T} {\pmb{W} \pmb{S} \pmb{W}^T}
\tag{1-6-4}
$$

![MsetCCA示意图](figures/MsetCCA.png)

在获取联合滤波器 $\pmb{W}$ 之后，对各试次训练数据分别滤波获得联合模板 $\pmb{Y}_k$ ：
$$
\pmb{Y}_k = 
\begin{bmatrix}
\pmb{\omega}_k^1 \pmb{X}_k^1\\
\pmb{\omega}_k^2 \pmb{X}_k^2\\
\vdots\\
\pmb{\omega}_k^{N_t} \pmb{X}_k^{N_t}\\
\end{bmatrix} \in \mathbb{R}^{N_t \times N_p}
\tag{1-6-5}
$$
由于 $\pmb{Y}_k$ 依然是多维数据，因此在分类过程中需要与测试数据 $\pmb{\mathcal{X}}$ 进行标准 CCA 过程的匹配。当然我们也可以使用二维 *Pearson* 相关系数，虽然可能不算一个很有创新点的 idea，但是林林总总的改进 tricks 凑个数攒一篇还是有可能的。具体流程见上图：
$$
\hat{k} = \underset{k} \argmax{\{\rho_k \ | \ \rho_k = CCA\left(\pmb{\mathcal{X},\pmb{Y}_k}\right)\}}
\tag{1-6-6}
$$
***

## 1.7 多重通道 CCA：MwayCCA
**(Multiway CCA)**

**[论文链接][MwayCCA] | 代码：[cca][cca(code)].mwaycca()**













[msCCA]: https://ieeexplore.ieee.org/document/9006809/
[Benchmark]: https://ieeexplore.ieee.org/document/7740878/
[UCSD]: https://dx.plos.org/10.1371/journal.pone.0140703
[BETA]: https://www.frontiersin.org/article/10.3389/fnins.2020.00627/full

[MsetCCA]: https://www.worldscientific.com/doi/abs/10.1142/S0129065714500130

[MwayCCA]: http://link.springer.com/10.1007/978-3-642-24955-6_35

[CSSFT]: http://iopscience.iop.org/article/10.1088/1741-2552/ac6b57

***