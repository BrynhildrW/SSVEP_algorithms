### 3.3 正余弦扩展 TRCA：(e)TRCA-R
**[论文链接][TRCA-R] | 代码：[trca][trca(code)].etrca_r()**

该算法依旧出自 *Wong* 的手笔。按照其提出的设计框架，-R 技术就是将原本为单位阵的空间投影矩阵替换为正余弦信号张成的投影空间 $\pmb{\mathcal{P}}$，与之类似的算法还有 MsetCCA1-R（未来更新）。在讲解 (e)TRCA-R 之前，我们先来观察 (e)TRCA 的目标函数在统一框架（ 1.3 节式 (1-3-4)，**Type I** ) 下的各部分组成：
$$
    \begin{cases}
        \pmb{\mathcal{Z}} = \pmb{\mathcal{I}}_{N_t,N_c} \left(\oplus_{i=1}^{N_t} \pmb{X}_k^i \right) \in \mathbb{R}^{N_c \times \left(N_t N_p \right)}\\
        \\
        \pmb{\mathcal{D}} = \pmb{I}_{N_t N_p} \in \mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}\\
        \\
        \pmb{\mathcal{P}} = {\pmb{\mathcal{I}}_{N_t,N_p}}^T \pmb{\mathcal{I}}_{N_t,N_p} \in \mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}
    \end{cases}
    \tag{3-3-1}
$$
为了更清楚地让大家明白这个框架到底干了什么事，我们来依次画一下各步骤的展开形态：
$$
    \pmb{\mathcal{Z}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_c} & \pmb{I}_{N_c} & \cdots & \pmb{I}_{N_c}
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_c \right)}}
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \pmb{0} & \cdots & \pmb{0}\\
            \pmb{0} & \pmb{X}_k^2 & \cdots & \pmb{0}\\
            \vdots & \vdots & \ddots & \vdots\\
            \pmb{0} & \pmb{0} & \cdots & \pmb{X}_k^{N_t}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_c \right) \times \left(N_t N_p \right)}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \pmb{X}_k^2 & \cdots & \pmb{X}_k^{N_t}
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \tag{3-3-2}
$$
$$
    \pmb{\mathcal{Z}} \pmb{\mathcal{D}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \pmb{X}_k^2 & \cdots & \pmb{X}_k^{N_t}
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p} & \pmb{0} & \cdots & \pmb{0}\\
            \pmb{0} & \pmb{I}_{N_p} & \cdots & \pmb{0}\\
            \vdots & \vdots & \ddots & \vdots\\
            \pmb{0} & \pmb{0} & \cdots & \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}} = \pmb{\mathcal{Z}}
    \tag{3-3-3}
$$
$$
    \pmb{\mathcal{P}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p}\\
            \pmb{I}_{N_p}\\
            \vdots\\
            \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times N_p}}
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p} & \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{N_p \times \left(N_t N_p \right)}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p} & \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
            \pmb{I}_{N_p} & \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
            \vdots & \vdots & \ddots & \vdots\\
            \pmb{I}_{N_p} & \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}}
    \tag{3-3-4}
$$
$$
    \pmb{\mathcal{Z}} \pmb{\mathcal{D}} \pmb{\mathcal{P}} = 
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \cdots & \pmb{X}_k^{N_t}\\
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \underbrace{
        \begin{bmatrix}
            \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
            \vdots & \ddots & \vdots\\
            \pmb{I}_{N_p} & \cdots & \pmb{I}_{N_p}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}} = 
    \underbrace{
        \begin{bmatrix}
            \sum_{i=1}^{N_t}{\pmb{X}_k^i} & \cdots & \sum_{i=1}^{N_t}{\pmb{X}_k^i}\\
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \tag{3-3-5}
$$
$$
    \pmb{\mathcal{Z}} \pmb{\mathcal{D}} \pmb{\mathcal{P}} {\pmb{\mathcal{P}}}^T {\pmb{\mathcal{D}}}^T {\pmb{\mathcal{Z}}}^T = 
    \underbrace{
        \begin{bmatrix}
            \sum_{i=1}^{N_t}{\pmb{X}_k^i} & \cdots & \sum_{i=1}^{N_t}{\pmb{X}_k^i}\\
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \underbrace{
        \begin{bmatrix}
            \sum_{i=1}^{N_t}{\pmb{X}_k^i}\\
            \vdots\\
            \sum_{i=1}^{N_t}{\pmb{X}_k^i}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times N_c}} = 
    N_t\sum_{j=1}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T
    \tag{3-3-6}
$$
$$
    \pmb{\mathcal{Z}} \pmb{\mathcal{D}} {\pmb{\mathcal{D}}}^T {\pmb{\mathcal{Z}}}^T = \pmb{\mathcal{Z}} \pmb{\mathcal{Z}}^T = 
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1 & \pmb{X}_k^2 & \cdots & \pmb{X}_k^{N_t}\\
        \end{bmatrix}}_{\mathbb{R}^{N_c \times \left(N_t N_p \right)}}
    \underbrace{
        \begin{bmatrix}
            \pmb{X}_k^1\\
            \pmb{X}_k^2\\
            \vdots\\
            \pmb{X}_k^{N_t}\\
        \end{bmatrix}}_{\mathbb{R}^{\left(N_t N_p \right) \times N_c}} = 
    \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T
    \tag{3-3-7}
$$
综上所述，仅需一维投影向量的情况下，*GEP* 方程可表示为式 (3-3-8)，忽略常系数影响后可发现该式与 (e)TRCA 的目标函数完全吻合。
$$
    \left(N_t\sum_{j=1}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) \pmb{\omega} = 
    \lambda \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) \pmb{\omega}
    \tag{3-3-8}
$$
在常规 (e)TRCA 中，正交投影矩阵 $\pmb{\mathcal{P}}$ 的本质作用仅是叠加。而在 (e)TRCA-R 中，*Wong* 将其改为了正余弦信号张成的投影空间，其余均与 (e)TRCA 保持一致：
$$
    \pmb{\mathcal{P}} = 
    \begin{bmatrix}
        \pmb{Q}_{\pmb{Y}_k}\\
        \vdots\\
        \pmb{Q}_{\pmb{Y}_k}\\
    \end{bmatrix}
    \begin{bmatrix}
        \pmb{Q}_{\pmb{Y}_k} & \cdots & \pmb{Q}_{\pmb{Y}_k}\\
    \end{bmatrix} = 
    \begin{bmatrix}
        \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T & \cdots & \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T\\
        \vdots & \ddots & \vdots\\
        \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T & \cdots & \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T\\
    \end{bmatrix}
    \tag{3-3-9}
$$
注意有 $\pmb{\mathcal{P}} = \pmb{\mathcal{P}} {\pmb{\mathcal{P}}}^T$，所以 (e)TRCA-R 的 *GEP* 方程可表示为：
$$
    \left(N_t\sum_{j=1}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^i \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T {\pmb{X}_k^i}^T \right) \pmb{\omega} = 
    \lambda \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) \pmb{\omega}
    \tag{3-3-10}
$$
平心而论，我很难直观地理解到底发生了什么事，以及为什么这个正交投影能够起作用。另外，由于 -R 技术在 $\pmb{X}_k^i {\pmb{X}_k^i}^T$ 的过程中插入了额外的矩阵乘法，导致 TRCA 复现过程中的一个重要 trick（通过叠加平均信号的矩阵乘法替代跨试次循环）不再生效，而 **np.einsum()** 函数在面对跨试次矩阵乘法时似乎存在一些内在的逻辑问题，其运算效率出奇地低，因此目前我推荐的复现方法依旧只有循环。
