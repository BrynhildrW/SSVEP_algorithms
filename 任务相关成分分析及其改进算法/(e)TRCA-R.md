# 正余弦扩展 TRCA
## (e)TRCA-R

---

[论文链接][TRCA-R]

该算法依旧出自 *Wong* 的手笔。按照其提出的设计框架，-R 技术就是将原本为单位阵的空间投影矩阵替换为正余弦信号张成的投影空间 $\pmb{\mathcal{P}}$，与之类似的算法还有 MsetCCA1-R。在讲解 (e)TRCA-R 之前，我们先来观察 (e)TRCA 的目标函数在统一框架（见 msCCA）下的各部分组成：
$$
\begin{cases}
\pmb{\mathcal{Z}} = \pmb{\mathcal{I}}_{N_t,N_c} \left(\oplus_{i=1}^{N_t} \pmb{X}_k^i \right) \in \mathbb{R}^{N_c \times \left(N_t N_p \right)}\\
\\
\pmb{\mathcal{D}} = \pmb{I}_{N_t N_p} \in \mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}\\
\\
\pmb{\mathcal{P}} = {\pmb{\mathcal{I}}_{N_t,N_p}}^T \pmb{\mathcal{I}}_{N_t,N_p} \in \mathbb{R}^{\left(N_t N_p \right) \times \left(N_t N_p \right)}
\end{cases}
\tag{1}
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
\tag{2}
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
\tag{3}
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
\tag{4}
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
\tag{5}
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
\tag{6}
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
\tag{7}
$$
综上所述，仅需一维投影向量的情况下，GEP 方程可表示为式（8），忽略常系数影响后可发现该式与 (e)TRCA 的目标函数完全吻合。
$$
\left(N_t\sum_{j=1}^{N_t} \sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) \pmb{w} = 
\lambda \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) \pmb{w}
\tag{8}
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
\tag{9}
$$
注意有 $\pmb{\mathcal{P}} = \pmb{\mathcal{P}} {\pmb{\mathcal{P}}}^T$，所以 (e)TRCA-R 的 GEP 方程可表示为：
$$
\left[N_t\sum_{j=1}^{N_t} \sum_{i=1}^{N_t} \left(\pmb{X}_k^i \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T\right) \left(\pmb{X}_k^j \pmb{Q}_{\pmb{Y}_k} {\pmb{Q}_{\pmb{Y}_k}}^T\right)^T \right] \pmb{w}^T = 
\lambda \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T \right) \pmb{w}^T
\tag{10}
$$
目标函数可以写为：
$$
\hat{\pmb{w}}_k = 
\underset{\pmb{w}_k} \argmax 
\dfrac{\pmb{w}_k \left(\bar{\pmb{X}}_k \pmb{P}_k \right) \left(\bar{\pmb{X}}_k \pmb{P}_k \right)^T {\pmb{w}_k}^T} {\pmb{w}_k \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T\right) {\pmb{w}_k}^T} = 
\dfrac{\pmb{w}_k \bar{\pmb{X}}_k \pmb{Y}_k^T \pmb{Y}_k \bar{\pmb{X}}_k^T {\pmb{w}_k}^T} {\pmb{w}_k \left(\sum_{i=1}^{N_t} \pmb{X}_k^i {\pmb{X}_k^i}^T\right) {\pmb{w}_k}^T}
\tag{11}
$$
通过（11）可以观察到，-R 技术的改进点在于通过正交投影矩阵 $\pmb{P}_k$ 进一步约束了优化函数的目标 $\bar{\pmb{X}}_k$ 。关于投影矩阵的作用，在 ms-eCCA 章节中已有介绍，此处不再赘述。

---

[TRCA-R]: https://ieeexplore.ieee.org/document/9006809/