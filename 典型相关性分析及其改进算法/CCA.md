# 标准CCA
## Standard Canonical Correlation Analysis, sCCA or CCA
***

### [论文链接][CCA]

对于具有 $N_e$ 个不同频率刺激的 SSVEP-BCI 系统，频率索引 $k$ 的人工构建正余弦模板 $\pmb{Y}_k$ 可表示为：
$$
\pmb{Y}_k = \begin{bmatrix} 
\sin \left(2 \pi f_k n \right)\\
\cos \left(2 \pi f_k n \right)\\
\sin \left(4 \pi f_k n \right)\\
\cos \left(4 \pi f_k n \right)\\
\vdots\\
\sin \left(2 N_h \pi f_k n \right)\\
\cos \left(2 N_h \pi f_k n \right)\\
\end{bmatrix} \in \mathbb{R}^{\left(2N_h \right) \times N_p}, \ n=\left[\dfrac{1}{f_s}, \dfrac{2}{f_s}, ..., \dfrac{N_p}{f_s} \right]
\tag{1}
$$
对于单试次多导联 EEG 测试数据 $\pmb{\mathcal{X}} \in \mathbb{R}^{N_c \times N_p}$ 以及假定的所属类别 $k$ ，CCA 的优化目标为一组投影向量 $\hat{\pmb{u}}_k$ 和 $\hat{\pmb{v}}_k$，使得一维信号 $\hat{\pmb{u}}_k \pmb{\mathcal{X}}$ 与 $\hat{\pmb{v}}_k \pmb{Y}_k$ 之间相关性最大化，其目标函数为：
$$
\hat{\pmb{u}}_k, \hat{\pmb{v}}_k 
= \underset{\pmb{u}_k, \pmb{v}_k} \argmax {\dfrac{{\rm {\rm Cov}} \left(\pmb{u}_k \pmb{\mathcal{X}}, \pmb{v}_k \pmb{Y}_k \right)} {\sqrt{{\rm Var} \left(\pmb{u}_k \pmb{\mathcal{X}} \right)} \sqrt{{\rm Var} \left(\pmb{v}_k \pmb{Y}_k \right)}}} 
= \underset{\pmb{u}_k, \pmb{v}_k} \argmax {\dfrac{\pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{v}_k}^T} {\sqrt{\pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}} {\pmb{u}_k}^T} \sqrt{\pmb{v}_k \pmb{C}_{\pmb{Y}_k \pmb{Y}_k} {\pmb{v}_k}^T}}}
\tag{2}
$$
$$
\begin{cases}
\pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}} = \dfrac{1}{N_p-1} \pmb{\mathcal{X}} {\pmb{\mathcal{X}}}^T \in \mathbb{R}^{N_c \times N_c}\\
\\
\pmb{C}_{\pmb{Y}_k \pmb{Y}_k} = \dfrac{1}{N_p-1} \pmb{Y}_k {\pmb{Y}_k}^T \in \mathbb{R}^{\left(2N_h \right) \times \left(2N_h \right)}\\
\\
\pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} = \dfrac{1}{N_p-1} \pmb{\mathcal{X}} {\pmb{Y}_k}^T \in \mathbb{R}^{N_c \times \left(2N_h \right)}\\
\\
\pmb{C}_{\pmb{Y}_k \pmb{\mathcal{X}}} = \dfrac{1}{N_p-1} \pmb{Y}_k {\pmb{\mathcal{X}}}^T \in \mathbb{R}^{\left(2N_h \right) \times N_c}\\
\end{cases}
\tag{3}
$$
根据最优化理论，函数 (2) 的等效形式为：
$$
\begin{cases}
\underset{\pmb{u}_k, \pmb{v}_k} \max \ \pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{v}_k}^T\\
\\
{\rm s.t.}\ \ \pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}} {\pmb{{u}}_k}^T =
\pmb{v}_k \pmb{C}_{\pmb{Y}_k \pmb{Y}_k} {\pmb{v}_k}^T = 1
\end{cases}
\tag{4}
$$

### 特征值分解法
利用 Lagrandian 乘子法构建多元函数 $J(\pmb{u}_k, \pmb{v}_k, \lambda, \theta)$：
$$
J = \pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{v}_k}^T - \dfrac{1}{2} \lambda \left(\pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}} {\pmb{u}_k}^T - 1 \right) - \dfrac{1}{2} \theta \left(\pmb{v}_k \pmb{C}_{\pmb{Y}_k \pmb{Y}_k} {\pmb{v}_k}^T - 1 \right)
\tag{5}
$$
对函数 $J$ 求偏导数并置零：
$$
\begin{cases}
\dfrac{\partial J} {\partial \pmb{u}_k} = 
\pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{v}_k}^T - \lambda \pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}} {\pmb{u}_k}^T = 0 \ \ ({\rm I})\\
\\
\dfrac{\partial J} {\partial \pmb{v}_k} = 
\pmb{C}_{\pmb{Y}_k \pmb{\mathcal{X}}} {\pmb{u}_k}^T - \theta \pmb{C}_{\pmb{Y}_k \pmb{Y}_k} {\pmb{v}_k}^T = 0 \ \ ({\rm II})
\end{cases}
\tag{6}
$$
消元化简后可知 $\lambda = \theta$。代回方程组可得：
$$
\begin{cases}
\pmb{u}_k * ({\rm I}) \to 
\pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{v}_k}^T - \lambda \pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}} {\pmb{u}_k}^T = 0\\
\\
\pmb{v}_k * ({\rm II}) \to 
\pmb{v}_k \pmb{C}_{\pmb{Y}_k \pmb{\mathcal{X}}} {\pmb{u}_k}^T - \theta \pmb{v}_k \pmb{C}_{\pmb{Y}_k \pmb{Y}_k} {\pmb{v}_k}^T = 0\\
\end{cases}
\tag{7}
$$
根据约束条件 (4) 可知：
$$
\lambda = \pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{v}_k}^T, \ 
\theta = \pmb{v}_k \pmb{C}_{\pmb{Y}_k \pmb{\mathcal{X}}} {\pmb{u}_k}^T
\tag{8}
$$
注意 $\lambda = {\theta}^T$，而当我们明确要求优化目标是**一维向量**的时候，式 (8) 的两个变量其实都是实数，所以它们相等。之后就是大家在解二元一次方程组时常用的代换消元过程（$\pmb{u}_k$ 与 $\pmb{v}_k$ 互相替换），我就不再演示了。最终应得到两个特征值方程：
$$
\begin{cases}
{\pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}}}^{-1} \pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{C}_{\pmb{Y}_k \pmb{Y}_k}}^{-1} \pmb{C}_{\pmb{Y}_k \pmb{\mathcal{X}}} {\pmb{u}_k}^T = {\lambda}^2 {\pmb{u}_k}^T\\
\\
{\pmb{C}_{\pmb{Y}_k \pmb{Y}_k}}^{-1} \pmb{C}_{\pmb{Y}_k \pmb{\mathcal{X}}} {\pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}}}^{-1} \pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{v}_k}^T = {\theta}^2 {\pmb{v}_k}^T
\end{cases}
\tag{9}
$$
对式 (9) 中的两个 Hermitte 矩阵分别进行特征值分解，取最大特征值对应的特征向量作为投影向量，即为所求。对所有的假定类别遍历上述过程，基于一维 Pearson 相关系数分别计算判别系数并比较大小，确定最终的结果输出 $\hat{k}$：
$$
\rho_k = {\rm corr} \left(\hat{\pmb{u}}_k \pmb{\mathcal{X}}, \hat{\pmb{v}}_k \pmb{Y}_k \right), \ \hat{k} = \underset{k} \argmax\{\rho_k\}
\tag{10}
$$

### 奇异值分解法
对式（4）展示的边界约束条件进行矩阵标准化，令 $\pmb{u}_k = \pmb{a}_k {\pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}}}^{-\frac{1}{2}}$、$\pmb{v}_k = \pmb{b}_k {\pmb{C}_{\pmb{Y}_k \pmb{Y}_k}}^{-\frac{1}{2}}$，则有：
$$
\begin{align}
\notag \pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}} {\pmb{{u}}_k}^T &= \pmb{a}_k {\pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}}}^{-\frac{1}{2}} \pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}} {\pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}}}^{-\frac{1}{2}} {\pmb{a}_k}^T = \pmb{a}_k {\pmb{a}_k}^T = 1\\
\notag \ \\
\notag \pmb{v}_k \pmb{C}_{\pmb{Y}_k \pmb{Y}_k} {\pmb{{v}}_k}^T &= \pmb{b}_k {\pmb{C}_{\pmb{Y}_k \pmb{Y}_k}}^{-\frac{1}{2}} \pmb{C}_{\pmb{Y}_k \pmb{Y}_k} {\pmb{C}_{\pmb{Y}_k \pmb{Y}_k}}^{-\frac{1}{2}} {\pmb{b}_k}^T = \pmb{b}_k {\pmb{b}_k}^T = 1\\
\end{align}
\tag{11}
$$
此时，优化目标转化为：
$$
\begin{cases}
\underset{\pmb{a}_k, \ \pmb{b}_k}{\max} \ \pmb{a}_k {\pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}}}^{-\frac{1}{2}} \pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{C}_{\pmb{Y}_k \pmb{Y}_k}}^{-\frac{1}{2}} {\pmb{b}_k}^T\\
\ \\
{\rm s.t.} \ \ \pmb{a}_k {\pmb{a}_k}^T = \pmb{b}_k {\pmb{b}_k}^T = 1
\end{cases}
\tag{12}
$$
如果将 $\pmb{a}_k$ 和 $\pmb{b}_k$ 视为矩阵 ${\pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}}}^{-\frac{1}{2}} \pmb{C}_{\pmb{\mathcal{X}} \pmb{Y}_k} {\pmb{C}_{\pmb{Y}_k \pmb{Y}_k}}^{-\frac{1}{2}}$ 某个奇异值对应的左右奇异向量，则目标函数（12）即为最大化奇异值，$\pmb{a}_k$ 和 $\pmb{b}_k$ 即为最大奇异值对应的左右奇异向量，转换之后即可求得 $\pmb{u}_k$ 与 $\pmb{v}_k$。需要指出的是，奇异值分解法和特征值分解法求得的投影向量在数值尺度上是有差别的。这是因为特征值分解保证的是特征向量（即 $\pmb{u}_k$ 与 $\pmb{v}_k$）标准正交，而奇异值分解保证的是奇异向量（即 $\pmb{a}_k$ 和 $\pmb{b}_k$）标准正交，尽管这种数值尺度差异不影响相关性匹配结果，但需要注意一下。此外，在求解速度方面，奇异值和特征值两种解法并没有显著差异。
***

[CCA]: http://ieeexplore.ieee.org/document/4203016/