# 个体 CCA
## Individual template based CCA, itCCA
***

#### [论文链接][itCCA]

据现业界大牛 Nakanishi 在一篇因“年少不懂事”发的[论文][ex1]（PLOS ONE）里说，IT-CCA 最早用于一种 c-VEP 信号（code modulated VEP）的解码，是 Bin 等人完成的研究。考虑到这个算法的思路实在过于简单，我还一度怀疑过他们是不是第一个提出该算法的团队。

不同于 SSVEP，c-VEP 信号的特征难以用正余弦波动性质描述，但确实存在稳定的类间差异性与类内统一性。因此基于正余弦信号的标准 CCA 算法并不适用于这种情况，Bin 等人在此基础上修改目标函数，把正余弦模板 $\pmb{Y}_k$ 替换成训练数据的叠加平均 $\bar{\pmb{X}}_k$，其余步骤与标准 CCA 完全一致：
$$
    \hat{\pmb{u}}_k, \hat{\pmb{v}}_k 
    = \underset{\pmb{u}_k, \pmb{v}_k} \argmax {\dfrac{{\rm Cov} \left(\pmb{u}_k \pmb{\mathcal{X}}, \pmb{v}_k \bar{\pmb{X}}_k \right)} {\sqrt{{\rm Var} \left(\pmb{u}_k \pmb{\mathcal{X}} \right)} \sqrt{{\rm Var} \left(\pmb{v}_k \bar{\pmb{X}}_k \right)}}}
    = \underset{\pmb{u}_k, \pmb{v}_k} \argmax {\dfrac{\pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \bar{\pmb{X}}_k} {\pmb{v}_k}^T} {\sqrt{\pmb{u}_k \pmb{C}_{\pmb{\mathcal{X}} \pmb{\mathcal{X}}} {\pmb{u}_k}^T} \sqrt{\pmb{v}_k \pmb{C}_{\bar{\pmb{X}}_k \bar{\pmb{X}}_k} {\pmb{v}_k}^T}}}\
    \tag{1}
$$
***

[itCCA]: https://iopscience.iop.org/article/10.1088/1741-2560/8/2/025015
[ex1]: https://dx.plos.org/10.1371/journal.pone.0140703