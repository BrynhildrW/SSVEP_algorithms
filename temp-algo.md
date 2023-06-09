---
html:
    toc: true
print_background: true
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>

# 判别空间模式
**Discriminant Spatial Patterns**

**[论文链接][DSP] | 代码：[special][dsp(code)].DSP()**

DSP 是基于线性判别分析 ( *Linear Discriminant Analysis, LDA* ) 设计的判别式模型及分类算法。判别式算法相比 TRCA、CCA 等生成式模型之间存在理念上的较大差异。

（1）生成式模型通常仅使用单类数据内部的各种信息，其根本目标是通过约束、投影等方法使得类内数据相似度提高，难以有效学习不同类别信息中的差异。当然，基于多类别甚至全体类别数据的生成式 SSVEP 算法也不是没有，例如ms-(e)TRCA、ms-CCA、ms-eCCA等。ms- 技术认为不同频率-相位刺激诱发的 SSVEP 信号在空间分布上遵循相同模式、适用于同一空间滤波器。因此这些算法在训练模型的过程中将各类数据协方差进行等权重叠加以达到增强数据利用率的效果；

（2）判别式模型

这种差异在应用得当的情况下可以转变为模型优势：

[DSP]: https://ieeexplore.ieee.org/document/8930304/
[dsp(code)]: https://github.com/BrynhildrW/SSVEP_algorithms/blob/main/programs/special.py