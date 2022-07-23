---
html:
    toc: true
print_background: true
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>

# 常见 SSVEP 信号处理算法（空间滤波器）

Github 的在线 Markdown 公式编译模块有 bug，想看公式过程的建议下载**算法说明**文件本地查看。鉴于部分公式推导步骤属于本人硕士学位论文内容，在此提醒各位，ctrl+C/V 请慎重。

更新进展：今天写了 msTRCA 的部分说明、吐槽（2022/7/23）
近期计划：看心情更新。

**建议各位同僚读完硕士赶紧去就业吧，千万不要盲目读博、投身火海。**

***
### 1. 典型相关性分析
**Canonical correlation analysis, CCA**
#### 1.1 标准 CCA：CCA
**[论文链接][CCA] | 代码：[cca][cca(code)].cca()** | **2022/7/10**
#### 1.2 扩展 CCA：eCCA
**(Extended CCA)**
**[论文链接][eCCA] | 代码：[cca][cca(code)].ecca()**
#### 1.3 多重刺激 CCA：msCCA
**(Multi-stimulus CCA)**
**[论文链接][msCCA] | 代码：[cca][cca(code)].mscca()**
#### 1.x 跨个体空间滤波器迁移：CSSFT
**(Cross-subject spatial filter transfer method)**
**[论文链接][CSSFT] | 代码：[cca][cca(code)].cssft()**

[cca(code)]: https://github.com/BrynhildrW/SSVEP_algorithms/blob/main/cca.py
[CCA]: http://ieeexplore.ieee.org/document/4203016/
[eCCA]: http://www.pnas.org/lookup/doi/10.1073/pnas.1508080112
[msCCA]: https://ieeexplore.ieee.org/document/9006809/
[CSSFT]: http://iopscience.iop.org/article/10.1088/1741-2552/ac6b57

***
### 2. 多变量同步化系数
**Multivariate synchronization index, MSI**
#### 2.1 标准 MSI：MSI
**[论文链接][MSI] | 代码：[msi][msi(code)].msi()**
#### 2.2 时域局部 MSI：tMSI
**(Temporally MSI)**
**[论文链接][tMSI] | 代码：[msi][msi(code)].tmsi()**
#### 2.3 扩展 MSI：eMSI
**(Extended MSI)**
**[论文链接][MSI] | 代码：[msi][msi(code)].emsi()**
[msi(code)]: temp
[MSI]: https://linkinghub.elsevier.com/retrieve/pii/S0165027013002677
[tMSI]: http://link.springer.com/10.1007/s11571-016-9398-9
[eMSI]: https://linkinghub.elsevier.com/retrieve/pii/S0925231217309980
***
### 3. 任务相关成分分析
**Task-related component analysis, TRCA**
#### 3.1 普通/集成 TRCA：(e)TRCA
**( (Ensemble) TRCA, (e)TRCA）**
**[论文链接][TRCA] | 代码：[trca][trca(code)].etrca()** | **2022/7/16**
[trca(code)]: temp
[TRCA]: https://ieeexplore.ieee.org/document/7904641/
#### 3.2 正余弦扩展 TRCA：(e)TRCA-R
**[论文链接][TRCA-R] | 代码：[trca][trca(code)].etrca_r()**
[TRCA-R]: https://ieeexplore.ieee.org/document/9006809/
#### 3.3 多重刺激 TRCA：ms-(e)TRCA
**(Multi-stimulus (e)TRCA)**
**[论文链接][ms-TRCA] | 代码：[trca][trca(code)].ms_etrca()** | **2022/7/23**
[ms-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
[Benchmark]:https://ieeexplore.ieee.org/document/7740878/
[UCSD]:https://dx.plos.org/10.1371/journal.pone.0140703
[BETA]:https://www.frontiersin.org/article/10.3389/fnins.2020.00627/full
#### 3.4 相似度约束 TRCA：sc-(e)TRCA
**(Similarity-constrained (e)TRCA)**
**[论文链接][sc-TRCA] | 代码：[trca][trca(code)].sc_etrca()**
#### 3.5 组 TRCA：gTRCA
**(Group TRCA)**
**[论文链接][gTRCA] | 代码：[trca][trca(code)].gtrca()**
#### 3.6 交叉相关性 TRCA：xTRCA
**(Cross-correlation TRCA)**
**[论文链接][xTRCA] | 代码：[trca][trca(code)].xtrca()**
[sc-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/abfdfa
[gTRCA]: temp
[xTRCA]: temp
[TDCA]: https://ieeexplore.ieee.org/document/9541393/
***
### x. 其它早期算法
#### x.1 最小能量组合：MEC
**(Minimun energy combination)**
**[论文链接][MEC] | 代码：[other][other(code)].mec()**
#### x.2 最大对比度组合：MCC
**Maximun contrast combination, MCC**
**[论文链接][MCC] | 代码：[other][other(code)].mcc()**
[other(code)]: temp
[MEC]: http://ieeexplore.ieee.org/document/4132932/
[MCC]: http://ieeexplore.ieee.org/document/4132932/
***






