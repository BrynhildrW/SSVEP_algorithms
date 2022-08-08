# 常见 SSVEP 信号处理算法（空间滤波器）

Github 的在线 Markdown 公式编译模块有 bug，想看公式过程的建议下载各章节PDF文件本地查看。鉴于部分公式推导步骤属于本人硕士学位论文内容，在此提醒各位，ctrl+C/V 请慎重。

目前我会以算法公式推导工作为主，大概频率是一周一个算法。后续会依次补上代码部分。凡标注（已完成）的算法，表示其说明文档与程序均已完成。

***
## 1. 典型相关性分析
**Canonical correlation analysis, CCA**

### 1.1 标准 CCA：CCA
**[论文链接][CCA] | 代码：[cca][cca(code)].cca()** （已完成）

### 1.2 扩展 CCA：eCCA
**(Extended CCA)** <br>
**[论文链接][eCCA] | 代码：[cca][cca(code)].ecca()**

### 1.3 多重刺激 CCA：msCCA
**(Multi-stimulus CCA)** <br>
**[论文链接][msCCA] | 代码：[cca][cca(code)].mscca()**

### 1.4 多重刺激扩展 CCA：ms-eCCA
**(Multi-stimulus CCA)** <br>
**[论文链接][ms-eTRCA] | 代码：[cca][cca(code)].msecca()**

### 1.x 跨个体空间滤波器迁移：CSSFT
**(Cross-subject spatial filter transfer method)** <br>
**[论文链接][CSSFT] | 代码：[cca][cca(code)].cssft()**

[cca(code)]: https://github.com/BrynhildrW/SSVEP_algorithms/blob/main/cca.py
[CCA]: http://ieeexplore.ieee.org/document/4203016/
[eCCA]: http://www.pnas.org/lookup/doi/10.1073/pnas.1508080112
[msCCA]: https://ieeexplore.ieee.org/document/9006809/
[CSSFT]: http://iopscience.iop.org/article/10.1088/1741-2552/ac6b57
***

## 2. 多变量同步化系数
**Multivariate synchronization index, MSI**

### 2.1 标准 MSI：MSI
**[论文链接][MSI] | 代码：[msi][msi(code)].msi()**

### 2.2 时域局部 MSI：tMSI
**(Temporally MSI)** <br>
**[论文链接][tMSI] | 代码：[msi][msi(code)].tmsi()**

### 2.3 扩展 MSI：eMSI
**(Extended MSI)** <br>
**[论文链接][MSI] | 代码：[msi][msi(code)].emsi()**

[msi(code)]: temp
[MSI]: https://linkinghub.elsevier.com/retrieve/pii/S0165027013002677
[tMSI]: http://link.springer.com/10.1007/s11571-016-9398-9
[eMSI]: https://linkinghub.elsevier.com/retrieve/pii/S0925231217309980
***

## 3. 任务相关成分分析
**Task-related component analysis, TRCA**

### 3.1 普通/集成 TRCA：(e)TRCA
**( (Ensemble) TRCA, (e)TRCA）** <br>
**[论文链接][TRCA] | 代码：[trca][trca(code)].etrca()** （已完成）

### 3.2 多重刺激 TRCA：ms-(e)TRCA
**(Multi-stimulus (e)TRCA)** <br>
**[论文链接][ms-TRCA] | 代码：[trca][trca(code)].ms_etrca()**

### 3.3 正余弦扩展 TRCA：(e)TRCA-R
**[论文链接][TRCA-R] | 代码：[trca][trca(code)].etrca_r()**

### 3.4 相似度约束 TRCA：sc-(e)TRCA
**(Similarity-constrained (e)TRCA)** <br>
**[论文链接][sc-TRCA] | 代码：[trca][trca(code)].sc_etrca()**

### 3.5 组 TRCA：gTRCA
**(Group TRCA)** <br>
**[论文链接][gTRCA] | 代码：[trca][trca(code)].gtrca()**

### 3.6 交叉相关性 TRCA：xTRCA
**(Cross-correlation TRCA)** <br>
**[论文链接][xTRCA] | 代码：[trca][trca(code)].xtrca()**

[trca(code)]: temp
[TRCA]: https://ieeexplore.ieee.org/document/7904641/
[TRCA-R]: https://ieeexplore.ieee.org/document/9006809/
[ms-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/ab2373
[sc-TRCA]: https://iopscience.iop.org/article/10.1088/1741-2552/abfdfa
[gTRCA]: temp
[xTRCA]: temp
[TDCA]: https://ieeexplore.ieee.org/document/9541393/
***

## x. 其它早期算法

### x.1 最小能量组合：MEC
**(Minimun energy combination)** <br>
**[论文链接][MEC] | 代码：[other][other(code)].mec()**

### x.2 最大对比度组合：MCC
**Maximun contrast combination, MCC** <br>
**[论文链接][MCC] | 代码：[other][other(code)].mcc()**

[other(code)]: temp
[MEC]: http://ieeexplore.ieee.org/document/4132932/
[MCC]: http://ieeexplore.ieee.org/document/4132932/
***
