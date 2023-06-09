# 跨个体迁移 TRCA
## Cross-subject transfer learning TRCA, TL-TRCA
***

[论文链接][TL-TRCA]

2023 年新鲜出炉的这一篇 TRCA 改进算法看上去好像集合了诸多文献的要义：[sc-(e)TRCA][ref1] 的正余弦模板拼接、[PT 投影][ref2]的最小二乘投影以及 [gTRCA][ref3] 的多受试者数据整合。可惜没能用上 [tlCCA][ref3] 的最小二乘受试者融合权重分配技术。为什么要说“好像”，是因为实际测试结果以及文献汇报的结果似乎没有想象得那么出众。当然这并不妨碍作者提出了一些富有新意的改进思路。

![TL-TRCA示意图](TL-TRCA.png)



***

[TL-TRCA]: https://ieeexplore.ieee.org/document/10057002/
[ref1]: https://iopscience.iop.org/article/10.1088/1741-2552/abfdfa
[ref2]: https://iopscience.iop.org/article/10.1088/1741-2552/abcb6e
[ref3]: http://www.nature.com/articles/s41598-019-56962-2

