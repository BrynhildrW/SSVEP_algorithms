Github的公式编辑有一些问题，每个算法的详细说明与公式推导我都放在个人的 FlowUs 页面中了

>典型相关性分析：https://flowus.cn/brynhildrw/share/7cca304a-f180-4c56-81fe-a3f3b746fd8c
>
>任务相关成分分析：https://flowus.cn/brynhildrw/share/e4cccd8a-c268-4cdb-9167-15b45d659ede
>
>数据增强 & 迁移学习算法：https://flowus.cn/brynhildrw/share/c31fcb70-2065-4ba5-9b7e-0f66e546b8fc

访问密码都是：tjuw。欢迎各位同行与我合理讨论与交流：215707269@qq.com

---

最近更新：2024/08/04

（1）在`utils`里更新了一些基本步骤的功能性子函数，如`generate_data_info()`、`spatial_filtering()`等等，进而优化了`cca`、`trca`、`dsp`以及`transfer`四个基本库的代码结构

---

未来更新计划

（1）cold-start

---

目前已支持：

（1）CCA 系列：`cca.CCA()`、`cca.MEC()`、`cca.MCC()`、`cca.MSI()`、`cca.ITCCA()`、`cca.ECCA()`、`cca.MSCCA()`、`cca.MS_ECCA`、`cca.MSETCCA1()`、`cca.TDCCA()` 以及各自的 filter-bank 版本

（2）TRCA 系列：`trca.TRCA()`、`trca.MS_TRCA()`、`trca.TRCA_R()`、`trca.SC_TRCA()` 以及各自的集成版本

（3）DSP 系列：`dsp.DSP()`、`dsp.TDCA()` 以及各自的 filter-bank 版本

（4）迁移学习算法：`transfer.TNSRE_20233250953()`、`transfer.STCCA()`、`transfer.TLCCA()`、`transfer.SDLST()`、`transfer.TNSRE_20233305202()`、`transfer.IISMC()`、`transfer.ASS_IISCCA()`、`transfer.ALPHA()` 以及各自的 filter-bank 版本
