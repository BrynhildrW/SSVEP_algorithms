Github的公式编辑有一些问题，每个算法的详细说明与公式推导我都放在个人的 FlowUs 页面中了

>典型相关性分析：https://flowus.cn/brynhildrw/share/7cca304a-f180-4c56-81fe-a3f3b746fd8c
>
>任务相关成分分析：https://flowus.cn/brynhildrw/share/e4cccd8a-c268-4cdb-9167-15b45d659ede
>
>数据增强 & 迁移学习算法：https://flowus.cn/brynhildrw/share/c31fcb70-2065-4ba5-9b7e-0f66e546b8fc

访问密码都是：tjuw。欢迎各位同行与我合理讨论与交流：215707269@qq.com

.md 文件以及使用的图片在三个中文命名的文件夹中，代码文件在 program 文件夹里。其它文件可能是废稿，可以不用管他们。

---

最近更新：2024/10/22

（1）更新了论文 “Leveraging Transfer Superposition Theory for StableState Visual Evoked Potential Cross-Subject Frequency Recognition” 的算法：`transfer.TBME_20243406603()`

（2）更新了论文 “Cross-Stimulus Transfer Method Using Common Impulse Response for Fast Calibration of SSVEP-Based BCIs” 的算法：`transfer.TIM_20243374314()`

---

未来更新计划

（1）在 flowUs 页面更新 DSP、TDCA、ASS-IISCCA 等代码已包含在库中、但是没有专栏页面的算法。

---

目前已支持：

（1）CCA 系列：`cca.CCA()`、`cca.MEC()`、`cca.MCC()`、`cca.MSI()`、`cca.ITCCA()`、`cca.ECCA()`、`cca.MSCCA()`、`cca.MS_ECCA`、`cca.MSETCCA1()`、`cca.TDCCA()` 以及各自的 filter-bank 版本

（2）TRCA 系列：`trca.TRCA()`、`trca.MS_TRCA()`、`trca.TRCA_R()`、`trca.SC_TRCA()` 以及各自的集成版本

（3）DSP 系列：`dsp.DSP()`、`dsp.TDCA()` 以及各自的 filter-bank 版本

（4）迁移学习算法：`transfer.TNSRE_20233250953()`、`transfer.STCCA()`、`transfer.TLCCA()`、`transfer.SDLST()`、`transfer.TNSRE_20233305202()`、`transfer.IISMC()`、`transfer.ASS_IISCCA()`、`transfer.ALPHA()`、`transfer.TIM_20243374314()`、`transfer.TBME_20243406603()`以及各自的 filter-bank 版本
