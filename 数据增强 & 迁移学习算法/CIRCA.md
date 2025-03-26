# 基于冲激响应成分分析的 SSVEP 解码算法
## Impulse response component analysis, IRCA
***

2024 年 Xiong Bang 在 TNSRE 上发表了一篇

$$
\hat{\pmb{r}}, \ \hat{\pmb{w}} = \underset{\pmb{r}, \ \pmb{w}}{\arg\min} \sum_{k=1}^{N_e} \left\| \pmb{r} \pmb{H}_k - \pmb{w} \bar{\pmb{X}}_k \right\|_2^2
$$

$$
\begin{cases}
\underset{\pmb{r}, \ \pmb{w}}{\min} \sum_{k=1}^{N_e} \left\| \pmb{r} \pmb{H}_k^{'} - \pmb{w} \bar{\pmb{X}}_k \right\|_2^2\\
\ \\
{\rm s.t.} \ \ \pmb{r}{\pmb{r}}^T = \pmb{w}{\pmb{w}}^T = 1
\end{cases} \ \ \Longrightarrow \ \ \hat{\pmb{r}}, \ \hat{\pmb{w}} = \underset{\pmb{r}, \ \pmb{w}}{\arg\min} \dfrac{\pmb{r} \left(\sum_{k=1}^{N_e} \pmb{H}_k^{'} {\bar{\pmb{X}}_k}^T \right) {\pmb{w}}^T }{\sqrt{\pmb{r} \left( \sum_{k=1}^{N_e} \pmb{H}_k^{'} {\pmb{H}_k^{'}}^T \right) {\pmb{r}}^T} \sqrt{\pmb{w} \left( \sum_{k=1}^{N_e} \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T \right) {\pmb{w}}^T}}
$$

$$
\hat{\pmb{r}}, \ \hat{\pmb{w}} = \underset{\pmb{r}, \ \pmb{w}}{\arg\min} \dfrac{\pmb{r} \left(\sum_{k=1}^{N_e} \pmb{H}_k {\bar{\pmb{X}}_k}^T \right) {\pmb{w}}^T }{\sqrt{\pmb{r} \left( \sum_{k=1}^{N_e} \pmb{H}_k {\pmb{H}_k}^T \right) {\pmb{r}}^T} \sqrt{\pmb{w} \left( \sum_{k=1}^{N_e} \bar{\pmb{X}}_k {\bar{\pmb{X}}_k}^T \right) {\pmb{w}}^T}}
$$

$$
{\rm Lanczos} (x) = \begin{cases}
{\rm sinc} (x) {\rm sinc} \left(\dfrac{x}{a} \right), \ \ |x| < a\\
0, \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ |x| \geqslant a
\end{cases}, \ \ \ {\rm sinc} (x) = \dfrac{{\rm sin} (\pi x)}{\pi x}
$$

$$
\pmb{H}_k^{'} = \pmb{H}_k \left( \pmb{I}_{N_p}+
\begin{bmatrix}
\pmb{0}^{(N_p-N_l)\times N_l} & \pmb{0}^{(N_p - N_l)\times(N_p-N_l)}\\
\ \\
\pmb{I}_{N_l} & \pmb{0}^{N_l \times(N_p-N_l)}
\end{bmatrix} \right)
$$

$$
\begin{cases}
\rho_{k,1} = {\rm corr} \left( \hat{\pmb{w}} \pmb{\mathcal{X}}, \ \ \hat{\pmb{w}} \bar{\pmb{X}}_k \right)\\
\ \\
\rho_{k,2} = {\rm corr} \left( \hat{\pmb{w}} \pmb{\mathcal{X}}, \ \ \hat{\pmb{r}} \pmb{H}_k^{'} \right)
\end{cases}, \ \ \ \ \rho_k = \sum_{i=1}^{2} {\rm sign} (\rho_{k,i}) \rho_{k,i}^2
$$

$$
\hat{\pmb{P}}_k^{(s)} = \underset{\pmb{P}_k^{(s)}}{\arg\min} \dfrac{1 - \rho}{N_t^{(s)}} \sum_{i=1}^{N_t^{(s)}} \left\| \pmb{w}^{(s)} \pmb{X}_k^{i,(s)} \pmb{P}_k^{(s)} - \pmb{w}^{(\tau)} \pmb{X}_k^{(\tau)} \right\|_F^2 + \rho \left\| \pmb{P}_k^{(s)} \right\|_F^2
$$

$$
\hat{\pmb{P}}^{(s)}_k = \underset{\pmb{P}_k^{(s)}}{\arg\min} \left\| \pmb{w}^{(s)} \bar{\pmb{X}}_k^{(s)} - \pmb{P}_k^{(s)} \pmb{w}^{(\tau)} \bar{\pmb{X}}_k^{(\tau)} \right\|_F^2, \ \ \ \ {\rm s.t.} \ {\pmb{P}_k^{(s)}}^T \pmb{P}_k^{(s)} = \pmb{I}
$$

$$
\hat{\pmb{Q}}^{(s)} = \underset{\pmb{Q}^{(s)}}{\arg\min} \left\| {\pmb{Q}^{(s)}}^T \pmb{C}^{(s)} \pmb{Q}^{(s)} - \pmb{C}^{(\tau)} \right\|_F^2 = {\pmb{C}^{(s)}}^{-\frac{1}{2}} {\pmb{C}^{(\tau)}}^{\frac{1}{2}}, \ \ \ \ \begin{cases}
\pmb{C}^{(s)} = \dfrac{\sum_{k=1}^{N_e}\sum_{i=1}^{N_t^{(s)}} \pmb{X}_k^{i,(s)} {\pmb{X}_k^{i,(s)}}^T}{N_eN_t^{(s)} N_p - 1}\\
\ \\
\pmb{C}^{(\tau)} = \dfrac{\sum_{k=1}^{N_e}\sum_{i=1}^{N_t^{(\tau)}} \pmb{X}_k^{i,(\tau)} {\pmb{X}_k^{i,(\tau)}}^T}{N_eN_t^{(\tau)} N_p - 1}
\end{cases}
$$

$$
\rho_k = {\rm corr} \left( \hat{\pmb{w}}^{(\tau)} \pmb{\mathcal{X}}, \ \ \hat{\pmb{w}}^{(\tau)} {\hat{\pmb{Q}}^{(s)}}^T \bar{\pmb{X}}_k^{(s)} \right) \ \ \ {\rm or} \ \ \ \rho_k = {\rm corr} \left( \hat{\pmb{w}}^{(s)} {\hat{\pmb{Q}}^{(s)}}^{-T} \pmb{\mathcal{X}}, \ \ \hat{\pmb{w}}^{(s)} \bar{\pmb{X}}_k^{(s)} \right)
$$

$$
\hat{\pmb{\psi}} = \underset{\pmb{\psi}}{\arg\min} \left\| \pmb{\psi} \pmb{A}_{\pmb{\psi}} - \pmb{b}_{\pmb{\psi}} \right\|_F^2, \ \ \ \ \hat{\pmb{\gamma}} = \underset{\pmb{\gamma}}{\arg\min} \left\| \pmb{\gamma} \pmb{A}_{\pmb{\gamma}} - \pmb{b}_{\pmb{\gamma}} \right\|_F^2
$$

$$
\begin{align}
\notag \hat{\pmb{\psi}} &= \underset{\pmb{\psi}}{\arg\min} \left\| \pmb{\psi} \pmb{A}_{\pmb{\psi}} - \pmb{b}_{\pmb{\psi}} \right\|_F^2, \ \ \ \ \pmb{A}_{\pmb{\psi}} = \begin{bmatrix}
\hat{\pmb{w}}^{(1)} \bar{\pmb{X}}_1^{(1)} & \cdots & \hat{\pmb{w}}^{(1)} \bar{\pmb{X}}_{N_e}^{(1)}\\
\vdots & \ddots & \vdots\\
\hat{\pmb{w}}^{(N_s)} \bar{\pmb{X}}_1^{(N_s)} & \cdots & \hat{\pmb{w}}^{(N_s)} \bar{\pmb{X}}_{N_e}^{(N_s)}\\
\end{bmatrix}, \ \ \ \ \pmb{b}_{\pmb{\psi}} = \begin{bmatrix}
\tilde{\pmb{w}}^{(\tau)} \bar{\pmb{X}}_1^{(\tau)} & \cdots & \tilde{\pmb{w}}^{(\tau)} \bar{\pmb{X}}_{N_e}^{(\tau)}
\end{bmatrix}\\
\notag \ \\
\notag \hat{\pmb{\gamma}} &= \underset{\pmb{\gamma}}{\arg\min} \left\| \pmb{\gamma} \pmb{A}_{\pmb{\gamma}} - \pmb{b}_{\pmb{\gamma}} \right\|_F^2, \ \ \ \ \pmb{A}_{\pmb{\gamma}} = \begin{bmatrix}
\hat{\pmb{r}}^{(1)} \pmb{H}_1^{'} & \cdots & \hat{\pmb{r}}^{(1)} \pmb{H}_{N_e}^{'}\\
\vdots & \ddots & \vdots\\
\hat{\pmb{r}}^{(N_s)} \pmb{H}_1^{'} & \cdots & \hat{\pmb{r}}^{(N_s)} \pmb{H}_{N_e}^{'}\\
\end{bmatrix}, \ \ \ \ \pmb{b}_{\pmb{\gamma}} = \begin{bmatrix}
\tilde{\pmb{r}}^{(\tau)} \pmb{H}_1^{'} & \cdots & \tilde{\pmb{r}}^{(\tau)} \pmb{H}_{N_e}^{'}
\end{bmatrix}\\
\end{align}
$$

$$
\begin{cases}
\rho_{k,1} = {\rm corr} \left( \hat{\pmb{w}}^{(\tau)} \pmb{\mathcal{X}}, \ \ \hat{\pmb{w}}^{(\tau)} \bar{\pmb{X}}_k^{(\tau)} \right)\\
\ \\
\rho_{k,2} = {\rm corr} \left( \hat{\pmb{w}}^{(\tau)} \pmb{\mathcal{X}}, \ \ \hat{\pmb{r}}^{(\tau)} \pmb{H}_k \right)\\
\ \\
\rho_{k,3} = {\rm corr} \left( \hat{\pmb{w}}^{(\tau)} \pmb{\mathcal{X}}, \ \ \sum_{s=1}^{N_s} \hat{\pmb{\psi}}(s) \hat{\pmb{w}}^{(s)} \bar{\pmb{X}}_k^{(s)} \right)\\
\ \\
\rho_{k,4} = {\rm corr} \left( \hat{\pmb{w}}^{(\tau)} \pmb{\mathcal{X}}, \ \ \sum_{s=1}^{N_s} \hat{\pmb{\gamma}}(s) \hat{\pmb{r}}^{(s)} \pmb{H}_k \right)\\
\end{cases}, \ \ \ \ \rho = \sum_{i} {\rm sign} \left( \rho_{k,i} \right) {\rho_{k,i}}^2
$$



$$
\begin{cases}
\left\{ \hat{\pmb{r}}, \ \hat{\pmb{w}} \right\} = \underset{\pmb{r}, \ \pmb{w}}{\arg\min} \left\| \pmb{r} \widetilde{\pmb{H}}_k - \pmb{w} \widetilde{\pmb{X}}_k \right\|_2^2\\
\ \\
{\rm s.t.} \ \ \pmb{r}{\pmb{r}}^T = \pmb{w}{\pmb{w}}^T = 1
\end{cases} \ \ \underset{{\rm ALS}}{\Longrightarrow} \ \ \begin{cases}
\pmb{r} = \pmb{w} \widetilde{\pmb{X}}_k {\widetilde{\pmb{H}}_k}^T {\left(\widetilde{\pmb{H}}_k {\widetilde{\pmb{H}}_k}^T \right)}^{-1}\\
\ \\
\pmb{w} = \pmb{r} \widetilde{\pmb{H}}_k {\widetilde{\pmb{X}}_k}^T {\left(\widetilde{\pmb{X}}_k {\widetilde{\pmb{X}}_k}^T \right)}^{-1}
\end{cases}
$$

$$
\begin{cases}
\rho_{k,1} = {\rm corr} \left( \hat{\pmb{w}} \pmb{\mathcal{X}}, \ \ \hat{\pmb{r}} \pmb{H}_k \right)\\
\ \\
\rho_{k,2} = {\rm CCA} \left( \pmb{\mathcal{X}}, \ \ \pmb{Y}_k \right)
\end{cases}, \ \ \ \ \rho_k = \sum_{i=1}^{2} {\rm sign} (\rho_{k,i}) \rho_{k,i}^2
$$