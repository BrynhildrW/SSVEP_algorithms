$$
\begin{align}
\notag \hat{\pmb{\psi}} &= \underset{\pmb{\psi}}{\arg\max} \sum_{s=1}^{N_s} {\left( \dfrac{\pmb{\psi} {\tilde{\pmb{w}}^{(s)}}^T}{\left\| \pmb{\psi} \right\|_2 \left\| \tilde{\pmb{w}}^{(s)} \right\|_2} \right)}^2 = \underset{\pmb{\psi}}{\arg\max} \dfrac{\pmb{\psi} \pmb{C}_{\pmb{w}} {\pmb{\psi}}^T}{\pmb{\psi} {\pmb{\psi}}^T}, \ \ \ \ \pmb{C}_{\pmb{w}} = \sum_{s=1}^{N_s} {\tilde{\pmb{w}}^{(s)}}^T \tilde{\pmb{w}}^{(s)}, \ \ \tilde{\pmb{w}}^{(s)} = \dfrac{\pmb{w}^{(s)}}{\left\| \pmb{w}^{(s)} \right\|_2}\\
\notag \ \\
\notag \hat{\pmb{\gamma}} &= \underset{\pmb{\gamma}}{\arg\max} \sum_{s=1}^{N_s} {\left( \dfrac{\pmb{\gamma} {\tilde{\pmb{r}}^{(s)}}^T}{\left\| \pmb{\gamma} \right\|_2 \left\| \tilde{\pmb{r}}^{(s)} \right\|_2} \right)}^2 = \underset{\pmb{\gamma}}{\arg\max} \dfrac{\pmb{\gamma} \pmb{C}_{\pmb{r}} {\pmb{\gamma}}^T}{\pmb{\gamma} {\pmb{\gamma}}^T}, \ \ \ \ \pmb{C}_{\pmb{r}} = \sum_{s=1}^{N_s} {\tilde{\pmb{r}}^{(s)}}^T \tilde{\pmb{r}}^{(s)}, \ \ \tilde{\pmb{r}}^{(s)} = \dfrac{\pmb{r}^{(s)}}{\left\| \pmb{r}^{(s)} \right\|_2}
\end{align}
$$
$$
\begin{align}
\notag \hat{\pmb{\psi}} &= \underset{\pmb{\psi}}{\arg\min} \left\| \pmb{\psi} \pmb{A}_{\pmb{\psi}} - \pmb{b}_{\pmb{\psi}} \right\|_F^2, \ \ \ \ \pmb{A}_{\pmb{\psi}} =
\begin{bmatrix}
\tilde{\pmb{w}}^{(1)} \bar{\pmb{X}}_1^{(1)} & \tilde{\pmb{w}}^{(1)} \bar{\pmb{X}}_2^{(1)} & \cdots & \tilde{\pmb{w}}^{(1)} \bar{\pmb{X}}_{N_e}^{(1)}\\
\ \\
\tilde{\pmb{w}}^{(2)} \bar{\pmb{X}}_1^{(2)} & \tilde{\pmb{w}}^{(2)} \bar{\pmb{X}}_2^{(2)} & \cdots & \tilde{\pmb{w}}^{(2)} \bar{\pmb{X}}_{N_e}^{(2)}\\
\ \\
\vdots & \vdots & \ddots & \vdots\\
\ \\
\tilde{\pmb{w}}^{(N_s)} \bar{\pmb{X}}_1^{(N_s)} & \tilde{\pmb{w}}^{(N_s)} \bar{\pmb{X}}_2^{(N_s)} & \cdots & \tilde{\pmb{w}}^{(N_s)} \bar{\pmb{X}}_{N_e}^{(N_s)}\\
\end{bmatrix}, \ \ \pmb{b}_{\pmb{\psi}} =
\begin{bmatrix}
\tilde{\pmb{w}}^{(\tau)} \bar{\pmb{X}}_1^{(\tau)} & \tilde{\pmb{w}}^{(\tau)} \bar{\pmb{X}}_2^{(\tau)} & \cdots & \tilde{\pmb{w}}^{(\tau)} \bar{\pmb{X}}_{N_e}^{(\tau)}
\end{bmatrix}\\
\notag \ \\
\notag \hat{\pmb{\gamma}} &= \underset{\pmb{\gamma}}{\arg\min} \left\| \pmb{\gamma} \pmb{A}_{\pmb{\gamma}} - \pmb{b}_{\pmb{\gamma}} \right\|_F^2, \ \ \ \ \pmb{A}_{\pmb{\gamma}} =
\begin{bmatrix}
\tilde{\pmb{r}}^{(1)} \bar{\pmb{H}}_1^{(1)} & \tilde{\pmb{r}}^{(1)} \bar{\pmb{H}}_2^{(1)} & \cdots & \tilde{\pmb{r}}^{(1)} \bar{\pmb{H}}_{N_e}^{(1)}\\
\ \\
\tilde{\pmb{r}}^{(2)} \bar{\pmb{H}}_1^{(2)} & \tilde{\pmb{r}}^{(2)} \bar{\pmb{H}}_2^{(2)} & \cdots & \tilde{\pmb{r}}^{(2)} \bar{\pmb{H}}_{N_e}^{(2)}\\
\ \\
\vdots & \vdots & \ddots & \vdots\\
\ \\
\tilde{\pmb{r}}^{(N_s)} \bar{\pmb{H}}_1^{(N_s)} & \tilde{\pmb{r}}^{(N_s)} \bar{\pmb{H}}_2^{(N_s)} & \cdots & \tilde{\pmb{r}}^{(N_s)} \bar{\pmb{H}}_{N_e}^{(N_s)}\\
\end{bmatrix}, \ \ \pmb{b}_{\pmb{\gamma}} =
\begin{bmatrix}
\tilde{\pmb{r}}^{(\tau)} \bar{\pmb{H}}_1^{(\tau)} & \tilde{\pmb{r}}^{(\tau)} \bar{\pmb{H}}_2^{(\tau)} & \cdots & \tilde{\pmb{r}}^{(\tau)} \bar{\pmb{H}}_{N_e}^{(\tau)}
\end{bmatrix}\\
\end{align}
$$
$$
\begin{align}
\notag \hat{\pmb{P}}_{\pmb{r}} &= \underset{\pmb{P}_{\pmb{r}}}{\arg\min} \dfrac{1 - \rho}{N_s} \sum_{s=1}^{N_s} \left\| \pmb{r}^{(s)} \pmb{P}_{\pmb{r}} - \pmb{r}^{(\tau)} \right\|_F^2 + \rho \left\| \pmb{P}_{\pmb{r}} \right\|_F^2\\
\notag \ \\
\notag \hat{\pmb{P}}_{\pmb{w},k} &= \underset{\pmb{P}_{\pmb{w},k}}{\arg\min} \dfrac{1 - \rho}{N_s} \sum_{s=1}^{N_s} \left\| \pmb{w}^{(s)} \bar{\pmb{X}}_k^{(s)} \pmb{P}_{\pmb{w},k} - \pmb{w}^{(\tau)} \bar{\pmb{X}}_k^{(\tau)} \right\|_F^2 + \rho \left\| \pmb{P}_{\pmb{w},k} \right\|_F^2\\
\end{align}
$$
$$
\eta_{\pmb{r}}^{(s)} = \dfrac{{\rm corr} \left( \pmb{r}^{(\tau)}, \ \ \pmb{r}^{(s)} \right)}{\sum_{s=1}^{N_s} {\rm corr} \left( \pmb{r}^{(\tau)}, \ \ \pmb{r}^{(s)} \right)}, \ \ \ \ \eta_{\pmb{w}}^{(s)} = \dfrac{\sum_{k=1}^{N_e} {\rm corr} \left( \pmb{w}^{(\tau)} \bar{\pmb{X}}_k^{(\tau)}, \ \ \pmb{w}^{(s)} \bar{\pmb{X}}_k^{(s)} \right)}{\sum_{s=1}^{N_s} \sum_{k=1}^{N_e} {\rm corr} \left( \pmb{w}^{(\tau)} \bar{\pmb{X}}_k^{(\tau)}, \ \ \pmb{w}^{(s)} \bar{\pmb{X}}_k^{(s)} \right)}
$$
$$
\hat{\pmb{P}}^{(\tau, s)} = \underset{\pmb{P}^{(\tau, s)}}{\arg\min} \left\| \pmb{r}^{(s)} - \pmb{r}^{(\tau)} {\pmb{P}^{(\tau, s)}}^T \right\|_F^2, \ \ \ \ s.t. \ \ \pmb{P}^{(\tau, s)} {\pmb{P}^{(\tau, s)}}^T = \pmb{I}
$$
$$
\begin{cases}
\hat{\pmb{A}}_k^{(s)} = \underset{\pmb{A}_k^{(s)}}{\arg\min} \dfrac{1}{N_t^{(s)}} \left\| \pmb{A}_k^{(s)} \pmb{w}^{(s)} \bar{\pmb{X}}_k^{(s)} - \pmb{X}_k^{i,(s)} \right\|_F^2\\
\ \\
\hat{\pmb{A}}_k^{(\tau)} = \underset{\pmb{A}_k^{(\tau)}}{\arg\min} \dfrac{1}{N_t^{(\tau)}} \left\| \pmb{A}_k^{(\tau)} \pmb{w}^{(\tau)} \bar{\pmb{X}}_k^{(\tau)} - \pmb{X}_k^{i,(\tau)} \right\|_F^2\\
\end{cases}, \ \ \ \ \hat{\pmb{P}}_k^{(\tau, s)} = \underset{\pmb{P}_k^{(\tau, s)}}{\arg\min} \left\| \hat{\pmb{A}}_k^{(s)} - \hat{\pmb{A}}_k^{(\tau)} {\pmb{P}_k^{(\tau, s)}}^T \right\|_F^2, \ \ \ \ s.t. \ \ \pmb{P}_k^{(\tau, s)} {\pmb{P}_k^{(\tau, s)}}^T = \pmb{I}
$$

$$
\rho_k = {\rm corr} \left( {\hat{\pmb{P}}_k^{(\tau, s)}}^T \hat{\pmb{w}}^{(\tau)} \pmb{\mathcal{X}}, \ \ \hat{\pmb{w}}^{(s)} \bar{\pmb{X}}_k^{(s)} \right)
$$
$$
\hat{\pmb{Q}}_k^{(s)} = \underset{\pmb{Q}_k^{(s)}}{\arg\min} \left\| {\pmb{Q}_k^{(s)}}^T \pmb{C}_k^{(s)} \pmb{Q}_k^{(s)} - \pmb{C}_k^{(\tau)} \right\|_F^2, \ \ \ \ \hat{\pmb{Q}}_k^{(s)} = {\pmb{C}_k^{(s)}}^{-\frac{1}{2}} {\pmb{C}_k^{(\tau)}}^{\frac{1}{2}}
$$

$$
\pmb{C}_k^{(s)} = \dfrac{1}{N_t^{(s)} N_p - 1} \sum_{i=1}^{N_t^{(s)}} \pmb{X}_k^{i,(s)} {\pmb{X}_k^{i,(s)}}^T, \ \ \ \ 
\begin{cases}
\pmb{C}_k^{(\tau)} = \dfrac{1}{N_t^{(\tau)} N_p - 1} \sum_{i=1}^{N_t^{(\tau)}} \pmb{X}_k^{i,(\tau)} {\pmb{X}_k^{i,(\tau)}}^T\\
\ \\
\pmb{C}^{(\tau)} = \dfrac{1}{N_p - 1} \pmb{\mathcal{X}} {\pmb{\mathcal{X}}}^T
\end{cases}
$$

$$
\rho_k = {\rm corr} \left( \hat{\pmb{w}}^{(\tau)} \pmb{\mathcal{X}}, \ \ \hat{\pmb{w}}^{(\tau)} {\hat{\pmb{Q}}_k^{(s)}}^T \bar{\pmb{X}}_k^{(s)}\right) \ \ {\rm or} \ \ \rho_k = {\rm corr} \left( \hat{\pmb{w}}^{(s)} {\hat{\pmb{Q}}_k^{(s)}}^{-T} \pmb{\mathcal{X}}, \ \ \hat{\pmb{w}}^{(s)} \bar{\pmb{X}}_k^{(s)}\right)
$$
$$
\rho_{k,1} = \dfrac{1}{N_s} \sum_{s=1}^{N_s} {\rm corr} \left( \tilde{\pmb{w}}_k^{(s)} \pmb{\mathcal{X}}, \ \ \tilde{\pmb{r}}_k^{(s)} \pmb{H}_k \right), \ \ \ \ \rho_{k,2} = {\rm corr} \left( \hat{\pmb{\psi}}_k \pmb{\mathcal{X}}, \ \ \hat{\pmb{\gamma}}_k \pmb{H}_k \right), \ \ \ \ \rho_{k,3} = {\rm CCA} \left( \pmb{\mathcal{X}}, \ \ \pmb{Y}_k \right), \ \ \ \ \rho_{k,4} = \dfrac{1}{N_s} \sum_{s=1}^{N_s} {\rm corr} \left( \tilde{\pmb{w}} \pmb{\mathcal{X}}, \ \ \tilde{\pmb{r}} \pmb{H}_k \right), \ \ \ \ \rho_{k,5} = {\rm corr} \left( \hat{\pmb{\psi}} \pmb{\mathcal{X}}, \ \ \hat{\pmb{\gamma}} \pmb{H}_k \right)\\
$$
