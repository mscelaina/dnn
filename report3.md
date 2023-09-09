训练集 $D=\{(x^{(1)},y^{(1)}),...,(x^{(n)},y^{(n)})\},x\in\R^d,y\in\R^l.$

在双隐层网络中, 分别有 $p,q$ 个隐层神经元. 输入分别记 $\alpha,\beta,\gamma,$ 阈值分别记 $\zeta,\eta,\theta,$ 连接权分别记 $u,v,w.$

$$
\begin{aligned}
    \alpha_c
    &=\sum_{i=1}^du_{ic}x_i,\\
    \beta_h
    &=\sum_{c=1}^pv_{ch}a_c,\\
    \gamma_j
    &=\sum_{h=1}^qw_{hj}b_h.\\
    \\
\end{aligned}
$$

激活函数使用 sigmoid 函数, 即

$$
f(x)=\cfrac{1}{1+e^{-x}},
$$

其具有性质

$$
f'(x)=f(x)(1-f(x)).
$$

模型

$$
\hat{y}^{(k)}_j=f(\gamma_j-\theta_j).
$$

损失函数

$$
E_k=\cfrac12\sum_{j=1}^l(\hat{y}^{(k)}_j-y^{(k)}_j)^2
$$

$$
\begin{aligned}
    \alpha_c
    &=\sum_{i=1}^du_{ic}x_i,\\
    \beta_h
    &=\sum_{c=1}^pv_{ch}a_c,\\
    \gamma_j
    &=\sum_{h=1}^qw_{hj}b_h.\\
    \\
\end{aligned}
$$

通过偏导获得递推式

$$
\begin{aligned}
    \cfrac{\partial\alpha_c}{\partial u_{ic}}
    &=x_i,
    &\cfrac{\partial\alpha_c}{\partial x_i}
    &=u_{ic},\\
    \cfrac{\partial\beta_h}{\partial v_{ch}}
    &=a_c,
    &\cfrac{\partial\beta_h}{\partial a_c}
    &=v_{ch},\\
    \cfrac{\partial\gamma_j}{\partial w_{hj}}
    &=b_h,
    &\cfrac{\partial\gamma_j}{\partial b_h}
    &=w_{hj},\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial\hat{y}^{(k)}_j}{\partial\gamma_j}
    &=f'(\gamma_j-\theta_j)\\
    &=\hat{y}^{(k)}_j(1-\hat{y}^{(k)}_j),\\
    \cfrac{\partial\hat{y}^{(k)}_j}{\partial\theta_j}
    &=-f'(\gamma_j-\theta_j)\\
    &=-\cfrac{\partial\hat{y}^{(k)}_j}{\partial\gamma_j}\\
    &=\hat{y}^{(k)}_j(\hat{y}^{(k)}_j-1),\\
    \\
    \cfrac{\partial b_h}{\partial\beta_h}
    &=f'(\beta_h-\eta_h)\\
    &=b_h(1-b_h),\\
    \cfrac{\partial b_h}{\partial\eta_h}
    &=-f'(\beta_h-\eta_h)\\
    &=-\cfrac{\partial b_h}{\partial\beta_h}\\
    &=b_h(b_h-1),\\
    \\
    \cfrac{\partial a_c}{\partial\alpha_c}
    &=f'(\alpha_c-\zeta_h)\\
    &=a_c(1-a_c),\\
    \cfrac{\partial a_c}{\partial\zeta_h}
    &=-f'(\alpha_c-\zeta_h)\\
    &=-\cfrac{\partial a_c}{\partial\alpha_c}\\
    &=a_c(a_c-1),\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial E_k}{\partial\hat{y}^{(k)}_j}
    &=\cfrac12\cdot2(\hat{y}^{(k)}_j-y^{(k)}_j)\\
    &=\hat{y}^{(k)}_j-y^{(k)}_j,\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial E_k}{\partial\theta_j}
    &=\cfrac{\partial E_k}{\partial\hat{y}^{(k)}_j}\cdot\cfrac{\partial\hat{y}^{(k)}_j}{\partial\theta_j}\\
    &=\hat{y}^{(k)}_j(1-\hat{y}^{(k)}_j)(y^{(k)}_j-\hat{y}^{(k)}_j)\\
    &=g_j,\\
    \cfrac{\partial E_k}{\partial\gamma_j}
    &=\cfrac{\partial E_k}{\partial\hat{y}^{(k)}_j}\cdot\cfrac{\partial\hat{y}^{(k)}_j}{\partial\gamma_j}\\
    &=-g_j,\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial E_k}{\partial b_h}
    &=\sum_{j=1}^l\cfrac{\partial E_k}{\partial\gamma_j}\cdot\cfrac{\partial\gamma_j}{\partial b_h}\\
    &=-\sum_{j=1}^lg_jw_{hj},\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial E_k}{\partial\eta_h}
    &=\cfrac{\partial E_k}{\partial b_h}\cdot\cfrac{\partial b_h}{\partial\eta_h}\\
    &=-\sum_{j=1}^lg_jw_{hj}b_h(b_h-1)\\
    &=b_h(1-b_h)\sum_{j=1}^lg_jw_{hj}\\
    &=e_h,\\
    \cfrac{\partial E_k}{\partial\beta_h}
    &=\cfrac{\partial E_k}{\partial b_h}\cdot\cfrac{\partial b_h}{\partial\beta_h}\\
    &=-e_h,\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial E_k}{\partial a_c}
    &=\sum_{h=1}^q\cfrac{\partial E_k}{\partial\beta_h}\cdot\cfrac{\partial\beta_h}{\partial a_c}\\
    &=-\sum_{h=1}^qe_hv_{ch},\\
    \\
\end{aligned}
$$


$$
\begin{aligned}
    \\
    \Delta\theta_j
    &=-\eta\cfrac{\partial E_k}{\partial\theta_j}\\
    &=-\eta g_j,\\
    \\
    \cfrac{\partial E_k}{\partial\gamma_h}
    &=\cfrac{\partial E_k}{\partial\beta_j}\cdot\cfrac{\partial\beta_j}{\partial b_h}\cdot\cfrac{\partial b_h}{\partial\gamma_h}\\
    &=e_h,\\
    \Delta\gamma_h
    &=-\eta\cfrac{\partial E_k}{\partial\gamma_h}\\
    &=-\eta e_h,
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial E_k}{\partial w_{hj}}
    &=\cfrac{\partial E_k}{\partial\beta_j}\cdot\cfrac{\partial\beta_j}{\partial w_{hj}}\\
    &=-g_jb_h,\\
    \Delta w_{hj}
    &=-\eta\cfrac{\partial E_k}{\partial w_{hj}}\\
    &=\eta g_jb_h,\\
    \\
    \cfrac{\partial E_k}{\partial v_{ih}}
    &=\cfrac{\partial E_k}{\partial\alpha_j}\cdot\cfrac{\partial\alpha_j}{\partial v_{ih}}\\
    &=-e_hx_i,\\
    \Delta v_{ih}
    &=-\eta\cfrac{\partial E_k}{\partial v_{ih}}\\
    &=\eta e_hx_i.\\
    \\
\end{aligned}
$$

#### 参考资料

Neural Network, 反向传播算法 的 Wikipedia 页面;

周志华 机器学习.
