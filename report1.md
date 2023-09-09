
#### 主要功能

神经网络是一种非线性统计性数据建模工具, 是由具有适应性的人工神经元 (neuron) 组成的广泛并行互联的网络, 模拟生物神经系统对真实世界物体所作出的交互反应, 对函数进行估计或近似.

#### 基本原理

神经元接收来自 $n$ 个神经元的输入信号 $x_i,$ 这些输入信号通过带权重 $w_i$ 的连接进行传递; 神经元将接收到的总输入值 $\displaystyle\sum_{i=1}^nw_ix_i$ 与阈值 $\theta$ 进行比较, 通过激活函数 (activation function) $f$ 处理产生输出 $y,$ 即

$$
y=f(\sum_{i=1}^nw_ix_i-\theta).
$$

#### 应用情况

在难以被传统基于规则的编程所解决的问题, 例如机器视觉和语音识别等问题中, 神经网络更具有优势.

#### 优点

性能高.

与其它机器学习算法相比, 获得的数据量和计算能力增加时, 性能提升较明显.

#### 不足

对数据量和计算能力有很高的要求.

#### BP 公式推导

训练集 $D=\{(x^{(1)},y^{(1)}),...,(x^{(n)},y^{(n)})\},x\in\R^d,y\in\R^l.$

在单隐层网络中, 有 $q$ 个隐层神经元. 输出层, 隐层神经元阈值分别记 $\theta,\gamma,$ 连接权分别记 $v,w.$

记隐层第 $h$ 个神经元输入为

$$
\alpha_h=\sum_{i=1}^dv_{ih}x_i,
$$

输出层第 $j$ 个神经元输入为

$$
\beta_j=\sum_{h=1}^qw_{hj}b_h.
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
\hat{y}^{(k)}_j=f(\beta_j-\theta_j).
$$

损失函数

$$
E_k=\cfrac12\sum_{j=1}^l(\hat{y}^{(k)}_j-y^{(k)}_j)^2
$$

通过偏导获得递推式

$$
\begin{aligned}
    \cfrac{\partial\alpha_h}{\partial v_{ih}}
    &=x_i,\\
    \cfrac{\partial\alpha_h}{\partial x_i}
    &=v_{ih},\\
    \cfrac{\partial\beta_j}{\partial w_{hj}}
    &=b_h,\\
    \cfrac{\partial\beta_j}{\partial b_h}
    &=w_{hj},\\
    \\
    \cfrac{\partial b_h}{\partial\alpha_h}
    &=f'(\alpha_h-\gamma_h)\\
    &=b_h(1-b_h),\\
    \cfrac{\partial b_h}{\partial\gamma_h}
    &=-f'(\alpha_h-\gamma_h)\\
    &=b_h(b_h-1),\\
    \cfrac{\partial\hat{y}^{(k)}_j}{\partial\beta_j}
    &=f'(\beta_j-\theta_j)\\
    &=\hat{y}^{(k)}_j(1-\hat{y}^{(k)}_j),\\
    \cfrac{\partial\hat{y}^{(k)}_j}{\partial\theta_j}
    &=-f'(\beta_j-\theta_j)\\
    &=\hat{y}^{(k)}_j(\hat{y}^{(k)}_j-1),
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial E_k}{\partial\hat{y}^{(k)}_j}
    &=\cfrac12\cdot2(\hat{y}^{(k)}_j-y^{(k)}_j)\\
    &=\hat{y}^{(k)}_j-y^{(k)}_j,
    \\
    g_j
    &=-\cfrac{\partial E_k}{\partial\beta_j}\\
    &=-\cfrac{\partial E_k}{\partial\hat{y}^{(k)}_j}\cdot\cfrac{\partial\hat{y}^{(k)}_j}{\partial\beta_j}\\
    &=\hat{y}^{(k)}_j(1-\hat{y}^{(k)}_j)(y^{(k)}_j-\hat{y}^{(k)}_j),\\
    e_h
    &=-\cfrac{\partial E_k}{\partial\alpha_h}\\
    &=-\cfrac{\partial E_k}{\partial b_h}\cdot\cfrac{\partial b_h}{\partial\alpha_h}\\
    &=-b_h(1-b_h)\sum_{j=1}^l\cfrac{\partial E_k}{\partial\beta_j}\cdot\cfrac{\partial\beta_j}{\partial b_h}\\
    &=b_h(1-b_h)\sum_{j=1}^lw_{hj}g_j,\\
    \\
    \cfrac{\partial E_k}{\partial\theta_j}
    &=\cfrac{\partial E_k}{\partial\hat{y}^{(k)}_j}\cdot\cfrac{\partial\hat{y}^{(k)}_j}{\partial\theta_j}\\
    &=g_j,\\
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
