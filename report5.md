训练集 $D=\{(x^{(1)},y^{(1)}),...,(x^{(n)},y^{(n)})\},x\in\R^d,y\in\R^l.$

在多隐层网络中, 有 $m$ 个隐层, 分别有 $q^{(c)},c=1,...,m$ 个隐层神经元 $\{b^{(c)}_1,...,b^{(c)}_{q^{(c)}}\}.$

输入分别记 $\alpha^{(c)},$ 阈值分别记 $\theta^{(c)},$ 连接权分别记 $w^{(c)}.$ 激活函数使用 softmax 函数, 即

$$
\begin{aligned}
    \alpha^{(c)}_j
    &=\sum_{i=1}^{q^{(c-1)}}w^{(c)}_{ij}b^{(c-1)}_i,\\
    b^{(c)}_j
    &=\cfrac{\exp(\alpha^{(c)}_j)}{\sum\exp(\alpha^{(c)})},\\
    j
    &=1,...,q^{(c)},\\
    c
    &=1,...,m+1.
    \\
\end{aligned}
$$

令

$$
\begin{aligned}
    q^{(m+1)}
    &=l,\\
    q^{(0)}
    &=d,\\
    b^{(0)}_i
    &=\cfrac{x_i}{\sum x},\\
    i
    &=1,...,d,\\
    \\
\end{aligned}
$$

可见有

$$
\begin{aligned}
    \sum b^{(c)}
    &=1,\\
    c
    &=0,...,m+1.
\end{aligned}
$$

模型

$$
\begin{aligned}
    \hat{y}_j
    &=b^{(m+1)}_j\\
    &=\cfrac{\exp(\alpha^{(m+1)}_j)}{\sum\exp(\alpha^{(m+1)})},\\
    j
    &=1,...,l.\\
\end{aligned}
$$

损失函数选择交叉熵

$$
E=-\sum_{j=1}^ly_j\log_2\hat{y}_j.
$$

通过偏导获得递推式

$$
\begin{aligned}
    z^{(m+1)}_j
    &=\cfrac{\partial E}{\partial b^{(m+1)}_j}\\
    &=\cfrac{\partial E}{\partial\hat{y}_j}\\
    &=\cfrac12\cdot2(\hat{y}_j-y_j)\\
    &=\hat{y}_j-y_j,\\
    j
    &=1,...,l.\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial\alpha^{(c)}_j}{\partial w^{(c)}_{ij}}
    &=b^{(c-1)}_i,\\
    \cfrac{\partial\alpha^{(c)}_j}{\partial b^{(c-1)}_i}
    &=w^{(c)}_{ij},\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    \cfrac{\partial b^{(c)}_j}{\alpha^{(c)}_j}
    &=f'(\alpha^{(c)}_j-\theta^{(c)}_j)\\
    &=f(\alpha^{(c)}_j-\theta^{(c)}_j)(1-f(\alpha^{(c)}_j-\theta^{(c)}_j))\\
    &=b^{(c)}_j(1-b^{(c)}_j),\\
    \cfrac{\partial b^{(c)}_j}{\theta^{(c)}_j}
    &=-f'(\alpha^{(c)}_j-\theta^{(c)}_j)\\
    &=-\cfrac{\partial b^{(c)}_j}{\alpha^{(c)}_j}\\
    &=b^{(c)}_j(b^{(c)}_j-1),\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    g^{(c)}_j
    &=\cfrac{\partial E}{\partial\theta^{(c)}_j}\\
    &=\cfrac{\partial E}{\partial b^{(c)}_j}\cdot\cfrac{\partial b^{(c)}_j}{\partial\theta^{(c)}_j}\\
    &=z^{(c)}_jb^{(c)}_j(b^{(c)}_j-1),\\
    \cfrac{\partial E}{\partial\alpha^{(c)}_j}
    &=\cfrac{\partial E}{\partial b^{(c)}_j}\cdot\cfrac{\partial b^{(c)}_j}{\partial\alpha^{(c)}_j}\\
    &=-g^{(c)}_j,\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    u^{(c)}_{ij}
    &=\cfrac{\partial E}{\partial w^{(c)}_{ij}}\\
    &=\cfrac{\partial E}{\partial\alpha^{(c)}_j}\cdot\cfrac{\partial\alpha^{(c)}_j}{\partial w^{(c)}_{ij}}\\
    &=-g^{(c)}_jb^{(c-1)}_i,\\
    z^{(c-1)}_i
    &=\cfrac{\partial E}{\partial b^{(c-1)}_i}\\
    &=\sum_{j=1}^{q^{(c)}}\cfrac{\partial E}{\partial\alpha^{(c)}_j}\cdot\cfrac{\partial\alpha^{(c)}_j}{\partial b^{(c-1)}_i}\\
    &=-\sum_{j=1}^{q^{(c)}}g^{(c)}_jw^{(c)}_{ij},\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    \\
    \Delta\theta^{(c)}_j
    &=-\eta^{(c)}\cfrac{\partial E}{\partial\theta^{(c)}_j}\\
    &=-\eta^{(c)}g^{(c)}_j,\\
    \\
    \Delta w^{(c)}_{ij}
    &=-\eta^{(c)}\cfrac{\partial E}{\partial w^{(c)}_{ij}}\\
    &=-\eta^{(c)}u^{(c)}_i\\
    &=\eta^{(c)}g^{(c)}_jb^{(c-1)}_i,\\
    \\
\end{aligned}
$$

$$
\begin{aligned}
    i
    &=1,...,q^{(c)},\\
    j
    &=1,...,q^{(c+1)},\\
    c
    &=1,...,m+1.\\
\end{aligned}
$$


#### 参考资料

Neural Network, 反向传播算法 的 Wikipedia 页面;

周志华 机器学习.
