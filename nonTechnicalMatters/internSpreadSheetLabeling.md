## 平均分散法

V:標準偏差対角行列  
P:相関行列  
$\Sigma = VPV$:共分散行列  
$w_m$:時価比率  
a:リスク回避度  
$\mu = a \Sigma w_m$:期待リターン

$\Sigma^{-1}$  
$\bf{1}$ベクトル  
$\Sigma^{-1}\mu$  
$\Sigma^{-1}\bf{1}$  
$A = \bf{1}'\Sigma^{-1}\bf{1}$  
$B = \mu'\Sigma^{-1}\bf{1}$  
$C = \mu' \Sigma^{-1}\mu$  

$\mu_p$  
$\lambda_1 = \frac{A\times \mu_p - B}{AC-B^2}$  
$\lambda_2 = \frac{C-B\times \mu_p}{AC-B^2}$  
$w'=\left(\lambda_1 \Sigma^{-1}\mu + \lambda_2 \Sigma^{-1}\bf{1}\right)'$  
$\sigma_p=\sqrt{w'\Sigma w}$

$\mu - r_f \bf{1}$  
$\Sigma^{-1}\left(\mu - r_f \bf{1} \right)$  
$\left(\mu - r_f \bf{1}\right)' \Sigma^{-1} \left( \mu - r_f \bf{1}\right)$  

$\lambda = \frac{\mu_p - r_f}{\left(\mu - r_f \bf{1}\right)' \Sigma^{-1} \left( \mu - r_f \bf{1}\right)}$  
$w' = \left(\lambda \Sigma^{-1}\left(\mu - r_f \bf{1} \right) \right)'$  
$\sigma_p = \sqrt{w'\Sigma w}$

## Black-Littermanモデル

$\tau$  
$F$  
$r_V$  
$\omega : 不確実性$  
$G = \left(F'\Omega^{-1}F + \left(\tau \Sigma \right)^{-1} \right)^{-1}$  
$GH = G \left(F'\Omega^{-1}r_V + \left(\tau \Sigma \right)^{-1}\mu_m \right)$  

$\left(\tau \Sigma \right)^{-1}$  
$\left(\tau \Sigma \right)^{-1}\mu_m$  
$F'\Omega^{-1} F$  
$F'\Omega^{-1}r_V$  
$\Omega^{-1}$  
$\Omega = DPD$: 不確実性共分散  
$D$:不確実性 対角行列  
$P$:相関行列

## CAPM
CAPMに基づく期待リターン

$\mathrm{資産 i}の期待リターン = \mathrm{無リスク金利} + \beta \times \left(\mathrm{Topixの期待リターン} - \mathrm{無リスク金利} \right)$


$ \mathbb{E}[r^i] = r_f + \beta \times \left(\mathbb{E}[r^M] - r_f\right)$