---
layout: post
title: "ISL_Chapter4_Exercises"
author: "hyeju.kim"
categories: ML
tags: [ISL]
image: chapter4_exercise.png
---

# Chapter4 Exercises

## Conceptual

1. $p(X) = \frac{e^{\beta_0+\beta_1X}}{1+e^{\beta_0+\beta_1X}}$  (4.2)

   $1-p(X) = \frac{1}{1+e^{\beta_0+\beta_1X}}$

   $\therefore \frac{p(X)}{1-p(X)} = e^{\beta_0+\beta_1X}$ (4.3)

 In other words, the logistic function representation and logit representation for the logistic regression model are equivalent.



2. Claim: Maximizing $p_k(x)$ is equivalent to maximizing $δ_k(x)$

   *Proof.* Let $x​$ remain fixed and observe that we are maximizing over the parameter $k​$. Suppose that $δ_k(x)≥δ_i(x)​$. We will show that $f_k(x)≥f_i(x)​$, From our assumption we have

   $x\frac{μ_k}{σ^2}−\frac{μ_k^2}{2σ^2}+log(π_k)≥x\frac{μ_i}{σ^2}−\frac{μ_i^2}{2σ^2}+log(π_i)$

   exponential function is monotonically increasing function,so the following inequality holds

   $π_k exp(x\frac{μ_k}{σ^2}−\frac{μ_k^2}{2σ^2}+log(π_k)) ≥ π_i exp(x\frac{μ_i}{σ^2}−\frac{μ_i^2}{2σ^2}+log(π_i))$

   Multiply this inequality by the positive constant

   $c=\frac{\frac{1}{\sqrt{2πσ}}exp(−\frac{1}{2σ^2}x^2)}{∑π_l\frac{1}{\sqrt{2πσ}}exp(−\frac{1}{2σ^2}(x-μ_l)^2)}$

   and we have that

   $\frac{π_k\frac{1}{\sqrt{2πσ}}exp(−\frac{1}{2σ^2}{(x-μ_k)^2})}{∑π_l\frac{1}{\sqrt{2πσ}}exp(−\frac{1}{2σ^2}(x-μ_l)^2)}  ≥ \frac{π_i\frac{1}{\sqrt{2πσ}}exp(−\frac{1}{2σ^2}{(x-μ_i)^2})}{∑π_l\frac{1}{\sqrt{2πσ}}exp(−\frac{1}{2σ^2}(x-μ_l)^2)}$

   or equivalently,$ f_k(x)≥f_i(x)$. Reversing these steps also holds.



[#proof link](http://blog.princehonest.com/stat-learning/ch4/2.html)



3. $p_k(x)=\frac{π_k\frac{1}{\sqrt{2πσ_k}}exp(−\frac{1}{2σ_k^2}{(x-μ_k)^2})}{∑π_l\frac{1}{\sqrt{2πσ}}exp(−\frac{1}{2σ^2}(x-μ_l)^2)}$

   $log(p_k(x))=\frac{logπ_k+log(\frac{1}{\sqrt{2πσ_k}})−\frac{1}{2σ_k^2}{(x-μ_k)^2}}{log(∑π_l\frac{1}{\sqrt{2πσ}}exp(−\frac{1}{2σ^2}(x-μ_l)^2))}$

   $log(p_k(x)) log(∑π_l\frac{1}{\sqrt{2πσ}}exp(−\frac{1}{2σ^2}(x-μ_l)^2)= logπ_k+log(\frac{1}{\sqrt{2πσ_k}})−\frac{1}{2σ_k^2}{(x-μ_k)^2}$

   $δ(x)= logπ_k+log(\frac{1}{\sqrt{2πσ_k}})−\frac{1}{2σ_k^2}{(x-μ_k)^2}$

   quadratic

[#proof link](http://blog.princehonest.com/stat-learning/ch4/3.html)

4.  **non-parametic approaches often perform poorly when p is large :  curse of dimensionality**

   (a)  0.1

   (b)  0.1 * 0.1 = 0.01

   (c)  0.1^100 

   (d) if p is very large, then (0.1)^p become extremely small. In this case, if there are very few training observations “near” any given test observation, the fraction of the available observations would be smaller. It is hard to predict with few observations

   (e) 0.1^(1/N)

5. **LDA vs. QDA**

   (a) QDA performs better on the training set , BUT LDA  performs better on the training set because overfitting problem could occur with linear boundaries

   (b) QDA performs better on the training set and test set with non-linear boundaries because QDA supposes that each class has its own covariance matrix.

   (c) **In general, with large sample, more flexible method QDA performs better because variance can be offset by large samples.**

   (d) False. **With fewer samples** the variance increase with flexibitliy of QDA leads to overfitting problem, lower test rate

   ​

   9. odds

   ​

   ​



