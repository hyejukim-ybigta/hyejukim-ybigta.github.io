---
layout: post
<<<<<<< HEAD
title: "ISL_Chapter3_Linear Regression"
=======
title: "ISL_Chapter_modified"
>>>>>>> 229cba7b3aec84f9dbf6c7a333c626be54fea306
author: "hyeju.kim"
categories: facts
tags: [ISL]
image: 
---
<<<<<<< HEAD
=======
3.3.3 Potential Problems

1. Non-linearity of the Data

Figure 3.9

- Residual plots are a useful graphical tool for identifying non-linearity
- If the residual plot indicates that there are non-linear associations in the
  data, then a simple approach is to use non-linear transformations of the
  predictors, such as logX,
  √
  X, and X2,



2. Correlation of Error Terms 

- Why might correlations among the error terms occur? Such correlations
  frequently occur in the context of time series data, which consists of obtime
  series
  servations for which measurements are obtained at discrete points in time.
- Figure 3.10
- Now there is a clear pattern in the
  residuals—adjacent residuals tend to take on similar values.

3. Non-constant Variance of Error Terms 

- heteroscedasticity
- When faced with this problem, one possible solution is to transform
  the response Y using a concave function such as log Y or
  √
  Y .

4. Outliers

- Residual plots
- observations for which the response yi is
  unusual given the predictor xi.

5. High Leverage Points 

- Figure 3.13
- high leverage observations tend to have
  a sizable impact on the estimated regression line.
- But in a multiple
  linear regression with many predictors, it is possible to have an observation
  that is well within the range of each individual predictor’s values, but that
  is unusual in terms of the full set of predictors.
- quantify an observation’s leverage:
  For a simple linear regression, statistic
  hi =
  1
  n
  +
  (xi − ¯x)2

  n

  i=1(xi − ¯x)2 .

(3.37)

	So if a given observation has a leverage statistic that greatly 		exceeds (p+1)/n, then we may suspect that the corresponding

point has high leverage.

6. Collinearity 

Collinearity refers to the situation in which two or more predictor variables are closely related to one another.
>>>>>>> 229cba7b3aec84f9dbf6c7a333c626be54fea306





## 3.5 Comparison of Linear Regression with K-Nearest Neighbors

- linear regression - parametric
- K-nearest neighbors regression (KNN regression). - non-parametric



- 두 개의 성능비교 : the parametric approach will outperform the nonparametric approach if the parametric form that has been selected is close to the true form of f.
  - true form of f: linear -> linear  reg  (MSE LOWER)>> KNN
  - true form of f: NONlinear -> linear  reg >> KNN (MSE LOWER)
    - IN REALITY, however, curse of dimensionality problem occurs. p(predictor 개수)의 개수가 커지면 결국 linear reg >> KNN reg












