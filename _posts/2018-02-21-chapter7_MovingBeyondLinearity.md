---
layout: post
title: "ISL_Chapter7_Moving Beyond Linearity"
author: "hyeju.kim"
categories: facts
tags: [ISL]
image: NaturalCubicSpline.png
---

# Chapter 7. Moving Beyond Linearity 

## 7.1 Polynomial Regression

![image](https://user-images.githubusercontent.com/32008883/36198832-12110b16-11bb-11e8-8b29-6543b9062dfe.png)

![image](https://user-images.githubusercontent.com/32008883/36199018-a3d2d7e6-11bb-11e8-99d8-222919a060b2.png)



## 7.2 Step Functions

- break the range of X into bins, and fit a different constant in each bin. 
- converting a continous variable into an ordered categorical variable

![image](https://user-images.githubusercontent.com/32008883/36199055-c365c37a-11bb-11e8-9ae1-68b3149d21d4.png)

![image](https://user-images.githubusercontent.com/32008883/36199200-23165abe-11bc-11e8-99e0-8d3b9b3dc669.png)



![image](https://user-images.githubusercontent.com/32008883/36199106-e8dcbeba-11bb-11e8-9196-2a02263e2925.png)



## 7.3 Basis Functions

![image](https://user-images.githubusercontent.com/32008883/36199271-6a0c5b9e-11bc-11e8-8e40-f5443c708049.png)

- the basis functions are $b_j(x_i)=x_j$ , and for piecewise constant functions they are $b_j(x_i)=I(c_j ≤ x_i < c_{j+1})$  



## 7.4 Regression Splines

- extension of polynomial regression and piecewise constant regression

### 7.4.1 Piecewise Polynomials

- fitting separate low-degree polynomials over differernt regions of X

![image](https://user-images.githubusercontent.com/32008883/36199833-16e76b14-11be-11e8-9033-04ef678a24b3.png)

### 7.4.2 Constraints and Splines



top-right : *constraint* that the fitted curve must be continuous

lower-left :  *constraint* that continous + have continuous first and second deriatives (called as *cubic spline*)

lower-right : *constraint* of continuity at each knot

![image](https://user-images.githubusercontent.com/32008883/36200143-10e1bd0e-11bf-11e8-82f0-1eec82ee5efb.png)

### 7.4.3 The Spline Basis Representation

![image](https://user-images.githubusercontent.com/32008883/36200488-36857a54-11c0-11e8-98c1-943716ae612e.png)

- start off with a basis for a cubic polynomial, and then add one *truncated power basis* function per knot

- a truncate power basis function is defined as

  ![image](https://user-images.githubusercontent.com/32008883/36200573-800380cc-11c0-11e8-877c-b972da6c0d43.png)

  ![image](https://user-images.githubusercontent.com/32008883/36200631-aaaf17d2-11c0-11e8-99dc-36d457a8f847.png)

- a *natural cubic spline* is a regression apline with additional *boundary constraints:*the function is requred to be linear at the boundary

### 7.4.4 Choosing the Number and Locations of the Knots

- cross- validation



### 7.4.5 Comparison to Polynomial Regression

- why **Regression spline >> Polynomial regression**?
  - splines introduce flexibilityh by increasing the number of knots but keeping the degree fixed
  - polynomials must use a high degree to produce flexible fits



## 7.5 Smoothing Splines

### 7.5.1 An Overview of Smoothing Splines

![image](https://user-images.githubusercontent.com/32008883/36202931-0acfaef8-11c9-11e8-86eb-02b63bef505b.png)

### 						Loss + Penalty 

$\lambda$ : nonnegative tuning parameter

- if $\lambda \to \infty$ , sensitive to changing

g : function that minimizes(7.11) known as a *smoothing spline*

- why penalty? 
  - $g\prime\prime(t)$ = amount by which the slope is changing
- the function g(x) that minimizes (7.11)  : (shrunken ver.) a  natural cubic spline with knots at $x_1, ... , x_n$.



### 7.5.2 Choosing the Smoothing Parameter $\lambda$

choosing $\lambda \to$ effective degrees of freedom ($df_\lambda$)

- cross-validation

![image](https://user-images.githubusercontent.com/32008883/36203299-5e3760c6-11ca-11e8-8b72-55bd67a4f527.png)



## 7.6 Local Regression

- choosing $s$ : cross-validation

smaller -> flexibillity up

- effective in varing coefficient model(a multiple linear regression  model that is global in some variable , but local in another, such as time)
- perform poorly if p is much larger than about 3  or 4

![image](https://user-images.githubusercontent.com/32008883/36205024-a3c22f58-11d0-11e8-9d28-8e896b3aaf88.png)



## 7.7 Generalized Additive Models

- a general framework for extending a standard linear model by allowing non-linear functions of each of the variable, while maintaining *addivitiy*
- *quantitative and qualitative*



### 7.7.1 GAM for Regression Problems

![image](https://user-images.githubusercontent.com/32008883/36205081-d003932c-11d0-11e8-8910-6422c27726c1.png)

->

![image](https://user-images.githubusercontent.com/32008883/36205101-e167a414-11d0-11e8-93e0-3a1a1d6126a9.png)

- bulding methods for fitting an addtitive model
- **Pros and Cons of GAMs**
  - fit a non-linear $f_j$ to *each* $X_j \to$ do not need dto manually try out many different transformations on each variable
  - potentially make more accurate predictions 
  - examine the effect of each $X_j $ on $Y$ individually while holding all of the other variables fixed -> useful for inference
  - smoothness of the function $f_j$  can ve summarized via degrees of freedom
  - do not include interaction terms
  - compromise between linear and fully nonparametric models



### 7.7.2 GAMs for Classification Problems

![image](https://user-images.githubusercontent.com/32008883/36205606-a57349e8-11d2-11e8-97f7-c562be59ae4d.png)





