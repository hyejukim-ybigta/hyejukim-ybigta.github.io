---
layout: post
title: "ISL_Chapter2_Exercises"
author: "hyeju.kim"
categories: ML
tags: [ISL]
image: chapter2_exercise.jpg
---

# Chapter2 Exercises

## Conceptual

1. For each of parts (a) through (d), indicate whether we would generally expect the performance of a flexible statistical learning method to be better or worse than an inflexible method. Justify your answer.

참고 사항:
https://stats.stackexchange.com/questions/69237/flexible-and-inflexible-models-in-machine-learning

Of course it depends on the underlying data which you should always explore to find out some of its characteristics before trying to fit a model but what I've learnt as general rules of thumb are:

(1)A flexible model allows you to take full advantage of a large sample size (large n).
(2)A flexible model will be necessary to find the nonlinear effect.
(3)A flexible model will cause you to fit too much of the noise in the problem (when variance of the error terms is high).


   (a) The sample size n is extremely large, and the number of predictors p is small. --**(1)번에 따르면 flexible model이 n이 큰 경우에 유용하다고 한다.**


   (a) The sample size n is extremely large, and the number of predictors p is small. --** (1)번에 따르면 flexible method는 n이 큰 경우에는 잘 작동하기 때무네 more inflexible methods work better with less features
   (b) The number of predictors p is extremely large, and the number of observations n is small. -- **n이 작은 경우에 flexible model을 쓰면 overfitting 문제가 발생할 수 있으므로 inflexible method가 선호된다.**

   (c) The relationship between the predictors and response is highly non-linear. -- **flexible methods가 non-linear relationship을 더 잘 보여준다.**

   (d) The variance of the error terms, i.e. σ2 = Var(), is extremely high. -- **flexible methods의 경우 error에 너무 fitting을 해서 overfitting 문제가 발생할 수 있다.**



2. Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide n and p.

   (a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry and theCEO salary. We are interested in understanding which factors affect CEO salary.   
   --**regression//inference//target : CEO salary//**
   **n = 500 firms in the US//p = profit, number of employees, industry**

   (b) We are considering launching a new product and wish to know whether it will be a success or a failure. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables. 
   --**classification//prediction//target:a success or failure of a new product//**
   **n= 20 similar products//p = price charged for the product, marketing budget, competition price, and ten other variables.**

   (c) We are interesting in predicting the % change in the US dollar in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the dollar, the % change in the US market, the % change in the British market, and the % change in the German market. 
   --**regression//prediction//target:the % change in the US dollar in relation to the weekly changes in the world stock markets.//**
   **n= weekly data(52 weeks) for all of 2012.//p = the % change in the US market, the % change in the British market, and the % change in the German market.**



3. We now revisit the bias-variance decomposition.

   참고자료: https://elitedatascience.com/bias-variance-tradeoff

   ​

   (a) figure 2.12 + figure 2.17

   ![chapter2_exercise_3_a](https://user-images.githubusercontent.com/32008883/37323314-3d5cb034-26c6-11e8-8c27-32a0d31f3ab0.jpg)

   ​

   (b) 

   ​	1) bias : flexibility가 증가하면 bias가 감소한다.

   ​	2) variance : flexibility가 증가하면 variance는 증가한다. bias-variance trade off를 잘 보여주고 있다.

   ​	3) test error : bias와 variance, var($\epsilon$)를 합친 모양의 곡선이다.   

   ​	4) trarinig error : flexibility 가 증가할 수록 training error가 감소한다. irreducible error보다 작아질 때는 overfitting문제가 발생한다고 볼 수 있다.

   ​	5) bayes error:  var($\epsilon$) 와 같으므로 test error의 lower limit이다. 

   왜 bayes error = irreducible error? 

   *Bayes error is the lowest possible prediction error that can be achieved and is the same as irreducible error.* 

   https://stats.stackexchange.com/questions/302900/what-is-bayes-error-in-machine-learning

4. -생략-

5.  flexible method와 inflexible method의 장단점을 서술하고 어떠한 경우에 좋은지 말하여라

   1) bias & variance :  flexible method의 경우 non-linear model에 더 적합이 잘되며, bias가 적어지게 한다. 하지만 더 많은parameter를 추정해야하며, overfitting의 문제가 발생할 수 있다. 즉, variance는 커진다.

   2) purpose(prediciton/inference) :  flexible method의 경우  prediction에 유리하고,  inflexible method의 경우 inference나 interpretability에 유리하다.

   ​

6. 1) paramatic metohds 는 먼저, f의 함수식에 대한 가정을 만든다. 그리고 이 모델을 fitting시키거나 training시킨다.

   - 장점 : parameter $\beta_0, \beta_1, ... \beta_p$ 에 대한 추정을 하는 것이므로 f  자체를 추정하는 것보다 간단하다. non-parametic보다 많은 n을 필요로 하지 않는다. 
   - 단점 :  모델이 true f와 같지 않을 수 있다. 또한 너무 복잡하게 모델을 세우면 overfitting문제가 발생할 수 있다.

   2) non-parametic methods 는 f 의 함수 형태에 대한 명확한 가정을 하지 않는다. 너무 구불구불하지 않게 f를 추정하는 것이 목표이다. 

   - 장점 : f에 대한 특정한 함수 형태에 대한 가정이 없기 때문에 더 많은 모양의 f들을 포용할 수 있다. 
   - 단점 :  많은 n을 필요로 한다.

7. The table below provides a training data set containing six observations, three predictors, and one qualitative response variable.

   ​

         Obs.   X1   X2   X3  Distance(0, 0, 0)   Y
       ---------------------------------------------
          1      0    3    0   3                   Red 
          2      2    0    0   2                   Red
          3      0    1    3   sqrt(10) ~ 3.2      Red
          4      0    1    2   sqrt(5) ~ 2.2       Green
          5      -1   0    1   sqrt(2) ~ 1.4       Green
          6      1    1    1   sqrt(3) ~ 1.7       Red
   ​
   Suppose we wish to use this data set to make a prediction for Y when X1 = X2 = X3 = 0 using K-nearest neighbors.

   (a) Compute the Euclidean distance between each observation and the test point, X1 = X2 = X3 = 0.

   1) $\sqrt{9} = 3$

   2) $\sqrt{4} = 2$

   3) $\sqrt{1+9} = \sqrt{10}$

   4) $\sqrt{1+4} = \sqrt{5}$

   5) $\sqrt{1+1} = \sqrt{2}$

   6) $\sqrt{1+1+1} = \sqrt{3}$ 

   ​

   (b) What is our prediction with K = 1? Why?

   **belongs to Green class because it is close to 5th observation.**

   (c) What is our prediction with K = 3? Why?

   **top 3 closest observations are 2,5,6. then estimated probabilities 1/3 for green class, and 2/3 for red class. Hence, KNN classifier would predict test observation would belong to red class.**

   (d) If the Bayes decision boundary in this problem is highly nonlinear, then would we expect the best value for K to be large or small? Why?

   **According to the textbook, as K grows, the method becomes less flexible and produces a decision boundary that is close to linear. So, if the problem is higly nonlinear, K should be small.**

   ​


2주차 발제

q1) 2개 표본이면 x-bar(x)가 0이 됨. (x값이 동일할 때)

q2) 주어진 x값에서 error term들의 평균이 0이다. 라는 가정을 만족시키지 못함(평균이 0인 난수,,)