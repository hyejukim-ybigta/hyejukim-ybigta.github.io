---
layout: post
title: "ISL_Chapter3_Exercises"
author: "hyeju.kim"
categories: ML
tags: [ISL]
image: chapter3_exercise.jpg
---

# Chapter3 Exercises

## Conceptual

1. ​

```This difference stems from the fact that in the simple regression case, the
slope term represents the average effect of a $1,000 increase in newspaper advertising, ignoring other predictors such as TV and radio . In contrast, in average effect of increasing newspaper spending by $1,000 while holding TV and radio fixed.
```

 Table 3.4 에서, 'TV' 에 대한 null hypothesis는 'radio와 newspaper가 고정되어 있을 때 TV광고는 sales에 영향이 없다'이다. radio, newspaper에 대한 null hypothesis는 동일하다. TV와 radio 변수는 p-value가 0.05보다 작기 때문에 null hypothesis를 reject할 수 있다. 따라서 radio와 newspaper가 고정되어 있을 때 TV광고는 sale에 영향이 있다. 또한 TV와 newspaper가 고정되어 있을 때 radio광고는 sale에 영향이 있다. newpaper 변수에서는 p-value가 0.05보다 크므로 null hypothesis를 reject할 수 없다.



2. 방법은 같으나 target이 양적(quantitative)이냐 양적(qualitative)이냐에 따라 다르다.



3. Y = 50 + 20(gpa) + 0.07(iq) + 35(gender) + 0.01(gpa * iq) - 10 (gpa * gender)

(a) male: (gender=0)  Y = 50 + 20(gpa) + 0.07(iq) + 0.01(gpa * iq) 

female: (gender=1)  Y = 50 + 20(gpa) + 0.07(iq) + 35+ 0.01(gpa * iq) - 10 (gpa)

male과 female은  50 + 20(gpa) + 0.07(iq) + 0.01(gpa * iq)  항이 동일하고 + 35- 10 (gpa) 에서 차이를 보인다. 만약 gpa가 높다면 (3.5 < gpa) male일때 female보다 더 salary가 높을 것이다.

즉, iii correct

(b) IQ = 110, gpa = 4, gender = 1

Y = 50 + 20 * 4 + 0.07 * 110 + 35 * 1 + 0.01 * ( 4 * 110) - 10 ( 4 * 1) = 137.1

(c) coefficient가 작다고 해서 의미가 없는 것이 아니다. 단위에 따라 coefficient가 작을 수 있기 때문에 p-value를 살펴보아야 한다. 



4. ​

(a) cubic regression의 경우가 더 training RSS가 낮을 것이다. 왜냐하면 더 flexible하기 때문이다. 

(b) test set에서는 오히려 linear regression의 RSS가 더 낮을 것이다. 덜 flexible하여 overfitting문제가 덜하기 때문이다.

*(c) flexibility 와 상관없이 항상 cubic regression의 training RSS가 더 낮다.*

(d)  information이 부족하다. 왜냐하면 linear보다 cubic에 가까울수록 cubic regression의 test RSS가 linear regression보다 낮을 수 있기 때문이다. 하지만 linear에 가깝다면 linear regression의 test RSS가 더 낮다. 




5. $\hat{y_i} = \sum_{k=1}^{n} {a_k}{y_k}$ , what is $a_k$?

   ### $\hat{y_i}=x_i\hat{\beta} = x_i \frac{\sum_{k=1}^{n}x_ky_k}{\sum_{s=1}^{n}x_s^2}=  \frac{\sum_{k=1}^{n}x_ix_ky_k}{\sum_{s=1}^{n}x_s^2} = \sum_{k=1}^{n}{\frac{x_ix_k}{\sum_{s=1}^{n}x_s^2}y_k} $

   ### $\therefore a_k = \frac{x_ix_k}{\sum_{s=1}^{n}x_s^2}$

   ​

6. simple linear regression에서 $\hat{y} = \hat{\beta_0} + \hat{\beta_1}x$

   $\hat{\beta_0} = \overline{y} - \hat{\beta_1}\overline{x}$

   $\therefore \hat{\beta_0} + \hat{\beta_1}x=   \overline{y} - \hat{\beta_1}\overline{x} + \hat{\beta_1}x$

   $\hat{y} =  \overline{y} - \hat{\beta_1}\overline{x} + \hat{\beta_1}x$에 $(\overline{x},\overline{y})$ 를 대입하면

   $\overline{y} = \overline{y} - \hat{\beta_1}\overline{x} + \hat{\beta_1}\overline{x} = \overline{y}  $

   ​

7. ​

   suppose that we have $n$ observations $(x1,y1),…,(xn,yn)(x1,y1),…,(xn,yn)$ from a simple linear regression

   $Y_i=\beta_0+\beta_1x_i+ε_i,$ where $i=1,…,n$

   Let us denote $\hat{y}_i = \hat{\beta_0} + \hat{\beta_1}x_i$ for $i=1,…,n$, where  $\hat{\beta_0}$  and $\hat{\beta_1}$ are the ordinary least squares estimators of the parameters $ \beta_0$ and $\beta_1$.

   The coefficient of the determination $r^2$ is defined by

   ### $r^2=\frac{\sum_{i=1}^{n}{(\hat{y}_i−\overline{y})}^2}{\sum_{i=1}^{n} {(y_i-\overline{y})}^2}$

   Using the facts that

   ### $\hat{\beta_1}=\frac{∑^n_{i=1}(x_i−\overline{x})(y_i−\overline{y})}{∑^n_{i=1}(x_i−\overline{x})^2}$

   and $\hat{\beta_0} = \overline{y} - \hat{\beta_1}\overline{x}$, we obtain

   $$\sum_{i=1}^{n}{(\hat{y}_i−\overline{y})}^2 = \sum_{i=1}^{n}{(\hat{\beta_0} + \hat{\beta_1}x_i−\overline{y})^2}$$

   $= \sum_{i=1}^{n}{(\overline{y} - \hat{\beta_1}\overline{x}+ \hat{\beta_1}x_i−\overline{y})^2}$

   $= \hat{\beta_1}^2\sum_{i=1}^{n}{(x_i- \overline{x})^2}$

   ### $=\frac{(∑^n_{i=1}(x_i−\overline{x})(y_i−\overline{y}))^2}{(∑^n_{i=1}(x_i−\overline{x})^2)^2}\sum_{i=1}^{n}{(x_i- \overline{x})^2}$

   ### $=\frac{(∑^n_{i=1}(x_i−\overline{x})(y_i−\overline{y}))^2}{∑^n_{i=1}(x_i−\overline{x})^2}$

   ### $\therefore$(3.17) R^2 statistic = $ r^2=\frac{\sum_{i=1}^{n}{(\hat{y}_i−\overline{y})}^2}{\sum_{i=1}^{n} {(y_i-\overline{y})}^2}$

   ###$=\frac{(∑^n_{i=1}(x_i−\overline{x})(y_i−\overline{y}))^2}{∑^n_{i=1}(x_i−\overline{x})^2\sum_{i=1}^{n} {(y_i-\overline{y})}^2} $= square of the correlation between X and Y(3.17)

   [증명출처](https://math.stackexchange.com/questions/129909/correlation-coefficient-and-determination-coefficient)

   ​

   ​

   ​

   ​