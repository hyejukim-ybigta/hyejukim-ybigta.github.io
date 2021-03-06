---
layout: post

title: "ISL_Chapter9_Support Vector Machines_korean"

author: "hyeju.kim"

categories: ML

tags: [ISL]

image: svm.jpg
---

# Chapter 9. Support Vector Machines

## 9.1 최대 마진 분류기(Maximal Margin Classifier)

### 9.1.1 초평면(Hyperplane)이란?

- p차원의 공간에서 초평면은 p-1차원의 평평한 아핀(affine) 공간이다. (*affine : 원점을 지나지 않는 부분공간)

  ex) 예를 들어 2차원에서, 초평면은 직선이다.


- 초평면은 p차원의 공간을 두 개로 나눈다. (초평면>0과 초평면<0에따라 )

### 9.1.2 초평면을 사용한 분류

![img](https://user-images.githubusercontent.com/32008883/36377130-3f1e4860-15b9-11e8-8e70-982891d4cd4d.png)

![img](https://user-images.githubusercontent.com/32008883/36377156-5a4b362a-15b9-11e8-84f2-6e30c7fff329.png)

![image](https://user-images.githubusercontent.com/32008883/36726405-ff89cbc6-1bfc-11e8-8a90-7e65ec21d268.png)

- 만약 blue class에 속할 때 1로 라벨링하고, purple class에 속할 때 -1로 라벨링한다고 하자. y값이 1인 class, 즉 blue class는 초평면이 0보다 큰 영역이고, y값이 -1인 class, 즉 purple class면 기준이 되는 초평면이 0보다 작은 영역이다. 즉, (9.8)과 같은 식을 얻을 수 있다. 

- 기준이 되는 초평면이 있다면, 이를 활용한 분류기를 얻을 수 있다. test 관측치 $$x^*$$ 가 초평면의 어떤 쪽에 속하는지, 즉 0보다 큰지 작은지에 따라 class를 분류할 수 있다. $$f(x^*) = \beta_0 + \beta_1x_1^* + \beta_2x_2^* + ... +\beta_px_p^*$$ 가 0보다 크면 class1에 배정, 0보다 작으면 class -1에 배정한다. $$f(x^*) $$의 크기를 사용하면, 할당에 대한 확신 정도를 알 수 있다. $$f(x^*)$$ 가 0에서 멀 수록 초평면에서 멀리 떨어져있다는 것인데, 그만큼 class 할당에 자신있다는 것이다. 

  ​

  ​




### 9.1.3 최대 마진 분류기(The Maximal Margin Classifier)

- 최대 마진 분류기는 가장 큰 마진을 가지는 초평면을 기준으로 분류한다. 

- 마진(margin) : 각각의 training 관측치들로터 초평면까지의 수직 거리 중에 최소 거리이다.

- 차원이 커질 때 overfitting문제가 발생할 수 있다.

- $$f(x^*)$$ 의 부호에 따라 test 관측치 $$x^*$$ 의 분류가 달라진다.

- Support Vector : 쉽게 생각하면 초평면 부근의 관측치들이다. 최대 마진 분류기에서는 margin 안 쪽, 즉 점선안 쪽에 놓인 점들을 support vector라고 볼 수 있다. 이 점들이 조금이라도 움직이면 최대 마진 초평면이 변경될 수 있기 때문에 이 점들(vector)가 초평면을 지지(support)한다고 볼 수 있다. 아래 그림에서 점선에 놓인 두개의 파란 점과 한 개의 보라 점이 Support Vector이다.

  ​

![img](https://user-images.githubusercontent.com/32008883/36379539-5a8b1ab6-15c2-11e8-9902-d52b51d4f9be.png)

### 9.1.4 최대 마진 분류기의 원리

- 이 분류기를 최적화하려면 어떻게 해야 할까?

![img](https://user-images.githubusercontent.com/32008883/36379334-79ca3b6a-15c1-11e8-8fb7-3e317ce91b07.png)

- - (9.11) : 각각의 관측치들이 초평면을 기준으로 margin 이상의 위치에 있어야 올바른 위치에 있다는 뜻이다. M은 양수인 margin이다. 

  - (9.10) : (9.10)의 조건이 있어야 $y_i(\beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + ... +\beta_px_{ip})$ 값이 초평면과 i번째 관측치 사이의 수직 거리가 된다.

    - why? : 점($$x_0,y_0,z_0$$)과 평면($$ax+by+cz+d=0$$) 사이의 거리 $$d=\frac{\mid ax_0 + by_0 + cz_0 + d  \mid}{\sqrt{a^2+b^2+c^2}} $$ 

      즉 (9.10)의 조건으로 분모 부분이 1 이되어서 [(9.6)](#9.1.2 초평면을 사용한 분류) 과 (9.7)에 의해 d 는 결국 $y_i(\beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + ... +\beta_px_{ip})$ 과 같아진다. (절대값 기호)

  - (9.10)&(9.11): 각각의 관측치가 초평면에서 올바른 위치에 있고, 초평면에서 적어도 M 거리를 가지고 있다는 것이다.

  - M : 기준이 되는 초평면에서의 마진(margin)

  - (9.9) : M을 가장 최대화하는 방향으로 최적화한다.

### 9.1.5 최대 마진 분류기로 분류하기 어려운 경우

- 분류할 수 있는 초평면이 없는 경우? 
  - soft margin이라는 것을 활용하여 정확히가 아니라 '대부분' 분류하는 초평면을 찾는 것이 support vector classifier이다.





## 9.2 Support Vector Classifier

### 9.2.1 Support Vector Classifier 소개

- 왜 Support Vector Classifier를 사용할까?
  - robustness 증가 : 현실에서는 정확히 나누어진 경우가 없고, support vector classifier는 '대부분' 잘 분류해내기 때문

- 몇몇의 관측치가 incorrect side of margin만 아니라 incorrect side of the hyperplane에 있도록 허락한다. 즉, 몇몇 관측치가 margin을 넘어서는 것을 허락하기 떄문에 margin이 *soft*하다고도 표현한다. 

  *incorrect side of margin : 1, 8

  *incorrect side of hyperplane & incorrect side of margin: 11, 12

  ![image](https://user-images.githubusercontent.com/32008883/36885092-269efc18-1e28-11e8-9334-806668cb0fa8.png)

  ​

  ​

### 9.2.2 자세히 살펴보는 Support Vector Classifier



![img](https://user-images.githubusercontent.com/32008883/36407065-db26b51e-163e-11e8-931c-d152f7f9df1a.png)

- - (9.14) :  (9.14)에 있는 $\epsilon_1,...,\epsilon_n$ 는  *slack variables*라고 부르는데, 개별 관측치가 margin이나 초평면의 잘못된 영역에 있는 것을 허락해주는 것이다. slack variable은 margin이나 초평면에 비교해서 i번째 관측치가 어디 있는지 알려준다.  

    - $\epsilon_i = 0$ 일 때, $i$ 번째 관측치는 margin기준 바른 영역에 위치한다.
    - $\epsilon_i > 0$ 일 때, $i$ 번째 관측치는 margin기준 반대 영역에 위치한다. 
    - $\epsilon_i > 1$ 일 때,  $i$ 번째 관측치는 hyperplane기준 반대 영역에 위치한다....(a)

    ​

  - (9.15):  slack variable 들의 총합의 기준이 되는 $C$ 는 마진이나 초평면 기준에 어긋나는 것들에 대한 민감도를 조절할 수 있는 튜닝 파라미터이다. 관측치들이 margin기준 반대 영역에 위치하는 잘못을 저질러도 봐주는 예산 정도라고 봐도 된다. 

    - $C = 0$ 이면 최대 마진 분류기와 같아진다.
    - $C > 0$ 이면 , 최대 C개의 관측치들이 hyperplane 기준 반대 영역에 위치한다. ((a)와 (9.15)에 의해)
    - 예산인 $C$ 가 증가하면, margin에 대한 violation을 더욱 봐주게 되므로, margin의 폭이 넓어진다. 
    - cross-validation 으로 $C$ 를 선택한다.
    - bias-variance trade-off :
      - $C$ 가 작을 때, low bias & high variance (엄격하게 하니까)
      - $C$ 가 클 때, high bias & low variance (대충 맞게 맞추니까)

  - (9.12)-(9.15): margin위에 있거나 margin을 넘어서는 support vector들만 초평면에 영향을 주고, 분류기가 형성된다. 마진으로부터 멀리 떨어진 관측치는 support vector classifier에 영향을 주지 않는다.

    - $C$ 가 클 때, margin이 넓어서, 많은 support vector가 있다 $\to$  high bias & low variance
    - $C$ 가 작을 때, 더 적은 support vectors $\to$  low bias & high variance
    - support vectors에 의해서만 분류 기준이 결정된다는 것은 초평면들에서 꽤 먼 관측지들에 대해서는 robust하다는 의미이다.

    ![img](https://user-images.githubusercontent.com/32008883/36407468-380d0cbc-1642-11e8-8d5a-3e099f7e52db.png)

    ![img](https://user-images.githubusercontent.com/32008883/36407476-47edf236-1642-11e8-8a3a-bec00f402398.png)

    ​

## 9.3 Support Vector Machines(SVM)

- SVM은 Support Vector Classifier 의 non-linear 버전이다. 

### 9.3.1 Classification with Non-linear Decision Boundaries

- 제곱, 세제곱, 등 다차항을 이용하여 feature space를 확장하는 방법

  "feature space?" 설명변수들의 공간. 참고> <https://stats.stackexchange.com/questions/46425/what-is-feature-space>

  - ex) 제곱항 - 2p개의 features($X_1,X_1^2,X_2,X_2^2, ... , X_p, X_p^2$)을 사용한 support vector classifier 

  - 앞에서 살펴본 (9.12)–(9.15) 은 다음과 같이 변화한다.

    ![img](https://user-images.githubusercontent.com/32008883/36463122-1b7bb912-170b-11e8-92a4-4009f2e6f82e.png)

  - 설명변수(predictor)를 변형하여 다른 함수를 사용할 수 있지만, 너무 많은 feature는 좋지 않다.

### 9.3.2 The Support Vector Machine

-  support vector classifier에 *kernels*를 사용하면 feature space를 확대할 수 있다.  (feature space를 확대하게 되면 non-linear한 경계에 유용하다.)

-  linear support vector classifier의 f(x)는 다음과 같이 표현할 수 있다.((9.14)의 수식 앞부분)

   -  $x$ 는 test observation이고 $x_i$ 는 training observation 이다.

   ![image](https://user-images.githubusercontent.com/32008883/36789744-ef181d04-1cd5-11e8-8f53-8fa62461258e.png)

   단, ![image](https://user-images.githubusercontent.com/32008883/36789789-1635c7ba-1cd6-11e8-9043-98ac282b0e81.png)

   표기는 관측치들의 내적값이다. 즉, linear classfier $f(x)$ 에서 계수를 계산하려면 내적값만 있으면 된다. (정확한 수식 유도는 너무 어려워서 생략....내적을 이용하여 표현할 수 있다는 것만 알고 넘어갑시다)

   *내적값을 사용하는 이유 : (9.12)~(9.15)에서 얻은 평면과 포인트사이 수직거리는 결국 내적과 관련이 있다. 단위벡터(unit vector) * 방향벡터는 평면과 점 사이 수직거리가 되기 떄문이다.

   관련 참고 내용(2번 내용 참고):<http://ifyouwanna.tistory.com/entry/%EB%82%B4%EC%A0%81%EC%9D%98-%ED%99%9C%EC%9A%A9> 

   ​

   (9. 18)에서  training observation이 support vector일 경우에만 $\alpha_i$ 가 0이 아니고 support vector가 아니면 $\alpha_i$가 0이다. 즉, support vector일 경우에만 유의미한 계수를 가진다. $S$가 support vector들의 집합이라면 (9.18)은 다음과 같이 표기할 수 있다.

   ![image](https://user-images.githubusercontent.com/32008883/36789978-a42e229c-1cd6-11e8-9aab-3d46f877f103.png)

   ​

   ​


- $K(x_i, x_{i^`})$ 

  SVM은 Support Vector Classifier에 non-linear 커널을 활용한 것인데, 커널의 역할과 종류들을 알아보자. 하지만 너무 수식적인 부분들은 어려워서 의미 정도만 파악하고 넘어가자...

  - *kernel* : 커널은 두 개 관측치의 유사도를 양적으로 표현하는 함수이다.  K라고 표기한다.

  ​

  - a linear kernel (Pearson (standard) correlation을 사용하여 유사도 측정):

    (참고: 상관계수와 내적의 관계 <https://wikidocs.net/6957>)

    ![img](https://user-images.githubusercontent.com/32008883/36468104-641927d0-1725-11e8-8581-a0ab01538a71.png)

    ​
    non-linear kernel 들을 살펴보기전에 **SVM**를 kernel로 사용하여 표현해보자.

    - support vector classifier가 non-linear kernel과 결합했을때, support vector machine이라고 한다.

      ![img](https://user-images.githubusercontent.com/32008883/36468162-9c73bf14-1725-11e8-9e48-b3887f91dcf3.png)

  - polynomial kernel(flexible. non-linear):

    - d: 양수

    ![img](https://user-images.githubusercontent.com/32008883/36468094-5318c418-1725-11e8-999d-765c83f1346a.png)

  - radial kernel(non-linear):

    ![img](https://user-images.githubusercontent.com/32008883/36468357-7c715ce8-1726-11e8-8385-0ff0575c7698.png)

    ​

    -  $\gamma$ : 양의 상수
    -  만약 주어진 test 관측치 $$x^* = (x^*_1 ... x^*_p)^T$$가 training 관측치로부터 유클리디안 거리가 크다면, $$\sum_{j=1}^p(x^*_j - x_{ij})^2$$ 가 클 것이고, (9.24)는 매우 작아진다. 즉, (9.23)에서 $$f(x^*)$$ 에 영향을 주지 않게 된다. 즉,  $$x^*$$ 에서 멀리 떨어진 training관측치 $$x_i$$ 는 test 관측치 분류에 어떠한 영향도 끼치지 않는다.
    -  the radial kernel은 오직 근처의 training 관측치들만 test관측치에 영향을 끼치기 때문에 *local*하다. 


- ​

  ![img](https://user-images.githubusercontent.com/32008883/36468484-dafc275c-1726-11e8-8de2-7c950e2287ee.png)

- 좌측 : polynomial kernel 사용 // 우측 : radial kernel 사용

  ​

  ​

  ​


### 9.3.2 Heart Disease Data에 적용

ROC curve : True positive rate (민감도) 와 False positive rate(1-특이도) 를 축으로 하는 곡선. ㄱ의 좌우대칭 형태일 수록 좋은 것. 

- ROC curve 과 민감도, 특이도에 관련된 부분은 다음 링크 참고 <https://sites.google.com/site/torajim/articles/performance_measure>
- 하단의 그래프는 test set기준 ROC curve


- 좌측 : SVM with polynomial kernel of degree d=1 // LDA  모두 비슷하게 잘 작동했다.
- 우측 :  SVM with a radial kernel , training set에서는 $\gamma $ 가 커질수록 ($\gamma = 10^{-1}$)일 때 분류를 가장 잘했으나 test 에서는  ($\gamma = 10^{-2} or   \gamma = 10^{-3}$)  가 ($\gamma = 10^{-1}$) 보다 더 좋은 결과를 얻었다. $\gamma$ 가 클 수록 거리가 먼 것에 대해 엄격해져서 너무 커지게 되면 overfitting문제가 발생한 것으로 보인다.



![img](https://user-images.githubusercontent.com/32008883/36712482-5b718c02-1bcc-11e8-867d-4d6551822c1d.png)



## 9.4 SVMs with More than Two Classes

### 9.4.1 One-Versus-One Classification

K개의 class가 있을 때, ${K}\choose{2}$ 개의 SVMs가 두개의 class씩 비교한다. 예를 들어, 하나의 SVM이  $k$번째 class(+1)과  $k^`$ 번째 class(-1)를 비교하여 관측치를 할당한다.최종적으로 ${K}\choose{2}$ 번의 분류 중 가장 많이 할당된 class를 부여한다. 



### 9.4.2 One-Versus-All Classification

$K$개의 SVMs가 하나의 class와 나머지 $K-1$ class들을 비교하는 방법이다.  $\beta_{0k} + \beta_{1k}x_1^* +  \beta_{2k}x_2^* + ... +  \beta_{pk}x_p^*$ 가 가장 큰 class에 관측치를 배정하는데,  hyperplane에서 가장 먼 경우 해당 class에 배정하는 것이다.  이는 test 관측치가 다른 class보다 $k$번째 class에 속할 확신의 정도라고 볼 수 있다.



## 9.5 로지스틱 회귀와 SVM의 관계

(9.12)-(9.15)는 다음과 같이 쓰일 수 있다.  

![img](https://user-images.githubusercontent.com/32008883/36713135-6169c798-1bcf-11e8-8df4-d0581e791a32.png)

- $\lambda$ 는 양수의 tuning 파라미터 

  - $\lambda$ 가 크면 $\beta_1,..,\beta_p$가 작아지고, 마진에 대한 violations에 대해 더 관대해진다. $\to$ a low-variance but high-bias classifier 

    $\lambda$가 작으면, 마진을 넘어서는 결과가 별로 없다.  $\to$  a high-variance but low-bias classifier

    $\lambda$ 는 (9.15)의  $C$과 대응한다. margin에 대한 violations와 bias, variance를 control하기 때문이다.

  - (9.25)의 패널티항인 $\lambda\sum_{j=1}^p \beta_j^2$ 는 ridge 패널티 항으로 볼 수 있고, 두 패널티 항 모두 bias-variance trade-off 역할을 한다.

- (9.25) 는 **"Loss + Penalty"**형태라고 볼 수 있다. (9.26)과 같이 표현할 수 있다.

  ![img](https://user-images.githubusercontent.com/32008883/36713282-225dd00c-1bd0-11e8-9fdb-4a7389fbc0b8.png)

  - ridege 회귀와 lasso 회귀에 적용하면, Loss function부분은 다음과 같다. 

    ![img](https://user-images.githubusercontent.com/32008883/36713309-4b607f04-1bd0-11e8-9f0c-e72929252a78.png)

  - (9.25)의 경우 Loss function 은 다음과 같다.

    ![img](https://user-images.githubusercontent.com/32008883/36713414-d1127940-1bd0-11e8-8348-51b95a44eaf6.png)

    - hinge loss라고 부르며, 로지스틱 회귀의 loss function 과 밀접히 관련있다.(수식적은 증명은 생략..)

    - 잘 분류가 됐다면, $y_i(\beta_0 + \beta_1x_{i1} + \beta_2x_{i2} + ... +\beta_px_{ip}) >= M(1-\epsilon) ... (9.14)$ 에서 $\epsilon = 0$ 이다. 그런데 이 hinge loss function으로는, margin은 1로 가정한다. 따라서 마진 기준 올바른 영역에 있는 관측치들은 loss function이 0이 된다.

      - support vector들만 분류기에 영향을 주고, 마진 기준 올바른 영역에 있는 관측치들은 loss function이 0이기 때문에 영향을 주지 않는다.

    -  하지만 로지스틱 회귀에서는, decision 경계에서 먼 관측치들은 loss function이 정확히 0이 아니라 점점 매우 작은 값을 보인다. 

      ​

      ![img](https://user-images.githubusercontent.com/32008883/36713595-c662b0d6-1bd1-11e8-92dc-b4fb80cbd17e.png)

- class가 잘 나뉘어져있다면, SVM이 로지스틱 회귀보다 선호된다. 겹치는 영역이 많다면, 로지스틱 회귀가 더 선호된다. 

  - 참고 : <https://stats.stackexchange.com/questions/254124/why-does-logistic-regression-become-unstable-when-classes-are-well-separated>

- 역사적인 이유에서, non-linear kernel의 적용은 로지스틱 회귀나 다른 방법들에서보다 SVM에서 더 많이 쓰인다.