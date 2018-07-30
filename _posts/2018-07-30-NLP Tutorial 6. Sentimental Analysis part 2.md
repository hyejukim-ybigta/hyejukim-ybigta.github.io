---
layout: post
title: "NLP_film_reviews_setimental_analysis_part2"
author: "hyeju.kim"
categories: NLP
tags: [NLP]
image: nlp_tutorial6.png
---



출처: https://programmers.co.kr/learn/courses/21/lessons/1701

** 다른 부분은 강의안과 동일하나 헷갈리는 두 부분을 좀 더 파헤쳐 보았습니다. **

- [평균벡터와 관련된 알고리즘 이해하기](#avg-feature-vector-from-scratch-by-hj)

- [bag of centroids와 관련된 알고리즘 이해하기](#bag-of-centroids-from-scratch-by-hj)


# gensim을 활용한 word2vec 
# -> avg feature vector/k means clustering 
# -> random forest classifier로 학습



## 튜토리얼 파트 2 Word Vectors
- 딥러닝 기법인 Word2Vec을 통해 단어를 벡터화해본다.
- t-SNE를 통해 벡터화한 데이터를 시각화해본다.
- 딥러닝과 지도학습의 랜덤포레스트를 사용하는 하이브리드 방식을 사용한다.

## Word2Vec(Word Embedding to Vector)

컴퓨터는 숫자만 인식할 수 있고 한글, 이미지는 바이너리 코드로 저장된다.

튜토리얼 파트1에서는 Bag of Word라는 개념을 사용해서 문자를 벡터화하여 머신러닝 알고리즘이 이해할 수 있도록 벡터화해주는 작업을 하였다.

- one hot encoding(예 [0000001000]) 혹은 Bag of Word에서 vector size가 매우 크고 sparse 하므로 neural net 성능이 잘 나오지 않는다.
- 주위 단어가 비슷하면 해당 단어의 의미는 유사하다라는 아이디어
- 단어를 트레이닝시킬 때 주위 단어를 label로 매치하여 최적화
- 단어를 의미를 내포한 dense vector로 매칭시키는 것

Word2Vec은 분산된 텍스트 표현을 사용하여 개념 간 유사성을 본다. 예를 들어, 파리와 프랑스가 베를린과 독일이 (수도와 나라) 같은 방식으로 관련되어 있음을 이해한다.

이미지 - 도시와 나라관의 관계

이미지 출처 : https://opensource.googleblog.com/2013/08/learning-meaning-behind-words.html

단어의 임베딩 과정을 실시간으로 시각화 : word embedding visual inspector

### CBOW and skip-gram

출처 : https://arxiv.org/pdf/1301.3781.pdf
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. In Proceedings of NIPS, 2013.

CBOW와 Skip-Gram 기법이 있다.

CBOW(continuous bag-of-words)는 전체 텍스트로 하나의 단어를 예측하기 때문에 작은 데이터 세트일 수록 유리하다.
아래 예제에서 __ 에 들어갈 단어를 예측한다.

1) __가 맛있다. 

2) __를 타는 것이 재미있다. 

3) 평소보다 두 __로 많이 먹어서 __가 아프다.

* Skip-Gram은 타겟 단어들로부터 원본 단어를 역으로 예측하는 것이다. CBOW와는 반대로 컨텍스트-타겟 쌍을 새로운 발견으로 처리하고 큰 규모의 데이터셋을 가질 때 유리하다.

* 배라는 단어 주변에 올 수 있는 단어를 예측한다.

1) *배*가 맛있다. 

2) *배*를 타는 것이 재미있다. 

3) 평소보다 두 *배*로 많이 먹어서 *배*가 아프다.

### word2vec 추가자료
- [word2vec 모델 · 텐서플로우 문서 한글 번역본](https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/tutorials/word2vec/)
- [Word2Vec으로 문장 분류하기 · ratsgo's blog](https://ratsgo.github.io/natural%20language%20processing/2017/03/08/word2vec/)
- Efficient Estimation of Word Representations in Vector Space
- Distributed Representations of Words and Phrases and their Compositionality
- CS224n: Natural Language Processing with Deep Learning
- Word2Vec Tutorial - The Skip-Gram Model · Chris McCormick

# Gensim -word2vec을 python에서!

- [gensim: models.word2vec – Deep learning with word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [gensim: Tutorials](https://radimrehurek.com/gensim/tutorial.html)
- [한국어와 NLTK, Gensim의 만남 - PyCon Korea 2015](https://www.lucypark.kr/docs/2015-pyconkr/#7)


## 먼저 저번시간에 했던 전처리


```python
# 출력이 너무 길어지지 않게 하기 위해 찍지 않도록 했으나
# 실제 학습할 때는 아래 두 줄을 주석처리 하는 것을 권장한다.
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd

train = pd.read_csv('data/labeledTrainData.tsv', 
                    header=0, delimiter='\t', quoting=3)
test = pd.read_csv('data/testData.tsv', 
                   header=0, delimiter='\t', quoting=3)
unlabeled_train = pd.read_csv('data/unlabeledTrainData.tsv', 
                              header=0, delimiter='\t', quoting=3)

print(train.shape)
print(test.shape)
print(unlabeled_train.shape)

print(train['review'].size)
print(test['review'].size)
print(unlabeled_train['review'].size)
```

    (25000, 3)
    (25000, 2)
    (50000, 2)
    25000
    25000
    50000



```python
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sentiment</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>"5814_8"</td>
      <td>1</td>
      <td>"With all this stuff going down at the moment ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"2381_9"</td>
      <td>1</td>
      <td>"\"The Classic War of the Worlds\" by Timothy ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>"7759_3"</td>
      <td>0</td>
      <td>"The film starts with a manager (Nicholas Bell...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"3630_4"</td>
      <td>0</td>
      <td>"It must be assumed that those who praised thi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>"9495_8"</td>
      <td>1</td>
      <td>"Superbly trashy and wondrously unpretentious ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>"12311_10"</td>
      <td>"Naturally in a film who's main themes are of ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>"8348_2"</td>
      <td>"This movie is a disaster within a disaster fi...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>"5828_4"</td>
      <td>"All in all, this is a movie for kids. We saw ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>"7186_2"</td>
      <td>"Afraid of the Dark left me with the impressio...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>"12128_7"</td>
      <td>"A very accurate depiction of small time mob l...</td>
    </tr>
  </tbody>
</table>
</div>



이 전 tutorial에서 했던 내용 그대로 class로 담겨있음


```python
import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from multiprocessing import Pool

class KaggleWord2VecUtility(object):

    @staticmethod
    def review_to_wordlist(review, remove_stopwords=False):
        # 1. HTML 제거
        review_text = BeautifulSoup(review, "html.parser").get_text()
        # 2. 특수문자를 공백으로 바꿔줌
        review_text = re.sub('[^a-zA-Z]', ' ', review_text)
        # 3. 소문자로 변환 후 나눈다.
        words = review_text.lower().split()
        # 4. 불용어 제거
        if remove_stopwords:
            stops = set(stopwords.words('english'))
            words = [w for w in words if not w in stops]
        # 5. 어간추출
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in words]
        # 6. 리스트 형태로 반환
        return(words)

    @staticmethod
    def review_to_join_words( review, remove_stopwords=False ):
        words = KaggleWord2VecUtility.review_to_wordlist(\
            review, remove_stopwords=False)
        join_words = ' '.join(words)
        return join_words

    @staticmethod
    def review_to_sentences( review, remove_stopwords=False ):
        # punkt tokenizer를 로드한다.
        """
        이 때, pickle을 사용하는데
        pickle을 통해 값을 저장하면 원래 변수에 연결 된 참조값 역시 저장된다.
        저장된 pickle을 다시 읽으면 변수에 연결되었던
        모든 레퍼런스가 계속 참조 상태를 유지한다.
        """
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # 1. nltk tokenizer를 사용해서 단어로 토큰화 하고 공백 등을 제거한다.
        raw_sentences = tokenizer.tokenize(review.strip())
        # 2. 각 문장을 순회한다.
        sentences = []
        for raw_sentence in raw_sentences:
            # 비어있다면 skip
            if len(raw_sentence) > 0:
                # 태그제거, 알파벳문자가 아닌 것은 공백으로 치환, 불용어제거
                sentences.append(\
                    KaggleWord2VecUtility.review_to_wordlist(\
                    raw_sentence, remove_stopwords))
        return sentences


    # 참고 : https://gist.github.com/yong27/7869662
    # http://www.racketracer.com/2016/07/06/pandas-in-parallel/
    # 속도 개선을 위해 멀티 스레드로 작업하도록
    @staticmethod
    def _apply_df(args):
        df, func, kwargs = args
        return df.apply(func, **kwargs)

    @staticmethod
    def apply_by_multiprocessing(df, func, **kwargs):
        # 키워드 항목 중 workers 파라메터를 꺼냄
        workers = kwargs.pop('workers')
        # 위에서 가져온 workers 수로 프로세스 풀을 정의
        pool = Pool(processes=workers)
        # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
        result = pool.map(KaggleWord2VecUtility._apply_df, [(d, func, kwargs)
                for d in np.array_split(df, workers)])
        pool.close()
        # 작업 결과를 합쳐서 반환
        return pd.concat(result)
```


```python
KaggleWord2VecUtility.review_to_wordlist(train['review'][0])[:10]
```




    ['with', 'all', 'this', 'stuff', 'go', 'down', 'at', 'the', 'moment', 'with']




```python
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to /home/jovyan/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.





    True




```python
sentences = []
for review in train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)
```


```python
for review in unlabeled_train["review"]:
    sentences += KaggleWord2VecUtility.review_to_sentences(
        review, remove_stopwords=False)
```


```python
len(sentences)
```




    795538




```python
sentences[0][:10]
```




    ['with', 'all', 'this', 'stuff', 'go', 'down', 'at', 'the', 'moment', 'with']




```python
sentences[795537][:10]
```




    ['pathmark', 'mean', 'save']



word2vec에서는 문맥이 중요하므로 remove_stopwords= False 

## Gensim으로 word2vec

-[gensim: models.word2vec – Deep learning with word2vec](https://radimrehurek.com/gensim/models/word2vec.html)

### Word2Vec 모델의 파라메터

- 아키텍처 : 아키텍처 옵션은 skip-gram (default) 또는 CBOW 모델이다. skip-gram (default)은 느리지만 더 나은 결과를 낸다.

- 학습 알고리즘 : Hierarchical softmax (default) 또는 negative 샘플링. 여기에서는 기본값이 잘 동작한다.

- 빈번하게 등장하는 단어에 대한 다운 샘플링 : Google 문서는 .00001에서 .001 사이의 값을 권장한다. 여기에서는 0.001에 가까운 값이 최종 모델의 정확도를 높이는 것으로 보여진다.

- 단어 벡터 차원 : 많은 feature를 사용한다고 항상 좋은 것은 아니지만 대체적으로 좀 더 나은 모델이 된다. 합리적인 값은 수십에서 수백 개가 될 수 있고 여기에서는 300으로 지정했다.

- 컨텍스트 / 창 크기 : 학습 알고리즘이 고려해야 하는 컨텍스트의 단어 수는 얼마나 될까? hierarchical softmax 를 위해 좀 더 큰 수가 좋지만 10 정도가 적당하다.

- Worker threads : 실행할 병렬 프로세스의 수로 컴퓨터마다 다르지만 대부분의 시스템에서 4에서 6 사이의 값을 사용하다.

- 최소 단어 수 : 어휘의 크기를 의미 있는 단어로 제한하는 데 도움이 된다. 모든 문서에서 여러 번 발생하지 않는 단어는 무시된다. 10에서 100 사이가 적당하며, 이 경진대회의 데이터는 각 영화가 30개씩의 리뷰가 있기 때문에 개별 영화 제목에 너무 많은 중요성이 붙는 것을 피하고자 최소 단어 수를 40으로 설정한다. 그 결과 전체 어휘 크기는 약 15,000단어가 된다. 높은 값은 제한 된 실행시간에 도움이 된다.


```python
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.INFO)
```


```python
# 파라메터값 지정
num_features = 300 # 문자 벡터 차원 수
min_word_count = 40 # 최소 문자 수
num_workers = 4 # 병렬 처리 스레드 수
context = 10 # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도수 Downsample

# 초기화 및 모델 학습
from gensim.models import word2vec

# 모델 학습
model = word2vec.Word2Vec(sentences, 
                          workers=num_workers, 
                          size=num_features, 
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)
model
```

    2018-07-30 05:29:27,353 : INFO : collecting all words and their counts
    2018-07-30 05:29:27,354 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    2018-07-30 05:29:27,449 : INFO : PROGRESS: at sentence #10000, processed 225803 words, keeping 12465 word types
    2018-07-30 05:29:27,519 : INFO : PROGRESS: at sentence #20000, processed 451892 words, keeping 17070 word types
    2018-07-30 05:29:27,584 : INFO : PROGRESS: at sentence #30000, processed 671314 words, keeping 20370 word types
    2018-07-30 05:29:27,652 : INFO : PROGRESS: at sentence #40000, processed 897814 words, keeping 23125 word types
    2018-07-30 05:29:27,715 : INFO : PROGRESS: at sentence #50000, processed 1116962 words, keeping 25365 word types
    2018-07-30 05:29:27,778 : INFO : PROGRESS: at sentence #60000, processed 1338403 words, keeping 27283 word types
    2018-07-30 05:29:27,840 : INFO : PROGRESS: at sentence #70000, processed 1561579 words, keeping 29024 word types
    2018-07-30 05:29:27,900 : INFO : PROGRESS: at sentence #80000, processed 1780886 words, keeping 30603 word types
    2018-07-30 05:29:27,958 : INFO : PROGRESS: at sentence #90000, processed 2004995 words, keeping 32223 word types
    2018-07-30 05:29:28,025 : INFO : PROGRESS: at sentence #100000, processed 2226966 words, keeping 33579 word types
    2018-07-30 05:29:28,085 : INFO : PROGRESS: at sentence #110000, processed 2446580 words, keeping 34827 word types
    2018-07-30 05:29:28,143 : INFO : PROGRESS: at sentence #120000, processed 2668775 words, keeping 36183 word types
    2018-07-30 05:29:28,203 : INFO : PROGRESS: at sentence #130000, processed 2894303 words, keeping 37353 word types
    2018-07-30 05:29:28,259 : INFO : PROGRESS: at sentence #140000, processed 3107005 words, keeping 38376 word types
    2018-07-30 05:29:28,324 : INFO : PROGRESS: at sentence #150000, processed 3332627 words, keeping 39556 word types
    2018-07-30 05:29:28,375 : INFO : PROGRESS: at sentence #160000, processed 3555315 words, keeping 40629 word types
    2018-07-30 05:29:28,422 : INFO : PROGRESS: at sentence #170000, processed 3778655 words, keeping 41628 word types
    2018-07-30 05:29:28,469 : INFO : PROGRESS: at sentence #180000, processed 3999236 words, keeping 42599 word types
    2018-07-30 05:29:28,518 : INFO : PROGRESS: at sentence #190000, processed 4224449 words, keeping 43461 word types
    2018-07-30 05:29:28,565 : INFO : PROGRESS: at sentence #200000, processed 4448603 words, keeping 44301 word types
    2018-07-30 05:29:28,616 : INFO : PROGRESS: at sentence #210000, processed 4669967 words, keeping 45212 word types
    2018-07-30 05:29:28,667 : INFO : PROGRESS: at sentence #220000, processed 4894968 words, keeping 46134 word types
    2018-07-30 05:29:28,715 : INFO : PROGRESS: at sentence #230000, processed 5117545 words, keeping 46986 word types
    2018-07-30 05:29:28,764 : INFO : PROGRESS: at sentence #240000, processed 5345050 words, keeping 47854 word types
    2018-07-30 05:29:28,809 : INFO : PROGRESS: at sentence #250000, processed 5559165 words, keeping 48699 word types
    2018-07-30 05:29:28,856 : INFO : PROGRESS: at sentence #260000, processed 5779146 words, keeping 49469 word types
    2018-07-30 05:29:28,903 : INFO : PROGRESS: at sentence #270000, processed 6000435 words, keeping 50416 word types
    2018-07-30 05:29:28,950 : INFO : PROGRESS: at sentence #280000, processed 6226314 words, keeping 51640 word types
    2018-07-30 05:29:28,997 : INFO : PROGRESS: at sentence #290000, processed 6449474 words, keeping 52754 word types
    2018-07-30 05:29:29,044 : INFO : PROGRESS: at sentence #300000, processed 6674077 words, keeping 53755 word types
    2018-07-30 05:29:29,091 : INFO : PROGRESS: at sentence #310000, processed 6899391 words, keeping 54734 word types
    2018-07-30 05:29:29,138 : INFO : PROGRESS: at sentence #320000, processed 7124278 words, keeping 55770 word types
    2018-07-30 05:29:29,184 : INFO : PROGRESS: at sentence #330000, processed 7346021 words, keeping 56687 word types
    2018-07-30 05:29:29,232 : INFO : PROGRESS: at sentence #340000, processed 7575533 words, keeping 57629 word types
    2018-07-30 05:29:29,278 : INFO : PROGRESS: at sentence #350000, processed 7798803 words, keeping 58485 word types
    2018-07-30 05:29:29,324 : INFO : PROGRESS: at sentence #360000, processed 8019466 words, keeping 59345 word types
    2018-07-30 05:29:29,371 : INFO : PROGRESS: at sentence #370000, processed 8246654 words, keeping 60161 word types
    2018-07-30 05:29:29,417 : INFO : PROGRESS: at sentence #380000, processed 8471801 words, keeping 61069 word types
    2018-07-30 05:29:29,465 : INFO : PROGRESS: at sentence #390000, processed 8701551 words, keeping 61810 word types
    2018-07-30 05:29:29,511 : INFO : PROGRESS: at sentence #400000, processed 8924500 words, keeping 62546 word types
    2018-07-30 05:29:29,557 : INFO : PROGRESS: at sentence #410000, processed 9145850 words, keeping 63263 word types
    2018-07-30 05:29:29,602 : INFO : PROGRESS: at sentence #420000, processed 9366930 words, keeping 64024 word types
    2018-07-30 05:29:29,649 : INFO : PROGRESS: at sentence #430000, processed 9594467 words, keeping 64795 word types
    2018-07-30 05:29:29,696 : INFO : PROGRESS: at sentence #440000, processed 9821218 words, keeping 65539 word types
    2018-07-30 05:29:29,742 : INFO : PROGRESS: at sentence #450000, processed 10044980 words, keeping 66378 word types
    2018-07-30 05:29:29,790 : INFO : PROGRESS: at sentence #460000, processed 10277740 words, keeping 67158 word types
    2018-07-30 05:29:29,836 : INFO : PROGRESS: at sentence #470000, processed 10505665 words, keeping 67775 word types
    2018-07-30 05:29:29,882 : INFO : PROGRESS: at sentence #480000, processed 10726049 words, keeping 68500 word types
    2018-07-30 05:29:29,929 : INFO : PROGRESS: at sentence #490000, processed 10952793 words, keeping 69256 word types
    2018-07-30 05:29:29,975 : INFO : PROGRESS: at sentence #500000, processed 11174449 words, keeping 69892 word types
    2018-07-30 05:29:30,021 : INFO : PROGRESS: at sentence #510000, processed 11399724 words, keeping 70593 word types
    2018-07-30 05:29:30,067 : INFO : PROGRESS: at sentence #520000, processed 11623075 words, keeping 71267 word types
    2018-07-30 05:29:30,113 : INFO : PROGRESS: at sentence #530000, processed 11847473 words, keeping 71877 word types
    2018-07-30 05:29:30,159 : INFO : PROGRESS: at sentence #540000, processed 12072088 words, keeping 72537 word types
    2018-07-30 05:29:30,206 : INFO : PROGRESS: at sentence #550000, processed 12297639 words, keeping 73212 word types
    2018-07-30 05:29:30,252 : INFO : PROGRESS: at sentence #560000, processed 12518929 words, keeping 73861 word types
    2018-07-30 05:29:30,299 : INFO : PROGRESS: at sentence #570000, processed 12748076 words, keeping 74431 word types
    2018-07-30 05:29:30,344 : INFO : PROGRESS: at sentence #580000, processed 12969572 words, keeping 75087 word types
    2018-07-30 05:29:30,391 : INFO : PROGRESS: at sentence #590000, processed 13195097 words, keeping 75733 word types
    2018-07-30 05:29:30,437 : INFO : PROGRESS: at sentence #600000, processed 13417295 words, keeping 76294 word types
    2018-07-30 05:29:30,482 : INFO : PROGRESS: at sentence #610000, processed 13638318 words, keeping 76952 word types
    2018-07-30 05:29:30,529 : INFO : PROGRESS: at sentence #620000, processed 13864643 words, keeping 77503 word types
    2018-07-30 05:29:30,575 : INFO : PROGRESS: at sentence #630000, processed 14088929 words, keeping 78066 word types
    2018-07-30 05:29:30,621 : INFO : PROGRESS: at sentence #640000, processed 14309712 words, keeping 78692 word types
    2018-07-30 05:29:30,669 : INFO : PROGRESS: at sentence #650000, processed 14535468 words, keeping 79295 word types
    2018-07-30 05:29:30,714 : INFO : PROGRESS: at sentence #660000, processed 14758258 words, keeping 79864 word types
    2018-07-30 05:29:30,760 : INFO : PROGRESS: at sentence #670000, processed 14981651 words, keeping 80381 word types
    2018-07-30 05:29:30,807 : INFO : PROGRESS: at sentence #680000, processed 15206483 words, keeping 80912 word types
    2018-07-30 05:29:30,852 : INFO : PROGRESS: at sentence #690000, processed 15428676 words, keeping 81482 word types
    2018-07-30 05:29:30,899 : INFO : PROGRESS: at sentence #700000, processed 15657382 words, keeping 82074 word types
    2018-07-30 05:29:30,945 : INFO : PROGRESS: at sentence #710000, processed 15880371 words, keeping 82560 word types
    2018-07-30 05:29:30,992 : INFO : PROGRESS: at sentence #720000, processed 16105658 words, keeping 83036 word types
    2018-07-30 05:29:31,039 : INFO : PROGRESS: at sentence #730000, processed 16332039 words, keeping 83571 word types
    2018-07-30 05:29:31,085 : INFO : PROGRESS: at sentence #740000, processed 16553072 words, keeping 84127 word types
    2018-07-30 05:29:31,130 : INFO : PROGRESS: at sentence #750000, processed 16771399 words, keeping 84599 word types
    2018-07-30 05:29:31,175 : INFO : PROGRESS: at sentence #760000, processed 16990803 words, keeping 85068 word types
    2018-07-30 05:29:31,221 : INFO : PROGRESS: at sentence #770000, processed 17217940 words, keeping 85644 word types
    2018-07-30 05:29:31,269 : INFO : PROGRESS: at sentence #780000, processed 17448086 words, keeping 86160 word types
    2018-07-30 05:29:31,316 : INFO : PROGRESS: at sentence #790000, processed 17675162 words, keeping 86665 word types
    2018-07-30 05:29:31,342 : INFO : collected 86996 word types from a corpus of 17798263 raw words and 795538 sentences
    2018-07-30 05:29:31,342 : INFO : Loading a fresh vocabulary
    2018-07-30 05:29:31,405 : INFO : effective_min_count=40 retains 11986 unique words (13% of original 86996, drops 75010)
    2018-07-30 05:29:31,405 : INFO : effective_min_count=40 leaves 17434026 word corpus (97% of original 17798263, drops 364237)
    2018-07-30 05:29:31,436 : INFO : deleting the raw counts dictionary of 86996 items
    2018-07-30 05:29:31,438 : INFO : sample=0.001 downsamples 50 most-common words
    2018-07-30 05:29:31,439 : INFO : downsampling leaves estimated 12872359 word corpus (73.8% of prior 17434026)
    2018-07-30 05:29:31,466 : INFO : estimated required memory for 11986 words and 300 dimensions: 34759400 bytes
    2018-07-30 05:29:31,467 : INFO : resetting layer weights
    2018-07-30 05:29:31,657 : INFO : training model with 4 workers on 11986 vocabulary and 300 features, using sg=0 hs=0 sample=0.001 negative=5 window=10
    2018-07-30 05:29:32,673 : INFO : EPOCH 1 - PROGRESS: at 5.76% examples, 738505 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:33,675 : INFO : EPOCH 1 - PROGRESS: at 11.84% examples, 754571 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:34,688 : INFO : EPOCH 1 - PROGRESS: at 18.25% examples, 771541 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:35,695 : INFO : EPOCH 1 - PROGRESS: at 24.15% examples, 766994 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:36,716 : INFO : EPOCH 1 - PROGRESS: at 29.98% examples, 760389 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:37,719 : INFO : EPOCH 1 - PROGRESS: at 35.73% examples, 754724 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:38,722 : INFO : EPOCH 1 - PROGRESS: at 41.54% examples, 753878 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:39,737 : INFO : EPOCH 1 - PROGRESS: at 47.34% examples, 752149 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:40,740 : INFO : EPOCH 1 - PROGRESS: at 53.38% examples, 754982 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:41,754 : INFO : EPOCH 1 - PROGRESS: at 58.84% examples, 749835 words/s, in_qsize 5, out_qsize 2
    2018-07-30 05:29:42,773 : INFO : EPOCH 1 - PROGRESS: at 64.00% examples, 740918 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:43,792 : INFO : EPOCH 1 - PROGRESS: at 69.93% examples, 741734 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:44,798 : INFO : EPOCH 1 - PROGRESS: at 76.05% examples, 744938 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:45,799 : INFO : EPOCH 1 - PROGRESS: at 81.87% examples, 745299 words/s, in_qsize 8, out_qsize 0
    2018-07-30 05:29:46,806 : INFO : EPOCH 1 - PROGRESS: at 87.59% examples, 744388 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:47,810 : INFO : EPOCH 1 - PROGRESS: at 92.56% examples, 737895 words/s, in_qsize 5, out_qsize 2
    2018-07-30 05:29:48,821 : INFO : EPOCH 1 - PROGRESS: at 98.23% examples, 736935 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:49,074 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-07-30 05:29:49,089 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-07-30 05:29:49,100 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-07-30 05:29:49,103 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-07-30 05:29:49,104 : INFO : EPOCH - 1 : training on 17798263 raw words (12872230 effective words) took 17.4s, 738154 effective words/s
    2018-07-30 05:29:50,136 : INFO : EPOCH 2 - PROGRESS: at 5.81% examples, 734331 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:51,150 : INFO : EPOCH 2 - PROGRESS: at 12.45% examples, 783350 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:52,156 : INFO : EPOCH 2 - PROGRESS: at 18.36% examples, 771376 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:53,175 : INFO : EPOCH 2 - PROGRESS: at 24.15% examples, 761043 words/s, in_qsize 5, out_qsize 2
    2018-07-30 05:29:54,181 : INFO : EPOCH 2 - PROGRESS: at 29.88% examples, 755185 words/s, in_qsize 6, out_qsize 1
    2018-07-30 05:29:55,194 : INFO : EPOCH 2 - PROGRESS: at 35.66% examples, 750360 words/s, in_qsize 6, out_qsize 1
    2018-07-30 05:29:56,212 : INFO : EPOCH 2 - PROGRESS: at 41.54% examples, 749470 words/s, in_qsize 8, out_qsize 1
    2018-07-30 05:29:57,222 : INFO : EPOCH 2 - PROGRESS: at 47.28% examples, 747799 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:58,230 : INFO : EPOCH 2 - PROGRESS: at 53.05% examples, 746774 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:29:59,243 : INFO : EPOCH 2 - PROGRESS: at 58.69% examples, 744721 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:00,257 : INFO : EPOCH 2 - PROGRESS: at 65.06% examples, 750850 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:01,269 : INFO : EPOCH 2 - PROGRESS: at 70.71% examples, 748396 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:02,291 : INFO : EPOCH 2 - PROGRESS: at 76.60% examples, 747875 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:03,310 : INFO : EPOCH 2 - PROGRESS: at 82.44% examples, 747072 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:04,325 : INFO : EPOCH 2 - PROGRESS: at 88.07% examples, 745175 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:05,331 : INFO : EPOCH 2 - PROGRESS: at 93.74% examples, 743861 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:06,344 : INFO : EPOCH 2 - PROGRESS: at 98.83% examples, 738285 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:06,506 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-07-30 05:30:06,516 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-07-30 05:30:06,524 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-07-30 05:30:06,527 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-07-30 05:30:06,528 : INFO : EPOCH - 2 : training on 17798263 raw words (12872424 effective words) took 17.4s, 739156 effective words/s
    2018-07-30 05:30:07,551 : INFO : EPOCH 3 - PROGRESS: at 5.48% examples, 702100 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:08,557 : INFO : EPOCH 3 - PROGRESS: at 11.95% examples, 759780 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:09,570 : INFO : EPOCH 3 - PROGRESS: at 17.46% examples, 737167 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:10,576 : INFO : EPOCH 3 - PROGRESS: at 23.21% examples, 735741 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:11,580 : INFO : EPOCH 3 - PROGRESS: at 28.64% examples, 727964 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:12,587 : INFO : EPOCH 3 - PROGRESS: at 33.99% examples, 719017 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:13,589 : INFO : EPOCH 3 - PROGRESS: at 39.48% examples, 717143 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:14,590 : INFO : EPOCH 3 - PROGRESS: at 44.79% examples, 713220 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:15,600 : INFO : EPOCH 3 - PROGRESS: at 50.28% examples, 712606 words/s, in_qsize 7, out_qsize 1
    2018-07-30 05:30:16,605 : INFO : EPOCH 3 - PROGRESS: at 55.61% examples, 709578 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:17,608 : INFO : EPOCH 3 - PROGRESS: at 61.42% examples, 713726 words/s, in_qsize 8, out_qsize 0
    2018-07-30 05:30:18,612 : INFO : EPOCH 3 - PROGRESS: at 66.80% examples, 711830 words/s, in_qsize 8, out_qsize 0
    2018-07-30 05:30:19,612 : INFO : EPOCH 3 - PROGRESS: at 72.20% examples, 710920 words/s, in_qsize 8, out_qsize 2
    2018-07-30 05:30:20,620 : INFO : EPOCH 3 - PROGRESS: at 78.31% examples, 715975 words/s, in_qsize 8, out_qsize 0
    2018-07-30 05:30:21,630 : INFO : EPOCH 3 - PROGRESS: at 83.73% examples, 713924 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:22,652 : INFO : EPOCH 3 - PROGRESS: at 89.63% examples, 716164 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:23,652 : INFO : EPOCH 3 - PROGRESS: at 95.53% examples, 718181 words/s, in_qsize 8, out_qsize 0
    2018-07-30 05:30:24,391 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-07-30 05:30:24,402 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-07-30 05:30:24,412 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-07-30 05:30:24,418 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-07-30 05:30:24,419 : INFO : EPOCH - 3 : training on 17798263 raw words (12871205 effective words) took 17.9s, 720060 effective words/s
    2018-07-30 05:30:25,432 : INFO : EPOCH 4 - PROGRESS: at 5.42% examples, 698383 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:26,436 : INFO : EPOCH 4 - PROGRESS: at 11.14% examples, 712058 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:27,439 : INFO : EPOCH 4 - PROGRESS: at 17.00% examples, 722222 words/s, in_qsize 8, out_qsize 0
    2018-07-30 05:30:28,444 : INFO : EPOCH 4 - PROGRESS: at 22.87% examples, 728399 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:29,453 : INFO : EPOCH 4 - PROGRESS: at 28.76% examples, 732925 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:30,459 : INFO : EPOCH 4 - PROGRESS: at 34.54% examples, 732583 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:31,464 : INFO : EPOCH 4 - PROGRESS: at 39.96% examples, 727410 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:32,471 : INFO : EPOCH 4 - PROGRESS: at 45.95% examples, 732376 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:33,475 : INFO : EPOCH 4 - PROGRESS: at 51.13% examples, 725332 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:34,485 : INFO : EPOCH 4 - PROGRESS: at 57.26% examples, 731332 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:35,497 : INFO : EPOCH 4 - PROGRESS: at 62.99% examples, 731665 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:36,509 : INFO : EPOCH 4 - PROGRESS: at 68.98% examples, 734355 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:37,511 : INFO : EPOCH 4 - PROGRESS: at 74.28% examples, 730632 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:38,518 : INFO : EPOCH 4 - PROGRESS: at 79.63% examples, 727038 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:39,519 : INFO : EPOCH 4 - PROGRESS: at 85.34% examples, 727582 words/s, in_qsize 8, out_qsize 0
    2018-07-30 05:30:40,524 : INFO : EPOCH 4 - PROGRESS: at 91.09% examples, 728379 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:41,529 : INFO : EPOCH 4 - PROGRESS: at 97.02% examples, 729896 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:42,059 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-07-30 05:30:42,060 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-07-30 05:30:42,063 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-07-30 05:30:42,074 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-07-30 05:30:42,075 : INFO : EPOCH - 4 : training on 17798263 raw words (12870862 effective words) took 17.6s, 729372 effective words/s
    2018-07-30 05:30:43,103 : INFO : EPOCH 5 - PROGRESS: at 5.54% examples, 708392 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:44,103 : INFO : EPOCH 5 - PROGRESS: at 11.38% examples, 725809 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:45,112 : INFO : EPOCH 5 - PROGRESS: at 17.23% examples, 729957 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:46,131 : INFO : EPOCH 5 - PROGRESS: at 23.09% examples, 731551 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:47,132 : INFO : EPOCH 5 - PROGRESS: at 29.04% examples, 738110 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:48,149 : INFO : EPOCH 5 - PROGRESS: at 34.60% examples, 730738 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:49,154 : INFO : EPOCH 5 - PROGRESS: at 40.13% examples, 727889 words/s, in_qsize 6, out_qsize 1
    2018-07-30 05:30:50,155 : INFO : EPOCH 5 - PROGRESS: at 46.11% examples, 733266 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:51,156 : INFO : EPOCH 5 - PROGRESS: at 52.37% examples, 741501 words/s, in_qsize 8, out_qsize 0
    2018-07-30 05:30:52,162 : INFO : EPOCH 5 - PROGRESS: at 58.29% examples, 744083 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:53,167 : INFO : EPOCH 5 - PROGRESS: at 63.78% examples, 740507 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:54,181 : INFO : EPOCH 5 - PROGRESS: at 69.31% examples, 737502 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:55,209 : INFO : EPOCH 5 - PROGRESS: at 74.85% examples, 734229 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:56,210 : INFO : EPOCH 5 - PROGRESS: at 80.37% examples, 732260 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:57,231 : INFO : EPOCH 5 - PROGRESS: at 85.79% examples, 729107 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:58,254 : INFO : EPOCH 5 - PROGRESS: at 91.32% examples, 727184 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:59,279 : INFO : EPOCH 5 - PROGRESS: at 96.30% examples, 720816 words/s, in_qsize 7, out_qsize 0
    2018-07-30 05:30:59,960 : INFO : worker thread finished; awaiting finish of 3 more threads
    2018-07-30 05:30:59,964 : INFO : worker thread finished; awaiting finish of 2 more threads
    2018-07-30 05:30:59,967 : INFO : worker thread finished; awaiting finish of 1 more threads
    2018-07-30 05:30:59,983 : INFO : worker thread finished; awaiting finish of 0 more threads
    2018-07-30 05:30:59,984 : INFO : EPOCH - 5 : training on 17798263 raw words (12870049 effective words) took 17.9s, 719391 effective words/s
    2018-07-30 05:30:59,985 : INFO : training on a 88991315 raw words (64356770 effective words) took 88.3s, 728620 effective words/s





    <gensim.models.word2vec.Word2Vec at 0x7fa5a1c13668>




```python
# 학습이 완료 되면 필요없는 메모리를 unload 시킨다.
model.init_sims(replace=True)

model_name = '300features_40minwords_10text'
# model_name = '300features_50minwords_20text'
model.save(model_name)
```

    2018-07-30 05:31:06,379 : INFO : precomputing L2-norms of word weight vectors
    2018-07-30 05:31:06,519 : INFO : saving Word2Vec object under 300features_40minwords_10text, separately None
    2018-07-30 05:31:06,520 : INFO : not storing attribute vectors_norm
    2018-07-30 05:31:06,521 : INFO : not storing attribute cum_table
    2018-07-30 05:31:07,231 : INFO : saved 300features_40minwords_10text


### 모델 결과 탐색


```python
# 유사도가 없는 단어 추출
model.wv.doesnt_match('man woman child kitchen'.split())
```




    'kitchen'




```python
model.wv.doesnt_match("france england germany berlin".split())
```

    2018-07-30 05:31:08,752 : WARNING : vectors for words {'germany', 'france'} are not present in the model, ignoring these words





    'england'




```python
# 가장 유사한 단어를 추출
model.wv.most_similar("man")
```




    [('woman', 0.6270360946655273),
     ('lad', 0.5032452344894409),
     ('businessman', 0.5023341178894043),
     ('ladi', 0.5011050701141357),
     ('gunman', 0.4861173629760742),
     ('millionair', 0.4683084487915039),
     ('men', 0.4623528718948364),
     ('widow', 0.4565393626689911),
     ('priest', 0.45500609278678894),
     ('doctor', 0.4431304335594177)]




```python
model.wv.most_similar("queen")
```




    [('princess', 0.6152504682540894),
     ('latifah', 0.5782067775726318),
     ('victoria', 0.5741004943847656),
     ('madam', 0.5545714497566223),
     ('stepmoth', 0.5542747378349304),
     ('eva', 0.5538949966430664),
     ('duchess', 0.5441377758979797),
     ('goddess', 0.5416854619979858),
     ('mistress', 0.5406708717346191),
     ('bride', 0.5390686988830566)]




```python
model.wv.most_similar("film")
```




    [('movi', 0.8590012788772583),
     ('flick', 0.6091418266296387),
     ('documentari', 0.5728434324264526),
     ('pictur', 0.5318986773490906),
     ('cinema', 0.528438925743103),
     ('masterpiec', 0.5145473480224609),
     ('sequel', 0.48509395122528076),
     ('it', 0.48382335901260376),
     ('effort', 0.4781997501850128),
     ('thriller', 0.47806960344314575)]




```python
# model.wv.most_similar("happy")
model.wv.most_similar("happi") # stemming 처리 시 
```




    [('unhappi', 0.4235895872116089),
     ('satisfi', 0.4199100732803345),
     ('sad', 0.41657835245132446),
     ('lucki', 0.4004006087779999),
     ('afraid', 0.39070242643356323),
     ('upset', 0.3887670040130615),
     ('glad', 0.37744760513305664),
     ('bitter', 0.35681796073913574),
     ('proud', 0.35108745098114014),
     ('comfort', 0.34909945726394653)]



## Word2Vec으로 벡터화 한 단어를 t-SNE 를 통해 시각화


```python
# 참고 https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

model_name = '300features_40minwords_10text'
model = g.Doc2Vec.load(model_name)

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X))
print(X[0][:10])
tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:100,:])
# X_tsne = tsne.fit_transform(X)
```

    2018-07-30 05:31:11,109 : INFO : loading Doc2Vec object from 300features_40minwords_10text
    2018-07-30 05:31:11,387 : INFO : loading wv recursively from 300features_40minwords_10text.wv.* with mmap=None
    2018-07-30 05:31:11,388 : INFO : setting ignored attribute vectors_norm to None
    2018-07-30 05:31:11,389 : INFO : loading vocabulary recursively from 300features_40minwords_10text.vocabulary.* with mmap=None
    2018-07-30 05:31:11,389 : INFO : loading trainables recursively from 300features_40minwords_10text.trainables.* with mmap=None
    2018-07-30 05:31:11,390 : INFO : setting ignored attribute cum_table to None
    2018-07-30 05:31:11,391 : INFO : loaded 300features_40minwords_10text


    11986
    [ 0.08282609 -0.06867442  0.05857689 -0.13006091 -0.10408346 -0.10806588
     -0.01058144  0.01877268  0.00539338 -0.06462868]



```python
df = pd.DataFrame(X_tsne, index=vocab[:100], columns=['x', 'y'])
df.shape
```




    (100, 2)




```python
df.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>with</th>
      <td>-7.182045</td>
      <td>17.190171</td>
    </tr>
    <tr>
      <th>all</th>
      <td>4.185929</td>
      <td>-64.974663</td>
    </tr>
    <tr>
      <th>this</th>
      <td>-72.129743</td>
      <td>-21.310174</td>
    </tr>
    <tr>
      <th>stuff</th>
      <td>22.817384</td>
      <td>-70.655331</td>
    </tr>
    <tr>
      <th>go</th>
      <td>-22.977635</td>
      <td>-64.070689</td>
    </tr>
    <tr>
      <th>down</th>
      <td>-56.804619</td>
      <td>-48.719601</td>
    </tr>
    <tr>
      <th>at</th>
      <td>-74.471437</td>
      <td>35.751977</td>
    </tr>
    <tr>
      <th>the</th>
      <td>-4.903970</td>
      <td>-80.308936</td>
    </tr>
    <tr>
      <th>moment</th>
      <td>71.176369</td>
      <td>-6.631626</td>
    </tr>
    <tr>
      <th>mj</th>
      <td>-24.538491</td>
      <td>2.037012</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()
```


![png](output_42_0.png)


# random forest classifier로 예측

### 평균 feature vector 구하기


```python
import numpy as np

def makeFeatureVec(words, model, num_features):
    """
    주어진 문장에서 단어 벡터의 평균을 구하는 함수
    """
    # 속도를 위해 0으로 채운 배열로 초기화한다.
    featureVec = np.zeros((num_features,),dtype="float32")

    nwords = 0.
    # Index2word는 모델의 사전에 있는 단어 명을 담은 리스트이다.
    # 속도를 위해 set 형태로 초기화한다.
    index2word_set = set(model.wv.index2word)
    # 루프를 돌며 모델 사전에 포함이 되는 단어라면 피처에 추가한다.
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 결과를 단어 수로 나누어 평균을 구한다.
    featureVec = np.divide(featureVec,nwords)
    return featureVec
```


```python
def getAvgFeatureVecs(reviews, model, num_features):
    # 리뷰 단어 목록의 각각에 대한 평균 feature 벡터를 계산하고 
    # 2D numpy 배열을 반환한다.

    # 카운터를 초기화한다.
    counter = 0.
    # 속도를 위해 2D 넘파이 배열을 미리 할당한다.
    reviewFeatureVecs = np.zeros(
        (len(reviews),num_features),dtype="float32")

    for review in reviews:
       # 매 1000개 리뷰마다 상태를 출력
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       # 평균 피처 벡터를 만들기 위해 위에서 정의한 함수를 호출한다.
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)
       # 카운터를 증가시킨다.
       counter = counter + 1.
    return reviewFeatureVecs

```


```python
# 멀티스레드로 4개의 워커를 사용해 처리한다.
def getCleanReviews(reviews):
    clean_reviews = []
    clean_reviews = KaggleWord2VecUtility.apply_by_multiprocessing(\
        reviews["review"], KaggleWord2VecUtility.review_to_wordlist,\
        workers=4)
    return clean_reviews
```

리뷰 단어 목록 각각에 대한 feature vector 로 만든 뒤 학습


```python
%time trainDataVecs = getAvgFeatureVecs(\
    getCleanReviews(train), model, num_features ) 
```

    Review 0 of 25000
    Review 1000 of 25000
    Review 2000 of 25000
    Review 3000 of 25000
    Review 4000 of 25000
    Review 5000 of 25000
    Review 6000 of 25000
    Review 7000 of 25000
    Review 8000 of 25000
    Review 9000 of 25000
    Review 10000 of 25000
    Review 11000 of 25000
    Review 12000 of 25000
    Review 13000 of 25000
    Review 14000 of 25000
    Review 15000 of 25000
    Review 16000 of 25000
    Review 17000 of 25000
    Review 18000 of 25000
    Review 19000 of 25000
    Review 20000 of 25000
    Review 21000 of 25000
    Review 22000 of 25000
    Review 23000 of 25000
    Review 24000 of 25000
    CPU times: user 54.8 s, sys: 1.07 s, total: 55.8 s
    Wall time: 1min 18s



```python
%time testDataVecs = getAvgFeatureVecs(\
        getCleanReviews(test), model, num_features )
```

    Review 0 of 25000
    Review 1000 of 25000
    Review 2000 of 25000
    Review 3000 of 25000
    Review 4000 of 25000
    Review 5000 of 25000
    Review 6000 of 25000
    Review 7000 of 25000
    Review 8000 of 25000
    Review 9000 of 25000
    Review 10000 of 25000
    Review 11000 of 25000
    Review 12000 of 25000
    Review 13000 of 25000
    Review 14000 of 25000
    Review 15000 of 25000
    Review 16000 of 25000
    Review 17000 of 25000
    Review 18000 of 25000
    Review 19000 of 25000
    Review 20000 of 25000
    Review 21000 of 25000
    Review 22000 of 25000
    Review 23000 of 25000
    Review 24000 of 25000
    CPU times: user 54.3 s, sys: 996 ms, total: 55.3 s
    Wall time: 1min 17s


### avg feature vector from scratch by hj

**1. avg featurevec 를 구하는 이유**

동일한 단어라도 각 리뷰마다 다른 word embedding값을 가질 것이다. 

이 값들을 평균내서 단어의 평균 word embedding 값을 구해주는 게 featurevec가 하는 일.


```python
# 1. 단어사전 300개 지정 
```


```python
featureVec = np.zeros((num_features,),dtype="float32")
nwords = 0
index2word_set = set(model.wv.index2word)
```


```python
#2. 각 리뷰마다, 단어가 가진 word embedding 값이 다를텐데 그 값들을 더해주는 과정.
# 즉 예를 들어, queen이란 단어가 단어사전에 있다고 하자.
# review 1 - word embedding value 1 of queen
# review 2 - word embedding value 2 of queen
# review 3 - word embedding value 3 of queen
# ...

# 이것들을 다 더하면
# feature vec에는 sum of [word embedding values of queen]이 들어갈 것이다. 
```


```python
for review in getCleanReviews(train):
    for word in review
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
```


```python
featureVec[:100] #총 300개.
```




    array([  43277.8359375 ,  -36582.87109375,  -69812.5       ,
            -15764.14746094,  -21503.10546875,   19546.38671875,
            108738.6015625 ,  -32815.83203125,  105890.0390625 ,
            -51010.33984375,  -14712.4453125 ,   73403.90625   ,
            -65969.265625  ,  136879.5625    ,  -30680.07226562,
            143764.203125  ,  -30274.37695312,   52444.234375  ,
            -89841.203125  ,  -40014.390625  , -165481.421875  ,
            -33258.7109375 ,  -20850.96289062,  -14847.16992188,
             53174.44921875,   -1347.15393066,   73878.9375    ,
             60239.734375  ,   41290.51953125,  -65331.40234375,
            153440.515625  ,  -33024.93359375,   45669.23046875,
            -50730.84375   ,  125627.140625  ,   43632.2109375 ,
             47208.79296875,  -62631.97265625,   -3389.45898438,
             -6763.79833984,  113081.3828125 ,  -21843.76367188,
             38117.83203125,  -14444.27539062,   13584.72167969,
             85891.4140625 ,   13480.09863281,   12926.14648438,
            134929.234375  ,  -12780.91113281,  -33238.19140625,
            -35529.09375   ,   56736.27734375,   25159.81835938,
            -62394.3984375 ,  -14746.43554688,  -83178.3203125 ,
             79496.78125   ,   10658.36328125,    2992.44287109,
             -6175.94091797,   28865.84375   ,   -1883.58129883,
             32427.1171875 , -123675.234375  ,  123198.75      ,
            -52450.09375   ,  166563.453125  ,  -37210.78515625,
             57060.8984375 ,  -74277.578125  ,  115813.296875  ,
            -63076.98828125,  -30099.60351562,   14265.02148438,
            -17157.86914062, -170868.109375  ,  -41254.5234375 ,
             60729.59765625,   24252.2890625 ,  -25122.06640625,
            -65625.46875   ,  -24979.99804688,   34203.85546875,
            -15394.63476562,  -72860.90625   ,  -22108.73632812,
             58017.92578125, -120817.265625  ,  -94287.1875    ,
              6480.33300781,  -16008.90429688,    8651.65527344,
           -129602.890625  ,    5216.33447266,  -57224.56640625,
            -35796.6796875 ,  -12472.36035156,   18314.2578125 ,
            -13862.18164062], dtype=float32)




```python
#3. 마지막으로 단어 사전의 단어마다 빈도 수가 다를 테니 빈도 수로 나누어준다.
# 즉 최종 feature vec은 mean of [word embedding values of queen]이 들어갈 것이다. 
# 그럼 이제  단어 사전의 단어마다 
```


```python
# 결과를 단어 수로 나누어 평균을 구한다.
featureVec = np.divide(featureVec,nwords)
```


```python
featureVec[:100]
```




    array([ 0.00746138, -0.00630713, -0.01203613, -0.00271784, -0.00370728,
            0.00336993,  0.01874725, -0.00565766,  0.01825614, -0.00879452,
           -0.00253652,  0.01265532, -0.01137353,  0.02359894, -0.00528945,
            0.02478589, -0.0052195 ,  0.00904173, -0.01548921, -0.00689874,
           -0.02853008, -0.00573402, -0.00359484, -0.00255975,  0.00916762,
           -0.00023226,  0.01273721,  0.01038573,  0.00711876, -0.01126356,
            0.02645415, -0.00569372,  0.00787368, -0.00874633,  0.02165894,
            0.00752248,  0.00813911, -0.01079816, -0.00058436, -0.00116612,
            0.01949597, -0.00376601,  0.00657176, -0.00249029,  0.0023421 ,
            0.01480824,  0.00232406,  0.00222855,  0.02326269, -0.00220351,
           -0.00573048, -0.00612545,  0.00978171,  0.00433772, -0.0107572 ,
           -0.00254238, -0.01434049,  0.01370577,  0.00183757,  0.00051592,
           -0.00106477,  0.00497666, -0.00032474,  0.00559065, -0.02132242,
            0.02124027, -0.00904274,  0.02871663, -0.00641538,  0.00983767,
           -0.01280594,  0.01996697, -0.01087489, -0.00518937,  0.00245938,
           -0.00295813, -0.02945879, -0.00711255,  0.01047018,  0.00418125,
           -0.00433121, -0.01131426, -0.00430672,  0.00589697, -0.00265414,
           -0.0125617 , -0.00381169,  0.01000267, -0.02082969, -0.01625573,
            0.00111725, -0.00276004,  0.0014916 , -0.02234439,  0.00089933,
           -0.00986589, -0.00617158, -0.00215032,  0.0031575 , -0.00238993], dtype=float32)



**2. reviewFeautreVec 구하는 이유**

모든 리뷰마다 avg feature vec를 구한다.


```python
reviewFeatureVecs = np.zeros(
    (len(getCleanReviews(train)),num_features),dtype="float32")
```


```python
reviewFeatureVecs.shape #각 리뷰들이 row, 단어사전의 단어들이 column
```




    (25000, 300)



각 리뷰마다 아까 만든 feature vec를 넣어준다. 

그러면 최종으로 각 리뷰별, 단어별, 평균 feature vec를 구할 수 있는데, 이것이 예측변수로 들어간다.


```python
#카운터 초기화
counter = 0

for review in getCleanReviews(train):
    # 평균 피처 벡터를 만들기 위해 위에서 정의한 함수를 호출한다.
    reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
        num_features)
    counter = counter + 1
```


```python
reviewFeatureVecs
```




    array([[ 0.00150529,  0.01103031, -0.00816323, ...,  0.00249564,
            -0.00765224,  0.00439896],
           [ 0.00482398, -0.02468516, -0.02431285, ...,  0.00930303,
            -0.0175909 ,  0.01331643],
           [ 0.00440439, -0.02945814, -0.01366866, ...,  0.02271648,
             0.00406455,  0.00833462],
           ..., 
           [ 0.00555219,  0.01114775, -0.00860306, ..., -0.00792819,
            -0.01058043,  0.01111708],
           [ 0.00368529, -0.02184346, -0.0250172 , ...,  0.0162076 ,
            -0.00263063,  0.02172502],
           [-0.00089606, -0.00426484, -0.00975944, ..., -0.00582595,
            -0.00395891,  0.00985096]], dtype=float32)




```python
reviewFeatureVecs.shape
```




    (25000, 300)



### 랜포로 학습


```python
a = getCleanReviews(train)[0]
```


```python
temp_vec = makeFeatureVec(a, model, num_features )
```


```python
temp_vec
```




    array([  1.50529016e-03,   1.10303052e-02,  -8.16323049e-03,
            -3.23904282e-03,  -3.79234785e-03,   7.16752233e-03,
             2.28214841e-02,  -8.59869365e-03,   9.04021319e-03,
            -1.79818217e-02,  -1.18850882e-03,   1.34315277e-02,
            -1.35411853e-02,   1.10117160e-02,  -8.29315628e-04,
             8.22135899e-03,  -4.93493117e-03,   8.47088452e-03,
            -1.57629792e-02,  -2.90230353e-04,  -2.37419736e-02,
            -3.09550879e-03,  -1.01268971e-02,   3.30725947e-04,
             1.16922113e-03,   5.46010863e-03,   3.21324100e-03,
             1.03572197e-02,   1.06110619e-02,  -1.41721945e-02,
             2.27137096e-02,  -9.78946593e-03,   2.25413777e-03,
            -5.38882893e-03,   1.76430847e-02,   7.17692217e-03,
             7.45031936e-03,  -6.22338243e-03,  -8.27366312e-04,
            -5.70786232e-03,   1.52504947e-02,  -2.03610631e-03,
             3.31725064e-03,  -1.68471690e-03,   8.14944389e-04,
             1.55954063e-02,   4.12699766e-03,   5.77879883e-03,
             2.73221899e-02,  -2.42081354e-03,  -1.05169369e-02,
            -9.42508224e-03,   1.14100222e-02,   1.14345765e-02,
            -1.29566053e-02,  -1.81207294e-03,  -2.11283602e-02,
             1.46933533e-02,  -1.27410807e-03,   3.49169504e-03,
             5.81657095e-03,   5.25891827e-03,   1.22939190e-03,
             4.80777724e-03,  -2.38338076e-02,   2.24672388e-02,
            -8.46160203e-03,   3.34953070e-02,  -1.27307763e-02,
             1.31522510e-02,  -7.55584380e-03,   2.36248896e-02,
            -3.12625081e-03,  -5.38261142e-03,  -2.34170395e-04,
            -8.74355808e-03,  -2.73867436e-02,  -2.02109059e-03,
             4.22215275e-03,   2.79379915e-03,  -3.95888137e-03,
            -1.63047686e-02,  -1.26792863e-02,   5.76392189e-03,
            -3.65250814e-03,  -2.23027896e-02,  -3.90141062e-03,
             1.17570795e-02,  -2.13839449e-02,  -2.00144202e-02,
            -3.87235050e-04,   2.57926900e-03,   6.71309826e-04,
            -2.31817998e-02,   6.49184175e-03,  -1.11919381e-02,
            -4.32327902e-03,  -3.83146922e-04,   1.09063424e-02,
            -5.13001019e-03,   1.13131534e-02,   8.77182651e-03,
            -1.09792314e-03,  -1.67138577e-02,   3.11374082e-03,
             5.84132550e-03,   1.16300657e-02,  -1.45317968e-02,
             9.89772193e-03,  -1.17971683e-02,  -2.35974207e-03,
             6.13020780e-03,  -2.80396105e-03,   2.05450244e-02,
            -2.19477918e-02,   1.24030206e-02,   1.51153409e-03,
            -1.78360250e-02,  -8.87757353e-03,   1.59267392e-02,
            -2.26870272e-02,   6.45816838e-03,  -6.26618834e-03,
            -1.75512005e-02,   4.79752431e-03,   1.35564525e-02,
             1.38718775e-02,  -3.31382942e-03,   4.31624288e-03,
             1.78880605e-03,  -6.12699240e-03,   1.25193251e-02,
            -6.05734458e-05,  -3.23085557e-03,  -2.80479551e-03,
             3.35541228e-03,  -3.07192386e-04,  -1.85803156e-02,
             8.25601537e-03,   8.56125448e-03,  -8.94240662e-03,
            -6.90470031e-03,   1.34476796e-02,   6.50821719e-03,
             1.02420049e-02,  -2.96295155e-03,   1.46283703e-02,
             8.88741575e-03,  -2.36950722e-03,  -1.72574557e-02,
             1.19573139e-02,   1.33726560e-02,  -2.51003206e-02,
            -4.95055225e-03,   2.32488848e-03,   1.71153061e-02,
             1.92360338e-02,   4.23272653e-03,  -2.11303849e-02,
            -2.40759645e-03,  -5.77615248e-03,   8.23669601e-04,
            -1.12839276e-02,  -1.71404630e-02,  -7.67260650e-03,
             3.31780012e-03,  -5.44408197e-03,  -6.26250810e-04,
            -9.33989882e-03,   6.13987958e-03,   3.47427325e-03,
            -3.75316216e-04,   1.84020903e-02,  -7.10275955e-03,
             9.10445955e-03,  -9.00686253e-04,   6.38570823e-03,
             2.08506882e-02,   8.75463430e-03,  -9.68101993e-03,
            -6.55411184e-03,  -8.44454020e-03,   1.14302794e-02,
            -1.71402898e-02,   4.48470283e-03,   1.25490604e-02,
            -1.39044020e-02,  -8.61189794e-03,  -4.70104022e-03,
            -1.75088327e-02,   9.86699481e-03,   1.71027728e-03,
             9.38883889e-03,  -5.62875578e-03,   1.60284410e-03,
            -5.31525118e-03,  -1.01583207e-03,  -2.27985065e-02,
             5.79485390e-03,   8.96170549e-03,   8.19395110e-03,
            -1.20327901e-02,  -7.66879786e-03,   5.86916506e-03,
             8.50342866e-03,  -4.34160698e-03,   9.53705143e-03,
            -6.92971051e-03,  -1.10006193e-02,  -4.00544796e-03,
             1.00136194e-02,  -6.02190499e-04,   1.30172763e-02,
            -2.04649498e-03,  -1.90648660e-02,   2.97072763e-03,
             8.65408801e-05,   8.42461362e-03,   1.42904045e-02,
             2.82599647e-02,  -1.42175118e-02,   4.65403358e-03,
            -1.55696990e-02,   2.11588107e-03,  -1.97566673e-02,
             1.55959381e-02,   5.78130549e-03,   4.66423668e-03,
            -2.65762722e-03,  -1.18183326e-02,  -5.38599445e-04,
             9.82161146e-03,  -1.12784712e-03,  -2.23190617e-03,
            -1.57582472e-04,   2.05157604e-03,  -4.06458881e-03,
            -1.42547942e-03,  -1.93381077e-03,   3.15721356e-03,
             8.41433369e-03,  -7.69853545e-03,   7.65910465e-03,
            -1.90479513e-02,   5.35252364e-03,   2.52401959e-02,
             4.24405187e-03,   2.07887287e-03,   2.43626116e-03,
             2.00178567e-02,  -2.98168219e-04,   3.93268140e-03,
            -2.90560955e-03,   1.19955102e-02,   5.37558226e-03,
            -4.82169585e-03,  -6.13087462e-03,   1.09401820e-02,
             6.85627060e-03,   1.80334633e-03,   7.24920025e-03,
            -3.47610633e-03,  -5.04601514e-03,  -6.23134151e-03,
             1.49964755e-02,   4.24604584e-03,   1.37440031e-02,
            -1.12940362e-02,  -8.58669169e-03,  -4.13474580e-03,
            -5.79788443e-03,   5.85982902e-03,   1.46555621e-03,
            -1.01159001e-02,   1.68268736e-02,  -2.80653015e-02,
             1.71622932e-02,  -5.02084848e-03,  -6.74569188e-03,
            -1.02934735e-02,   3.20384558e-03,  -6.39671134e-03,
            -3.93164204e-03,  -1.01541597e-02,  -1.19620254e-02,
             1.28276006e-05,   9.94103402e-03,  -3.85524589e-03,
            -4.89852764e-03,   2.62143486e-03,   8.13133828e-03,
             5.12429746e-03,   7.15774624e-03,   1.43537382e-02,
             1.99573524e-02,   2.52066902e-03,  -2.73921364e-03,
             2.49564415e-03,  -7.65224127e-03,   4.39895643e-03], dtype=float32)




```python
trainDataVecs
```




    array([[ 0.00150529,  0.01103031, -0.00816323, ...,  0.00249564,
            -0.00765224,  0.00439896],
           [ 0.00482398, -0.02468516, -0.02431285, ...,  0.00930303,
            -0.0175909 ,  0.01331643],
           [ 0.00440439, -0.02945814, -0.01366866, ...,  0.02271648,
             0.00406455,  0.00833462],
           ..., 
           [ 0.00555219,  0.01114775, -0.00860306, ..., -0.00792819,
            -0.01058043,  0.01111708],
           [ 0.00368529, -0.02184346, -0.0250172 , ...,  0.0162076 ,
            -0.00263063,  0.02172502],
           [-0.00089606, -0.00426484, -0.00975944, ..., -0.00582595,
            -0.00395891,  0.00985096]], dtype=float32)




```python
trainDataVecs.shape
```




    (25000, 300)




```python
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(
    n_estimators = 100, n_jobs = -1, random_state=2018)
```


```python
%time forest = forest.fit( trainDataVecs, train["sentiment"] )
```

    CPU times: user 1min 6s, sys: 116 ms, total: 1min 6s
    Wall time: 2.7 s



```python
from sklearn.model_selection import cross_val_score
%time score = np.mean(cross_val_score(\
    forest, trainDataVecs, \
    train['sentiment'], cv=10, scoring='roc_auc'))
```

    CPU times: user 9min 45s, sys: 792 ms, total: 9min 46s
    Wall time: 25.4 s



```python
score
```




    0.90643337600000007




```python
result = forest.predict( testDataVecs )
```


```python
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv('data/Word2Vec_AverageVectors_{0:.5f}.csv'.format(score), 
              index=False, quoting=3 )
```


```python
output_sentiment = output['sentiment'].value_counts()
print(output_sentiment[0] - output_sentiment[1])
output_sentiment
```

    -26





    1    12513
    0    12487
    Name: sentiment, dtype: int64




```python
#별 의미 없음
import seaborn as sns 
%matplotlib inline

fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.countplot(train['sentiment'], ax=axes[0])
sns.countplot(output['sentiment'], ax=axes[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa59da50d68>




![png](output_81_1.png)


### avg feature vector 대신 knn으로 centroid구하기

[DeepLearningMovies/KaggleWord2VecUtility.py at master · wendykan/DeepLearningMovies](https://github.com/wendykan/DeepLearningMovies/blob/master/KaggleWord2VecUtility.py)

캐글에 링크 되어 있는 github 튜토리얼을 참고하여 만들었으며 파이썬2로 되어있는 소스를 파이썬3에 맞게 일부 수정하였다.


**K-means (K평균) 클러스터링으로 데이터 묶기**

[K-평균 알고리즘 - 위키백과, 우리 모두의 백과사전](https://ko.wikipedia.org/wiki/K-%ED%8F%89%EA%B7%A0_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)

클러스터링은 비지도 학습 기법

클러스터링은 유사성 등 개념에 기초해 몇몇 그룹으로 분류하는 기법

클러스터링의 목적은 샘플(실수로 구성된 n 차원의 벡터)을 내부적으로는 비슷하지만, 외부적으로 공통분모가 없는 여러 그룹으로 묶는 것

특정 차원의 범위가 다른 차원과 차이가 크면 클러스터링하기 전에 스케일을 조정해야 한다.

1. 최초 센트로이드(centroid)(중심점)로 k개의 벡터를 무작위로 선정한다.
2. 각 샘플을 그 위치에서 가장 가까운 센트로이드에 할당한다.
3. 센트로이드의 위치를 재계산한다.
4. 센트로이드가 더는 움직이지 않을 때까지 2와 3을 반복한다.

참고 : [책] 모두의 데이터 과학(with 파이썬)

### 첫 번째 시도(average feature vectors)

튜토리얼2의 코드로 벡터의 평균을 구한다.

### 두 번째 시도(K-means)

- Word2Vec은 의미가 관련 있는 단어들의 클러스터를 생성하기 때문에 클러스터 내의 단어 유사성을 이용하는 것이다.
- 이런 식으로 벡터를 그룹화하는 것을 vector quantization(벡터 양자화)라고 한다.
- 이를 위해서는 K-means와 같은 클러스터링 알고리즘을 사용하여 클러스터라는 단어의 중심을 찾아야 한다.
- 비지도 학습인 K-means를 통해 클러스터링하고 지도학습인 랜덤포레스트로 리뷰가 추천인지 아닌지를 예측한다.



```python
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from bs4 import BeautifulSoup
import re
import time

from nltk.corpus import stopwords
import nltk.data

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
model = Word2Vec.load('300features_40minwords_10text')
model
```

    2018-07-30 08:03:31,267 : INFO : loading Word2Vec object from 300features_40minwords_10text
    2018-07-30 08:03:31,551 : INFO : loading wv recursively from 300features_40minwords_10text.wv.* with mmap=None
    2018-07-30 08:03:31,551 : INFO : setting ignored attribute vectors_norm to None
    2018-07-30 08:03:31,552 : INFO : loading vocabulary recursively from 300features_40minwords_10text.vocabulary.* with mmap=None
    2018-07-30 08:03:31,553 : INFO : loading trainables recursively from 300features_40minwords_10text.trainables.* with mmap=None
    2018-07-30 08:03:31,553 : INFO : setting ignored attribute cum_table to None
    2018-07-30 08:03:31,554 : INFO : loaded 300features_40minwords_10text





    <gensim.models.word2vec.Word2Vec at 0x7fa59db335f8>




```python
# 숫자로 단어를 표현
# Word2Vec 모델은 어휘의 각 단어에 대한 feature 벡터로 구성되며
# 'syn0'이라는 넘파이 배열로 저장된다.
# syn0의 행 수는 모델 어휘의 단어 수
# 컬럼 수는 2부에서 설정한 피처 벡터의 크기
type(model.wv.syn0)
```




    numpy.ndarray




```python
# syn0의 행 수는 모델 어휘의 단어 수
# 열 수는 2부에서 설정한 특징 벡터의 크기
model.wv.syn0.shape
```




    (11986, 300)




```python
model.wv.syn0 #단어들의 차원을 줄인 것
```




    array([[ 0.06857584, -0.08908626, -0.1413338 , ..., -0.02047134,
            -0.03055766,  0.03220269],
           [-0.04243432, -0.07301593,  0.00064677, ..., -0.03542022,
             0.01490266, -0.00769218],
           [ 0.00236901,  0.03051756,  0.01223689, ...,  0.03757854,
            -0.09623639, -0.10434743],
           ..., 
           [-0.02697999, -0.12175807,  0.04290784, ...,  0.02816296,
             0.04635599,  0.04288006],
           [ 0.03608203, -0.09203785, -0.0087913 , ..., -0.02622958,
             0.04311008,  0.00610909],
           [ 0.10187781, -0.00168801, -0.06760626, ..., -0.03277631,
             0.0010244 ,  0.01257479]], dtype=float32)




```python
# 개별 단어 벡터 접근
model.wv['flower'].shape
```




    (300,)




```python
model.wv['flower'][:10]
```




    array([ 0.05890045, -0.00891029,  0.06552412, -0.01939453,  0.0625994 ,
           -0.08876541, -0.00085926,  0.0417149 ,  0.08830464,  0.04886935], dtype=float32)




```python
# 단어 벡터에서 k-means를 실행하고 일부 클러스터를 찍어본다.
start = time.time() # 시작시간

# 클러스터의 크기 "k"를 어휘 크기의 1/5이나 평균 5단어로 설정한다.
word_vectors = model.wv.syn0 # 어휘의 feature vector
num_clusters = word_vectors.shape[0] / 5
num_clusters = int(num_clusters)

# K means 를 정의하고 학습시킨다.
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# 끝난 시간에서 시작시각을 빼서 걸린 시간을 구한다.
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")
```

    Time taken for K Means clustering:  244.466890335083 seconds.



```python
# 각 어휘 단어를 클러스터 번호에 매핑되게 word/Index 사전을 만든다.
idx = list(idx)
names = model.wv.index2word
word_centroid_map = {names[i]: idx[i] for i in range(len(names))}
#     word_centroid_map = dict(zip( model.wv.index2word, idx ))
```


```python
len(idx)
```




    11986




```python
#unique한 클러스터 개수
myset = set(idx)
print (len(myset))
```

    2397



```python
dict(list(word_centroid_map.items())[0:20])
```




    {'a': 1632,
     'and': 315,
     'as': 1428,
     'but': 1358,
     'film': 296,
     'for': 768,
     'i': 1622,
     'in': 602,
     'is': 1751,
     'it': 729,
     'movi': 296,
     'of': 1576,
     's': 1751,
     'that': 179,
     'the': 2027,
     'this': 729,
     'to': 1429,
     'was': 879,
     'with': 1152,
     'you': 1849}




```python

# 첫 번째 클러스터의 처음 10개를 출력

for cluster in range(0,10):
    # 클러스터 번호를 출력
    print("\nCluster {}".format(cluster))

    ## 클러스터마다 클러스터에 해당되는 단어를 가져온다. 
    words = []
    for i in range(0,len(list(word_centroid_map.values()))):
        if( list(word_centroid_map.values())[i] == cluster ):
            words.append(list(word_centroid_map.keys())[i])
    print(words)
```


    Cluster 0
    ['underdog', 'squall']
    
    Cluster 1
    ['cartwright']
    
    Cluster 2
    ['cower']
    
    Cluster 3
    ['hectic']
    
    Cluster 4
    ['ellen', 'bate', 'kathi', 'mia', 'angi', 'farrow', 'amber', 'dickinson', 'theron', 'barkin', 'charliz']
    
    Cluster 5
    ['popul', 'slave', 'tribe', 'oppress', 'tribal', 'enslav', 'tibetan', 'endang', 'displac', 'cleans', 'erad', 'indigen', 'migrat', 'outnumb']
    
    Cluster 6
    ['confus', 'disjoint', 'convolut', 'muddl', 'messi', 'jumbl', 'threadbar']
    
    Cluster 7
    ['reaper', 'seren', 'stairway']
    
    Cluster 8
    ['bread', 'dice', 'sandwich', 'banana', 'butter', 'cigar', 'pussi', 'loaf', 'cereal']
    
    Cluster 9
    ['understand', 'imagin', 'explain', 'justifi', 'deni', 'understood', 'grasp', 'discern', 'comprehend', 'fathom', 'deciph', 'divulg']



```python
"""
판다스로 데이터프레임 형태의 데이터로 읽어온다.
QUOTE_MINIMAL (0), QUOTE_ALL (1), 
QUOTE_NONNUMERIC (2) or QUOTE_NONE (3).

그리고 이전 튜토리얼에서 했던 것처럼 clean_train_reviews 와 
clean_test_reviews 로 텍스트를 정제한다.
"""
train = pd.read_csv('data/labeledTrainData.tsv', 
                    header=0, delimiter="\t", quoting=3)
test = pd.read_csv('data/testData.tsv', 
                   header=0, delimiter="\t", quoting=3)
# unlabeled_train = pd.read_csv( 'data/unlabeledTrainData.tsv', header=0,  delimiter="\t", quoting=3 )
```


```python
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to /home/jovyan/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.





    True




```python
# 학습 리뷰를 정제한다.
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(
        KaggleWord2VecUtility.review_to_wordlist( review, \
        remove_stopwords=True ))
```


```python
# 테스트 리뷰를 정제한다.
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(
        KaggleWord2VecUtility.review_to_wordlist( review, \
        remove_stopwords=True ))
```


```python
# bags of centroids 생성
# 속도를 위해 centroid 학습 세트 bag을 미리 할당한다.
train_centroids = np.zeros((train["review"].size, num_clusters), \
    dtype="float32" )

train_centroids.shape

```




    (25000, 2397)



### bag of centroids from scratch by hj

** create_bag_of_centroid를 만드는 이유**

각 리뷰마다, 리뷰안의 단어들이 어떠한 클러스트를 가지고 있는지 클러스트들을 쌓는다

그래서 bag_of_centroids는 리뷰들의 개수만큼 생긴다.


```python
# centroid 는 두 클러스터의 중심점을 정의 한 다음 중심점의 거리를 측정한 것 <--먼말??그냥 cluster들의 넘버 모아 놓은 거 아님?
def create_bag_of_centroids( wordlist, word_centroid_map ):

    # 클러스터의 수는 word / centroid map에서 가장 높은 클러스트 인덱스와 같다.
    num_centroids = max( word_centroid_map.values() ) + 1

    # 속도를 위해 bag of centroids vector를 미리 할당한다.
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )

    # 루프를 돌며 단어가 word_centroid_map에 있다면
    # 해당하는 클러스터의 수를 하나씩 증가시켜 준다.
    for word in wordlist: ## wordlist = 각 review
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    # bag of centroids를 반환한다.
    return bag_of_centroids
```


```python
num_centroids = max( word_centroid_map.values() ) + 1
bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
```


```python
bag_of_centroids.shape
```




    (2397,)




```python
#dice라는 단어가 리뷰안에 들어가 있다면 클러스터 8 추가
```


```python
word_centroid_map['dice'] 
```




    8



** test_centroids가 모든 리뷰에 대해 클러스트들을 쌓은것**


```python
train_centroids.shape
```




    (25000, 2397)




```python
# 학습 리뷰를 bags of centroids로 변환한다.
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# 테스트 리뷰도 같은 방법으로 반복해 준다.
test_centroids = np.zeros(( test["review"].size, num_clusters), \
    dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1


# 랜덤포레스트를 사용하여 학습시키고 예측
forest = RandomForestClassifier(n_estimators = 100)

# train 데이터의 레이블을 통해 학습시키고 예측한다.
# 시간이 좀 소요되기 때문에 %time을 통해 걸린 시간을 찍도록 함
print("Fitting a random forest to labeled training data...")
%time forest = forest.fit(train_centroids, train["sentiment"])
```

    Fitting a random forest to labeled training data...
    CPU times: user 19.1 s, sys: 32 ms, total: 19.2 s
    Wall time: 19.1 s


** 도대체 centroid를 왜 피팅과정에 넣는 것인가? 에 대한 답 **

train_centroid를 데이터프레임이라고 생각한다면, train_cetroid가 가지고 있는 변수들은 cluster1,cluster2, ...,cluster2397 들일 것이다. 즉, clsuter 몇 번에 몇 개의 단어가 속했는지가, 종속변수인 sentiment를 예측할 수 있게 하는 것이다. 


```python
train_centroids[0]
```




    array([ 0.,  0.,  0., ...,  0.,  0.,  0.], dtype=float32)




```python
train["sentiment"][0]
```




    1




```python
from sklearn.model_selection import cross_val_score
%time score = np.mean(cross_val_score(\
    forest, train_centroids, train['sentiment'], cv=10,\
    scoring='roc_auc'))
```

    CPU times: user 2min 49s, sys: 640 ms, total: 2min 50s
    Wall time: 2min 49s



```python
%time result = forest.predict(test_centroids)
```

    CPU times: user 1.4 s, sys: 12 ms, total: 1.42 s
    Wall time: 1.41 s



```python
score
```




    0.91752092800000007




```python
# 결과를 csv로 저장
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("data/submit_BagOfCentroids_{0:.5f}.csv".format(score), index=False, quoting=3)
```


```python
fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.countplot(train['sentiment'], ax=axes[0])
sns.countplot(output['sentiment'], ax=axes[1])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa4e7320358>




![png](output_121_1.png)



```python
output_sentiment = output['sentiment'].value_counts()
print(output_sentiment[0] - output_sentiment[1])
output_sentiment
```

    274





    0    12637
    1    12363
    Name: sentiment, dtype: int64



### 왜 이 튜토리얼에서는 Bag of Words가 더 좋은 결과를 가져올까?
벡터를 평균화하고 centroids를 사용하면 단어 순서가 어긋나며 Bag of Words 개념과 매우 비슷하다. 성능이 (표준 오차의 범위 내에서) 비슷하기 때문에 튜토리얼 1, 2, 3이 동등한 결과를 가져온다.

첫째, Word2Vec을 더 많은 텍스트로 학습시키면 성능이 좋아진다. Google의 결과는 10 억 단어가 넘는 코퍼스에서 배운 단어 벡터를 기반으로 한다. 학습 레이블이 있거나 레이블이 없는 학습 세트는 단지 대략 천팔백만 단어 정도다. 편의상 Word2Vec은 Google의 원래 C 도구에서 출력되는 사전 학습된 모델을 로드하는 기능을 제공하기 때문에 C로 모델을 학습 한 다음 Python으로 가져올 수도 있다.

둘째, 출판된 자료들에서 분산 워드 벡터 기술은 Bag of Words 모델보다 우수한 것으로 나타났다. 이 논문에서는 IMDB 데이터 집합에 단락 벡터 (Paragraph Vector)라는 알고리즘을 사용하여 현재까지의 최첨단 결과 중 일부를 생성한다. 단락 벡터는 단어 순서 정보를 보존하는 반면 벡터 평균화 및 클러스터링은 단어 순서를 잃어버리기 때문에 여기에서 시도하는 방식보다 부분적으로 더 좋다.
