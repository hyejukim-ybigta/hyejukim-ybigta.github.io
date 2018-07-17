---
layout: post

title: "nlp_seq2seq_tutorial"

author: "hyeju.kim"

categories: NLP

tags: [NLP]

image: 
---



# Seq2Seq Tutorial : 일정한 날짜 형식으로 반환하기 

본 포스트는 [여기](https://github.com/sachinruk/deepschool.io/blob/master/DL-Keras_Tensorflow/Lesson%2019%20-%20Seq2Seq%20-%20Date%20translator.ipynb) 내용을 한글로 번역한 것입니다.

![seq2seq](https://user-images.githubusercontent.com/32008883/42751807-f18baf98-8926-11e8-97c5-5f1ea327e572.png)

seq2seq 모델은 두개의 lstm 모델을 가지고 있다. encoder는 many to one lstm 으로 인풋값들이 하나의 hidden value을 만들게 된다. decoder쪽은 many to many 구조인데, 바로 앞에서 나온 결과 값이 그다음에도 반영이 되므로 many to many 구조이다. 