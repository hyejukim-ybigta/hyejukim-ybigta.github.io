# semantic similarity

### 개념

텍스트 내 단어 끼리의 similarity를 정량화한 것

### 활용 용도

- 비슷한 단어끼리 그룹화 가능
- natural language understanding에서 블록을 만들 수 있음
  - textual entailment(텍스트 전체의 내용을 반영하여 텍스트 내부의 sentence 의미 추론)
  - paraphrasing

### 측정 방법 using WordNet

**1. Word Net이란? **

영어의 의미 어휘 목록. 영어 단어의 의미관계를 저장해 놓은 데이터베이스라고 생각하면 됨. 

- 계층구조(hierarchy)를 바탕으로 semantic similarity 판단 가능
- 단어 형태(verb,noun,adejctive)에 따라 각각의 계층구조를 가지고 있음

![image](https://user-images.githubusercontent.com/32008883/44022086-99a4a436-9f21-11e8-8979-4c16b7ae4fe8.png)



### 2. Path Similarity

- 두 단어 사이에 가장 짧은 path찾는 것, path distance의 반대 개념이다.
- path distance(deer,elk) = 1, path distance(deer,giraffe) = 2
- PathSim(deer,elk) = 1/ (1+1) = 0.5, PathSim(deer,giraffe) = 1/ (1+2) = 0.33

**3. Lin Similarity**

- Lowest common subsumer(LCS)에 기반하여 계산
- LCS란 두 단어의 가장 가까운 공통 분모
- LCS(deer, elk) = deer , LCS(deer,giraffe) = ruminant
- LinSim(u,v) = 2 * logP(LCS(u,v))/(logP(u)+logP(v)) 
- 상단의 P(u)는 큰 말뭉치로 학습하여 계산
- 









find similariy between 2 means

path similarity - find the path between the two concepts

similariy measure inversly related to path distance

elk&deer - distance 1 - pathsim(1/1+1)

elk&giraffe - distance 2 -pathsim 1/(1+2)

Lowest common subsumer(LCS)

deer&giraffe - ruminant 

deer&elk = deer



Lin Similarity

based on the information contained in the LCS of the 2 concepts



python

deer.n.01:  first syntax(noun) meaning of deer

path_similarity 

not distance 

collocations and distributional similarity

distributional similarity



strength of association between words





topic modeling



topic modeling 6:00

what's known

(1,(2,(3



PLSA // LDA





